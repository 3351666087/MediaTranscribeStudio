"""Offline production runners for the dynamic speaker cascade.

All model paths are explicit local filesystem paths.  These runners never
download models, never enable telemetry, and keep heavy imports/model loading
lazy so the worker can validate jobs before reserving GPU memory.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .adapters import AdapterContext
from .asr_evidence import (
    ASR_CANDIDATE_SET_SCHEMA_VERSION,
    build_asr_candidate_set,
    model_identity_from_manifest,
)
from .errors import WorkerError
from .language import (
    normalize_language_tag,
    normalize_qwen_language_candidates,
    qwen_language_for_request,
    qwen_supported_primary_language_tags,
    reconcile_detected_languages,
)
from .models import TranscriptSegment
from .speaker_change_detection import (
    EnergyValley,
    Pcm16kMonoTimeline,
    SpeakerChangeDetectionConfig,
    SpeakerEmbeddingWindow,
    plan_speaker_changes,
)
from .speaker_pipeline import (
    AsrHypothesis,
    EmbeddingRecord,
    OverlapDecision,
    PreparedAudio,
    ReviewCandidate,
    ReviewProposal,
    SpeechWindow,
)
from .voice_activity import build_voice_activity

VadStageObserver = Callable[[Mapping[str, Any]], None]
_THIRD_PARTY_STDOUT_LOCK = threading.Lock()
_WINDOWS_SAFE_SHARD_LIMIT_BYTES = 1024 * 1024 * 1024
_WINDOWS_FULL_GPU_MINIMUM_BYTES = 14 * 1024 * 1024 * 1024
_WINDOWS_ASR_GPU_BUDGET = "2GiB"
_WINDOWS_ASR_CPU_BUDGET = "8GiB"
_WINDOWS_ALIGNER_GPU_BUDGET = "1500MiB"
_WINDOWS_ALIGNER_CPU_BUDGET = "6GiB"


def _is_windows_runtime() -> bool:
    return os.name == "nt"


def _is_mps_device(value: str) -> bool:
    return str(value).strip().casefold().split(":", 1)[0] == "mps"


def _mps_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def _optional_result_score(result: Any, names: Sequence[str]) -> float | None:
    for name in names:
        raw = getattr(result, name, None)
        if (
            raw is not None
            and not isinstance(raw, bool)
            and isinstance(raw, (int, float))
        ):
            score = float(raw)
            if math.isfinite(score):
                return score
    return None


def _prepare_eres2netv2_mps_pipeline(
    pipeline: Any,
    *,
    requested_device: str,
) -> Any:
    """Move ModelScope ERes2NetV2 after its CPU-only device validation."""

    if not _is_mps_device(requested_device):
        return pipeline
    if not _mps_is_available():
        raise WorkerError(
            "ERES2NETV2_DEVICE_UNAVAILABLE",
            "ERes2NetV2 requested MPS but the runtime cannot provide it",
            details={
                "requestedDevice": requested_device,
                "modelQualityChanged": False,
            },
        )
    try:
        import torch
    except ImportError as exc:
        raise WorkerError(
            "ERES2NETV2_RUNTIME_MISSING",
            "ERes2NetV2 requires PyTorch for MPS execution",
        ) from exc

    model = getattr(pipeline, "model", None)
    embedding_model = getattr(model, "embedding_model", None)
    if model is None or embedding_model is None:
        raise WorkerError(
            "ERES2NETV2_RUNTIME_INCOMPATIBLE",
            "ModelScope ERes2NetV2 does not expose its embedding model",
            details={
                "requestedDevice": requested_device,
                "modelQualityChanged": False,
            },
        )

    target = torch.device(requested_device)
    try:
        moved = embedding_model.to(target)
        if moved is not None:
            model.embedding_model = moved
        model.embedding_model.eval()
        model.device = target
    except Exception as exc:
        resource_error = _eres2netv2_resource_error(
            exc,
            phase="model-device-transfer",
            requested_device=requested_device,
            clip_count=0,
        )
        if resource_error is not None:
            raise resource_error from exc
        raise WorkerError(
            "ERES2NETV2_DEVICE_CONFIGURATION_FAILED",
            "ERes2NetV2 could not initialize on the requested MPS device",
            details={
                "requestedDevice": requested_device,
                "exceptionType": type(exc).__name__,
                "modelQualityChanged": False,
            },
        ) from exc
    return pipeline


def _local_model_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"{label} is missing: {path}")
    if path.is_symlink() or bool(
        getattr(os.path, "isjunction", lambda _path: False)(path)
    ):
        raise ValueError(f"{label} must not be a symlink or junction")
    return path.resolve(strict=True)


def _executable(value: str | Path, label: str) -> str:
    text = os.fspath(value).strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    if Path(text).is_absolute() or any(
        separator and separator in text for separator in (os.sep, os.altsep)
    ):
        path = Path(text).expanduser()
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
        return str(path.resolve(strict=True))
    resolved = shutil.which(text)
    if not resolved:
        raise ValueError(f"{label} is not available on PATH: {text}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_snapshot() -> dict[str, float]:
    output = {"ramMb": 0.0, "vramMb": 0.0}
    try:
        import psutil

        output["ramMb"] = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except (ImportError, OSError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            output["vramMb"] = torch.cuda.max_memory_allocated() / (
                1024.0 * 1024.0
            )
    except (ImportError, RuntimeError):
        pass
    return output


def _checkpoint_layout(path: Path) -> dict[str, int]:
    shards = tuple(
        item.stat().st_size
        for item in path.glob("*.safetensors")
        if item.is_file()
    )
    return {
        "shardCount": len(shards),
        "largestShardBytes": max(shards, default=0),
        "totalShardBytes": sum(shards),
    }


def _windows_low_commit_layout_required(path: Path, label: str) -> None:
    if not _is_windows_runtime():
        return
    layout = _checkpoint_layout(path)
    if layout["largestShardBytes"] <= _WINDOWS_SAFE_SHARD_LIMIT_BYTES:
        return
    raise WorkerError(
        "QWEN3_CHECKPOINT_RESHARD_REQUIRED",
        (
            f"{label} contains a safetensors shard that is too large for "
            "reliable low-commit Windows loading"
        ),
        details={
            **layout,
            "safeShardLimitBytes": _WINDOWS_SAFE_SHARD_LIMIT_BYTES,
            "remediation": "STREAM_RESHARD_WITH_TOOLS_RESHARD_SAFETENSORS",
            "modelQualityChanged": False,
        },
    )


def _is_windows_pagefile_error(error: BaseException) -> bool:
    if not _is_windows_runtime():
        return False
    if isinstance(error, OSError) and getattr(error, "winerror", None) == 1455:
        return True
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "os error 1455",
            "winerror 1455",
            "页面文件太小",
            "paging file is too small",
        )
    )


def _is_accelerator_out_of_memory(error: BaseException) -> bool:
    if type(error).__name__ == "OutOfMemoryError":
        return True
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "cuda out of memory",
            "cuda error: out of memory",
            "hip out of memory",
            "mps backend out of memory",
            "defaultcpuallocator: not enough memory",
        )
    )


def _cuda_device_index(device_map: str) -> int:
    normalized = str(device_map).strip().casefold()
    if normalized.startswith("cuda:"):
        try:
            return max(0, int(normalized.split(":", 1)[1]))
        except ValueError:
            return 0
    return 0


def _cuda_total_memory_bytes(device_index: int) -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(device_index).total_memory)
    except (ImportError, RuntimeError, ValueError):
        return None


def _qwen_load_policy(
    device_map: str,
    *,
    forced_aligner: bool,
    probe_accelerator: bool,
) -> dict[str, Any]:
    """Return a stable Windows policy without probing injected test factories."""

    normalized = str(device_map).strip().casefold()
    if not _is_windows_runtime() or normalized in {"cpu", "cpu:0"}:
        return {"device_map": device_map}
    if not probe_accelerator:
        return {"device_map": device_map}

    device_index = _cuda_device_index(device_map)
    total_memory = _cuda_total_memory_bytes(device_index)
    explicit_full_gpu = normalized in {
        "cuda",
        f"cuda:{device_index}",
    }
    if (
        explicit_full_gpu
        and total_memory is not None
        and total_memory >= _WINDOWS_FULL_GPU_MINIMUM_BYTES
    ):
        return {"device_map": device_map}

    return {
        "device_map": "balanced",
        "max_memory": {
            device_index: (
                _WINDOWS_ALIGNER_GPU_BUDGET
                if forced_aligner
                else _WINDOWS_ASR_GPU_BUDGET
            ),
            "cpu": (
                _WINDOWS_ALIGNER_CPU_BUDGET
                if forced_aligner
                else _WINDOWS_ASR_CPU_BUDGET
            ),
        },
        "offload_state_dict": True,
    }


def _qwen_resource_error(
    error: BaseException,
    *,
    phase: str,
    requested_device_map: str,
    max_inference_batch_size: int,
) -> WorkerError | None:
    details = {
        "phase": phase,
        "requestedDeviceMap": requested_device_map,
        "maxInferenceBatchSize": max_inference_batch_size,
        "modelQualityChanged": False,
    }
    if _is_windows_pagefile_error(error):
        return WorkerError(
            "QWEN3_WINDOWS_COMMIT_EXHAUSTED",
            (
                "Windows committed-memory capacity was exhausted while "
                "loading or running Qwen3-ASR"
            ),
            details={
                **details,
                "remediation": (
                    "USE_STREAM_RESHARDED_CHECKPOINT_OR_INCREASE_WINDOWS_PAGEFILE"
                ),
            },
        )
    if isinstance(error, MemoryError) or _is_accelerator_out_of_memory(error):
        return WorkerError(
            "QWEN3_ACCELERATOR_MEMORY_EXHAUSTED",
            "Qwen3-ASR exceeded the bounded accelerator memory budget",
            details={
                **details,
                "remediation": (
                    "REDUCE_CONCURRENT_GPU_WORK_OR_USE_BALANCED_DEVICE_MAP"
                ),
            },
        )
    return None


def _campp_resource_error(
    error: BaseException,
    *,
    phase: str,
    requested_device: str,
    embedding_batch_size: int,
) -> WorkerError | None:
    if not (
        isinstance(error, MemoryError)
        or _is_accelerator_out_of_memory(error)
    ):
        return None
    return WorkerError(
        "CAMPP_ACCELERATOR_MEMORY_EXHAUSTED",
        "CAM++ exceeded the bounded accelerator memory budget",
        details={
            "phase": phase,
            "requestedDevice": requested_device,
            "embeddingBatchSize": embedding_batch_size,
            "modelQualityChanged": False,
            "remediation": (
                "RELEASE_PREVIOUS_GPU_STAGE_OR_REDUCE_CONCURRENT_GPU_WORK"
            ),
        },
    )


def _eres2netv2_resource_error(
    error: BaseException,
    *,
    phase: str,
    requested_device: str,
    clip_count: int,
) -> WorkerError | None:
    memory_error = isinstance(error, MemoryError)
    accelerator_error = _is_accelerator_out_of_memory(error)
    if not (memory_error or accelerator_error):
        return None
    normalized_device = requested_device.strip().casefold()
    if normalized_device in {"cpu", "cpu:0"}:
        return WorkerError(
            "ERES2NETV2_HOST_MEMORY_EXHAUSTED",
            "ERes2NetV2 exceeded the bounded host-memory budget",
            details={
                "phase": phase,
                "requestedDevice": requested_device,
                "clipCount": clip_count,
                "modelQualityChanged": False,
                "remediation": (
                    "REDUCE_CLIP_BATCH_SIZE_OR_CLOSE_MEMORY_INTENSIVE_PROCESSES"
                ),
            },
            retryable=False,
        )
    return WorkerError(
        "ERES2NETV2_ACCELERATOR_MEMORY_EXHAUSTED",
        "ERes2NetV2 exceeded the bounded accelerator memory budget",
        details={
            "phase": phase,
            "requestedDevice": requested_device,
            "clipCount": clip_count,
            "modelQualityChanged": False,
            "remediation": (
                "RELEASE_PREVIOUS_GPU_STAGE_OR_REDUCE_CONCURRENT_GPU_WORK"
            ),
        },
        retryable=True,
    )


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


def _load_audio(path: str | Path) -> tuple[Any, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise WorkerError(
            "AUDIO_RUNTIME_MISSING",
            "soundfile is required by the local production runners",
        ) from exc
    samples, sample_rate = sf.read(
        os.fspath(path),
        dtype="float32",
        always_2d=False,
    )
    if getattr(samples, "ndim", 1) == 2:
        samples = samples.mean(axis=1)
    return samples, int(sample_rate)


def _slice_audio(
    samples: Any,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
) -> Any:
    start = max(0, round(start_ms * sample_rate / 1000))
    end = min(len(samples), round(end_ms * sample_rate / 1000))
    if end <= start:
        raise WorkerError(
            "AUDIO_WINDOW_INVALID",
            "speech window resolves to an empty audio slice",
        )
    return samples[start:end]


class _SharedPcmStore:
    """Small process-local LRU for the normalized mono PCM timeline.

    The persisted WAV remains the cross-process cache.  This store removes
    repeated full-file reads inside one worker process while keeping memory
    bounded.  A miss is serialized so concurrent stages cannot load the same
    timeline more than once.
    """

    def __init__(
        self,
        *,
        max_entries: int = 4,
        max_bytes: int = 1024 * 1024 * 1024,
    ) -> None:
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("shared PCM store limits must be positive")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[str, tuple[Any, int, int]] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.RLock()

    @staticmethod
    def _sample_bytes(samples: Any) -> int:
        raw = getattr(samples, "nbytes", None)
        if isinstance(raw, int) and raw >= 0:
            return raw
        try:
            return max(0, len(samples) * 4)
        except TypeError:
            return 0

    def _trim(self) -> None:
        while len(self._entries) > self.max_entries or (
            self._total_bytes > self.max_bytes and len(self._entries) > 1
        ):
            _key, (_samples, _rate, size_bytes) = self._entries.popitem(
                last=False
            )
            self._total_bytes -= size_bytes

    def put(self, key: str, samples: Any, sample_rate: int) -> None:
        size_bytes = self._sample_bytes(samples)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._total_bytes -= previous[2]
            self._entries[key] = (samples, int(sample_rate), size_bytes)
            self._total_bytes += size_bytes
            self._trim()

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], tuple[Any, int]],
    ) -> tuple[Any, int]:
        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is not None:
                self._entries[key] = cached
                return cached[0], cached[1]
            samples, sample_rate = loader()
            size_bytes = self._sample_bytes(samples)
            self._entries[key] = (samples, int(sample_rate), size_bytes)
            self._total_bytes += size_bytes
            self._trim()
            return samples, int(sample_rate)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._total_bytes,
                "maxEntries": self.max_entries,
                "maxBytes": self.max_bytes,
            }


_SHARED_PCM_STORE = _SharedPcmStore(
    max_entries=max(1, int(os.environ.get("MTS_PCM_CACHE_ENTRIES", "4"))),
    max_bytes=max(
        1,
        int(
            os.environ.get(
                "MTS_PCM_CACHE_BYTES",
                str(1024 * 1024 * 1024),
            )
        ),
    ),
)


def _pcm_buffer_key(source_fingerprint: str, audio_path: str | Path) -> str:
    resolved = str(Path(audio_path).resolve(strict=False))
    path_digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return f"pcm16k:{source_fingerprint}:{path_digest}"


def _pcm_buffer_id(prepared: PreparedAudio) -> str:
    if not prepared.audio_path:
        raise WorkerError(
            "PREPARED_AUDIO_MISSING",
            "the prepared timeline has no normalized audio path",
        )
    return _pcm_buffer_key(
        prepared.source_fingerprint,
        prepared.audio_path,
    )


def _shared_pcm_for_prepared(prepared: PreparedAudio) -> tuple[Any, int, str]:
    if not prepared.audio_path or not Path(prepared.audio_path).is_file():
        raise WorkerError(
            "PREPARED_AUDIO_MISSING",
            "the persisted normalized audio timeline is missing",
        )
    buffer_id = _pcm_buffer_id(prepared)
    samples, sample_rate = _SHARED_PCM_STORE.get_or_load(
        buffer_id,
        lambda: _load_audio(prepared.audio_path),
    )
    if sample_rate != 16_000:
        raise WorkerError(
            "NORMALIZED_AUDIO_INVALID",
            "normalized audio must be mono 16 kHz",
        )
    return samples, sample_rate, buffer_id


class FfmpegFunAsrPreparationAdapter:
    """Normalize once with FFmpeg, then produce cached FunASR VAD boundaries."""

    adapter_id = "ffmpeg-funasr-vad-boundary"
    version = "1.2.0"

    def __init__(
        self,
        *,
        vad_model_path: str | Path,
        ffmpeg_executable: str | Path = "ffmpeg",
        device: str = "cpu",
        minimum_window_ms: int = 120,
        model_factory: Callable[..., Any] | None = None,
        stage_observer: VadStageObserver | None = None,
    ) -> None:
        self.vad_model_path = _local_model_path(
            vad_model_path, "FunASR VAD model"
        )
        self.ffmpeg_executable = _executable(
            ffmpeg_executable, "ffmpeg executable"
        )
        self.device = str(device)
        self.minimum_window_ms = int(minimum_window_ms)
        if self.minimum_window_ms < 1:
            raise ValueError("minimum_window_ms must be positive")
        self._model_factory = model_factory
        self._stage_observer = stage_observer
        self._vad_model: Any = None
        self._load_lock = threading.Lock()

    def _observe_stage(
        self,
        stage: str,
        status: str,
        *,
        started_at: float | None = None,
        error: Exception | None = None,
        **details: Any,
    ) -> None:
        """Send path-free VAD diagnostics to an injected observer.

        Observability must never write to stdout because stdout is reserved for
        the worker JSONL protocol. Observer failures are deliberately isolated
        from media processing so a telemetry consumer cannot fail a job.
        """

        observer = self._stage_observer
        if observer is None:
            return
        event: dict[str, Any] = {
            "adapterId": self.adapter_id,
            "stage": stage,
            "status": status,
            "device": self.device,
        }
        if started_at is not None:
            event["elapsedMs"] = max(
                0.0,
                (time.perf_counter() - started_at) * 1000.0,
            )
        if error is not None:
            event["errorCode"] = (
                error.code
                if isinstance(error, WorkerError)
                else type(error).__name__
            )
        event.update(details)
        try:
            observer(event)
        except Exception:
            # The JSONL worker and model pipeline must remain independent from
            # optional diagnostics sinks.
            return

    def _model(self) -> Any:
        with self._load_lock:
            if self._vad_model is None:
                # Some FunASR releases print their version during import.
                # stdout is the worker's JSONL protocol, so third-party model
                # initialization must never be allowed to write to it.
                with _THIRD_PARTY_STDOUT_LOCK, redirect_stdout(io.StringIO()):
                    factory = self._model_factory
                    if factory is None:
                        try:
                            from funasr import AutoModel
                        except ImportError as exc:
                            raise WorkerError(
                                "FUNASR_RUNTIME_MISSING",
                                "FunASR is required for production VAD",
                            ) from exc
                        factory = AutoModel
                    self._vad_model = factory(
                        model=str(self.vad_model_path),
                        device=self.device,
                        disable_update=True,
                        disable_pbar=True,
                        ncpu=1,
                    )
            return self._vad_model

    @staticmethod
    def _vad_intervals(raw: Any) -> list[tuple[int, int]]:
        candidates: Any = raw
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) == 1 and isinstance(raw[0], Mapping):
                candidates = raw[0].get("value")
        elif isinstance(raw, Mapping):
            candidates = raw.get("value")
        if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes)
        ):
            raise WorkerError(
                "FUNASR_VAD_RESULT_INVALID",
                "FunASR VAD did not return an interval array",
            )
        output: list[tuple[int, int]] = []
        for index, item in enumerate(candidates):
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) < 2
            ):
                raise WorkerError(
                    "FUNASR_VAD_RESULT_INVALID",
                    f"VAD interval {index} is malformed",
                )
            start_ms = int(round(float(item[0])))
            end_ms = int(round(float(item[1])))
            if start_ms < 0 or end_ms <= start_ms:
                raise WorkerError(
                    "FUNASR_VAD_RESULT_INVALID",
                    f"VAD interval {index} has invalid boundaries",
                )
            output.append((start_ms, end_ms))
        return output

    def _normalize(
        self,
        source_path: Path,
        output_path: Path,
        context: AdapterContext,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.stem}.{uuid.uuid4().hex}.tmp.wav"
        )
        command = [
            self.ffmpeg_executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if _is_windows_runtime()
            else 0
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=creationflags,
        )
        try:
            while process.poll() is None:
                if context.cancellation.wait(0.05):
                    process.kill()
                    process.wait(timeout=5)
                    context.raise_if_cancelled()
            stderr = (process.stderr.read() if process.stderr else b"").decode(
                "utf-8", errors="replace"
            )
            if process.returncode != 0:
                raise WorkerError(
                    "FFMPEG_NORMALIZATION_FAILED",
                    "FFmpeg could not normalize the source media",
                    details={"diagnostic": stderr[-2000:]},
                )
            os.replace(temporary, output_path)
        finally:
            if process.stderr is not None:
                process.stderr.close()
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def prepare(
        self,
        source_path: Path,
        *,
        normalization_profile: str,
        context: AdapterContext,
    ) -> PreparedAudio:
        context.raise_if_cancelled()
        source_fingerprint = _sha256_file(source_path)
        normalized_path = (
            context.output_directory
            / ".pipeline"
            / "audio"
            / f"{source_fingerprint}.mono-16khz.wav"
        )
        decode_started = time.perf_counter()
        normalize_cache_hit = normalized_path.is_file()
        self._observe_stage(
            "normalize",
            "started",
            cacheHit=normalize_cache_hit,
        )
        try:
            if not normalize_cache_hit:
                self._normalize(source_path, normalized_path, context)
        except Exception as exc:
            self._observe_stage(
                "normalize",
                "failed",
                started_at=decode_started,
                error=exc,
                cacheHit=normalize_cache_hit,
            )
            raise
        normalization_ms = (time.perf_counter() - decode_started) * 1000.0
        self._observe_stage(
            "normalize",
            "completed",
            started_at=decode_started,
            cacheHit=normalize_cache_hit,
        )

        pcm_started = time.perf_counter()
        self._observe_stage("pcm_load", "started")
        try:
            samples, sample_rate = _load_audio(normalized_path)
            if sample_rate != 16000:
                raise WorkerError(
                    "NORMALIZED_AUDIO_INVALID",
                    "normalized audio must be mono 16 kHz",
                )
            duration_ms = max(1, round(len(samples) * 1000 / sample_rate))
        except Exception as exc:
            self._observe_stage(
                "pcm_load",
                "failed",
                started_at=pcm_started,
                error=exc,
            )
            raise
        self._observe_stage(
            "pcm_load",
            "completed",
            started_at=pcm_started,
            sampleRate=sample_rate,
            audioDurationMs=duration_ms,
        )
        normalized_audio_path = str(normalized_path.resolve(strict=True))
        _SHARED_PCM_STORE.put(
            _pcm_buffer_key(source_fingerprint, normalized_audio_path),
            samples,
            sample_rate,
        )
        context.raise_if_cancelled()
        vad_started = time.perf_counter()
        model_load_started = time.perf_counter()
        model_was_cached = self._vad_model is not None
        self._observe_stage(
            "model_load",
            "started",
            cacheHit=model_was_cached,
        )
        try:
            model = self._model()
        except Exception as exc:
            self._observe_stage(
                "model_load",
                "failed",
                started_at=model_load_started,
                error=exc,
                cacheHit=model_was_cached,
            )
            raise
        self._observe_stage(
            "model_load",
            "completed",
            started_at=model_load_started,
            cacheHit=model_was_cached,
        )

        inference_started = time.perf_counter()
        self._observe_stage(
            "inference",
            "started",
            audioDurationMs=duration_ms,
            sampleRate=sample_rate,
        )
        try:
            raw_vad = model.generate(
                input=str(normalized_path),
                cache={},
                is_final=True,
            )
            intervals = [
                (start_ms, min(end_ms, duration_ms))
                for start_ms, end_ms in self._vad_intervals(raw_vad)
                if min(end_ms, duration_ms) - start_ms
                >= self.minimum_window_ms
            ]
        except Exception as exc:
            self._observe_stage(
                "inference",
                "failed",
                started_at=inference_started,
                error=exc,
                audioDurationMs=duration_ms,
                sampleRate=sample_rate,
            )
            raise
        self._observe_stage(
            "inference",
            "completed",
            started_at=inference_started,
            audioDurationMs=duration_ms,
            sampleRate=sample_rate,
            speechWindowCount=len(intervals),
            classification=(
                "speech-candidates-detected"
                if intervals
                else "no-speech-candidates-detected"
            ),
        )
        if not intervals:
            voice_activity = build_voice_activity(
                job_id=context.job_id,
                source_sha256=source_fingerprint,
                media_duration_ms=duration_ms,
                normalization_profile=normalization_profile,
                provider={
                    "id": self.adapter_id,
                    "version": self.version,
                },
                windows=(),
                minimum_window_ms=self.minimum_window_ms,
                classification="no-speech-candidates-detected",
                has_transcribable_speech=False,
            )
            raise WorkerError(
                "NO_SPEECH_DETECTED",
                "FunASR VAD found no speech windows",
                details={"voiceActivity": voice_activity},
            )
        vad_ms = (time.perf_counter() - vad_started) * 1000.0
        windows = tuple(
            SpeechWindow(
                window_id=f"window-{index:06d}",
                start_ms=start_ms,
                end_ms=end_ms,
                metadata={"turnId": f"turn-{index:06d}"},
            )
            for index, (start_ms, end_ms) in enumerate(intervals, start=1)
        )
        return PreparedAudio(
            duration_ms=duration_ms,
            source_fingerprint=source_fingerprint,
            normalization_profile=normalization_profile,
            windows=windows,
            stage_durations_ms={
                "decode": normalization_ms,
                "normalize": normalization_ms,
                "vad": vad_ms,
                "boundary": 0.0,
            },
            audio_path=normalized_audio_path,
        )


class LocalQwen3AsrAdapter:
    """Qwen3-ASR-1.7B runner using explicit local model directories only."""

    adapter_id = "Qwen3-ASR-1.7B"
    version = "1.5.0"

    def __init__(
        self,
        *,
        model_path: str | Path,
        forced_aligner_path: str | Path | None = None,
        device_map: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        max_inference_batch_size: int = 2,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = _local_model_path(model_path, "Qwen3-ASR model")
        self.forced_aligner_path = (
            _local_model_path(forced_aligner_path, "Qwen3 forced aligner")
            if forced_aligner_path is not None
            else None
        )
        self.device_map = str(device_map)
        self.torch_dtype = str(torch_dtype)
        if (
            isinstance(max_inference_batch_size, bool)
            or int(max_inference_batch_size) < 1
        ):
            raise ValueError("max_inference_batch_size must be positive")
        self.max_inference_batch_size = int(max_inference_batch_size)
        self._model_factory = model_factory
        self._model_identity = model_identity_from_manifest(
            self.model_path,
            injected_fixture=model_factory is not None,
        )
        self._forced_aligner_identity = (
            model_identity_from_manifest(
                self.forced_aligner_path,
                injected_fixture=model_factory is not None,
            )
            if self.forced_aligner_path is not None
            else None
        )
        self._model_instance: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def evidence_cache_identity(self) -> dict[str, Any]:
        """Invalidate ASR caches when evidence or local model identity changes."""

        return {
            "candidateSetSchemaVersion": ASR_CANDIDATE_SET_SCHEMA_VERSION,
            "asrModel": dict(self._model_identity),
            "forcedAlignerModel": (
                dict(self._forced_aligner_identity)
                if self._forced_aligner_identity is not None
                else None
            ),
        }

    def _model(self) -> Any:
        with self._load_lock:
            if self._model_instance is None:
                _windows_low_commit_layout_required(
                    self.model_path,
                    "Qwen3-ASR model",
                )
                if self.forced_aligner_path is not None:
                    _windows_low_commit_layout_required(
                        self.forced_aligner_path,
                        "Qwen3 forced aligner",
                    )
                factory = self._model_factory
                if factory is None:
                    try:
                        from qwen_asr import Qwen3ASRModel
                    except ImportError as exc:
                        raise WorkerError(
                            "QWEN3_ASR_RUNTIME_MISSING",
                            "qwen-asr is required for production transcription",
                        ) from exc
                    factory = Qwen3ASRModel.from_pretrained
                probe_accelerator = self._model_factory is None
                model_policy = _qwen_load_policy(
                    self.device_map,
                    forced_aligner=False,
                    probe_accelerator=probe_accelerator,
                )
                kwargs: dict[str, Any] = {
                    "pretrained_model_name_or_path": str(self.model_path),
                    **model_policy,
                    "dtype": self.torch_dtype,
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                    "max_inference_batch_size": (
                        self.max_inference_batch_size
                    ),
                }
                if self.forced_aligner_path is not None:
                    kwargs["forced_aligner"] = str(self.forced_aligner_path)
                    aligner_policy = _qwen_load_policy(
                        self.device_map,
                        forced_aligner=True,
                        probe_accelerator=probe_accelerator,
                    )
                    kwargs["forced_aligner_kwargs"] = {
                        **aligner_policy,
                        "dtype": self.torch_dtype,
                        "local_files_only": True,
                        "low_cpu_mem_usage": True,
                    }
                try:
                    self._model_instance = factory(**kwargs)
                except Exception as exc:
                    resource_error = _qwen_resource_error(
                        exc,
                        phase="model-load",
                        requested_device_map=self.device_map,
                        max_inference_batch_size=(
                            self.max_inference_batch_size
                        ),
                    )
                    if resource_error is None:
                        raise
                    _release_accelerator_memory()
                    raise resource_error from exc
            return self._model_instance

    def release_resources(self) -> None:
        """Idempotently unload ASR and forced-aligner weights between stages.

        The complete Qwen object owns both model graphs.  Dropping the final
        reference under the inference/load locks, followed by a full Python
        collection and CUDA allocator flush, prevents CAM++ from competing
        with resident ASR weights on memory-constrained accelerators.
        """

        with self._inference_lock:
            with self._load_lock:
                model = self._model_instance
                self._model_instance = None
            if model is not None:
                del model
        _release_accelerator_memory()

    def validate_requested_language(self, requested_language: str) -> str:
        """Fail before media preparation when an explicit prompt is unsupported."""

        try:
            normalized = normalize_language_tag(
                requested_language,
                allow_auto=True,
            )
            qwen_language_for_request(normalized)
        except ValueError as exc:
            raise WorkerError(
                "QWEN3_ASR_LANGUAGE_UNSUPPORTED",
                "The installed Qwen3-ASR model does not support the requested explicit language",
                details={
                    "requestedLanguage": str(requested_language),
                    "supportedPrimaryLanguageTags": list(
                        qwen_supported_primary_language_tags()
                    ),
                    "autoDetectionAvailable": True,
                },
            ) from exc
        return normalized

    def transcribe_batch(
        self,
        prepared: PreparedAudio,
        windows: Sequence[SpeechWindow],
        context: AdapterContext,
        *,
        requested_language: str,
    ) -> list[AsrHypothesis]:
        context.raise_if_cancelled()
        normalized_request_language = self.validate_requested_language(
            requested_language
        )
        qwen_language = qwen_language_for_request(normalized_request_language)
        if not prepared.audio_path or not Path(prepared.audio_path).is_file():
            raise WorkerError(
                "PREPARED_AUDIO_MISSING",
                "Qwen3-ASR requires a persisted normalized audio path",
            )
        with self._inference_lock:
            model = self._model()
            context.raise_if_cancelled()
            samples, sample_rate, pcm_buffer_id = _shared_pcm_for_prepared(
                prepared
            )
            audio_batch = [
                (
                    _slice_audio(
                        samples,
                        sample_rate,
                        window.start_ms,
                        window.end_ms,
                    ),
                    sample_rate,
                )
                for window in windows
            ]
            transcribe_kwargs: dict[str, Any] = {
                "audio": audio_batch,
                "return_time_stamps": self.forced_aligner_path is not None,
            }
            if qwen_language is not None:
                transcribe_kwargs["language"] = [qwen_language] * len(
                    audio_batch
                )
            try:
                results = model.transcribe(**transcribe_kwargs)
            except Exception as exc:
                resource_error = _qwen_resource_error(
                    exc,
                    phase="inference",
                    requested_device_map=self.device_map,
                    max_inference_batch_size=(
                        self.max_inference_batch_size
                    ),
                )
                if resource_error is None:
                    raise
                _release_accelerator_memory()
                raise resource_error from exc
            if (
                isinstance(results, Sequence)
                and not isinstance(results, (str, bytes))
                and len(results) == len(windows)
            ):
                results = list(results)
                for index, result in enumerate(results):
                    if str(getattr(result, "text", "") or "").strip():
                        continue
                    context.raise_if_cancelled()
                    retry_kwargs: dict[str, Any] = {
                        "audio": [audio_batch[index]],
                        "return_time_stamps": (
                            self.forced_aligner_path is not None
                        ),
                    }
                    if qwen_language is not None:
                        retry_kwargs["language"] = [qwen_language]
                    try:
                        retry_results = model.transcribe(**retry_kwargs)
                    except Exception as exc:
                        resource_error = _qwen_resource_error(
                            exc,
                            phase="inference-retry",
                            requested_device_map=self.device_map,
                            max_inference_batch_size=1,
                        )
                        if resource_error is None:
                            raise
                        _release_accelerator_memory()
                        raise resource_error from exc
                    if (
                        isinstance(retry_results, Sequence)
                        and not isinstance(retry_results, (str, bytes))
                        and len(retry_results) == 1
                    ):
                        results[index] = retry_results[0]
        context.raise_if_cancelled()
        if (
            not isinstance(results, Sequence)
            or isinstance(results, (str, bytes))
            or len(results) != len(windows)
        ):
            raise WorkerError(
                "QWEN3_ASR_RESULT_INVALID",
                "Qwen3-ASR must return one result per speech window",
            )
        output: list[AsrHypothesis] = []
        for window, result in zip(windows, results):
            text = str(getattr(result, "text", "") or "").strip()
            if not text:
                output.append(
                    AsrHypothesis(
                        window_id=window.window_id,
                        text="",
                        confidence=0.0,
                        evidence={
                            "model": "Qwen3-ASR-1.7B",
                            "pcmBufferId": pcm_buffer_id,
                            "confidenceAvailable": False,
                            "requestedLanguage": normalized_request_language,
                            "qwenPromptLanguage": qwen_language,
                            "disposition": "rejected-non-lexical",
                            "rejectionReason": (
                                "EMPTY_AFTER_INDIVIDUAL_RETRY"
                            ),
                            "individualRetryCount": 1,
                            "windowDurationMs": (
                                window.end_ms - window.start_ms
                            ),
                            "timestamps": [],
                        },
                    )
                )
                continue
            timestamp_result = getattr(result, "time_stamps", None)
            timestamps: list[dict[str, Any]] = []
            if timestamp_result is not None:
                items = getattr(timestamp_result, "items", None)
                if (
                    not isinstance(items, Sequence)
                    or isinstance(items, (str, bytes, bytearray))
                ):
                    raise WorkerError(
                        "QWEN3_ASR_TIMESTAMP_INVALID",
                        "Qwen3-ASR forced alignment must expose an items array",
                        details={"windowId": window.window_id},
                    )
                for index, item in enumerate(items):
                    try:
                        raw_start = getattr(item, "start_time")
                        raw_end = getattr(item, "end_time")
                        if isinstance(raw_start, bool) or isinstance(raw_end, bool):
                            raise TypeError("boolean timestamp")
                        start_seconds = float(raw_start)
                        end_seconds = float(raw_end)
                    except (AttributeError, TypeError, ValueError) as exc:
                        raise WorkerError(
                            "QWEN3_ASR_TIMESTAMP_INVALID",
                            "Qwen3-ASR returned a malformed forced-alignment item",
                            details={
                                "windowId": window.window_id,
                                "itemIndex": index,
                                "exceptionType": type(exc).__name__,
                            },
                        ) from exc
                    if not math.isfinite(start_seconds) or not math.isfinite(
                        end_seconds
                    ):
                        raise WorkerError(
                            "QWEN3_ASR_TIMESTAMP_INVALID",
                            "Qwen3-ASR returned a non-finite forced-alignment timestamp",
                            details={
                                "windowId": window.window_id,
                                "itemIndex": index,
                            },
                        )
                    start_ms = window.start_ms + round(start_seconds * 1000.0)
                    end_ms = window.start_ms + round(end_seconds * 1000.0)
                    start_ms = max(
                        window.start_ms,
                        min(start_ms, window.end_ms),
                    )
                    end_ms = max(
                        start_ms,
                        min(end_ms, window.end_ms),
                    )
                    timestamps.append(
                        {
                            "text": str(getattr(item, "text", "")),
                            "startMs": start_ms,
                            "endMs": end_ms,
                        }
                    )
            raw_language = getattr(result, "language", None)
            language_candidates = normalize_qwen_language_candidates(
                raw_language
            )
            window_language = reconcile_detected_languages(
                [
                    {
                        "languageCandidates": language_candidates,
                        "speechDurationMs": window.end_ms - window.start_ms,
                    }
                ],
                requested_language=normalized_request_language,
            )
            acoustic_score = _optional_result_score(
                result,
                ("acoustic_score", "acousticScore"),
            )
            decode_score = _optional_result_score(
                result,
                (
                    "decode_score",
                    "decodeScore",
                    "avg_logprob",
                    "average_logprob",
                    "score",
                ),
            )
            candidate_set = build_asr_candidate_set(
                model_id="Qwen3-ASR-1.7B",
                model_revision=self._model_identity["modelRevision"],
                model_manifest_sha256=(
                    self._model_identity["modelManifestSha256"]
                ),
                model_identity_status=(
                    self._model_identity["modelIdentityStatus"]
                ),
                source_audio_sha256=prepared.source_fingerprint,
                normalization_profile=prepared.normalization_profile,
                source_window_id=window.window_id,
                start_ms=window.start_ms,
                end_ms=window.end_ms,
                hypotheses=[
                    {
                        "text": text,
                        "language": window_language,
                        "tokens": [
                            item
                            for item in timestamps
                            if str(item.get("text") or "").strip()
                        ],
                        "acousticScore": acoustic_score,
                        "acousticScoreStatus": (
                            "available"
                            if acoustic_score is not None
                            else "provider-unavailable"
                        ),
                        "decodeScore": decode_score,
                        "decodeScoreStatus": (
                            "available"
                            if decode_score is not None
                            else "provider-unavailable"
                        ),
                    }
                ],
            )
            output.append(
                AsrHypothesis(
                    window_id=window.window_id,
                    text=text,
                    confidence=0.0,
                    evidence={
                        "model": "Qwen3-ASR-1.7B",
                        "pcmBufferId": pcm_buffer_id,
                        "confidenceAvailable": False,
                        "requestedLanguage": normalized_request_language,
                        "qwenPromptLanguage": qwen_language,
                        "rawLanguage": (
                            raw_language
                            if isinstance(
                                raw_language,
                                (str, int, float, bool, list, tuple, type(None)),
                            )
                            else str(raw_language)
                        ),
                        "languageCandidates": list(language_candidates),
                        "language": window_language,
                        "timestamps": timestamps,
                        **candidate_set,
                    },
                )
            )
        return output


def _to_vector(value: Any) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 1
        and isinstance(value[0], Sequence)
    ):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise WorkerError(
            "SPEAKER_EMBEDDING_RESULT_INVALID",
            "speaker model did not return an embedding vector",
        )
    vector = tuple(float(item) for item in value)
    if not vector or any(not math.isfinite(item) for item in vector):
        raise WorkerError(
            "SPEAKER_EMBEDDING_RESULT_INVALID",
            "speaker embedding contains invalid values",
        )
    return vector


class LocalFunAsrCamPlusAdapter:
    """Full-corpus CAM++ runner plus conservative VAD-internal refinement.

    One shared, normalized PCM timeline and one lazy CAM++ model instance are
    reused for both multi-resolution change detection and final turn
    embeddings.  Only acoustically strong proposals are applied
    automatically; every other proposal remains attached to the source VAD as
    auditable review evidence.
    """

    adapter_id = "CAM++"
    version = "2.4.0"
    refinement_method = "cam-plus-multiresolution-language-window-v5"

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "cuda:0",
        embedding_batch_size: int = 128,
        fine_window_ms: int = 1_200,
        fine_step_ms: int = 375,
        context_window_ms: int = 3_000,
        context_step_ms: int = 1_200,
        merge_radius_ms: int = 250,
        cross_resolution_support_radius_ms: int = 600,
        min_consensus_change_score: float = 0.46,
        min_consensus_acoustic_confidence: float = 0.60,
        min_cross_resolution_support_score: float = 0.20,
        min_local_peak_prominence: float = 0.05,
        min_resulting_interval_ms: int = 700,
        max_language_window_ms: int = 12_000,
        language_split_search_ms: int = 1_000,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = _local_model_path(model_path, "CAM++ model")
        self.device = str(device)
        self.embedding_batch_size = int(embedding_batch_size)
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        self.fine_window_ms = int(fine_window_ms)
        self.fine_step_ms = int(fine_step_ms)
        self.context_window_ms = int(context_window_ms)
        self.context_step_ms = int(context_step_ms)
        self.merge_radius_ms = int(merge_radius_ms)
        self.cross_resolution_support_radius_ms = int(
            cross_resolution_support_radius_ms
        )
        self.min_consensus_change_score = float(min_consensus_change_score)
        self.min_consensus_acoustic_confidence = float(
            min_consensus_acoustic_confidence
        )
        self.min_cross_resolution_support_score = float(
            min_cross_resolution_support_score
        )
        self.min_local_peak_prominence = float(min_local_peak_prominence)
        self.min_resulting_interval_ms = int(min_resulting_interval_ms)
        self.max_language_window_ms = int(max_language_window_ms)
        self.language_split_search_ms = int(language_split_search_ms)
        for field_name in (
            "fine_window_ms",
            "fine_step_ms",
            "context_window_ms",
            "context_step_ms",
            "merge_radius_ms",
            "cross_resolution_support_radius_ms",
            "min_resulting_interval_ms",
            "max_language_window_ms",
            "language_split_search_ms",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "min_consensus_change_score",
            "min_consensus_acoustic_confidence",
            "min_cross_resolution_support_score",
            "min_local_peak_prominence",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if (
            self.max_language_window_ms
            <= self.language_split_search_ms
            + self.min_resulting_interval_ms
        ):
            raise ValueError(
                "max_language_window_ms must exceed "
                "language_split_search_ms + min_resulting_interval_ms"
            )
        self._model_factory = model_factory
        self._model_instance: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.RLock()

    def refinement_identity(self) -> Mapping[str, Any]:
        """Stable cache identity for the VAD-internal refinement stage."""

        return {
            "method": self.refinement_method,
            "adapterId": self.adapter_id,
            "adapterVersion": self.version,
            "fineWindowMs": self.fine_window_ms,
            "fineStepMs": self.fine_step_ms,
            "contextWindowMs": self.context_window_ms,
            "contextStepMs": self.context_step_ms,
            "mergeRadiusMs": self.merge_radius_ms,
            "crossResolutionSupportRadiusMs": (
                self.cross_resolution_support_radius_ms
            ),
            "minConsensusChangeScore": self.min_consensus_change_score,
            "minConsensusAcousticConfidence": (
                self.min_consensus_acoustic_confidence
            ),
            "minCrossResolutionSupportScore": (
                self.min_cross_resolution_support_score
            ),
            "minLocalPeakProminence": self.min_local_peak_prominence,
            "minResultingIntervalMs": self.min_resulting_interval_ms,
            "maxLanguageWindowMs": self.max_language_window_ms,
            "languageSplitSearchMs": self.language_split_search_ms,
            "embeddingBatchSize": self.embedding_batch_size,
        }

    def _model(self) -> Any:
        with self._load_lock:
            if self._model_instance is None:
                factory = self._model_factory
                if factory is None:
                    try:
                        from funasr import AutoModel
                    except ImportError as exc:
                        raise WorkerError(
                            "FUNASR_RUNTIME_MISSING",
                            "FunASR is required for CAM++",
                        ) from exc
                    factory = AutoModel
                try:
                    self._model_instance = factory(
                        model=str(self.model_path),
                        device=self.device,
                        disable_update=True,
                    )
                except Exception as exc:
                    resource_error = _campp_resource_error(
                        exc,
                        phase="model-load",
                        requested_device=self.device,
                        embedding_batch_size=self.embedding_batch_size,
                    )
                    if resource_error is None:
                        raise
                    _release_accelerator_memory()
                    raise resource_error from exc
            return self._model_instance

    def release_resources(self) -> None:
        """Idempotently unload CAM++ before another heavyweight GPU stage."""

        with self._inference_lock:
            with self._load_lock:
                model = self._model_instance
                self._model_instance = None
        if model is not None:
            del model
        _release_accelerator_memory()

    @staticmethod
    def _extract_rows(raw: Any, expected: int) -> list[tuple[float, ...]]:
        if isinstance(raw, Mapping):
            raw = [raw]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise WorkerError(
                "SPEAKER_EMBEDDING_RESULT_INVALID",
                "CAM++ result must be an array",
            )
        rows: list[tuple[float, ...]] = []
        if len(raw) == expected and all(isinstance(item, Mapping) for item in raw):
            rows = [_to_vector(item.get("spk_embedding")) for item in raw]
        elif len(raw) == 1 and isinstance(raw[0], Mapping):
            matrix = raw[0].get("spk_embedding")
            if hasattr(matrix, "detach"):
                matrix = matrix.detach()
            if hasattr(matrix, "cpu"):
                matrix = matrix.cpu()
            if hasattr(matrix, "tolist"):
                matrix = matrix.tolist()
            if (
                isinstance(matrix, Sequence)
                and not isinstance(matrix, (str, bytes))
                and len(matrix) == expected
            ):
                rows = [_to_vector(item) for item in matrix]
        if len(rows) != expected:
            raise WorkerError(
                "SPEAKER_EMBEDDING_RESULT_INVALID",
                "CAM++ must return one embedding per speech window",
            )
        return rows

    def _embed_slices(
        self,
        slices: Sequence[Any],
        context: AdapterContext,
    ) -> list[tuple[float, ...]]:
        rows: list[tuple[float, ...]] = []
        with self._inference_lock:
            model = self._model()
            for offset in range(0, len(slices), self.embedding_batch_size):
                context.raise_if_cancelled()
                batch = list(
                    slices[offset : offset + self.embedding_batch_size]
                )
                try:
                    raw = model.generate(
                        input=batch,
                        batch_size=len(batch),
                        disable_pbar=True,
                    )
                except Exception as exc:
                    resource_error = _campp_resource_error(
                        exc,
                        phase="embedding-inference",
                        requested_device=self.device,
                        embedding_batch_size=len(batch),
                    )
                    if resource_error is None:
                        raise
                    self.release_resources()
                    raise resource_error from exc
                rows.extend(self._extract_rows(raw, len(batch)))
        return rows

    @staticmethod
    def _stable_starts(
        start_ms: int,
        end_ms: int,
        window_ms: int,
        step_ms: int,
    ) -> tuple[int, ...]:
        """Return deterministic, end-covering sliding-window starts."""

        duration_ms = end_ms - start_ms
        if duration_ms <= window_ms:
            return (start_ms,)
        last_start = end_ms - window_ms
        starts = list(range(start_ms, last_start + 1, step_ms))
        if starts[-1] != last_start:
            starts.append(last_start)
        return tuple(starts)

    def _sliding_windows(
        self,
        source: SpeechWindow,
        *,
        resolution: str,
        window_ms: int,
        step_ms: int,
    ) -> tuple[SpeechWindow, ...]:
        starts = self._stable_starts(
            source.start_ms,
            source.end_ms,
            window_ms,
            step_ms,
        )
        output: list[SpeechWindow] = []
        for index, start_ms in enumerate(starts, start=1):
            end_ms = min(source.end_ms, start_ms + window_ms)
            output.append(
                SpeechWindow(
                    window_id=(
                        f"{source.window_id}.{resolution}{index:04d}."
                        f"{start_ms}-{end_ms}"
                    ),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    metadata={
                        "sourceVadWindowId": source.window_id,
                        "resolution": resolution,
                    },
                )
            )
        return tuple(output)

    @staticmethod
    def _energy_valleys(
        samples: Any,
        sample_rate: int,
        source: SpeechWindow,
        *,
        frame_ms: int = 40,
        step_ms: int = 20,
    ) -> tuple[EnergyValley, ...]:
        """Find inexpensive PCM energy minima used only for localization.

        The acoustic CAM++ change remains the sole candidate source.  If the
        PCM object cannot be represented as a numeric array (for example a
        lightweight test double), localization simply falls back to the
        acoustic midpoint.
        """

        try:
            import numpy as np

            raw = _slice_audio(
                samples,
                sample_rate,
                source.start_ms,
                source.end_ms,
            )
            values = np.asarray(raw, dtype=np.float32).reshape(-1)
            frame_samples = max(1, round(frame_ms * sample_rate / 1000))
            step_samples = max(1, round(step_ms * sample_rate / 1000))
            if len(values) < frame_samples * 3:
                return ()
            squared = np.square(values, dtype=np.float64)
            cumulative = np.concatenate(
                (np.zeros(1, dtype=np.float64), np.cumsum(squared))
            )
            starts = np.arange(
                0,
                len(values) - frame_samples + 1,
                step_samples,
                dtype=np.int64,
            )
            energies = (
                cumulative[starts + frame_samples] - cumulative[starts]
            ) / frame_samples
            if len(energies) < 3:
                return ()
            valleys: list[EnergyValley] = []
            for index in range(1, len(energies) - 1):
                current = float(energies[index])
                left = float(energies[index - 1])
                right = float(energies[index + 1])
                if current > left or current >= right:
                    continue
                neighborhood = max(left, right, 1e-12)
                confidence = max(
                    0.0,
                    min(1.0, (neighborhood - current) / neighborhood),
                )
                if confidence < 0.20:
                    continue
                relative_ms = round(
                    (int(starts[index]) + frame_samples / 2)
                    * 1000
                    / sample_rate
                )
                timestamp_ms = min(
                    source.end_ms - 1,
                    max(source.start_ms, source.start_ms + relative_ms),
                )
                valleys.append(
                    EnergyValley(
                        timestamp_ms=timestamp_ms,
                        confidence=round(confidence, 12),
                        marker_id=(
                            f"energy:{source.window_id}:{timestamp_ms}"
                        ),
                    )
                )
            return tuple(valleys)
        except (ImportError, TypeError, ValueError, AttributeError):
            return ()

    @staticmethod
    def _proposal_priority(
        resolution: str,
        proposal: Any,
    ) -> tuple[float, float, int, int, str]:
        return (
            -float(proposal.change_score),
            -float(proposal.acoustic_confidence),
            0 if resolution == "fine" else 1,
            int(proposal.split_ms),
            str(proposal.proposal_id),
        )

    def _cross_resolution_consensus(
        self,
        resolution: str,
        proposal: Any,
        candidates: Sequence[tuple[str, Any]],
    ) -> Mapping[str, Any] | None:
        """Recommend review for a strong, locally prominent multi-scale peak."""

        if tuple(proposal.review_reasons) != ("LOW_ACOUSTIC_CONFIDENCE",):
            return None
        change_score = float(proposal.change_score)
        acoustic_confidence = float(proposal.acoustic_confidence)
        if (
            change_score < self.min_consensus_change_score
            or acoustic_confidence
            < self.min_consensus_acoustic_confidence
        ):
            return None

        acoustic_boundary_ms = int(proposal.acoustic_boundary_ms)
        same_resolution_neighbors = [
            item
            for candidate_resolution, item in candidates
            if candidate_resolution == resolution
            and item.proposal_id != proposal.proposal_id
            and abs(int(item.acoustic_boundary_ms) - acoustic_boundary_ms)
            <= self.cross_resolution_support_radius_ms
        ]
        strongest_neighbor_score = max(
            (
                float(item.change_score)
                for item in same_resolution_neighbors
            ),
            default=0.0,
        )
        local_prominence = round(
            change_score - strongest_neighbor_score,
            12,
        )
        if local_prominence < self.min_local_peak_prominence:
            return None

        cross_resolution_support = [
            (candidate_resolution, item)
            for candidate_resolution, item in candidates
            if candidate_resolution != resolution
            and abs(int(item.acoustic_boundary_ms) - acoustic_boundary_ms)
            <= self.cross_resolution_support_radius_ms
            and float(item.change_score)
            >= self.min_cross_resolution_support_score
        ]
        if not cross_resolution_support:
            return None
        cross_resolution_support.sort(
            key=lambda item: (
                abs(
                    int(item[1].acoustic_boundary_ms)
                    - acoustic_boundary_ms
                ),
                self._proposal_priority(item[0], item[1]),
            )
        )
        support_resolution, support = cross_resolution_support[0]
        if (
            proposal.boundary_source == "ACOUSTIC_MIDPOINT"
            and support.boundary_source == "ACOUSTIC_MIDPOINT"
        ):
            return None
        return {
            "reasonCode": (
                "CROSS_RESOLUTION_LOCAL_PROMINENCE_REVIEW_RECOMMENDATION"
            ),
            "applicationPolicy": "review-only",
            "applyAutomatically": False,
            "proposalId": str(proposal.proposal_id),
            "resolution": resolution,
            "splitMs": int(proposal.split_ms),
            "acousticBoundaryMs": acoustic_boundary_ms,
            "changeScore": change_score,
            "acousticConfidence": acoustic_confidence,
            "localPeakProminence": local_prominence,
            "support": {
                "proposalId": str(support.proposal_id),
                "resolution": support_resolution,
                "splitMs": int(support.split_ms),
                "acousticBoundaryMs": int(support.acoustic_boundary_ms),
                "changeScore": float(support.change_score),
                "boundarySource": str(support.boundary_source),
            },
            "thresholds": {
                "supportRadiusMs": self.cross_resolution_support_radius_ms,
                "minChangeScore": self.min_consensus_change_score,
                "minAcousticConfidence": (
                    self.min_consensus_acoustic_confidence
                ),
                "minSupportChangeScore": (
                    self.min_cross_resolution_support_score
                ),
                "minLocalPeakProminence": (
                    self.min_local_peak_prominence
                ),
            },
        }

    def _automatic_split_analysis(
        self,
        source: SpeechWindow,
        plans: Mapping[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Mapping[str, Any], ...]]:
        """Merge cross-resolution proposals and enforce safe turn lengths."""

        candidates: list[tuple[str, Any]] = []
        for resolution in sorted(plans):
            candidates.extend(
                (resolution, proposal)
                for proposal in plans[resolution].proposals
            )
        candidates.sort(
            key=lambda item: (
                int(item[1].split_ms),
                self._proposal_priority(item[0], item[1]),
            )
        )
        clusters: list[list[tuple[str, Any]]] = []
        for item in candidates:
            if (
                not clusters
                or int(item[1].split_ms)
                - int(clusters[-1][-1][1].split_ms)
                > self.merge_radius_ms
            ):
                clusters.append([item])
            else:
                clusters[-1].append(item)

        selected: list[tuple[str, Any]] = []
        consensus_recommendations: list[Mapping[str, Any]] = []
        for cluster in clusters:
            winner = min(
                cluster,
                key=lambda item: self._proposal_priority(
                    item[0], item[1]
                ),
            )
            if winner[1].apply_automatically:
                selected.append(winner)
                continue
            consensus = self._cross_resolution_consensus(
                winner[0],
                winner[1],
                candidates,
            )
            if consensus is not None:
                consensus_recommendations.append(consensus)

        selected.sort(key=lambda item: int(item[1].split_ms))
        selected = [
            item
            for item in selected
            if int(item[1].split_ms) - source.start_ms
            >= self.min_resulting_interval_ms
            and source.end_ms - int(item[1].split_ms)
            >= self.min_resulting_interval_ms
        ]
        while True:
            conflict_index = next(
                (
                    index
                    for index in range(len(selected) - 1)
                    if int(selected[index + 1][1].split_ms)
                    - int(selected[index][1].split_ms)
                    < self.min_resulting_interval_ms
                ),
                None,
            )
            if conflict_index is None:
                break
            left = selected[conflict_index]
            right = selected[conflict_index + 1]
            if self._proposal_priority(left[0], left[1]) <= (
                self._proposal_priority(right[0], right[1])
            ):
                del selected[conflict_index + 1]
            else:
                del selected[conflict_index]
        return (
            tuple(int(item[1].split_ms) for item in selected),
            tuple(dict(item) for item in consensus_recommendations),
        )

    def _automatic_splits(
        self,
        source: SpeechWindow,
        plans: Mapping[str, Any],
    ) -> tuple[int, ...]:
        return self._automatic_split_analysis(source, plans)[0]

    def _language_duration_splits(
        self,
        source: SpeechWindow,
        speaker_change_splits: Sequence[int],
        energy_valleys: Sequence[EnergyValley],
    ) -> tuple[tuple[int, ...], tuple[Mapping[str, Any], ...]]:
        """Bound ASR language windows without inventing speaker changes."""

        speaker_boundaries = (
            source.start_ms,
            *speaker_change_splits,
            source.end_ms,
        )
        split_points: list[int] = []
        split_evidence: list[Mapping[str, Any]] = []
        for span_start, span_end in zip(
            speaker_boundaries,
            speaker_boundaries[1:],
        ):
            cursor = span_start
            while span_end - cursor > self.max_language_window_ms:
                latest_split = min(
                    cursor + self.max_language_window_ms,
                    span_end - self.min_resulting_interval_ms,
                )
                target_split = max(
                    cursor + self.min_resulting_interval_ms,
                    latest_split - self.language_split_search_ms,
                )
                earliest_split = max(
                    cursor + self.min_resulting_interval_ms,
                    target_split - self.language_split_search_ms,
                )
                candidates = [
                    marker
                    for marker in energy_valleys
                    if earliest_split
                    <= marker.timestamp_ms
                    <= latest_split
                ]
                selected = (
                    min(
                        candidates,
                        key=lambda marker: (
                            abs(marker.timestamp_ms - target_split),
                            -marker.confidence,
                            marker.timestamp_ms,
                            marker.marker_id,
                        ),
                    )
                    if candidates
                    else None
                )
                split_ms = (
                    selected.timestamp_ms
                    if selected is not None
                    else target_split
                )
                if split_ms <= cursor or split_ms >= span_end:
                    raise WorkerError(
                        "LANGUAGE_WINDOW_SPLIT_INVALID",
                        "language window refinement could not make progress",
                        details={"sourceWindowId": source.window_id},
                    )
                split_points.append(split_ms)
                split_evidence.append(
                    {
                        "targetMs": target_split,
                        "splitMs": split_ms,
                        "searchStartMs": earliest_split,
                        "searchEndMs": latest_split,
                        "localizer": (
                            "energy-valley"
                            if selected is not None
                            else "deterministic-duration-cap"
                        ),
                        **(
                            {
                                "energyValleyMarkerId": selected.marker_id,
                                "energyValleyConfidence": (
                                    selected.confidence
                                ),
                            }
                            if selected is not None
                            else {}
                        ),
                    }
                )
                cursor = split_ms
        return tuple(split_points), tuple(split_evidence)

    def refine_windows(
        self,
        prepared: PreparedAudio,
        context: AdapterContext,
    ) -> PreparedAudio:
        """Split long FunASR VAD spans at strong multi-resolution changes."""

        context.raise_if_cancelled()
        started = time.perf_counter()
        samples, sample_rate, pcm_buffer_id = _shared_pcm_for_prepared(
            prepared
        )
        pcm_timeline = Pcm16kMonoTimeline(
            buffer_id=pcm_buffer_id,
            sample_count=len(samples),
            sample_rate_hz=sample_rate,
        )
        detection_config = SpeakerChangeDetectionConfig(
            min_resulting_interval_ms=self.min_resulting_interval_ms,
        )

        descriptors: list[tuple[SpeechWindow, str, SpeechWindow]] = []
        by_source: dict[
            str,
            dict[str, tuple[SpeechWindow, ...]],
        ] = {}
        for source in prepared.windows:
            resolutions = {
                "fine": self._sliding_windows(
                    source,
                    resolution="fine",
                    window_ms=self.fine_window_ms,
                    step_ms=self.fine_step_ms,
                ),
                "context": self._sliding_windows(
                    source,
                    resolution="context",
                    window_ms=self.context_window_ms,
                    step_ms=self.context_step_ms,
                ),
            }
            by_source[source.window_id] = resolutions
            for resolution in ("fine", "context"):
                descriptors.extend(
                    (source, resolution, window)
                    for window in resolutions[resolution]
                )

        slices = [
            _slice_audio(
                samples,
                sample_rate,
                window.start_ms,
                window.end_ms,
            )
            for _source, _resolution, window in descriptors
        ]
        rows = self._embed_slices(slices, context)
        if len(rows) != len(descriptors):
            raise WorkerError(
                "SPEAKER_EMBEDDING_RESULT_INVALID",
                "CAM++ refinement lost a sliding-window embedding",
            )
        embeddings: dict[str, SpeakerEmbeddingWindow] = {}
        for (source, resolution, window), vector in zip(
            descriptors,
            rows,
        ):
            raw_confidence = source.metadata.get(
                "speakerChangeEmbeddingConfidence",
                1.0,
            )
            try:
                confidence = max(0.0, min(1.0, float(raw_confidence)))
            except (TypeError, ValueError):
                confidence = 0.0
            embeddings[window.window_id] = SpeakerEmbeddingWindow(
                window_id=window.window_id,
                start_ms=window.start_ms,
                end_ms=window.end_ms,
                embedding=vector,
                confidence=confidence,
                overlap_risk=bool(
                    source.metadata.get("overlapRisk", False)
                ),
                evidence={
                    "model": "CAM++",
                    "resolution": resolution,
                    "sourceVadWindowId": source.window_id,
                    "pcmBufferId": pcm_buffer_id,
                },
            )

        refined: list[SpeechWindow] = []
        for source in prepared.windows:
            context.raise_if_cancelled()
            energy_valleys = self._energy_valleys(
                samples,
                sample_rate,
                source,
            )
            plans: dict[str, Any] = {}
            for resolution in ("fine", "context"):
                plans[resolution] = plan_speaker_changes(
                    vad_start_ms=source.start_ms,
                    vad_end_ms=source.end_ms,
                    windows=tuple(
                        embeddings[window.window_id]
                        for window in by_source[source.window_id][resolution]
                    ),
                    energy_valleys=energy_valleys,
                    pcm_timeline=pcm_timeline,
                    config=detection_config,
                )
            speaker_change_splits = self._automatic_splits(source, plans)
            _analyzed_splits, consensus_recommendations = (
                self._automatic_split_analysis(source, plans)
            )
            (
                language_duration_splits,
                language_split_evidence,
            ) = self._language_duration_splits(
                source,
                speaker_change_splits,
                energy_valleys,
            )
            applied_splits = tuple(
                sorted(
                    {
                        *speaker_change_splits,
                        *language_duration_splits,
                    }
                )
            )
            evidence = {
                "version": self.version,
                "method": self.refinement_method,
                "pcmBufferId": pcm_buffer_id,
                "automaticSplitsMs": list(speaker_change_splits),
                "speakerChangeSplitsMs": list(speaker_change_splits),
                "crossResolutionConsensusRecommendations": [
                    dict(item) for item in consensus_recommendations
                ],
                "languageDurationSplitsMs": list(language_duration_splits),
                "appliedSplitsMs": list(applied_splits),
                "maxLanguageWindowMs": self.max_language_window_ms,
                "languageSplitSearchMs": self.language_split_search_ms,
                "languageSplitEvidence": [
                    dict(item) for item in language_split_evidence
                ],
                "reviewRequired": any(
                    plan.review_required for plan in plans.values()
                ),
                "plans": {
                    resolution: plans[resolution].as_dict()
                    for resolution in ("fine", "context")
                },
            }
            boundaries = (
                source.start_ms,
                *applied_splits,
                source.end_ms,
            )
            if not applied_splits:
                metadata = dict(source.metadata)
                metadata.setdefault(
                    "sourceVadWindowId",
                    source.window_id,
                )
                metadata.setdefault(
                    "turnId",
                    (
                        f"turn:{source.window_id}:"
                        f"{source.start_ms}-{source.end_ms}"
                    ),
                )
                metadata["speakerChangeRefinement"] = evidence
                refined.append(replace(source, metadata=metadata))
                continue
            speaker_boundaries = (
                source.start_ms,
                *speaker_change_splits,
                source.end_ms,
            )
            source_turn_id = (
                str(source.metadata["turnId"]).strip()
                if isinstance(source.metadata.get("turnId"), str)
                and str(source.metadata["turnId"]).strip()
                else f"turn:{source.window_id}:{source.start_ms}-{source.end_ms}"
            )
            for index, (start_ms, end_ms) in enumerate(
                zip(boundaries, boundaries[1:]),
                start=1,
            ):
                speaker_span = next(
                    (
                        (speaker_start, speaker_end)
                        for speaker_start, speaker_end in zip(
                            speaker_boundaries,
                            speaker_boundaries[1:],
                        )
                        if speaker_start <= start_ms
                        and end_ms <= speaker_end
                    ),
                    None,
                )
                if speaker_span is None:
                    raise WorkerError(
                        "LANGUAGE_WINDOW_SPLIT_INVALID",
                        "refined language window crossed a speaker boundary",
                        details={"sourceWindowId": source.window_id},
                    )
                turn_id = (
                    source_turn_id
                    if not speaker_change_splits
                    else (
                        f"turn:{source.window_id}:"
                        f"{speaker_span[0]}-{speaker_span[1]}"
                    )
                )
                metadata = dict(source.metadata)
                metadata.update(
                    {
                        "sourceVadWindowId": source.window_id,
                        "turnId": turn_id,
                        "speakerChangeRefinement": evidence,
                    }
                )
                refined.append(
                    SpeechWindow(
                        window_id=(
                            f"{source.window_id}."
                            f"{'rf' if language_duration_splits else 'sc'}"
                            f"{index:02d}"
                        ),
                        start_ms=start_ms,
                        end_ms=end_ms,
                        boundary_conflict=source.boundary_conflict,
                        locked_speaker_id=source.locked_speaker_id,
                        metadata=metadata,
                    )
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings = dict(prepared.stage_durations_ms)
        timings["speakerChangeRefinement"] = elapsed_ms
        timings["boundary"] = float(timings.get("boundary", 0.0)) + elapsed_ms
        return replace(
            prepared,
            windows=tuple(refined),
            stage_durations_ms=timings,
        )

    def embed_batch(
        self,
        prepared: PreparedAudio,
        windows: Sequence[SpeechWindow],
        context: AdapterContext,
    ) -> list[EmbeddingRecord]:
        context.raise_if_cancelled()
        if not prepared.audio_path or not Path(prepared.audio_path).is_file():
            raise WorkerError(
                "PREPARED_AUDIO_MISSING",
                "CAM++ requires a persisted normalized audio path",
            )
        samples, sample_rate, pcm_buffer_id = _shared_pcm_for_prepared(
            prepared
        )
        slices = [
            _slice_audio(
                samples,
                sample_rate,
                window.start_ms,
                window.end_ms,
            )
            for window in windows
        ]
        rows = self._embed_slices(slices, context)
        context.raise_if_cancelled()
        return [
            EmbeddingRecord(
                window_id=window.window_id,
                vector=vector,
                confidence=1.0,
                evidence={
                    "model": "CAM++",
                    "pcmBufferId": pcm_buffer_id,
                    "embeddingBatchSize": self.embedding_batch_size,
                    "modelPathSha256": hashlib.sha256(
                        str(self.model_path).encode("utf-8")
                    ).hexdigest(),
                },
            )
            for window, vector in zip(windows, rows)
        ]


class LocalERes2NetV2Verifier:
    """Selective ERes2NetV2 verifier using top-2 speaker exemplars only."""

    adapter_id = "ERes2NetV2"
    version = "1.1.0"

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "gpu",
        decision_margin: float = 0.05,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = _local_model_path(model_path, "ERes2NetV2 model")
        self.device = str(device)
        self.decision_margin = float(decision_margin)
        if not 0.0 <= self.decision_margin <= 1.0:
            raise ValueError("decision_margin must be between 0 and 1")
        self._pipeline_factory = pipeline_factory
        self._pipeline_instance: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.RLock()

    def _pipeline(self) -> Any:
        with self._load_lock:
            if self._pipeline_instance is None:
                factory = self._pipeline_factory
                if factory is None:
                    try:
                        from modelscope.pipelines import pipeline
                        from modelscope.utils.constant import Tasks
                    except ImportError as exc:
                        raise WorkerError(
                            "ERES2NETV2_RUNTIME_MISSING",
                            "ModelScope speaker verification runtime is incomplete",
                        ) from exc

                    def factory(**kwargs: Any) -> Any:
                        return pipeline(task=Tasks.speaker_verification, **kwargs)

                try:
                    load_device = (
                        "cpu" if _is_mps_device(self.device) else self.device
                    )
                    pipeline = factory(
                        model=str(self.model_path),
                        device=load_device,
                    )
                    self._pipeline_instance = (
                        _prepare_eres2netv2_mps_pipeline(
                            pipeline,
                            requested_device=self.device,
                        )
                    )
                except Exception as exc:
                    resource_error = _eres2netv2_resource_error(
                        exc,
                        phase="model-load",
                        requested_device=self.device,
                        clip_count=0,
                    )
                    if resource_error is None:
                        raise
                    _release_accelerator_memory()
                    raise resource_error from exc
            return self._pipeline_instance

    def release_resources(self) -> None:
        """Idempotently unload the shared verifier between GPU stages."""

        with self._inference_lock:
            with self._load_lock:
                pipeline = self._pipeline_instance
                self._pipeline_instance = None
                if pipeline is not None:
                    del pipeline
                _release_accelerator_memory()

    @staticmethod
    def _ranked_speakers(segment: TranscriptSegment) -> tuple[str, ...]:
        ranked: list[str] = []
        seen: set[str] = set()
        for item in sorted(
            segment.speaker_scores,
            key=lambda score: (-score.score, score.speaker_id),
        ):
            if item.speaker_id in seen:
                continue
            seen.add(item.speaker_id)
            ranked.append(item.speaker_id)
            if len(ranked) == 2:
                break
        return tuple(ranked)

    @staticmethod
    def _candidate_segment(
        candidate: ReviewCandidate,
        segments: Mapping[str, TranscriptSegment],
    ) -> TranscriptSegment:
        try:
            return segments[candidate.segment_id]
        except KeyError:
            raise WorkerError(
                "ERES2NETV2_CANDIDATE_INVALID",
                "ERes2NetV2 candidate segment is missing from the transcript",
                details={"segmentId": candidate.segment_id},
            ) from None

    @staticmethod
    def _references_by_speaker(
        segments: Mapping[str, TranscriptSegment],
    ) -> dict[str, tuple[TranscriptSegment, ...]]:
        references: dict[str, list[TranscriptSegment]] = {}
        for segment in segments.values():
            if segment.overlapping or segment.speaker_margin <= 0.0:
                continue
            references.setdefault(segment.speaker_id, []).append(segment)
        return {
            speaker_id: tuple(
                sorted(
                    choices,
                    key=lambda segment: (
                        -segment.speaker_margin,
                        -(segment.end_ms - segment.start_ms),
                        segment.start_ms,
                        segment.segment_id,
                    ),
                )
            )
            for speaker_id, choices in references.items()
        }

    @staticmethod
    def _reference_segments(
        candidate: ReviewCandidate,
        segments: Mapping[str, TranscriptSegment],
        references_by_speaker: Mapping[
            str, Sequence[TranscriptSegment]
        ] | None = None,
    ) -> tuple[TranscriptSegment, ...]:
        target = LocalERes2NetV2Verifier._candidate_segment(
            candidate,
            segments,
        )
        reference_pool = (
            references_by_speaker
            if references_by_speaker is not None
            else LocalERes2NetV2Verifier._references_by_speaker(segments)
        )
        output: list[TranscriptSegment] = []
        for speaker_id in LocalERes2NetV2Verifier._ranked_speakers(target):
            reference = next(
                (
                    segment
                    for segment in reference_pool.get(speaker_id, ())
                    if segment.segment_id != target.segment_id
                ),
                None,
            )
            if reference is not None:
                output.append(reference)
        return tuple(output)

    def cache_material(
        self,
        candidate: ReviewCandidate,
        segments: Mapping[str, TranscriptSegment],
    ) -> Mapping[str, Any]:
        references_by_speaker = self._references_by_speaker(segments)
        return {
            "modelPath": str(self.model_path),
            "references": [
                {
                    "segmentId": segment.segment_id,
                    "speakerId": segment.speaker_id,
                    "startMs": segment.start_ms,
                    "endMs": segment.end_ms,
                    "audioPath": self._audio_path(segment),
                }
                for segment in self._reference_segments(
                    candidate,
                    segments,
                    references_by_speaker,
                )
            ],
        }

    @staticmethod
    def _audio_path(segment: TranscriptSegment) -> str | None:
        preparation = segment.evidence.get("preparation")
        audio_path = (
            preparation.get("audioPath")
            if isinstance(preparation, Mapping)
            else None
        )
        if isinstance(audio_path, str) and audio_path.strip():
            return audio_path
        return None

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise WorkerError(
                "ERES2NETV2_RESULT_INVALID",
                "ERes2NetV2 returned embeddings with inconsistent dimensions",
                details={
                    "leftDimensions": len(left),
                    "rightDimensions": len(right),
                },
            )
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(item * item for item in left))
        right_norm = math.sqrt(sum(item * item for item in right))
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            raise WorkerError(
                "ERES2NETV2_RESULT_INVALID",
                "ERes2NetV2 returned a zero-norm embedding",
            )
        return numerator / (left_norm * right_norm)

    def _embeddings(self, audio: Sequence[Any]) -> list[tuple[float, ...]]:
        resource_error: WorkerError | None = None
        with self._inference_lock:
            pipeline = self._pipeline()
            try:
                result = pipeline(list(audio), output_emb=True)
            except Exception as exc:
                resource_error = _eres2netv2_resource_error(
                    exc,
                    phase="embedding-inference",
                    requested_device=self.device,
                    clip_count=len(audio),
                )
                if resource_error is None:
                    raise
            if resource_error is not None:
                with self._load_lock:
                    resident = self._pipeline_instance
                    self._pipeline_instance = None
                    del pipeline
                    if resident is not None:
                        del resident
                    _release_accelerator_memory()
        if resource_error is not None:
            raise resource_error from None
        raw = result.get("embs") if isinstance(result, Mapping) else None
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != len(audio)
        ):
            raise WorkerError(
                "ERES2NETV2_RESULT_INVALID",
                "ERes2NetV2 must return one embedding per requested clip",
            )
        return [_to_vector(item) for item in raw]

    def review_batch(
        self,
        candidates: Sequence[ReviewCandidate],
        segments: Mapping[str, TranscriptSegment],
        context: AdapterContext,
    ) -> list[ReviewProposal]:
        candidate_ids = [
            candidate.segment_id for candidate in candidates
        ]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise WorkerError(
                "ERES2NETV2_CANDIDATE_INVALID",
                "ERes2NetV2 candidate batch contains duplicate segment ids",
                details={"candidateIds": candidate_ids},
            )
        candidate_segments = {
            candidate.segment_id: self._candidate_segment(
                candidate,
                segments,
            )
            for candidate in candidates
        }

        references_by_speaker = self._references_by_speaker(segments)
        proposals: dict[str, ReviewProposal] = {}
        plans: list[dict[str, Any]] = []
        clip_keys: list[tuple[str, int, int]] = []
        clip_index: dict[tuple[str, int, int], int] = {}

        def register_clip(
            segment: TranscriptSegment,
        ) -> tuple[str, int, int]:
            audio_path = self._audio_path(segment)
            if audio_path is None or not Path(audio_path).is_file():
                raise WorkerError(
                    "PREPARED_AUDIO_MISSING",
                    "ERes2NetV2 requires persisted normalized audio",
                    details={"segmentId": segment.segment_id},
                )
            key = (audio_path, segment.start_ms, segment.end_ms)
            if key not in clip_index:
                clip_index[key] = len(clip_keys)
                clip_keys.append(key)
            return key

        for candidate in candidates:
            context.raise_if_cancelled()
            segment = candidate_segments[candidate.segment_id]
            references = self._reference_segments(
                candidate,
                segments,
                references_by_speaker,
            )
            ranked = self._ranked_speakers(segment)
            reference_by_speaker = {
                reference.speaker_id: reference for reference in references
            }
            usable = [
                speaker_id
                for speaker_id in ranked
                if speaker_id in reference_by_speaker
            ]
            if not usable:
                proposals[segment.segment_id] = ReviewProposal(
                    segment_id=segment.segment_id,
                    source="acoustic",
                    reason_code="ERES2NETV2_NO_REFERENCE",
                    evidence_refs=(f"audio:{segment.segment_id}",),
                    confidence=0.0,
                    exit_reason="NO_REFERENCE",
                    resource=_resource_snapshot(),
                )
                continue
            if len(usable) == 1:
                only_speaker = usable[0]
                proposals[segment.segment_id] = ReviewProposal(
                    segment_id=segment.segment_id,
                    source="acoustic",
                    reason_code="ERES2NETV2_SINGLE_REFERENCE",
                    evidence_refs=(
                        f"audio:{segment.segment_id}",
                        f"audio:{reference_by_speaker[only_speaker].segment_id}",
                    ),
                    confidence=0.0,
                    exit_reason="SINGLE_REFERENCE_INSUFFICIENT",
                    resource=_resource_snapshot(),
                )
                continue

            candidate_key = register_clip(segment)
            reference_keys = {
                speaker_id: register_clip(
                    reference_by_speaker[speaker_id]
                )
                for speaker_id in usable
            }
            plans.append(
                {
                    "segment": segment,
                    "usable": tuple(usable),
                    "referenceBySpeaker": reference_by_speaker,
                    "candidateKey": candidate_key,
                    "referenceKeys": reference_keys,
                }
            )

        embeddings_by_key: dict[
            tuple[str, int, int], tuple[float, ...]
        ] = {}
        if plans:
            context.raise_if_cancelled()
            audio_by_path: dict[str, tuple[Any, int]] = {}
            clips: list[Any] = []
            for audio_path, start_ms, end_ms in clip_keys:
                context.raise_if_cancelled()
                if audio_path not in audio_by_path:
                    audio_by_path[audio_path] = _load_audio(audio_path)
                samples, sample_rate = audio_by_path[audio_path]
                clips.append(
                    _slice_audio(
                        samples,
                        sample_rate,
                        start_ms,
                        end_ms,
                    )
                )
            context.raise_if_cancelled()
            embeddings = self._embeddings(clips)
            embeddings_by_key = {
                key: embeddings[index]
                for index, key in enumerate(clip_keys)
            }

        for plan in plans:
            context.raise_if_cancelled()
            segment = plan["segment"]
            usable = plan["usable"]
            reference_by_speaker = plan["referenceBySpeaker"]
            candidate_embedding = embeddings_by_key[plan["candidateKey"]]
            scores = {
                speaker_id: self._cosine(
                    candidate_embedding,
                    embeddings_by_key[
                        plan["referenceKeys"][speaker_id]
                    ],
                )
                for speaker_id in usable
            }
            ranked_secondary = sorted(
                scores,
                key=lambda speaker_id: (-scores[speaker_id], speaker_id),
            )
            winner = ranked_secondary[0]
            margin = scores[winner] - scores[ranked_secondary[1]]
            evidence_refs = tuple(
                [
                    f"audio:{segment.segment_id}",
                    *[
                        f"audio:{reference_by_speaker[speaker_id].segment_id}"
                        for speaker_id in usable
                    ],
                ]
            )
            if margin < self.decision_margin:
                proposals[segment.segment_id] = ReviewProposal(
                    segment_id=segment.segment_id,
                    source="acoustic",
                    reason_code="ERES2NETV2_LOW_MARGIN",
                    evidence_refs=evidence_refs,
                    confidence=max(0.0, min(1.0, margin)),
                    exit_reason="LOW_MARGIN_UNRESOLVED",
                    resource=_resource_snapshot(),
                )
                continue
            apply_change = winner != segment.speaker_id
            proposals[segment.segment_id] = ReviewProposal(
                segment_id=segment.segment_id,
                source="acoustic",
                speaker_id=winner if apply_change else None,
                reason_code=(
                    "ERES2NETV2_TOP2_REASSIGN"
                    if apply_change
                    else "ERES2NETV2_VERIFY"
                ),
                evidence_refs=evidence_refs,
                confidence=max(0.0, min(1.0, margin)),
                exit_reason=(
                    "VERIFIED_SPEAKER_CHANGE"
                    if apply_change
                    else "VERIFIED_NO_CHANGE"
                ),
                resource=_resource_snapshot(),
            )
        return [
            proposals[candidate.segment_id] for candidate in candidates
        ]


class LocalPyannoteAuditAdapter:
    """Local pyannote overlap detector and candidate-only audit.

    Full-corpus inference is only enabled when production configuration selects
    the explicit pyannote fallback.  Audio is always supplied as an in-memory
    waveform so the runtime never depends on TorchCodec file decoding.
    """

    adapter_id = "pyannote-community-1"
    version = "2.4.0"
    telemetry_enabled = False

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "cuda",
        pipeline_factory: Callable[..., Any] | None = None,
        python_executable: str | Path | None = None,
        isolated_inference_runner: Callable[..., Any] | None = None,
        inference_timeout_seconds: float = 300.0,
    ) -> None:
        self.model_path = _local_model_path(model_path, "pyannote model")
        self.device = str(device)
        self._pipeline_factory = pipeline_factory
        self.python_executable = self._resolve_python(python_executable)
        if pipeline_factory is not None and (
            self.python_executable is not None
            or isolated_inference_runner is not None
        ):
            raise ValueError(
                "pipeline_factory and isolated inference are mutually exclusive"
            )
        self._isolated_inference_runner = isolated_inference_runner
        self.inference_timeout_seconds = float(inference_timeout_seconds)
        if (
            not math.isfinite(self.inference_timeout_seconds)
            or self.inference_timeout_seconds <= 0.0
        ):
            raise ValueError("inference_timeout_seconds must be positive")
        self._pipeline_instance: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.RLock()
        os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

    @property
    def _uses_isolated_runtime(self) -> bool:
        return (
            self.python_executable is not None
            or self._isolated_inference_runner is not None
        )

    @staticmethod
    def _resolve_python(value: str | Path | None) -> str | None:
        if value is None:
            return None
        text = os.fspath(value).strip()
        if not text:
            raise ValueError("python_executable must not be empty")
        contains_separator = any(
            separator and separator in text for separator in (os.sep, os.altsep)
        )
        if Path(text).is_absolute() or contains_separator:
            candidate = Path(text).expanduser()
            if not candidate.is_file():
                raise ValueError("python_executable must be a local file")
            return str(candidate.resolve(strict=True))
        resolved = shutil.which(text)
        if not resolved:
            raise ValueError("python_executable is not available on PATH")
        return resolved

    def _pipeline(self) -> Any:
        with self._load_lock:
            if self._pipeline_instance is None:
                factory = self._pipeline_factory
                try:
                    if factory is None:
                        from pyannote.audio import Pipeline
                        import torch

                        pipeline = Pipeline.from_pretrained(str(self.model_path))
                        pipeline.to(torch.device(self.device))
                        self._pipeline_instance = pipeline
                    else:
                        self._pipeline_instance = factory(
                            model=str(self.model_path),
                            device=self.device,
                            telemetry_enabled=False,
                        )
                    if not callable(self._pipeline_instance):
                        raise TypeError("pyannote pipeline is not callable")
                except WorkerError:
                    raise
                except ImportError as exc:
                    raise WorkerError(
                        "PYANNOTE_RUNTIME_MISSING",
                        "pyannote.audio and torch are required for audit mode",
                        details={"exceptionType": type(exc).__name__},
                    ) from exc
                except Exception as exc:
                    raise WorkerError(
                        "PYANNOTE_RUNTIME_INCOMPATIBLE",
                        "pyannote runtime could not be initialized safely",
                        details={
                            "exceptionType": type(exc).__name__,
                            "message": str(exc)[:500],
                        },
                    ) from exc
            return self._pipeline_instance

    def release_resources(self) -> None:
        """Idempotently unload pyannote before leaving its cascade stage."""

        with self._inference_lock:
            with self._load_lock:
                pipeline = self._pipeline_instance
                self._pipeline_instance = None
                if pipeline is not None:
                    del pipeline
                if not self._uses_isolated_runtime:
                    _release_accelerator_memory()

    @staticmethod
    def _normalize_isolated_turns(
        values: Any,
        *,
        start_ms: int,
        end_ms: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            raise WorkerError(
                "PYANNOTE_RESULT_INVALID",
                "isolated pyannote output is missing speaker turns",
            )
        turns: list[dict[str, Any]] = []
        for index, raw in enumerate(values):
            if not isinstance(raw, Mapping):
                raise WorkerError(
                    "PYANNOTE_RESULT_INVALID",
                    "isolated pyannote returned a malformed speaker turn",
                    details={"turnIndex": index},
                )
            turn_start = raw.get("startMs")
            turn_end = raw.get("endMs")
            local_speaker = raw.get("localSpeaker")
            if (
                isinstance(turn_start, bool)
                or not isinstance(turn_start, int)
                or isinstance(turn_end, bool)
                or not isinstance(turn_end, int)
                or not isinstance(local_speaker, str)
                or not local_speaker.strip()
                or turn_start < start_ms
                or turn_end > end_ms
                or turn_end <= turn_start
            ):
                raise WorkerError(
                    "PYANNOTE_RESULT_INVALID",
                    "isolated pyannote returned an out-of-range speaker turn",
                    details={"turnIndex": index},
                )
            turns.append(
                {
                    "startMs": turn_start,
                    "endMs": turn_end,
                    "localSpeaker": local_speaker.strip(),
                }
            )
        return sorted(
            turns,
            key=lambda item: (
                item["startMs"],
                item["endMs"],
                item["localSpeaker"],
            ),
        )

    def _run_isolated_inference(
        self,
        audio_path: Path,
        *,
        start_ms: int,
        end_ms: int,
        context: AdapterContext,
        failure_code: str,
        failure_message: str,
        details: Mapping[str, Any],
        speaker_count_constraints: Mapping[str, int] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        context.raise_if_cancelled()
        constraints, _ = self._speaker_count_constraints(
            speaker_count_constraints
        )
        if self._isolated_inference_runner is not None:
            raw_result = self._isolated_inference_runner(
                audio_path=audio_path,
                model_path=self.model_path,
                device=self.device,
                start_ms=start_ms,
                end_ms=end_ms,
                context=context,
                speaker_count_constraints=constraints or None,
            )
            regular_raw = (
                raw_result.get("speakerTurns")
                if isinstance(raw_result, Mapping)
                else raw_result
            )
            exclusive_raw = (
                raw_result.get("exclusiveSpeakerTurns")
                if isinstance(raw_result, Mapping)
                else None
            )
            return (
                self._normalize_isolated_turns(
                    regular_raw,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                (
                    self._normalize_isolated_turns(
                        exclusive_raw,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                    if exclusive_raw is not None
                    else None
                ),
            )
        if self.python_executable is None:
            raise WorkerError(
                "PYANNOTE_RUNTIME_MISSING",
                "isolated pyannote Python is not configured",
            )
        script = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "pyannote_runtime.py"
        )
        if not script.is_file():
            raise WorkerError(
                "PYANNOTE_RUNTIME_MISSING",
                "isolated pyannote runtime entrypoint is missing",
            )
        request = {
            "schemaVersion": "1.1.0",
            "modelPath": str(self.model_path),
            "audioPath": str(audio_path.resolve(strict=True)),
            "device": self.device,
            "startMs": start_ms,
            "endMs": end_ms,
            "speakerCountConstraints": constraints or None,
        }
        environment = dict(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYANNOTE_METRICS_ENABLED": "0",
                "DO_NOT_TRACK": "1",
            }
        )
        payload = json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                [self.python_executable, "-I", str(script)],
                cwd=str(script.parents[1]),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            pending_input: bytes | None = payload
            deadline = time.monotonic() + self.inference_timeout_seconds
            while True:
                try:
                    stdout, stderr = process.communicate(
                        input=pending_input,
                        timeout=min(
                            0.25,
                            max(0.01, deadline - time.monotonic()),
                        ),
                    )
                    break
                except subprocess.TimeoutExpired:
                    pending_input = None
                    context.raise_if_cancelled()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("isolated pyannote inference timed out")
            if len(stdout) > 2 * 1024 * 1024:
                raise ValueError("isolated pyannote output exceeds the limit")
            if process.returncode != 0:
                raise RuntimeError(
                    f"isolated pyannote exited with code {process.returncode}"
                )
            response = json.loads(stdout.decode("utf-8"))
            if (
                not isinstance(response, Mapping)
                or response.get("schemaVersion") not in {"1.0.0", "1.1.0"}
                or response.get("status") != "ok"
            ):
                raise ValueError("isolated pyannote response is malformed")
            turns = self._normalize_isolated_turns(
                response.get("speakerTurns"),
                start_ms=start_ms,
                end_ms=end_ms,
            )
            exclusive_turns = (
                self._normalize_isolated_turns(
                    response.get("exclusiveSpeakerTurns"),
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
                if response.get("schemaVersion") == "1.1.0"
                else None
            )
        except WorkerError:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
            raise
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
            raise WorkerError(
                failure_code,
                failure_message,
                details={
                    **dict(details),
                    "exceptionType": type(exc).__name__,
                },
            ) from exc
        context.raise_if_cancelled()
        return turns, exclusive_turns

    def _infer_waveform(
        self,
        samples: Any,
        sample_rate: int,
        context: AdapterContext,
        *,
        failure_code: str,
        failure_message: str,
        details: Mapping[str, Any],
        speaker_count_constraints: Mapping[str, int] | None = None,
    ) -> Any:
        context.raise_if_cancelled()
        try:
            import torch
        except ImportError as exc:
            raise WorkerError(
                "PYANNOTE_RUNTIME_MISSING",
                "torch is required for pyannote inference",
            ) from exc
        waveform = torch.from_numpy(samples).unsqueeze(0)
        _, inference_kwargs = self._speaker_count_constraints(
            speaker_count_constraints
        )
        try:
            with self._inference_lock:
                pipeline = self._pipeline()
                result = pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    **inference_kwargs,
                )
        except WorkerError:
            raise
        except Exception as exc:
            raise WorkerError(
                failure_code,
                failure_message,
                details={
                    **dict(details),
                    "exceptionType": type(exc).__name__,
                },
            ) from exc
        context.raise_if_cancelled()
        return result

    @staticmethod
    def _speaker_count_constraints(
        value: Mapping[str, int] | None,
    ) -> tuple[dict[str, int], dict[str, int]]:
        if value is None:
            return {}, {}
        if not isinstance(value, Mapping):
            raise WorkerError(
                "PYANNOTE_SPEAKER_COUNT_CONSTRAINT_INVALID",
                "pyannote speaker-count constraints must be an object",
            )
        constraints = dict(value)
        keys = set(constraints)
        if keys == {"numSpeakers"}:
            count = constraints["numSpeakers"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise WorkerError(
                    "PYANNOTE_SPEAKER_COUNT_CONSTRAINT_INVALID",
                    "pyannote numSpeakers must be a positive integer",
                )
            return (
                {"numSpeakers": count},
                {"num_speakers": count},
            )
        if keys == {"minSpeakers", "maxSpeakers"}:
            minimum = constraints["minSpeakers"]
            maximum = constraints["maxSpeakers"]
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or minimum < 1
                or maximum < minimum
            ):
                raise WorkerError(
                    "PYANNOTE_SPEAKER_COUNT_CONSTRAINT_INVALID",
                    "pyannote speaker-count bounds are invalid",
                )
            return (
                {
                    "minSpeakers": minimum,
                    "maxSpeakers": maximum,
                },
                {
                    "min_speakers": minimum,
                    "max_speakers": maximum,
                },
            )
        raise WorkerError(
            "PYANNOTE_SPEAKER_COUNT_CONSTRAINT_INVALID",
            "pyannote speaker-count constraints use unsupported fields",
        )

    @staticmethod
    def _annotation_from_result(
        result: Any,
        *,
        field: str = "speaker_diarization",
        required: bool = True,
    ) -> Any:
        annotation = None
        if isinstance(result, Mapping):
            annotation = result.get(field)
        else:
            annotation = getattr(result, field, None)
        if (
            field == "speaker_diarization"
            and annotation is None
            and callable(getattr(result, "itertracks", None))
        ):
            annotation = result
        if annotation is None and not required:
            return None
        if annotation is None or not callable(
            getattr(annotation, "itertracks", None)
        ):
            raise WorkerError(
                "PYANNOTE_RESULT_INVALID",
                "pyannote output does not contain a speaker diarization Annotation",
            )
        return annotation

    @staticmethod
    def _speaker_turns(
        result: Any,
        segment: TranscriptSegment,
        *,
        field: str = "speaker_diarization",
        required: bool = True,
    ) -> list[dict[str, Any]] | None:
        annotation = LocalPyannoteAuditAdapter._annotation_from_result(
            result,
            field=field,
            required=required,
        )
        if annotation is None:
            return None
        try:
            tracks = annotation.itertracks(yield_label=True)
        except Exception as exc:
            raise WorkerError(
                "PYANNOTE_RESULT_INVALID",
                "pyannote Annotation tracks could not be enumerated",
                details={"exceptionType": type(exc).__name__},
            ) from exc

        turns: list[dict[str, Any]] = []
        try:
            for index, raw in enumerate(tracks):
                if (
                    not isinstance(raw, Sequence)
                    or isinstance(raw, (str, bytes, bytearray))
                    or len(raw) not in {2, 3}
                ):
                    raise TypeError(f"track {index} is malformed")
                turn = raw[0]
                label = raw[-1]
                raw_start = getattr(turn, "start")
                raw_end = getattr(turn, "end")
                if isinstance(raw_start, bool) or isinstance(raw_end, bool):
                    raise TypeError(f"track {index} uses boolean timestamps")
                start_seconds = float(raw_start)
                end_seconds = float(raw_end)
                if not math.isfinite(start_seconds) or not math.isfinite(
                    end_seconds
                ):
                    raise ValueError(f"track {index} has non-finite timestamps")
                local_label = str(label or "").strip()
                if not local_label:
                    raise ValueError(f"track {index} has no speaker label")
                start_ms = segment.start_ms + round(start_seconds * 1000.0)
                end_ms = segment.start_ms + round(end_seconds * 1000.0)
                start_ms = max(segment.start_ms, min(start_ms, segment.end_ms))
                end_ms = max(start_ms, min(end_ms, segment.end_ms))
                if end_ms == start_ms:
                    continue
                turns.append(
                    {
                        "startMs": start_ms,
                        "endMs": end_ms,
                        "localSpeaker": local_label,
                    }
                )
        except WorkerError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise WorkerError(
                "PYANNOTE_RESULT_INVALID",
                "pyannote returned malformed speaker turns",
                details={"exceptionType": type(exc).__name__},
            ) from exc
        return sorted(
            turns,
            key=lambda item: (
                item["startMs"],
                item["endMs"],
                item["localSpeaker"],
            ),
        )

    @staticmethod
    def _overlap_intervals(
        turns: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        raw_intervals: list[dict[str, Any]] = []
        for left_index, left in enumerate(turns):
            for right in turns[left_index + 1 :]:
                if left["localSpeaker"] == right["localSpeaker"]:
                    continue
                start_ms = max(int(left["startMs"]), int(right["startMs"]))
                end_ms = min(int(left["endMs"]), int(right["endMs"]))
                if end_ms <= start_ms:
                    continue
                raw_intervals.append(
                    {
                        "startMs": start_ms,
                        "endMs": end_ms,
                        "localSpeakers": sorted(
                            {
                                str(left["localSpeaker"]),
                                str(right["localSpeaker"]),
                            }
                        ),
                    }
                )
        merged: list[dict[str, Any]] = []
        for interval in sorted(
            raw_intervals,
            key=lambda item: (
                item["startMs"],
                item["endMs"],
                tuple(item["localSpeakers"]),
            ),
        ):
            if not merged or interval["startMs"] > merged[-1]["endMs"]:
                merged.append(
                    {
                        "startMs": interval["startMs"],
                        "endMs": interval["endMs"],
                        "localSpeakers": list(interval["localSpeakers"]),
                    }
                )
                continue
            merged[-1]["endMs"] = max(
                int(merged[-1]["endMs"]),
                int(interval["endMs"]),
            )
            merged[-1]["localSpeakers"] = sorted(
                {
                    *merged[-1]["localSpeakers"],
                    *interval["localSpeakers"],
                }
            )
        return merged

    def detect_batch(
        self,
        prepared: PreparedAudio,
        windows: Sequence[SpeechWindow],
        context: AdapterContext,
        *,
        speaker_count_constraints: Mapping[str, int] | None = None,
    ) -> list[OverlapDecision]:
        """Detect exact overlap intervals once for the normalized timeline."""

        if not windows:
            return []
        if not prepared.audio_path or not Path(prepared.audio_path).is_file():
            raise WorkerError(
                "PREPARED_AUDIO_MISSING",
                "pyannote overlap detection requires persisted normalized audio",
            )
        pcm_buffer_id = _pcm_buffer_id(prepared)
        constraints, _ = self._speaker_count_constraints(
            speaker_count_constraints
        )
        if self._uses_isolated_runtime:
            turns, exclusive_turns = self._run_isolated_inference(
                Path(prepared.audio_path),
                start_ms=0,
                end_ms=prepared.duration_ms,
                context=context,
                failure_code="PYANNOTE_OVERLAP_INFERENCE_FAILED",
                failure_message="pyannote failed while detecting overlap",
                details={"windowCount": len(windows)},
                speaker_count_constraints=constraints or None,
            )
        else:
            samples, sample_rate, pcm_buffer_id = _shared_pcm_for_prepared(
                prepared
            )
            result = self._infer_waveform(
                samples,
                sample_rate,
                context,
                failure_code="PYANNOTE_OVERLAP_INFERENCE_FAILED",
                failure_message="pyannote failed while detecting overlap",
                details={"windowCount": len(windows)},
                speaker_count_constraints=constraints or None,
            )
            timeline = type(
                "_PyannoteTimeline",
                (),
                {"start_ms": 0, "end_ms": prepared.duration_ms},
            )()
            turns = self._speaker_turns(result, timeline)
            exclusive_turns = self._speaker_turns(
                result,
                timeline,
                field="exclusive_speaker_diarization",
                required=False,
            )
        if turns is None:
            raise WorkerError(
                "PYANNOTE_RESULT_INVALID",
                "pyannote regular speaker timeline is missing",
            )
        overlap_intervals = self._overlap_intervals(turns)
        global_local_speakers = sorted(
            {str(turn["localSpeaker"]) for turn in turns}
        )
        serialized_turns = json.dumps(
            turns,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        full_timeline_inference = {
            "scope": "full-normalized-timeline",
            "startMs": 0,
            "endMs": prepared.duration_ms,
            "turnCount": len(turns),
            "localSpeakerCount": len(global_local_speakers),
            "localSpeakers": global_local_speakers,
            "speakerTurns": [dict(turn) for turn in turns],
            "speakerTurnsSha256": hashlib.sha256(serialized_turns).hexdigest(),
            "exclusiveNative": exclusive_turns is not None,
            "speakerCountConstraints": constraints or None,
        }
        if exclusive_turns is not None:
            serialized_exclusive_turns = json.dumps(
                exclusive_turns,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            full_timeline_inference.update(
                {
                    "exclusiveTurnCount": len(exclusive_turns),
                    "exclusiveSpeakerTurns": [
                        dict(turn) for turn in exclusive_turns
                    ],
                    "exclusiveSpeakerTurnsSha256": hashlib.sha256(
                        serialized_exclusive_turns
                    ).hexdigest(),
                }
            )

        output: list[OverlapDecision] = []
        for window in windows:
            context.raise_if_cancelled()
            window_turns = [
                {
                    **dict(turn),
                    "startMs": max(window.start_ms, int(turn["startMs"])),
                    "endMs": min(window.end_ms, int(turn["endMs"])),
                }
                for turn in turns
                if int(turn["startMs"]) < window.end_ms
                and int(turn["endMs"]) > window.start_ms
            ]
            window_overlap = [
                {
                    **dict(interval),
                    "startMs": max(
                        window.start_ms,
                        int(interval["startMs"]),
                    ),
                    "endMs": min(
                        window.end_ms,
                        int(interval["endMs"]),
                    ),
                }
                for interval in overlap_intervals
                if int(interval["startMs"]) < window.end_ms
                and int(interval["endMs"]) > window.start_ms
            ]
            window_exclusive_turns = (
                [
                    {
                        **dict(turn),
                        "startMs": max(
                            window.start_ms,
                            int(turn["startMs"]),
                        ),
                        "endMs": min(
                            window.end_ms,
                            int(turn["endMs"]),
                        ),
                    }
                    for turn in exclusive_turns
                    if int(turn["startMs"]) < window.end_ms
                    and int(turn["endMs"]) > window.start_ms
                ]
                if exclusive_turns is not None
                else None
            )
            overlapping = bool(window_overlap)
            evidence: dict[str, Any] = {
                "detectorStatus": "EVALUATED",
                "overlapDetectorRun": True,
                "reviewStatus": (
                    "REVIEW_REQUIRED" if overlapping else "NOT_REQUIRED"
                ),
                "model": self.adapter_id,
                "pcmBufferId": pcm_buffer_id,
                "confidenceKind": "binary-annotation-no-posterior",
                "calibratedConfidence": False,
                "fullTimelineInference": dict(full_timeline_inference),
                "speakerTurns": window_turns,
                "overlapIntervals": window_overlap,
                "localSpeakerCount": len(
                    {
                        str(turn["localSpeaker"])
                        for turn in window_turns
                    }
                ),
            }
            if window_exclusive_turns is not None:
                evidence["exclusiveSpeakerTurns"] = window_exclusive_turns
            if overlapping:
                evidence["reasonCode"] = "PYANNOTE_OVERLAP_DETECTED"
            output.append(
                OverlapDecision(
                    window_id=window.window_id,
                    overlapping=overlapping,
                    confidence=0.5,
                    evidence=evidence,
                )
            )
        return output

    def review_batch(
        self,
        candidates: Sequence[ReviewCandidate],
        segments: Mapping[str, TranscriptSegment],
        context: AdapterContext,
    ) -> list[ReviewProposal]:
        output: list[ReviewProposal] = []
        for candidate in candidates:
            context.raise_if_cancelled()
            segment = segments[candidate.segment_id]
            preparation = segment.evidence.get("preparation")
            audio_path = (
                preparation.get("audioPath")
                if isinstance(preparation, Mapping)
                else None
            )
            if not isinstance(audio_path, str) or not Path(audio_path).is_file():
                raise WorkerError(
                    "PREPARED_AUDIO_MISSING",
                    "pyannote audit requires persisted normalized audio",
                )
            overlap = segment.evidence.get("overlap")
            cached_turns = (
                overlap.get("speakerTurns")
                if isinstance(overlap, Mapping)
                else None
            )
            if isinstance(cached_turns, list):
                turns = self._normalize_isolated_turns(
                    cached_turns,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                )
            elif self._uses_isolated_runtime:
                turns = self._run_isolated_inference(
                    Path(audio_path),
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    context=context,
                    failure_code="PYANNOTE_INFERENCE_FAILED",
                    failure_message=(
                        "pyannote failed while auditing a difficult segment"
                    ),
                    details={"segmentId": segment.segment_id},
                )
            else:
                samples, sample_rate = _load_audio(audio_path)
                clip = _slice_audio(
                    samples,
                    sample_rate,
                    segment.start_ms,
                    segment.end_ms,
                )
                result = self._infer_waveform(
                    clip,
                    sample_rate,
                    context,
                    failure_code="PYANNOTE_INFERENCE_FAILED",
                    failure_message=(
                        "pyannote failed while auditing a difficult segment"
                    ),
                    details={"segmentId": segment.segment_id},
                )
                turns = self._speaker_turns(result, segment)
            local_speakers = sorted(
                {str(turn["localSpeaker"]) for turn in turns}
            )
            overlap_intervals = self._overlap_intervals(turns)
            conflict_reasons: list[str] = ["ANONYMOUS_LABELS_NOT_AUTO_MAPPED"]
            if not turns:
                conflict_reasons.append("NO_SPEAKER_TURNS")
            if len(local_speakers) > 1:
                conflict_reasons.append("MULTIPLE_LOCAL_SPEAKERS")
            if overlap_intervals:
                conflict_reasons.append("LOCAL_OVERLAP")
            if "OVERLAP" in candidate.reasons or segment.overlapping:
                conflict_reasons.append("UPSTREAM_OVERLAP")
            audit_evidence = {
                "schemaVersion": 1,
                "segmentId": segment.segment_id,
                "segmentStartMs": segment.start_ms,
                "segmentEndMs": segment.end_ms,
                "speakerTurns": turns,
                "localSpeakerCount": len(local_speakers),
                "localSpeakers": local_speakers,
                "overlapIntervals": overlap_intervals,
                "conflictProposal": {
                    "action": "HUMAN_REVIEW",
                    "automaticSpeakerOverride": False,
                    "currentSpeakerId": segment.speaker_id,
                    "reasonCodes": sorted(set(conflict_reasons)),
                },
            }
            audit_reference = (
                f"pyannote:{segment.segment_id}:"
                + json.dumps(
                    audit_evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            output.append(
                ReviewProposal(
                    segment_id=segment.segment_id,
                    source="acoustic",
                    reason_code="PYANNOTE_CONFLICT_PROPOSAL",
                    evidence_refs=(
                        f"audio:{segment.segment_id}",
                        audit_reference,
                    ),
                    confidence=0.0,
                    exit_reason="HUMAN_REVIEW_REQUIRED",
                    resource=_resource_snapshot(),
                )
            )
        return output


__all__ = [
    "FfmpegFunAsrPreparationAdapter",
    "LocalERes2NetV2Verifier",
    "LocalFunAsrCamPlusAdapter",
    "LocalPyannoteAuditAdapter",
    "LocalQwen3AsrAdapter",
]
