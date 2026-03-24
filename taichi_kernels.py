"""
taichi_kernels.py - Taichi / NumPy accelerated kernels.

策略：
  - macOS 优先使用 Metal，其余平台按可用后端选择
  - 每个函数都有 NumPy fallback，且 NumPy 也做分块避免内存爆炸
  - 绝不一次性分配超过设定上限的内存
"""

import logging
import math
import os
import platform
import numpy as np
import threading
from typing import List, Tuple, Optional

try:
    import sitecustomize  # noqa: F401
except Exception:
    sitecustomize = None  # type: ignore[assignment]

from runtime_paths import RUNTIME_CACHE_ROOT

logger = logging.getLogger(__name__)

_TI_INITIALIZED = False
ti = None
_TI_INIT_THREAD_ID: Optional[int] = None
_TI_THREAD_GUARD_LOGGED = False

# 单次Taichi/NumPy分配上限（samples数）
# 10M samples ≈ 40MB float32，安全值
MAX_CHUNK_SAMPLES = 10_000_000


def _macos_version_tuple() -> Tuple[int, int, int]:
    if os.name == "nt" or os.environ.get("OS", "").lower() == "windows_nt":
        return (0, 0, 0)
    if os.sys.platform != "darwin":
        return (0, 0, 0)

    raw = str(platform.mac_ver()[0] or "").strip()
    if not raw:
        return (0, 0, 0)
    values: List[int] = []
    for part in raw.split(".")[:3]:
        try:
            values.append(int(part))
        except Exception:
            values.append(0)
    while len(values) < 3:
        values.append(0)
    return (values[0], values[1], values[2])


def _should_skip_taichi_metal(arch: str) -> bool:
    normalized = str(arch or "cpu").strip().lower()
    if os.sys.platform != "darwin" or normalized not in {"metal", "mps"}:
        return False
    if str(os.environ.get("MTS_DISABLE_TAICHI_METAL", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        major, minor, patch = _macos_version_tuple()
        logger.info(
            "Taichi Metal disabled by env on macOS %d.%d.%d; using CPU fallback",
            major,
            minor,
            patch,
        )
        return True
    return False


def _taichi_requires_main_thread() -> bool:
    # Taichi's LLVM runtime is notably fragile on Windows when initialized
    # off the main thread. macOS worker-thread usage is acceptable as long as
    # the same thread performs initialization and execution.
    return os.name == "nt"


def _taichi_offline_cache_dir() -> str:
    cache_dir = RUNTIME_CACHE_ROOT / "taichi"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def _reset_taichi_runtime() -> None:
    global _TI_INITIALIZED, ti, _TI_INIT_THREAD_ID
    try:
        if ti is not None and hasattr(ti, "reset"):
            ti.reset()
    except Exception:
        pass
    ti = None
    _TI_INITIALIZED = False
    _TI_INIT_THREAD_ID = None


def _taichi_runtime_self_test() -> bool:
    if ti is None:
        return False

    try:
        src = ti.field(dtype=ti.f32, shape=16)
        dst = ti.field(dtype=ti.f32, shape=16)
        src.from_numpy(np.linspace(0.0, 1.0, 16, dtype=np.float32))

        @ti.kernel
        def _warmup():
            for i in dst:
                dst[i] = src[i] * 2.0 + 1.0

        _warmup()
        out = dst.to_numpy()
        return bool(
            isinstance(out, np.ndarray)
            and out.shape == (16,)
            and np.all(np.isfinite(out))
        )
    except Exception as e:
        logger.warning(f"Taichi runtime self-test failed ({e}), using NumPy fallback")
        _reset_taichi_runtime()
        return False


def init_taichi(arch: str = "cpu", default_fp: int = 32) -> bool:
    """初始化 Taichi，失败时自动降级到 CPU 或纯 NumPy。"""
    global _TI_INITIALIZED, ti, _TI_INIT_THREAD_ID

    if _TI_INITIALIZED:
        return True

    if _should_skip_taichi_metal(arch):
        arch = "cpu"

    if _taichi_requires_main_thread() and threading.current_thread() is not threading.main_thread():
        logger.info(
            "Taichi disabled: init requested from non-main thread on Windows, using NumPy fallback"
        )
        ti = None
        _TI_INITIALIZED = False
        _TI_INIT_THREAD_ID = None
        return False

    try:
        import taichi as _ti
        ti = _ti

        arch_map = {
            "cuda": getattr(ti, "cuda", ti.cpu),
            "gpu": getattr(ti, "gpu", ti.cpu),
            "vulkan": getattr(ti, "vulkan", ti.cpu),
            "metal": getattr(ti, "metal", ti.cpu),
            "mps": getattr(ti, "metal", ti.cpu),
            "cpu": ti.cpu,
        }
        target = arch_map.get(str(arch or "cpu").lower(), ti.cpu)
        init_kwargs = {
            "default_fp": ti.f32,
            "offline_cache_file_path": _taichi_offline_cache_dir(),
        }

        def _safe_ti_init(target_arch):
            try:
                ti.init(arch=target_arch, **init_kwargs)
            except TypeError as e:
                if "unexpected keyword argument" not in str(e or ""):
                    raise
                fallback_kwargs = dict(init_kwargs)
                fallback_kwargs.pop("offline_cache_file_path", None)
                ti.init(arch=target_arch, **fallback_kwargs)

        # 尝试目标架构
        try:
            _safe_ti_init(target)
            if not _taichi_runtime_self_test():
                raise RuntimeError("taichi runtime self-test failed")
            _TI_INITIALIZED = True
            _TI_INIT_THREAD_ID = threading.get_ident()
            logger.info(f"Taichi initialized: arch={arch}")
            return True
        except Exception as e:
            logger.warning(f"Taichi {arch} failed ({e}), trying CPU")
            _reset_taichi_runtime()

        # CPU兜底
        try:
            if ti is None:
                import taichi as _ti
                ti = _ti
            _safe_ti_init(ti.cpu)
            if not _taichi_runtime_self_test():
                raise RuntimeError("taichi CPU runtime self-test failed")
            _TI_INITIALIZED = True
            _TI_INIT_THREAD_ID = threading.get_ident()
            logger.info("Taichi initialized: arch=cpu (fallback)")
            return True
        except Exception as e2:
            logger.warning(f"Taichi CPU failed ({e2}), pure NumPy mode")
            _reset_taichi_runtime()
            return False

    except ImportError:
        logger.warning("Taichi not installed, using NumPy")
        return False


def _can_use_taichi_runtime() -> bool:
    global _TI_THREAD_GUARD_LOGGED
    if not _TI_INITIALIZED or ti is None:
        return False
    current_id = threading.get_ident()
    if _taichi_requires_main_thread() and threading.current_thread() is not threading.main_thread():
        if not _TI_THREAD_GUARD_LOGGED:
            logger.info("Taichi bypassed on Windows worker thread; using NumPy fallback")
            _TI_THREAD_GUARD_LOGGED = True
        return False
    if _TI_INIT_THREAD_ID is not None and current_id != _TI_INIT_THREAD_ID:
        if not _TI_THREAD_GUARD_LOGGED:
            logger.info("Taichi bypassed on non-init thread; using NumPy fallback")
            _TI_THREAD_GUARD_LOGGED = True
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Audio Normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_audio_fast(audio_np: np.ndarray) -> np.ndarray:
    """Peak-normalize，大数组安全处理"""
    audio_f32 = np.ascontiguousarray(audio_np, dtype=np.float32)
    n = len(audio_f32)
    if n == 0:
        return audio_f32

    # NumPy normalize 极快且零额外内存，直接用
    peak = np.max(np.abs(audio_f32))
    if peak < 1e-8:
        return audio_f32
    return audio_f32 / peak


# ═══════════════════════════════════════════════════════════════════════════
# RMS Energy（分块计算，绝不爆内存）
# ═══════════════════════════════════════════════════════════════════════════

def compute_rms_energy_fast(
    audio_np: np.ndarray,
    frame_size: int = 512,
    hop_size: int = 256,
) -> np.ndarray:
    """
    计算逐帧RMS能量
    对长音频分块处理，避免内存/显存爆炸
    """
    audio_f32 = np.ascontiguousarray(audio_np, dtype=np.float32)
    n = len(audio_f32)

    if n < frame_size:
        rms_val = np.sqrt(np.mean(audio_f32 ** 2)) if n > 0 else 0.0
        return np.array([rms_val], dtype=np.float32)

    n_frames = (n - frame_size) // hop_size + 1

    if _can_use_taichi_runtime():
        try:
            return _rms_taichi_chunked(audio_f32, n, n_frames, frame_size, hop_size)
        except Exception as e:
            logger.debug(f"Taichi RMS failed ({e}), NumPy fallback")

    return _rms_numpy_chunked(audio_f32, n, n_frames, frame_size, hop_size)


def _rms_taichi_chunked(
    audio_f32: np.ndarray,
    n: int,
    n_frames: int,
    frame_size: int,
    hop_size: int,
) -> np.ndarray:
    """Taichi CUDA分块RMS计算"""
    # 每次处理的帧数上限（控制显存）
    max_frames_per_batch = MAX_CHUNK_SAMPLES // frame_size
    if max_frames_per_batch < 1:
        max_frames_per_batch = 1

    all_rms = np.zeros(n_frames, dtype=np.float32)

    for batch_start in range(0, n_frames, max_frames_per_batch):
        batch_end = min(batch_start + max_frames_per_batch, n_frames)
        batch_n_frames = batch_end - batch_start

        # 计算这批帧需要的音频范围
        audio_start = batch_start * hop_size
        audio_end = min((batch_end - 1) * hop_size + frame_size, n)
        audio_chunk = audio_f32[audio_start:audio_end]
        chunk_len = len(audio_chunk)

        # Taichi fields
        src = ti.field(dtype=ti.f32, shape=chunk_len)
        rms = ti.field(dtype=ti.f32, shape=batch_n_frames)
        src.from_numpy(audio_chunk)

        fs = frame_size
        hs = hop_size
        cl = chunk_len
        bnf = batch_n_frames

        @ti.kernel
        def calc():
            for fi in range(bnf):
                start = fi * hs
                energy = 0.0
                count = 0
                for j in range(fs):
                    idx = start + j
                    if idx < cl:
                        energy += src[idx] * src[idx]  # noqa: F821
                        count += 1
                if count > 0:
                    rms[fi] = ti.sqrt(energy / ti.cast(count, ti.f32))  # noqa: F821
                else:
                    rms[fi] = 0.0  # noqa: F821

        calc()
        all_rms[batch_start:batch_end] = rms.to_numpy()
        del src, rms

    return all_rms


def _rms_numpy_chunked(
    audio_f32: np.ndarray,
    n: int,
    n_frames: int,
    frame_size: int,
    hop_size: int,
) -> np.ndarray:
    """
    NumPy分块RMS计算
    绝不一次性创建 (n_frames, frame_size) 大矩阵
    """
    all_rms = np.zeros(n_frames, dtype=np.float32)

    # 每批处理的帧数（控制内存：batch_frames * frame_size * 4bytes）
    # 目标：每批 < 100MB
    max_mem_bytes = 100 * 1024 * 1024  # 100MB
    max_batch_frames = max(1, max_mem_bytes // (frame_size * 4))

    for batch_start in range(0, n_frames, max_batch_frames):
        batch_end = min(batch_start + max_batch_frames, n_frames)
        batch_size = batch_end - batch_start

        # 构建索引矩阵（仅这一小批）
        frame_offsets = np.arange(batch_size) * hop_size + batch_start * hop_size
        sample_offsets = np.arange(frame_size)

        # (batch_size, frame_size)
        indices = frame_offsets[:, None] + sample_offsets[None, :]
        indices = np.clip(indices, 0, n - 1)

        frames = audio_f32[indices]  # (batch_size, frame_size)
        all_rms[batch_start:batch_end] = np.sqrt(np.mean(frames ** 2, axis=1))

    return all_rms


# ═══════════════════════════════════════════════════════════════════════════
# VAD Mask
# ═══════════════════════════════════════════════════════════════════════════

def compute_vad_mask_fast(
    rms_energy: np.ndarray,
    threshold: float = 0.01,
    min_speech_frames: int = 5,
    min_silence_frames: int = 10,
) -> np.ndarray:
    """VAD掩码（数据量小，纯NumPy即可）"""
    mask = (rms_energy > threshold).astype(np.int32)
    return _apply_min_duration(mask, min_speech_frames, min_silence_frames)


def _apply_min_duration(
    mask: np.ndarray, min_speech: int, min_silence: int
) -> np.ndarray:
    result = mask.copy()
    n = len(result)
    i = 0
    while i < n:
        val = result[i]
        j = i
        while j < n and result[j] == val:
            j += 1
        seg_len = j - i
        if val == 1 and seg_len < min_speech:
            result[i:j] = 0
        elif val == 0 and seg_len < min_silence:
            result[i:j] = 1
        i = j
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Chunk Splitting
# ═══════════════════════════════════════════════════════════════════════════

def choose_chunk_duration_sec(
    audio_duration_sec: float,
    chunk_mode: str = "auto",
    fixed_chunk_sec: float = 300.0,
    min_chunk_sec: float = 90.0,
    max_chunk_sec: float = 480.0,
    target_chunk_sec: float = 240.0,
) -> float:
    """Choose chunk duration automatically by audio length (or keep fixed mode)."""
    duration = max(0.0, float(audio_duration_sec or 0.0))
    mode = str(chunk_mode or "auto").strip().lower()
    fixed = max(1.0, float(fixed_chunk_sec or 300.0))
    min_chunk = max(15.0, float(min_chunk_sec or 90.0))
    max_chunk = max(min_chunk, float(max_chunk_sec or max(min_chunk, fixed)))
    target = max(min_chunk, min(max_chunk, float(target_chunk_sec or 240.0)))

    if mode == "fixed":
        return max(min_chunk, min(max_chunk, fixed))

    if duration <= 0:
        return target
    if duration <= min_chunk * 1.15:
        return duration

    chunk_count = max(1, int(round(duration / target)))
    chunk_count = max(chunk_count, int(math.ceil(duration / max_chunk)))
    chunk_duration = duration / float(chunk_count)
    return max(min_chunk, min(max_chunk, chunk_duration))


def split_audio_chunks_fast(
    audio_np: np.ndarray,
    sample_rate: int,
    chunk_duration_sec: float,
    overlap_sec: float = 1.0,
    vad_mask: Optional[np.ndarray] = None,
    frame_hop_samples: int = 256,
) -> List[Tuple[np.ndarray, float, float]]:
    """智能分块，跳过静音段"""
    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    n = len(audio_np)
    chunk_samples = int(chunk_duration_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    if chunk_samples <= 0:
        raise ValueError("chunk size computed as 0, check chunk_duration_sec/sample_rate")
    step = chunk_samples - overlap_samples
    if step <= 0:
        step = max(chunk_samples // 2, 1)

    chunks = []
    pos = 0

    while pos < n:
        end = min(pos + chunk_samples, n)
        start_sec = pos / sample_rate
        end_sec = end / sample_rate

        # VAD检查
        if vad_mask is not None:
            fs = pos // frame_hop_samples
            fe = min(end // frame_hop_samples, len(vad_mask))
            if fs < fe and np.mean(vad_mask[fs:fe]) < 0.05:
                pos += step
                logger.debug(f"Skipped silent [{start_sec:.1f}s-{end_sec:.1f}s]")
                continue

        # Avoid unconditional copy here to reduce memory traffic.
        # ascontiguousarray returns a view when possible.
        chunk_audio = np.ascontiguousarray(audio_np[pos:end], dtype=np.float32)
        chunks.append((chunk_audio, start_sec, end_sec))
        pos += step

    logger.info(
        f"Split: {len(chunks)} chunks "
        f"(chunk={chunk_duration_sec}s, overlap={overlap_sec}s)"
    )
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Segment Merging / Dedup
# ═══════════════════════════════════════════════════════════════════════════

def merge_overlapping_segments(
    segments: List[dict],
    merge_gap_sec: float = 0.3,
    same_speaker_only: bool = True,
) -> List[dict]:
    """合并相邻/重叠段落"""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.get("start", 0))
    merged = [sorted_segs[0].copy()]

    for seg in sorted_segs[1:]:
        prev = merged[-1]
        can_merge = (
            seg.get("start", 0) <= prev.get("end", 0) + merge_gap_sec
            and (
                not same_speaker_only
                or seg.get("speaker") == prev.get("speaker")
            )
        )
        if can_merge:
            prev["end"] = max(prev.get("end", 0), seg.get("end", 0))
            prev_text = str(prev.get("text", "") or "")
            curr_text = str(seg.get("text", "") or "")
            overlap_len = _suffix_prefix_overlap(prev_text, curr_text)
            if overlap_len >= 5:
                curr_text = curr_text[overlap_len:].lstrip()
            if curr_text.strip():
                joiner = " " if prev_text and not prev_text.endswith((" ", "\n")) else ""
                prev["text"] = prev_text + joiner + curr_text
        else:
            merged.append(seg.copy())

    return merged


def _speaker_value(seg: dict) -> str:
    return str(seg.get("speaker", "") or "").strip()


def _normalize_text_for_match(text: str) -> str:
    return " ".join(str(text or "").strip().split()).lower()


def _suffix_prefix_overlap(
    left_text: str,
    right_text: str,
    min_match_chars: int = 5,
    max_scan_chars: int = 120,
) -> int:
    left = str(left_text or "")
    right = str(right_text or "")
    if not left or not right:
        return 0

    best = 0
    limit = min(len(left), len(right), max_scan_chars)
    for length in range(min_match_chars, limit + 1):
        if left.endswith(right[:length]):
            best = length
    return best


def deduplicate_overlap_text(
    segments: List[dict],
    overlap_sec: float = 1.0,
    same_speaker_only: bool = True,
    keep_cross_speaker_overlap: bool = True,
) -> List[dict]:
    """去除重叠区域的重复文本"""
    if len(segments) <= 1:
        return segments

    sorted_segs = sorted(
        segments,
        key=lambda s: (
            float(s.get("start", 0) or 0.0),
            float(s.get("end", 0) or 0.0),
        ),
    )

    max_overlap = max(float(overlap_sec), 0.0) + 0.5
    result = [sorted_segs[0].copy()]

    for curr_raw in sorted_segs[1:]:
        curr = curr_raw.copy()
        prev = result[-1]

        prev_start = float(prev.get("start", 0) or 0.0)
        prev_end = float(prev.get("end", 0) or 0.0)
        curr_start = float(curr.get("start", 0) or 0.0)
        curr_end = float(curr.get("end", 0) or 0.0)
        overlap_len = prev_end - curr_start

        prev_text = str(prev.get("text", "") or "")
        curr_text = str(curr.get("text", "") or "")
        prev_norm = _normalize_text_for_match(prev_text)
        curr_norm = _normalize_text_for_match(curr_text)
        same_speaker = _speaker_value(prev) == _speaker_value(curr)

        near_same_time = (
            abs(prev_start - curr_start) <= 0.35
            and abs(prev_end - curr_end) <= 0.35
        )
        if prev_norm and prev_norm == curr_norm and near_same_time:
            prev_conf = float(prev.get("confidence", 0.0) or 0.0)
            curr_conf = float(curr.get("confidence", 0.0) or 0.0)
            if curr_conf > prev_conf:
                result[-1] = curr
            continue

        allow_trim = (not same_speaker_only) or same_speaker
        if allow_trim and 0 < overlap_len <= max_overlap:
            overlap_chars = _suffix_prefix_overlap(prev_text, curr_text)
            if overlap_chars >= 5:
                curr["text"] = curr_text[overlap_chars:].lstrip()
                curr.pop("words", None)
                if same_speaker or not keep_cross_speaker_overlap:
                    curr["start"] = max(curr_start, prev_end)

        if curr.get("text", "").strip():
            result.append(curr)

    return result


def smooth_segment_boundaries(
    segments: List[dict],
    max_gap_sec: float = 0.45,
    max_overlap_sec: float = 1.2,
    min_duration_sec: float = 0.08,
) -> List[dict]:
    """
    Smooth small gap/overlap between adjacent segments without merging text blocks.
    Keeps sentence-level granularity while making chunk boundaries natural.
    """
    if len(segments) <= 1:
        return [seg.copy() for seg in segments]

    fixed = sorted(
        (seg.copy() for seg in segments),
        key=lambda s: float(s.get("start", 0.0) or 0.0),
    )
    min_dur = max(0.02, float(min_duration_sec or 0.08))
    max_gap = max(0.0, float(max_gap_sec or 0.45))
    max_overlap = max(0.0, float(max_overlap_sec or 1.2))

    for idx in range(1, len(fixed)):
        prev = fixed[idx - 1]
        curr = fixed[idx]

        prev_start = float(prev.get("start", 0.0) or 0.0)
        prev_end = float(prev.get("end", prev_start) or prev_start)
        curr_start = float(curr.get("start", 0.0) or 0.0)
        curr_end = float(curr.get("end", curr_start) or curr_start)

        if prev_end <= prev_start:
            prev_end = prev_start + min_dur
        if curr_end <= curr_start:
            curr_end = curr_start + min_dur

        gap = curr_start - prev_end
        should_smooth = (0.0 < gap <= max_gap) or (0.0 < -gap <= max_overlap)
        if not should_smooth:
            prev["end"] = round(prev_end, 3)
            curr["start"] = round(curr_start, 3)
            continue

        boundary = (prev_end + curr_start) * 0.5
        new_prev_end = max(prev_start + min_dur, boundary)
        new_curr_start = min(curr_end - min_dur, boundary)

        if new_prev_end > new_curr_start:
            middle = (prev_start + curr_end) * 0.5
            new_prev_end = min(
                prev_end,
                max(prev_start + min_dur, middle - min_dur * 0.5),
            )
            new_curr_start = max(
                curr_start,
                min(curr_end - min_dur, middle + min_dur * 0.5),
            )
            if new_prev_end > new_curr_start:
                new_prev_end = min(prev_end, curr_end - min_dur)
                new_curr_start = max(curr_start, prev_start + min_dur)

        prev["end"] = round(new_prev_end, 3)
        curr["start"] = round(new_curr_start, 3)

    return fixed
