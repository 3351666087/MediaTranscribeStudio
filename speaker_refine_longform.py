#!/usr/bin/env python3
"""
Long-form speaker refinement for transcript JSON outputs.

This script is designed as a conservative post-refiner for diarization output:

1. Split the original transcript into contiguous speaker runs instead of trusting
   the raw speaker label globally.
2. Compute global speaker embeddings for each run.
3. Merge runs with strong acoustic evidence while respecting cannot-link
   constraints such as overlapping timestamps.
4. Keep very short / low-evidence runs as standalone speakers unless there is
   clear evidence that they belong to an existing speaker.

The structure is intentionally aligned with the underlying principles used by
NeMo diarization clustering:
  - global speaker embeddings matter more than local duration heuristics
  - speaker labels need long-range stitching, not just local smoothing
  - conservative history-like merging is safer than aggressive collapsing

Usage:
    python3 speaker_refine_longform.py path/to/transcript.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import sitecustomize  # noqa: F401
except Exception:
    sitecustomize = None  # type: ignore[assignment]

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "speaker_refine_longform.py requires numpy. "
        "Run it with the project environment, for example: "
        "/Users/null3351/Desktop/1/.venv-macos/bin/python speaker_refine_longform.py ..."
    ) from exc

from runtime_paths import find_tool_executable


APP_ROOT = Path(__file__).resolve().parent


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_vec(values: Any) -> Optional[np.ndarray]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-6:
        return None
    return np.ascontiguousarray(arr / norm, dtype=np.float32)


def _cosine_similarity(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> float:
    a = _normalize_vec(left)
    b = _normalize_vec(right)
    if a is None or b is None:
        return float("nan")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _resample_audio_linear_np(x: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if x.size <= 0:
        return np.zeros((0,), dtype=np.float32)

    src = int(source_sr)
    dst = int(target_sr)
    if src <= 0 or dst <= 0 or src == dst:
        return np.ascontiguousarray(x, dtype=np.float32).reshape(-1)

    in_arr = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out_len = max(1, int(round(float(in_arr.size) * float(dst) / float(src))))
    if out_len == in_arr.size:
        return in_arr
    src_idx = np.arange(in_arr.size, dtype=np.float32)
    dst_idx = np.linspace(0.0, float(max(0, in_arr.size - 1)), out_len, dtype=np.float32)
    return np.interp(dst_idx, src_idx, in_arr).astype(np.float32, copy=False)


def _energy_score(samples: np.ndarray) -> float:
    if samples.size <= 0:
        return 0.0
    wav = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.maximum(1e-9, np.mean(wav * wav))))
    if wav.size < 128:
        return rms

    frame = min(wav.size, 512)
    window = np.hanning(frame).astype(np.float32, copy=False)
    frag = wav[:frame] if wav.size == frame else wav[(wav.size - frame) // 2 : (wav.size + frame) // 2]
    spec = np.abs(np.fft.rfft(frag * window))
    spec = np.maximum(spec, 1e-8)
    flatness = float(np.exp(np.mean(np.log(spec))) / np.maximum(np.mean(spec), 1e-8))
    return rms * max(0.05, 1.05 - flatness)


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
    if audio.size <= 0 or sample_rate <= 0:
        return []

    total_samples = int(audio.size)
    min_sec = max(0.4, float(min_sec))
    max_sec = max(min_sec, float(max_sec))
    view_sec = max(min_sec, min(float(view_sec), max_sec))

    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec + 1e-3, float(end_sec))
    duration = end_sec - start_sec

    local_start = max(0.0, start_sec - edge_pad_sec)
    local_end = min(float(total_samples) / float(sample_rate), end_sec + edge_pad_sec)
    local_duration = max(1e-3, local_end - local_start)

    local_s = max(0, min(total_samples, int(round(local_start * sample_rate))))
    local_e = max(local_s + 1, min(total_samples, int(round(local_end * sample_rate))))
    local_clip = np.ascontiguousarray(audio[local_s:local_e], dtype=np.float32)
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

    candidates: List[Tuple[float, int, np.ndarray]] = []
    step = max(1, target_view_samples // 2)
    center_start = max(0, min(local_clip.size - target_view_samples, (local_clip.size - target_view_samples) // 2))
    center_clip = local_clip[center_start : center_start + target_view_samples]
    candidates.append((_energy_score(center_clip) + 0.03, center_start, center_clip))

    for offset in range(0, max(1, local_clip.size - target_view_samples + 1), step):
        frag = local_clip[offset : offset + target_view_samples]
        score = _energy_score(frag)
        if score <= 0.0:
            continue
        candidates.append((score, offset, frag))

    chosen: List[Tuple[int, np.ndarray]] = []
    for _score, offset, frag in sorted(candidates, key=lambda item: (-item[0], abs(item[1] - center_start))):
        if any(abs(offset - existing_offset) < max(1, target_view_samples // 2) for existing_offset, _ in chosen):
            continue
        chosen.append((offset, np.ascontiguousarray(frag, dtype=np.float32)))
        if len(chosen) >= max(1, int(max_views)):
            break

    if not chosen:
        chosen = [(center_start, center_clip)]

    return [frag for _offset, frag in sorted(chosen, key=lambda item: item[0])]


class BaseEmbeddingBackend:
    name = "base"
    sample_rate = 16000

    def embed_clip(self, clip: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


class LightweightEmbeddingBackend(BaseEmbeddingBackend):
    name = "lightweight"
    sample_rate = 16000

    def embed_clip(self, clip: np.ndarray) -> Optional[np.ndarray]:
        wav = _resample_audio_linear_np(clip, source_sr=self.sample_rate, target_sr=self.sample_rate)
        if wav.size <= 0:
            return None

        min_samples = max(256, int(0.8 * self.sample_rate))
        if wav.size < min_samples:
            wav = np.pad(wav, (0, int(min_samples - wav.size)), mode="constant")
        wav = np.ascontiguousarray(wav, dtype=np.float32)

        frame = max(128, int(self.sample_rate * 0.025))
        hop = max(64, int(self.sample_rate * 0.010))
        if wav.size < frame:
            wav = np.pad(wav, (0, int(frame - wav.size)), mode="constant")

        starts = list(range(0, max(1, wav.size - frame + 1), hop))
        if not starts:
            starts = [0]

        window = np.hanning(frame).astype(np.float32, copy=False)
        nyq = max(1.0, float(self.sample_rate) * 0.5)
        frame_feats: List[List[float]] = []
        for start in starts:
            frag = wav[start : start + frame]
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
            frame_feats.append(
                [
                    rms,
                    zcr,
                    centroid_hz / nyq,
                    rolloff_hz / nyq,
                    flatness,
                ]
            )

        ff = np.asarray(frame_feats, dtype=np.float32)
        if ff.ndim != 2 or ff.shape[0] <= 0:
            return None

        emb = np.concatenate(
            [
                ff.mean(axis=0),
                ff.std(axis=0),
                np.array([float(wav.size) / float(max(1, self.sample_rate))], dtype=np.float32),
            ],
            axis=0,
        )
        return _normalize_vec(emb)


class NemoSpeakerEmbeddingBackend(BaseEmbeddingBackend):
    name = "nemo_titanet"
    sample_rate = 16000

    def __init__(self, model_spec: str, *, device: str = "cpu", allow_download: bool = False):
        try:
            import torch
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except Exception as exc:
            raise RuntimeError(f"NeMo speaker backend unavailable: {exc}") from exc

        self._torch = torch
        self._model_cls = EncDecSpeakerLabelModel
        self._device = "cuda" if str(device).strip().lower().startswith("cuda") and torch.cuda.is_available() else "cpu"
        self._allow_download = bool(allow_download)
        self._model = self._load_model(model_spec)
        self._model.eval()
        self._model.freeze()
        self._model.to(self._device)

    @staticmethod
    def _candidate_model_paths(explicit: str) -> Iterable[Path]:
        raw = str(explicit or "").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.exists() and p.is_file():
                yield p.resolve()

        known_names = [
            "speakerverification_en_titanet_large.nemo",
            "titanet_large.nemo",
        ]
        roots = [
            APP_ROOT / "output_files" / ".nemo_msdd",
            APP_ROOT / "output_files",
            APP_ROOT,
        ]
        for root in roots:
            if not root.exists():
                continue
            for name in known_names:
                direct = root / name
                if direct.exists() and direct.is_file():
                    yield direct.resolve()
            for name in known_names:
                matches = sorted(root.rglob(name))
                for match in matches[:2]:
                    if match.is_file():
                        yield match.resolve()

    def _load_model(self, model_spec: str):
        last_error: Optional[Exception] = None
        for candidate in self._candidate_model_paths(model_spec):
            try:
                return self._model_cls.restore_from(str(candidate), map_location=self._device)
            except Exception as exc:
                last_error = exc

        model_name = str(model_spec or "").strip() or "nvidia/speakerverification_en_titanet_large"
        if self._allow_download:
            try:
                return self._model_cls.from_pretrained(model_name=model_name, map_location=self._device)
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"Failed to load NeMo speaker model: {last_error}") from last_error
        raise RuntimeError(
            "No local NeMo speaker model found. Pass --speaker-model /path/to/model.nemo "
            "or use --allow-model-download."
        )

    def embed_clip(self, clip: np.ndarray) -> Optional[np.ndarray]:
        wav = _resample_audio_linear_np(clip, source_sr=self.sample_rate, target_sr=self.sample_rate)
        if wav.size <= 0:
            return None
        wav = np.ascontiguousarray(wav, dtype=np.float32).reshape(-1)
        if wav.size < int(0.5 * self.sample_rate):
            wav = np.pad(wav, (0, int(0.5 * self.sample_rate) - wav.size), mode="constant")

        with self._torch.no_grad():
            emb, _ = self._model.infer_segment(wav)
        if emb is None:
            return None

        arr = emb.detach().to("cpu").float().numpy().reshape(-1)
        return _normalize_vec(arr)


def _choose_backend(
    backend_name: str,
    *,
    speaker_model: str,
    device: str,
    allow_model_download: bool,
) -> BaseEmbeddingBackend:
    requested = str(backend_name or "auto").strip().lower()
    if requested in {"auto", "nemo", "nemo_titanet"}:
        try:
            return NemoSpeakerEmbeddingBackend(
                model_spec=speaker_model,
                device=device,
                allow_download=allow_model_download,
            )
        except Exception as exc:
            if requested in {"nemo", "nemo_titanet"}:
                raise
            print(f"[speaker-refine] NeMo backend unavailable, fallback to lightweight backend: {exc}", file=sys.stderr)
    return LightweightEmbeddingBackend()


def _decode_audio_with_ffmpeg(audio_path: Path, target_sr: int) -> Optional[Tuple[np.ndarray, int]]:
    ffmpeg_bin = find_tool_executable("ffmpeg")
    if not ffmpeg_bin:
        return None

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-f",
        "f32le",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception:
        return None

    audio = np.frombuffer(proc.stdout or b"", dtype=np.float32)
    if audio.size <= 0:
        return None
    return np.ascontiguousarray(audio, dtype=np.float32), int(target_sr)


def _decode_audio_with_soundfile(audio_path: Path, target_sr: int) -> Optional[Tuple[np.ndarray, int]]:
    try:
        import soundfile as sf
    except Exception:
        sf = None

    if sf is not None:
        try:
            audio, sr = sf.read(str(audio_path), always_2d=False, dtype="float32")
            if isinstance(audio, np.ndarray) and audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
            if audio.size > 0:
                if int(sr) != int(target_sr):
                    audio = _resample_audio_linear_np(audio, source_sr=int(sr), target_sr=int(target_sr))
                    sr = target_sr
                return audio, int(sr)
        except Exception:
            pass

    try:
        import torchaudio
    except Exception:
        torchaudio = None

    if torchaudio is not None:
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            wav = waveform.detach().cpu().float().numpy()
            if wav.ndim == 2:
                wav = np.mean(wav, axis=0)
            wav = np.ascontiguousarray(wav.reshape(-1), dtype=np.float32)
            if wav.size > 0:
                if int(sr) != int(target_sr):
                    wav = _resample_audio_linear_np(wav, source_sr=int(sr), target_sr=int(target_sr))
                    sr = target_sr
                return wav, int(sr)
        except Exception:
            pass

    return None


def decode_audio(audio_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    resolved = _decode_audio_with_ffmpeg(audio_path, target_sr=target_sr)
    if resolved is not None:
        return resolved

    resolved = _decode_audio_with_soundfile(audio_path, target_sr=target_sr)
    if resolved is not None:
        return resolved

    raise RuntimeError(f"Failed to decode audio: {audio_path}")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _label_for_index(index: int) -> str:
    idx = int(index)
    if idx < 0:
        idx = 0
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if idx < len(letters):
        return letters[idx]
    return f"S{idx}"


@dataclass
class SegmentItem:
    index: int
    start: float
    end: float
    text: str
    raw_speaker: str
    payload: Dict[str, Any]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class RunItem:
    run_id: int
    raw_speaker: str
    segment_indices: List[int]
    start: float
    end: float
    centroid: Optional[np.ndarray]
    view_embeddings: List[np.ndarray]
    quality: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def count(self) -> int:
        return len(self.segment_indices)


@dataclass
class RawSpeakerStats:
    speaker: str
    duration: float
    run_count: int
    centroid: Optional[np.ndarray]
    cohesion: float

    @property
    def stable(self) -> bool:
        return (
            (self.duration >= 10.0 or self.run_count >= 4)
            and math.isfinite(self.cohesion)
            and self.cohesion >= 0.68
        )


@dataclass
class Cluster:
    cluster_id: int
    run_ids: List[int]
    segment_indices: List[int]
    raw_speakers: set[str]
    centroid: Optional[np.ndarray]
    duration: float
    quality: float
    cohesion: float
    stable_hint: bool
    first_start: float
    last_end: float

    @property
    def run_count(self) -> int:
        return len(self.run_ids)

    @property
    def segment_count(self) -> int:
        return len(self.segment_indices)

    @property
    def short(self) -> bool:
        return self.duration <= 2.0 or self.run_count <= 1


def _load_transcript(path: Path) -> Tuple[Dict[str, Any], List[SegmentItem]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError(f"Invalid transcript JSON: missing 'segments' list in {path}")

    segments: List[SegmentItem] = []
    for idx, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            continue
        start = max(0.0, _safe_float(item.get("start"), 0.0))
        end = max(start + 1e-3, _safe_float(item.get("end"), start))
        speaker = str(item.get("speaker", "0") or "0").strip() or "0"
        text = str(item.get("text", "") or "")
        segments.append(
            SegmentItem(
                index=idx,
                start=start,
                end=end,
                text=text,
                raw_speaker=speaker,
                payload=dict(item),
            )
        )

    if not segments:
        raise ValueError(f"No usable segments found in {path}")

    segments.sort(key=lambda item: (item.start, item.end, item.index))
    return payload, segments


def _resolve_audio_path(payload: Dict[str, Any], transcript_path: Path, explicit_audio_path: str) -> Path:
    if explicit_audio_path:
        audio_path = Path(explicit_audio_path).expanduser()
        if audio_path.exists():
            return audio_path.resolve()
        raise FileNotFoundError(f"Audio path not found: {audio_path}")

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    candidate_values = [
        metadata.get("source_path"),
        metadata.get("audio_path"),
        metadata.get("media_path"),
    ]
    for raw in candidate_values:
        if not raw:
            continue
        candidate = Path(str(raw)).expanduser()
        if candidate.exists():
            return candidate.resolve()
        local = (transcript_path.parent / candidate.name).resolve()
        if local.exists():
            return local

    raise FileNotFoundError(
        "Could not resolve source audio. Pass --audio-path or ensure metadata.source_path exists."
    )


def _split_runs(segments: Sequence[SegmentItem], join_gap_sec: float) -> List[List[int]]:
    runs: List[List[int]] = []
    current: List[int] = []
    prev: Optional[SegmentItem] = None
    for idx, seg in enumerate(segments):
        if prev is None:
            current = [idx]
        else:
            gap = max(0.0, seg.start - prev.end)
            if seg.raw_speaker == prev.raw_speaker and gap <= join_gap_sec:
                current.append(idx)
            else:
                runs.append(current)
                current = [idx]
        prev = seg
    if current:
        runs.append(current)
    return runs


def _embed_run(
    run_id: int,
    run_indices: Sequence[int],
    segments: Sequence[SegmentItem],
    audio: np.ndarray,
    sample_rate: int,
    backend: BaseEmbeddingBackend,
    *,
    min_view_sec: float,
    max_view_sec: float,
    per_view_sec: float,
    max_views: int,
) -> RunItem:
    first = segments[run_indices[0]]
    last = segments[run_indices[-1]]
    views = _choose_embedding_views(
        audio=audio,
        sample_rate=sample_rate,
        start_sec=first.start,
        end_sec=last.end,
        min_sec=min_view_sec,
        max_sec=max_view_sec,
        view_sec=per_view_sec,
        max_views=max_views,
    )

    embeddings: List[np.ndarray] = []
    quality_values: List[float] = []
    for view in views:
        view_audio = _resample_audio_linear_np(view, source_sr=sample_rate, target_sr=backend.sample_rate)
        emb = backend.embed_clip(view_audio)
        if emb is None:
            continue
        embeddings.append(emb)
        quality_values.append(_energy_score(view_audio))

    centroid = None
    if embeddings:
        centroid = _normalize_vec(np.mean(np.stack(embeddings, axis=0), axis=0))

    quality = max(0.0, float(np.mean(quality_values))) if quality_values else 0.0
    return RunItem(
        run_id=run_id,
        raw_speaker=first.raw_speaker,
        segment_indices=list(run_indices),
        start=first.start,
        end=last.end,
        centroid=centroid,
        view_embeddings=embeddings,
        quality=quality,
    )


def _cluster_centroid(
    run_ids: Sequence[int],
    runs: Dict[int, RunItem],
) -> Tuple[Optional[np.ndarray], float, float]:
    vectors: List[np.ndarray] = []
    weights: List[float] = []
    for run_id in run_ids:
        run = runs[run_id]
        if run.centroid is None:
            continue
        vectors.append(run.centroid)
        weights.append(max(0.25, min(8.0, run.duration)) * max(0.1, run.quality))
    if not vectors:
        return None, 0.0, float("nan")

    w = np.asarray(weights, dtype=np.float32)
    mat = np.stack(vectors, axis=0)
    centroid = _normalize_vec(np.average(mat, axis=0, weights=w))
    if centroid is None:
        return None, 0.0, float("nan")

    sims = []
    for vec, weight in zip(vectors, weights):
        sim = _cosine_similarity(vec, centroid)
        if math.isfinite(sim):
            sims.append((sim, weight))
    if not sims:
        return centroid, float(np.sum(weights)), float("nan")

    total_weight = sum(weight for _sim, weight in sims)
    cohesion = sum(sim * weight for sim, weight in sims) / max(total_weight, 1e-6)
    return centroid, float(np.sum(weights)), float(cohesion)


def _build_raw_speaker_stats(runs: Sequence[RunItem]) -> Dict[str, RawSpeakerStats]:
    grouped: Dict[str, List[RunItem]] = {}
    for run in runs:
        grouped.setdefault(run.raw_speaker, []).append(run)

    out: Dict[str, RawSpeakerStats] = {}
    for speaker, items in grouped.items():
        centroid, _weight_sum, cohesion = _cluster_centroid([item.run_id for item in items], {item.run_id: item for item in items})
        duration = sum(item.duration for item in items)
        out[speaker] = RawSpeakerStats(
            speaker=speaker,
            duration=float(duration),
            run_count=len(items),
            centroid=centroid,
            cohesion=float(cohesion) if math.isfinite(cohesion) else float("nan"),
        )
    return out


def _build_initial_clusters(runs: Sequence[RunItem], raw_stats: Dict[str, RawSpeakerStats]) -> Dict[int, Cluster]:
    out: Dict[int, Cluster] = {}
    for run in runs:
        stat = raw_stats.get(run.raw_speaker)
        stable_hint = bool(stat and stat.stable and run.duration >= 0.6)
        cohesion = 1.0 if run.centroid is not None else float("nan")
        out[run.run_id] = Cluster(
            cluster_id=run.run_id,
            run_ids=[run.run_id],
            segment_indices=list(run.segment_indices),
            raw_speakers={run.raw_speaker},
            centroid=run.centroid,
            duration=float(run.duration),
            quality=float(run.quality),
            cohesion=float(cohesion),
            stable_hint=stable_hint,
            first_start=float(run.start),
            last_end=float(run.end),
        )
    return out


def _has_time_conflict(cluster_a: Cluster, cluster_b: Cluster, segments: Sequence[SegmentItem], overlap_tol: float = 0.05) -> bool:
    a_indices = sorted(cluster_a.segment_indices)
    b_indices = sorted(cluster_b.segment_indices)
    if not a_indices or not b_indices:
        return False

    left = [(segments[idx].start, segments[idx].end) for idx in a_indices]
    right = [(segments[idx].start, segments[idx].end) for idx in b_indices]
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        a_start, a_end = left[i]
        b_start, b_end = right[j]
        overlap = min(a_end, b_end) - max(a_start, b_start)
        if overlap > overlap_tol:
            return True
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return False


def _cross_run_support(cluster_a: Cluster, cluster_b: Cluster, runs: Dict[int, RunItem]) -> float:
    sims: List[float] = []
    for run_id_a in cluster_a.run_ids:
        run_a = runs[run_id_a]
        if run_a.centroid is None:
            continue
        for run_id_b in cluster_b.run_ids:
            run_b = runs[run_id_b]
            if run_b.centroid is None:
                continue
            sim = _cosine_similarity(run_a.centroid, run_b.centroid)
            if math.isfinite(sim):
                sims.append(sim)

    if not sims:
        return 0.0

    sims.sort(reverse=True)
    top = sims[: min(8, len(sims))]
    strong_frac = sum(1.0 for value in top if value >= 0.78) / max(1, len(top))
    return 0.5 * float(np.mean(top)) + 0.5 * strong_frac


def _merged_cluster(
    next_cluster_id: int,
    cluster_a: Cluster,
    cluster_b: Cluster,
    runs: Dict[int, RunItem],
    raw_stats: Dict[str, RawSpeakerStats],
) -> Cluster:
    run_ids = sorted(cluster_a.run_ids + cluster_b.run_ids)
    segment_indices = sorted(cluster_a.segment_indices + cluster_b.segment_indices)
    centroid, _weight_sum, cohesion = _cluster_centroid(run_ids, runs)
    raw_speakers = set(cluster_a.raw_speakers) | set(cluster_b.raw_speakers)
    duration = float(cluster_a.duration + cluster_b.duration)
    quality = float(np.mean([cluster_a.quality, cluster_b.quality]))
    stable_hint = False
    for speaker in raw_speakers:
        stat = raw_stats.get(speaker)
        if stat and stat.stable:
            stable_hint = True
            break

    return Cluster(
        cluster_id=next_cluster_id,
        run_ids=run_ids,
        segment_indices=segment_indices,
        raw_speakers=raw_speakers,
        centroid=centroid,
        duration=duration,
        quality=quality,
        cohesion=float(cohesion) if math.isfinite(cohesion) else float("nan"),
        stable_hint=stable_hint,
        first_start=min(cluster_a.first_start, cluster_b.first_start),
        last_end=max(cluster_a.last_end, cluster_b.last_end),
    )


def _pair_merge_score(
    cluster_a: Cluster,
    cluster_b: Cluster,
    *,
    runs: Dict[int, RunItem],
    raw_stats: Dict[str, RawSpeakerStats],
    merged_preview: Cluster,
) -> Tuple[float, float, Dict[str, Any]]:
    similarity = _cosine_similarity(cluster_a.centroid, cluster_b.centroid)
    if not math.isfinite(similarity):
        return float("nan"), float("inf"), {"reason": "no_embedding"}

    shared_raw = sorted(cluster_a.raw_speakers & cluster_b.raw_speakers)
    support = _cross_run_support(cluster_a, cluster_b, runs)
    both_long = (
        cluster_a.duration >= 8.0
        and cluster_b.duration >= 8.0
        and cluster_a.run_count >= 2
        and cluster_b.run_count >= 2
    )
    either_short = cluster_a.short or cluster_b.short
    shared_raw_cohesion = max(
        [raw_stats[s].cohesion for s in shared_raw if s in raw_stats and math.isfinite(raw_stats[s].cohesion)],
        default=float("nan"),
    )

    threshold = 0.84
    if shared_raw:
        threshold = 0.78 if math.isfinite(shared_raw_cohesion) and shared_raw_cohesion >= 0.70 else 0.84
    elif either_short:
        threshold = 0.81
    if both_long:
        threshold = max(threshold, 0.88)
    if support < 0.55 and not shared_raw:
        threshold += 0.03

    adjusted = similarity
    if shared_raw:
        adjusted += 0.03
    if support >= 0.82:
        adjusted += 0.02
    elif support < 0.40:
        adjusted -= 0.02
    if cluster_a.stable_hint and cluster_b.stable_hint and not shared_raw:
        adjusted -= 0.02

    merged_cohesion = merged_preview.cohesion
    min_input_cohesion = min(
        value for value in [cluster_a.cohesion, cluster_b.cohesion] if math.isfinite(value)
    ) if any(math.isfinite(value) for value in [cluster_a.cohesion, cluster_b.cohesion]) else float("nan")
    if math.isfinite(merged_cohesion):
        if merged_cohesion < 0.58 and not either_short:
            threshold = float("inf")
        elif math.isfinite(min_input_cohesion) and merged_cohesion + 0.10 < min_input_cohesion and not either_short:
            threshold = float("inf")

    info = {
        "similarity": round(similarity, 4),
        "support": round(support, 4),
        "threshold": round(threshold, 4) if math.isfinite(threshold) else None,
        "adjusted_score": round(adjusted, 4),
        "shared_raw_speakers": shared_raw,
        "merged_cohesion": round(merged_cohesion, 4) if math.isfinite(merged_cohesion) else None,
        "cluster_a_duration": round(cluster_a.duration, 3),
        "cluster_b_duration": round(cluster_b.duration, 3),
    }
    return adjusted, threshold, info


def _greedy_merge_clusters(
    clusters: Dict[int, Cluster],
    *,
    segments: Sequence[SegmentItem],
    runs: Dict[int, RunItem],
    raw_stats: Dict[str, RawSpeakerStats],
) -> Tuple[Dict[int, Cluster], List[Dict[str, Any]]]:
    active = dict(clusters)
    next_cluster_id = max(active) + 1 if active else 0
    merge_log: List[Dict[str, Any]] = []

    while True:
        active_ids = sorted(active)
        best_choice: Optional[Tuple[float, float, int, int, Cluster, Dict[str, Any]]] = None
        for idx, cluster_id_a in enumerate(active_ids):
            cluster_a = active[cluster_id_a]
            for cluster_id_b in active_ids[idx + 1 :]:
                cluster_b = active[cluster_id_b]
                if _has_time_conflict(cluster_a, cluster_b, segments):
                    continue

                merged_preview = _merged_cluster(
                    next_cluster_id=next_cluster_id,
                    cluster_a=cluster_a,
                    cluster_b=cluster_b,
                    runs=runs,
                    raw_stats=raw_stats,
                )
                score, threshold, info = _pair_merge_score(
                    cluster_a,
                    cluster_b,
                    runs=runs,
                    raw_stats=raw_stats,
                    merged_preview=merged_preview,
                )
                if not math.isfinite(score) or not math.isfinite(threshold):
                    continue
                if score < threshold:
                    continue

                margin = score - threshold
                candidate = (margin, score, cluster_id_a, cluster_id_b, merged_preview, info)
                if best_choice is None or candidate[:2] > best_choice[:2]:
                    best_choice = candidate

        if best_choice is None:
            break

        _margin, score, cluster_id_a, cluster_id_b, merged_cluster, info = best_choice
        cluster_a = active.pop(cluster_id_a)
        cluster_b = active.pop(cluster_id_b)
        merged_cluster.cluster_id = next_cluster_id
        active[next_cluster_id] = merged_cluster
        merge_log.append(
            {
                "merge_type": "global_merge",
                "from_clusters": [cluster_id_a, cluster_id_b],
                "from_raw_speakers": [sorted(cluster_a.raw_speakers), sorted(cluster_b.raw_speakers)],
                "to_cluster": next_cluster_id,
                "score": round(score, 4),
                **info,
            }
        )
        next_cluster_id += 1

    return active, merge_log


def _nearest_context_cluster(
    cluster: Cluster,
    clusters: Dict[int, Cluster],
    segments: Sequence[SegmentItem],
) -> Tuple[Optional[int], Optional[int]]:
    segment_to_cluster: Dict[int, int] = {}
    for cluster_id, item in clusters.items():
        for seg_index in item.segment_indices:
            segment_to_cluster[seg_index] = cluster_id

    seg_indices = sorted(cluster.segment_indices)
    if not seg_indices:
        return None, None

    left_cluster = None
    right_cluster = None
    first_index = seg_indices[0]
    last_index = seg_indices[-1]

    for idx in range(first_index - 1, -1, -1):
        candidate = segment_to_cluster.get(idx)
        if candidate is not None and candidate != cluster.cluster_id:
            left_cluster = candidate
            break

    for idx in range(last_index + 1, len(segments)):
        candidate = segment_to_cluster.get(idx)
        if candidate is not None and candidate != cluster.cluster_id:
            right_cluster = candidate
            break

    return left_cluster, right_cluster


def _attach_minor_clusters(
    clusters: Dict[int, Cluster],
    *,
    segments: Sequence[SegmentItem],
    runs: Dict[int, RunItem],
    raw_stats: Dict[str, RawSpeakerStats],
) -> Tuple[Dict[int, Cluster], List[Dict[str, Any]]]:
    active = dict(clusters)
    attach_log: List[Dict[str, Any]] = []
    next_cluster_id = max(active) + 1 if active else 0

    def _best_target(source_cluster: Cluster) -> Tuple[Optional[int], float, Dict[str, Any]]:
        left_context, right_context = _nearest_context_cluster(source_cluster, active, segments)
        best_target = None
        best_score = float("-inf")
        best_info: Dict[str, Any] = {}
        for target_id, target_cluster in active.items():
            if target_id == source_cluster.cluster_id:
                continue
            if _has_time_conflict(source_cluster, target_cluster, segments):
                continue

            merged_preview = _merged_cluster(
                next_cluster_id=next_cluster_id,
                cluster_a=source_cluster,
                cluster_b=target_cluster,
                runs=runs,
                raw_stats=raw_stats,
            )
            score, threshold, info = _pair_merge_score(
                source_cluster,
                target_cluster,
                runs=runs,
                raw_stats=raw_stats,
                merged_preview=merged_preview,
            )
            if not math.isfinite(score):
                continue

            neighbor_bonus = 0.0
            if left_context == target_id:
                neighbor_bonus += 0.04
            if right_context == target_id:
                neighbor_bonus += 0.04
            adjusted_threshold = threshold
            if neighbor_bonus >= 0.08 and math.isfinite(adjusted_threshold):
                adjusted_threshold = min(adjusted_threshold, 0.76)
            if math.isfinite(adjusted_threshold) and score + neighbor_bonus < adjusted_threshold:
                continue

            total_score = score + neighbor_bonus
            if total_score > best_score:
                best_target = target_id
                best_score = total_score
                best_info = {
                    **info,
                    "neighbor_bonus": round(neighbor_bonus, 4),
                    "left_context_cluster": left_context,
                    "right_context_cluster": right_context,
                }
        return best_target, best_score, best_info

    candidate_ids = [
        cluster_id
        for cluster_id, cluster in sorted(active.items(), key=lambda item: (item[1].duration, item[1].first_start))
        if cluster.duration <= 1.8 or (cluster.run_count == 1 and cluster.duration <= 3.0)
    ]

    for cluster_id in candidate_ids:
        cluster = active.get(cluster_id)
        if cluster is None:
            continue

        best_target, best_score, info = _best_target(cluster)
        if best_target is None:
            continue

        target_cluster = active.get(best_target)
        if target_cluster is None:
            continue

        merged_cluster = _merged_cluster(
            next_cluster_id=next_cluster_id,
            cluster_a=cluster,
            cluster_b=target_cluster,
            runs=runs,
            raw_stats=raw_stats,
        )
        active.pop(cluster_id, None)
        active.pop(best_target, None)
        active[next_cluster_id] = merged_cluster
        attach_log.append(
            {
                "merge_type": "minor_attach",
                "from_cluster": cluster_id,
                "to_cluster": best_target,
                "new_cluster": next_cluster_id,
                "score": round(best_score, 4),
                "from_raw_speakers": sorted(cluster.raw_speakers),
                "to_raw_speakers": sorted(target_cluster.raw_speakers),
                **info,
            }
        )
        next_cluster_id += 1

    return active, attach_log


def _assign_labels(clusters: Dict[int, Cluster]) -> Tuple[Dict[int, str], Dict[int, int]]:
    ordered = sorted(
        clusters.items(),
        key=lambda item: (
            item[1].first_start,
            item[1].last_end,
            min(item[1].segment_indices) if item[1].segment_indices else item[0],
        ),
    )
    cluster_labels: Dict[int, str] = {}
    cluster_order: Dict[int, int] = {}
    for ordinal, (cluster_id, _cluster) in enumerate(ordered):
        cluster_labels[cluster_id] = _label_for_index(ordinal)
        cluster_order[cluster_id] = ordinal
    return cluster_labels, cluster_order


def refine_speakers(
    transcript_path: Path,
    *,
    audio_path: str = "",
    backend_name: str = "auto",
    speaker_model: str = "",
    device: str = "cpu",
    allow_model_download: bool = False,
    join_gap_sec: float = 0.75,
    min_view_sec: float = 0.8,
    max_view_sec: float = 6.0,
    per_view_sec: float = 1.8,
    max_views: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload, segments = _load_transcript(transcript_path)
    resolved_audio_path = _resolve_audio_path(payload, transcript_path, audio_path)
    backend = _choose_backend(
        backend_name,
        speaker_model=speaker_model,
        device=device,
        allow_model_download=allow_model_download,
    )

    audio, sample_rate = decode_audio(resolved_audio_path, target_sr=backend.sample_rate)
    if audio.size <= 0:
        raise RuntimeError(f"Decoded empty audio: {resolved_audio_path}")

    run_groups = _split_runs(segments, join_gap_sec=join_gap_sec)
    runs: List[RunItem] = []
    for run_id, run_indices in enumerate(run_groups):
        runs.append(
            _embed_run(
                run_id=run_id,
                run_indices=run_indices,
                segments=segments,
                audio=audio,
                sample_rate=sample_rate,
                backend=backend,
                min_view_sec=min_view_sec,
                max_view_sec=max_view_sec,
                per_view_sec=per_view_sec,
                max_views=max_views,
            )
        )

    runs_by_id = {run.run_id: run for run in runs}
    raw_stats = _build_raw_speaker_stats(runs)
    initial_clusters = _build_initial_clusters(runs, raw_stats)

    refined_clusters, merge_log = _greedy_merge_clusters(
        initial_clusters,
        segments=segments,
        runs=runs_by_id,
        raw_stats=raw_stats,
    )
    refined_clusters, attach_log = _attach_minor_clusters(
        refined_clusters,
        segments=segments,
        runs=runs_by_id,
        raw_stats=raw_stats,
    )

    cluster_labels, cluster_order = _assign_labels(refined_clusters)
    run_to_cluster: Dict[int, int] = {}
    for cluster_id, cluster in refined_clusters.items():
        for run_id in cluster.run_ids:
            run_to_cluster[run_id] = cluster_id

    segment_cluster_map: Dict[int, int] = {}
    for run in runs:
        cluster_id = run_to_cluster.get(run.run_id)
        if cluster_id is None:
            continue
        for segment_index in run.segment_indices:
            segment_cluster_map[segment_index] = cluster_id

    output_by_original_index: Dict[int, Dict[str, Any]] = {}
    for sorted_pos, seg in enumerate(segments):
        cluster_id = segment_cluster_map.get(sorted_pos)
        label = cluster_labels.get(cluster_id, seg.raw_speaker)
        updated = dict(seg.payload)
        updated["speaker_raw"] = seg.raw_speaker
        updated["speaker"] = label
        if cluster_id is not None:
            updated["speaker_refine_cluster_id"] = int(cluster_order.get(cluster_id, -1))
        output_by_original_index[seg.index] = updated

    output_segments = [
        output_by_original_index[idx]
        for idx in sorted(output_by_original_index)
    ]

    input_speakers = sorted({seg.raw_speaker for seg in segments})
    output_speakers = [cluster_labels[cluster_id] for cluster_id in sorted(cluster_labels, key=lambda cid: cluster_order[cid])]

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata["num_segments"] = len(output_segments)
    metadata["num_speakers"] = len(output_speakers)
    metadata["speakers"] = output_speakers
    metadata["speaker_refinement"] = {
        "refiner": "longform_graph_v1",
        "generated_at": datetime.now().isoformat(),
        "backend": backend.name,
        "audio_path": str(resolved_audio_path),
        "input_speakers": input_speakers,
        "output_speakers": output_speakers,
        "input_num_speakers": len(input_speakers),
        "output_num_speakers": len(output_speakers),
        "input_runs": len(runs),
        "global_merges": len(merge_log),
        "minor_attaches": len(attach_log),
        "join_gap_sec": round(join_gap_sec, 3),
        "principles": [
            "run_level_initialization",
            "global_embedding_merging",
            "overlap_cannot_link",
            "conservative_minor_attachment",
        ],
    }

    refined_payload = {
        "metadata": metadata,
        "segments": output_segments,
    }

    report = {
        "version": "longform_graph_v1",
        "generated_at": datetime.now().isoformat(),
        "transcript_path": str(transcript_path.resolve()),
        "audio_path": str(resolved_audio_path),
        "backend": backend.name,
        "input": {
            "segments": len(segments),
            "raw_speakers": input_speakers,
            "num_raw_speakers": len(input_speakers),
            "runs": len(runs),
        },
        "output": {
            "speakers": output_speakers,
            "num_speakers": len(output_speakers),
            "clusters": len(refined_clusters),
        },
        "raw_speaker_stats": {
            speaker: {
                "duration": round(stat.duration, 3),
                "run_count": stat.run_count,
                "cohesion": round(stat.cohesion, 4) if math.isfinite(stat.cohesion) else None,
                "stable": stat.stable,
            }
            for speaker, stat in sorted(raw_stats.items())
        },
        "cluster_summary": [
            {
                "cluster_id": cluster_order[cluster_id],
                "speaker": cluster_labels[cluster_id],
                "raw_speakers": sorted(cluster.raw_speakers),
                "duration": round(cluster.duration, 3),
                "run_count": cluster.run_count,
                "segment_count": cluster.segment_count,
                "cohesion": round(cluster.cohesion, 4) if math.isfinite(cluster.cohesion) else None,
            }
            for cluster_id, cluster in sorted(
                refined_clusters.items(),
                key=lambda item: cluster_order[item[0]],
            )
        ],
        "merge_log": merge_log + attach_log,
    }
    return refined_payload, report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine long-form speaker labels in transcript JSON output.")
    parser.add_argument("transcript_json", help="Path to the transcript JSON produced by the app.")
    parser.add_argument("--audio-path", default="", help="Optional source audio / video path. Defaults to metadata.source_path.")
    parser.add_argument("--output", default="", help="Output refined JSON path. Default: <input>.refined.json")
    parser.add_argument("--report", default="", help="Optional refinement report JSON path. Default: <input>.refine_report.json")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "lightweight", "nemo"),
        help="Embedding backend. 'auto' prefers NeMo and falls back to lightweight.",
    )
    parser.add_argument("--speaker-model", default="", help="Optional local NeMo speaker model path.")
    parser.add_argument("--device", default="cpu", help="Embedding device for NeMo backend. Default: cpu")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow NeMo backend to download the speaker model when not found locally.",
    )
    parser.add_argument("--join-gap-sec", type=float, default=0.75, help="Max gap for merging adjacent same-speaker segments into one run.")
    parser.add_argument("--min-view-sec", type=float, default=0.8, help="Minimum audio span used for run embedding.")
    parser.add_argument("--max-view-sec", type=float, default=6.0, help="Maximum audio span per run view.")
    parser.add_argument("--per-view-sec", type=float, default=1.8, help="Preferred duration of one embedding view.")
    parser.add_argument("--max-views", type=int, default=3, help="Maximum embedding views per run.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input transcript JSON.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    transcript_path = Path(args.transcript_json).expanduser().resolve()
    if not transcript_path.exists():
        print(f"Transcript JSON not found: {transcript_path}", file=sys.stderr)
        return 2

    if args.in_place and args.output:
        print("--in-place and --output cannot be used together.", file=sys.stderr)
        return 2

    output_path = Path(args.output).expanduser() if args.output else transcript_path.with_name(f"{transcript_path.stem}.refined.json")
    if args.in_place:
        output_path = transcript_path
    report_path = Path(args.report).expanduser() if args.report else transcript_path.with_name(f"{transcript_path.stem}.refine_report.json")

    try:
        refined_payload, report = refine_speakers(
            transcript_path=transcript_path,
            audio_path=args.audio_path,
            backend_name=args.backend,
            speaker_model=args.speaker_model,
            device=args.device,
            allow_model_download=bool(args.allow_model_download),
            join_gap_sec=float(args.join_gap_sec),
            min_view_sec=float(args.min_view_sec),
            max_view_sec=float(args.max_view_sec),
            per_view_sec=float(args.per_view_sec),
            max_views=max(1, int(args.max_views)),
        )
    except Exception as exc:
        print(f"Speaker refinement failed: {exc}", file=sys.stderr)
        return 1

    _atomic_write_json(output_path, refined_payload)
    _atomic_write_json(report_path, report)

    input_count = report["input"]["num_raw_speakers"]
    output_count = report["output"]["num_speakers"]
    print(f"Refined speakers: {input_count} -> {output_count}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
