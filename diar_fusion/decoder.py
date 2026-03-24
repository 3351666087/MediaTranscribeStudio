from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from runtime_paths import APP_ROOT

from .calibration import FEATURE_NAMES, PosteriorFusionCalibrator, resolve_calibrator_path
from .native_backend import load_posterior_fusion_backend
from .training import dump_decoder_example, resolve_example_dump_dir
from .utils import (
    build_overlap_seed_regions,
    build_strong_reference_speaker_map,
    count_diar_speakers,
    merge_adjacent_speaker_turns,
    merge_time_regions,
    normalize_diar_segments_for_fusion,
    safe_bool,
    safe_float,
    safe_int,
)


@dataclass
class PosteriorFusionDecodeResult:
    segments: List[Dict[str, Any]]
    overlap_seed_regions: List[Dict[str, float]]
    route: str
    target_count: int
    overlap_tracks: List[Dict[str, Any]] = field(default_factory=list)
    calibrator_updates: int = 0
    example_dump_path: str = ""


class PosteriorFusionDecoder:
    """
    Framewise posterior fusion for diarization backends.

    Objective:
      maximize sum_t phi_t(z_t)^T w
             + prior_scale * pi(z_t)
             + stay_bonus * I[z_t = z_{t-1}]
             - switch_penalty * (1 - boundary_t) * I[z_t != z_{t-1}]

    where phi_t encodes backend support, pairwise agreement, boundary support and
    exclusivity cues, while pi is a global speaker prior derived from agreement,
    exclusive support, anchor strength, and a Sortformer saturation penalty. A
    lightweight online calibrator updates w from high-consensus pseudo labels to
    adapt to the project domain.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._backend = None
        self._calibrators: Dict[str, PosteriorFusionCalibrator] = {}

    @staticmethod
    def _hybrid_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        root = cfg if isinstance(cfg, dict) else {}
        hybrid_cfg = root.get("hybrid_fusion", {}) if isinstance(root, dict) else {}
        return hybrid_cfg if isinstance(hybrid_cfg, dict) else {}

    def _posterior_cfg(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        hybrid_cfg = self._hybrid_cfg(cfg)
        posterior_cfg = hybrid_cfg.get("posterior_decoder", {}) if isinstance(hybrid_cfg, dict) else {}
        if isinstance(posterior_cfg, dict):
            return posterior_cfg
        return {}

    @staticmethod
    def _requested_num_speakers(cfg: Optional[Dict[str, Any]]) -> int:
        root = cfg if isinstance(cfg, dict) else {}
        return max(0, safe_int(root.get("num_speakers", 0), 0))

    @staticmethod
    def _manual_count_weight_scale(
        backend_name: str,
        *,
        backend_count: int,
        requested_num_speakers: int,
    ) -> float:
        requested = max(0, int(requested_num_speakers or 0))
        count = max(0, int(backend_count or 0))
        if requested <= 0 or count <= 0:
            return 1.0
        if count == requested:
            return {
                "msdd": 3.8,
                "pyannote": 3.5,
                "sortformer": 3.2,
            }.get(str(backend_name or ""), 3.4)
        distance = abs(count - requested)
        if distance <= 1:
            return 0.18
        if distance == 2:
            return 0.10
        return 0.05

    def _calibrator(
        self,
        cfg: Optional[Dict[str, Any]],
        *,
        output_root: Optional[Path] = None,
    ) -> PosteriorFusionCalibrator:
        posterior_cfg = self._posterior_cfg(cfg)
        calib_cfg = posterior_cfg.get("calibrator", {}) if isinstance(posterior_cfg, dict) else {}
        calib_cfg = calib_cfg if isinstance(calib_cfg, dict) else {}
        persist_raw = str(calib_cfg.get("persist_path", "output_files/.posterior_fusion_calibrator.json") or "")
        persist_path = resolve_calibrator_path(
            persist_raw,
            root_dir=APP_ROOT,
            output_root=output_root,
        )
        cache_key = str(persist_path)
        calibrator = self._calibrators.get(cache_key)
        if calibrator is not None:
            return calibrator
        calibrator = PosteriorFusionCalibrator(cfg=calib_cfg, persist_path=persist_path)
        self._calibrators[cache_key] = calibrator
        return calibrator

    def _native_backend(self):
        if self._backend is not None:
            return self._backend
        self._backend = load_posterior_fusion_backend()
        return self._backend

    @staticmethod
    def _normalize_prior(values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.size <= 0:
            return vector
        mean = float(np.mean(vector))
        std = float(np.std(vector))
        if std < 1e-6:
            return np.zeros_like(vector, dtype=np.float32)
        return ((vector - mean) / std).astype(np.float32)

    @staticmethod
    def _backend_merge_gap(root_cfg: Dict[str, Any], key: str, default_value: float) -> float:
        section = root_cfg.get(key, {}) if isinstance(root_cfg, dict) else {}
        if not isinstance(section, dict):
            section = {}
        return safe_float(section.get("merge_gap_sec", default_value), default_value)

    def _heuristic_target_speaker_count(
        self,
        *,
        msdd_count: int,
        pyannote_count: int,
        sortformer_count: int,
        sortformer_turn_count: int,
        audio_duration: float,
        max_speakers: int,
        cfg: Optional[Dict[str, Any]],
    ) -> int:
        hybrid_cfg = self._hybrid_cfg(cfg)
        keep_msdd_longform_sec = max(
            60.0,
            safe_float(hybrid_cfg.get("count_longform_preserve_msdd_sec", 8 * 60), 8 * 60),
        )
        clamp_delta = max(
            0,
            safe_int(hybrid_cfg.get("count_msdd_pyannote_delta", 1), 1),
        )
        target = 0
        if msdd_count > 0 and pyannote_count > 0:
            if float(audio_duration) >= keep_msdd_longform_sec and msdd_count > pyannote_count:
                target = msdd_count
            else:
                target = max(pyannote_count, min(msdd_count, pyannote_count + clamp_delta))
        elif pyannote_count > 0:
            target = pyannote_count
        elif msdd_count > 0:
            target = msdd_count
        elif sortformer_count > 0:
            target = min(sortformer_count, 4)
        return int(max(1, min(max_speakers, target or 1)))

    @staticmethod
    def _count_proximity_weight(
        backend_count: int,
        candidate_count: int,
        *,
        soft_radius: float,
    ) -> float:
        backend_value = int(backend_count)
        candidate_value = int(candidate_count)
        if backend_value <= 0 or candidate_value <= 0:
            return 0.0
        radius = max(0.25, float(soft_radius))
        distance = abs(float(backend_value) - float(candidate_value))
        return float(1.0 / (1.0 + distance / radius))

    def _choose_target_speaker_count(
        self,
        *,
        msdd_count: int,
        pyannote_count: int,
        sortformer_count: int,
        sortformer_turn_count: int,
        audio_duration: float,
        max_speakers: int,
        quality_scores: Optional[Dict[str, float]],
        msdd_weight: float,
        pyannote_weight: float,
        sortformer_weight: float,
        cfg: Optional[Dict[str, Any]],
    ) -> tuple[int, Dict[str, Any]]:
        requested_num_speakers = self._requested_num_speakers(cfg)
        if requested_num_speakers > 0:
            locked_target = int(max(1, min(max_speakers, requested_num_speakers)))
            return locked_target, {
                "mode": "manual",
                "locked": True,
                "heuristic_target": locked_target,
                "weighted_mean": float(locked_target),
                "candidate_meta": {
                    locked_target: {
                        "score": float(locked_target),
                        "locked": True,
                        "direct_voters": [
                            name
                            for name, count in (
                                ("msdd", msdd_count),
                                ("pyannote", pyannote_count),
                                ("sortformer", sortformer_count),
                            )
                            if int(count) == locked_target
                        ],
                    }
                },
            }

        hybrid_cfg = self._hybrid_cfg(cfg)
        posterior_cfg = self._posterior_cfg(cfg)
        heuristic_target = self._heuristic_target_speaker_count(
            msdd_count=msdd_count,
            pyannote_count=pyannote_count,
            sortformer_count=sortformer_count,
            sortformer_turn_count=sortformer_turn_count,
            audio_duration=audio_duration,
            max_speakers=max_speakers,
            cfg=cfg,
        )
        base_support_floor = max(
            0.05,
            min(0.60, safe_float(posterior_cfg.get("count_quality_floor", 0.12), 0.12)),
        )
        soft_radius = max(
            0.5,
            safe_float(posterior_cfg.get("count_soft_radius", 1.0), 1.0),
        )
        consensus_bonus = max(
            0.0,
            safe_float(posterior_cfg.get("count_consensus_bonus", 0.18), 0.18),
        )
        pair_bonus = max(
            0.0,
            safe_float(posterior_cfg.get("count_pair_bonus", 0.16), 0.16),
        )
        heuristic_bonus = max(
            0.0,
            safe_float(posterior_cfg.get("count_heuristic_bonus", 0.18), 0.18),
        )
        direct_support_bonus = max(
            0.0,
            safe_float(posterior_cfg.get("count_direct_support_bonus", 0.45), 0.45),
        )
        mean_distance_penalty = max(
            0.0,
            safe_float(posterior_cfg.get("count_mean_distance_penalty", 0.08), 0.08),
        )
        sortformer_only_penalty = max(
            0.0,
            safe_float(posterior_cfg.get("count_sortformer_only_penalty", 0.22), 0.22),
        )
        sortformer_cap_penalty = max(
            0.0,
            safe_float(posterior_cfg.get("count_sortformer_cap_penalty", 0.12), 0.12),
        )
        keep_msdd_longform_sec = max(
            60.0,
            safe_float(hybrid_cfg.get("count_longform_preserve_msdd_sec", 8 * 60), 8 * 60),
        )
        longform_msdd_bonus = max(
            0.0,
            safe_float(posterior_cfg.get("count_longform_msdd_bonus", 0.18), 0.18),
        )

        backend_counts = {
            "msdd": int(max(0, msdd_count)),
            "pyannote": int(max(0, pyannote_count)),
            "sortformer": int(max(0, sortformer_count)),
        }
        backend_weights = {
            "msdd": max(0.05, float(msdd_weight)),
            "pyannote": max(0.05, float(pyannote_weight)),
            "sortformer": max(0.05, float(sortformer_weight)),
        }
        normalized_quality = {
            name: max(
                base_support_floor,
                min(1.0, safe_float((quality_scores or {}).get(name, 0.0), 0.0)),
            )
            for name, count in backend_counts.items()
            if count > 0
        }
        backend_support = {
            name: backend_weights[name] * normalized_quality[name]
            for name in normalized_quality
        }

        candidate_counts = {
            int(max(1, min(max_speakers, value)))
            for value in (
                heuristic_target,
                backend_counts["msdd"],
                backend_counts["pyannote"],
                backend_counts["sortformer"],
            )
            if int(value) > 0
        }
        if not candidate_counts:
            candidate_counts = {1}

        total_support = float(sum(backend_support.values()))
        weighted_mean = float(heuristic_target)
        if total_support > 1e-6:
            weighted_mean = sum(
                float(backend_counts[name]) * float(backend_support.get(name, 0.0))
                for name in backend_support
            ) / total_support

        candidate_meta: Dict[int, Dict[str, Any]] = {}
        sortformer_limit = min(4, max(1, max_speakers))
        for candidate in sorted(candidate_counts):
            score = 0.0
            direct_voters: List[str] = []
            direct_support = 0.0
            per_backend: Dict[str, float] = {}
            for backend_name, backend_count in backend_counts.items():
                if backend_count <= 0:
                    continue
                support = float(backend_support.get(backend_name, 0.0))
                proximity = self._count_proximity_weight(
                    backend_count,
                    candidate,
                    soft_radius=soft_radius,
                )
                contribution = support * proximity
                per_backend[backend_name] = contribution
                score += contribution
                if backend_count == candidate:
                    direct_voters.append(backend_name)
                    direct_support += support

            direct_vote_count = len(direct_voters)
            if direct_support > 0.0:
                score += direct_support_bonus * direct_support
            if direct_vote_count > 1 and direct_support > 0.0:
                score += consensus_bonus * (direct_support / float(direct_vote_count)) * float(direct_vote_count - 1)
            if msdd_count > 0 and pyannote_count > 0 and msdd_count == pyannote_count == candidate:
                pair_support = 0.5 * (
                    float(backend_support.get("msdd", 0.0))
                    + float(backend_support.get("pyannote", 0.0))
                )
                score += pair_bonus * pair_support
            elif candidate == heuristic_target and total_support > 0.0:
                heuristic_support = max(
                    float(backend_support.get("msdd", 0.0)),
                    float(backend_support.get("pyannote", 0.0)),
                    direct_support / float(max(1, direct_vote_count)),
                )
                score += heuristic_bonus * heuristic_support
            if (
                candidate == msdd_count
                and candidate == heuristic_target
                and float(audio_duration) >= keep_msdd_longform_sec
                and msdd_count > pyannote_count
            ):
                score += longform_msdd_bonus * float(backend_support.get("msdd", 0.0))

            if (
                backend_counts.get("sortformer", 0) == candidate
                and direct_voters == ["sortformer"]
                and candidate > heuristic_target
            ):
                score -= sortformer_only_penalty * float(backend_support.get("sortformer", 0.0)) * float(
                    candidate - heuristic_target
                )
            if (
                backend_counts.get("sortformer", 0) == candidate
                and candidate >= sortformer_limit
                and candidate > max(msdd_count, pyannote_count)
            ):
                score -= sortformer_cap_penalty * float(backend_support.get("sortformer", 0.0))

            score -= mean_distance_penalty * abs(float(candidate) - float(weighted_mean))
            candidate_meta[int(candidate)] = {
                "score": float(score),
                "direct_votes": int(direct_vote_count),
                "direct_support": float(direct_support),
                "per_backend": per_backend,
            }

        ranked_candidates = sorted(
            candidate_meta.items(),
            key=lambda item: (
                float(item[1]["score"]),
                int(item[1]["direct_votes"]),
                float(item[1]["direct_support"]),
                -abs(float(item[0]) - float(weighted_mean)),
                -abs(float(item[0]) - float(heuristic_target)),
                -float(item[0]),
            ),
            reverse=True,
        )
        target = int(ranked_candidates[0][0]) if ranked_candidates else int(heuristic_target)
        return target, {
            "heuristic_target": int(heuristic_target),
            "weighted_mean": float(weighted_mean),
            "backend_support": backend_support,
            "candidate_meta": candidate_meta,
        }

    @staticmethod
    def _build_boundary_scores(
        *,
        duration: float,
        frame_hop_sec: float,
        snap_sec: float,
        msdd_segments: List[Dict[str, Any]],
        pyannote_segments: List[Dict[str, Any]],
        sortformer_segments: List[Dict[str, Any]],
        msdd_weight: float,
        pyannote_weight: float,
        sortformer_weight: float,
    ) -> np.ndarray:
        num_frames = max(1, int(math.ceil(max(duration, 0.0) / max(frame_hop_sec, 1e-3))))
        centers = (np.arange(num_frames, dtype=np.float32) + 0.5) * float(frame_hop_sec)
        centers = np.clip(centers, 0.0, max(0.0, duration))
        points: List[tuple[float, float]] = [(0.0, pyannote_weight), (max(0.0, duration), pyannote_weight)]
        for segments, weight in (
            (msdd_segments, msdd_weight),
            (pyannote_segments, pyannote_weight),
            (sortformer_segments, sortformer_weight),
        ):
            for item in segments:
                points.append((float(item.get("start", 0.0) or 0.0), float(weight)))
                points.append((float(item.get("end", 0.0) or 0.0), float(weight)))
        if not points:
            return np.zeros((num_frames,), dtype=np.float32)
        scores = np.zeros((num_frames,), dtype=np.float32)
        norm = max(1e-6, float(sum(weight for _point, weight in points)))
        decay = max(1e-3, float(snap_sec))
        for point, weight in points:
            scores += float(weight) * np.exp(-np.abs(centers - float(point)) / decay)
        return np.clip(scores / norm, 0.0, 1.0)

    @staticmethod
    def _segments_to_frame_matrix(
        *,
        diar_segments: List[Dict[str, Any]],
        speaker_to_index: Dict[str, int],
        num_frames: int,
        frame_hop_sec: float,
        canonical_map: Optional[Dict[str, str]] = None,
        allow_unmapped: bool = False,
    ) -> np.ndarray:
        matrix = np.zeros((num_frames, len(speaker_to_index)), dtype=np.float32)
        if not diar_segments or not speaker_to_index:
            return matrix
        for item in diar_segments:
            raw_speaker = str(item.get("speaker", "") or "").strip()
            if not raw_speaker:
                continue
            canonical = str(canonical_map.get(raw_speaker, "")) if isinstance(canonical_map, dict) else raw_speaker
            canonical = canonical.strip()
            if not canonical and allow_unmapped:
                canonical = raw_speaker
            if canonical not in speaker_to_index:
                continue
            start = max(0.0, float(item.get("start", 0.0) or 0.0))
            end = max(start, float(item.get("end", start) or start))
            if end <= start:
                continue
            frame_start = max(0, int(math.floor(start / max(frame_hop_sec, 1e-6))))
            frame_end = min(num_frames - 1, int(math.ceil(end / max(frame_hop_sec, 1e-6))))
            for frame_index in range(frame_start, frame_end + 1):
                win_start = float(frame_index) * frame_hop_sec
                win_end = win_start + frame_hop_sec
                overlap = max(0.0, min(end, win_end) - max(start, win_start))
                if overlap <= 0.0:
                    continue
                score = min(1.0, overlap / max(frame_hop_sec, 1e-6))
                idx = int(speaker_to_index[canonical])
                if score > matrix[frame_index, idx]:
                    matrix[frame_index, idx] = score
        return matrix

    @staticmethod
    def _exclusive_support(pyannote_matrix: np.ndarray, overlap_mask: np.ndarray) -> np.ndarray:
        if pyannote_matrix.size <= 0:
            return np.zeros_like(pyannote_matrix, dtype=np.float32)
        active = (pyannote_matrix > 0.25).astype(np.float32)
        counts = np.sum(active, axis=1, keepdims=True)
        exclusive = np.where(counts <= 1.0, pyannote_matrix, 0.0)
        if overlap_mask.size > 0:
            exclusive = exclusive * (1.0 - overlap_mask.reshape(-1, 1))
        return np.clip(exclusive, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _frame_coverage(matrix: np.ndarray) -> float:
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.ndim != 2 or arr.size <= 0:
            return 0.0
        active = np.any(arr > 0.25, axis=1)
        if active.size <= 0:
            return 0.0
        return float(np.mean(active.astype(np.float32)))

    def _backend_quality_score(
        self,
        *,
        matrix: np.ndarray,
        peer_matrices: List[np.ndarray],
        diar_segments: List[Dict[str, Any]],
        audio_duration: float,
        consensus_count: int,
    ) -> float:
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.ndim != 2 or arr.size <= 0 or not diar_segments:
            return 0.0

        support_mass = float(np.sum(arr, dtype=np.float32))
        if support_mass <= 1e-6:
            return 0.0

        peer_union = np.zeros_like(arr, dtype=np.float32)
        valid_peers = [np.asarray(item, dtype=np.float32) for item in peer_matrices if np.asarray(item).shape == arr.shape]
        if valid_peers:
            peer_union = np.maximum.reduce(valid_peers).astype(np.float32)

        agreement = 0.5
        exclusive_ratio = 0.0
        coverage_penalty = 0.0
        if valid_peers:
            agreement = float(np.sum(np.minimum(arr, peer_union), dtype=np.float32)) / max(support_mass, 1e-6)
            exclusive_ratio = float(np.sum(np.clip(arr - peer_union, 0.0, 1.0), dtype=np.float32)) / max(
                support_mass, 1e-6
            )
            peer_coverage = float(np.mean([self._frame_coverage(item) for item in valid_peers]))
            if peer_coverage > 0.0:
                own_coverage = self._frame_coverage(arr)
                coverage_penalty = min(1.0, abs(own_coverage - peer_coverage) / max(0.10, peer_coverage + 0.05))

        durations = [
            max(
                0.0,
                float(item.get("end", item.get("start", 0.0)) or 0.0) - float(item.get("start", 0.0) or 0.0),
            )
            for item in diar_segments
        ]
        valid_durations = [value for value in durations if value > 0.0]
        short_turn_ratio = (
            float(sum(1 for value in valid_durations if value <= 0.60)) / float(len(valid_durations))
            if valid_durations
            else 1.0
        )
        minutes = max(1e-3, float(audio_duration) / 60.0)
        turn_density = float(len(valid_durations)) / minutes
        turn_penalty = min(1.0, turn_density / 28.0)

        own_count = max(1, count_diar_speakers(diar_segments))
        ref_count = max(1, int(consensus_count))
        count_penalty = min(1.0, abs(own_count - ref_count) / max(1.0, float(ref_count)))

        quality = (
            0.48 * max(0.0, min(1.0, agreement))
            + 0.16 * (1.0 - max(0.0, min(1.0, exclusive_ratio)))
            + 0.14 * (1.0 - max(0.0, min(1.0, short_turn_ratio)))
            + 0.10 * (1.0 - max(0.0, min(1.0, turn_penalty)))
            + 0.12 * (1.0 - max(0.0, min(1.0, count_penalty)))
        )
        quality *= max(0.45, 1.0 - 0.50 * max(0.0, min(1.0, coverage_penalty)))
        return float(np.clip(quality, 0.12, 1.0))

    def _backend_quality_scores(
        self,
        *,
        msdd_matrix: np.ndarray,
        pyannote_matrix: np.ndarray,
        sortformer_matrix: np.ndarray,
        msdd_segments: List[Dict[str, Any]],
        pyannote_segments: List[Dict[str, Any]],
        sortformer_segments: List[Dict[str, Any]],
        audio_duration: float,
    ) -> Dict[str, float]:
        counts = [
            count
            for count in (
                count_diar_speakers(msdd_segments),
                count_diar_speakers(pyannote_segments),
                count_diar_speakers(sortformer_segments),
            )
            if count > 0
        ]
        consensus_count = int(round(float(np.median(np.asarray(counts, dtype=np.float32))))) if counts else 1
        return {
            "msdd": self._backend_quality_score(
                matrix=msdd_matrix,
                peer_matrices=[pyannote_matrix, sortformer_matrix],
                diar_segments=msdd_segments,
                audio_duration=audio_duration,
                consensus_count=consensus_count,
            ),
            "pyannote": self._backend_quality_score(
                matrix=pyannote_matrix,
                peer_matrices=[msdd_matrix, sortformer_matrix],
                diar_segments=pyannote_segments,
                audio_duration=audio_duration,
                consensus_count=consensus_count,
            ),
            "sortformer": self._backend_quality_score(
                matrix=sortformer_matrix,
                peer_matrices=[msdd_matrix, pyannote_matrix],
                diar_segments=sortformer_segments,
                audio_duration=audio_duration,
                consensus_count=consensus_count,
            ),
        }

    def _feature_tensor(
        self,
        *,
        msdd_matrix: np.ndarray,
        pyannote_matrix: np.ndarray,
        sortformer_matrix: np.ndarray,
        boundary_scores: np.ndarray,
        overlap_mask: np.ndarray,
        anchor_mask: np.ndarray,
        quality_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        max_frames = max(msdd_matrix.shape[0], pyannote_matrix.shape[0], sortformer_matrix.shape[0])
        max_speakers = max(msdd_matrix.shape[1], pyannote_matrix.shape[1], sortformer_matrix.shape[1])
        tensor = np.zeros((max_frames, max_speakers, len(FEATURE_NAMES)), dtype=np.float32)
        msdd = msdd_matrix if msdd_matrix.size > 0 else np.zeros((max_frames, max_speakers), dtype=np.float32)
        pyannote = pyannote_matrix if pyannote_matrix.size > 0 else np.zeros((max_frames, max_speakers), dtype=np.float32)
        sortformer = sortformer_matrix if sortformer_matrix.size > 0 else np.zeros((max_frames, max_speakers), dtype=np.float32)

        agreement = (
            (msdd > 0.25).astype(np.float32)
            + (pyannote > 0.25).astype(np.float32)
            + (sortformer > 0.25).astype(np.float32)
        ) / 3.0
        exclusive = self._exclusive_support(pyannote, overlap_mask)
        boundary = boundary_scores.reshape(-1, 1).astype(np.float32)

        tensor[:, :, 0] = msdd
        tensor[:, :, 1] = pyannote
        tensor[:, :, 2] = sortformer
        tensor[:, :, 3] = np.minimum(msdd, pyannote)
        tensor[:, :, 4] = np.minimum(msdd, sortformer)
        tensor[:, :, 5] = np.minimum(pyannote, sortformer)
        tensor[:, :, 6] = agreement
        tensor[:, :, 7] = boundary * np.maximum.reduce([msdd, pyannote, sortformer])
        tensor[:, :, 8] = exclusive
        tensor[:, :, 9] = anchor_mask.reshape(1, -1).astype(np.float32)
        tensor[:, :, 10] = msdd * float((quality_scores or {}).get("msdd", 0.0))
        tensor[:, :, 11] = pyannote * float((quality_scores or {}).get("pyannote", 0.0))
        tensor[:, :, 12] = sortformer * float((quality_scores or {}).get("sortformer", 0.0))
        return tensor

    def _global_speaker_prior(
        self,
        *,
        msdd_matrix: np.ndarray,
        pyannote_matrix: np.ndarray,
        sortformer_matrix: np.ndarray,
        feature_tensor: np.ndarray,
        anchor_mask: np.ndarray,
        msdd_weight: float,
        pyannote_weight: float,
        sortformer_weight: float,
        sortformer_saturated: bool,
        cfg: Optional[Dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        posterior_cfg = self._posterior_cfg(cfg)
        prior_cfg = posterior_cfg.get("global_prior", {}) if isinstance(posterior_cfg, dict) else {}
        prior_cfg = prior_cfg if isinstance(prior_cfg, dict) else {}
        enabled = safe_bool(prior_cfg.get("enabled", True), True)

        num_speakers = feature_tensor.shape[1] if feature_tensor.ndim == 3 else 0
        zeros = np.zeros((num_speakers,), dtype=np.float32)
        if not enabled or num_speakers <= 0:
            return zeros, zeros

        prior_scale = max(0.0, safe_float(prior_cfg.get("emission_scale", 0.18), 0.18))
        agreement_bonus = max(0.0, safe_float(prior_cfg.get("agreement_bonus", 0.14), 0.14))
        exclusive_bonus = max(0.0, safe_float(prior_cfg.get("exclusive_bonus", 0.22), 0.22))
        anchor_bonus = max(0.0, safe_float(prior_cfg.get("anchor_bonus", 0.16), 0.16))
        unanchored_penalty = max(0.0, safe_float(prior_cfg.get("unanchored_penalty", 0.12), 0.12))
        sortformer_cap_penalty = max(
            0.0,
            safe_float(prior_cfg.get("sortformer_cap_penalty", 0.28), 0.28),
        )

        msdd_mass = np.sum(msdd_matrix, axis=0, dtype=np.float32)
        pyannote_mass = np.sum(pyannote_matrix, axis=0, dtype=np.float32)
        sortformer_mass = np.sum(sortformer_matrix, axis=0, dtype=np.float32)
        agreement_mass = np.sum(feature_tensor[:, :, 6], axis=0, dtype=np.float32)
        exclusive_mass = np.sum(feature_tensor[:, :, 8], axis=0, dtype=np.float32)

        score = (
            msdd_mass * float(msdd_weight)
            + pyannote_mass * float(pyannote_weight)
            + sortformer_mass * float(sortformer_weight)
            + agreement_bonus * agreement_mass
            + exclusive_bonus * exclusive_mass
            + anchor_bonus * np.asarray(anchor_mask, dtype=np.float32)
            - unanchored_penalty * (1.0 - np.asarray(anchor_mask, dtype=np.float32))
        )

        if sortformer_saturated:
            sortformer_only_mass = np.sum(
                np.clip(sortformer_matrix - np.maximum(msdd_matrix, pyannote_matrix), 0.0, 1.0),
                axis=0,
                dtype=np.float32,
            )
            score = score - sortformer_cap_penalty * sortformer_only_mass

        bias = self._normalize_prior(score) * float(prior_scale)
        return score.astype(np.float32), bias.astype(np.float32)

    @staticmethod
    def _decode_labels_python(
        emissions: np.ndarray,
        boundary_scores: np.ndarray,
        *,
        switch_penalty: float,
        stay_bonus: float,
        boundary_relief: float,
    ) -> np.ndarray:
        frames, speakers = emissions.shape
        dp = np.zeros((frames, speakers), dtype=np.float32)
        back = np.zeros((frames, speakers), dtype=np.int32)
        dp[0] = emissions[0]
        for frame_index in range(1, frames):
            boundary = float(boundary_scores[frame_index]) if frame_index < boundary_scores.shape[0] else 0.0
            switch_cost = max(0.0, float(switch_penalty) - float(boundary_relief) * boundary)
            prev_scores = dp[frame_index - 1]
            best_prev = int(np.argmax(prev_scores))
            for speaker_index in range(speakers):
                stay_score = prev_scores[speaker_index] + float(stay_bonus)
                switch_score = prev_scores[best_prev] - switch_cost
                if stay_score >= switch_score:
                    dp[frame_index, speaker_index] = emissions[frame_index, speaker_index] + stay_score
                    back[frame_index, speaker_index] = int(speaker_index)
                else:
                    dp[frame_index, speaker_index] = emissions[frame_index, speaker_index] + switch_score
                    back[frame_index, speaker_index] = int(best_prev)
        labels = np.zeros((frames,), dtype=np.int32)
        labels[-1] = int(np.argmax(dp[-1]))
        for frame_index in range(frames - 1, 0, -1):
            labels[frame_index - 1] = int(back[frame_index, labels[frame_index]])
        return labels

    @staticmethod
    def _segments_from_labels(
        labels: np.ndarray,
        speakers: List[str],
        *,
        duration: float,
        frame_hop_sec: float,
        min_turn_sec: float,
    ) -> List[Dict[str, Any]]:
        if labels.size <= 0 or not speakers:
            return []
        segments: List[Dict[str, Any]] = []
        current_label = int(labels[0])
        start_frame = 0
        for frame_index in range(1, labels.shape[0] + 1):
            label_changed = frame_index >= labels.shape[0] or int(labels[frame_index]) != current_label
            if not label_changed:
                continue
            start = float(start_frame) * frame_hop_sec
            end = min(duration, float(frame_index) * frame_hop_sec)
            if end - start >= min_turn_sec:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": str(speakers[current_label]),
                    }
                )
            if frame_index < labels.shape[0]:
                start_frame = frame_index
                current_label = int(labels[frame_index])
        return segments

    @staticmethod
    def _recover_missing_speaker_labels(
        labels: np.ndarray,
        *,
        msdd_matrix: np.ndarray,
        pyannote_matrix: np.ndarray,
        sortformer_matrix: np.ndarray,
        target_count: int,
        frame_hop_sec: float,
        min_turn_sec: float,
    ) -> np.ndarray:
        if labels.size <= 0:
            return labels
        matrices = {
            "msdd": np.asarray(msdd_matrix, dtype=np.float32),
            "pyannote": np.asarray(pyannote_matrix, dtype=np.float32),
            "sortformer": np.asarray(sortformer_matrix, dtype=np.float32),
        }
        num_speakers = max(matrix.shape[1] for matrix in matrices.values() if matrix.ndim == 2)
        if num_speakers <= 1:
            return labels

        updated = np.asarray(labels, dtype=np.int32).copy()
        present = {int(item) for item in updated.tolist()}
        if len(present) >= min(max(1, int(target_count)), num_speakers):
            return updated

        min_frames = max(1, int(math.ceil(max(min_turn_sec, frame_hop_sec) / max(frame_hop_sec, 1e-6))))
        speaker_backend_scores: List[tuple[float, int, str]] = []
        for speaker_index in range(num_speakers):
            if speaker_index in present:
                continue
            backend_name = ""
            backend_mass = 0.0
            for name, matrix in matrices.items():
                if matrix.ndim != 2 or matrix.shape[1] <= speaker_index:
                    continue
                mass = float(np.sum(matrix[:, speaker_index], dtype=np.float32))
                if mass > backend_mass:
                    backend_name = name
                    backend_mass = mass
            if backend_name and backend_mass > 0.0:
                speaker_backend_scores.append((backend_mass, speaker_index, backend_name))

        speaker_backend_scores.sort(reverse=True)
        for _mass, speaker_index, backend_name in speaker_backend_scores:
            if len(present) >= min(max(1, int(target_count)), num_speakers):
                break
            backend_matrix = matrices.get(backend_name)
            if backend_matrix is None or backend_matrix.ndim != 2 or backend_matrix.shape[1] <= speaker_index:
                continue
            speaker_track = backend_matrix[:, speaker_index]
            if speaker_track.size <= 0 or float(np.max(speaker_track)) < 0.55:
                continue
            current_support = backend_matrix[np.arange(updated.shape[0]), updated]
            takeover_mask = (
                (speaker_track >= 0.55)
                & (speaker_track >= current_support + 0.18)
            )
            takeover_mask = takeover_mask | (
                (speaker_track >= 0.75)
                & (current_support <= 0.10)
            )
            if not np.any(takeover_mask):
                continue

            start_idx: Optional[int] = None
            adopted = False
            for frame_index in range(takeover_mask.shape[0] + 1):
                active = frame_index < takeover_mask.shape[0] and bool(takeover_mask[frame_index])
                if active:
                    if start_idx is None:
                        start_idx = frame_index
                    continue
                if start_idx is None:
                    continue
                if frame_index - start_idx >= min_frames:
                    updated[start_idx:frame_index] = int(speaker_index)
                    adopted = True
                start_idx = None
            if adopted:
                present.add(int(speaker_index))
        return updated

    @staticmethod
    def _renumber_segments(
        segments: List[Dict[str, Any]],
        *,
        fallback_speakers: Optional[List[str]] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        if not segments:
            mapping = {
                str(speaker): str(idx)
                for idx, speaker in enumerate(list(fallback_speakers or []))
            }
            return [], mapping
        stats: Dict[str, tuple[float, float, int]] = {}
        for idx, item in enumerate(segments):
            speaker = str(item.get("speaker", "") or "").strip()
            start = float(item.get("start", 0.0) or 0.0)
            end = max(start, float(item.get("end", start) or start))
            duration = max(0.0, end - start)
            prev = stats.get(speaker)
            if prev is None:
                stats[speaker] = (duration, start, idx)
            else:
                stats[speaker] = (prev[0] + duration, min(prev[1], start), min(prev[2], idx))
        ordering = sorted(
            stats.items(),
            key=lambda item: (-float(item[1][0]), float(item[1][1]), int(item[1][2]), str(item[0])),
        )
        label_map = {speaker: str(order_idx) for order_idx, (speaker, _stats) in enumerate(ordering)}
        next_index = len(label_map)
        for speaker in list(fallback_speakers or []):
            raw = str(speaker or "").strip()
            if not raw or raw in label_map:
                continue
            label_map[raw] = str(next_index)
            next_index += 1
        return [
            {
                "start": float(item.get("start", 0.0) or 0.0),
                "end": max(float(item.get("start", 0.0) or 0.0), float(item.get("end", item.get("start", 0.0) or 0.0) or 0.0)),
                "speaker": label_map.get(str(item.get("speaker", "0")), "0"),
            }
            for item in segments
        ], label_map

    @staticmethod
    def _remap_track_speakers(
        tracks: List[Dict[str, Any]],
        *,
        speaker_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        remapped: List[Dict[str, Any]] = []
        for item in tracks or []:
            speaker = str(item.get("speaker", "") or "").strip()
            if not speaker:
                continue
            remapped.append(
                {
                    **item,
                    "speaker": str(speaker_map.get(speaker, speaker)),
                }
            )
        return remapped

    @staticmethod
    def _make_open_canonical_speaker_id(prefix: str, speaker: str) -> str:
        raw = str(speaker or "").strip() or "0"
        normalized = "".join(char if char.isalnum() else "_" for char in raw).strip("_") or "0"
        safe_prefix = "".join(char if char.isalnum() else "_" for char in str(prefix or "spk")).strip("_") or "spk"
        return f"{safe_prefix}_{normalized}_u"

    def _overlap_decode_cfg(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        posterior_cfg = self._posterior_cfg(cfg)
        payload = posterior_cfg.get("overlap_decode", {}) if isinstance(posterior_cfg, dict) else {}
        return payload if isinstance(payload, dict) else {}

    def _dump_cfg(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        posterior_cfg = self._posterior_cfg(cfg)
        payload = posterior_cfg.get("dump_examples", {}) if isinstance(posterior_cfg, dict) else {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _segments_from_active_mask(
        *,
        active_mask: np.ndarray,
        scores: np.ndarray,
        speakers: List[str],
        duration: float,
        frame_hop_sec: float,
        min_segment_sec: float,
    ) -> List[Dict[str, Any]]:
        if active_mask.ndim != 2 or active_mask.shape[1] != len(speakers):
            return []
        tracks: List[Dict[str, Any]] = []
        num_frames = active_mask.shape[0]
        for speaker_index, speaker in enumerate(speakers):
            start_idx: Optional[int] = None
            conf_values: List[float] = []
            for frame_index in range(num_frames + 1):
                is_on = frame_index < num_frames and bool(active_mask[frame_index, speaker_index])
                if is_on:
                    if start_idx is None:
                        start_idx = frame_index
                        conf_values = []
                    conf_values.append(float(scores[frame_index, speaker_index]))
                    continue
                if start_idx is None:
                    continue
                start = float(start_idx) * frame_hop_sec
                end = min(duration, float(frame_index) * frame_hop_sec)
                if end - start >= min_segment_sec:
                    tracks.append(
                        {
                            "start": start,
                            "end": end,
                            "speaker": str(speaker),
                            "confidence": float(np.mean(conf_values)) if conf_values else 0.0,
                        }
                    )
                start_idx = None
                conf_values = []
        return tracks

    @staticmethod
    def _regions_from_frame_mask(
        *,
        frame_mask: np.ndarray,
        duration: float,
        frame_hop_sec: float,
        min_region_sec: float,
        merge_gap_sec: float,
    ) -> List[Dict[str, float]]:
        mask = np.asarray(frame_mask, dtype=bool).reshape(-1)
        if mask.size <= 0:
            return []
        raw_regions: List[Dict[str, float]] = []
        start_idx: Optional[int] = None
        for frame_index in range(mask.shape[0] + 1):
            is_on = frame_index < mask.shape[0] and bool(mask[frame_index])
            if is_on:
                if start_idx is None:
                    start_idx = frame_index
                continue
            if start_idx is None:
                continue
            raw_regions.append(
                {
                    "start": float(start_idx) * frame_hop_sec,
                    "end": min(duration, float(frame_index) * frame_hop_sec),
                }
            )
            start_idx = None
        return merge_time_regions(
            raw_regions,
            max_gap_sec=max(0.0, merge_gap_sec),
            min_duration_sec=max(0.05, min_region_sec),
            max_end_sec=max(0.0, duration),
        )

    def _decode_overlap_tracks(
        self,
        *,
        activity_probs: np.ndarray,
        primary_labels: np.ndarray,
        feature_tensor: np.ndarray,
        speakers: List[str],
        overlap_mask: np.ndarray,
        duration: float,
        frame_hop_sec: float,
        cfg: Optional[Dict[str, Any]],
        base_activity_threshold: float,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        overlap_cfg = self._overlap_decode_cfg(cfg)
        if not safe_bool(overlap_cfg.get("enabled", True), True):
            return [], []
        if activity_probs.ndim != 2 or activity_probs.shape[0] <= 0 or activity_probs.shape[1] <= 1:
            return [], []

        primary_min = max(
            0.15,
            safe_float(overlap_cfg.get("primary_min_prob", base_activity_threshold + 0.02), base_activity_threshold + 0.02),
        )
        secondary_min = max(
            0.10,
            safe_float(overlap_cfg.get("secondary_min_prob", max(0.18, base_activity_threshold - 0.08)), max(0.18, base_activity_threshold - 0.08)),
        )
        frame_overlap_gate = max(0.0, min(1.0, safe_float(overlap_cfg.get("frame_overlap_gate", 0.34), 0.34)))
        agreement_gate = max(0.0, min(1.0, safe_float(overlap_cfg.get("agreement_gate", 0.34), 0.34)))
        seed_bonus = max(0.0, safe_float(overlap_cfg.get("seed_bonus", 0.08), 0.08))
        max_active = max(2, safe_int(overlap_cfg.get("max_active_speakers", 2), 2))
        min_overlap_sec = max(0.08, safe_float(overlap_cfg.get("min_overlap_sec", max(frame_hop_sec * 2.0, 0.16)), max(frame_hop_sec * 2.0, 0.16)))
        merge_gap_sec = max(0.0, safe_float(overlap_cfg.get("merge_gap_sec", max(frame_hop_sec * 0.5, 0.08)), max(frame_hop_sec * 0.5, 0.08)))

        active_mask = np.zeros_like(activity_probs, dtype=np.bool_)
        for frame_index in range(activity_probs.shape[0]):
            primary_idx = int(primary_labels[min(frame_index, primary_labels.shape[0] - 1)])
            ranked = np.argsort(activity_probs[frame_index])[::-1]
            selected: List[int] = []
            if 0 <= primary_idx < activity_probs.shape[1]:
                selected.append(primary_idx)
            frame_seed = float(overlap_mask[min(frame_index, overlap_mask.shape[0] - 1)]) >= frame_overlap_gate
            for speaker_index in ranked.tolist():
                if speaker_index in selected:
                    continue
                if len(selected) >= max_active:
                    break
                prob = float(activity_probs[frame_index, speaker_index])
                boosted = prob + seed_bonus * float(overlap_mask[min(frame_index, overlap_mask.shape[0] - 1)])
                agreement = float(feature_tensor[frame_index, speaker_index, 6]) if feature_tensor.ndim == 3 else 0.0
                if frame_seed:
                    if boosted >= secondary_min:
                        selected.append(speaker_index)
                elif boosted >= max(primary_min, secondary_min + 0.06) and agreement >= agreement_gate:
                    selected.append(speaker_index)
            if len(selected) <= 1:
                continue
            for speaker_index in selected:
                active_mask[frame_index, speaker_index] = True

        overlap_tracks = self._segments_from_active_mask(
            active_mask=active_mask,
            scores=activity_probs,
            speakers=speakers,
            duration=duration,
            frame_hop_sec=frame_hop_sec,
            min_segment_sec=min_overlap_sec,
        )
        overlap_regions = self._regions_from_frame_mask(
            frame_mask=np.sum(active_mask.astype(np.int32), axis=1) > 1,
            duration=duration,
            frame_hop_sec=frame_hop_sec,
            min_region_sec=min_overlap_sec,
            merge_gap_sec=merge_gap_sec,
        )
        return overlap_tracks, overlap_regions

    @staticmethod
    def _project_canonical_segments(
        diar_segments: List[Dict[str, Any]],
        *,
        canonical_map: Optional[Dict[str, str]],
        kept_speakers: List[str],
    ) -> List[Dict[str, Any]]:
        kept = {str(item) for item in kept_speakers}
        projected: List[Dict[str, Any]] = []
        for item in diar_segments:
            raw = str(item.get("speaker", "") or "").strip()
            canonical = str((canonical_map or {}).get(raw, raw)).strip()
            if canonical not in kept:
                continue
            projected.append(
                {
                    "start": float(item.get("start", 0.0) or 0.0),
                    "end": max(float(item.get("start", 0.0) or 0.0), float(item.get("end", item.get("start", 0.0) or 0.0) or 0.0)),
                    "speaker": canonical,
                }
            )
        return normalize_diar_segments_for_fusion(
            projected,
            min_turn_sec=0.02,
            merge_gap_sec=0.0,
            normalize_ids=False,
        )

    def _maybe_dump_example(
        self,
        *,
        file_name: str,
        duration: float,
        frame_hop_sec: float,
        speakers: List[str],
        feature_tensor: np.ndarray,
        boundary_scores: np.ndarray,
        overlap_mask: np.ndarray,
        primary_labels: np.ndarray,
        activity_probs: np.ndarray,
        msdd_segments: List[Dict[str, Any]],
        pyannote_segments: List[Dict[str, Any]],
        sortformer_segments: List[Dict[str, Any]],
        overlap_tracks: List[Dict[str, Any]],
        route: str,
        cfg: Optional[Dict[str, Any]],
        output_root: Optional[Path] = None,
    ) -> str:
        dump_cfg = self._dump_cfg(cfg)
        if not safe_bool(dump_cfg.get("enabled", False), False):
            return ""
        output_dir = resolve_example_dump_dir(
            str(dump_cfg.get("output_dir", "output_files/posterior_fusion_examples") or ""),
            root_dir=APP_ROOT,
            output_root=output_root,
        )
        try:
            meta_path = dump_decoder_example(
                output_dir=output_dir,
                source_name=file_name or "audio",
                duration=duration,
                frame_hop_sec=frame_hop_sec,
                speakers=speakers,
                feature_tensor=feature_tensor,
                boundary_scores=boundary_scores,
                overlap_mask=overlap_mask,
                primary_labels=primary_labels,
                activity_probs=activity_probs,
                msdd_segments=msdd_segments,
                pyannote_segments=pyannote_segments,
                sortformer_segments=sortformer_segments,
                overlap_tracks=overlap_tracks,
                route=route,
            )
            return str(meta_path)
        except Exception as exc:
            self.logger.warning("  Posterior fusion example dump failed%s: %s", f" ({file_name})" if file_name else "", exc)
            return ""

    def decode(
        self,
        *,
        audio_duration: float,
        candidates: Dict[str, List[Dict[str, Any]]],
        cfg: Optional[Dict[str, Any]] = None,
        file_name: str = "",
        output_root: Optional[Path] = None,
    ) -> PosteriorFusionDecodeResult:
        hybrid_cfg = self._hybrid_cfg(cfg)
        posterior_cfg = self._posterior_cfg(cfg)
        if not safe_bool(hybrid_cfg.get("enabled", True), True):
            return PosteriorFusionDecodeResult(segments=[], overlap_seed_regions=[], route="", target_count=0)
        if not safe_bool(posterior_cfg.get("enabled", True), True):
            return PosteriorFusionDecodeResult(segments=[], overlap_seed_regions=[], route="", target_count=0)

        root_cfg = cfg if isinstance(cfg, dict) else {}
        sf_cfg = root_cfg.get("sortformer", {}) if isinstance(root_cfg.get("sortformer", {}), dict) else {}
        py_cfg = root_cfg.get("pyannote_fallback", {}) if isinstance(root_cfg.get("pyannote_fallback", {}), dict) else {}
        overlap_cfg = root_cfg.get("overlap_handling", {}) if isinstance(root_cfg.get("overlap_handling", {}), dict) else {}

        msdd_segments = normalize_diar_segments_for_fusion(
            list(candidates.get("msdd") or []),
            min_turn_sec=0.05,
            merge_gap_sec=self._backend_merge_gap(root_cfg, "sortformer", 0.08),
        )
        pyannote_segments = normalize_diar_segments_for_fusion(
            list(candidates.get("pyannote") or []),
            min_turn_sec=0.05,
            merge_gap_sec=safe_float(py_cfg.get("merge_gap_sec", 0.08), 0.08),
        )
        sortformer_segments = normalize_diar_segments_for_fusion(
            list(candidates.get("sortformer") or []),
            min_turn_sec=0.05,
            merge_gap_sec=safe_float(sf_cfg.get("merge_gap_sec", 0.08), 0.08),
        )
        if not msdd_segments:
            return PosteriorFusionDecodeResult(segments=[], overlap_seed_regions=[], route="", target_count=0)

        duration = max(0.0, float(audio_duration))
        if duration <= 0.0:
            duration = max(float(msdd_segments[-1].get("end", 0.0) or 0.0), 0.0)
        frame_hop_sec = max(0.04, safe_float(posterior_cfg.get("frame_hop_sec", 0.08), 0.08))
        min_turn_sec = max(
            0.05,
            safe_float(posterior_cfg.get("min_turn_sec", hybrid_cfg.get("min_piece_sec", 0.20)), 0.20),
        )
        snap_sec = max(0.04, safe_float(hybrid_cfg.get("boundary_snap_sec", 0.18), 0.18))
        num_frames = max(1, int(math.ceil(max(duration, frame_hop_sec) / frame_hop_sec)))
        requested_num_speakers = self._requested_num_speakers(cfg)
        max_speakers = max(
            1,
            safe_int(root_cfg.get("max_speakers", 8), 8),
            requested_num_speakers,
        )

        py_weight = max(0.1, safe_float(hybrid_cfg.get("pyannote_weight", 1.25), 1.25))
        msdd_weight = max(0.1, safe_float(hybrid_cfg.get("msdd_weight", 1.0), 1.0))
        sf_weight = max(0.05, safe_float(hybrid_cfg.get("sortformer_weight", 0.55), 0.55))
        sortformer_count = count_diar_speakers(sortformer_segments)

        map_overlap_ratio = max(0.1, min(0.99, safe_float(hybrid_cfg.get("map_min_overlap_ratio", 0.62), 0.62)))
        map_overlap_sec = max(0.25, safe_float(hybrid_cfg.get("map_min_overlap_sec", 1.2), 1.2))
        map_margin_ratio = max(1.0, safe_float(hybrid_cfg.get("map_min_margin_ratio", 1.25), 1.25))

        canonical_speakers: List[str] = []
        if pyannote_segments:
            canonical_speakers = sorted({str(item.get("speaker", "0")) for item in pyannote_segments})
            msdd_map = build_strong_reference_speaker_map(
                msdd_segments,
                pyannote_segments,
                min_overlap_ratio=map_overlap_ratio,
                min_overlap_sec=map_overlap_sec,
                min_margin_ratio=map_margin_ratio,
            )
            sortformer_map = build_strong_reference_speaker_map(
                sortformer_segments,
                pyannote_segments,
                min_overlap_ratio=min(0.99, map_overlap_ratio + 0.08),
                min_overlap_sec=map_overlap_sec,
                min_margin_ratio=map_margin_ratio,
            )
            route_parts = ["NeMo-MSDD", "community-1"]
        else:
            canonical_speakers = sorted({str(item.get("speaker", "0")) for item in msdd_segments})
            msdd_map = {str(item): str(item) for item in canonical_speakers}
            sortformer_map = build_strong_reference_speaker_map(
                sortformer_segments,
                msdd_segments,
                min_overlap_ratio=map_overlap_ratio,
                min_overlap_sec=map_overlap_sec,
                min_margin_ratio=map_margin_ratio,
            )
            route_parts = ["NeMo-MSDD"]
        if sortformer_segments:
            route_parts.append("Sortformer")

        seen_canonical = {str(item) for item in canonical_speakers}
        for backend_name, source_segments, canonical_map in (
            ("msdd", msdd_segments, msdd_map),
            ("sortformer", sortformer_segments, sortformer_map),
        ):
            raw_speakers = sorted(
                {
                    str(item.get("speaker", "") or "").strip()
                    for item in source_segments
                    if str(item.get("speaker", "") or "").strip()
                }
            )
            for raw_speaker in raw_speakers:
                canonical = str(canonical_map.get(raw_speaker, "") or "").strip()
                if canonical:
                    seen_canonical.add(canonical)
                    continue
                unique_id = self._make_open_canonical_speaker_id(backend_name, raw_speaker)
                suffix = 2
                while unique_id in seen_canonical:
                    unique_id = f"{self._make_open_canonical_speaker_id(backend_name, raw_speaker)}_{suffix}"
                    suffix += 1
                canonical_map[raw_speaker] = unique_id
                canonical_speakers.append(unique_id)
                seen_canonical.add(unique_id)

        speaker_to_index = {speaker: idx for idx, speaker in enumerate(canonical_speakers)}
        if not speaker_to_index:
            return PosteriorFusionDecodeResult(segments=[], overlap_seed_regions=[], route="", target_count=0)

        msdd_matrix = self._segments_to_frame_matrix(
            diar_segments=msdd_segments,
            speaker_to_index=speaker_to_index,
            num_frames=num_frames,
            frame_hop_sec=frame_hop_sec,
            canonical_map=msdd_map,
            allow_unmapped=False,
        )
        pyannote_matrix = self._segments_to_frame_matrix(
            diar_segments=pyannote_segments,
            speaker_to_index=speaker_to_index,
            num_frames=num_frames,
            frame_hop_sec=frame_hop_sec,
            allow_unmapped=False,
        )
        sortformer_matrix = self._segments_to_frame_matrix(
            diar_segments=sortformer_segments,
            speaker_to_index=speaker_to_index,
            num_frames=num_frames,
            frame_hop_sec=frame_hop_sec,
            canonical_map=sortformer_map,
            allow_unmapped=False,
        )

        backend_quality_scores = self._backend_quality_scores(
            msdd_matrix=msdd_matrix,
            pyannote_matrix=pyannote_matrix,
            sortformer_matrix=sortformer_matrix,
            msdd_segments=msdd_segments,
            pyannote_segments=pyannote_segments,
            sortformer_segments=sortformer_segments,
            audio_duration=duration,
        )
        msdd_count = count_diar_speakers(msdd_segments)
        pyannote_count = count_diar_speakers(pyannote_segments)
        if requested_num_speakers > 0:
            msdd_scale = self._manual_count_weight_scale(
                "msdd",
                backend_count=msdd_count,
                requested_num_speakers=requested_num_speakers,
            )
            pyannote_scale = self._manual_count_weight_scale(
                "pyannote",
                backend_count=pyannote_count,
                requested_num_speakers=requested_num_speakers,
            )
            sortformer_scale = self._manual_count_weight_scale(
                "sortformer",
                backend_count=sortformer_count,
                requested_num_speakers=requested_num_speakers,
            )
            msdd_weight *= msdd_scale
            py_weight *= pyannote_scale
            sf_weight *= sortformer_scale
            self.logger.info(
                "  Posterior fusion manual speaker count lock: requested=%d, counts(msdd=%d, pyannote=%d, sortformer=%d), weight_scales(msdd=%.2f, pyannote=%.2f, sortformer=%.2f)",
                requested_num_speakers,
                msdd_count,
                pyannote_count,
                sortformer_count,
                float(msdd_scale),
                float(pyannote_scale),
                float(sortformer_scale),
            )
        target_count, count_decision = self._choose_target_speaker_count(
            msdd_count=msdd_count,
            pyannote_count=pyannote_count,
            sortformer_count=sortformer_count,
            sortformer_turn_count=len(sortformer_segments),
            audio_duration=duration,
            max_speakers=max_speakers,
            quality_scores=backend_quality_scores,
            msdd_weight=msdd_weight,
            pyannote_weight=py_weight,
            sortformer_weight=sf_weight,
            cfg=cfg,
        )
        count_score_text = ", ".join(
            f"{count}:{float(meta.get('score', 0.0)):.2f}"
            for count, meta in sorted((count_decision.get("candidate_meta", {}) or {}).items())
        )
        self.logger.info(
            "  Posterior fusion count decision: msdd=%d, pyannote=%d, sortformer=%d, heuristic=%d, mean=%.2f, scores=[%s] -> target=%d",
            msdd_count,
            pyannote_count,
            sortformer_count,
            int(count_decision.get("heuristic_target", target_count)),
            float(count_decision.get("weighted_mean", float(target_count))),
            count_score_text,
            int(target_count),
        )
        sortformer_limit = min(4, max(1, max_speakers))
        sortformer_saturated = bool(
            sortformer_segments
            and sortformer_count >= sortformer_limit
            and int(target_count) > sortformer_limit
        )
        if sortformer_saturated:
            sf_saturation_scale = max(
                0.1,
                min(
                    1.0,
                    safe_float(posterior_cfg.get("sortformer_saturation_scale", 0.65), 0.65),
                ),
            )
            sf_weight *= sf_saturation_scale
            support_mask = np.maximum(msdd_matrix, pyannote_matrix).astype(np.float32)
            sortformer_matrix = sortformer_matrix * np.maximum(support_mask, sf_saturation_scale)
        overlap_seed_regions = build_overlap_seed_regions(
            msdd_segments=msdd_segments,
            pyannote_segments=pyannote_segments,
            sortformer_segments=sortformer_segments,
            hybrid_cfg=hybrid_cfg,
            overlap_cfg=overlap_cfg,
        )
        overlap_mask = self._segments_to_frame_matrix(
            diar_segments=[dict(item, speaker="ovl") for item in overlap_seed_regions],
            speaker_to_index={"ovl": 0},
            num_frames=num_frames,
            frame_hop_sec=frame_hop_sec,
            allow_unmapped=True,
        )[:, 0]
        boundary_scores = self._build_boundary_scores(
            duration=duration,
            frame_hop_sec=frame_hop_sec,
            snap_sec=snap_sec,
            msdd_segments=msdd_segments,
            pyannote_segments=pyannote_segments,
            sortformer_segments=sortformer_segments,
            msdd_weight=msdd_weight,
            pyannote_weight=py_weight,
            sortformer_weight=sf_weight,
        )
        anchor_mask = np.maximum(
            (msdd_matrix > 0.01).any(axis=0),
            (pyannote_matrix > 0.01).any(axis=0),
        ).astype(np.float32)
        sortformer_anchor = (sortformer_matrix > 0.20).any(axis=0).astype(np.float32)
        anchor_mask = np.maximum(anchor_mask, sortformer_anchor * 0.5).astype(np.float32)
        feature_tensor = self._feature_tensor(
            msdd_matrix=msdd_matrix,
            pyannote_matrix=pyannote_matrix,
            sortformer_matrix=sortformer_matrix,
            boundary_scores=boundary_scores,
            overlap_mask=overlap_mask,
            anchor_mask=anchor_mask,
            quality_scores=backend_quality_scores,
        )
        self.logger.info(
            "  Posterior fusion backend quality: msdd=%.2f, pyannote=%.2f, sortformer=%.2f",
            float(backend_quality_scores.get("msdd", 0.0)),
            float(backend_quality_scores.get("pyannote", 0.0)),
            float(backend_quality_scores.get("sortformer", 0.0)),
        )

        calibrator = self._calibrator(cfg, output_root=output_root)
        emissions = calibrator.emission_scores(feature_tensor)
        speaker_scores, speaker_prior_bias = self._global_speaker_prior(
            msdd_matrix=msdd_matrix,
            pyannote_matrix=pyannote_matrix,
            sortformer_matrix=sortformer_matrix,
            feature_tensor=feature_tensor,
            anchor_mask=anchor_mask,
            msdd_weight=msdd_weight,
            pyannote_weight=py_weight,
            sortformer_weight=sf_weight,
            sortformer_saturated=sortformer_saturated,
            cfg=cfg,
        )
        keep_count = min(max(1, target_count), len(canonical_speakers))
        keep_indices_ranked = np.argsort(speaker_scores)[::-1][:keep_count]
        keep_indices = np.asarray(keep_indices_ranked, dtype=np.int32)
        keep_indices = np.sort(keep_indices.astype(np.int32))
        kept_feature_tensor = feature_tensor[:, keep_indices, :]
        kept_msdd = msdd_matrix[:, keep_indices]
        kept_pyannote = pyannote_matrix[:, keep_indices]
        kept_sortformer = sortformer_matrix[:, keep_indices]
        emissions = emissions[:, keep_indices] + speaker_prior_bias[keep_indices].reshape(1, -1)
        kept_speakers = [canonical_speakers[int(idx)] for idx in keep_indices.tolist()]
        kept_speakers_ranked = [canonical_speakers[int(idx)] for idx in keep_indices_ranked.tolist()]
        kept_boundary_scores = np.ascontiguousarray(boundary_scores, dtype=np.float32)
        activity_probs = calibrator.activity_probabilities(kept_feature_tensor)

        switch_penalty = max(0.05, safe_float(posterior_cfg.get("switch_penalty", 0.42), 0.42))
        stay_bonus = max(0.0, safe_float(posterior_cfg.get("stay_bonus", 0.16), 0.16))
        boundary_relief = max(0.0, safe_float(posterior_cfg.get("boundary_relief", 0.28), 0.28))
        use_native = safe_bool(posterior_cfg.get("use_native_viterbi", True), True)
        labels = None
        native_backend_used = False
        if use_native:
            backend = self._native_backend()
            if backend is not None:
                labels = backend.decode_labels(
                    np.ascontiguousarray(emissions, dtype=np.float32),
                    kept_boundary_scores,
                    switch_penalty=switch_penalty,
                    stay_bonus=stay_bonus,
                    boundary_relief=boundary_relief,
                )
                native_backend_used = labels is not None
        if labels is None:
            labels = self._decode_labels_python(
                np.ascontiguousarray(emissions, dtype=np.float32),
                kept_boundary_scores,
                switch_penalty=switch_penalty,
                stay_bonus=stay_bonus,
                boundary_relief=boundary_relief,
            )
        labels = self._recover_missing_speaker_labels(
            labels,
            msdd_matrix=kept_msdd,
            pyannote_matrix=kept_pyannote,
            sortformer_matrix=kept_sortformer,
            target_count=keep_count,
            frame_hop_sec=frame_hop_sec,
            min_turn_sec=min_turn_sec,
        )

        primary_segments = self._segments_from_labels(
            labels,
            kept_speakers,
            duration=duration,
            frame_hop_sec=frame_hop_sec,
            min_turn_sec=min_turn_sec,
        )
        primary_segments = merge_adjacent_speaker_turns(
            primary_segments,
            merge_gap_sec=max(frame_hop_sec * 0.5, 0.04),
        )
        overlap_tracks, posterior_overlap_regions = self._decode_overlap_tracks(
            activity_probs=activity_probs,
            primary_labels=labels,
            feature_tensor=kept_feature_tensor,
            speakers=kept_speakers,
            overlap_mask=overlap_mask,
            duration=duration,
            frame_hop_sec=frame_hop_sec,
            cfg=cfg,
            base_activity_threshold=float(calibrator.activity_threshold),
        )
        segments, speaker_label_map = self._renumber_segments(
            primary_segments,
            fallback_speakers=kept_speakers_ranked,
        )
        overlap_tracks = self._remap_track_speakers(
            overlap_tracks,
            speaker_map=speaker_label_map,
        )
        if posterior_overlap_regions:
            overlap_merge_gap = max(
                frame_hop_sec * 0.5,
                safe_float(
                    (self._overlap_decode_cfg(cfg).get("merge_gap_sec", 0.08)),
                    0.08,
                ),
            )
            overlap_seed_regions = merge_time_regions(
                list(overlap_seed_regions) + list(posterior_overlap_regions),
                max_gap_sec=overlap_merge_gap,
                min_duration_sec=max(0.08, frame_hop_sec * 2.0),
                max_end_sec=max(0.0, duration),
            )

        positive_counts = [
            count for count in (count_diar_speakers(msdd_segments), count_diar_speakers(pyannote_segments), sortformer_count)
            if count > 0
        ]
        count_spread = (max(positive_counts) - min(positive_counts)) if positive_counts else 0
        allow_online_update = (
            count_spread <= 1
            and count_diar_speakers(primary_segments) >= min(keep_count, max(positive_counts or [keep_count]))
        )
        pseudo = calibrator.build_pseudo_labels(
            msdd=kept_msdd,
            pyannote=kept_pyannote,
            sortformer=kept_sortformer,
        )
        updated = 0
        if allow_online_update and pseudo["indices"].size > 0:
            pseudo_features = kept_feature_tensor[pseudo["indices"]]
            updated = calibrator.update(pseudo_features, pseudo["targets"])
        pseudo_activity = calibrator.build_pseudo_activity_targets(
            msdd=kept_msdd,
            pyannote=kept_pyannote,
            sortformer=kept_sortformer,
        )
        if allow_online_update and pseudo_activity["indices"].size > 0:
            pseudo_activity_features = kept_feature_tensor[pseudo_activity["indices"]]
            updated += calibrator.update_activity(
                pseudo_activity_features,
                pseudo_activity["targets"],
                mask=pseudo_activity["mask"],
            )
        if allow_online_update and updated > 0:
            calibrator.save_state()
        elif not allow_online_update:
            updated = 0

        route = f"PosteriorFusion[{'+'.join(route_parts)}]"
        dump_path = self._maybe_dump_example(
            file_name=file_name,
            duration=duration,
            frame_hop_sec=frame_hop_sec,
            speakers=kept_speakers,
            feature_tensor=kept_feature_tensor,
            boundary_scores=kept_boundary_scores,
            overlap_mask=overlap_mask,
            primary_labels=labels,
            activity_probs=activity_probs,
            msdd_segments=self._project_canonical_segments(msdd_segments, canonical_map=msdd_map, kept_speakers=kept_speakers),
            pyannote_segments=self._project_canonical_segments(pyannote_segments, canonical_map=None, kept_speakers=kept_speakers),
            sortformer_segments=self._project_canonical_segments(sortformer_segments, canonical_map=sortformer_map, kept_speakers=kept_speakers),
            overlap_tracks=overlap_tracks,
            route=route,
            cfg=cfg,
            output_root=output_root,
        )

        if segments:
            self.logger.info(
                "  Posterior fusion decode complete: speakers=%d (target=%d), turns=%d, overlap_tracks=%d, pseudo_updates=%d, viterbi=%s%s",
                count_diar_speakers(segments),
                int(target_count),
                len(segments),
                len(overlap_tracks),
                int(updated),
                "native" if native_backend_used else "python",
                f" ({file_name})" if file_name else "",
            )

        return PosteriorFusionDecodeResult(
            segments=segments,
            overlap_seed_regions=overlap_seed_regions,
            route=route,
            target_count=int(target_count),
            overlap_tracks=overlap_tracks,
            calibrator_updates=int(updated),
            example_dump_path=str(dump_path or ""),
        )
