from __future__ import annotations

import hashlib
import json
import copy
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from output_layout import resolve_runtime_artifact_path

from .calibration import PosteriorFusionCalibrator
from .utils import count_diar_speakers, diar_speaker_overlap_matrix, normalize_diar_segments_for_fusion, safe_float, safe_int


def resolve_example_dump_dir(raw_path: str, *, root_dir: Path, output_root: Optional[Path] = None) -> Path:
    path = Path(str(raw_path or "").strip() or "output_files/posterior_fusion_examples")
    if not path.is_absolute():
        normalized = str(raw_path or "").strip().replace("\\", "/")
        if output_root is not None and normalized in {
            "",
            "posterior_fusion_examples",
            "output_files/posterior_fusion_examples",
            "_runtime_artifacts/posterior_fusion_examples",
        }:
            path = resolve_runtime_artifact_path(output_root, "posterior_fusion_examples")
        else:
            path = root_dir / path
    return path


def _stable_example_id(source_name: str, duration: float) -> str:
    raw = f"{source_name}|{duration:.3f}".encode("utf-8", errors="replace")
    digest = hashlib.sha1(raw).hexdigest()[:10]
    stem = Path(source_name or "audio").stem[:48] or "audio"
    return f"{stem}_{digest}"


def dump_decoder_example(
    *,
    output_dir: Path,
    source_name: str,
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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    example_id = _stable_example_id(source_name, duration)
    npz_path = output_dir / f"{example_id}.npz"
    meta_path = output_dir / f"{example_id}.json"

    np.savez_compressed(
        npz_path,
        feature_tensor=np.asarray(feature_tensor, dtype=np.float32),
        boundary_scores=np.asarray(boundary_scores, dtype=np.float32),
        overlap_mask=np.asarray(overlap_mask, dtype=np.float32),
        primary_labels=np.asarray(primary_labels, dtype=np.int32),
        activity_probs=np.asarray(activity_probs, dtype=np.float32),
    )
    metadata = {
        "example_id": example_id,
        "source_name": str(source_name or ""),
        "duration": float(duration),
        "frame_hop_sec": float(frame_hop_sec),
        "speakers": [str(item) for item in speakers],
        "route": str(route or ""),
        "msdd_segments": list(msdd_segments or []),
        "pyannote_segments": list(pyannote_segments or []),
        "sortformer_segments": list(sortformer_segments or []),
        "overlap_tracks": list(overlap_tracks or []),
        "npz_path": str(npz_path),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), "utf-8")
    return meta_path


def load_rttm_segments(rttm_path: Path) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if not rttm_path.exists():
        return segments
    with rttm_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = str(line or "").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            try:
                start = max(0.0, float(parts[3]))
                dur = max(0.0, float(parts[4]))
            except Exception:
                continue
            speaker = str(parts[7] or "").strip()
            if not speaker or dur <= 0.0:
                continue
            segments.append(
                {
                    "start": start,
                    "end": start + dur,
                    "speaker": speaker,
                }
            )
    return normalize_diar_segments_for_fusion(segments, min_turn_sec=0.02, merge_gap_sec=0.0, normalize_ids=False)


def resolve_reference_rttm(
    *,
    metadata: Dict[str, Any],
    rttm_dir: Path,
) -> Optional[Path]:
    source_name = str(metadata.get("source_name", "") or "").strip()
    if not source_name:
        return None
    stem = Path(source_name).stem
    exact = rttm_dir / f"{stem}.rttm"
    if exact.exists():
        return exact
    candidates = sorted(rttm_dir.glob(f"{stem}*.rttm"))
    return candidates[0] if candidates else None


def _segment_total_duration(diar_segments: List[Dict[str, Any]]) -> float:
    total = 0.0
    for item in diar_segments or []:
        start = max(0.0, float(item.get("start", 0.0) or 0.0))
        end = max(start, float(item.get("end", start) or start))
        total += max(0.0, end - start)
    return total


def _canonical_alignment_score(
    *,
    reference_segments: List[Dict[str, Any]],
    candidate_segments: List[Dict[str, Any]],
    mapping: Dict[str, str],
) -> float:
    if not reference_segments or not candidate_segments or not mapping:
        return -1.0
    overlap_matrix = diar_speaker_overlap_matrix(reference_segments, candidate_segments)
    matched_overlap = 0.0
    for reference_speaker, canonical_speaker in mapping.items():
        matched_overlap += float((overlap_matrix.get(str(reference_speaker), {}) or {}).get(str(canonical_speaker), 0.0))
    reference_total = max(1e-6, _segment_total_duration(reference_segments))
    candidate_total = max(1e-6, _segment_total_duration(candidate_segments))
    coverage = matched_overlap / reference_total
    purity = matched_overlap / candidate_total
    mapped_ratio = float(len(mapping)) / max(1.0, float(count_diar_speakers(reference_segments)))
    count_gap = abs(count_diar_speakers(candidate_segments) - count_diar_speakers(reference_segments))
    return (
        0.52 * coverage
        + 0.24 * purity
        + 0.24 * mapped_ratio
        - 0.04 * float(count_gap)
    )


def select_canonical_segments(
    metadata: Dict[str, Any],
    *,
    reference_segments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    candidates: List[tuple[str, List[Dict[str, Any]]]] = []
    for key in ("pyannote_segments", "msdd_segments", "sortformer_segments"):
        segments = metadata.get(key)
        if isinstance(segments, list) and segments:
            candidates.append(
                (
                    key,
                    normalize_diar_segments_for_fusion(
                        segments,
                        min_turn_sec=0.02,
                        merge_gap_sec=0.0,
                        normalize_ids=False,
                    ),
                )
            )
    if not candidates:
        return []
    if not reference_segments:
        return candidates[0][1]

    best_segments = candidates[0][1]
    best_score = -1.0
    for _key, candidate_segments in candidates:
        mapping = align_reference_speakers(
            reference_segments=reference_segments,
            canonical_segments=candidate_segments,
        )
        score = _canonical_alignment_score(
            reference_segments=reference_segments,
            candidate_segments=candidate_segments,
            mapping=mapping,
        )
        if score > best_score:
            best_score = score
            best_segments = candidate_segments
    return best_segments


def align_reference_speakers(
    *,
    reference_segments: List[Dict[str, Any]],
    canonical_segments: List[Dict[str, Any]],
) -> Dict[str, str]:
    if not reference_segments or not canonical_segments:
        return {}
    matrix = diar_speaker_overlap_matrix(reference_segments, canonical_segments)
    mapping: Dict[str, str] = {}
    used_canonical: set[str] = set()
    for ref_speaker, overlaps in sorted(matrix.items()):
        ranked = sorted(overlaps.items(), key=lambda item: (float(item[1]), str(item[0])), reverse=True)
        for canonical_speaker, overlap in ranked:
            if overlap <= 0.0 or canonical_speaker in used_canonical:
                continue
            mapping[str(ref_speaker)] = str(canonical_speaker)
            used_canonical.add(str(canonical_speaker))
            break
    return mapping


def _segments_to_frame_targets(
    *,
    diar_segments: List[Dict[str, Any]],
    speaker_to_index: Dict[str, int],
    num_frames: int,
    frame_hop_sec: float,
    speaker_map: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    matrix = np.zeros((num_frames, len(speaker_to_index)), dtype=np.float32)
    if not diar_segments or not speaker_to_index:
        return matrix
    for item in diar_segments:
        raw_speaker = str(item.get("speaker", "") or "").strip()
        mapped = str((speaker_map or {}).get(raw_speaker, raw_speaker)).strip()
        if mapped not in speaker_to_index:
            continue
        start = max(0.0, float(item.get("start", 0.0) or 0.0))
        end = max(start, float(item.get("end", start) or start))
        if end <= start:
            continue
        start_idx = max(0, int(np.floor(start / max(frame_hop_sec, 1e-6))))
        end_idx = min(num_frames - 1, int(np.ceil(end / max(frame_hop_sec, 1e-6))))
        for frame_index in range(start_idx, end_idx + 1):
            win_start = float(frame_index) * frame_hop_sec
            win_end = win_start + frame_hop_sec
            overlap = max(0.0, min(end, win_end) - max(start, win_start))
            if overlap <= 0.0:
                continue
            matrix[frame_index, speaker_to_index[mapped]] = max(
                matrix[frame_index, speaker_to_index[mapped]],
                min(1.0, overlap / max(frame_hop_sec, 1e-6)),
            )
    return matrix


def build_supervision_from_reference(
    *,
    metadata: Dict[str, Any],
    feature_tensor: np.ndarray,
    frame_hop_sec: float,
    reference_segments: List[Dict[str, Any]],
) -> Optional[Dict[str, np.ndarray]]:
    speakers = [str(item) for item in list(metadata.get("speakers") or [])]
    if feature_tensor.ndim != 3 or feature_tensor.shape[1] != len(speakers) or not speakers:
        return None

    canonical_segments = select_canonical_segments(
        metadata,
        reference_segments=reference_segments,
    )
    mapping = align_reference_speakers(
        reference_segments=reference_segments,
        canonical_segments=canonical_segments,
    )
    if not mapping:
        return None

    speaker_to_index = {speaker: idx for idx, speaker in enumerate(speakers)}
    ref_matrix = _segments_to_frame_targets(
        diar_segments=reference_segments,
        speaker_to_index=speaker_to_index,
        num_frames=feature_tensor.shape[0],
        frame_hop_sec=frame_hop_sec,
        speaker_map=mapping,
    )
    if ref_matrix.size <= 0 or not np.any(ref_matrix > 0.0):
        return None

    activity_targets = (ref_matrix > 0.05).astype(np.float32)
    activity_mask = np.zeros_like(activity_targets, dtype=np.float32)
    for ref_speaker, canonical_speaker in mapping.items():
        if canonical_speaker in speaker_to_index:
            activity_mask[:, speaker_to_index[canonical_speaker]] = 1.0

    primary_indices: List[int] = []
    primary_targets: List[int] = []
    for frame_index in range(ref_matrix.shape[0]):
        row = ref_matrix[frame_index]
        if not np.any(row > 0.0):
            continue
        primary_indices.append(frame_index)
        primary_targets.append(int(np.argmax(row)))

    return {
        "primary_indices": np.asarray(primary_indices, dtype=np.int64),
        "primary_targets": np.asarray(primary_targets, dtype=np.int64),
        "activity_targets": activity_targets,
        "activity_mask": activity_mask,
    }


def _activity_f1(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float,
    mask: Optional[np.ndarray] = None,
) -> float:
    pred = (np.asarray(probs) >= float(threshold)).astype(np.float32)
    ref = (np.asarray(targets) >= 0.5).astype(np.float32)
    if mask is None:
        valid = np.ones_like(ref, dtype=np.float32)
    else:
        valid = np.asarray(mask, dtype=np.float32)
    tp = float(np.sum(pred * ref * valid))
    fp = float(np.sum(pred * (1.0 - ref) * valid))
    fn = float(np.sum((1.0 - pred) * ref * valid))
    if tp <= 0.0:
        return 0.0
    precision = tp / max(tp + fp, 1e-6)
    recall = tp / max(tp + fn, 1e-6)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _binary_f1(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float32)
    ref_arr = np.asarray(ref, dtype=np.float32)
    tp = float(np.sum(pred_arr * ref_arr))
    fp = float(np.sum(pred_arr * (1.0 - ref_arr)))
    fn = float(np.sum((1.0 - pred_arr) * ref_arr))
    if tp <= 0.0:
        return 0.0
    precision = tp / max(tp + fp, 1e-6)
    recall = tp / max(tp + fn, 1e-6)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _overlap_frame_f1(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float,
    mask: Optional[np.ndarray] = None,
) -> float:
    prob_arr = np.asarray(probs, dtype=np.float32)
    target_arr = np.asarray(targets, dtype=np.float32)
    if mask is None:
        valid = np.ones_like(target_arr, dtype=np.float32)
    else:
        valid = np.asarray(mask, dtype=np.float32)
    pred = ((prob_arr >= float(threshold)).astype(np.float32) * valid).sum(axis=1) >= 2.0
    ref = ((target_arr >= 0.5).astype(np.float32) * valid).sum(axis=1) >= 2.0
    return _binary_f1(pred.astype(np.float32), ref.astype(np.float32))


def _primary_accuracy(
    calibrator: PosteriorFusionCalibrator,
    *,
    examples: List[Dict[str, Any]],
) -> float:
    primary_correct = 0
    primary_total = 0
    for item in examples:
        feature_tensor = item["feature_tensor"]
        supervision = item["supervision"]
        primary_idx = supervision["primary_indices"]
        if primary_idx.size <= 0:
            continue
        primary_probs = calibrator.probabilities(feature_tensor[primary_idx])
        pred = np.argmax(primary_probs, axis=1)
        primary_correct += int(np.sum(pred == supervision["primary_targets"]))
        primary_total += int(primary_idx.size)
    if primary_total <= 0:
        return 0.0
    return float(primary_correct) / float(primary_total)


def _candidate_list(
    raw_values: Any,
    *,
    default: List[float],
) -> List[float]:
    if isinstance(raw_values, (list, tuple)):
        values: List[float] = []
        for item in raw_values:
            try:
                values.append(float(item))
            except Exception:
                continue
        if values:
            return values
    return list(default)


def _activity_sample_weights(
    *,
    supervision: Dict[str, np.ndarray],
    activity_loss_weight: float,
    overlap_loss_weight: float,
) -> np.ndarray:
    targets = np.asarray(supervision["activity_targets"], dtype=np.float32)
    mask = np.asarray(supervision["activity_mask"], dtype=np.float32)
    weights = np.ones_like(targets, dtype=np.float32) * max(0.05, float(activity_loss_weight))
    overlap_frames = (np.sum((targets >= 0.5).astype(np.float32) * mask, axis=1) >= 2.0).astype(np.float32)
    if np.any(overlap_frames > 0.0):
        positive_overlap = (targets >= 0.5).astype(np.float32) * overlap_frames[:, None]
        weights += positive_overlap * max(0.0, float(overlap_loss_weight) - 1.0) * max(0.05, float(activity_loss_weight))
    return weights * mask


def _pseudo_activity_sample_weights(
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    base_weight: float,
) -> np.ndarray:
    target_arr = np.asarray(targets, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=np.float32)
    weights = np.ones_like(target_arr, dtype=np.float32) * max(0.02, float(base_weight))
    weights += (target_arr >= 0.5).astype(np.float32) * max(0.0, float(base_weight) * 0.35)
    return weights * mask_arr


def _adapt_calibrator_with_consensus(
    calibrator: PosteriorFusionCalibrator,
    *,
    examples: List[Dict[str, Any]],
    pseudo_epochs: int,
    pseudo_primary_weight: float,
    pseudo_activity_weight: float,
) -> int:
    if pseudo_epochs <= 0 or not examples:
        return 0
    total_updates = 0
    primary_weight = max(0.02, float(pseudo_primary_weight))
    activity_weight = max(0.02, float(pseudo_activity_weight))
    for _epoch in range(max(1, int(pseudo_epochs))):
        for item in examples:
            feature_tensor = PosteriorFusionCalibrator._coerce_feature_tensor(item["feature_tensor"])
            if feature_tensor.ndim != 3 or feature_tensor.shape[2] < 3:
                continue
            msdd = feature_tensor[:, :, 0]
            pyannote = feature_tensor[:, :, 1]
            sortformer = feature_tensor[:, :, 2]

            pseudo = calibrator.build_pseudo_labels(
                msdd=msdd,
                pyannote=pyannote,
                sortformer=sortformer,
            )
            if pseudo["indices"].size > 0:
                total_updates += calibrator.update(
                    feature_tensor[pseudo["indices"]],
                    pseudo["targets"],
                    sample_weights=np.full(
                        (pseudo["indices"].size,),
                        primary_weight,
                        dtype=np.float32,
                    ),
                )

            pseudo_activity = calibrator.build_pseudo_activity_targets(
                msdd=msdd,
                pyannote=pyannote,
                sortformer=sortformer,
            )
            if pseudo_activity["indices"].size > 0:
                total_updates += calibrator.update_activity(
                    feature_tensor[pseudo_activity["indices"]],
                    pseudo_activity["targets"],
                    mask=pseudo_activity["mask"],
                    sample_weights=_pseudo_activity_sample_weights(
                        pseudo_activity["targets"],
                        pseudo_activity["mask"],
                        base_weight=activity_weight,
                    ),
                )
    return int(total_updates)


def _evaluate_calibrator(
    calibrator: PosteriorFusionCalibrator,
    *,
    examples: List[Dict[str, Any]],
    threshold_grid: List[float],
    objective_weights: Dict[str, float],
) -> Dict[str, float]:
    best_metrics = {
        "activity_threshold": float(calibrator.activity_threshold),
        "activity_f1": 0.0,
        "overlap_f1": 0.0,
        "primary_accuracy": _primary_accuracy(calibrator, examples=examples),
        "objective_score": -1.0,
    }
    for threshold in threshold_grid:
        activity_scores: List[float] = []
        overlap_scores: List[float] = []
        for item in examples:
            feature_tensor = item["feature_tensor"]
            supervision = item["supervision"]
            probs = calibrator.activity_probabilities(feature_tensor)
            activity_scores.append(
                _activity_f1(
                    probs,
                    supervision["activity_targets"],
                    threshold=float(threshold),
                    mask=supervision["activity_mask"],
                )
            )
            overlap_scores.append(
                _overlap_frame_f1(
                    probs,
                    supervision["activity_targets"],
                    threshold=float(threshold),
                    mask=supervision["activity_mask"],
                )
            )
        activity_f1 = float(np.mean(activity_scores)) if activity_scores else 0.0
        overlap_f1 = float(np.mean(overlap_scores)) if overlap_scores else 0.0
        objective_score = (
            float(objective_weights.get("primary", 1.0)) * best_metrics["primary_accuracy"]
            + float(objective_weights.get("activity", 1.0)) * activity_f1
            + float(objective_weights.get("overlap", 1.35)) * overlap_f1
        )
        candidate_tuple = (objective_score, overlap_f1, activity_f1, best_metrics["primary_accuracy"])
        best_tuple = (
            float(best_metrics["objective_score"]),
            float(best_metrics["overlap_f1"]),
            float(best_metrics["activity_f1"]),
            float(best_metrics["primary_accuracy"]),
        )
        if candidate_tuple > best_tuple:
            best_metrics = {
                "activity_threshold": float(threshold),
                "activity_f1": activity_f1,
                "overlap_f1": overlap_f1,
                "primary_accuracy": best_metrics["primary_accuracy"],
                "objective_score": objective_score,
            }
    return best_metrics


def train_calibrator_from_example_dir(
    *,
    examples_dir: Path,
    rttm_dir: Path,
    output_path: Path,
    calibrator_cfg: Optional[Dict[str, Any]] = None,
    epochs: int = 6,
    trainer_cfg: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    example_paths = sorted(examples_dir.glob("*.json"))
    base_calibrator_cfg = copy.deepcopy(calibrator_cfg or {})
    search_cfg = trainer_cfg if isinstance(trainer_cfg, dict) else {}

    labeled_examples: List[Dict[str, Any]] = []
    adaptation_examples: List[Dict[str, Any]] = []
    for meta_path in example_paths:
        try:
            metadata = json.loads(meta_path.read_text("utf-8"))
        except Exception:
            continue
        npz_path = Path(str(metadata.get("npz_path", meta_path.with_suffix(".npz"))))
        if not npz_path.is_absolute():
            npz_path = meta_path.parent / npz_path.name
        if not npz_path.exists():
            continue
        with np.load(npz_path) as bundle:
            feature_tensor = np.asarray(bundle["feature_tensor"], dtype=np.float32)
        feature_tensor = PosteriorFusionCalibrator._coerce_feature_tensor(feature_tensor)
        example_entry = {
            "id": str(metadata.get("example_id", meta_path.stem)),
            "feature_tensor": feature_tensor,
        }
        adaptation_examples.append(example_entry)
        rttm_path = resolve_reference_rttm(metadata=metadata, rttm_dir=rttm_dir)
        if rttm_path is None or not rttm_path.exists():
            continue
        reference_segments = load_rttm_segments(rttm_path)
        if not reference_segments:
            continue
        frame_hop_sec = safe_float(metadata.get("frame_hop_sec", 0.08), 0.08)
        supervision = build_supervision_from_reference(
            metadata=metadata,
            feature_tensor=feature_tensor,
            frame_hop_sec=frame_hop_sec,
            reference_segments=reference_segments,
        )
        if supervision is None:
            continue
        labeled_examples.append(
            {
                **example_entry,
                "supervision": supervision,
            }
        )

    if not labeled_examples and not adaptation_examples:
        raise RuntimeError(f"No trainable posterior-fusion examples found in {examples_dir} with RTTM from {rttm_dir}")

    if progress_callback is not None:
        progress_callback(
            "Posterior fusion trainer: "
            f"loaded {len(labeled_examples)} labeled example(s) and "
            f"{max(0, len(adaptation_examples) - len(labeled_examples))} unlabeled example(s) "
            f"from {examples_dir} with RTTM from {rttm_dir}."
        )

    threshold_min = safe_float(search_cfg.get("threshold_min", 0.20), 0.20)
    threshold_max = safe_float(search_cfg.get("threshold_max", 0.75), 0.75)
    threshold_num = max(3, safe_int(search_cfg.get("threshold_num", 12), 12))
    threshold_grid = np.linspace(threshold_min, threshold_max, num=threshold_num, dtype=np.float32).tolist()
    primary_loss_candidates = _candidate_list(
        search_cfg.get("primary_loss_candidates"),
        default=[0.85, 1.0, 1.20],
    )
    activity_loss_candidates = _candidate_list(
        search_cfg.get("activity_loss_candidates"),
        default=[0.90, 1.0, 1.15],
    )
    overlap_loss_candidates = _candidate_list(
        search_cfg.get("overlap_loss_candidates"),
        default=[1.0, 1.35, 1.70],
    )
    objective_cfg = search_cfg.get("objective_weights", {}) or {}
    objective_weights = {
        "primary": safe_float(objective_cfg.get("primary", 1.0), 1.0),
        "activity": safe_float(objective_cfg.get("activity", 1.0), 1.0),
        "overlap": safe_float(objective_cfg.get("overlap", 1.35), 1.35),
    }
    pseudo_epochs = max(0, safe_int(search_cfg.get("pseudo_epochs", 2), 2))
    pseudo_primary_weight = max(0.02, safe_float(search_cfg.get("pseudo_primary_weight", 0.32), 0.32))
    pseudo_activity_weight = max(0.02, safe_float(search_cfg.get("pseudo_activity_weight", 0.28), 0.28))

    best_state: Optional[PosteriorFusionCalibrator] = None
    best_metrics: Dict[str, Any] = {
        "objective_score": -1.0,
        "activity_f1": 0.0,
        "overlap_f1": 0.0,
        "primary_accuracy": 0.0,
        "activity_threshold": float(base_calibrator_cfg.get("activity_threshold", 0.46) or 0.46),
    }
    best_search: Dict[str, Any] = {}
    if labeled_examples:
        candidate_grid = list(product(primary_loss_candidates, activity_loss_candidates, overlap_loss_candidates))
        total_candidates = len(candidate_grid)
        for candidate_index, (primary_loss_weight, activity_loss_weight, overlap_loss_weight) in enumerate(
            candidate_grid,
            start=1,
        ):
            calibrator = PosteriorFusionCalibrator(cfg=base_calibrator_cfg, persist_path=None)
            primary_weight = max(0.05, float(primary_loss_weight))
            activity_weight = max(0.05, float(activity_loss_weight))
            overlap_weight = max(1.0, float(overlap_loss_weight))
            if progress_callback is not None:
                progress_callback(
                    "Posterior fusion trainer: "
                    f"search {candidate_index}/{total_candidates} "
                    f"(primary={primary_weight:.2f}, activity={activity_weight:.2f}, overlap={overlap_weight:.2f})"
                )
            for _epoch in range(max(1, int(epochs))):
                for item in labeled_examples:
                    feature_tensor = item["feature_tensor"]
                    supervision = item["supervision"]
                    primary_idx = supervision["primary_indices"]
                    if primary_idx.size > 0:
                        calibrator.update(
                            feature_tensor[primary_idx],
                            supervision["primary_targets"],
                            sample_weights=np.full((primary_idx.size,), primary_weight, dtype=np.float32),
                        )
                    calibrator.update_activity(
                        feature_tensor,
                        supervision["activity_targets"],
                        mask=supervision["activity_mask"],
                        sample_weights=_activity_sample_weights(
                            supervision=supervision,
                            activity_loss_weight=activity_weight,
                            overlap_loss_weight=overlap_weight,
                        ),
                    )

            metrics = _evaluate_calibrator(
                calibrator,
                examples=labeled_examples,
                threshold_grid=threshold_grid,
                objective_weights=objective_weights,
            )
            candidate_tuple = (
                float(metrics["objective_score"]),
                float(metrics["overlap_f1"]),
                float(metrics["activity_f1"]),
                float(metrics["primary_accuracy"]),
            )
            best_tuple = (
                float(best_metrics.get("objective_score", -1.0)),
                float(best_metrics.get("overlap_f1", 0.0)),
                float(best_metrics.get("activity_f1", 0.0)),
                float(best_metrics.get("primary_accuracy", 0.0)),
            )
            if candidate_tuple > best_tuple:
                calibrator.activity_threshold = float(metrics["activity_threshold"])
                best_state = calibrator
                best_metrics = dict(metrics)
                best_search = {
                    "primary_loss_weight": primary_weight,
                    "activity_loss_weight": activity_weight,
                    "overlap_loss_weight": overlap_weight,
                }
    else:
        best_state = PosteriorFusionCalibrator(cfg=base_calibrator_cfg, persist_path=None)
        best_search = {"mode": "consensus_only"}
        best_metrics = {
            "objective_score": 0.0,
            "activity_f1": 0.0,
            "overlap_f1": 0.0,
            "primary_accuracy": 0.0,
            "activity_threshold": float(best_state.activity_threshold),
        }

    if best_state is None:
        raise RuntimeError("Posterior fusion trainer failed to produce a valid calibrator state.")

    labeled_ids = {str(item.get("id", "")) for item in labeled_examples}
    consensus_examples = [
        item for item in adaptation_examples if str(item.get("id", "")) not in labeled_ids
    ]
    if not consensus_examples and not labeled_examples:
        consensus_examples = list(adaptation_examples)

    consensus_updates = 0
    if pseudo_epochs > 0 and consensus_examples:
        if progress_callback is not None:
            progress_callback(
                "Posterior fusion trainer: "
                f"consensus adaptation on {len(consensus_examples)} example(s) "
                f"for {pseudo_epochs} epoch(s)"
            )
        consensus_updates = _adapt_calibrator_with_consensus(
            best_state,
            examples=consensus_examples,
            pseudo_epochs=pseudo_epochs,
            pseudo_primary_weight=pseudo_primary_weight,
            pseudo_activity_weight=pseudo_activity_weight,
        )
    if labeled_examples:
        best_metrics = _evaluate_calibrator(
            best_state,
            examples=labeled_examples,
            threshold_grid=threshold_grid,
            objective_weights=objective_weights,
        )
        best_state.activity_threshold = float(best_metrics["activity_threshold"])

    best_state.persist_path = output_path
    best_state.training_metadata = {
        "epochs": max(1, int(epochs)),
        "objective_weights": dict(objective_weights),
        "loss_search": dict(best_search),
        "pseudo_adaptation": {
            "epochs": int(pseudo_epochs),
            "primary_weight": float(pseudo_primary_weight),
            "activity_weight": float(pseudo_activity_weight),
            "examples": int(len(consensus_examples)),
            "updates": int(consensus_updates),
        },
        "threshold_grid": {
            "min": float(threshold_min),
            "max": float(threshold_max),
            "num": int(threshold_num),
        },
        "examples": int(len(labeled_examples)),
        "adaptation_examples": int(len(adaptation_examples)),
        "metrics": {
            "objective_score": float(best_metrics.get("objective_score", 0.0)),
            "primary_accuracy": float(best_metrics.get("primary_accuracy", 0.0)),
            "activity_f1": float(best_metrics.get("activity_f1", 0.0)),
            "overlap_f1": float(best_metrics.get("overlap_f1", 0.0)),
        },
    }
    best_state.save_state()
    if progress_callback is not None:
        progress_callback(
            "Posterior fusion trainer: "
            f"best threshold={float(best_metrics['activity_threshold']):.3f}, "
            f"objective={float(best_metrics['objective_score']):.4f}, "
            f"consensus_updates={int(consensus_updates)}"
        )

    return {
        "examples": len(labeled_examples),
        "adaptation_examples": len(adaptation_examples),
        "unlabeled_examples": max(0, len(adaptation_examples) - len(labeled_examples)),
        "epochs": max(1, int(epochs)),
        "activity_threshold": float(best_metrics["activity_threshold"]),
        "activity_f1": float(best_metrics["activity_f1"]),
        "overlap_f1": float(best_metrics["overlap_f1"]),
        "primary_accuracy": float(best_metrics["primary_accuracy"]),
        "objective_score": float(best_metrics["objective_score"]),
        "loss_search": dict(best_search),
        "pseudo_adaptation": {
            "epochs": int(pseudo_epochs),
            "primary_weight": float(pseudo_primary_weight),
            "activity_weight": float(pseudo_activity_weight),
            "updates": int(consensus_updates),
        },
        "objective_weights": dict(objective_weights),
        "output_path": str(output_path),
        "weights": best_state.describe_weights(),
    }
