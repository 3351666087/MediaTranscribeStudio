from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from output_layout import resolve_runtime_artifact_path


FEATURE_NAMES = (
    "msdd",
    "pyannote",
    "sortformer",
    "msdd_pyannote_agree",
    "msdd_sortformer_agree",
    "pyannote_sortformer_agree",
    "agreement_ratio",
    "boundary_support",
    "exclusive_support",
    "anchor_support",
    "msdd_quality_support",
    "pyannote_quality_support",
    "sortformer_quality_support",
)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -30.0, 30.0))
    denom = np.sum(exp_scores, axis=1, keepdims=True)
    return exp_scores / np.clip(denom, 1e-6, None)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -30.0, 30.0)))


class PosteriorFusionCalibrator:
    """
    Lightweight calibrator for posterior fusion.

    Two coupled heads are maintained:
      1. primary speaker head: shared-feature softmax over speakers
      2. activity head: per-speaker sigmoid used for overlap / multi-label decoding

    The state is intentionally small so it can be adapted online and also trained
    offline from dumped development examples with RTTM supervision.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, persist_path: Optional[Path] = None):
        root = cfg if isinstance(cfg, dict) else {}
        self.enabled = bool(root.get("enabled", True))
        self.learning_rate = float(root.get("online_learning_rate", 0.03) or 0.03)
        self.l2_reg = float(root.get("l2_reg", 0.002) or 0.002)
        self.min_pseudo_margin = float(root.get("min_pseudo_margin", 0.12) or 0.12)
        self.min_pseudo_models = int(root.get("min_pseudo_models", 2) or 2)
        self.activity_threshold = float(root.get("activity_threshold", 0.46) or 0.46)

        primary_cfg = root.get("feature_weights", {}) or {}
        self.weights = np.array(
            [
                float(primary_cfg.get(name, default))
                for name, default in (
                    ("msdd", 1.20),
                    ("pyannote", 1.35),
                    ("sortformer", 0.72),
                    ("msdd_pyannote_agree", 0.70),
                    ("msdd_sortformer_agree", 0.42),
                    ("pyannote_sortformer_agree", 0.50),
                    ("agreement_ratio", 0.35),
                    ("boundary_support", 0.22),
                    ("exclusive_support", 0.32),
                    ("anchor_support", 0.18),
                    ("msdd_quality_support", 0.0),
                    ("pyannote_quality_support", 0.0),
                    ("sortformer_quality_support", 0.0),
                )
            ],
            dtype=np.float32,
        )

        activity_cfg = root.get("activity_feature_weights", {}) or {}
        self.activity_weights = np.array(
            [
                float(activity_cfg.get(name, default))
                for name, default in (
                    ("msdd", 1.10),
                    ("pyannote", 1.18),
                    ("sortformer", 0.58),
                    ("msdd_pyannote_agree", 0.82),
                    ("msdd_sortformer_agree", 0.48),
                    ("pyannote_sortformer_agree", 0.58),
                    ("agreement_ratio", 0.52),
                    ("boundary_support", -0.08),
                    ("exclusive_support", 0.18),
                    ("anchor_support", 0.10),
                    ("msdd_quality_support", 0.0),
                    ("pyannote_quality_support", 0.0),
                    ("sortformer_quality_support", 0.0),
                )
            ],
            dtype=np.float32,
        )
        self.activity_bias = float(root.get("activity_bias", -1.05) or -1.05)

        self.persist_path = Path(persist_path) if persist_path is not None else None
        self.updates = 0
        self.training_metadata: Dict[str, Any] = {}
        self._load_state()

    def _load_state(self) -> None:
        path = self.persist_path
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text("utf-8"))
        except Exception:
            return
        if list(payload.get("feature_names") or []) != list(FEATURE_NAMES):
            return

        raw_primary = payload.get("primary_weights", payload.get("weights"))
        raw_activity = payload.get("activity_weights")
        if not isinstance(raw_primary, list) or len(raw_primary) != len(FEATURE_NAMES):
            return
        try:
            self.weights = np.asarray(raw_primary, dtype=np.float32)
            if isinstance(raw_activity, list) and len(raw_activity) == len(FEATURE_NAMES):
                self.activity_weights = np.asarray(raw_activity, dtype=np.float32)
            self.activity_bias = float(payload.get("activity_bias", self.activity_bias) or self.activity_bias)
            self.activity_threshold = float(
                payload.get("activity_threshold", self.activity_threshold) or self.activity_threshold
            )
            self.updates = int(payload.get("updates", 0) or 0)
            raw_training = payload.get("training_metadata")
            if isinstance(raw_training, dict):
                self.training_metadata = dict(raw_training)
        except Exception:
            return

    def save_state(self) -> None:
        path = self.persist_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "feature_names": list(FEATURE_NAMES),
                "weights": [float(value) for value in self.weights.tolist()],
                "primary_weights": [float(value) for value in self.weights.tolist()],
                "activity_weights": [float(value) for value in self.activity_weights.tolist()],
                "activity_bias": float(self.activity_bias),
                "activity_threshold": float(self.activity_threshold),
                "updates": int(self.updates),
            }
            if self.training_metadata:
                payload["training_metadata"] = dict(self.training_metadata)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
            tmp_path.replace(path)
        except Exception:
            return

    @staticmethod
    def _coerce_feature_tensor(features: np.ndarray) -> np.ndarray:
        tensor = np.asarray(features, dtype=np.float32)
        if tensor.ndim != 3:
            return tensor
        feature_dim = int(tensor.shape[2])
        target_dim = len(FEATURE_NAMES)
        if feature_dim == target_dim:
            return tensor
        if feature_dim > target_dim:
            return np.asarray(tensor[:, :, :target_dim], dtype=np.float32)
        padded = np.zeros((tensor.shape[0], tensor.shape[1], target_dim), dtype=np.float32)
        padded[:, :, :feature_dim] = tensor
        return padded

    def emission_scores(self, features: np.ndarray) -> np.ndarray:
        tensor = self._coerce_feature_tensor(features)
        return np.tensordot(tensor, self.weights, axes=([-1], [0]))

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        scores = self.emission_scores(features)
        return _softmax(scores)

    def activity_logits(self, features: np.ndarray) -> np.ndarray:
        tensor = self._coerce_feature_tensor(features)
        return np.tensordot(tensor, self.activity_weights, axes=([-1], [0])) + float(self.activity_bias)

    def activity_probabilities(self, features: np.ndarray) -> np.ndarray:
        return _sigmoid(self.activity_logits(features))

    def update(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        sample_weights: Optional[np.ndarray] = None,
    ) -> int:
        if not self.enabled:
            return 0
        x = self._coerce_feature_tensor(features)
        y = np.asarray(targets, dtype=np.int64)
        if x.ndim != 3 or x.shape[0] <= 0 or x.shape[2] != len(FEATURE_NAMES):
            return 0
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            return 0
        if sample_weights is None:
            weights = np.ones((x.shape[0],), dtype=np.float32)
        else:
            weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
            if weights.shape[0] != x.shape[0]:
                return 0

        scores = self.emission_scores(x)
        probs = _softmax(scores)
        grad = np.zeros_like(self.weights)
        weight_sum = 0.0
        for sample_index in range(x.shape[0]):
            target_index = int(y[sample_index])
            if target_index < 0 or target_index >= x.shape[1]:
                continue
            sample_weight = max(0.0, float(weights[sample_index]))
            if sample_weight <= 0.0:
                continue
            target = np.zeros((x.shape[1],), dtype=np.float32)
            target[target_index] = 1.0
            diff = probs[sample_index] - target
            grad += np.sum(x[sample_index] * diff[:, None], axis=0) * sample_weight
            weight_sum += sample_weight
        if weight_sum <= 0.0:
            return 0
        grad /= max(weight_sum, 1e-6)
        grad += self.l2_reg * self.weights
        self.weights = self.weights - self.learning_rate * grad.astype(np.float32)
        applied = int(max(1, round(weight_sum)))
        self.updates += applied
        return applied

    def update_activity(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> int:
        if not self.enabled:
            return 0
        x = self._coerce_feature_tensor(features)
        y = np.asarray(targets, dtype=np.float32)
        if x.ndim != 3 or x.shape[0] <= 0 or x.shape[2] != len(FEATURE_NAMES):
            return 0
        if y.ndim != 2 or y.shape[0] != x.shape[0] or y.shape[1] != x.shape[1]:
            return 0

        logits = self.activity_logits(x)
        probs = _sigmoid(logits)
        if mask is None:
            valid_mask = np.ones_like(y, dtype=np.float32)
        else:
            valid_mask = np.asarray(mask, dtype=np.float32)
            if valid_mask.shape != y.shape:
                return 0
        if sample_weights is None:
            weight_mask = np.ones_like(y, dtype=np.float32)
        else:
            weight_mask = np.asarray(sample_weights, dtype=np.float32)
            if weight_mask.shape != y.shape:
                return 0

        diff = (probs - y) * valid_mask * weight_mask
        denom = float(np.sum(valid_mask * weight_mask))
        if denom <= 0.0:
            return 0
        grad = np.sum(x * diff[:, :, None], axis=(0, 1)) / denom
        grad += self.l2_reg * self.activity_weights
        bias_grad = float(np.sum(diff)) / denom
        self.activity_weights = self.activity_weights - self.learning_rate * grad.astype(np.float32)
        self.activity_bias = float(self.activity_bias - self.learning_rate * bias_grad)
        self.updates += int(max(1, round(denom)))
        return int(max(1, round(denom)))

    def build_pseudo_labels(
        self,
        *,
        msdd: np.ndarray,
        pyannote: np.ndarray,
        sortformer: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        matrices = [np.asarray(msdd), np.asarray(pyannote), np.asarray(sortformer)]
        available = [matrix for matrix in matrices if matrix.size > 0]
        if not available:
            return {"indices": np.zeros((0,), dtype=np.int64), "targets": np.zeros((0,), dtype=np.int64)}

        num_frames = available[0].shape[0]
        samples: list[int] = []
        targets: list[int] = []
        strong_cut = 0.55
        for frame_index in range(num_frames):
            votes: Dict[int, int] = {}
            strengths: Dict[int, float] = {}
            for matrix in available:
                if frame_index >= matrix.shape[0] or matrix.shape[1] <= 0:
                    continue
                row = matrix[frame_index]
                best_idx = int(np.argmax(row))
                best_val = float(row[best_idx])
                sorted_row = np.sort(row)
                second_val = float(sorted_row[-2]) if row.shape[0] > 1 else 0.0
                if best_val < strong_cut or (best_val - second_val) < self.min_pseudo_margin:
                    continue
                votes[best_idx] = votes.get(best_idx, 0) + 1
                strengths[best_idx] = strengths.get(best_idx, 0.0) + best_val
            if not votes:
                continue
            winner, winner_votes = max(
                votes.items(),
                key=lambda item: (int(item[1]), float(strengths.get(item[0], 0.0)), -int(item[0])),
            )
            if winner_votes < max(1, self.min_pseudo_models):
                continue
            samples.append(frame_index)
            targets.append(int(winner))
        return {
            "indices": np.asarray(samples, dtype=np.int64),
            "targets": np.asarray(targets, dtype=np.int64),
        }

    def build_pseudo_activity_targets(
        self,
        *,
        msdd: np.ndarray,
        pyannote: np.ndarray,
        sortformer: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        matrices = [np.asarray(msdd), np.asarray(pyannote), np.asarray(sortformer)]
        available = [matrix for matrix in matrices if matrix.size > 0]
        if not available:
            return {
                "indices": np.zeros((0,), dtype=np.int64),
                "targets": np.zeros((0, 0), dtype=np.float32),
                "mask": np.zeros((0, 0), dtype=np.float32),
            }

        num_frames = available[0].shape[0]
        num_speakers = available[0].shape[1]
        strong_cut = 0.50
        weak_cut = 0.12

        active_votes = np.zeros((num_frames, num_speakers), dtype=np.float32)
        inactive_votes = np.zeros((num_frames, num_speakers), dtype=np.float32)
        for matrix in available:
            if matrix.shape[:2] != (num_frames, num_speakers):
                continue
            active_votes += (matrix >= strong_cut).astype(np.float32)
            inactive_votes += (matrix <= weak_cut).astype(np.float32)

        targets = np.zeros((num_frames, num_speakers), dtype=np.float32)
        mask = np.zeros((num_frames, num_speakers), dtype=np.float32)
        positive = active_votes >= float(max(1, self.min_pseudo_models))
        negative = inactive_votes >= float(max(1, self.min_pseudo_models))
        targets[positive] = 1.0
        mask[positive | negative] = 1.0

        frame_indices = np.nonzero(np.any(mask > 0.0, axis=1))[0].astype(np.int64)
        return {
            "indices": frame_indices,
            "targets": targets[frame_indices],
            "mask": mask[frame_indices],
        }

    def describe_weights(self) -> Dict[str, Any]:
        return {
            "primary": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, self.weights.tolist())
            },
            "activity": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, self.activity_weights.tolist())
            },
            "activity_bias": float(self.activity_bias),
            "activity_threshold": float(self.activity_threshold),
        }


def resolve_calibrator_path(raw_path: str, *, root_dir: Path, output_root: Optional[Path] = None) -> Path:
    path = Path(str(raw_path or "").strip() or "output_files/.posterior_fusion_calibrator.json")
    if not path.is_absolute():
        normalized = str(raw_path or "").strip().replace("\\", "/")
        if output_root is not None and normalized in {
            "",
            ".posterior_fusion_calibrator.json",
            "output_files/.posterior_fusion_calibrator.json",
            "_runtime_artifacts/posterior_fusion_calibrator.json",
        }:
            path = resolve_runtime_artifact_path(output_root, "posterior_fusion_calibrator.json")
        else:
            path = root_dir / path
    return path
