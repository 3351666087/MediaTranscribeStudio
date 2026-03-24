from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config import Config
from diar_fusion.training import train_calibrator_from_example_dir


def _parse_float_list(text: str):
    values = []
    for item in str(text or "").split(","):
        piece = item.strip()
        if not piece:
            continue
        values.append(float(piece))
    return values


def main() -> int:
    cfg = Config()
    posterior_cfg = (
        cfg["asr"].get("nemo_msdd", {}).get("hybrid_fusion", {}).get("posterior_decoder", {}) or {}
    )
    calibrator_cfg = dict(posterior_cfg.get("calibrator", {}) or {})
    trainer_cfg = dict(posterior_cfg.get("trainer", {}) or {})

    parser = argparse.ArgumentParser(description="Train posterior-fusion calibrator from dumped dev examples and RTTM supervision.")
    parser.add_argument(
        "--examples-dir",
        default=str(trainer_cfg.get("examples_dir", "output_files/posterior_fusion_examples")),
        help="Directory containing dumped posterior-fusion example .json/.npz files.",
    )
    parser.add_argument(
        "--rttm-dir",
        default=str(trainer_cfg.get("rttm_dir", "output_files/posterior_fusion_rttm")),
        help="Directory containing reference RTTM files.",
    )
    parser.add_argument(
        "--output",
        default=str(trainer_cfg.get("output_path", "output_files/.posterior_fusion_calibrator.json")),
        help="Output calibrator state path.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(trainer_cfg.get("epochs", 6) or 6),
        help="Number of offline training epochs.",
    )
    parser.add_argument("--threshold-min", type=float, default=None, help="Optional search grid lower bound.")
    parser.add_argument("--threshold-max", type=float, default=None, help="Optional search grid upper bound.")
    parser.add_argument("--threshold-num", type=int, default=None, help="Optional number of threshold candidates.")
    parser.add_argument("--primary-loss-candidates", type=str, default="", help="Comma-separated primary loss weights.")
    parser.add_argument("--activity-loss-candidates", type=str, default="", help="Comma-separated activity loss weights.")
    parser.add_argument("--overlap-loss-candidates", type=str, default="", help="Comma-separated overlap emphasis weights.")
    args = parser.parse_args()

    output_path = Path(str(args.output))
    if not output_path.is_absolute():
        output_path = APP_ROOT / output_path

    if args.threshold_min is not None:
        trainer_cfg["threshold_min"] = float(args.threshold_min)
    if args.threshold_max is not None:
        trainer_cfg["threshold_max"] = float(args.threshold_max)
    if args.threshold_num is not None:
        trainer_cfg["threshold_num"] = int(args.threshold_num)
    if args.primary_loss_candidates.strip():
        trainer_cfg["primary_loss_candidates"] = _parse_float_list(args.primary_loss_candidates)
    if args.activity_loss_candidates.strip():
        trainer_cfg["activity_loss_candidates"] = _parse_float_list(args.activity_loss_candidates)
    if args.overlap_loss_candidates.strip():
        trainer_cfg["overlap_loss_candidates"] = _parse_float_list(args.overlap_loss_candidates)

    metrics = train_calibrator_from_example_dir(
        examples_dir=Path(str(args.examples_dir)).expanduser(),
        rttm_dir=Path(str(args.rttm_dir)).expanduser(),
        output_path=output_path,
        calibrator_cfg=calibrator_cfg,
        epochs=max(1, int(args.epochs)),
        trainer_cfg=trainer_cfg,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
