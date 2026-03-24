from __future__ import annotations

import copy
import json
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal, Slot

from config import Config
from diar_fusion.training import train_calibrator_from_example_dir
from pipeline import TranscriptionPipeline
from runtime_paths import resolve_app_writable_path
from utils import setup_logging


def _get_nested(data: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value = data
    for part in dotted_key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _set_nested(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    current = data
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        node = current.get(part)
        if not isinstance(node, dict):
            node = {}
            current[part] = node
        current = node
    current[parts[-1]] = value


def _resolve_ui_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if not path.is_absolute():
        path = resolve_app_writable_path(path)
    return path


class UILogHandler(logging.Handler):
    """Bridge Python logging records to Qt signals."""

    def __init__(self, emit_signal):
        super().__init__()
        self._emit_signal = emit_signal

    def emit(self, record) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self._emit_signal.emit(msg)


class PipelineWorker(QObject):
    """Run the transcription pipeline inside a background QThread."""

    log_message = Signal(str)
    status_message = Signal(str)
    file_started = Signal(int, int, str)
    file_step = Signal(dict)
    file_finished = Signal(dict)
    run_started = Signal(int)
    model_download = Signal(dict)
    run_finished = Signal(dict)
    run_failed = Signal(str)

    def __init__(
        self,
        config_data: Dict[str, Any],
        files: List[str],
        input_root: str,
        run_options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.config_data = config_data
        self.files = files
        self.input_root = input_root
        self.run_options = dict(run_options or {})
        self.pause_event = threading.Event()
        self._pipeline: Optional[TranscriptionPipeline] = None
        self._ui_log_handler: Optional[UILogHandler] = None
        self._completion_payload: Optional[Dict[str, Any]] = None

    def _setup_logging(self, cfg: Config) -> None:
        requested_level = str(cfg.get("logging.level", "INFO") or "INFO").upper()
        level = "DEBUG" if requested_level not in {"DEBUG", "NOTSET"} else requested_level
        log_file = cfg.get("logging.log_file", "pipeline.log")
        setup_logging(level=level, log_file=log_file)

        self._ui_log_handler = UILogHandler(self.log_message)
        self._ui_log_handler.setLevel(logging.DEBUG)
        self._ui_log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logging.getLogger().addHandler(self._ui_log_handler)

    def _teardown_logging(self) -> None:
        if self._ui_log_handler:
            logging.getLogger().removeHandler(self._ui_log_handler)
            self._ui_log_handler = None

    def _on_progress(self, event: str, payload: Dict[str, Any]) -> None:
        if event == "run_start":
            self.run_started.emit(int(payload.get("total", 0)))
        elif event == "file_start":
            self.file_started.emit(
                int(payload.get("index", 0)),
                int(payload.get("total", 0)),
                str(payload.get("file_path", "")),
            )
        elif event == "file_step":
            normalized = dict(payload or {})
            normalized["file_path"] = str(normalized.get("file_path", ""))
            normalized["step"] = str(normalized.get("step", ""))
            self.file_step.emit(normalized)
        elif event == "file_done":
            self.file_finished.emit(dict(payload.get("result", {})))
        elif event == "model_download":
            self.model_download.emit(dict(payload))
        elif event == "paused":
            self.status_message.emit("已暂停，等待继续")
        elif event == "resumed":
            self.status_message.emit("正在继续处理")
        elif event == "run_complete":
            self._completion_payload = dict(payload)

    @Slot()
    def pause(self) -> None:
        self.pause_event.set()
        self.status_message.emit("暂停请求已发送")

    @Slot()
    def resume(self) -> None:
        self.pause_event.clear()
        self.status_message.emit("正在恢复执行")

    @Slot()
    def request_stop(self) -> None:
        if self._pipeline:
            self._pipeline.request_shutdown()
            self.status_message.emit("停止请求已发送")

    @Slot()
    def run(self) -> None:
        try:
            cfg = Config(config_path=None)
            cfg._data = copy.deepcopy(self.config_data)
            cfg._validate()
            if bool(self.run_options.get("disable_llm_processing", False)):
                llm_cfg = cfg._data.setdefault("llm", {})
                llm_cfg["enabled"] = False
                llm_cfg.setdefault("speaker_arbitration", {})["enabled"] = False
                llm_cfg.setdefault("optimize_language", {})["enabled"] = False
                cfg._data.setdefault("translation", {})["enabled"] = False
            if bool(self.run_options.get("disable_resume", False)):
                cfg._data.setdefault("resume", {})["enabled"] = False
            overrides = self.run_options.get("config_overrides")
            if isinstance(overrides, dict):
                for dotted_key, value in overrides.items():
                    if not str(dotted_key or "").strip():
                        continue
                    _set_nested(cfg._data, str(dotted_key), value)
            cfg._data.setdefault("logging", {})["show_progress"] = False
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

            self._setup_logging(cfg)
            self.status_message.emit("正在初始化流水线...")

            self._pipeline = TranscriptionPipeline(
                cfg,
                progress_callback=self._on_progress,
                pause_event=self.pause_event,
                enable_signal_handlers=False,
            )
            self._pipeline.run(
                media_files=[Path(p) for p in self.files],
                input_dir=Path(self.input_root),
            )

            if self._completion_payload is None:
                total = len(self.files)
                ok = sum(1 for r in self._pipeline.all_results if r.get("status") == "OK")
                self._completion_payload = {
                    "total": total,
                    "processed": len(self._pipeline.all_results),
                    "ok": ok,
                }

            self.run_finished.emit(self._completion_payload)

        except Exception:
            self.run_failed.emit(traceback.format_exc())
        finally:
            self._teardown_logging()
            self._pipeline = None


class PosteriorFusionTrainerWorker(QObject):
    """Run posterior-fusion offline calibration in a background QThread."""

    log_message = Signal(str)
    status_message = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        config_data: Dict[str, Any],
        *,
        examples_dir: str,
        rttm_dir: str,
        output_path: str,
        epochs: int,
    ):
        super().__init__()
        self.config_data = copy.deepcopy(config_data)
        self.examples_dir = str(examples_dir or "")
        self.rttm_dir = str(rttm_dir or "")
        self.output_path = str(output_path or "")
        self.epochs = max(1, int(epochs))

    def _emit_progress(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        self.status_message.emit(text)
        self.log_message.emit(text)

    @Slot()
    def run(self) -> None:
        try:
            cfg = Config(config_path=None)
            cfg._data = copy.deepcopy(self.config_data)
            cfg._validate()
            posterior_cfg = (
                _get_nested(cfg._data, "asr.nemo_msdd.hybrid_fusion.posterior_decoder", {}) or {}
            )
            calibrator_cfg = dict((posterior_cfg.get("calibrator", {}) or {}))
            trainer_cfg = dict((posterior_cfg.get("trainer", {}) or {}))

            self._emit_progress("Posterior Fusion: 正在读取开发集样本和 RTTM 标注...")
            metrics = train_calibrator_from_example_dir(
                examples_dir=_resolve_ui_path(self.examples_dir),
                rttm_dir=_resolve_ui_path(self.rttm_dir),
                output_path=_resolve_ui_path(self.output_path),
                calibrator_cfg=calibrator_cfg,
                epochs=self.epochs,
                trainer_cfg=trainer_cfg,
                progress_callback=self._emit_progress,
            )
            self.log_message.emit(json.dumps(metrics, ensure_ascii=False, indent=2))
            self.finished.emit(metrics)
        except Exception:
            self.failed.emit(traceback.format_exc())
