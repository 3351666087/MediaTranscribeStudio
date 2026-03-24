"""
pipeline.py - Main transcription pipeline.
"""

import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import sitecustomize  # noqa: F401
except Exception:
    sitecustomize = None  # type: ignore[assignment]

import numpy as np
from tqdm import tqdm

from config import AUDIO_EXTENSIONS, Config
from output_layout import (
    resolve_output_subdir,
    resolve_output_temp_dir,
    resolve_runtime_artifact_path,
    resolve_runtime_artifacts_root,
)
from utils import (
    GracefulShutdown,
    get_media_duration,
    human_readable_duration,
    human_readable_size,
    log_accelerator_status,
    scan_media_files,
    setup_torch_performance,
    smart_empty_cache,
)
from audio_extractor import AudioExtractor
from llm_processor import LLMProcessor
from output_formatter import OutputFormatter
from report_generator import ReportGenerator
from video_text_overlay import VideoTextOverlayEngine
from taichi_kernels import (
    choose_chunk_duration_sec,
    compute_rms_energy_fast,
    compute_vad_mask_fast,
    deduplicate_overlap_text,
    init_taichi,
    normalize_audio_fast,
    smooth_segment_boundaries,
    split_audio_chunks_fast,
)
from transcriber import Transcriber, TranscriptionSegment

logger = logging.getLogger(__name__)


def _stream_supports_text(stream, text: str) -> bool:
    try:
        encoding = getattr(stream, "encoding", None) or sys.stdout.encoding or "utf-8"
        str(text).encode(encoding)
        return True
    except Exception:
        return False


def _stream_safe_text(stream, text: str) -> str:
    value = str(text)
    try:
        encoding = getattr(stream, "encoding", None) or sys.stdout.encoding or "utf-8"
        return value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except Exception:
        return value.encode("ascii", errors="replace").decode("ascii")


class TranscriptionPipeline:
    """End-to-end transcription pipeline."""

    def __init__(
        self,
        config: Config,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        pause_event=None,
        enable_signal_handlers: bool = True,
    ):
        self.config = config
        self.shutdown = GracefulShutdown(enable_signal_handlers=enable_signal_handlers)
        self.progress_callback = progress_callback
        self.pause_event = pause_event
        self.output_dir = Path(self.config["paths"]["output_dir"])
        self.runtime_artifacts_dir = resolve_runtime_artifacts_root(self.output_dir)

        resume_cfg = self.config.get("resume", {}) or {}
        self.resume_enabled = bool(resume_cfg.get("enabled", True))
        self.persist_segments = bool(resume_cfg.get("persist_segments", True))
        self.resume_dir = resolve_runtime_artifact_path(self.output_dir, "resume")
        self.session_state_path = self.resume_dir / "session_state.json"
        self.session_state: Dict[str, Any] = {"version": 1, "files": {}}
        if self.resume_enabled:
            self.resume_dir.mkdir(parents=True, exist_ok=True)
            self.session_state = self._load_session_state()

        # Best-effort silence for third-party noise logs.
        os.environ.setdefault("ONELOGGER_DISABLED", "1")
        os.environ.setdefault("ONE_LOGGER_DISABLED", "1")

        setup_torch_performance(config)

        ti_cfg = config["taichi"]
        if ti_cfg.get("enabled", True):
            init_taichi(
                arch=ti_cfg.get("arch", "cpu"),
                default_fp=ti_cfg.get("default_fp", 32),
            )

        logger.info("Initializing pipeline components...")
        self.audio_extractor = AudioExtractor(config)
        self.transcriber = Transcriber(config, progress_callback=self._emit_progress)
        self.formatter = OutputFormatter(config)
        self.report_gen = ReportGenerator(config)
        self.video_overlay = VideoTextOverlayEngine(config)

        self.llm = LLMProcessor(config)
        if self.llm.enabled:
            self.llm.start()
            modes: List[str] = []
            if getattr(self.llm, "analysis_enabled", False):
                modes.append("analysis")
            if getattr(self.llm, "translation_enabled", False):
                target = str(getattr(self.llm, "translation_target_language", "") or "")
                modes.append(f"translation->{target or 'unknown'}")
            mode_text = ", ".join(modes) if modes else "unknown"
            logger.info(f"LLM services started ({mode_text}).")
        else:
            reason = getattr(self.llm, "disabled_reason", "unknown")
            logger.info(f"LLM services disabled: {reason}")

        self.all_results: List[Dict[str, Any]] = []
        logger.info("Pipeline initialized.")
        log_accelerator_status("INIT ")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    @staticmethod
    def _file_fingerprint(file_path: Path) -> Dict[str, Any]:
        stat = file_path.stat()
        return {
            "size": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        }

    @staticmethod
    def _fingerprint_matches(file_path: Path, payload: Dict[str, Any]) -> bool:
        try:
            current = TranscriptionPipeline._file_fingerprint(file_path)
            saved_size = TranscriptionPipeline._safe_int(payload.get("file_size"), -1)
            saved_mtime = TranscriptionPipeline._safe_float(payload.get("file_mtime"), -1.0)
            if saved_size < 0 or saved_mtime < 0:
                return False
            if current["size"] != saved_size:
                return False
            return abs(current["mtime"] - saved_mtime) <= 0.5
        except Exception:
            return False

    @staticmethod
    def _file_key(file_path: Path) -> str:
        raw = str(file_path.resolve()).encode("utf-8", errors="replace")
        return hashlib.sha1(raw).hexdigest()

    def _checkpoint_path(self, file_path: Path) -> Path:
        key = self._file_key(file_path)
        stem = file_path.stem[:48] or "file"
        return self.resume_dir / f"{stem}_{key[:10]}.json"

    @staticmethod
    def _serialize_segments(segments: List[TranscriptionSegment]) -> List[Dict[str, Any]]:
        return [seg.to_dict() for seg in segments]

    @staticmethod
    def _deserialize_segments(payload: Any) -> List[TranscriptionSegment]:
        if not isinstance(payload, list):
            return []
        segments: List[TranscriptionSegment] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                segments.append(TranscriptionSegment(**item))
            except Exception:
                continue
        return segments

    def _result_snapshot(self, result: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "source_file",
            "source_path",
            "duration",
            "num_segments",
            "diarization_route",
            "elapsed",
            "status",
            "error",
            "detected_language",
            "pre_detected_language",
            "translation_applied",
            "translated_language",
            "language_optimized",
            "txt_path",
            "json_path",
            "html_path",
            "pdf_path",
            "burned_video_path",
            "subtitle_srt_path",
            "subtitle_ass_path",
            "subtitle_backend",
            "subtitle_stream_codec",
            "subtitle_stream_embedded",
        ]
        return {k: result.get(k) for k in keys if k in result}

    def _speaker_constraints_payload(
        self,
        *,
        allowed_speakers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        fixed_num = self._safe_int(self.config.get("asr.nemo_msdd.num_speakers", 0), 0)
        min_speakers = self._safe_int(self.config.get("asr.nemo_msdd.min_speakers", 1), 1)
        max_speakers = self._safe_int(self.config.get("asr.nemo_msdd.max_speakers", 8), 8)
        return {
            "mode": "manual" if fixed_num > 0 else "auto",
            "count_locked": bool(fixed_num > 0),
            "num_speakers": fixed_num,
            "min_speakers": min_speakers,
            "max_speakers": max(max_speakers, min_speakers),
            "speaker_labels": str(self.config.get("output.speaker_labels", "ABCDEFGHIJKLMNOPQRSTUVWXYZ") or ""),
            "allowed_speakers": list(allowed_speakers or []),
        }

    def _load_session_state(self) -> Dict[str, Any]:
        if not self.session_state_path.exists():
            return {"version": 1, "files": {}}
        try:
            with open(self.session_state_path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            files = payload.get("files")
            if not isinstance(files, dict):
                files = {}
            return {
                "version": int(payload.get("version", 1)),
                "files": files,
            }
        except Exception as e:
            logger.warning(f"Failed to load resume session state: {e}")
            return {"version": 1, "files": {}}

    def _save_session_state(self):
        if not self.resume_enabled:
            return
        payload = dict(self.session_state)
        payload["updated_at"] = time.time()
        try:
            self._atomic_write_json(self.session_state_path, payload)
        except Exception as e:
            logger.debug(f"Failed to save resume session state: {e}")

    def _load_checkpoint(self, file_path: Path) -> Dict[str, Any]:
        if not self.resume_enabled:
            return {}
        checkpoint_path = self._checkpoint_path(file_path)
        if not checkpoint_path.exists():
            return {}
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            if not self._fingerprint_matches(file_path, payload):
                logger.info(
                    "  Resume checkpoint invalidated (source file changed): "
                    f"{file_path.name}"
                )
                return {}
            return payload
        except Exception as e:
            logger.warning(f"Failed to load checkpoint for {file_path.name}: {e}")
            return {}

    def _write_checkpoint(self, file_path: Path, payload: Dict[str, Any]):
        if not self.resume_enabled:
            return
        checkpoint_path = self._checkpoint_path(file_path)
        merged = self._load_checkpoint(file_path)
        merged.update(payload)
        merged["version"] = 1
        merged["source_file"] = file_path.name
        merged["source_path"] = str(file_path.resolve())
        merged["updated_at"] = time.time()
        fingerprint = self._file_fingerprint(file_path)
        merged["file_size"] = fingerprint["size"]
        merged["file_mtime"] = fingerprint["mtime"]
        if not self.persist_segments:
            merged.pop("segments", None)

        try:
            self._atomic_write_json(checkpoint_path, merged)
        except Exception as e:
            logger.debug(f"Failed to write checkpoint for {file_path.name}: {e}")
            return

        entry = {
            "source_file": file_path.name,
            "source_path": str(file_path.resolve()),
            "status": str(merged.get("status", "processing") or "processing"),
            "stage": str(merged.get("stage", "") or ""),
            "checkpoint_path": str(checkpoint_path),
            "file_size": fingerprint["size"],
            "file_mtime": fingerprint["mtime"],
            "updated_at": merged["updated_at"],
            "result": merged.get("result", {}),
        }
        files = self.session_state.setdefault("files", {})
        files[self._file_key(file_path)] = entry
        self._save_session_state()

    @staticmethod
    def _checkpoint_report_has_analysis(checkpoint: Dict[str, Any]) -> bool:
        if not isinstance(checkpoint, dict):
            return False
        return bool(
            checkpoint.get("report_has_analysis", False)
            or checkpoint.get("analysis_in_report", False)
        )

    @staticmethod
    def _analysis_has_content(analysis: Any) -> bool:
        if analysis is None:
            return False
        return bool(
            str(getattr(analysis, "summary", "") or "").strip()
            or list(getattr(analysis, "key_points", []) or [])
            or list(getattr(analysis, "action_items", []) or [])
            or list(getattr(analysis, "topics", []) or [])
        )

    def _report_resume_compatible(
        self,
        checkpoint: Dict[str, Any],
        *,
        want_analysis: bool,
        file_name: str = "",
    ) -> bool:
        have_analysis = self._checkpoint_report_has_analysis(checkpoint)
        if bool(want_analysis) == bool(have_analysis):
            return True
        logger.info(
            "  Resume report invalidated: analysis section changed "
            "(want=%s, checkpoint=%s)%s",
            bool(want_analysis),
            bool(have_analysis),
            f" ({file_name})" if file_name else "",
        )
        return False

    def _path_exists(self, path_value: Any) -> bool:
        if not path_value:
            return False
        try:
            return Path(str(path_value)).exists()
        except Exception:
            return False

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    def _overlay_outputs_ready_for_resume(self, output_paths: Dict[str, Any]) -> bool:
        if not isinstance(output_paths, dict):
            return False

        video_burned = output_paths.get("video_burned")
        subtitle_srt = output_paths.get("subtitle_srt")
        if not self._path_exists(video_burned) or not self._path_exists(subtitle_srt):
            return False

        subtitle_ass = output_paths.get("subtitle_ass") or subtitle_srt
        if getattr(self.video_overlay, "embed_subtitle_stream", False):
            try:
                stream_info = self.video_overlay.describe_embedded_subtitle_stream(
                    Path(str(video_burned))
                )
            except Exception as exc:
                logger.warning("  Overlay resume validation failed: %s", exc)
                return False

            stream_count = int(stream_info.get("subtitle_stream_count", 0) or 0)
            if stream_count <= 0:
                logger.info(
                    "  Resume overlay output is missing embedded subtitle stream; attempting repair."
                )
                try:
                    repair_meta = self.video_overlay.ensure_embedded_subtitle_stream(
                        Path(str(video_burned)),
                        srt_path=Path(str(subtitle_srt)),
                        ass_path=Path(str(subtitle_ass)),
                    )
                except Exception as exc:
                    logger.warning("  Embedded subtitle stream repair failed: %s", exc)
                    return False
                if repair_meta:
                    output_paths.update(repair_meta)
                try:
                    stream_info = self.video_overlay.describe_embedded_subtitle_stream(
                        Path(str(video_burned))
                    )
                except Exception as exc:
                    logger.warning(
                        "  Overlay subtitle stream verification failed after repair: %s",
                        exc,
                    )
                    return False
                stream_count = int(stream_info.get("subtitle_stream_count", 0) or 0)
                if stream_count <= 0 and self.video_overlay.can_embed_subtitle_stream(
                    Path(str(video_burned)),
                    srt_path=Path(str(subtitle_srt)),
                    ass_path=Path(str(subtitle_ass)),
                ):
                    logger.warning(
                        "  Overlay subtitle stream is still missing after repair attempt."
                    )
                    return False

            if stream_count > 0:
                output_paths["subtitle_stream_embedded"] = "true"
                subtitle_codec = str(stream_info.get("subtitle_codec", "") or "").strip()
                if subtitle_codec:
                    output_paths["subtitle_stream_codec"] = subtitle_codec

        return True

    def _checkpoint_outputs_exist(
        self,
        output_paths: Dict[str, Any],
        require_report: bool = False,
    ) -> bool:
        if not isinstance(output_paths, dict):
            return False
        if not self._path_exists(output_paths.get("txt")):
            return False
        if not self._path_exists(output_paths.get("json")):
            return False
        if require_report and self.config.get("report.generate_html", True):
            if not self._path_exists(output_paths.get("html")):
                return False
        if require_report and self.config.get("report.generate_pdf", True):
            if not self._path_exists(output_paths.get("pdf")):
                return False
        return True

    def _load_completed_result(self, file_path: Path) -> Optional[Dict[str, Any]]:
        if not self.resume_enabled:
            return None
        files = self.session_state.get("files", {})
        entry = files.get(self._file_key(file_path))
        if not isinstance(entry, dict):
            return None
        if str(entry.get("status", "")).lower() != "done":
            return None
        if not self._fingerprint_matches(file_path, entry):
            return None

        checkpoint = self._load_checkpoint(file_path)
        if not checkpoint:
            return None
        if str(checkpoint.get("status", "")).lower() != "done":
            return None
        want_analysis = bool(
            getattr(self.llm, "analysis_enabled", False)
            and self._safe_int(checkpoint.get("num_segments", 0), 0) > 0
        )
        if not self._report_resume_compatible(
            checkpoint,
            want_analysis=want_analysis,
            file_name=file_path.name,
        ):
            return None

        output_paths = checkpoint.get("output_paths", {})
        if not self._checkpoint_outputs_exist(output_paths, require_report=False):
            return None
        if self.video_overlay.is_enabled_for(file_path):
            if not self._overlay_outputs_ready_for_resume(output_paths):
                return None

        snapshot = checkpoint.get("result", {})
        if not isinstance(snapshot, dict):
            snapshot = {}
        result: Dict[str, Any] = {
            "source_file": file_path.name,
            "source_path": str(file_path),
            "duration": self._safe_float(
                snapshot.get("duration", checkpoint.get("duration", 0.0)), 0.0
            ),
            "num_segments": self._safe_int(
                snapshot.get("num_segments", checkpoint.get("num_segments", 0)), 0
            ),
            "diarization_route": str(
                snapshot.get("diarization_route", checkpoint.get("diarization_route", ""))
                or ""
            ),
            "segments": [],
            "elapsed": self._safe_float(snapshot.get("elapsed", 0.0), 0.0),
            "pre_detected_language": str(
                snapshot.get(
                    "pre_detected_language",
                    checkpoint.get("pre_detected_language", ""),
                )
                or ""
            ),
            "detected_language": str(
                snapshot.get("detected_language", checkpoint.get("detected_language", ""))
                or ""
            ),
            "translation_applied": bool(
                snapshot.get(
                    "translation_applied",
                    checkpoint.get("translation_applied", False),
                )
            ),
            "translated_language": str(
                snapshot.get(
                    "translated_language",
                    checkpoint.get("translated_language", ""),
                )
                or ""
            ),
            "language_optimized": bool(
                snapshot.get(
                    "language_optimized",
                    checkpoint.get("language_optimized", False),
                )
            ),
            "status": "OK",
            "error": "",
        }
        for key, alias in (
            ("txt", "txt_path"),
            ("json", "json_path"),
            ("html", "html_path"),
            ("pdf", "pdf_path"),
            ("video_burned", "burned_video_path"),
            ("subtitle_srt", "subtitle_srt_path"),
            ("subtitle_ass", "subtitle_ass_path"),
        ):
            value = output_paths.get(key)
            if value and self._path_exists(value):
                result[alias] = str(value)
        backend = output_paths.get("subtitle_backend")
        if backend:
            result["subtitle_backend"] = str(backend)
        subtitle_stream_codec = output_paths.get("subtitle_stream_codec")
        if subtitle_stream_codec:
            result["subtitle_stream_codec"] = str(subtitle_stream_codec)
        if output_paths.get("subtitle_stream_embedded"):
            result["subtitle_stream_embedded"] = self._coerce_bool(
                output_paths.get("subtitle_stream_embedded")
            )
        if self.persist_segments:
            result["segments"] = self._deserialize_segments(checkpoint.get("segments", []))
            if not result["num_segments"]:
                result["num_segments"] = len(result["segments"])
        return result

    def _emit_progress(self, event: str, **payload):
        if not self.progress_callback:
            return
        try:
            self.progress_callback(event, payload)
        except Exception as e:
            logger.debug(f"Progress callback failed ({event}): {e}")

    def _wait_if_paused(self):
        if self.pause_event is None:
            return

        announced = False
        while self.pause_event.is_set() and not self.shutdown.should_stop:
            if not announced:
                announced = True
                logger.info("Paused. Waiting to resume...")
                self._emit_progress("paused")
            time.sleep(0.2)

        if announced:
            logger.info("Resumed.")
            self._emit_progress("resumed")

    def request_shutdown(self):
        self.shutdown.request_shutdown()

    def run(
        self,
        media_files: Optional[List[Path]] = None,
        input_dir: Optional[Path] = None,
    ):
        if input_dir is None:
            input_dir = Path(self.config["paths"]["input_dir"])
        else:
            input_dir = Path(input_dir)

        if media_files is None:
            media_files = scan_media_files(str(input_dir))
        else:
            media_files = [Path(p) for p in media_files]

        if not media_files:
            logger.warning(
                f"No media files in {input_dir}\n"
                f"  Put .mp4 .mkv .mov .avi .mp3 .wav .flac etc. in: {input_dir}"
            )
            self._emit_progress("run_complete", total=0, processed=0, ok=0, elapsed=0.0)
            return

        total_files = len(media_files)
        logger.info(f"Pipeline: {total_files} file(s)")
        self._emit_progress("run_start", total=total_files)
        t0 = time.time()

        progress_stream = getattr(sys, "stderr", None) or getattr(sys, "stdout", None)
        pbar = tqdm(
            media_files,
            desc="Processing",
            unit="file",
            disable=not self.config["logging"].get("show_progress", True),
            ascii=not _stream_supports_text(progress_stream, "█⚡"),
        )

        for idx, fp in enumerate(pbar, 1):
            self._wait_if_paused()
            if self.shutdown.should_stop:
                logger.warning("Shutdown requested.")
                break

            postfix_text = fp.name[:30]
            try:
                pbar.set_postfix_str(postfix_text, refresh=True)
            except UnicodeEncodeError:
                pbar.set_postfix_str(
                    _stream_safe_text(getattr(pbar, "fp", progress_stream), postfix_text),
                    refresh=True,
                )
            self._emit_progress(
                "file_start",
                index=idx,
                total=total_files,
                file_path=str(fp),
                overall_percent=round(((idx - 1) / max(1, total_files)) * 100.0, 2),
            )

            resumed = self._load_completed_result(fp)
            if resumed is not None:
                logger.info(
                    f"Resuming from checkpoint: {fp.name} is already complete, skipping."
                )
                result = resumed
            else:
                result = self._process_file(
                    fp,
                    input_dir,
                    file_index=idx,
                    total_files=total_files,
                )
            self.all_results.append(result)

            try:
                self.formatter.write_summary(self.all_results)
            except Exception as e:
                logger.debug(f"Failed to write rolling summary: {e}")

            self._emit_progress(
                "file_done",
                index=idx,
                total=total_files,
                result=result,
                overall_percent=round((idx / max(1, total_files)) * 100.0, 2),
            )

        try:
            self.formatter.write_summary(self.all_results)
        except Exception as e:
            logger.error(f"Failed to write summary: {e}")

        elapsed = time.time() - t0
        total_audio = sum(r.get("duration", 0.0) for r in self.all_results)
        ok = sum(1 for r in self.all_results if r.get("status") == "OK")

        logger.info("=" * 60)
        logger.info(f"Complete! {ok}/{total_files} OK")
        logger.info(f"  Audio: {human_readable_duration(total_audio)}")
        logger.info(f"  Time:  {human_readable_duration(elapsed)}")
        if total_audio > 0:
            logger.info(f"  RTF:   {elapsed / total_audio:.3f}")
        logger.info("=" * 60)

        self._emit_progress(
            "run_complete",
            total=total_files,
            processed=len(self.all_results),
            ok=ok,
            elapsed=elapsed,
            total_audio=total_audio,
        )

        self._cleanup()

    def _process_file(
        self,
        file_path: Path,
        input_dir: Path,
        file_index: int = 1,
        total_files: int = 1,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "source_file": file_path.name,
            "source_path": str(file_path),
            "duration": 0.0,
            "num_segments": 0,
            "diarization_route": "",
            "segments": [],
            "elapsed": 0.0,
            "status": "OK",
            "error": "",
        }
        t0 = time.time()
        llm_job_id = ""
        current_step = "start"
        analysis = None
        speaker_arbitration_meta: Dict[str, Any] = {}
        file_output_dir = resolve_output_subdir(
            output_root=self.output_dir,
            source_file=file_path,
            input_dir=input_dir,
        )
        file_temp_dir = resolve_output_temp_dir(
            output_root=self.output_dir,
            source_file=file_path,
            input_dir=input_dir,
        )
        self.transcriber.set_runtime_temp_dir(file_temp_dir)
        stage_rank = {
            "start": 0,
            "extract": 1,
            "preprocess": 2,
            "transcribe": 3,
            "transcribe_done": 4,
            "translate": 5,
            "translate_done": 6,
            "optimize_language": 7,
            "optimize_language_done": 8,
            "write_text": 9,
            "write_text_done": 10,
            "report": 11,
            "report_done": 12,
            "render_video": 13,
            "render_video_done": 14,
            "done": 15,
        }
        checkpoint = self._load_checkpoint(file_path)
        resume_stage = str(checkpoint.get("stage", "") or "")
        resume_rank = stage_rank.get(resume_stage, 0)
        duration = self._safe_float(checkpoint.get("duration", 0.0), 0.0)
        sr = self._safe_int(checkpoint.get("sample_rate", 0), 0)
        speech_ratio = self._safe_float(checkpoint.get("speech_ratio", 0.0), 0.0)
        pre_detected_lang = str(checkpoint.get("pre_detected_language", "") or "")
        detected_lang = str(checkpoint.get("detected_language", "") or "")
        translated_language = str(checkpoint.get("translated_language", "") or "")
        translation_applied = bool(checkpoint.get("translation_applied", False))
        language_optimized = bool(checkpoint.get("language_optimized", False))
        speaker_languages = checkpoint.get("speaker_languages", {})
        if not isinstance(speaker_languages, dict):
            speaker_languages = {}
        output_paths: Dict[str, str] = {}
        cp_output_paths = checkpoint.get("output_paths", {})
        if isinstance(cp_output_paths, dict):
            output_paths = {
                str(k): str(v)
                for k, v in cp_output_paths.items()
                if k and v
            }
        overlay_enabled_for_file = self.video_overlay.is_enabled_for(file_path)
        is_audio_only_input = (
            file_path.suffix.lower() in AUDIO_EXTENSIONS and not overlay_enabled_for_file
        )
        translation_wanted = bool(self.config.get("translation.enabled", False))
        optimize_wanted = bool(self.config.get("llm.optimize_language.enabled", False))
        step_weights: Dict[str, float] = {
            "extract": 0.10,
            "preprocess": 0.08,
            "transcribe": 0.44,
            "translate": 0.10,
            "optimize_language": 0.08,
            "write_text": 0.09,
            "report": 0.08,
            "render_video": 0.03,
        }
        planned_steps: List[str] = ["extract", "preprocess", "transcribe"]
        if translation_wanted:
            planned_steps.append("translate")
        if optimize_wanted:
            planned_steps.append("optimize_language")
        planned_steps.extend(["write_text", "report"])
        if overlay_enabled_for_file:
            planned_steps.append("render_video")
        total_step_weight = sum(step_weights.get(step, 0.0) for step in planned_steps) or 1.0
        step_before_weight: Dict[str, float] = {}
        rolling_weight = 0.0
        for step_name in planned_steps:
            step_before_weight[step_name] = rolling_weight
            rolling_weight += step_weights.get(step_name, 0.0)
        step_index_map = {name: idx + 1 for idx, name in enumerate(planned_steps)}

        if is_audio_only_input:
            logger.info(
                "  Audio-only input detected: subtitle burn-in stage is disabled for %s.",
                file_path.name,
            )

        def _emit_step_progress(
            step: str,
            phase: str = "start",
            within_step: Optional[float] = None,
            **extra_payload: Any,
        ):
            if step not in step_index_map:
                return
            before = step_before_weight.get(step, 0.0)
            span = step_weights.get(step, 0.0)
            if within_step is None:
                if phase in {"done", "resume", "skip"}:
                    file_ratio = (before + span) / total_step_weight
                else:
                    file_ratio = before / total_step_weight
            else:
                t = max(0.0, min(1.0, float(within_step)))
                file_ratio = (before + span * t) / total_step_weight

            file_percent = max(0.0, min(100.0, file_ratio * 100.0))
            total_safe = max(1, int(total_files))
            file_index_safe = max(1, int(file_index))
            overall_ratio = (file_index_safe - 1 + file_ratio) / total_safe
            overall_percent = max(0.0, min(100.0, overall_ratio * 100.0))
            self._emit_progress(
                "file_step",
                file_path=str(file_path),
                step=step,
                phase=phase,
                step_index=step_index_map.get(step, 0),
                step_total=len(planned_steps),
                file_percent=round(file_percent, 2),
                overall_percent=round(overall_percent, 2),
                within_step=(
                    max(0.0, min(1.0, float(within_step)))
                    if within_step is not None
                    else None
                ),
                **extra_payload,
            )

        segments: List[TranscriptionSegment] = []

        try:
            logger.info("-" * 60)
            logger.info(
                f"Processing: {file_path.name} "
                f"({human_readable_size(file_path.stat().st_size)})"
            )
            if checkpoint:
                logger.info(
                    f"  Resume checkpoint detected: stage={resume_stage or 'unknown'}"
                )

            self._write_checkpoint(
                file_path,
                {
                    "status": "processing",
                    "stage": "start",
                    "result": self._result_snapshot(result),
                },
            )

            if resume_rank >= stage_rank["transcribe_done"]:
                segments = self._deserialize_segments(checkpoint.get("segments", []))
                if segments:
                    logger.info(
                        f"  Resume: loaded {len(segments)} segments from checkpoint."
                    )
                    if duration <= 0:
                        duration = max((float(s.end) for s in segments), default=0.0)
                    _emit_step_progress("extract", "resume")
                    _emit_step_progress("preprocess", "resume")
                    _emit_step_progress("transcribe", "resume")
                else:
                    logger.warning(
                        "  Resume checkpoint has no segments, restarting transcription."
                    )
                    resume_rank = 0

            if not segments:
                self._wait_if_paused()
                if self.shutdown.should_stop:
                    raise KeyboardInterrupt
                if duration <= 0:
                    duration = get_media_duration(file_path)
                if duration > 0:
                    logger.info(f"  Duration: {human_readable_duration(duration)}")
                else:
                    logger.warning(
                        "  Duration unavailable from ffprobe, "
                        "will fallback to decoded audio length."
                    )

                current_step = "extract"
                logger.info("  [1/6] Extracting audio...")
                _emit_step_progress("extract", "start")
                self._write_checkpoint(file_path, {"status": "processing", "stage": current_step})
                audio_np, sr = self.audio_extractor.extract(file_path)
                decoded_duration = (
                    (len(audio_np) / sr)
                    if sr and sr > 0 and audio_np is not None
                    else 0.0
                )
                if duration <= 0 and decoded_duration > 0:
                    duration = decoded_duration
                    logger.info(
                        "  Duration fallback from decoded audio: "
                        f"{human_readable_duration(duration)}"
                    )
                result["duration"] = duration
                _emit_step_progress("extract", "done")

                current_step = "preprocess"
                logger.info("  [2/6] Preprocessing (normalize + VAD)...")
                self._wait_if_paused()
                if self.shutdown.should_stop:
                    raise KeyboardInterrupt
                _emit_step_progress("preprocess", "start")
                self._write_checkpoint(file_path, {"status": "processing", "stage": current_step})
                audio_np = normalize_audio_fast(audio_np)
                rms = compute_rms_energy_fast(audio_np, frame_size=512, hop_size=256)
                threshold = max(np.percentile(rms, 15), 0.005) if len(rms) > 0 else 0.005
                vad_mask = compute_vad_mask_fast(rms, threshold=threshold)
                speech_ratio = np.mean(vad_mask) if len(vad_mask) > 0 else 0.0
                logger.info(f"  VAD: {speech_ratio * 100:.1f}% speech")
                _emit_step_progress("preprocess", "done")

                current_step = "transcribe"
                logger.info("  [3/6] Transcribing...")
                self._wait_if_paused()
                if self.shutdown.should_stop:
                    raise KeyboardInterrupt
                _emit_step_progress("transcribe", "start")
                self._write_checkpoint(file_path, {"status": "processing", "stage": current_step})
                audio_cfg = self.config["audio"]
                audio_len = len(audio_np) / sr
                chunk_overlap = max(
                    0.0,
                    self._safe_float(audio_cfg.get("overlap_sec", 1.0), 1.0),
                )
                fixed_chunk_default = self._safe_float(
                    audio_cfg.get("auto_chunk_target_sec", 240),
                    240.0,
                )
                chunk_dur = choose_chunk_duration_sec(
                    audio_duration_sec=audio_len,
                    chunk_mode=str(audio_cfg.get("chunk_mode", "auto") or "auto"),
                    fixed_chunk_sec=self._safe_float(
                        audio_cfg.get("chunk_duration_sec", fixed_chunk_default),
                        fixed_chunk_default,
                    ),
                    min_chunk_sec=self._safe_float(
                        audio_cfg.get("auto_chunk_min_sec", 90),
                        90.0,
                    ),
                    max_chunk_sec=self._safe_float(
                        audio_cfg.get("auto_chunk_max_sec", 420),
                        420.0,
                    ),
                    target_chunk_sec=self._safe_float(
                        audio_cfg.get("auto_chunk_target_sec", 240),
                        240.0,
                    ),
                )
                logger.info(
                    f"  Chunk plan: {chunk_dur:.1f}s target, overlap={chunk_overlap:.2f}s"
                )
                fallback_lang = self.config.get("language.fallback_language", "zh")
                use_nemo_msdd = self.transcriber.use_nemo_msdd_pipeline()
                speaker_languages = {}

                pre_detected_lang = self.transcriber.detect_audio_language(
                    audio_np,
                    sample_rate=sr,
                    file_name=file_path.name,
                )
                pre_detect_meta = dict(
                    getattr(self.transcriber, "last_language_probe", {}) or {}
                )
                pre_detected_prob = self._safe_float(
                    pre_detect_meta.get("probability", 0.0), 0.0
                )
                detected_lang = pre_detected_lang
                logger.info(
                    f"  Pre-detected language: {pre_detected_lang} "
                    f"(p={pre_detected_prob:.2f})"
                )

                if self.config.get("asr.engine", "auto") == "auto":
                    min_switch_prob = self._safe_float(
                        self.config.get(
                            "language.probe_engine_switch_min_probability",
                            0.55,
                        ),
                        0.55,
                    )
                    if pre_detected_prob >= min_switch_prob:
                        switched = self.transcriber.ensure_engine_for_language(detected_lang)
                        if switched:
                            logger.info(
                                f"  ASR engine selected for {detected_lang}: "
                                f"{self.transcriber.engine_name}"
                            )
                    else:
                        logger.info(
                            "  Skip language-based engine switch due low "
                            f"pre-detect confidence (p={pre_detected_prob:.2f} "
                            f"< {min_switch_prob:.2f})."
                        )

                def run_transcription() -> List[TranscriptionSegment]:
                    should_chunk = audio_len > (chunk_dur + max(3.0, chunk_overlap))
                    if should_chunk:
                        chunks = split_audio_chunks_fast(
                            audio_np,
                            sr,
                            chunk_duration_sec=chunk_dur,
                            overlap_sec=chunk_overlap,
                            vad_mask=vad_mask,
                            frame_hop_samples=256,
                        )
                        total_chunks = max(1, len(chunks))

                        def _on_chunk_progress(chunk_payload: Dict[str, Any]) -> None:
                            try:
                                chunk_index = int(chunk_payload.get("chunk_index", 0) or 0)
                            except (TypeError, ValueError):
                                chunk_index = 0
                            try:
                                chunk_total = int(chunk_payload.get("chunk_total", total_chunks) or total_chunks)
                            except (TypeError, ValueError):
                                chunk_total = total_chunks
                            chunk_total = max(1, chunk_total)
                            phase = str(chunk_payload.get("phase", "progress") or "progress").strip().lower()
                            error_text = str(chunk_payload.get("error", "") or "").strip()

                            if phase == "start":
                                ratio = (max(1, chunk_index) - 1) / chunk_total
                                phase_name = "progress"
                            elif phase in {"done", "failed"}:
                                ratio = max(1, chunk_index) / chunk_total
                                phase_name = "failed" if phase == "failed" else "progress"
                            else:
                                ratio = max(1, chunk_index) / chunk_total
                                phase_name = "progress"

                            ratio = max(0.0, min(0.995, float(ratio)))
                            detail = (
                                (error_text[:72] if error_text else f"{chunk_index}/{chunk_total}")
                                if phase_name == "failed"
                                else f"{chunk_index}/{chunk_total}"
                            )
                            _emit_step_progress(
                                "transcribe",
                                phase=phase_name,
                                within_step=ratio,
                                chunk_index=chunk_index,
                                chunk_total=chunk_total,
                                phase_detail=detail,
                            )

                        chunk_segments = self.transcriber.transcribe_chunks(
                            chunks,
                            sample_rate=sr,
                            file_name=file_path.name,
                            file_path=str(file_path),
                            map_speakers=False,
                            overlap_sec=chunk_overlap,
                            chunk_progress_callback=_on_chunk_progress,
                        )

                        seg_dicts = [s.to_dict() for s in chunk_segments]
                        seg_dicts = deduplicate_overlap_text(
                            seg_dicts,
                            chunk_overlap,
                            same_speaker_only=True,
                            keep_cross_speaker_overlap=True,
                        )
                        seg_dicts = smooth_segment_boundaries(
                            seg_dicts,
                            max_gap_sec=max(0.2, min(0.8, chunk_overlap + 0.25)),
                            max_overlap_sec=max(0.5, chunk_overlap + 0.8),
                        )
                        merged_segments = [TranscriptionSegment(**d) for d in seg_dicts]
                        if merged_segments:
                            merged_segments = self.transcriber.assign_speakers(
                                audio_np,
                                sample_rate=sr,
                                segments=merged_segments,
                                file_name=file_path.name,
                                map_speakers=True,
                            )
                        return merged_segments
                    logger.info("  Chunk plan resolved to single-pass transcription.")
                    return self.transcriber.transcribe(
                        audio_np,
                        sample_rate=sr,
                        file_name=file_path.name,
                        language_override=detected_lang,
                    )

                segments = run_transcription()
                result["diarization_route"] = str(
                    getattr(self.transcriber, "last_diarization_route", "") or ""
                )

                if segments:
                    speaker_languages = dict(self.transcriber.last_speaker_languages)
                    seg_lang_scores: Dict[str, float] = {}
                    for seg in segments:
                        seg_lang = self.transcriber._normalize_language_tag(
                            str(getattr(seg, "language", "") or "")
                        )
                        if not seg_lang:
                            continue
                        seg_duration = max(
                            0.2, float(getattr(seg, "end", 0.0)) - float(getattr(seg, "start", 0.0))
                        )
                        seg_lang_scores[seg_lang] = (
                            seg_lang_scores.get(seg_lang, 0.0) + seg_duration
                        )
                    dominant_seg_lang = ""
                    dominant_seg_ratio = 0.0
                    if seg_lang_scores:
                        dominant_seg_lang, dominant_seg_score = max(
                            seg_lang_scores.items(),
                            key=lambda item: item[1],
                        )
                        total_seg_score = sum(seg_lang_scores.values())
                        if total_seg_score > 0:
                            dominant_seg_ratio = dominant_seg_score / total_seg_score

                    reconcile_min_ratio = self._safe_float(
                        self.config.get("language.segment_reconcile_min_ratio", 0.65),
                        0.65,
                    )
                    reconcile_max_probe_prob = self._safe_float(
                        self.config.get(
                            "language.segment_reconcile_pre_detect_max_probability",
                            0.70,
                        ),
                        0.70,
                    )
                    normalized_detected_lang = self.transcriber._normalize_language_tag(
                        detected_lang
                    )
                    if (
                        dominant_seg_lang
                        and (
                            not normalized_detected_lang
                            or normalized_detected_lang in ("unknown", fallback_lang)
                            or (
                                dominant_seg_lang != normalized_detected_lang
                                and pre_detected_prob <= reconcile_max_probe_prob
                                and dominant_seg_ratio >= reconcile_min_ratio
                            )
                        )
                    ):
                        if dominant_seg_lang != normalized_detected_lang:
                            logger.info(
                                "  Language reconciled by ASR segments: "
                                f"{normalized_detected_lang or 'unknown'} -> "
                                f"{dominant_seg_lang} "
                                f"(ratio={dominant_seg_ratio * 100:.0f}%, "
                                f"pre-detect p={pre_detected_prob:.2f})"
                            )
                        detected_lang = dominant_seg_lang

                    sample = " ".join(s.text for s in segments[:20])
                    text_lang = self.transcriber._normalize_language_tag(
                        self.llm.detect_language_sync(sample)
                    )
                    if text_lang and text_lang != "unknown" and (
                        detected_lang in ("unknown", "", fallback_lang)
                        or not detected_lang
                        or pre_detected_prob < 0.50
                    ):
                        if text_lang != detected_lang:
                            logger.info(
                                f"  Language refined by transcript text: "
                                f"{detected_lang or 'unknown'} -> {text_lang}"
                            )
                        detected_lang = text_lang
                    if speaker_languages:
                        logger.info(
                            f"  Source language: {detected_lang} "
                            f"(speaker map: {speaker_languages})"
                        )
                    else:
                        logger.info(f"  Source language: {detected_lang}")
                    for seg in segments:
                        if not seg.language:
                            seg.language = detected_lang
                elif use_nemo_msdd:
                    logger.info("  NeMo MSDD enabled but no ASR segments were produced.")

                if duration <= 0 and segments:
                    seg_duration = max((float(s.end) for s in segments), default=0.0)
                    if seg_duration > 0:
                        duration = seg_duration
                        result["duration"] = duration
                        logger.info(
                            "  Duration fallback from segments: "
                            f"{human_readable_duration(duration)}"
                        )

                self._write_checkpoint(
                    file_path,
                    {
                        "status": "processing",
                        "stage": "transcribe_done",
                        "duration": duration,
                        "sample_rate": sr,
                        "speech_ratio": float(speech_ratio),
                        "pre_detected_language": pre_detected_lang,
                        "detected_language": detected_lang,
                        "speaker_languages": speaker_languages,
                        "num_segments": len(segments),
                        "segments": self._serialize_segments(segments),
                    },
                )
                _emit_step_progress("transcribe", "done")

            fallback_lang = self.config.get("language.fallback_language", "zh")
            if not pre_detected_lang:
                pre_detected_lang = detected_lang or fallback_lang
            if not detected_lang:
                detected_lang = pre_detected_lang or fallback_lang

            source_detected_lang = detected_lang
            translation_target = str(
                self.config.get("translation.target_language", "zh") or "zh"
            ).strip().lower()
            if segments and translation_wanted:
                if resume_rank >= stage_rank["translate_done"]:
                    logger.info(
                        "  [3.5/6] Resume: translation stage already completed, skipping."
                    )
                    translated_language = str(
                        checkpoint.get("translated_language", translated_language) or ""
                    )
                    translation_applied = bool(
                        checkpoint.get("translation_applied", translation_applied)
                    )
                    if translated_language:
                        detected_lang = translated_language
                    _emit_step_progress("translate", "resume")
                else:
                    current_step = "translate"
                    logger.info("  [3.5/6] Translating transcript segments...")
                    self._wait_if_paused()
                    if self.shutdown.should_stop:
                        raise KeyboardInterrupt
                    _emit_step_progress("translate", "start")
                    self._write_checkpoint(
                        file_path,
                        {"status": "processing", "stage": current_step},
                    )
                    translation_result = self.llm.translate_segments_sync(
                        segments=segments,
                        source_language=source_detected_lang,
                        target_language=translation_target,
                        timeout=300.0,
                        pause_checker=self._wait_if_paused,
                        cancel_checker=lambda: self.shutdown.should_stop,
                    )
                    translated_language = str(
                        translation_result.get("target_language", "") or ""
                    ).strip().lower()
                    translation_applied = bool(translation_result.get("applied", False))
                    translation_reason = str(translation_result.get("reason", "") or "")
                    translated_count = int(
                        translation_result.get("translated_count", 0) or 0
                    )
                    skipped_count = int(
                        translation_result.get("skipped_count", 0) or 0
                    )
                    if translation_applied and translated_language:
                        detected_lang = translated_language
                    if translation_reason == "same_language":
                        logger.info(
                            "  Translation skipped: source language equals target language."
                        )
                    elif translation_reason == "translation_disabled":
                        logger.info("  Translation disabled by configuration.")
                    elif translation_reason not in {"ok", ""}:
                        logger.warning(f"  Translation not applied: {translation_reason}")
                    else:
                        logger.info(
                            f"  Translation complete: translated={translated_count}, "
                            f"unchanged={skipped_count}"
                        )
                    self._write_checkpoint(
                        file_path,
                        {
                            "status": "processing",
                            "stage": "translate_done",
                            "detected_language": detected_lang,
                            "translated_language": translated_language,
                            "translation_applied": translation_applied,
                            "num_segments": len(segments),
                            "segments": self._serialize_segments(segments),
                        },
                    )
                    _emit_step_progress("translate", "done")
            elif translation_wanted:
                _emit_step_progress("translate", "skip")

            if segments and optimize_wanted:
                if resume_rank >= stage_rank["optimize_language_done"]:
                    logger.info(
                        "  [3.8/6] Resume: language optimization already completed, skipping."
                    )
                    language_optimized = bool(
                        checkpoint.get("language_optimized", language_optimized)
                    )
                    _emit_step_progress("optimize_language", "resume")
                else:
                    current_step = "optimize_language"
                    logger.info("  [3.8/6] Optimizing transcript language...")
                    self._wait_if_paused()
                    if self.shutdown.should_stop:
                        raise KeyboardInterrupt
                    _emit_step_progress("optimize_language", "start")
                    self._write_checkpoint(
                        file_path,
                        {"status": "processing", "stage": current_step},
                    )
                    optimize_timeout = 360.0
                    optimize_texts = [
                        str(getattr(seg, "text", "") or "")
                        for seg in segments
                        if str(getattr(seg, "text", "") or "").strip()
                    ]
                    if hasattr(self.llm, "estimate_optimize_wait_timeout"):
                        try:
                            optimize_timeout = float(
                                self.llm.estimate_optimize_wait_timeout(optimize_texts)
                            )
                        except Exception:
                            optimize_timeout = 360.0
                    logger.info(
                        "  Language optimization timeout budget: %.1fs (segments=%d)",
                        optimize_timeout,
                        len(optimize_texts),
                    )
                    optimize_result = self.llm.optimize_segments_sync(
                        segments=segments,
                        source_language=detected_lang,
                        timeout=optimize_timeout,
                        pause_checker=self._wait_if_paused,
                        cancel_checker=lambda: self.shutdown.should_stop,
                    )
                    optimize_reason = str(optimize_result.get("reason", "") or "")
                    optimized_count = int(optimize_result.get("optimized_count", 0) or 0)
                    skipped_count = int(optimize_result.get("skipped_count", 0) or 0)
                    language_optimized = bool(
                        optimize_result.get("applied", False) and optimized_count > 0
                    )
                    if optimize_reason == "optimize_disabled":
                        logger.info("  Language optimization disabled by configuration.")
                    elif optimize_reason not in {"ok", ""}:
                        logger.warning(f"  Language optimization not applied: {optimize_reason}")
                    else:
                        logger.info(
                            "  Language optimization complete: optimized=%d, unchanged=%d",
                            optimized_count,
                            skipped_count,
                        )
                    self._write_checkpoint(
                        file_path,
                        {
                            "status": "processing",
                            "stage": "optimize_language_done",
                            "language_optimized": language_optimized,
                            "detected_language": detected_lang,
                            "translated_language": translated_language,
                            "translation_applied": translation_applied,
                            "num_segments": len(segments),
                            "segments": self._serialize_segments(segments),
                        },
                    )
                    _emit_step_progress("optimize_language", "done")
            elif optimize_wanted:
                _emit_step_progress("optimize_language", "skip")

            if segments:
                logger.info(
                    "  Final language: %s (source=%s, translated=%s, translation_applied=%s, optimized=%s)",
                    detected_lang,
                    source_detected_lang,
                    translated_language or "-",
                    translation_applied,
                    language_optimized,
                )

            analysis_wanted_for_report = bool(
                getattr(self.llm, "analysis_enabled", False) and segments
            )
            speaker_constraints = self._speaker_constraints_payload(
                allowed_speakers=sorted(
                    {
                        str(getattr(seg, "speaker", "") or "").strip()
                        for seg in segments
                        if str(getattr(seg, "speaker", "") or "").strip()
                    }
                ),
            )
            speaker_arbitration_wanted = bool(
                getattr(self.llm, "speaker_arbitration_enabled", False) and segments
            )
            if speaker_arbitration_wanted and int(speaker_constraints.get("num_speakers", 0) or 0) > 0:
                speaker_arbitration_wanted = False
                speaker_arbitration_meta = {
                    "applied": False,
                    "reason": "manual_speaker_count_locked",
                    "speaker_constraints": dict(speaker_constraints),
                }
                logger.info(
                    "  Skipping LLM speaker semantic arbitration because manual speaker count is locked: num=%d",
                    int(speaker_constraints.get("num_speakers", 0) or 0),
                )

            if analysis_wanted_for_report and analysis is None:
                logger.info("  Preparing LLM summary before output generation...")
                llm_job_id = ""
                llm_job_id = self.llm.submit_analysis(
                    segments,
                    file_id=file_path.stem,
                    response_language=detected_lang,
                )
                if llm_job_id and hasattr(self.llm, "describe_analysis_job"):
                    try:
                        job_meta = self.llm.describe_analysis_job(llm_job_id)
                    except Exception:
                        job_meta = {}
                    if job_meta:
                        logger.info(
                            "  LLM analysis plan: mode=%s, strategy=%s, chars=%d, chunks=%d",
                            str(job_meta.get("analysis_mode", "") or "pending"),
                            str(job_meta.get("request_strategy", "") or "single_pass"),
                            int(job_meta.get("total_chars", 0) or 0),
                            max(1, int(job_meta.get("total_chunks", 0) or 0)),
                        )
                if llm_job_id:
                    logger.info("  Waiting for LLM summary...")
                    llm_timeout = 120.0
                    total_chars = 0
                    try:
                        total_chars = sum(
                            len(str(getattr(seg, "text", "") or ""))
                            for seg in segments
                        )
                        if hasattr(self.llm, "estimate_analysis_wait_timeout"):
                            llm_timeout = float(
                                self.llm.estimate_analysis_wait_timeout(total_chars)
                            )
                        else:
                            llm_timeout += min(900.0, (total_chars / 20000.0) * 60.0)
                    except Exception:
                        pass
                    analysis = self.llm.wait_for_analysis(
                        llm_job_id,
                        timeout=llm_timeout,
                        pause_checker=self._wait_if_paused,
                        cancel_checker=lambda: self.shutdown.should_stop,
                    )
                    if analysis and not analysis.error:
                        meta = dict(getattr(analysis, "metadata", {}) or {})
                        logger.info(
                            "  LLM summary ready: mode=%s, strategy=%s, chunks=%s/%s, reduce=%s, summary=%d chars",
                            str(meta.get("mode", "") or "single_pass"),
                            str(meta.get("request_strategy", "") or "single_pass"),
                            int(meta.get("completed_chunks", 0) or 0),
                            max(1, int(meta.get("chunk_count", 0) or 1)),
                            str(meta.get("reduce_stage", "") or "single_pass"),
                            len(analysis.summary),
                        )
                    elif analysis and analysis.error:
                        logger.warning("  LLM summary incomplete: %s", analysis.error)
                    else:
                        logger.warning("  LLM summary timed out")

            if speaker_arbitration_wanted and not self._analysis_has_content(analysis):
                fallback_reason = str(
                    getattr(analysis, "error", "") or "speaker_arbitration_summary_missing"
                )
                try:
                    fallback_analysis = self.llm.build_report_fallback_analysis(
                        segments,
                        response_language=detected_lang,
                        reason=fallback_reason,
                    )
                except Exception as e:
                    fallback_analysis = None
                    logger.warning(f"  Local speaker-arbitration fallback analysis failed: {e}")
                if self._analysis_has_content(fallback_analysis):
                    logger.warning(
                        "  LLM analysis unavailable for speaker arbitration; using local fallback summary (%s).",
                        fallback_reason,
                    )
                    analysis = fallback_analysis

            if (
                speaker_arbitration_wanted
                and analysis is not None
                and self._analysis_has_content(analysis)
            ):
                logger.info(
                    "  Running final speaker semantic arbitration with constraints: mode=%s, num=%d, min=%d, max=%d",
                    str(speaker_constraints.get("mode", "") or "auto"),
                    int(speaker_constraints.get("num_speakers", 0) or 0),
                    int(speaker_constraints.get("min_speakers", 0) or 0),
                    int(speaker_constraints.get("max_speakers", 0) or 0),
                )
                speaker_arbitration_meta = self.llm.semantic_arbitrate_segments_sync(
                    segments=segments,
                    analysis=analysis,
                    response_language=detected_lang,
                    speaker_constraints=speaker_constraints,
                    timeout=420.0,
                    pause_checker=self._wait_if_paused,
                    cancel_checker=lambda: self.shutdown.should_stop,
                )
                changed_segments = int(
                    speaker_arbitration_meta.get("changed_segments", 0) or 0
                )
                if speaker_arbitration_meta.get("applied", False):
                    merged_group_count = int(
                        speaker_arbitration_meta.get("merged_group_count", 0) or 0
                    )
                    logger.info(
                        "  Speaker semantic arbitration applied: changed=%d, merged_groups=%d, profiles=%d",
                        changed_segments,
                        merged_group_count,
                        len(
                            dict(
                                speaker_arbitration_meta.get("speaker_profiles", {}) or {}
                            )
                        ),
                    )
                    speaker_languages = {
                        str(getattr(seg, "speaker", "") or "").strip(): str(
                            getattr(seg, "language", "") or ""
                        ).strip()
                        for seg in segments
                        if str(getattr(seg, "speaker", "") or "").strip()
                    }
                else:
                    logger.info(
                        "  Speaker semantic arbitration skipped: %s",
                        str(speaker_arbitration_meta.get("reason", "") or "not_applied"),
                    )
                try:
                    analysis.metadata = dict(getattr(analysis, "metadata", {}) or {})
                    analysis.metadata["speaker_arbitration"] = {
                        "applied": bool(speaker_arbitration_meta.get("applied", False)),
                        "changed_segments": changed_segments,
                        "merged_group_count": int(
                            speaker_arbitration_meta.get("merged_group_count", 0) or 0
                        ),
                        "merged_segments_removed": int(
                            speaker_arbitration_meta.get("merged_segments_removed", 0) or 0
                        ),
                        "reason": str(speaker_arbitration_meta.get("reason", "") or ""),
                        "speaker_profiles": dict(
                            speaker_arbitration_meta.get("speaker_profiles", {}) or {}
                        ),
                        "speaker_constraints": dict(
                            speaker_arbitration_meta.get("speaker_constraints", {}) or {}
                        ),
                    }
                except Exception:
                    pass

            result["num_segments"] = len(segments)
            result["segments"] = segments

            current_step = "write_text"
            reuse_text_outputs = (
                resume_rank >= stage_rank["write_text_done"]
                and self._checkpoint_outputs_exist(output_paths, require_report=False)
            )
            if reuse_text_outputs:
                logger.info("  [4/6] Resume: TXT + JSON already exists, skipping.")
                if output_paths.get("txt"):
                    result["txt_path"] = str(output_paths["txt"])
                if output_paths.get("json"):
                    result["json_path"] = str(output_paths["json"])
                _emit_step_progress("write_text", "resume")
            else:
                logger.info("  [4/6] Writing TXT + JSON...")
                self._wait_if_paused()
                if self.shutdown.should_stop:
                    raise KeyboardInterrupt
                _emit_step_progress("write_text", "start")
                self._write_checkpoint(file_path, {"status": "processing", "stage": current_step})
                out_paths = self.formatter.write_results(
                    segments=segments,
                    source_file=file_path,
                    input_dir=input_dir,
                    duration=duration,
                    output_dir=file_output_dir,
                    metadata={
                        "engine": self.transcriber.engine_name,
                        "sample_rate": sr,
                        "speech_ratio": round(speech_ratio, 3),
                        "language": detected_lang,
                        "source_language": source_detected_lang,
                        "translation_applied": translation_applied,
                        "translated_language": translated_language,
                        "language_optimized": language_optimized,
                        "pre_detected_language": pre_detected_lang,
                        "diarization_route": result.get("diarization_route", ""),
                        "speaker_languages": speaker_languages,
                        "speaker_constraints": speaker_constraints,
                        "speaker_arbitration": speaker_arbitration_meta,
                        "duration_sec": round(duration, 3),
                    },
                )
                for key, value in out_paths.items():
                    output_paths[str(key)] = str(value)
                    result[f"{key}_path"] = str(value)
                self._write_checkpoint(
                    file_path,
                    {
                        "status": "processing",
                        "stage": "write_text_done",
                        "segments": self._serialize_segments(segments),
                        "speaker_languages": speaker_languages,
                        "output_paths": output_paths,
                        "result": self._result_snapshot(result),
                    },
                )
                _emit_step_progress("write_text", "done")

            current_step = "report"
            analysis_wanted_for_report = bool(
                getattr(self.llm, "analysis_enabled", False) and segments
            )
            reuse_report_outputs = (
                resume_rank >= stage_rank["report_done"]
                and self._checkpoint_outputs_exist(output_paths, require_report=True)
                and self._report_resume_compatible(
                    checkpoint,
                    want_analysis=analysis_wanted_for_report,
                    file_name=file_path.name,
                )
            )
            if reuse_report_outputs:
                logger.info("  [5/6] Resume: HTML/PDF report already exists, skipping.")
                if output_paths.get("html"):
                    result["html_path"] = str(output_paths["html"])
                if output_paths.get("pdf"):
                    result["pdf_path"] = str(output_paths["pdf"])
                _emit_step_progress("report", "resume")
            else:
                logger.info("  [5/6] Generating HTML/PDF report...")
                self._wait_if_paused()
                if self.shutdown.should_stop:
                    raise KeyboardInterrupt
                _emit_step_progress("report", "start")
                self._write_checkpoint(file_path, {"status": "processing", "stage": current_step})
                if analysis is None and getattr(self.llm, "analysis_enabled", False) and segments:
                    llm_job_id = ""
                    logger.info("  [3.5] LLM analysis not cached yet, submitting for report...")
                    llm_job_id = self.llm.submit_analysis(
                        segments,
                        file_id=file_path.stem,
                        response_language=detected_lang,
                    )
                    if llm_job_id and hasattr(self.llm, "describe_analysis_job"):
                        try:
                            job_meta = self.llm.describe_analysis_job(llm_job_id)
                        except Exception:
                            job_meta = {}
                        if job_meta:
                            logger.info(
                                "  LLM analysis plan: mode=%s, strategy=%s, chars=%d, chunks=%d",
                                str(job_meta.get("analysis_mode", "") or "pending"),
                                str(job_meta.get("request_strategy", "") or "single_pass"),
                                int(job_meta.get("total_chars", 0) or 0),
                                max(1, int(job_meta.get("total_chunks", 0) or 0)),
                            )
                            chunk_sizes = list(job_meta.get("chunk_sizes", []) or [])
                            if chunk_sizes:
                                logger.info(
                                    "  LLM chunk plan chars: %s",
                                    ", ".join(
                                        f"{index + 1}:{size}"
                                        for index, size in enumerate(chunk_sizes)
                                    ),
                                )
                if analysis is None and llm_job_id:
                    logger.info("  Waiting for LLM analysis...")
                    llm_timeout = 120.0
                    total_chars = 0
                    try:
                        total_chars = sum(
                            len(str(getattr(seg, "text", "") or ""))
                            for seg in segments
                        )
                        if hasattr(self.llm, "estimate_analysis_wait_timeout"):
                            llm_timeout = float(
                                self.llm.estimate_analysis_wait_timeout(total_chars)
                            )
                        else:
                            llm_timeout += min(900.0, (total_chars / 20000.0) * 60.0)
                    except Exception:
                        pass
                    logger.debug(
                        "  LLM wait timeout budget: %.1fs (chars=%d)",
                        llm_timeout,
                        total_chars,
                    )
                    analysis = self.llm.wait_for_analysis(
                        llm_job_id,
                        timeout=llm_timeout,
                        pause_checker=self._wait_if_paused,
                        cancel_checker=lambda: self.shutdown.should_stop,
                    )
                    if analysis and not analysis.error:
                        meta = dict(getattr(analysis, "metadata", {}) or {})
                        logger.info(
                            "  LLM analysis complete: mode=%s, strategy=%s, chunks=%s/%s, reduce=%s, "
                            "summary=%d chars, points=%d, actions=%d",
                            str(meta.get("mode", "") or "single_pass"),
                            str(meta.get("request_strategy", "") or "single_pass"),
                            int(meta.get("completed_chunks", 0) or 0),
                            max(1, int(meta.get("chunk_count", 0) or 1)),
                            str(meta.get("reduce_stage", "") or "single_pass"),
                            len(analysis.summary),
                            len(analysis.key_points),
                            len(analysis.action_items),
                        )
                    elif analysis and analysis.error:
                        meta = dict(getattr(analysis, "metadata", {}) or {})
                        logger.warning(
                            "  LLM analysis incomplete: %s | mode=%s strategy=%s phase=%s chunks=%s/%s reduce_started=%s",
                            analysis.error,
                            str(meta.get("mode", "") or "unknown"),
                            str(meta.get("request_strategy", "") or "unknown"),
                            str(meta.get("phase", "") or meta.get("stage", "") or "unknown"),
                            int(meta.get("completed_chunks", 0) or 0),
                            max(1, int(meta.get("chunk_count", 0) or 1)),
                            bool(meta.get("reduce_started", False)),
                        )
                    else:
                        logger.warning("  LLM analysis timed out")

                if analysis_wanted_for_report and not self._analysis_has_content(analysis):
                    fallback_reason = str(
                        getattr(analysis, "error", "") or "llm_report_content_missing"
                    )
                    try:
                        fallback_analysis = self.llm.build_report_fallback_analysis(
                            segments,
                            response_language=detected_lang,
                            reason=fallback_reason,
                        )
                    except Exception as e:
                        fallback_analysis = None
                        logger.warning(f"  Local report fallback analysis failed: {e}")
                    if self._analysis_has_content(fallback_analysis):
                        logger.warning(
                            "  LLM analysis unavailable for report; using local fallback summary (%s).",
                            fallback_reason,
                        )
                        analysis = fallback_analysis

                report_has_analysis = self._analysis_has_content(analysis)
                report_paths = self.report_gen.generate(
                    segments=segments,
                    source_file=file_path,
                    input_dir=input_dir,
                    duration=duration,
                    output_dir=file_output_dir,
                    metadata={
                        "engine": self.transcriber.engine_name,
                        "language": detected_lang,
                        "source_language": source_detected_lang,
                        "translation_applied": translation_applied,
                        "translated_language": translated_language,
                        "language_optimized": language_optimized,
                        "pre_detected_language": pre_detected_lang,
                        "speaker_languages": speaker_languages,
                        "speaker_constraints": speaker_constraints,
                        "speaker_arbitration": speaker_arbitration_meta,
                        "duration_sec": round(duration, 3),
                    },
                    analysis=analysis,
                )
                for key, value in report_paths.items():
                    output_paths[str(key)] = str(value)
                    result[f"{key}_path"] = str(value)
                self._write_checkpoint(
                    file_path,
                    {
                        "status": "processing",
                        "stage": "report_done",
                        "report_has_analysis": report_has_analysis,
                        "segments": self._serialize_segments(segments),
                        "speaker_languages": speaker_languages,
                        "output_paths": output_paths,
                        "result": self._result_snapshot(result),
                    },
                )
                _emit_step_progress("report", "done")

            if overlay_enabled_for_file:
                current_step = "render_video"
                reuse_overlay_outputs = (
                    resume_rank >= stage_rank["render_video_done"]
                    and self._overlay_outputs_ready_for_resume(output_paths)
                )
                if reuse_overlay_outputs:
                    logger.info("  [6/7] Resume: subtitle video already exists, skipping.")
                    if output_paths.get("video_burned"):
                        result["burned_video_path"] = str(output_paths["video_burned"])
                    if output_paths.get("subtitle_srt"):
                        result["subtitle_srt_path"] = str(output_paths["subtitle_srt"])
                    if output_paths.get("subtitle_ass"):
                        result["subtitle_ass_path"] = str(output_paths["subtitle_ass"])
                    if output_paths.get("subtitle_backend"):
                        result["subtitle_backend"] = str(output_paths["subtitle_backend"])
                    if output_paths.get("subtitle_stream_codec"):
                        result["subtitle_stream_codec"] = str(output_paths["subtitle_stream_codec"])
                    if output_paths.get("subtitle_stream_embedded"):
                        result["subtitle_stream_embedded"] = self._coerce_bool(
                            output_paths["subtitle_stream_embedded"]
                        )
                    if result.get("subtitle_stream_embedded"):
                        logger.info(
                            "  [6/7] Embedded subtitle stream ready: codec=%s",
                            str(result.get("subtitle_stream_codec", "") or "unknown"),
                        )
                    _emit_step_progress("render_video", "resume")
                elif not segments:
                    logger.warning("  [6/7] No segments available, skip subtitle video burn-in.")
                    self._write_checkpoint(
                        file_path,
                        {
                            "status": "processing",
                            "stage": "render_video_done",
                            "output_paths": output_paths,
                            "result": self._result_snapshot(result),
                        },
                    )
                    _emit_step_progress("render_video", "skip")
                else:
                    logger.info("  [6/7] Rendering subtitle video...")
                    self._wait_if_paused()
                    if self.shutdown.should_stop:
                        raise KeyboardInterrupt
                    _emit_step_progress("render_video", "start")
                    self._write_checkpoint(file_path, {"status": "processing", "stage": current_step})

                    overlay_paths = self.video_overlay.render(
                        source_file=file_path,
                        segments=segments,
                        output_dir=file_output_dir,
                        temp_dir=file_temp_dir,
                    )
                    for key, value in overlay_paths.items():
                        output_paths[str(key)] = str(value)

                    if output_paths.get("video_burned"):
                        result["burned_video_path"] = str(output_paths["video_burned"])
                    if output_paths.get("subtitle_srt"):
                        result["subtitle_srt_path"] = str(output_paths["subtitle_srt"])
                    if output_paths.get("subtitle_ass"):
                        result["subtitle_ass_path"] = str(output_paths["subtitle_ass"])
                    if output_paths.get("subtitle_backend"):
                        result["subtitle_backend"] = str(output_paths["subtitle_backend"])
                    if output_paths.get("subtitle_stream_codec"):
                        result["subtitle_stream_codec"] = str(output_paths["subtitle_stream_codec"])
                    if output_paths.get("subtitle_stream_embedded"):
                        result["subtitle_stream_embedded"] = self._coerce_bool(
                            output_paths["subtitle_stream_embedded"]
                        )
                    if result.get("subtitle_stream_embedded"):
                        logger.info(
                            "  [6/7] Embedded subtitle stream written: codec=%s",
                            str(result.get("subtitle_stream_codec", "") or "unknown"),
                        )

                    self._write_checkpoint(
                        file_path,
                        {
                            "status": "processing",
                            "stage": "render_video_done",
                            "segments": self._serialize_segments(segments),
                            "speaker_languages": speaker_languages,
                            "output_paths": output_paths,
                            "result": self._result_snapshot(result),
                        },
                    )
                    _emit_step_progress("render_video", "done")

            result["detected_language"] = detected_lang
            result["pre_detected_language"] = pre_detected_lang
            result["translation_applied"] = translation_applied
            result["translated_language"] = translated_language
            result["language_optimized"] = language_optimized
            result["status"] = "OK"
            result["error"] = ""
            self._write_checkpoint(
                file_path,
                {
                    "status": "done",
                    "stage": "done",
                    "duration": duration,
                    "sample_rate": sr,
                    "speech_ratio": float(speech_ratio),
                    "pre_detected_language": pre_detected_lang,
                    "detected_language": detected_lang,
                    "translated_language": translated_language,
                    "translation_applied": translation_applied,
                    "language_optimized": language_optimized,
                    "speaker_languages": speaker_languages,
                    "num_segments": len(segments),
                    "segments": self._serialize_segments(segments),
                    "output_paths": output_paths,
                    "result": self._result_snapshot(result),
                },
            )

        except KeyboardInterrupt:
            result["detected_language"] = detected_lang
            result["pre_detected_language"] = pre_detected_lang
            result["translation_applied"] = translation_applied
            result["translated_language"] = translated_language
            result["language_optimized"] = language_optimized
            result["status"] = "INTERRUPTED"
            result["error"] = "User interrupted"
            self._write_checkpoint(
                file_path,
                {
                    "status": "interrupted",
                    "stage": current_step,
                    "duration": duration,
                    "sample_rate": sr,
                    "speech_ratio": float(speech_ratio),
                    "pre_detected_language": pre_detected_lang,
                    "detected_language": detected_lang,
                    "translated_language": translated_language,
                    "translation_applied": translation_applied,
                    "language_optimized": language_optimized,
                    "speaker_languages": speaker_languages,
                    "num_segments": len(segments),
                    "segments": self._serialize_segments(segments),
                    "output_paths": output_paths,
                    "result": self._result_snapshot(result),
                },
            )

        except Exception as e:
            result["detected_language"] = detected_lang
            result["pre_detected_language"] = pre_detected_lang
            result["translation_applied"] = translation_applied
            result["translated_language"] = translated_language
            result["language_optimized"] = language_optimized
            result["status"] = "ERROR"
            result["error"] = str(e)
            logger.error(f"Failed: {file_path.name}: {e}")
            logger.debug(traceback.format_exc())
            self._write_checkpoint(
                file_path,
                {
                    "status": "error",
                    "stage": current_step,
                    "duration": duration,
                    "sample_rate": sr,
                    "speech_ratio": float(speech_ratio),
                    "pre_detected_language": pre_detected_lang,
                    "detected_language": detected_lang,
                    "translated_language": translated_language,
                    "translation_applied": translation_applied,
                    "language_optimized": language_optimized,
                    "speaker_languages": speaker_languages,
                    "num_segments": len(segments),
                    "segments": self._serialize_segments(segments),
                    "output_paths": output_paths,
                    "result": self._result_snapshot(result),
                },
            )

        finally:
            result["num_segments"] = len(segments)
            result["segments"] = segments
            result["elapsed"] = time.time() - t0
            self.transcriber.set_runtime_temp_dir(None)
            if file_temp_dir.exists():
                shutil.rmtree(file_temp_dir, ignore_errors=True)
            logger.info(
                f"  DONE {file_path.name} | {result['status']} | "
                f"{result['num_segments']} segs | {result['elapsed']:.1f}s"
            )
            self._write_checkpoint(
                file_path,
                {
                    "stage": "done" if result.get("status") == "OK" else current_step,
                    "result": self._result_snapshot(result),
                },
            )
            cache_interval = max(
                0,
                self._safe_int(
                    self.config.get("performance.empty_cache_interval", 5),
                    5,
                ),
            )
            cache_threshold = self._safe_float(
                self.config.get("performance.cache_cleanup_threshold_pct", 96.0),
                96.0,
            )
            cache_threshold = max(50.0, min(99.5, cache_threshold))
            should_force_cache_cleanup = (
                cache_interval > 0 and (file_index % cache_interval == 0)
            )
            smart_empty_cache(
                force=should_force_cache_cleanup,
                threshold_pct=cache_threshold,
            )
            if should_force_cache_cleanup:
                logger.debug(
                    "Periodic runtime cache cleanup after file %d (interval=%d).",
                    file_index,
                    cache_interval,
                )
            gc.collect()

        return result

    def _cleanup(self):
        logger.info("Cleaning up...")
        try:
            self.transcriber.cleanup()
        except Exception:
            pass
        try:
            self.llm.stop()
        except Exception:
            pass
        self._save_session_state()
        smart_empty_cache(force=True)
        gc.collect()
        self.shutdown.restore()
