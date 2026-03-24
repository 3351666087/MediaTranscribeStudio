#!/usr/bin/env python3
"""
main.py - Entry point for transcription pipeline (CLI + optional UI).
"""

import argparse
import importlib.metadata
import importlib.util
import logging
import os
import shutil
import sys
import ctypes
from pathlib import Path
from typing import Optional

from runtime_paths import APP_ROOT, DEFAULT_BUNDLED_RUNTIME_ENV

PROJECT_ROOT = APP_ROOT.resolve()
BUNDLED_RUNTIME_ENV = DEFAULT_BUNDLED_RUNTIME_ENV


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

from utils import (
    accelerator_display_name,
    mlx_is_available,
    mlx_status,
    mlx_whisper_is_available,
    mlx_whisper_status,
    mps_status,
    mps_unavailable_reason,
    smart_empty_cache,
)


def _make_stdio_encoding_safe():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def _hide_console_window_if_present():
    if os.name != "nt":
        return
    try:
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Media Transcription Pipeline",
    )
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument(
        "--engine",
        "-e",
        choices=[
            "auto",
            "funasr",
            "faster_whisper",
            "faster-whisper",
            "mlx_whisper",
            "mlx-whisper",
        ],
        default=None,
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--no-taichi", action="store_true", help="Disable Taichi")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM analysis")
    parser.add_argument("--no-pdf", action="store_true", help="Disable PDF generation")
    parser.add_argument("--no-html", action="store_true", help="Disable HTML generation")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--ui", action="store_true", help="Launch modern desktop UI")
    return parser.parse_args()


def print_banner():
    print("=" * 72)
    print(" Media Transcription Pipeline")
    print(" ASR + LLM Analysis + HTML/PDF Report + macOS-friendly runtime")
    print("=" * 72)


def check_environment():
    """Print environment diagnostics."""
    import torch

    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  Device:     {accelerator_display_name()}")
    if torch.cuda.is_available():
        print(f"  CUDA:       Yes (v{torch.version.cuda})")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  GPU [{i}]:    {props.name} ({mem_gb:.1f} GB)")
    else:
        backend = getattr(torch.backends, "mps", None)
        if backend is not None:
            built = bool(backend.is_built())
            raw_available = bool(backend.is_available())
            runtime_available, runtime_reason = mps_status()
            print(
                "  MPS:        "
                f"built={built}, available={raw_available}, runtime_ok={runtime_available}"
            )
            if built and not runtime_available:
                reason = runtime_reason or mps_unavailable_reason()
                if reason:
                    print(f"  MPS note:   {reason}")
        else:
            print("  MPS:        unavailable")
        mlx_core_ok, mlx_core_detail = mlx_status()
        mlx_whisper_ok, mlx_whisper_detail = mlx_whisper_status()
        print(
            "  MLX:        "
            f"core={mlx_core_ok} whisper={mlx_whisper_ok}"
        )
        if mlx_core_detail or mlx_whisper_detail:
            print(
                "  MLX note:   "
                f"core={mlx_core_detail or '-'} | whisper={mlx_whisper_detail or '-'}"
            )

    def _module_installed(module_name: str) -> bool:
        normalized = str(module_name or "").strip().lower().replace("-", "_")
        if normalized == "mlx":
            return mlx_is_available()
        if normalized == "mlx_whisper":
            return mlx_whisper_is_available()
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def _safe_dist_version(dist_name: str) -> str:
        try:
            return importlib.metadata.version(dist_name)
        except Exception:
            return ""

    libs = {
        "torchaudio": ("torchaudio", "torchaudio"),
        "funasr": ("funasr", "funasr"),
        "faster-whisper": ("faster_whisper", "faster-whisper"),
        "mlx-whisper": ("mlx_whisper", "mlx-whisper"),
        "mlx": ("mlx", "mlx"),
        "nemo-toolkit": ("nemo", "nemo-toolkit"),
        "pyannote.audio": ("pyannote.audio", "pyannote.audio"),
        "omegaconf": ("omegaconf", "omegaconf"),
        "playwright": ("playwright", "playwright"),
        "taichi": ("taichi", "taichi"),
        "openai": ("openai", "openai"),
        "pdfkit": ("pdfkit", "pdfkit"),
        "soundfile": ("soundfile", "soundfile"),
    }
    print()
    for name, (module_name, dist_name) in libs.items():
        installed = _module_installed(module_name)
        if not installed:
            print(f"  {name:16s}: NOT INSTALLED")
            continue
        version = _safe_dist_version(dist_name)
        print(f"  {name:16s}: {version or 'INSTALLED'}")

    wk = shutil.which("wkhtmltopdf")
    wk_display = wk if wk else "NOT FOUND (PDF may be disabled)"
    print(f"  {'wkhtmltopdf':16s}: {wk_display}")
    print()


def launch_ui(config_path: Optional[str]):
    try:
        from ui_app import launch_ui_app
    except ImportError as e:
        print("Failed to load UI module.")
        print("Install UI dependency first: pip install PySide6")
        print(f"Details: {e}")
        sys.exit(1)

    launch_ui_app(config_path=config_path)


def main():
    _make_stdio_encoding_safe()
    args = parse_args()

    # Default behavior: launch UI when main.py is started without CLI args.
    if args.ui or len(sys.argv) == 1:
        _hide_console_window_if_present()
        launch_ui(args.config)
        return

    print_banner()
    check_environment()

    from config import Config
    from pipeline import TranscriptionPipeline
    from utils import setup_logging

    import torch

    config = Config(config_path=args.config)

    if args.input:
        config._data["paths"]["input_dir"] = os.path.abspath(args.input)
    if args.output:
        config._data["paths"]["output_dir"] = os.path.abspath(args.output)
    if args.engine:
        config._data["asr"]["engine"] = args.engine.replace("-", "_")
    if args.dtype:
        config._data["performance"]["dtype"] = args.dtype
    if args.no_taichi:
        config._data["taichi"]["enabled"] = False
    if args.no_llm:
        llm_cfg = config._data.setdefault("llm", {})
        llm_cfg["enabled"] = False
        llm_cfg.setdefault("optimize_language", {})["enabled"] = False
        config._data.setdefault("translation", {})["enabled"] = False
    if args.no_pdf:
        config._data.setdefault("report", {})["generate_pdf"] = False
    if args.no_html:
        config._data.setdefault("report", {})["generate_html"] = False

    log_level = "DEBUG" if args.verbose else config.get("logging.level", "INFO")
    log_file = config.get("logging.log_file", "pipeline.log")
    setup_logging(level=log_level, log_file=log_file)

    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded")
    if BUNDLED_RUNTIME_ENV:
        logger.info(
            "  Bundled runtime env: %s",
            ", ".join(sorted(BUNDLED_RUNTIME_ENV.keys())),
        )
    logger.info(f"  Input:  {config['paths']['input_dir']}")
    logger.info(f"  Output: {config['paths']['output_dir']}")
    logger.info(f"  Engine: {config['asr']['engine']}")
    logger.info(f"  Device: {accelerator_display_name()}")
    startup_region = str(os.environ.get("MTS_DOWNLOAD_REGION", "") or "").strip()
    startup_route = str(os.environ.get("MTS_HF_PRIMARY_ENDPOINT", "") or "").strip()
    startup_note = str(os.environ.get("MTS_DOWNLOAD_REGION_NOTE", "") or "").strip()
    if startup_region:
        logger.info(
            "  Download route: region=%s, primary=%s",
            startup_region,
            startup_route or "<unset>",
        )
        if startup_note:
            logger.info("  Download route note: %s", startup_note)
    logger.info(f"  LLM:    {'ON' if config.get('llm.enabled', False) else 'OFF'}")
    logger.info(f"  HTML:   {'ON' if config.get('report.generate_html', True) else 'OFF'}")
    logger.info(f"  PDF:    {'ON' if config.get('report.generate_pdf', True) else 'OFF'}")

    try:
        pipeline = TranscriptionPipeline(config)
        pipeline.run()

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        smart_empty_cache(force=True)

    logger.info("All done.")


if __name__ == "__main__":
    main()
