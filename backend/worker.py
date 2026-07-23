"""Command-line entry point for the explicit offline production worker.

The production process has one dependency graph, loaded from a strict JSON
configuration. It never falls back to unavailable development adapters. All
machine-readable output, including startup failures and diagnostic modes, is
emitted as one-object-per-line JSON.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from .composition import build_production_composition
from .documents import utc_now
from .errors import WorkerError
from .models import PROTOCOL_VERSION
from .production_config import (
    ProductionConfig,
    ProductionConfigError,
    apply_offline_environment,
    production_diagnostics,
    run_production_preflight,
)
from .protocol import JsonlEmitter, WorkerProtocol, run_jsonl_loop
from .runtime_preload import preload_production_runtime

_CONFIG_ENVIRONMENT_VARIABLE = "MTS_PRODUCTION_CONFIG"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MediaTranscribeStudio offline production JSONL worker"
    )
    parser.add_argument(
        "--config",
        help=(
            "strict production JSON configuration; defaults to "
            f"{_CONFIG_ENVIRONMENT_VARIABLE}"
        ),
    )
    parser.add_argument(
        "--input-root",
        action="append",
        help=(
            "override an allowed media input root; may be repeated and replaces "
            "the configured list"
        ),
    )
    parser.add_argument("--output-root", help="override the configured output root")
    parser.add_argument("--max-workers", type=int, help="override worker concurrency")
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument(
        "--preflight",
        action="store_true",
        help="validate local production dependencies and exit",
    )
    diagnostic.add_argument(
        "--diagnose",
        action="store_true",
        help="emit the path-free production topology and preflight report",
    )
    return parser


def _configure_streams() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="strict")


def _control_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    request_id: str = "startup",
) -> dict[str, Any]:
    return {
        "schemaVersion": PROTOCOL_VERSION,
        "requestId": request_id,
        "timestamp": utc_now(),
        "type": event_type,
        "payload": payload,
    }


def _resolve_config_path(argument: str | None) -> Path:
    value = argument or os.environ.get(_CONFIG_ENVIRONMENT_VARIABLE)
    if not value or not value.strip():
        raise ProductionConfigError(
            "production configuration is required",
            details={
                "component": "production-config",
                "acceptedSources": ["--config", _CONFIG_ENVIRONMENT_VARIABLE],
            },
        )
    return Path(value.strip()).expanduser()


def _load_effective_config(args: argparse.Namespace) -> ProductionConfig:
    config = ProductionConfig.load(_resolve_config_path(args.config))
    return config.with_runtime_overrides(
        allowed_input_roots=args.input_root,
        allowed_output_root=args.output_root,
        max_workers=args.max_workers,
    )


def _startup_error(error: BaseException) -> WorkerError:
    if isinstance(error, WorkerError):
        return error
    return WorkerError(
        "PRODUCTION_STARTUP_FAILED",
        "production worker failed closed during startup",
        details={"exceptionType": type(error).__name__},
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_streams()
    # Keep a private reference to the protocol stream, then route every
    # ordinary Python ``print`` in this process to stderr. Model runtimes such
    # as FunASR can emit banners lazily while a job is already running; without
    # this process-wide firewall a single banner corrupts the strict JSONL
    # transport. JsonlEmitter writes only to the captured protocol stream.
    emitter = JsonlEmitter(sys.stdout)
    with redirect_stdout(sys.stderr):
        try:
            config = _load_effective_config(args)
            apply_offline_environment()
            preflight = run_production_preflight(
                config,
                probe_runtime_imports=config.runtime.strict_startup_preflight,
            )

            if args.preflight:
                emitter.emit(
                    _control_event(
                        "worker.preflight.completed",
                        preflight.as_dict(),
                    )
                )
                return 0 if preflight.passed else 2

            if args.diagnose:
                emitter.emit(
                    _control_event(
                        "worker.diagnostics.completed",
                        production_diagnostics(config, preflight),
                    )
                )
                return 0 if preflight.passed else 2

            preflight.raise_if_failed()
            preload_production_runtime(
                include_pyannote=config.speaker.pyannote_mode != "disabled",
            )
            composition = build_production_composition(
                config,
                event_sink=emitter.emit,
                preflight_report=preflight,
                probe_runtime_imports=False,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            emitter.emit(
                _control_event(
                    "worker.startup.failed",
                    _startup_error(exc).as_payload(),
                )
            )
            return 2

        protocol = WorkerProtocol(
            composition.service,
            emitter,
            max_line_bytes=config.runtime.max_line_bytes,
        )
        run_jsonl_loop(
            input_stream=sys.stdin,
            protocol=protocol,
            service=composition.service,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
