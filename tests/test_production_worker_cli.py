from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend import ProductionConfigError
from backend.production_config import ProductionPreflightReport
from backend.worker import main


class FakeService:
    def shutdown(self, *, cancel: bool, wait: bool) -> None:
        del cancel, wait


class ProductionWorkerCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "production.json"
        self.config_path.write_text("{}\n", encoding="utf-8")
        self.config = SimpleNamespace(
            runtime=SimpleNamespace(
                strict_startup_preflight=True,
                max_line_bytes=8192,
            ),
        )
        self.config.with_runtime_overrides = lambda **_kwargs: self.config
        self.passed_report = ProductionPreflightReport(
            config_fingerprint="a" * 64,
            checks=(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def output_line(buffer: io.StringIO) -> dict:
        lines = [line for line in buffer.getvalue().splitlines() if line]
        if len(lines) != 1:
            raise AssertionError(f"expected one JSONL line, got {lines!r}")
        return json.loads(lines[0])

    def test_missing_config_fails_closed_without_path_disclosure(self) -> None:
        buffer = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), redirect_stdout(buffer):
            exit_code = main(["--preflight"])
        self.assertEqual(exit_code, 2)
        event = self.output_line(buffer)
        self.assertEqual(event["type"], "worker.startup.failed")
        self.assertEqual(
            event["payload"]["code"], "PRODUCTION_CONFIG_INVALID"
        )
        self.assertNotIn(str(self.root), json.dumps(event))

    def test_preflight_emits_one_jsonl_event_and_exits(self) -> None:
        buffer = io.StringIO()
        with (
            patch(
                "backend.worker.ProductionConfig.load",
                return_value=self.config,
            ),
            patch(
                "backend.worker.run_production_preflight",
                return_value=self.passed_report,
            ) as preflight,
            patch("backend.worker.apply_offline_environment"),
            redirect_stdout(buffer),
        ):
            exit_code = main(
                ["--config", str(self.config_path), "--preflight"]
            )
        self.assertEqual(exit_code, 0)
        event = self.output_line(buffer)
        self.assertEqual(event["type"], "worker.preflight.completed")
        self.assertEqual(event["payload"]["status"], "passed")
        preflight.assert_called_once_with(
            self.config,
            probe_runtime_imports=True,
        )

    def test_diagnostics_does_not_build_or_load_models(self) -> None:
        buffer = io.StringIO()
        diagnostics = {
            "mode": "offline-production",
            "speakerCardinality": {"fixedFivePersonLimit": False},
        }
        with (
            patch(
                "backend.worker.ProductionConfig.load",
                return_value=self.config,
            ),
            patch(
                "backend.worker.run_production_preflight",
                return_value=self.passed_report,
            ),
            patch(
                "backend.worker.production_diagnostics",
                return_value=diagnostics,
            ),
            patch(
                "backend.worker.build_production_composition"
            ) as composition,
            patch("backend.worker.apply_offline_environment"),
            redirect_stdout(buffer),
        ):
            exit_code = main(
                ["--config", str(self.config_path), "--diagnose"]
            )
        self.assertEqual(exit_code, 0)
        self.assertFalse(composition.called)
        event = self.output_line(buffer)
        self.assertEqual(event["type"], "worker.diagnostics.completed")
        self.assertFalse(
            event["payload"]["speakerCardinality"]["fixedFivePersonLimit"]
        )

    def test_startup_uses_config_line_limit_and_production_composition(self) -> None:
        buffer = io.StringIO()
        composition = SimpleNamespace(service=FakeService())
        captured: dict[str, object] = {}

        class FakeProtocol:
            def __init__(self, service, emitter, *, max_line_bytes):
                captured.update(
                    service=service,
                    emitter=emitter,
                    max_line_bytes=max_line_bytes,
                )

        with (
            patch(
                "backend.worker.ProductionConfig.load",
                return_value=self.config,
            ),
            patch(
                "backend.worker.run_production_preflight",
                return_value=self.passed_report,
            ),
            patch(
                "backend.worker.build_production_composition",
                return_value=composition,
            ) as build,
            patch("backend.worker.WorkerProtocol", FakeProtocol),
            patch("backend.worker.run_jsonl_loop") as loop,
            patch("backend.worker.apply_offline_environment"),
            redirect_stdout(buffer),
        ):
            exit_code = main(["--config", str(self.config_path)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["max_line_bytes"], 8192)
        build.assert_called_once()
        loop.assert_called_once()
        self.assertEqual(buffer.getvalue(), "")

    def test_unexpected_startup_error_is_sanitized(self) -> None:
        buffer = io.StringIO()
        with (
            patch(
                "backend.worker.ProductionConfig.load",
                side_effect=RuntimeError(f"secret path: {self.root}"),
            ),
            redirect_stdout(buffer),
        ):
            exit_code = main(["--config", str(self.config_path)])
        self.assertEqual(exit_code, 2)
        event = self.output_line(buffer)
        self.assertEqual(
            event["payload"]["code"], "PRODUCTION_STARTUP_FAILED"
        )
        serialized = json.dumps(event)
        self.assertNotIn(str(self.root), serialized)
        self.assertEqual(
            event["payload"]["details"]["exceptionType"], "RuntimeError"
        )

    def test_worker_error_payload_is_preserved(self) -> None:
        buffer = io.StringIO()
        error = ProductionConfigError(
            "invalid config",
            details={"component": "production-config"},
        )
        with (
            patch(
                "backend.worker.ProductionConfig.load",
                side_effect=error,
            ),
            redirect_stdout(buffer),
        ):
            exit_code = main(["--config", str(self.config_path)])
        self.assertEqual(exit_code, 2)
        event = self.output_line(buffer)
        self.assertEqual(event["payload"], error.as_payload())


if __name__ == "__main__":
    unittest.main()
