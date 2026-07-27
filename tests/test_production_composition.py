from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from backend.composition import (
    ProductionFactories,
    build_production_composition,
)
from backend.production_config import (
    ProductionConfig,
    ProductionConfigError,
    _probe_runtime_import,
    production_diagnostics,
    run_production_preflight,
)
from backend.speaker_pipeline import SpeakerPipeline


class RecordingFactory:
    def __init__(self, result: Any = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.result = result if result is not None else SimpleNamespace()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.result


class FakePipeline:
    version = "9.0-test"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeService:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class ProductionCompositionTests(unittest.TestCase):
    @staticmethod
    def _write_model_fixture(path: Path, *, model_key: str) -> None:
        payload = path / "weights.bin"
        content = f"fixture:{model_key}".encode("utf-8")
        payload.write_bytes(content)
        manifest = {
            "schemaVersion": "1.0.0",
            "provider": "modelscope",
            "modelKey": model_key,
            "repoId": f"fixture/{model_key}",
            "revision": "fixture-revision",
            "totalBytes": len(content),
            "files": [
                {
                    "path": payload.name,
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ],
        }
        (path / ".mts-model-manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.input_root = self.root / "input"
        self.output_root = self.root / "output"
        self.cache_root = self.root / "cache"
        self.input_root.mkdir()
        self.output_root.mkdir()
        model_fixtures = {
            "vad": "funasrVad",
            "asr": "qwen3Asr",
            "aligner": "qwen3ForcedAligner",
            "cam": "camPlus",
            "eres": "eres2netV2",
            "pyannote": "pyannoteCommunity1",
        }
        for name, model_key in model_fixtures.items():
            path = self.root / name
            path.mkdir()
            self._write_model_fixture(path, model_key=model_key)
        self.jar = self.root / "renderer.jar"
        with zipfile.ZipFile(self.jar, "w") as archive:
            archive.writestr("META-INF/MANIFEST.MF", "Main-Class: test.Main\n")
            archive.writestr(
                "com/openhtmltopdf/pdfboxout/PdfRendererBuilder.class", b""
            )
            archive.writestr(
                "org/apache/pdfbox/pdmodel/PDDocument.class", b""
            )
        self.ffmpeg = self.root / "ffmpeg.exe"
        self.java = self.root / "java.exe"
        self.pyannote_python = self.root / "pyannote-python.exe"
        self.ffmpeg.write_bytes(b"fixture")
        self.java.write_bytes(b"fixture")
        self.pyannote_python.write_bytes(b"fixture")
        self.config_path = self.root / "production.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def mapping(self, *, pyannote_mode: str = "fallback") -> dict[str, Any]:
        return {
            "schemaVersion": "1.0.0",
            "mode": "offline-production",
            "offline": True,
            "paths": {
                "allowedInputRoots": ["input"],
                "allowedOutputRoot": "output",
                "cacheRoot": "cache",
            },
            "models": {
                "funasrVad": "vad",
                "qwen3Asr": "asr",
                "qwen3ForcedAligner": "aligner",
                "camPlus": "cam",
                "eres2netV2": "eres",
                "pyannote": (
                    "pyannote" if pyannote_mode != "disabled" else None
                ),
            },
            "executables": {
                "ffmpeg": str(self.ffmpeg),
                "java": str(self.java),
                "pdfRendererJar": "renderer.jar",
                "pyannotePython": str(self.pyannote_python),
            },
            "runtime": {
                "maxWorkers": 2,
                "maxPendingJobs": 3,
                "modelResidency": "worker",
                "heartbeatIntervalSeconds": 7.5,
                "strictStartupPreflight": True,
            },
            "speaker": {
                "maxAutoSpeakers": None,
                "maxClusteringWindows": 12_345,
                "maxClusteringWorkItems": 234_567,
                "countStabilityRuns": 5,
                "eigengapLandmarkLimit": 128,
                "maxLanguageWindowMs": 11_000,
                "languageSplitSearchMs": 900,
                "pyannoteMappingMarginThreshold": 0.07,
                "pyannotePrimaryDominanceThreshold": 0.65,
                "pyannoteMode": pyannote_mode,
                "overlapRecoveryAsrMaxNewTokens": 72,
                "localLlmMode": "suggestion-only",
            },
            "pdf": {
                "minimumScore": 85,
                "maxRounds": 5,
            },
        }

    def load(self, *, pyannote_mode: str = "fallback") -> ProductionConfig:
        self.config_path.write_text(
            json.dumps(self.mapping(pyannote_mode=pyannote_mode)),
            encoding="utf-8",
        )
        return ProductionConfig.load(self.config_path)

    def test_load_resolves_relative_local_paths_and_dynamic_cardinality(self) -> None:
        config = self.load()
        self.assertEqual(config.paths.allowed_input_roots, (self.input_root,))
        self.assertEqual(config.models.qwen3_asr, self.root / "asr")
        self.assertIsNone(config.speaker.max_auto_speakers)
        self.assertEqual(config.speaker.max_clustering_windows, 12_345)
        self.assertEqual(config.speaker.max_clustering_work_items, 234_567)
        self.assertEqual(config.speaker.count_stability_runs, 5)
        self.assertEqual(config.speaker.eigengap_landmark_limit, 128)
        self.assertEqual(config.speaker.max_language_window_ms, 11_000)
        self.assertEqual(config.speaker.language_split_search_ms, 900)
        self.assertEqual(config.speaker.pyannote_mapping_margin_threshold, 0.07)
        self.assertEqual(
            config.speaker.pyannote_primary_dominance_threshold,
            0.65,
        )
        self.assertEqual(config.speaker.pyannote_mode, "fallback")
        self.assertEqual(
            config.speaker.overlap_recovery_asr_max_new_tokens,
            72,
        )
        self.assertEqual(config.runtime.vad_device, "cpu")
        self.assertEqual(config.runtime.model_residency, "worker")
        self.assertEqual(config.runtime.heartbeat_interval_seconds, 7.5)
        self.assertTrue(config.offline)

    def test_rejects_pdf_font_not_bundled_by_renderer(self) -> None:
        mapping = self.mapping()
        mapping["pdf"]["preferredFont"] = "MTS CJK"
        self.config_path.write_text(
            json.dumps(mapping),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ProductionConfigError,
            "pdf.preferredFont must be one of LXGW WenKai",
        ):
            ProductionConfig.load(self.config_path)

    def test_unknown_fields_fail_closed(self) -> None:
        value = self.mapping()
        value["models"]["remoteModelId"] = "organization/model"
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            ProductionConfigError, "unsupported fields"
        ):
            ProductionConfig.load(self.config_path)

    def test_pyannote_mode_and_model_must_agree(self) -> None:
        value = self.mapping(pyannote_mode="fallback")
        value["models"]["pyannote"] = None
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            ProductionConfigError, "models.pyannote is required"
        ):
            ProductionConfig.load(self.config_path)

    def test_pyannote_mode_requires_isolated_python(self) -> None:
        value = self.mapping(pyannote_mode="fallback")
        del value["executables"]["pyannotePython"]
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            ProductionConfigError,
            "executables.pyannotePython is required",
        ):
            ProductionConfig.load(self.config_path)

    def test_removed_pyannote_audit_mode_requires_explicit_migration(
        self,
    ) -> None:
        for removed_value in ("audit", " audit "):
            with self.subTest(removed_value=removed_value):
                with self.assertRaises(ProductionConfigError) as captured:
                    self.load(pyannote_mode=removed_value)

                error = captured.exception
                self.assertEqual(
                    error.code,
                    "PRODUCTION_CONFIG_MIGRATION_REQUIRED",
                )
                self.assertEqual(
                    error.details["field"],
                    "speaker.pyannoteMode",
                )
                self.assertEqual(error.details["removedValue"], "audit")
                self.assertEqual(
                    error.details["supportedValues"],
                    ["disabled", "fallback"],
                )

    def test_secondary_fraction_cannot_consume_the_full_corpus(self) -> None:
        value = self.mapping()
        value["speaker"]["maxSecondaryFraction"] = 1.0
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            ProductionConfigError, "must be less than 1.0"
        ):
            ProductionConfig.load(self.config_path)

    def test_preflight_is_path_free_and_checks_all_runtime_layers(self) -> None:
        config = self.load()
        report = run_production_preflight(
            config,
            runtime_probe=lambda module: module != "pyannote.audio",
            probe_executables=False,
        )
        self.assertFalse(report.passed)
        diagnostics = production_diagnostics(config, report)
        serialized = json.dumps(diagnostics, ensure_ascii=False)
        self.assertNotIn(str(self.root), serialized)
        self.assertIn("runtime-pyannote", serialized)
        self.assertIn("runtime-simplejson", serialized)
        self.assertIn("difficult-segments-only", serialized)
        self.assertFalse(
            diagnostics["speakerCardinality"]["fixedFivePersonLimit"]
        )

    def test_runtime_import_probe_allows_slow_cold_start(self) -> None:
        completed = CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=b"ok\n",
            stderr=b"",
        )

        with patch(
            "backend.production_config.subprocess.run",
            return_value=completed,
        ) as run:
            self.assertTrue(_probe_runtime_import("funasr"))

        self.assertEqual(run.call_args.kwargs["timeout"], 120.0)
        code = run.call_args.args[0][-1]
        self.assertLess(code.index("import setuptools"), code.index("importlib"))

    def test_preflight_checks_funasr_campplus_registration_module(self) -> None:
        config = self.load()
        probed: list[str] = []

        report = run_production_preflight(
            config,
            runtime_probe=lambda module: probed.append(module) or True,
            probe_executables=False,
        )

        self.assertTrue(report.passed)
        self.assertIn("funasr.models.campplus.model", probed)
        self.assertIn(
            "runtime-funasr-campplus",
            {check.check_id for check in report.checks},
        )

    def test_preflight_rejects_present_but_corrupted_model_content(self) -> None:
        config = self.load()
        (config.models.cam_plus / "weights.bin").write_bytes(b"corrupted")

        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        by_id = {check.check_id: check for check in report.checks}

        self.assertTrue(by_id["cam-plus-model"].passed)
        self.assertFalse(by_id["cam-plus-model-integrity"].passed)
        self.assertEqual(
            by_id["cam-plus-model-integrity"].reason_code,
            "LOCKED_MODEL_CONTENT_INTEGRITY",
        )
        self.assertFalse(report.passed)

    def test_preflight_write_probes_are_concurrency_safe(self) -> None:
        config = self.load()

        def run_once() -> bool:
            return run_production_preflight(
                config,
                runtime_probe=lambda _module: True,
                probe_executables=False,
            ).passed

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _index: run_once(), range(32)))

        self.assertEqual(results, [True] * 32)
        self.assertEqual(
            list(config.paths.allowed_output_root.glob(".mts-write-probe-*")),
            [],
        )
        self.assertEqual(
            list(config.paths.cache_root.glob(".mts-write-probe-*")),
            [],
        )

    def test_composition_never_uses_unavailable_adapters(self) -> None:
        config = self.load()
        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        preparation = RecordingFactory(SimpleNamespace(adapter_id="prep"))
        asr = RecordingFactory(SimpleNamespace(adapter_id="asr"))
        embedding = RecordingFactory(SimpleNamespace(adapter_id="cam"))
        secondary = RecordingFactory(SimpleNamespace(adapter_id="eres"))
        pyannote = RecordingFactory(
            SimpleNamespace(adapter_id="pyannote", telemetry_enabled=False)
        )
        cache = RecordingFactory(SimpleNamespace())
        assembler = RecordingFactory(SimpleNamespace())
        java_client = RecordingFactory(SimpleNamespace())
        renderer = RecordingFactory(SimpleNamespace(adapter_id="renderer"))
        composition = build_production_composition(
            config,
            preflight_report=report,
            factories=ProductionFactories(
                preparation=preparation,
                asr=asr,
                embedding=embedding,
                secondary=secondary,
                pyannote=pyannote,
                cache=cache,
                pipeline=FakePipeline,
                assembler=assembler,
                java_client_from_jar=java_client,
                renderer=renderer,
                service=FakeService,
            ),
        )
        service = composition.service
        self.assertIsInstance(service, FakeService)
        self.assertEqual(service.kwargs["max_workers"], 2)
        self.assertEqual(service.kwargs["max_pending_jobs"], 3)
        pipeline = service.kwargs["transcription_adapter"]
        self.assertIsInstance(pipeline, FakePipeline)
        self.assertIsNotNone(pipeline.kwargs["secondary_adapter"])
        self.assertIsNotNone(pipeline.kwargs["pyannote_adapter"])
        self.assertIsNone(pipeline.kwargs["config"].max_auto_speakers)
        self.assertEqual(
            pipeline.kwargs["config"].max_clustering_windows,
            12_345,
        )
        self.assertEqual(
            pipeline.kwargs["config"].max_clustering_work_items,
            234_567,
        )
        self.assertEqual(
            pipeline.kwargs["config"].count_stability_runs,
            5,
        )
        self.assertEqual(
            pipeline.kwargs["config"].eigengap_landmark_limit,
            128,
        )
        self.assertEqual(len(secondary.calls), 1)
        self.assertEqual(len(pyannote.calls), 1)
        self.assertEqual(
            pyannote.calls[0][1]["python_executable"],
            str(self.pyannote_python),
        )
        self.assertEqual(preparation.calls[0][1]["device"], "cpu")
        self.assertEqual(
            embedding.calls[0][1]["max_language_window_ms"],
            11_000,
        )
        self.assertEqual(
            embedding.calls[0][1]["language_split_search_ms"],
            900,
        )

    def test_real_config_builds_real_speaker_pipeline_config(self) -> None:
        config = self.load(pyannote_mode="fallback")
        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        composition = build_production_composition(
            config,
            preflight_report=report,
            factories=ProductionFactories(
                preparation=RecordingFactory(
                    SimpleNamespace(adapter_id="prep")
                ),
                asr=RecordingFactory(SimpleNamespace(adapter_id="asr")),
                embedding=RecordingFactory(
                    SimpleNamespace(adapter_id="cam")
                ),
                secondary=RecordingFactory(
                    SimpleNamespace(adapter_id="eres")
                ),
                pyannote=RecordingFactory(
                    SimpleNamespace(
                        adapter_id="pyannote",
                        telemetry_enabled=False,
                    )
                ),
                cache=RecordingFactory(SimpleNamespace()),
                assembler=RecordingFactory(SimpleNamespace()),
                java_client_from_jar=RecordingFactory(SimpleNamespace()),
                renderer=RecordingFactory(
                    SimpleNamespace(adapter_id="renderer")
                ),
                service=FakeService,
            ),
        )

        pipeline = composition.service.kwargs["transcription_adapter"]
        self.assertIsInstance(pipeline, SpeakerPipeline)
        self.assertEqual(pipeline.config.pyannote_mode, "fallback")
        self.assertEqual(pipeline.config.max_clustering_windows, 12_345)
        self.assertEqual(
            pipeline.config.max_clustering_work_items,
            234_567,
        )
        self.assertEqual(pipeline.config.count_stability_runs, 5)
        self.assertEqual(pipeline.config.eigengap_landmark_limit, 128)
        self.assertEqual(
            pipeline.config.pyannote_mapping_margin_threshold,
            0.07,
        )
        self.assertEqual(
            pipeline.config.pyannote_primary_dominance_threshold,
            0.65,
        )
        self.assertEqual(
            pipeline.config.overlap_recovery_asr_max_new_tokens,
            72,
        )
        self.assertIsNotNone(pipeline.pyannote_adapter)
        self.assertIs(
            pipeline.overlap_adapter,
            pipeline.pyannote_adapter,
        )
        self.assertEqual(
            composition.service.kwargs["heartbeat_interval_seconds"],
            7.5,
        )

    def test_explicit_vad_device_override_is_preserved(self) -> None:
        value = self.mapping()
        value["runtime"]["vadDevice"] = "cuda:7"
        self.config_path.write_text(json.dumps(value), encoding="utf-8")

        config = ProductionConfig.load(self.config_path)

        self.assertEqual(config.runtime.vad_device, "cuda:7")

    def test_disabled_pyannote_is_not_instantiated(self) -> None:
        config = self.load(pyannote_mode="disabled")
        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        pyannote = RecordingFactory()
        factories = ProductionFactories(
            preparation=RecordingFactory(),
            asr=RecordingFactory(),
            embedding=RecordingFactory(),
            secondary=RecordingFactory(),
            pyannote=pyannote,
            cache=RecordingFactory(),
            pipeline=FakePipeline,
            assembler=RecordingFactory(),
            java_client_from_jar=RecordingFactory(),
            renderer=RecordingFactory(),
            service=FakeService,
        )
        composition = build_production_composition(
            config,
            preflight_report=report,
            factories=factories,
        )
        pipeline = composition.service.kwargs["transcription_adapter"]
        self.assertIsNone(pipeline.kwargs["pyannote_adapter"])
        self.assertIsNone(pipeline.kwargs["overlap_adapter"])
        self.assertEqual(pyannote.calls, [])


if __name__ == "__main__":
    unittest.main()
