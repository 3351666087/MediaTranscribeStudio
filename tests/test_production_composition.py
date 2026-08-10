from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from backend.composition import (
    ProductionFactories,
    build_production_composition,
)
from backend.local_llm import AnthropicProvider, create_llm_provider
from backend.production_config import (
    ProductionConfig,
    ProductionConfigError,
    _probe_python_cuda_available,
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
        self.ffprobe = self.root / "ffprobe.exe"
        self.java = self.root / "java.exe"
        self.pyannote_python = self.root / "pyannote-python.exe"
        self.ffmpeg.write_bytes(b"fixture")
        self.ffprobe.write_bytes(b"fixture")
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
                "localLlmModel": "qwen3.5:9b",
                "localLlmModelDigest": "a" * 64,
                "localLlmTimeoutSeconds": 444,
                "localLlmContextTokens": 32_768,
                "localLlmOutputTokens": 4_096,
                "localLlmBatchSize": 6,
                "localLlmMaxRounds": 4,
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

    def explicit_secondary_mapping(
        self,
        *,
        model_path: Path,
        registry_model_id: str,
        manifest_model_key: str,
        manifest_sha256: str | None = None,
    ) -> dict[str, Any]:
        value = self.mapping()
        del value["models"]["eres2netV2"]
        manifest_path = model_path / ".mts-model-manifest.json"
        value["models"]["secondarySpeakerVerifier"] = {
            "path": str(model_path),
            "deploymentSlot": "secondary-speaker-verification",
            "registryModelId": registry_model_id,
            "manifestModelKey": manifest_model_key,
            "manifestSha256": (
                manifest_sha256
                if manifest_sha256 is not None
                else hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            ),
            "adapterId": "modelscope-eres2netv2",
        }
        return value

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
            config.speaker.local_llm_model_digest,
            "sha256:" + "a" * 64,
        )
        self.assertEqual(config.speaker.local_llm_timeout_seconds, 444.0)
        self.assertEqual(config.speaker.local_llm_context_tokens, 32_768)
        self.assertEqual(config.speaker.local_llm_output_tokens, 4_096)
        self.assertEqual(config.speaker.local_llm_batch_size, 6)
        self.assertEqual(config.speaker.local_llm_max_rounds, 4)
        self.assertEqual(
            config.speaker.overlap_recovery_asr_max_new_tokens,
            72,
        )
        self.assertEqual(config.runtime.vad_device, "cpu")
        self.assertEqual(config.runtime.model_residency, "worker")
        self.assertEqual(config.runtime.heartbeat_interval_seconds, 7.5)
        self.assertTrue(config.offline)
        self.assertEqual(
            config.models.secondary_speaker_verifier.deployment_slot,
            "secondary-speaker-verification",
        )
        self.assertEqual(
            config.models.secondary_speaker_verifier.registry_model_id,
            "eres2netv2",
        )
        self.assertIsNone(
            config.models.secondary_speaker_verifier.manifest_sha256
        )
        self.assertTrue(
            config.models.secondary_speaker_verifier.legacy_binding
        )

    def test_configurable_remote_native_provider_uses_preset_without_network(self) -> None:
        value = self.mapping()
        value["mode"] = "configurable-production"
        value["offline"] = False
        value["runtime"]["pyannoteDevice"] = "cpu"
        del value["speaker"]["localLlmModelDigest"]
        value["speaker"]["localLlmModel"] = "claude-sonnet-4-5"
        value["llm"] = {
            "provider": "anthropic",
            "endpoint": "https://api.anthropic.com/v1",
            "apiKeyEnv": "MTS_TEST_ANTHROPIC_KEY",
            "requireApiKey": True,
        }
        self.config_path.write_text(json.dumps(value), encoding="utf-8")

        config = ProductionConfig.load(self.config_path)

        self.assertEqual(config.llm.provider, "anthropic")
        self.assertEqual(config.llm.model, "claude-sonnet-4-5")
        self.assertEqual(config.llm.network_policy, "remote-explicit")
        self.assertEqual(config.llm.api_key_env, "MTS_TEST_ANTHROPIC_KEY")
        self.assertIsNone(config.speaker.local_llm_model_digest)
        # Factory construction is side-effect free; no HTTP call occurs until
        # generate_json is explicitly invoked.
        provider = create_llm_provider(config.llm)
        self.assertIsInstance(provider, AnthropicProvider)
        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        self.assertTrue(report.passed)

    def test_configurable_remote_provider_can_use_model_different_from_offline_default(self) -> None:
        value = self.mapping()
        value["mode"] = "configurable-production"
        value["offline"] = False
        value["runtime"]["pyannoteDevice"] = "cpu"
        # Keep the audited offline champion in the legacy speaker field while
        # selecting a replaceable remote deployment in the provider block.
        value["speaker"]["localLlmModel"] = "qwen3.5:27b-q4_K_M"
        del value["speaker"]["localLlmModelDigest"]
        value["llm"] = {
            "provider": "enterprise-relay",
            "model": "deployment-b",
            "endpoint": "https://relay.example/v1",
            "apiKeyEnv": "RELAY_API_KEY",
        }
        self.config_path.write_text(json.dumps(value), encoding="utf-8")

        config = ProductionConfig.load(self.config_path)

        self.assertEqual(config.llm.provider, "enterprise-relay")
        self.assertEqual(config.llm.model, "deployment-b")
        self.assertEqual(config.speaker.local_llm_model, "qwen3.5:27b-q4_K_M")
        self.assertTrue(config.llm.require_api_key)

    def test_composition_remote_provider_factory_is_native_and_does_not_touch_network(self) -> None:
        value = self.mapping()
        value["mode"] = "configurable-production"
        value["offline"] = False
        value["runtime"]["pyannoteDevice"] = "cpu"
        del value["speaker"]["localLlmModelDigest"]
        value["speaker"]["localLlmModel"] = "claude-sonnet-4-5"
        value["llm"] = {
            "provider": "anthropic",
            "endpoint": "https://api.anthropic.com/v1",
            "apiKeyEnv": "MTS_TEST_ANTHROPIC_KEY",
            "requireApiKey": True,
        }
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        config = ProductionConfig.load(self.config_path)
        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        factories = ProductionFactories(
            preparation=RecordingFactory(),
            asr=RecordingFactory(),
            embedding=RecordingFactory(),
            secondary=RecordingFactory(),
            pyannote=RecordingFactory(),
            separation=RecordingFactory(),
            cache=RecordingFactory(),
            pipeline=FakePipeline,
            assembler=RecordingFactory(),
            java_client_from_jar=RecordingFactory(),
            renderer=RecordingFactory(),
            media_probe=RecordingFactory(),
            subtitle_delivery_executor=RecordingFactory(),
            subtitle_visual_qa=RecordingFactory(),
            service=FakeService,
        )
        composition = build_production_composition(
            config,
            preflight_report=report,
            factories=factories,
        )
        request = SimpleNamespace(
            business_config=SimpleNamespace(
                model="claude-sonnet-4-5",
                translation_targets=(),
            ),
            local_llm_model="claude-sonnet-4-5",
            local_llm_endpoint="https://api.anthropic.com/v1",
        )

        provider = composition.service.kwargs["business_provider_factory"](request)

        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(provider.config.endpoint, "https://api.anthropic.com/v1")

    def test_configurable_loopback_provider_still_requires_digest(self) -> None:
        value = self.mapping()
        value["mode"] = "configurable-production"
        value["offline"] = False
        value["runtime"]["pyannoteDevice"] = "cpu"
        del value["speaker"]["localLlmModelDigest"]
        value["llm"] = {
            "provider": "ollama-loopback",
            "model": "qwen3.5:27b-q4_K_M",
            "endpoint": "http://127.0.0.1:11434",
        }
        value["speaker"]["localLlmModel"] = "qwen3.5:27b-q4_K_M"
        self.config_path.write_text(json.dumps(value), encoding="utf-8")

        with self.assertRaisesRegex(
            ProductionConfigError,
            "localLlmModelDigest",
        ):
            ProductionConfig.load(self.config_path)

    def test_explicit_challenger_binding_controls_manifest_and_fingerprint(
        self,
    ) -> None:
        challenger = self.root / "eres-wide"
        challenger.mkdir()
        self._write_model_fixture(
            challenger,
            model_key="eres2netV2LargeCandidate",
        )
        value = self.explicit_secondary_mapping(
            model_path=challenger,
            registry_model_id="eres2netv2-w24s4ep4",
            manifest_model_key="eres2netV2LargeCandidate",
        )
        self.config_path.write_text(json.dumps(value), encoding="utf-8")

        challenger_config = ProductionConfig.load(self.config_path)
        binding = challenger_config.models.secondary_speaker_verifier
        self.assertEqual(binding.path, challenger)
        self.assertEqual(binding.registry_model_id, "eres2netv2-w24s4ep4")
        self.assertEqual(
            binding.manifest_model_key,
            "eres2netV2LargeCandidate",
        )
        self.assertTrue(binding.manifest_sha256.startswith("sha256:"))
        self.assertFalse(binding.legacy_binding)
        report = run_production_preflight(
            challenger_config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        by_id = {check.check_id: check for check in report.checks}
        self.assertTrue(report.passed)
        self.assertTrue(
            by_id["secondary-speaker-verifier-model-integrity"].passed
        )
        self.assertNotEqual(challenger_config.fingerprint(), self.load().fingerprint())

    def test_secondary_binding_is_exclusive_and_manifest_pinned(self) -> None:
        value = self.mapping()
        value["models"]["secondarySpeakerVerifier"] = {
            "path": "eres",
            "deploymentSlot": "secondary-speaker-verification",
            "registryModelId": "eres2netv2",
            "manifestModelKey": "eres2netV2",
            "manifestSha256": "a" * 64,
            "adapterId": "modelscope-eres2netv2",
        }
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ProductionConfigError, "exactly one"):
            ProductionConfig.load(self.config_path)

        del value["models"]["eres2netV2"]
        value["models"]["secondarySpeakerVerifier"]["manifestSha256"] = (
            "b" * 64
        )
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        config = ProductionConfig.load(self.config_path)
        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        by_id = {check.check_id: check for check in report.checks}
        self.assertFalse(
            by_id["secondary-speaker-verifier-model-integrity"].passed
        )

        del value["models"]["secondarySpeakerVerifier"]
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ProductionConfigError, "exactly one"):
            ProductionConfig.load(self.config_path)

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

    def test_local_model_is_configurable(self) -> None:
        value = self.mapping()
        value["speaker"]["localLlmModel"] = "candidate-structural:14b"
        self.config_path.write_text(json.dumps(value), encoding="utf-8")

        config = ProductionConfig.load(self.config_path)

        self.assertEqual(
            config.speaker.local_llm_model,
            "candidate-structural:14b",
        )

    def test_local_model_generation_limits_fail_closed(self) -> None:
        value = self.mapping()
        del value["speaker"]["localLlmModelDigest"]
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            ProductionConfigError,
            "missing required fields",
        ):
            ProductionConfig.load(self.config_path)

        value = self.mapping()
        value["speaker"]["localLlmModelDigest"] = "not-a-digest"
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            ProductionConfigError,
            "localLlmModelDigest",
        ):
            ProductionConfig.load(self.config_path)

        value = self.mapping()
        value["speaker"]["localLlmContextTokens"] = 1024
        value["speaker"]["localLlmOutputTokens"] = 2048
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            ProductionConfigError,
            "localLlmOutputTokens",
        ):
            ProductionConfig.load(self.config_path)

        value = self.mapping()
        value["speaker"]["localLlmBatchSize"] = 33
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            ProductionConfigError,
            "localLlmBatchSize must be between 1 and 32",
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

    def test_isolated_python_cuda_probe_requires_available_sentinel(self) -> None:
        completed = CompletedProcess(
            args=[str(self.pyannote_python)],
            returncode=0,
            stdout=b"available\n",
            stderr=b"",
        )

        with patch(
            "backend.production_config.subprocess.run",
            return_value=completed,
        ) as run:
            self.assertTrue(
                _probe_python_cuda_available(str(self.pyannote_python))
            )

        self.assertEqual(run.call_args.args[0][0], str(self.pyannote_python))
        self.assertIn("torch.cuda.is_available()", run.call_args.args[0][-1])
        self.assertEqual(run.call_args.kwargs["timeout"], 120.0)

        completed.stdout = b"unavailable\n"
        with patch(
            "backend.production_config.subprocess.run",
            return_value=completed,
        ):
            self.assertFalse(
                _probe_python_cuda_available(str(self.pyannote_python))
            )

    def test_preflight_rejects_unavailable_isolated_pyannote_cuda(self) -> None:
        config = self.load(pyannote_mode="fallback")

        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            pyannote_cuda_probe=lambda _python: False,
            probe_executables=False,
        )
        by_id = {check.check_id: check for check in report.checks}

        self.assertFalse(report.passed)
        self.assertFalse(by_id["runtime-pyannote-cuda"].passed)
        self.assertEqual(
            by_id["runtime-pyannote-cuda"].reason_code,
            "ISOLATED_PYANNOTE_CUDA_REQUIRED",
        )
        with self.assertRaises(ProductionConfigError) as captured:
            report.raise_if_failed()
        self.assertIn(
            "runtime-pyannote-cuda",
            captured.exception.details["failedChecks"],
        )

    def test_preflight_skips_cuda_probe_for_cpu_pyannote(self) -> None:
        config = self.load(pyannote_mode="fallback")
        config = replace(
            config,
            runtime=replace(config.runtime, pyannote_device="cpu"),
        )

        cuda_probe = RecordingFactory(False)
        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            pyannote_cuda_probe=cuda_probe,
            probe_executables=False,
        )

        self.assertTrue(report.passed)
        self.assertEqual(cuda_probe.calls, [])
        self.assertNotIn(
            "runtime-pyannote-cuda",
            {check.check_id for check in report.checks},
        )

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

    def test_preflight_accepts_byte_preserving_reshard_manifest(self) -> None:
        config = self.load()
        manifest_path = (
            config.models.qwen3_forced_aligner / ".mts-model-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "schemaVersion": "1.1.0",
                "kind": "derived-safetensors-reshard",
                "reshard": {
                    "schemaVersion": "1.0.0",
                    "tool": "tools/reshard_safetensors.py",
                    "exactTensorBytesPreserved": True,
                    "modelQualityChanged": False,
                    "source": {
                        field: manifest[field]
                        for field in (
                            "modelKey",
                            "provider",
                            "repoId",
                            "revision",
                        )
                    },
                },
            }
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )

        self.assertTrue(report.passed)

    def test_preflight_rejects_reshard_without_preservation_evidence(self) -> None:
        config = self.load()
        manifest_path = (
            config.models.qwen3_forced_aligner / ".mts-model-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "schemaVersion": "1.1.0",
                "kind": "derived-safetensors-reshard",
                "reshard": {
                    "schemaVersion": "1.0.0",
                    "tool": "tools/reshard_safetensors.py",
                    "exactTensorBytesPreserved": False,
                    "modelQualityChanged": False,
                    "source": {
                        field: manifest[field]
                        for field in (
                            "modelKey",
                            "provider",
                            "repoId",
                            "revision",
                        )
                    },
                },
            }
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        by_id = {check.check_id: check for check in report.checks}

        self.assertFalse(
            by_id["qwen3-forced-aligner-model-integrity"].passed
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
        subtitle_delivery_executor = RecordingFactory(
            SimpleNamespace(adapter_id="subtitle-delivery")
        )
        subtitle_visual_qa = RecordingFactory(
            SimpleNamespace(adapter_id="subtitle-visual-qa")
        )
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
                subtitle_delivery_executor=subtitle_delivery_executor,
                subtitle_visual_qa=subtitle_visual_qa,
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
        secondary_kwargs = secondary.calls[0][1]
        self.assertEqual(
            secondary_kwargs["deployment_slot"],
            "secondary-speaker-verification",
        )
        self.assertEqual(secondary_kwargs["registry_model_id"], "eres2netv2")
        self.assertEqual(secondary_kwargs["manifest_model_key"], "eres2netV2")
        self.assertIsNone(secondary_kwargs["manifest_sha256"])
        self.assertEqual(
            secondary_kwargs["adapter_id"],
            "modelscope-eres2netv2",
        )
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
        self.assertEqual(len(subtitle_delivery_executor.calls), 1)
        subtitle_delivery_kwargs = subtitle_delivery_executor.calls[0][1]
        self.assertIs(
            subtitle_delivery_kwargs["probe"],
            service.kwargs["media_probe"],
        )
        self.assertEqual(
            subtitle_delivery_kwargs["ffmpeg_command"],
            (str(self.ffmpeg),),
        )
        self.assertIs(
            service.kwargs["subtitle_delivery_executor"],
            subtitle_delivery_executor.result,
        )
        self.assertEqual(len(subtitle_visual_qa.calls), 1)
        subtitle_visual_qa_kwargs = subtitle_visual_qa.calls[0][1]
        self.assertEqual(
            subtitle_visual_qa_kwargs["ffmpeg_path"],
            str(self.ffmpeg),
        )
        self.assertIs(
            subtitle_visual_qa_kwargs["probe"],
            service.kwargs["media_probe"],
        )
        self.assertIs(
            service.kwargs["subtitle_visual_qa_hook"],
            subtitle_visual_qa.result,
        )
        request = SimpleNamespace(
            business_config=SimpleNamespace(
                model="qwen3.5:9b",
                translation_targets=(),
            ),
            local_llm_endpoint="http://127.0.0.1:11434",
        )
        business_provider = service.kwargs["business_provider_factory"](request)
        semantic_provider = service.kwargs["semantic_provider_factory"](request)
        semantic_orchestrator = service.kwargs[
            "semantic_orchestrator_factory"
        ](
            request,
            SimpleNamespace(raise_if_cancelled=lambda: None),
        )
        self.assertEqual(business_provider.config.keep_alive, "10m")
        self.assertEqual(semantic_provider.config.keep_alive, "10m")
        self.assertEqual(business_provider.config.timeout_seconds, 444.0)
        self.assertEqual(semantic_provider.config.context_tokens, 32_768)
        self.assertEqual(semantic_provider.config.output_tokens, 4_096)
        self.assertEqual(
            semantic_provider.config.expected_model_digest,
            "sha256:" + "a" * 64,
        )
        self.assertEqual(semantic_orchestrator.arbitrator.batch_size, 6)
        self.assertEqual(semantic_orchestrator.max_rounds, 4)
        self.assertFalse(business_provider.config.release_on_close)
        self.assertFalse(semantic_provider.config.release_on_close)

        request.business_config.model = "unpinned-candidate:latest"
        with self.assertRaisesRegex(
            ValueError,
            "must match the digest-pinned local LLM model",
        ):
            service.kwargs["business_provider_factory"](request)

    def test_stage_residency_stays_warm_and_explicitly_releases(self) -> None:
        config = self.load()
        config = replace(
            config,
            runtime=replace(config.runtime, model_residency="stage"),
        )
        report = run_production_preflight(
            config,
            runtime_probe=lambda _module: True,
            probe_executables=False,
        )
        composition = build_production_composition(
            config,
            preflight_report=report,
            factories=ProductionFactories(
                preparation=RecordingFactory(),
                asr=RecordingFactory(),
                embedding=RecordingFactory(),
                secondary=RecordingFactory(),
                pyannote=RecordingFactory(),
                cache=RecordingFactory(),
                pipeline=FakePipeline,
                assembler=RecordingFactory(),
                java_client_from_jar=RecordingFactory(),
                renderer=RecordingFactory(),
                service=FakeService,
            ),
        )
        request = SimpleNamespace(
            business_config=SimpleNamespace(model="qwen3.5:9b"),
            local_llm_endpoint="http://127.0.0.1:11434",
        )
        service = composition.service

        business_provider = service.kwargs["business_provider_factory"](request)
        semantic_provider = service.kwargs["semantic_provider_factory"](request)

        self.assertEqual(business_provider.config.keep_alive, "5m")
        self.assertEqual(semantic_provider.config.keep_alive, "5m")
        self.assertTrue(business_provider.config.release_on_close)
        self.assertTrue(semantic_provider.config.release_on_close)

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
