from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
import shutil
import struct
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.adapters import AdapterContext
from backend.errors import WorkerError
from backend.models import SpeakerScore, TranscriptSegment
from backend import production_runners
from backend.production_runners import (
    FfmpegFunAsrPreparationAdapter,
    LocalERes2NetV2Verifier,
    LocalFunAsrCamPlusAdapter,
    LocalPyannoteAuditAdapter,
    LocalQwen3AsrAdapter,
)
from backend.speaker_change_detection import EnergyValley
from backend.speaker_pipeline import (
    OverlapDecision,
    PreparedAudio,
    ReviewCandidate,
    SpeechWindow,
)


def write_wave(path: Path, amplitudes: list[float], *, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    samples_per_part = rate
    for amplitude in amplitudes:
        value = max(-32767, min(32767, round(amplitude * 32767)))
        for _ in range(samples_per_part):
            frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


class FakePcmSlice:
    def __init__(self, owner: "FakePcmTimeline", start: int, end: int) -> None:
        self.owner = owner
        self.start = start
        self.end = end


class FakePcmTimeline:
    marker = "RAW_PCM_MUST_NOT_ENTER_EVIDENCE"

    def __init__(self, sample_count: int, *, nbytes: int | None = None) -> None:
        self.sample_count = sample_count
        self.nbytes = sample_count * 4 if nbytes is None else nbytes
        self.slice_calls: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, item: slice) -> FakePcmSlice:
        if not isinstance(item, slice):
            raise TypeError("fake PCM only supports slicing")
        start = 0 if item.start is None else int(item.start)
        end = self.sample_count if item.stop is None else int(item.stop)
        self.slice_calls.append((start, end))
        return FakePcmSlice(self, start, end)


class FakeVadModel:
    def generate(self, **kwargs):
        return [{"value": [[0, 900], [1000, 1900]]}]


class FailingVadModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        raise WorkerError(
            "FUNASR_VAD_INFERENCE_FAILED",
            "synthetic VAD inference failure",
        )


class FakeQwenModel:
    def __init__(self) -> None:
        self.language_requests: list[list[str] | None] = []

    def transcribe(self, *, audio, return_time_stamps, language=None):
        self.language_requests.append(
            list(language) if language is not None else None
        )
        return [
            SimpleNamespace(
                text=f"中文窗口{index + 1}",
                language="Chinese",
                time_stamps=None,
            )
            for index in range(len(audio))
        ]


class FakeCamModel:
    def generate(self, *, input, batch_size, disable_pbar):
        return [
            {"spk_embedding": [[float(index + 1), 1.0]]}
            for index in range(len(input))
        ]


class FakeEResPipeline:
    def __call__(self, audio, output_emb):
        embeddings = []
        for clip in audio:
            mean = float(abs(clip).mean())
            embeddings.append([mean, 1.0 - mean])
        return {"embs": embeddings}


class FixedEResPipeline:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, audio, output_emb):
        return {"embs": self.embeddings[: len(audio)]}


class FakeAnnotation:
    def __init__(self, tracks):
        self.tracks = tracks

    def itertracks(self, *, yield_label):
        if not yield_label:
            raise AssertionError("yield_label=True is required")
        return iter(self.tracks)


class FakePyannotePipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, payload):
        self.calls.append(payload)
        return self.result


class ProductionRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        production_runners._SHARED_PCM_STORE.clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.audio = self.root / "audio.wav"
        write_wave(self.audio, [0.1, 0.9, 0.1, 0.9])
        self.context = AdapterContext(
            "runner-job",
            self.output,
            threading.Event(),
        )
        self.qwen_model = self.root / "qwen"
        self.cam_model = self.root / "cam"
        self.eres_model = self.root / "eres"
        self.vad_model = self.root / "vad"
        self.forced_aligner_model = self.root / "forced-aligner"
        self.pyannote_model = self.root / "pyannote"
        for path in (
            self.qwen_model,
            self.cam_model,
            self.eres_model,
            self.vad_model,
            self.forced_aligner_model,
            self.pyannote_model,
        ):
            path.mkdir()

    def tearDown(self) -> None:
        production_runners._SHARED_PCM_STORE.clear()
        self.temporary.cleanup()

    def prepared(self) -> PreparedAudio:
        return PreparedAudio(
            duration_ms=4000,
            source_fingerprint="a" * 64,
            normalization_profile="mono-16khz-f32-v1",
            windows=(
                SpeechWindow("window-1", 0, 1000),
                SpeechWindow("window-2", 1000, 2000),
            ),
            stage_durations_ms={
                "decode": 0.0,
                "normalize": 0.0,
                "vad": 0.0,
                "boundary": 0.0,
            },
            audio_path=str(self.audio),
        )

    def test_qwen3_runner_uses_local_only_model_and_preserves_raw_text(self) -> None:
        captured = {}
        model = FakeQwenModel()

        def factory(**kwargs):
            captured.update(kwargs)
            return model

        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=factory,
            device_map="cpu",
            torch_dtype="float32",
        )
        results = adapter.transcribe_batch(
            self.prepared(),
            self.prepared().windows,
            self.context,
            requested_language="auto",
        )
        self.assertTrue(captured["local_files_only"])
        self.assertEqual(captured["dtype"], "float32")
        self.assertNotIn("torch_dtype", captured)
        self.assertTrue(captured["low_cpu_mem_usage"])
        self.assertEqual(captured["max_inference_batch_size"], 2)
        self.assertEqual(
            [item.text for item in results],
            ["中文窗口1", "中文窗口2"],
        )
        self.assertEqual(model.language_requests, [None])
        self.assertEqual(
            [item.evidence["language"] for item in results],
            ["zh", "zh"],
        )
        self.assertTrue(
            all(item.evidence["confidenceAvailable"] is False for item in results)
        )

    def test_qwen3_retries_an_empty_batch_result_individually(self) -> None:
        class EmptyThenRecoveredModel:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def transcribe(
                self,
                *,
                audio,
                return_time_stamps,
                language=None,
            ):
                self.calls.append(len(audio))
                if len(audio) == 2:
                    return [
                        SimpleNamespace(
                            text="第一段",
                            language="Chinese",
                            time_stamps=None,
                        ),
                        SimpleNamespace(
                            text="",
                            language="Chinese",
                            time_stamps=None,
                        ),
                    ]
                return [
                    SimpleNamespace(
                        text="第二段恢复",
                        language="Chinese",
                        time_stamps=None,
                    )
                ]

        model = EmptyThenRecoveredModel()
        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=lambda **kwargs: model,
            device_map="cpu",
            torch_dtype="float32",
        )

        results = adapter.transcribe_batch(
            self.prepared(),
            self.prepared().windows,
            self.context,
            requested_language="auto",
        )

        self.assertEqual(model.calls, [2, 1])
        self.assertEqual([item.text for item in results], ["第一段", "第二段恢复"])
        self.assertNotIn("disposition", results[1].evidence)

    def test_qwen3_rejects_persistent_non_lexical_window_without_fabrication(
        self,
    ) -> None:
        class PermanentlyEmptyModel:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def transcribe(
                self,
                *,
                audio,
                return_time_stamps,
                language=None,
            ):
                self.calls.append(len(audio))
                return [
                    SimpleNamespace(
                        text=("第一段" if len(audio) == 2 and index == 0 else ""),
                        language="Chinese",
                        time_stamps=None,
                    )
                    for index in range(len(audio))
                ]

        model = PermanentlyEmptyModel()
        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=lambda **kwargs: model,
            device_map="cpu",
            torch_dtype="float32",
        )

        results = adapter.transcribe_batch(
            self.prepared(),
            self.prepared().windows,
            self.context,
            requested_language="auto",
        )

        self.assertEqual(model.calls, [2, 1])
        self.assertEqual(results[1].text, "")
        self.assertEqual(results[1].confidence, 0.0)
        self.assertEqual(
            results[1].evidence["disposition"],
            "rejected-non-lexical",
        )
        self.assertEqual(
            results[1].evidence["rejectionReason"],
            "EMPTY_AFTER_INDIVIDUAL_RETRY",
        )
        restored = production_runners.AsrHypothesis.from_mapping(
            results[1].as_dict()
        )
        self.assertEqual(restored, results[1])

    def test_qwen3_runner_rejects_unsupported_language_before_model_load(self) -> None:
        model_loads = 0

        def factory(**kwargs):
            nonlocal model_loads
            model_loads += 1
            return FakeQwenModel()

        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=factory,
            device_map="cpu",
            torch_dtype="float32",
        )

        with self.assertRaises(WorkerError) as captured:
            adapter.validate_requested_language("he-IL")

        self.assertEqual(
            captured.exception.code,
            "QWEN3_ASR_LANGUAGE_UNSUPPORTED",
        )
        self.assertEqual(model_loads, 0)
        self.assertTrue(captured.exception.details["autoDetectionAvailable"])
        self.assertIn(
            "en",
            captured.exception.details["supportedPrimaryLanguageTags"],
        )

    def test_qwen3_windows_balanced_policy_uses_bounded_memory(self) -> None:
        with (
            mock.patch.object(
                production_runners,
                "_is_windows_runtime",
                return_value=True,
            ),
            mock.patch.object(
                production_runners,
                "_cuda_total_memory_bytes",
                return_value=8 * 1024 * 1024 * 1024,
            ),
        ):
            asr = production_runners._qwen_load_policy(
                "cuda:0",
                forced_aligner=False,
                probe_accelerator=True,
            )
            aligner = production_runners._qwen_load_policy(
                "cuda:0",
                forced_aligner=True,
                probe_accelerator=True,
            )

        self.assertEqual(asr["device_map"], "balanced")
        self.assertEqual(
            asr["max_memory"],
            {0: "2GiB", "cpu": "8GiB"},
        )
        self.assertTrue(asr["offload_state_dict"])
        self.assertEqual(aligner["device_map"], "balanced")
        self.assertEqual(
            aligner["max_memory"],
            {0: "1500MiB", "cpu": "6GiB"},
        )

    def test_qwen3_cpu_policy_and_fake_factory_avoid_gpu_probe(self) -> None:
        with (
            mock.patch.object(
                production_runners,
                "_is_windows_runtime",
                return_value=True,
            ),
            mock.patch.object(
                production_runners,
                "_cuda_total_memory_bytes",
            ) as probe,
        ):
            cpu = production_runners._qwen_load_policy(
                "cpu",
                forced_aligner=False,
                probe_accelerator=True,
            )
            injected = production_runners._qwen_load_policy(
                "cuda:0",
                forced_aligner=False,
                probe_accelerator=False,
            )

        self.assertEqual(cpu, {"device_map": "cpu"})
        self.assertEqual(injected, {"device_map": "cuda:0"})
        probe.assert_not_called()

    def test_qwen3_forced_aligner_receives_safe_canonical_kwargs(self) -> None:
        captured: dict[str, object] = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return FakeQwenModel()

        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            forced_aligner_path=self.forced_aligner_model,
            model_factory=factory,
            device_map="cpu",
            torch_dtype="float32",
        )
        adapter.transcribe_batch(
            self.prepared(),
            self.prepared().windows,
            self.context,
            requested_language="auto",
        )

        aligner = captured["forced_aligner_kwargs"]
        self.assertIsInstance(aligner, dict)
        assert isinstance(aligner, dict)
        self.assertEqual(aligner["device_map"], "cpu")
        self.assertEqual(aligner["dtype"], "float32")
        self.assertTrue(aligner["local_files_only"])
        self.assertTrue(aligner["low_cpu_mem_usage"])

    def test_qwen3_windows_rejects_oversized_shard_without_path_leak(
        self,
    ) -> None:
        shard = self.qwen_model / "model.safetensors"
        shard.write_bytes(b"xx")
        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=lambda **kwargs: FakeQwenModel(),
            device_map="cpu",
            torch_dtype="float32",
        )

        with (
            mock.patch.object(
                production_runners,
                "_is_windows_runtime",
                return_value=True,
            ),
            mock.patch.object(
                production_runners,
                "_WINDOWS_SAFE_SHARD_LIMIT_BYTES",
                1,
            ),
            self.assertRaises(WorkerError) as captured,
        ):
            adapter.transcribe_batch(
                self.prepared(),
                self.prepared().windows,
                self.context,
                requested_language="auto",
            )

        self.assertEqual(
            captured.exception.code,
            "QWEN3_CHECKPOINT_RESHARD_REQUIRED",
        )
        serialized = json.dumps(captured.exception.as_payload())
        self.assertNotIn(str(self.qwen_model), serialized)
        self.assertFalse(captured.exception.details["modelQualityChanged"])

    def test_qwen3_converts_windows_1455_without_path_leak(self) -> None:
        private_path = str(self.qwen_model)

        def factory(**kwargs):
            error = OSError(f"{private_path}: os error 1455")
            error.winerror = 1455
            raise error

        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=factory,
            device_map="cpu",
            torch_dtype="float32",
        )
        with (
            mock.patch.object(
                production_runners,
                "_is_windows_runtime",
                return_value=True,
            ),
            self.assertRaises(WorkerError) as captured,
        ):
            adapter.transcribe_batch(
                self.prepared(),
                self.prepared().windows,
                self.context,
                requested_language="auto",
            )

        self.assertEqual(
            captured.exception.code,
            "QWEN3_WINDOWS_COMMIT_EXHAUSTED",
        )
        serialized = json.dumps(captured.exception.as_payload())
        self.assertNotIn(private_path, serialized)

    def test_qwen3_serializes_inference_calls(self) -> None:
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        class SerialModel:
            def transcribe(self, *, audio, return_time_stamps, language=None):
                nonlocal active, maximum_active
                with state_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                with state_lock:
                    active -= 1
                return [
                    SimpleNamespace(
                        text=f"window {index}",
                        language="English",
                        time_stamps=None,
                    )
                    for index in range(len(audio))
                ]

        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=lambda **kwargs: SerialModel(),
            device_map="cpu",
            torch_dtype="float32",
        )

        def run(_index: int):
            return adapter.transcribe_batch(
                self.prepared(),
                self.prepared().windows,
                self.context,
                requested_language="auto",
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(run, range(3)))

        self.assertEqual(len(results), 3)
        self.assertEqual(maximum_active, 1)

    def test_qwen3_release_resources_is_idempotent_and_reloads_lazily(
        self,
    ) -> None:
        models: list[FakeQwenModel] = []

        def factory(**_kwargs):
            model = FakeQwenModel()
            models.append(model)
            return model

        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=factory,
            device_map="cpu",
            torch_dtype="float32",
        )
        adapter.transcribe_batch(
            self.prepared(),
            self.prepared().windows,
            self.context,
            requested_language="auto",
        )
        self.assertEqual(len(models), 1)
        self.assertIs(adapter._model_instance, models[0])

        with mock.patch.object(
            production_runners,
            "_release_accelerator_memory",
        ) as release_memory:
            adapter.release_resources()
            adapter.release_resources()

        self.assertIsNone(adapter._model_instance)
        self.assertEqual(release_memory.call_count, 2)

        adapter.transcribe_batch(
            self.prepared(),
            self.prepared().windows,
            self.context,
            requested_language="auto",
        )
        self.assertEqual(len(models), 2)
        self.assertIs(adapter._model_instance, models[1])

    def test_qwen3_runner_normalizes_or_defaults_result_languages(self) -> None:
        result_languages = ("pt_br", None, "not a valid tag")

        class LanguageQwenModel:
            def transcribe(
                self,
                *,
                audio,
                return_time_stamps,
                language=None,
            ):
                return [
                    SimpleNamespace(
                        text=f"Source text {index + 1}",
                        language=result_languages[index],
                        time_stamps=None,
                    )
                    for index in range(len(audio))
                ]

        prepared = PreparedAudio(
            duration_ms=4000,
            source_fingerprint="b" * 64,
            normalization_profile="mono-16khz-f32-v1",
            windows=(
                SpeechWindow("window-1", 0, 1000),
                SpeechWindow("window-2", 1000, 2000),
                SpeechWindow("window-3", 2000, 3000),
            ),
            stage_durations_ms={
                "decode": 0.0,
                "normalize": 0.0,
                "vad": 0.0,
                "boundary": 0.0,
            },
            audio_path=str(self.audio),
        )
        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=lambda **kwargs: LanguageQwenModel(),
            device_map="cpu",
            torch_dtype="float32",
        )

        results = adapter.transcribe_batch(
            prepared,
            prepared.windows,
            self.context,
            requested_language="auto",
        )

        self.assertEqual(
            [item.evidence["language"] for item in results],
            ["pt-BR", "und", "und"],
        )

    def test_qwen_and_cam_reuse_one_pcm_load_without_serializing_pcm(self) -> None:
        prepared = self.prepared()
        pcm = FakePcmTimeline(64_000)
        qwen_slices: list[FakePcmSlice] = []
        cam_slices: list[FakePcmSlice] = []

        class CapturingQwenModel:
            def transcribe(
                self,
                *,
                audio,
                return_time_stamps,
                language=None,
            ):
                qwen_slices.extend(item[0] for item in audio)
                return [
                    SimpleNamespace(
                        text=f"中文窗口{index + 1}",
                        language="Chinese",
                        time_stamps=None,
                    )
                    for index in range(len(audio))
                ]

        class CapturingCamModel:
            def generate(self, *, input, batch_size, disable_pbar):
                cam_slices.extend(input)
                return [
                    {"spk_embedding": [[float(index + 1), 1.0]]}
                    for index in range(len(input))
                ]

        qwen = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=lambda **kwargs: CapturingQwenModel(),
            device_map="cpu",
            torch_dtype="float32",
        )
        cam = LocalFunAsrCamPlusAdapter(
            model_path=self.cam_model,
            model_factory=lambda **kwargs: CapturingCamModel(),
            device="cpu",
        )

        with mock.patch.object(
            production_runners,
            "_load_audio",
            return_value=(pcm, 16_000),
        ) as load_audio:
            asr_results = qwen.transcribe_batch(
                prepared,
                prepared.windows,
                self.context,
                requested_language="auto",
            )
            embedding_results = cam.embed_batch(
                prepared,
                prepared.windows,
                self.context,
            )

        load_audio.assert_called_once_with(prepared.audio_path)
        self.assertTrue(qwen_slices)
        self.assertTrue(cam_slices)
        self.assertTrue(
            all(item.owner is pcm for item in (*qwen_slices, *cam_slices))
        )
        pcm_buffer_ids = {
            item.evidence["pcmBufferId"]
            for item in (*asr_results, *embedding_results)
        }
        self.assertEqual(len(pcm_buffer_ids), 1)
        self.assertTrue(next(iter(pcm_buffer_ids)).startswith("pcm16k:"))

        evidence_json = json.dumps(
            {
                "asr": [dict(item.evidence) for item in asr_results],
                "embeddings": [
                    dict(item.evidence) for item in embedding_results
                ],
            },
            ensure_ascii=False,
        )
        self.assertNotIn(FakePcmTimeline.marker, evidence_json)
        self.assertNotIn("slice_calls", evidence_json)

    def test_shared_pcm_store_single_flight_and_bounded_lru(self) -> None:
        store = production_runners._SharedPcmStore(
            max_entries=2,
            max_bytes=8,
        )
        pcm = FakePcmTimeline(1, nbytes=4)
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        loader_calls: list[int] = []
        loader_lock = threading.Lock()

        def loader():
            with loader_lock:
                loader_calls.append(1)
            time.sleep(0.03)
            return pcm, 16_000

        def load_from_worker(_index: int):
            barrier.wait(timeout=2)
            return store.get_or_load("shared", loader)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(load_from_worker, range(worker_count)))

        self.assertEqual(len(loader_calls), 1)
        self.assertTrue(all(samples is pcm for samples, _rate in results))
        self.assertTrue(all(rate == 16_000 for _samples, rate in results))
        self.assertEqual(
            store.snapshot(),
            {
                "entries": 1,
                "bytes": 4,
                "maxEntries": 2,
                "maxBytes": 8,
            },
        )

        lru = production_runners._SharedPcmStore(
            max_entries=2,
            max_bytes=8,
        )
        pcm_a = FakePcmTimeline(1, nbytes=4)
        pcm_b = FakePcmTimeline(1, nbytes=4)
        pcm_c = FakePcmTimeline(1, nbytes=4)
        lru.put("a", pcm_a, 16_000)
        lru.put("b", pcm_b, 16_000)

        def unexpected_loader():
            raise AssertionError("recently used LRU entry should remain cached")

        cached_a, _rate = lru.get_or_load("a", unexpected_loader)
        self.assertIs(cached_a, pcm_a)
        lru.put("c", pcm_c, 16_000)
        self.assertEqual(
            lru.snapshot(),
            {
                "entries": 2,
                "bytes": 8,
                "maxEntries": 2,
                "maxBytes": 8,
            },
        )

        reload_calls: list[int] = []

        def reload_b():
            reload_calls.append(1)
            return pcm_b, 16_000

        reloaded_b, _rate = lru.get_or_load("b", reload_b)
        self.assertIs(reloaded_b, pcm_b)
        self.assertEqual(reload_calls, [1])
        self.assertEqual(lru.snapshot()["entries"], 2)
        self.assertEqual(lru.snapshot()["bytes"], 8)

    def test_qwen3_forced_alignment_uses_items_seconds_offset_and_clamp(self) -> None:
        class ForcedAlignmentModel:
            def transcribe(
                self,
                *,
                audio,
                return_time_stamps,
                language=None,
            ):
                self.return_time_stamps = return_time_stamps
                return [
                    SimpleNamespace(
                        text="中文原文",
                        language="Chinese",
                        time_stamps=SimpleNamespace(
                            items=[
                                SimpleNamespace(
                                    text="前",
                                    start_time=-0.25,
                                    end_time=0.125,
                                ),
                                SimpleNamespace(
                                    text="中",
                                    start_time=0.5,
                                    end_time=2.0,
                                ),
                                SimpleNamespace(
                                    text="逆",
                                    start_time=0.9,
                                    end_time=0.2,
                                ),
                            ]
                        ),
                    )
                ]

        model = ForcedAlignmentModel()
        adapter = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            forced_aligner_path=self.forced_aligner_model,
            model_factory=lambda **kwargs: model,
            device_map="cpu",
            torch_dtype="float32",
        )
        result = adapter.transcribe_batch(
            self.prepared(),
            [SpeechWindow("window-offset", 1000, 2000)],
            self.context,
            requested_language="auto",
        )
        self.assertTrue(model.return_time_stamps)
        self.assertEqual(
            result[0].evidence["timestamps"],
            [
                {"text": "前", "startMs": 1000, "endMs": 1125},
                {"text": "中", "startMs": 1500, "endMs": 2000},
                {"text": "逆", "startMs": 1900, "endMs": 1900},
            ],
        )

    def test_qwen3_forced_alignment_fails_closed_on_malformed_timestamps(self) -> None:
        malformed_values = (
            SimpleNamespace(time_stamps=SimpleNamespace()),
            SimpleNamespace(
                time_stamps=SimpleNamespace(
                    items=[
                        SimpleNamespace(
                            text="坏",
                            start_time=True,
                            end_time=0.1,
                        )
                    ]
                )
            ),
            SimpleNamespace(
                time_stamps=SimpleNamespace(
                    items=[
                        SimpleNamespace(
                            text="坏",
                            start_time=float("nan"),
                            end_time=0.1,
                        )
                    ]
                )
            ),
            SimpleNamespace(
                time_stamps=SimpleNamespace(
                    items=[
                        SimpleNamespace(
                            text="坏",
                            start_time=0.0,
                            end_time=float("inf"),
                        )
                    ]
                )
            ),
        )
        for index, malformed in enumerate(malformed_values):
            with self.subTest(index=index):
                malformed.text = "中文原文"
                malformed.language = "Chinese"

                class MalformedModel:
                    def transcribe(self, **kwargs):
                        return [malformed]

                adapter = LocalQwen3AsrAdapter(
                    model_path=self.qwen_model,
                    forced_aligner_path=self.forced_aligner_model,
                    model_factory=lambda **kwargs: MalformedModel(),
                    device_map="cpu",
                    torch_dtype="float32",
                )
                with self.assertRaises(WorkerError) as captured:
                    adapter.transcribe_batch(
                        self.prepared(),
                        [SpeechWindow("window-offset", 1000, 2000)],
                        self.context,
                        requested_language="auto",
                    )
                self.assertEqual(
                    captured.exception.code,
                    "QWEN3_ASR_TIMESTAMP_INVALID",
                )

    def test_cam_plus_runner_batches_every_requested_window(self) -> None:
        captured = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return FakeCamModel()

        adapter = LocalFunAsrCamPlusAdapter(
            model_path=self.cam_model,
            model_factory=factory,
            device="cpu",
        )
        results = adapter.embed_batch(
            self.prepared(),
            self.prepared().windows,
            self.context,
        )
        self.assertEqual(captured["model"], str(self.cam_model.resolve()))
        self.assertEqual([item.window_id for item in results], ["window-1", "window-2"])
        self.assertEqual(results[0].vector, (1.0, 1.0))
        self.assertEqual(results[1].vector, (2.0, 1.0))

    def test_cam_plus_model_load_oom_is_structured_and_path_free(self) -> None:
        class OutOfMemoryError(RuntimeError):
            pass

        private_path = str(self.cam_model)

        def factory(**_kwargs):
            raise OutOfMemoryError(f"{private_path}: CUDA out of memory")

        adapter = LocalFunAsrCamPlusAdapter(
            model_path=self.cam_model,
            model_factory=factory,
            device="cuda:0",
        )

        with self.assertRaises(WorkerError) as captured:
            adapter.embed_batch(
                self.prepared(),
                self.prepared().windows,
                self.context,
            )

        self.assertEqual(
            captured.exception.code,
            "CAMPP_ACCELERATOR_MEMORY_EXHAUSTED",
        )
        self.assertEqual(captured.exception.details["phase"], "model-load")
        self.assertFalse(captured.exception.details["modelQualityChanged"])
        self.assertNotIn(
            private_path,
            json.dumps(captured.exception.as_payload()),
        )

    def test_cam_plus_inference_oom_unloads_model_and_is_structured(
        self,
    ) -> None:
        class OutOfMemoryError(RuntimeError):
            pass

        class OomCamModel:
            def generate(self, **_kwargs):
                raise OutOfMemoryError("CUDA out of memory")

        model = OomCamModel()
        adapter = LocalFunAsrCamPlusAdapter(
            model_path=self.cam_model,
            model_factory=lambda **_kwargs: model,
            device="cuda:0",
            embedding_batch_size=2,
        )

        with self.assertRaises(WorkerError) as captured:
            adapter.embed_batch(
                self.prepared(),
                self.prepared().windows,
                self.context,
            )

        self.assertEqual(
            captured.exception.code,
            "CAMPP_ACCELERATOR_MEMORY_EXHAUSTED",
        )
        self.assertEqual(
            captured.exception.details["phase"],
            "embedding-inference",
        )
        self.assertEqual(
            captured.exception.details["embeddingBatchSize"],
            2,
        )
        self.assertIsNone(adapter._model_instance)

    def test_cam_plus_bounded_batches_lazy_model_and_stable_output_order(
        self,
    ) -> None:
        windows = tuple(
            SpeechWindow(
                f"window-{index + 1}",
                index * 500,
                (index + 1) * 500,
            )
            for index in range(5)
        )
        prepared = PreparedAudio(
            duration_ms=4000,
            source_fingerprint="b" * 64,
            normalization_profile="mono-16khz-f32-v1",
            windows=windows,
            stage_durations_ms={
                "decode": 0.0,
                "normalize": 0.0,
                "vad": 0.0,
                "boundary": 0.0,
            },
            audio_path=str(self.audio),
        )
        pcm = FakePcmTimeline(64_000)
        factory_calls: list[dict[str, object]] = []
        generate_calls: list[tuple[int, tuple[int, ...]]] = []

        class OrderedCamModel:
            def generate(self, *, input, batch_size, disable_pbar):
                generate_calls.append(
                    (batch_size, tuple(item.start for item in input))
                )
                return [
                    {
                        "spk_embedding": [
                            [float(item.start), float(item.end)]
                        ]
                    }
                    for item in input
                ]

        model = OrderedCamModel()

        def factory(**kwargs):
            factory_calls.append(kwargs)
            return model

        adapter = LocalFunAsrCamPlusAdapter(
            model_path=self.cam_model,
            model_factory=factory,
            device="cpu",
            embedding_batch_size=2,
        )
        with mock.patch.object(
            production_runners,
            "_load_audio",
            return_value=(pcm, 16_000),
        ) as load_audio:
            first = adapter.embed_batch(prepared, windows, self.context)
            second = adapter.embed_batch(prepared, windows, self.context)

        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(
            [size for size, _starts in generate_calls],
            [2, 2, 1, 2, 2, 1],
        )
        self.assertEqual(
            [starts for _size, starts in generate_calls[:3]],
            [(0, 8000), (16000, 24000), (32000,)],
        )
        self.assertEqual(
            [item.window_id for item in first],
            [item.window_id for item in windows],
        )
        self.assertEqual(
            [item.vector for item in first],
            [item.vector for item in second],
        )
        self.assertEqual(
            [item.vector for item in first],
            [
                (0.0, 8000.0),
                (8000.0, 16000.0),
                (16000.0, 24000.0),
                (24000.0, 32000.0),
                (32000.0, 40000.0),
            ],
        )
        self.assertTrue(
            all(item.evidence["embeddingBatchSize"] == 2 for item in first)
        )
        pcm_buffer_ids = {
            item.evidence["pcmBufferId"] for item in (*first, *second)
        }
        self.assertEqual(len(pcm_buffer_ids), 1)
        load_audio.assert_called_once_with(prepared.audio_path)

    def test_cam_plus_multiresolution_refinement_preserves_strong_a_b_a(
        self,
    ) -> None:
        prepared = PreparedAudio(
            duration_ms=4000,
            source_fingerprint="c" * 64,
            normalization_profile="mono-16khz-f32-v1",
            windows=(SpeechWindow("vad-1", 0, 3600),),
            stage_durations_ms={
                "decode": 0.0,
                "normalize": 0.0,
                "vad": 0.0,
                "boundary": 0.0,
            },
            audio_path=str(self.audio),
        )
        pcm = FakePcmTimeline(64_000)
        factory_calls: list[dict[str, object]] = []
        generate_batches: list[int] = []

        class AbaCamModel:
            def generate(self, *, input, batch_size, disable_pbar):
                generate_batches.append(batch_size)
                output = []
                for item in input:
                    center_ms = (item.start + item.end) * 1000 / 32_000
                    vector = (
                        [0.0, 1.0]
                        if 1200 <= center_ms < 2400
                        else [1.0, 0.0]
                    )
                    output.append({"spk_embedding": [vector]})
                return output

        model = AbaCamModel()

        def factory(**kwargs):
            factory_calls.append(kwargs)
            return model

        adapter = LocalFunAsrCamPlusAdapter(
            model_path=self.cam_model,
            model_factory=factory,
            device="cpu",
            embedding_batch_size=3,
        )
        with mock.patch.object(
            production_runners,
            "_load_audio",
            return_value=(pcm, 16_000),
        ) as load_audio:
            refined = adapter.refine_windows(prepared, self.context)
            full_turn_embeddings = adapter.embed_batch(
                refined,
                refined.windows,
                self.context,
            )

        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(generate_batches[:4], [3, 3, 3, 1])
        self.assertEqual(generate_batches[4:], [3])
        self.assertEqual(
            [window.window_id for window in refined.windows],
            ["vad-1.sc01", "vad-1.sc02", "vad-1.sc03"],
        )
        self.assertEqual(
            [
                (window.start_ms, window.end_ms)
                for window in refined.windows
            ],
            [(0, 1162), (1162, 2287), (2287, 3600)],
        )
        self.assertEqual(
            [item.window_id for item in full_turn_embeddings],
            ["vad-1.sc01", "vad-1.sc02", "vad-1.sc03"],
        )
        for left, right in zip(refined.windows, refined.windows[1:]):
            self.assertEqual(left.end_ms, right.start_ms)
        evidence = refined.windows[0].metadata[
            "speakerChangeRefinement"
        ]
        self.assertEqual(evidence["automaticSplitsMs"], [1162, 2287])
        self.assertFalse(evidence["reviewRequired"])
        self.assertEqual(
            refined.windows[1].metadata["turnId"],
            "turn:vad-1:1162-2287",
        )
        self.assertEqual(
            evidence["plans"]["fine"]["automaticSplitsMs"],
            [1162, 2287],
        )
        self.assertGreater(
            refined.stage_durations_ms["speakerChangeRefinement"],
            0.0,
        )
        self.assertNotIn(FakePcmTimeline.marker, json.dumps(refined.as_dict()))
        load_audio.assert_called_once_with(prepared.audio_path)

    def test_language_windows_use_energy_valleys_and_detect_code_switches(
        self,
    ) -> None:
        prepared = PreparedAudio(
            duration_ms=30_000,
            source_fingerprint="9" * 64,
            normalization_profile="mono-16khz-f32-v1",
            windows=(
                SpeechWindow(
                    "vad-code-switch",
                    0,
                    30_000,
                    metadata={"turnId": "turn-code-switch"},
                ),
            ),
            stage_durations_ms={
                "decode": 0.0,
                "normalize": 0.0,
                "vad": 0.0,
                "boundary": 0.0,
            },
            audio_path=str(self.audio),
        )
        pcm = FakePcmTimeline(480_000)

        class StableCamModel:
            def generate(self, *, input, batch_size, disable_pbar):
                return [
                    {"spk_embedding": [[1.0, 0.0]]}
                    for _item in input
                ]

        class CodeSwitchQwenModel:
            def transcribe(
                self,
                *,
                audio,
                return_time_stamps,
                language=None,
            ):
                results = []
                for index, (clip, _rate) in enumerate(audio):
                    center_ms = (clip.start + clip.end) * 1000 / 32_000
                    detected = (
                        "English"
                        if center_ms < 10_000
                        else "Chinese"
                        if center_ms < 22_000
                        else "Spanish"
                    )
                    results.append(
                        SimpleNamespace(
                            text=f"code-switch-{index + 1}",
                            language=detected,
                            time_stamps=None,
                        )
                    )
                return results

        cam = LocalFunAsrCamPlusAdapter(
            model_path=self.cam_model,
            model_factory=lambda **_kwargs: StableCamModel(),
            device="cpu",
        )
        qwen = LocalQwen3AsrAdapter(
            model_path=self.qwen_model,
            model_factory=lambda **_kwargs: CodeSwitchQwenModel(),
            device_map="cpu",
            torch_dtype="float32",
        )
        valleys = (
            EnergyValley(11_200, 0.80, "language-valley-1"),
            EnergyValley(22_000, 0.90, "language-valley-2"),
        )
        with (
            mock.patch.object(
                production_runners,
                "_load_audio",
                return_value=(pcm, 16_000),
            ),
            mock.patch.object(
                cam,
                "_energy_valleys",
                return_value=valleys,
            ),
            mock.patch.object(
                cam,
                "_automatic_splits",
                return_value=(),
            ),
        ):
            refined = cam.refine_windows(prepared, self.context)
            hypotheses = qwen.transcribe_batch(
                refined,
                refined.windows,
                self.context,
                requested_language="auto",
            )

        self.assertEqual(
            [
                (window.start_ms, window.end_ms)
                for window in refined.windows
            ],
            [(0, 11_200), (11_200, 22_000), (22_000, 30_000)],
        )
        self.assertTrue(
            all(
                window.end_ms - window.start_ms <= 12_000
                for window in refined.windows
            )
        )
        self.assertEqual(
            {window.metadata["turnId"] for window in refined.windows},
            {"turn-code-switch"},
        )
        self.assertEqual(
            [item.evidence["language"] for item in hypotheses],
            ["en", "zh", "es"],
        )
        evidence = refined.windows[0].metadata[
            "speakerChangeRefinement"
        ]
        self.assertEqual(evidence["speakerChangeSplitsMs"], [])
        self.assertEqual(
            evidence["languageDurationSplitsMs"],
            [11_200, 22_000],
        )
        self.assertEqual(evidence["appliedSplitsMs"], [11_200, 22_000])
        self.assertEqual(
            [
                item["localizer"]
                for item in evidence["languageSplitEvidence"]
            ],
            ["energy-valley", "energy-valley"],
        )

    def test_cam_plus_refinement_keeps_review_only_evidence_without_split(
        self,
    ) -> None:
        scenarios = (
            (
                "low-confidence",
                {"speakerChangeEmbeddingConfidence": 0.25},
                "LOW_EMBEDDING_CONFIDENCE",
            ),
            (
                "overlap-risk",
                {"overlapRisk": True},
                "OVERLAP_RISK_NOT_EVALUATED",
            ),
        )

        class SingleChangeCamModel:
            def generate(self, *, input, batch_size, disable_pbar):
                return [
                    {
                        "spk_embedding": [
                            (
                                [1.0, 0.0]
                                if (item.start + item.end) < 32_000
                                else [0.0, 1.0]
                            )
                        ]
                    }
                    for item in input
                ]

        for suffix, metadata, expected_reason in scenarios:
            with self.subTest(scenario=suffix):
                production_runners._SHARED_PCM_STORE.clear()
                prepared = PreparedAudio(
                    duration_ms=4000,
                    source_fingerprint=(
                        "d" * 63 + ("1" if suffix == "low-confidence" else "2")
                    ),
                    normalization_profile="mono-16khz-f32-v1",
                    windows=(
                        SpeechWindow(
                            f"vad-{suffix}",
                            0,
                            3000,
                            metadata=metadata,
                        ),
                    ),
                    stage_durations_ms={
                        "decode": 0.0,
                        "normalize": 0.0,
                        "vad": 0.0,
                        "boundary": 0.0,
                    },
                    audio_path=str(self.audio),
                )
                adapter = LocalFunAsrCamPlusAdapter(
                    model_path=self.cam_model,
                    model_factory=lambda **_kwargs: SingleChangeCamModel(),
                    device="cpu",
                )
                with mock.patch.object(
                    production_runners,
                    "_load_audio",
                    return_value=(FakePcmTimeline(64_000), 16_000),
                ):
                    refined = adapter.refine_windows(
                        prepared,
                        self.context,
                    )

                self.assertEqual(len(refined.windows), 1)
                self.assertEqual(
                    refined.windows[0].window_id,
                    f"vad-{suffix}",
                )
                evidence = refined.windows[0].metadata[
                    "speakerChangeRefinement"
                ]
                self.assertTrue(evidence["reviewRequired"])
                self.assertEqual(evidence["automaticSplitsMs"], [])
                proposals = [
                    proposal
                    for plan in evidence["plans"].values()
                    for proposal in plan["proposals"]
                ]
                self.assertTrue(proposals)
                self.assertTrue(
                    any(
                        expected_reason in proposal["reviewReasons"]
                        for proposal in proposals
                    )
                )
                self.assertTrue(
                    all(
                        not proposal["applyAutomatically"]
                        for proposal in proposals
                    )
                )

    def test_cam_plus_refinement_rejects_short_middle_turn_and_is_stable(
        self,
    ) -> None:
        prepared = PreparedAudio(
            duration_ms=4000,
            source_fingerprint="e" * 64,
            normalization_profile="mono-16khz-f32-v1",
            windows=(SpeechWindow("vad-short", 0, 3000),),
            stage_durations_ms={
                "decode": 0.0,
                "normalize": 0.0,
                "vad": 0.0,
                "boundary": 0.0,
            },
            audio_path=str(self.audio),
        )

        class ShortAbaCamModel:
            def generate(self, *, input, batch_size, disable_pbar):
                output = []
                for item in input:
                    center_ms = (item.start + item.end) * 1000 / 32_000
                    vector = (
                        [0.0, 1.0]
                        if 1500 <= center_ms < 1900
                        else [1.0, 0.0]
                    )
                    output.append({"spk_embedding": [vector]})
                return output

        adapter = LocalFunAsrCamPlusAdapter(
            model_path=self.cam_model,
            model_factory=lambda **_kwargs: ShortAbaCamModel(),
            device="cpu",
        )
        with mock.patch.object(
            production_runners,
            "_load_audio",
            return_value=(FakePcmTimeline(64_000), 16_000),
        ):
            first = adapter.refine_windows(prepared, self.context)
            second = adapter.refine_windows(prepared, self.context)

        self.assertEqual([item.as_dict() for item in first.windows], [
            item.as_dict() for item in second.windows
        ])
        self.assertEqual(len(first.windows), 1)
        evidence = first.windows[0].metadata["speakerChangeRefinement"]
        self.assertEqual(evidence["automaticSplitsMs"], [])
        fine_proposals = evidence["plans"]["fine"]["proposals"]
        self.assertEqual(len(fine_proposals), 2)
        self.assertTrue(
            all(
                "SHORT_RESULTING_INTERVAL" in item["reviewReasons"]
                for item in fine_proposals
            )
        )

    def test_eres2netv2_uses_only_candidate_and_top2_exemplars(self) -> None:
        def segment(
            segment_id: str,
            start_ms: int,
            speaker_id: str,
            scores: tuple[float, float],
            margin: float,
        ) -> TranscriptSegment:
            return TranscriptSegment(
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=start_ms + 1000,
                speaker_id=speaker_id,
                raw_text="中文原文",
                normalized_text="中文原文",
                display_text="中文原文",
                confidence=0.9,
                speaker_scores=(
                    SpeakerScore("speaker-1", scores[0]),
                    SpeakerScore("speaker-2", scores[1]),
                ),
                speaker_margin=margin,
                evidence={"preparation": {"audioPath": str(self.audio)}},
            )

        segments = {
            "candidate": segment("candidate", 1000, "speaker-1", (0.55, 0.50), 0.05),
            "reference-1": segment("reference-1", 0, "speaker-1", (0.95, 0.05), 0.90),
            "reference-2": segment("reference-2", 3000, "speaker-2", (0.05, 0.95), 0.90),
        }
        factory_calls = []

        def factory(**kwargs):
            factory_calls.append(kwargs)
            return FakeEResPipeline()

        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="cpu",
            decision_margin=0.05,
            pipeline_factory=factory,
        )
        candidate = ReviewCandidate(
            "candidate",
            ("LOW_MARGIN",),
            False,
        )
        proposals = verifier.review_batch(
            [candidate],
            segments,
            self.context,
        )
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(proposals[0].speaker_id, "speaker-2")
        self.assertEqual(
            proposals[0].exit_reason,
            "VERIFIED_SPEAKER_CHANGE",
        )
        cache_material = verifier.cache_material(candidate, segments)
        self.assertEqual(len(cache_material["references"]), 2)

    def test_eres2netv2_mps_loads_through_cpu_then_moves_embedding_model(
        self,
    ) -> None:
        class EmbeddingModel:
            def __init__(self) -> None:
                self.devices: list[str] = []
                self.eval_calls = 0

            def to(self, device):
                self.devices.append(str(device))
                return self

            def eval(self):
                self.eval_calls += 1
                return self

        model = SimpleNamespace(
            device="cpu",
            embedding_model=EmbeddingModel(),
        )
        pipeline = SimpleNamespace(model=model)
        factory_calls: list[dict[str, object]] = []

        def factory(**kwargs):
            factory_calls.append(kwargs)
            return pipeline

        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="mps",
            pipeline_factory=factory,
        )
        with mock.patch.object(
            production_runners,
            "_mps_is_available",
            return_value=True,
        ):
            loaded = verifier._pipeline()

        self.assertIs(loaded, pipeline)
        self.assertEqual(factory_calls[0]["device"], "cpu")
        self.assertEqual(model.embedding_model.devices, ["mps"])
        self.assertEqual(model.embedding_model.eval_calls, 1)
        self.assertEqual(str(model.device), "mps")

    def test_eres2netv2_mps_unavailable_fails_before_device_transfer(
        self,
    ) -> None:
        pipeline = SimpleNamespace(
            model=SimpleNamespace(
                device="cpu",
                embedding_model=mock.Mock(),
            )
        )
        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="mps",
            pipeline_factory=lambda **_kwargs: pipeline,
        )

        with (
            mock.patch.object(
                production_runners,
                "_mps_is_available",
                return_value=False,
            ),
            self.assertRaises(WorkerError) as captured,
        ):
            verifier._pipeline()

        self.assertEqual(
            captured.exception.code,
            "ERES2NETV2_DEVICE_UNAVAILABLE",
        )
        pipeline.model.embedding_model.to.assert_not_called()
        self.assertIsNone(verifier._pipeline_instance)

    def test_eres2netv2_serializes_shared_pipeline_inference(self) -> None:
        class ConcurrentTrackingPipeline:
            def __init__(self) -> None:
                self.active = 0
                self.maximum_active = 0
                self.lock = threading.Lock()

            def __call__(self, audio, output_emb):
                self.assert_output_emb(output_emb)
                with self.lock:
                    self.active += 1
                    self.maximum_active = max(
                        self.maximum_active,
                        self.active,
                    )
                time.sleep(0.03)
                with self.lock:
                    self.active -= 1
                return {"embs": [[1.0, 0.0] for _ in audio]}

            @staticmethod
            def assert_output_emb(output_emb) -> None:
                if output_emb is not True:
                    raise AssertionError("output_emb=True is required")

        pipeline = ConcurrentTrackingPipeline()
        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="cpu",
            pipeline_factory=lambda **_kwargs: pipeline,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda _index: verifier._embeddings([[0.1, 0.2]]),
                    range(2),
                )
            )

        self.assertEqual(pipeline.maximum_active, 1)
        self.assertEqual(results, [[(1.0, 0.0)], [(1.0, 0.0)]])

    def test_eres2netv2_release_resources_is_idempotent_and_reloads(
        self,
    ) -> None:
        instances = []

        class Pipeline:
            def __call__(self, audio, output_emb):
                return {"embs": [[1.0, 0.0] for _ in audio]}

        def factory(**_kwargs):
            pipeline = Pipeline()
            instances.append(pipeline)
            return pipeline

        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="cpu",
            pipeline_factory=factory,
        )
        verifier._embeddings([[0.1]])

        with mock.patch.object(
            production_runners,
            "_release_accelerator_memory",
        ) as release_memory:
            verifier.release_resources()
            verifier.release_resources()

        self.assertIsNone(verifier._pipeline_instance)
        self.assertEqual(release_memory.call_count, 2)

        verifier._embeddings([[0.2]])
        self.assertEqual(len(instances), 2)
        self.assertIs(verifier._pipeline_instance, instances[-1])

    def test_eres2netv2_release_holds_locks_until_finalize_and_flush(
        self,
    ) -> None:
        finalize_started = threading.Event()
        allow_finalize = threading.Event()
        second_load_started = threading.Event()
        events: list[str] = []
        factory_calls = 0

        class Pipeline:
            def __init__(self, generation: int) -> None:
                self.generation = generation

            def __call__(self, audio, output_emb):
                return {"embs": [[1.0, 0.0] for _ in audio]}

            def __del__(self) -> None:
                if self.generation != 1:
                    return
                events.append("finalize-start")
                finalize_started.set()
                allow_finalize.wait(timeout=2.0)
                events.append("finalize-end")

        def factory(**_kwargs):
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 2:
                second_load_started.set()
                events.append("load-2")
            return Pipeline(factory_calls)

        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="cpu",
            pipeline_factory=factory,
        )
        verifier._embeddings([[0.1]])

        with mock.patch.object(
            production_runners,
            "_release_accelerator_memory",
            side_effect=lambda: events.append("flush"),
        ):
            release_thread = threading.Thread(
                target=verifier.release_resources,
            )
            release_thread.start()
            self.assertTrue(finalize_started.wait(timeout=1.0))

            inference_thread = threading.Thread(
                target=lambda: verifier._embeddings([[0.2]]),
            )
            inference_thread.start()
            self.assertFalse(second_load_started.wait(timeout=0.1))

            allow_finalize.set()
            release_thread.join(timeout=2.0)
            inference_thread.join(timeout=2.0)

        self.assertFalse(release_thread.is_alive())
        self.assertFalse(inference_thread.is_alive())
        self.assertEqual(
            events[:4],
            ["finalize-start", "finalize-end", "flush", "load-2"],
        )

    def test_eres2netv2_model_load_oom_is_retryable_and_path_free(
        self,
    ) -> None:
        class OutOfMemoryError(RuntimeError):
            pass

        private_path = str(self.eres_model)

        def factory(**_kwargs):
            raise OutOfMemoryError(
                f"{private_path}: CUDA out of memory"
            )

        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="gpu",
            pipeline_factory=factory,
        )

        with mock.patch.object(
            production_runners,
            "_release_accelerator_memory",
        ) as release_memory:
            with self.assertRaises(WorkerError) as captured:
                verifier._embeddings([[0.1]])

        error = captured.exception
        self.assertEqual(
            error.code,
            "ERES2NETV2_ACCELERATOR_MEMORY_EXHAUSTED",
        )
        self.assertTrue(error.retryable)
        self.assertEqual(error.details["phase"], "model-load")
        self.assertEqual(error.details["clipCount"], 0)
        self.assertFalse(error.details["modelQualityChanged"])
        self.assertNotIn(private_path, json.dumps(error.as_payload()))
        self.assertIsNone(verifier._pipeline_instance)
        release_memory.assert_called_once_with()

    def test_eres2netv2_cpu_memory_error_is_host_memory_failure(
        self,
    ) -> None:
        class HostOomPipeline:
            def __call__(self, audio, output_emb):
                raise MemoryError("host allocation failed")

        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="cpu",
            pipeline_factory=lambda **_kwargs: HostOomPipeline(),
        )

        with self.assertRaises(WorkerError) as captured:
            verifier._embeddings([[0.1]])

        error = captured.exception
        self.assertEqual(
            error.code,
            "ERES2NETV2_HOST_MEMORY_EXHAUSTED",
        )
        self.assertFalse(error.retryable)
        self.assertEqual(error.details["requestedDevice"], "cpu")
        self.assertNotIn("accelerator", str(error).casefold())
        self.assertIsNone(verifier._pipeline_instance)

    def test_eres2netv2_inference_oom_finalizes_before_allocator_flush(
        self,
    ) -> None:
        events: list[str] = []

        class OutOfMemoryError(RuntimeError):
            pass

        class OomPipeline:
            def __call__(self, audio, output_emb):
                raise OutOfMemoryError("CUDA out of memory")

            def __del__(self) -> None:
                events.append("finalize")

        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="gpu",
            pipeline_factory=lambda **_kwargs: OomPipeline(),
        )

        with mock.patch.object(
            production_runners,
            "_release_accelerator_memory",
            side_effect=lambda: events.append("flush"),
        ):
            with self.assertRaises(WorkerError):
                verifier._embeddings([[0.1]])

        self.assertEqual(events, ["finalize", "flush"])

    def test_eres2netv2_inference_oom_unloads_shared_pipeline(
        self,
    ) -> None:
        class OutOfMemoryError(RuntimeError):
            pass

        class OomPipeline:
            def __call__(self, audio, output_emb):
                raise OutOfMemoryError("CUDA out of memory")

        verifier = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="gpu",
            pipeline_factory=lambda **_kwargs: OomPipeline(),
        )

        with mock.patch.object(
            production_runners,
            "_release_accelerator_memory",
        ) as release_memory:
            with self.assertRaises(WorkerError) as captured:
                verifier._embeddings([[0.1], [0.2]])

        error = captured.exception
        self.assertEqual(
            error.code,
            "ERES2NETV2_ACCELERATOR_MEMORY_EXHAUSTED",
        )
        self.assertTrue(error.retryable)
        self.assertEqual(error.details["phase"], "embedding-inference")
        self.assertEqual(error.details["clipCount"], 2)
        self.assertIsNone(verifier._pipeline_instance)
        release_memory.assert_called_once_with()

    def test_eres2netv2_distinguishes_reference_and_margin_outcomes(self) -> None:
        def segment(
            segment_id: str,
            start_ms: int,
            speaker_id: str,
            scores: tuple[float, float],
        ) -> TranscriptSegment:
            return TranscriptSegment(
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=start_ms + 1000,
                speaker_id=speaker_id,
                raw_text="中文原文",
                normalized_text="中文原文",
                display_text="中文原文",
                confidence=0.9,
                speaker_scores=(
                    SpeakerScore("speaker-1", scores[0]),
                    SpeakerScore("speaker-2", scores[1]),
                ),
                speaker_margin=abs(scores[0] - scores[1]),
                evidence={"preparation": {"audioPath": str(self.audio)}},
            )

        candidate = ReviewCandidate("candidate", ("LOW_MARGIN",), False)
        no_reference_segments = {
            "candidate": segment("candidate", 2000, "speaker-1", (0.51, 0.49))
        }
        factory_calls = []
        no_reference = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="cpu",
            pipeline_factory=lambda **kwargs: factory_calls.append(kwargs),
        ).review_batch([candidate], no_reference_segments, self.context)[0]
        self.assertEqual(no_reference.exit_reason, "NO_REFERENCE")
        self.assertEqual(no_reference.reason_code, "ERES2NETV2_NO_REFERENCE")
        self.assertEqual(factory_calls, [])

        single_reference_segments = {
            **no_reference_segments,
            "reference-1": segment(
                "reference-1",
                0,
                "speaker-1",
                (0.95, 0.05),
            ),
        }
        single_reference = LocalERes2NetV2Verifier(
            model_path=self.eres_model,
            device="cpu",
            pipeline_factory=lambda **kwargs: factory_calls.append(kwargs),
        ).review_batch([candidate], single_reference_segments, self.context)[0]
        self.assertEqual(
            single_reference.exit_reason,
            "SINGLE_REFERENCE_INSUFFICIENT",
        )
        self.assertEqual(
            single_reference.reason_code,
            "ERES2NETV2_SINGLE_REFERENCE",
        )
        self.assertEqual(factory_calls, [])

        complete_segments = {
            **single_reference_segments,
            "reference-2": segment(
                "reference-2",
                1000,
                "speaker-2",
                (0.05, 0.95),
            ),
        }
        outcome_cases = (
            (
                [[1.0, 0.0], [0.8, 0.6], [0.8, -0.6]],
                "LOW_MARGIN_UNRESOLVED",
                None,
            ),
            (
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                "VERIFIED_NO_CHANGE",
                None,
            ),
            (
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                "VERIFIED_SPEAKER_CHANGE",
                "speaker-2",
            ),
        )
        for embeddings, exit_reason, speaker_id in outcome_cases:
            with self.subTest(exit_reason=exit_reason):
                verifier = LocalERes2NetV2Verifier(
                    model_path=self.eres_model,
                    device="cpu",
                    decision_margin=0.05,
                    pipeline_factory=lambda **kwargs: FixedEResPipeline(
                        embeddings
                    ),
                )
                proposal = verifier.review_batch(
                    [candidate],
                    complete_segments,
                    self.context,
                )[0]
                self.assertEqual(proposal.exit_reason, exit_reason)
                self.assertEqual(proposal.speaker_id, speaker_id)

    def test_eres2netv2_top2_ranking_has_deterministic_tie_break(self) -> None:
        segment = TranscriptSegment(
            segment_id="candidate",
            start_ms=0,
            end_ms=1000,
            speaker_id="speaker-2",
            raw_text="中文原文",
            normalized_text="中文原文",
            display_text="中文原文",
            confidence=0.9,
            speaker_scores=(
                SpeakerScore("speaker-2", 0.5),
                SpeakerScore("speaker-1", 0.5),
                SpeakerScore("speaker-3", 0.4),
            ),
            speaker_margin=0.0,
        )
        self.assertEqual(
            LocalERes2NetV2Verifier._ranked_speakers(segment),
            ("speaker-1", "speaker-2"),
        )

    def test_pyannote_retains_auditable_turn_overlap_and_conflict_evidence(
        self,
    ) -> None:
        annotation = FakeAnnotation(
            [
                (SimpleNamespace(start=-0.2, end=0.8), "track-1", "LOCAL_A"),
                (SimpleNamespace(start=0.5, end=1.5), "track-2", "LOCAL_B"),
                (SimpleNamespace(start=1.8, end=3.0), "track-3", "LOCAL_A"),
            ]
        )
        pipeline = FakePyannotePipeline(
            SimpleNamespace(speaker_diarization=annotation)
        )
        adapter = LocalPyannoteAuditAdapter(
            model_path=self.pyannote_model,
            device="cpu",
            pipeline_factory=lambda **kwargs: pipeline,
        )
        segment = TranscriptSegment(
            segment_id="candidate",
            start_ms=1000,
            end_ms=3000,
            speaker_id="speaker-1",
            raw_text="中文原文",
            normalized_text="中文原文",
            display_text="中文原文",
            confidence=0.9,
            speaker_scores=(
                SpeakerScore("speaker-1", 0.52),
                SpeakerScore("speaker-2", 0.48),
            ),
            speaker_margin=0.04,
            evidence={"preparation": {"audioPath": str(self.audio)}},
        )
        proposal = adapter.review_batch(
            [ReviewCandidate("candidate", ("LOW_MARGIN",), False)],
            {"candidate": segment},
            self.context,
        )[0]
        self.assertEqual(proposal.reason_code, "PYANNOTE_CONFLICT_PROPOSAL")
        self.assertEqual(proposal.exit_reason, "HUMAN_REVIEW_REQUIRED")
        self.assertIsNone(proposal.speaker_id)
        encoded = next(
            ref for ref in proposal.evidence_refs if ref.startswith("pyannote:")
        )
        evidence = json.loads(encoded.split(":", 2)[2])
        self.assertEqual(
            evidence["speakerTurns"],
            [
                {"startMs": 1000, "endMs": 1800, "localSpeaker": "LOCAL_A"},
                {"startMs": 1500, "endMs": 2500, "localSpeaker": "LOCAL_B"},
                {"startMs": 2800, "endMs": 3000, "localSpeaker": "LOCAL_A"},
            ],
        )
        self.assertEqual(
            evidence["overlapIntervals"],
            [
                {
                    "startMs": 1500,
                    "endMs": 1800,
                    "localSpeakers": ["LOCAL_A", "LOCAL_B"],
                }
            ],
        )
        self.assertEqual(evidence["localSpeakerCount"], 2)
        self.assertFalse(
            evidence["conflictProposal"]["automaticSpeakerOverride"]
        )
        self.assertEqual(len(pipeline.calls), 1)

    def test_pyannote_detects_exact_overlap_intervals_once_per_timeline(
        self,
    ) -> None:
        annotation = FakeAnnotation(
            [
                (SimpleNamespace(start=-0.2, end=0.8), "track-1", "LOCAL_A"),
                (SimpleNamespace(start=0.5, end=1.5), "track-2", "LOCAL_B"),
                (SimpleNamespace(start=1.2, end=2.2), "track-3", "LOCAL_A"),
            ]
        )
        pipeline = FakePyannotePipeline(
            SimpleNamespace(speaker_diarization=annotation)
        )
        adapter = LocalPyannoteAuditAdapter(
            model_path=self.pyannote_model,
            device="cpu",
            pipeline_factory=lambda **kwargs: pipeline,
        )

        decisions = adapter.detect_batch(
            self.prepared(),
            (
                SpeechWindow("window-1", 0, 1000),
                SpeechWindow("window-2", 1000, 2000),
            ),
            self.context,
        )

        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual(
            [decision.window_id for decision in decisions],
            ["window-1", "window-2"],
        )
        self.assertTrue(all(isinstance(item, OverlapDecision) for item in decisions))
        self.assertTrue(all(item.overlapping for item in decisions))
        self.assertEqual(
            decisions[0].evidence["overlapIntervals"],
            [
                {
                    "startMs": 500,
                    "endMs": 800,
                    "localSpeakers": ["LOCAL_A", "LOCAL_B"],
                }
            ],
        )
        self.assertEqual(
            decisions[1].evidence["overlapIntervals"],
            [
                {
                    "startMs": 1200,
                    "endMs": 1500,
                    "localSpeakers": ["LOCAL_A", "LOCAL_B"],
                }
            ],
        )
        self.assertEqual(
            decisions[0].evidence["confidenceKind"],
            "binary-annotation-no-posterior",
        )
        self.assertFalse(decisions[0].evidence["calibratedConfidence"])
        self.assertTrue(decisions[0].evidence["overlapDetectorRun"])
        full_timeline = decisions[0].evidence["fullTimelineInference"]
        self.assertEqual(
            full_timeline,
            decisions[1].evidence["fullTimelineInference"],
        )
        self.assertEqual(full_timeline["scope"], "full-normalized-timeline")
        self.assertEqual(full_timeline["startMs"], 0)
        self.assertEqual(full_timeline["endMs"], 4000)
        self.assertEqual(full_timeline["turnCount"], 3)
        self.assertEqual(full_timeline["localSpeakerCount"], 2)
        self.assertEqual(
            full_timeline["localSpeakers"],
            ["LOCAL_A", "LOCAL_B"],
        )
        self.assertRegex(full_timeline["speakerTurnsSha256"], r"^[0-9a-f]{64}$")

    def test_pyannote_isolated_runtime_receives_one_full_timeline(
        self,
    ) -> None:
        calls: list[dict[str, Any]] = []

        def isolated_runner(**kwargs):
            calls.append(kwargs)
            return [
                {"startMs": 0, "endMs": 800, "localSpeaker": "LOCAL_A"},
                {"startMs": 500, "endMs": 1500, "localSpeaker": "LOCAL_B"},
                {"startMs": 1200, "endMs": 2000, "localSpeaker": "LOCAL_A"},
            ]

        adapter = LocalPyannoteAuditAdapter(
            model_path=self.pyannote_model,
            device="cpu",
            isolated_inference_runner=isolated_runner,
        )
        decisions = adapter.detect_batch(
            self.prepared(),
            (
                SpeechWindow("window-1", 0, 1000),
                SpeechWindow("window-2", 1000, 2000),
            ),
            self.context,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["start_ms"], 0)
        self.assertEqual(calls[0]["end_ms"], 4000)
        self.assertEqual(
            decisions[0].evidence["fullTimelineInference"],
            decisions[1].evidence["fullTimelineInference"],
        )
        self.assertEqual(
            [item.evidence["overlapIntervals"] for item in decisions],
            [
                [
                    {
                        "startMs": 500,
                        "endMs": 800,
                        "localSpeakers": ["LOCAL_A", "LOCAL_B"],
                    }
                ],
                [
                    {
                        "startMs": 1200,
                        "endMs": 1500,
                        "localSpeakers": ["LOCAL_A", "LOCAL_B"],
                    }
                ],
            ],
        )

    def test_pyannote_review_reuses_isolated_overlap_turns(self) -> None:
        def unexpected_runner(**_kwargs):
            raise AssertionError("review must reuse full-timeline overlap evidence")

        adapter = LocalPyannoteAuditAdapter(
            model_path=self.pyannote_model,
            device="cpu",
            isolated_inference_runner=unexpected_runner,
        )
        segment = TranscriptSegment(
            segment_id="candidate",
            start_ms=1000,
            end_ms=3000,
            speaker_id="speaker-1",
            raw_text="中文原文",
            normalized_text="中文原文",
            display_text="中文原文",
            confidence=0.9,
            speaker_scores=(
                SpeakerScore("speaker-1", 0.52),
                SpeakerScore("speaker-2", 0.48),
            ),
            speaker_margin=0.04,
            evidence={
                "preparation": {"audioPath": str(self.audio)},
                "overlap": {
                    "speakerTurns": [
                        {
                            "startMs": 1000,
                            "endMs": 1800,
                            "localSpeaker": "LOCAL_A",
                        },
                        {
                            "startMs": 1500,
                            "endMs": 2500,
                            "localSpeaker": "LOCAL_B",
                        },
                    ]
                },
            },
        )

        proposal = adapter.review_batch(
            [ReviewCandidate("candidate", ("OVERLAP",), False)],
            {"candidate": segment},
            self.context,
        )[0]

        encoded = next(
            ref
            for ref in proposal.evidence_refs
            if ref.startswith("pyannote:")
        )
        evidence = json.loads(encoded.split(":", 2)[2])
        self.assertEqual(evidence["localSpeakerCount"], 2)
        self.assertEqual(len(evidence["overlapIntervals"]), 1)

    def test_pyannote_runtime_initialization_incompatibility_fails_closed(
        self,
    ) -> None:
        def incompatible_factory(**kwargs):
            raise AttributeError("unsupported pyannote API")

        adapter = LocalPyannoteAuditAdapter(
            model_path=self.pyannote_model,
            device="cpu",
            pipeline_factory=incompatible_factory,
        )
        with self.assertRaises(WorkerError) as captured:
            adapter._pipeline()
        self.assertEqual(
            captured.exception.code,
            "PYANNOTE_RUNTIME_INCOMPATIBLE",
        )

    def test_pyannote_release_resources_is_idempotent_and_reloads(
        self,
    ) -> None:
        instances = []

        def factory(**_kwargs):
            pipeline = FakePyannotePipeline(
                SimpleNamespace(speaker_diarization=FakeAnnotation([]))
            )
            instances.append(pipeline)
            return pipeline

        adapter = LocalPyannoteAuditAdapter(
            model_path=self.pyannote_model,
            device="cpu",
            pipeline_factory=factory,
        )
        first = adapter._pipeline()

        with mock.patch.object(
            production_runners,
            "_release_accelerator_memory",
        ) as release_memory:
            adapter.release_resources()
            adapter.release_resources()

        self.assertIsNone(adapter._pipeline_instance)
        self.assertEqual(release_memory.call_count, 2)
        second = adapter._pipeline()
        self.assertIsNot(first, second)
        self.assertEqual(len(instances), 2)

    def test_ffmpeg_funasr_preparation_persists_normalized_audio(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is not available")

        def factory(**kwargs):
            self.assertEqual(kwargs["model"], str(self.vad_model.resolve()))
            self.assertTrue(kwargs["disable_update"])
            self.assertTrue(kwargs["disable_pbar"])
            self.assertEqual(kwargs["ncpu"], 1)
            return FakeVadModel()

        adapter = FfmpegFunAsrPreparationAdapter(
            vad_model_path=self.vad_model,
            ffmpeg_executable=ffmpeg,
            device="cpu",
            model_factory=factory,
        )
        prepared = adapter.prepare(
            self.audio,
            normalization_profile="mono-16khz-f32-v1",
            context=self.context,
        )
        self.assertTrue(Path(prepared.audio_path).is_file())
        self.assertEqual(len(prepared.windows), 2)
        self.assertEqual(
            [(item.start_ms, item.end_ms) for item in prepared.windows],
            [(0, 900), (1000, 1900)],
        )

    def test_funasr_empty_vad_returns_auditable_no_speech_evidence(
        self,
    ) -> None:
        class EmptyVadModel:
            def generate(self, **kwargs):
                return [{"value": []}]

        def normalize(source_path, output_path, context):
            context.raise_if_cancelled()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, output_path)

        adapter = FfmpegFunAsrPreparationAdapter(
            vad_model_path=self.vad_model,
            ffmpeg_executable=Path(__file__),
            model_factory=lambda **_kwargs: EmptyVadModel(),
        )
        with (
            mock.patch.object(adapter, "_normalize", side_effect=normalize),
            mock.patch.object(
                production_runners,
                "_load_audio",
                return_value=(FakePcmTimeline(32_000), 16_000),
            ),
            self.assertRaises(WorkerError) as captured,
        ):
            adapter.prepare(
                self.audio,
                normalization_profile="mono-16khz-f32-v1",
                context=self.context,
            )

        self.assertEqual(captured.exception.code, "NO_SPEECH_DETECTED")
        activity = captured.exception.details["voiceActivity"]
        self.assertEqual(
            activity["classification"],
            "no-speech-candidates-detected",
        )
        self.assertFalse(activity["hasSpeechCandidates"])
        self.assertFalse(activity["hasTranscribableSpeech"])
        self.assertEqual(activity["mediaDurationMs"], 2_000)
        self.assertEqual(activity["speechRatio"], 0.0)

    def test_funasr_vad_is_cpu_isolated_observable_and_stdout_silent(
        self,
    ) -> None:
        factory_calls = []
        observations = []

        def factory(**kwargs):
            print("synthetic third-party model banner")
            factory_calls.append(kwargs)
            return FakeVadModel()

        def normalize(source_path, output_path, context):
            context.raise_if_cancelled()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, output_path)

        adapter = FfmpegFunAsrPreparationAdapter(
            vad_model_path=self.vad_model,
            ffmpeg_executable=Path(__file__),
            model_factory=factory,
            stage_observer=observations.append,
        )
        output = io.StringIO()
        samples = FakePcmTimeline(32_000)
        with (
            mock.patch.object(adapter, "_normalize", side_effect=normalize),
            mock.patch.object(
                production_runners,
                "_load_audio",
                return_value=(samples, 16_000),
            ),
            redirect_stdout(output),
        ):
            prepared = adapter.prepare(
                self.audio,
                normalization_profile="mono-16khz-f32-v1",
                context=self.context,
            )

        self.assertEqual(output.getvalue(), "")
        self.assertEqual(adapter.device, "cpu")
        self.assertEqual(
            factory_calls,
            [
                {
                    "model": str(self.vad_model.resolve()),
                    "device": "cpu",
                    "disable_update": True,
                    "disable_pbar": True,
                    "ncpu": 1,
                }
            ],
        )
        self.assertEqual(
            [
                (event["stage"], event["status"])
                for event in observations
            ],
            [
                ("normalize", "started"),
                ("normalize", "completed"),
                ("pcm_load", "started"),
                ("pcm_load", "completed"),
                ("model_load", "started"),
                ("model_load", "completed"),
                ("inference", "started"),
                ("inference", "completed"),
            ],
        )
        self.assertEqual(prepared.duration_ms, 2_000)
        self.assertEqual(
            observations[-1]["speechWindowCount"],
            len(prepared.windows),
        )
        self.assertTrue(
            all(event["device"] == "cpu" for event in observations)
        )

    def test_funasr_vad_failure_is_observed_and_stops_the_adapter(
        self,
    ) -> None:
        observations = []
        model = FailingVadModel()

        def normalize(source_path, output_path, context):
            context.raise_if_cancelled()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, output_path)

        adapter = FfmpegFunAsrPreparationAdapter(
            vad_model_path=self.vad_model,
            ffmpeg_executable=Path(__file__),
            model_factory=lambda **_kwargs: model,
            stage_observer=observations.append,
        )
        with (
            mock.patch.object(adapter, "_normalize", side_effect=normalize),
            mock.patch.object(
                production_runners,
                "_load_audio",
                return_value=(FakePcmTimeline(16_000), 16_000),
            ),
            self.assertRaises(WorkerError) as captured,
        ):
            adapter.prepare(
                self.audio,
                normalization_profile="mono-16khz-f32-v1",
                context=self.context,
            )

        self.assertEqual(
            captured.exception.code,
            "FUNASR_VAD_INFERENCE_FAILED",
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(
            [
                (event["stage"], event["status"])
                for event in observations
            ],
            [
                ("normalize", "started"),
                ("normalize", "completed"),
                ("pcm_load", "started"),
                ("pcm_load", "completed"),
                ("model_load", "started"),
                ("model_load", "completed"),
                ("inference", "started"),
                ("inference", "failed"),
            ],
        )
        self.assertEqual(
            observations[-1]["errorCode"],
            "FUNASR_VAD_INFERENCE_FAILED",
        )

    def test_model_paths_must_be_explicit_local_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            LocalQwen3AsrAdapter(model_path="Qwen/Qwen3-ASR-1.7B")
        with self.assertRaisesRegex(ValueError, "missing"):
            LocalFunAsrCamPlusAdapter(model_path="iic/campplus")


if __name__ == "__main__":
    unittest.main()
