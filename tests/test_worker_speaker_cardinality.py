from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend import SpeakerCountPolicy, WorkerError, canonical_speaker_ids

from test_worker_support import (
    FakeDynamicRenderer,
    FakeTranscriptionAdapter,
    make_service,
    read_json,
    result_mapping,
)


class SpeakerPolicyTests(unittest.TestCase):
    def test_manual_cardinalities_are_dynamic(self) -> None:
        for count in (1, 3, 5, 8):
            with self.subTest(count=count):
                policy = SpeakerCountPolicy.from_payload(
                    {
                        "speakerCountMode": "manual",
                        "speakerCount": count,
                        "speakerRoles": [
                            f"角色-{index}" for index in range(1, count + 1)
                        ],
                    }
                )
                self.assertEqual(policy.manual_count, count)
                self.assertEqual(
                    canonical_speaker_ids(count),
                    tuple(
                        f"speaker-{index}" for index in range(1, count + 1)
                    ),
                )

    def test_manual_rejects_non_positive_and_non_integer(self) -> None:
        for value in (0, -1, True, 2.5, "5"):
            with self.subTest(value=value):
                with self.assertRaises(WorkerError):
                    SpeakerCountPolicy.from_payload(
                        {
                            "speakerCountMode": "manual",
                            "speakerCount": value,
                        }
                    )
        self.assertEqual(
            SpeakerCountPolicy.from_payload(
                {
                    "speakerCountMode": "manual",
                    "speakerCount": 129,
                }
            ).manual_count,
            129,
        )

    def test_manual_roles_must_match_and_be_unique(self) -> None:
        with self.assertRaises(WorkerError):
            SpeakerCountPolicy.from_payload(
                {
                    "speakerCountMode": "manual",
                    "speakerCount": 3,
                    "speakerRoles": ["主持人", "工程师"],
                }
            )
        with self.assertRaises(WorkerError):
            SpeakerCountPolicy.from_payload(
                {
                    "speakerCountMode": "manual",
                    "speakerCount": 2,
                    "speakerRoles": ["主持人", "主持人"],
                }
            )

    def test_hybrid_requires_valid_bounds_and_prior(self) -> None:
        policy = SpeakerCountPolicy.from_payload(
            {
                "speakerCountMode": "hybrid",
                "speakerCountBounds": {"min": 2, "max": 8},
                "speakerCountPrior": 5,
            }
        )
        self.assertEqual((policy.minimum, policy.maximum, policy.prior), (2, 8, 5))
        for payload in (
            {
                "speakerCountMode": "hybrid",
                "speakerCountBounds": {"min": 8, "max": 2},
            },
            {
                "speakerCountMode": "hybrid",
                "speakerCountBounds": {"min": 2, "max": 8},
                "speakerCountPrior": 9,
            },
            {"speakerCountMode": "hybrid"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(WorkerError):
                    SpeakerCountPolicy.from_payload(payload)


class DynamicWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_root = self.root / "input"
        self.output_root = self.root / "output"
        self.input_root.mkdir()
        self.output_root.mkdir()
        self.source = self.input_root / "meeting.mov"
        self.source.write_bytes(b"synthetic media")
        self.services = []

    def tearDown(self) -> None:
        for service in self.services:
            service.shutdown(cancel=True, wait=True)
        self.temporary.cleanup()

    def run_job(
        self,
        count: int,
        *,
        mode: str = "manual",
        render: bool = False,
        result=None,
        extra=None,
    ):
        adapter = FakeTranscriptionAdapter(result or result_mapping(count))
        service = make_service(
            self.input_root,
            self.output_root,
            adapter=adapter,
            renderer=FakeDynamicRenderer(),
        )
        self.services.append(service)
        payload = {
            "jobId": f"job-{count}-{len(self.services)}",
            "sourcePath": str(self.source),
            "outputDirectory": f"job-{count}-{len(self.services)}",
            "speakerCountMode": mode,
            "renderPdf": render,
        }
        if mode == "manual":
            payload["speakerCount"] = count
        elif mode == "hybrid":
            payload["speakerCountBounds"] = {"min": 1, "max": count + 2}
            payload["speakerCountPrior"] = count
        if extra:
            payload.update(extra)
        service.start(payload)
        return service, service.wait(payload["jobId"]), payload

    def test_one_three_five_and_eight_speakers_complete(self) -> None:
        for count in (1, 3, 5, 8):
            with self.subTest(count=count):
                service, status, payload = self.run_job(count)
                self.assertEqual(status["status"], "completed")
                document = read_json(
                    self.output_root
                    / payload["outputDirectory"]
                    / "transcript-document.v2.json"
                )
                self.assertEqual(
                    document["speakerPolicy"]["speakerIds"],
                    list(canonical_speaker_ids(count)),
                )
                self.assertEqual(document["speakerPolicy"]["resolvedCount"], count)
                self.assertEqual(len(document["speakers"]), count)

    def test_five_is_only_a_regression_fixture(self) -> None:
        _, status, payload = self.run_job(5)
        self.assertEqual(status["status"], "completed")
        document = read_json(
            self.output_root
            / payload["outputDirectory"]
            / "transcript-document.v2.json"
        )
        self.assertEqual(document["schemaVersion"], "2.0.0")
        self.assertNotIn("expectedCount", document["speakerPolicy"])

    def test_auto_persists_estimate_confidence_and_candidate_range(self) -> None:
        _, status, payload = self.run_job(3, mode="auto")
        self.assertEqual(status["status"], "completed")
        document = read_json(
            self.output_root
            / payload["outputDirectory"]
            / "transcript-document.v2.json"
        )
        estimate = document["speakerPolicy"]["estimate"]
        self.assertEqual(estimate["estimatedCount"], 3)
        self.assertEqual(estimate["candidateRange"], {"min": 3, "max": 3})
        self.assertGreater(estimate["confidence"], 0.9)

    def test_auto_missing_estimate_fails_closed(self) -> None:
        _, status, _ = self.run_job(
            3,
            mode="auto",
            result=result_mapping(3, estimate=False),
        )
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error"]["code"], "SPEAKER_COUNT_ESTIMATE_MISSING")

    def test_auto_candidate_range_must_contain_estimate(self) -> None:
        broken = result_mapping(3)
        broken["speakerCountEstimate"]["candidateRange"] = {"min": 4, "max": 7}
        _, status, _ = self.run_job(3, mode="auto", result=broken)
        self.assertEqual(status["status"], "failed")
        self.assertIn(
            status["error"]["code"],
            {"ADAPTER_RESULT_INVALID", "INVALID_REQUEST"},
        )

    def test_hybrid_rejects_estimate_or_range_outside_human_bounds(self) -> None:
        broken = result_mapping(5, candidate_min=1, candidate_max=7)
        _, status, _ = self.run_job(
            5,
            mode="hybrid",
            result=broken,
            extra={"speakerCountBounds": {"min": 2, "max": 6}},
        )
        self.assertEqual(status["status"], "failed")
        self.assertEqual(
            status["error"]["code"], "HYBRID_CANDIDATE_RANGE_OUT_OF_BOUNDS"
        )

    def test_non_five_dynamic_renderer_can_complete(self) -> None:
        _, status, payload = self.run_job(3, render=True)
        self.assertEqual(status["status"], "completed")
        checkpoint = read_json(
            self.output_root / payload["outputDirectory"] / "checkpoint.v2.json"
        )
        self.assertEqual(checkpoint["rendererVersion"], "test-2")
        self.assertEqual(checkpoint["qualityStatus"], "passed")
        self.assertTrue(
            (self.output_root / payload["outputDirectory"] / "transcript.pdf").is_file()
        )


if __name__ == "__main__":
    unittest.main()
