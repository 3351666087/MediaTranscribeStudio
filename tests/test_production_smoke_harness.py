from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import psutil

from tools.run_production_smoke import (
    BatchSmokeJob,
    HarnessSettings,
    ProductionBatchSmokeHarness,
    ProductionSmokeHarness,
    ReviewDecisionPlan,
    SmokePaths,
    absolute_python_executable,
    build_parser,
    build_start_payload,
    calculate_job_hard_timeout_seconds,
    load_review_decision_plan,
)


FAKE_WORKER = r"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


behavior = sys.argv[1]
capture_path = Path(sys.argv[2])


def emit(value):
    sys.stdout.buffer.write(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )
    sys.stdout.buffer.flush()


start_raw = sys.stdin.buffer.readline()
if not start_raw:
    raise SystemExit(20)
start = json.loads(start_raw.decode("utf-8", errors="strict"))
capture_path.write_bytes(
    (
        json.dumps(start, ensure_ascii=False, indent=2)
        + "\n"
    ).encode("utf-8")
)
job_id = start["payload"]["jobId"]
emit(
    {
        "schemaVersion": "1.0.0",
        "requestId": start["requestId"],
        "timestamp": "2026-07-22T00:00:00Z",
        "type": "command.accepted",
        "payload": {"jobId": job_id, "status": "queued"},
    }
)
emit(
    {
        "schemaVersion": "1.0.0",
        "eventId": "evt-started",
        "jobId": job_id,
        "sequence": 0,
        "timestamp": "2026-07-22T00:00:01Z",
        "type": "job.started",
        "payload": {"status": "running"},
    }
)
sys.stderr.buffer.write("诊断：模型加载日志\n".encode("utf-8"))
sys.stderr.buffer.flush()

if behavior == "hang":
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    emit(
        {
            "schemaVersion": "1.0.0",
            "eventId": "evt-child",
            "jobId": job_id,
            "sequence": 1,
            "timestamp": "2026-07-22T00:00:02Z",
            "type": "stage.started",
            "payload": {"stage": "blocked", "childPid": child.pid},
        }
    )
    sys.stderr.buffer.write(
        ("阻塞诊断 childPid=%d\n" % child.pid).encode("utf-8")
    )
    sys.stderr.buffer.flush()
    time.sleep(60)
    raise SystemExit(0)

if behavior == "early":
    sys.stderr.buffer.write("提前退出\n".encode("utf-8"))
    sys.stderr.buffer.flush()
    raise SystemExit(7)

if behavior == "invalid-json":
    sys.stdout.buffer.write(b"\xff\xfe-not-json\n")
    sys.stdout.buffer.flush()
    time.sleep(60)
    raise SystemExit(0)

if behavior == "silent":
    time.sleep(60)
    raise SystemExit(0)

if behavior == "heartbeat-hard":
    for sequence in range(1, 40):
        time.sleep(0.02)
        emit(
            {
                "schemaVersion": "1.0.0",
                "eventId": "evt-heartbeat-" + str(sequence),
                "jobId": job_id,
                "sequence": sequence,
                "timestamp": "2026-07-22T00:00:02Z",
                "type": "stage.progress",
                "payload": {
                    "stage": "transcription",
                    "kind": "heartbeat",
                },
            }
        )
    time.sleep(60)
    raise SystemExit(0)

if behavior == "reject":
    emit(
        {
            "schemaVersion": "1.0.0",
            "requestId": start["requestId"],
            "timestamp": "2026-07-22T00:00:02Z",
            "type": "command.rejected",
            "payload": {"code": "INVALID_REQUEST"},
        }
    )
elif behavior == "review":
    emit(
        {
            "schemaVersion": "1.0.0",
            "eventId": "evt-review",
            "jobId": job_id,
            "sequence": 1,
            "timestamp": "2026-07-22T00:00:02Z",
            "type": "review.required",
            "payload": {
                "openCount": 2,
                "echoedRoles": start["payload"].get("speakerRoles", []),
                "renderPdf": start["payload"]["renderPdf"],
            },
        }
    )
elif behavior == "failed":
    emit(
        {
            "schemaVersion": "1.0.0",
            "eventId": "evt-failed",
            "jobId": job_id,
            "sequence": 1,
            "timestamp": "2026-07-22T00:00:02Z",
            "type": "job.failed",
            "payload": {"code": "FAKE_FAILURE"},
        }
    )
else:
    emit(
        {
            "schemaVersion": "1.0.0",
            "eventId": "evt-completed",
            "jobId": job_id,
            "sequence": 1,
            "timestamp": "2026-07-22T00:00:02Z",
            "type": "job.completed",
            "payload": {
                "status": "completed",
                "echoedRoles": start["payload"].get("speakerRoles", []),
                "renderPdf": start["payload"]["renderPdf"],
            },
        }
    )

# If the harness closes stdin at job.start time, this read returns EOF and the
# test worker exits without a shutdown acknowledgement.
shutdown_raw = sys.stdin.buffer.readline()
if not shutdown_raw:
    raise SystemExit(21)
shutdown = json.loads(shutdown_raw.decode("utf-8", errors="strict"))
if shutdown["type"] != "worker.shutdown":
    raise SystemExit(22)
emit(
    {
        "schemaVersion": "1.0.0",
        "requestId": shutdown["requestId"],
        "timestamp": "2026-07-22T00:00:03Z",
        "type": "command.completed",
        "payload": {"status": "shutdown-requested"},
    }
)
"""


def test_absolute_python_executable_preserves_venv_symlink(
    tmp_path: Path,
) -> None:
    base_python = tmp_path / "base-python"
    base_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    try:
        venv_python.symlink_to(base_python)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            raise unittest.SkipTest(
                "Windows symlink privilege is not enabled"
            ) from exc
        raise

    assert absolute_python_executable(venv_python) == str(venv_python)

BATCH_FAKE_WORKER = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path


capture_path = Path(sys.argv[1])
commands = []
fail_second = len(sys.argv) > 2 and sys.argv[2] == "fail-second"
job_count = 0


def emit(value):
    sys.stdout.buffer.write(
        (
            json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    )
    sys.stdout.buffer.flush()


sys.stderr.buffer.write("model-preloaded-once\n".encode("utf-8"))
sys.stderr.buffer.flush()
for raw in sys.stdin.buffer:
    command = json.loads(raw.decode("utf-8", errors="strict"))
    commands.append(command)
    capture_path.write_text(
        json.dumps(commands, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    request_id = command["requestId"]
    if command["type"] == "job.start":
        job_count += 1
        job_id = command["payload"]["jobId"]
        emit(
            {
                "schemaVersion": "1.0.0",
                "requestId": request_id,
                "timestamp": "2026-07-24T00:00:00Z",
                "type": "command.accepted",
                "payload": {"jobId": job_id, "status": "queued"},
            }
        )
        emit(
            {
                "schemaVersion": "1.0.0",
                "eventId": "evt-start-" + job_id,
                "jobId": job_id,
                "sequence": 0,
                "timestamp": "2026-07-24T00:00:01Z",
                "type": "job.started",
                "payload": {"status": "running"},
            }
        )
        if fail_second and job_count == 2:
            raise SystemExit(7)
        emit(
            {
                "schemaVersion": "1.0.0",
                "eventId": "evt-review-" + job_id,
                "jobId": job_id,
                "sequence": 1,
                "timestamp": "2026-07-24T00:00:02Z",
                "type": "review.required",
                "payload": {"openCount": 1},
            }
        )
    elif command["type"] == "worker.health":
        emit(
            {
                "schemaVersion": "1.0.0",
                "requestId": request_id,
                "timestamp": "2026-07-24T00:00:03Z",
                "type": "command.completed",
                "payload": {
                    "status": "ok",
                    "activeOutputClaims": 0,
                },
            }
        )
    elif command["type"] == "worker.shutdown":
        emit(
            {
                "schemaVersion": "1.0.0",
                "requestId": request_id,
                "timestamp": "2026-07-24T00:00:04Z",
                "type": "command.completed",
                "payload": {"status": "shutdown-requested"},
            }
        )
        break
"""


REVIEW_FLOW_FAKE_WORKER = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path


capture_path = Path(sys.argv[1])
behavior = sys.argv[2] if len(sys.argv) > 2 else "normal"
commands = []
job_id = None
sequence = 0
items = []
decisions = []


def emit(value):
    sys.stdout.buffer.write(
        (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )
    sys.stdout.buffer.flush()


def response(command, response_type, payload):
    emit(
        {
            "schemaVersion": "1.0.0",
            "requestId": command["requestId"],
            "timestamp": "2026-08-07T00:00:00Z",
            "type": response_type,
            "payload": payload,
        }
    )


def job_event(event_type, payload):
    global sequence
    emit(
        {
            "schemaVersion": "1.0.0",
            "eventId": "evt-%s-%d" % (event_type, sequence),
            "jobId": job_id,
            "sequence": sequence,
            "timestamp": "2026-08-07T00:00:00Z",
            "type": event_type,
            "payload": payload,
        }
    )
    sequence += 1


def open_count():
    return sum(item["status"] == "open" for item in items)


for raw in sys.stdin.buffer:
    command = json.loads(raw.decode("utf-8", errors="strict"))
    commands.append(command)
    capture_path.write_text(
        json.dumps(commands, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command_type = command["type"]
    payload = command["payload"]
    if command_type == "job.start":
        job_id = payload["jobId"]
        sequence = 0
        decisions = []
        items = [
            {"id": "item-a", "status": "open", "modelScore": 0.01},
            {
                "id": "item-extra" if behavior == "queue-mismatch" else "item-b",
                "status": "open",
                "modelScore": 0.99,
            },
        ]
        response(command, "command.accepted", {"jobId": job_id, "status": "queued"})
        job_event("job.started", {"status": "running"})
        if behavior == "direct-completed":
            job_event("job.completed", {"status": "completed"})
        else:
            job_event("review.required", {"openCount": 2})
    elif command_type == "speaker.merge":
        decision = {
            "decisionId": payload["decisionId"],
            "command": "speaker.merge",
            "confidence": payload["confidence"],
        }
        decisions.append(decision)
        response(
            command,
            "command.completed",
            {
                "jobId": job_id,
                "status": "review_required",
                "command": "speaker.merge",
                "decision": decision,
                "openCount": open_count(),
            },
        )
    elif command_type == "review.queue":
        response(
            command,
            "command.completed",
            {
                "jobId": job_id,
                "status": "review_required",
                "openCount": open_count(),
                "queue": {
                    "jobId": job_id,
                    "openCount": open_count(),
                    "items": items,
                    "decisions": decisions,
                },
            },
        )
    elif command_type == "review.submit":
        item = next(item for item in items if item["id"] == payload["itemId"])
        item["status"] = "accepted" if payload["action"] == "accept" else "rejected"
        decision = {
            "decisionId": payload["decisionId"],
            "command": "review.submit",
            "confidence": payload["confidence"],
        }
        decisions.append(decision)
        job_event("review.decision.persisted", {"decision": decision})
        response(
            command,
            "command.completed",
            {
                "jobId": job_id,
                "status": "review_required",
                "command": "review.submit",
                "decision": decision,
                "openCount": open_count(),
            },
        )
    elif command_type == "job.resume":
        if behavior == "completed-before-accepted":
            job_event("job.completed", {"status": "completed", "operation": "resume"})
        response(command, "command.accepted", {"jobId": job_id, "status": "queued"})
        if behavior != "completed-before-accepted":
            job_event("job.completed", {"status": "completed", "operation": "resume"})
    elif command_type == "worker.health":
        response(
            command,
            "command.completed",
            {"status": "ok", "activeOutputClaims": 0},
        )
    elif command_type == "worker.shutdown":
        response(command, "command.completed", {"status": "shutdown-requested"})
        break
"""


def review_plan(job_id: str) -> ReviewDecisionPlan:
    audit = {"actor": "codex-reviewer", "source": "human"}
    return ReviewDecisionPlan(
        job_id=job_id,
        pre_review_commands=(
            {
                "type": "speaker.merge",
                "sourceSpeakerId": "speaker-3",
                "targetSpeakerId": "speaker-1",
                "decisionId": "merge-0001",
                "reason": "Independent listening found one speaker.",
                "evidence": ["audio:full"],
                "confidence": 0.88,
                "audit": audit,
            },
            {
                "type": "speaker.merge",
                "sourceSpeakerId": "speaker-2",
                "targetSpeakerId": "speaker-1",
                "decisionId": "merge-0002",
                "reason": "The remaining voice is the same speaker.",
                "evidence": ["audio:full", "context:transcript"],
                "confidence": 0.79,
                "audit": audit,
            },
        ),
        decisions=(
            {
                "itemId": "item-a",
                "action": "accept",
                "decisionId": "item-0001",
                "reason": "The first queue item is resolved by listening.",
                "evidence": ["audio:0-10"],
                "confidence": 0.91,
                "audit": audit,
            },
            {
                "itemId": "item-b",
                "action": "reject",
                "decisionId": "item-0002",
                "reason": "The second queue proposal contradicts the audio.",
                "evidence": ["audio:10-20"],
                "confidence": 0.42,
                "audit": audit,
            },
        ),
    )


class ProductionSmokeHarnessTests(unittest.TestCase):
    def test_duration_rtf_budget_is_bounded_and_rejects_invalid_inputs(
        self,
    ) -> None:
        self.assertEqual(
            calculate_job_hard_timeout_seconds(
                duration_seconds=10.0,
                cold_start_p95_seconds=240.0,
                rtf_p95=25.0,
                safety_margin_seconds=60.0,
                minimum_seconds=300.0,
                maximum_seconds=7200.0,
            ),
            550.0,
        )
        self.assertEqual(
            calculate_job_hard_timeout_seconds(
                duration_seconds=0.1,
                cold_start_p95_seconds=1.0,
                rtf_p95=1.0,
                safety_margin_seconds=1.0,
                minimum_seconds=300.0,
                maximum_seconds=7200.0,
            ),
            300.0,
        )
        with self.assertRaisesRegex(ValueError, "maximum_seconds"):
            calculate_job_hard_timeout_seconds(
                duration_seconds=1.0,
                cold_start_p95_seconds=1.0,
                rtf_p95=1.0,
                safety_margin_seconds=1.0,
                minimum_seconds=10.0,
                maximum_seconds=5.0,
            )

    def test_cli_default_uses_supported_business_prompt_version(self) -> None:
        args = build_parser().parse_args(
            [
                "--config",
                "production.json",
                "--source",
                "sample.wav",
                "--output-dir",
                "output",
            ]
        )

        self.assertEqual(args.business_prompt_version, "business-v3")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worker_script = self.root / "fake_worker.py"
        self.worker_script.write_text(FAKE_WORKER, encoding="utf-8")
        self.source = self.root / "会议片段.wav"
        self.source.write_bytes(b"RIFF")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def paths(self, name: str) -> SmokePaths:
        return SmokePaths(
            event_log=self.root / f"{name}-events.jsonl",
            stderr_log=self.root / f"{name}-stderr.log",
            result_json=self.root / f"{name}-result.json",
        )

    def harness(
        self,
        behavior: str,
        name: str,
        *,
        settings: HarnessSettings | None = None,
    ) -> tuple[ProductionSmokeHarness, Path]:
        capture = self.root / f"{name}-start.json"
        return (
            ProductionSmokeHarness(
                worker_command=(
                    sys.executable,
                    "-u",
                    str(self.worker_script),
                    behavior,
                    str(capture),
                ),
                cwd=self.root,
                paths=self.paths(name),
                settings=settings
                or HarnessSettings(
                    timeout_seconds=5.0,
                    shutdown_timeout_seconds=2.0,
                    cleanup_timeout_seconds=2.0,
                ),
                environment=os.environ.copy(),
            ),
            capture,
        )

    def review_harness(
        self,
        name: str,
        *,
        behavior: str = "normal",
    ) -> tuple[ProductionSmokeHarness, Path]:
        worker = self.root / f"{name}-review-worker.py"
        worker.write_text(REVIEW_FLOW_FAKE_WORKER, encoding="utf-8")
        capture = self.root / f"{name}-review-commands.json"
        return (
            ProductionSmokeHarness(
                worker_command=(
                    sys.executable,
                    "-u",
                    str(worker),
                    str(capture),
                    behavior,
                ),
                cwd=self.root,
                paths=self.paths(name),
                settings=HarnessSettings(
                    timeout_seconds=5.0,
                    shutdown_timeout_seconds=2.0,
                    cleanup_timeout_seconds=2.0,
                ),
                environment=os.environ.copy(),
            ),
            capture,
        )

    def payload(
        self,
        name: str,
        *,
        mode: str = "manual",
        render_pdf: bool = False,
    ) -> dict:
        kwargs = {}
        if mode == "manual":
            kwargs.update(speaker_count=5)
        elif mode == "hybrid":
            kwargs.update(
                speaker_count_min=4,
                speaker_count_max=7,
                speaker_count_prior=5,
            )
        return build_start_payload(
            job_id=f"smoke-{name}",
            source_path=self.source,
            output_directory=self.root / f"{name}-output",
            speaker_count_mode=mode,
            render_pdf=render_pdf,
            **kwargs,
        )

    @staticmethod
    def read_events(path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    @staticmethod
    def wait_pid_gone(pid: int, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid):
                return True
            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    return True
            except psutil.Error:
                return True
            time.sleep(0.05)
        return not psutil.pid_exists(pid)

    def test_keeps_stdin_open_until_completed_then_shuts_down_with_utf8_roles(
        self,
    ) -> None:
        harness, capture = self.harness("completed", "completed")
        result = harness.run(self.payload("completed"))

        self.assertEqual(result.status, "observed")
        self.assertEqual(result.terminal_type, "job.completed")
        self.assertTrue(result.shutdown_acknowledged)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.forced_cleanup_pids, ())
        self.assertEqual(
            result.terminal_event["payload"]["echoedRoles"],
            ["角色1", "角色2", "角色3", "角色4", "角色5"],
        )
        self.assertFalse(result.terminal_event["payload"]["renderPdf"])

        start = json.loads(capture.read_text(encoding="utf-8"))
        self.assertEqual(
            start["payload"]["speakerRoles"],
            ["角色1", "角色2", "角色3", "角色4", "角色5"],
        )
        self.assertFalse(start["payload"]["renderPdf"])
        self.assertIn(
            "诊断：模型加载日志",
            self.paths("completed").stderr_log.read_text(encoding="utf-8"),
        )
        events = self.read_events(self.paths("completed").event_log)
        self.assertLess(
            next(i for i, event in enumerate(events) if event["type"] == "job.completed"),
            next(
                i
                for i, event in enumerate(events)
                if event.get("requestId") == "production-smoke-shutdown"
            ),
        )
        persisted = json.loads(
            self.paths("completed").result_json.read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["terminal_type"], "job.completed")

    def test_review_required_and_job_failed_are_observed_before_shutdown(self) -> None:
        for behavior, expected_status in (
            ("review", "observed"),
            ("failed", "job-failed"),
        ):
            with self.subTest(behavior=behavior):
                harness, _capture = self.harness(behavior, behavior)
                result = harness.run(self.payload(behavior))
                self.assertEqual(result.status, expected_status)
                self.assertEqual(
                    result.terminal_type,
                    "review.required" if behavior == "review" else "job.failed",
                )
                self.assertTrue(result.shutdown_acknowledged)
                self.assertEqual(result.exit_code, 0)

    def test_review_plan_loader_requires_explicit_manual_fields_and_merge_order(
        self,
    ) -> None:
        expected = review_plan("smoke-loader")
        document = {
            "schemaVersion": "1.0.0",
            "artifactType": "production-review-decisions",
            "jobId": expected.job_id,
            "automaticScoring": False,
            "preReviewCommands": [
                dict(command) for command in expected.pre_review_commands
            ],
            "decisions": [dict(decision) for decision in expected.decisions],
        }
        path = self.root / "review-decisions.json"
        path.write_text(
            json.dumps(document, ensure_ascii=False),
            encoding="utf-8",
        )

        loaded = load_review_decision_plan(
            path,
            expected_job_id="smoke-loader",
        )

        self.assertEqual(
            [command["type"] for command in loaded.pre_review_commands],
            ["speaker.merge", "speaker.merge"],
        )
        self.assertEqual(
            loaded.all_decision_ids,
            ("merge-0001", "merge-0002", "item-0001", "item-0002"),
        )
        self.assertEqual(
            [decision["confidence"] for decision in loaded.decisions],
            [0.91, 0.42],
        )

        codex_document = json.loads(json.dumps(document))
        for operation in (
            *codex_document["preReviewCommands"],
            *codex_document["decisions"],
        ):
            operation["audit"]["source"] = "codex-agent"
        path.write_text(json.dumps(codex_document), encoding="utf-8")
        codex_loaded = load_review_decision_plan(path)
        self.assertEqual(
            {
                operation["audit"]["source"]
                for operation in (
                    *codex_loaded.pre_review_commands,
                    *codex_loaded.decisions,
                )
            },
            {"codex-agent"},
        )

        invalid_documents = []
        automatic = json.loads(json.dumps(document))
        automatic["automaticScoring"] = True
        invalid_documents.append(automatic)
        missing_confidence = json.loads(json.dumps(document))
        del missing_confidence["decisions"][0]["confidence"]
        invalid_documents.append(missing_confidence)
        duplicate_id = json.loads(json.dumps(document))
        duplicate_id["decisions"][0]["decisionId"] = "merge-0001"
        invalid_documents.append(duplicate_id)
        missing_source = json.loads(json.dumps(document))
        del missing_source["preReviewCommands"][0]["audit"]["source"]
        invalid_documents.append(missing_source)
        unapproved_source = json.loads(json.dumps(document))
        unapproved_source["decisions"][0]["audit"]["source"] = (
            "model-self-review"
        )
        invalid_documents.append(unapproved_source)
        for index, invalid in enumerate(invalid_documents):
            with self.subTest(index=index):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_review_decision_plan(path)

    def test_review_plan_merges_then_resolves_exact_queue_and_resumes(
        self,
    ) -> None:
        harness, capture = self.review_harness(
            "review-flow",
            behavior="completed-before-accepted",
        )
        payload = self.payload("review-flow", mode="auto")

        result = harness.run(
            payload,
            review_decisions=review_plan(payload["jobId"]),
        )

        self.assertEqual(result.status, "observed")
        self.assertEqual(result.terminal_type, "job.completed")
        self.assertTrue(result.shutdown_acknowledged)
        commands = json.loads(capture.read_text(encoding="utf-8"))
        self.assertEqual(
            [command["type"] for command in commands],
            [
                "job.start",
                "speaker.merge",
                "speaker.merge",
                "review.queue",
                "review.submit",
                "review.submit",
                "review.queue",
                "job.resume",
                "worker.shutdown",
            ],
        )
        manual_commands = [
            command
            for command in commands
            if command["type"] in {"speaker.merge", "review.submit"}
        ]
        self.assertEqual(
            [command["payload"]["confidence"] for command in manual_commands],
            [0.88, 0.79, 0.91, 0.42],
        )
        events = self.read_events(self.paths("review-flow").event_log)
        completed_index = next(
            index
            for index, event in enumerate(events)
            if event.get("type") == "job.completed"
        )
        resume_index = next(
            index
            for index, event in enumerate(events)
            if event.get("requestId")
            == "production-smoke-start-review-resume"
        )
        self.assertLess(completed_index, resume_index)

    def test_review_plan_fails_closed_when_queue_items_do_not_match(
        self,
    ) -> None:
        harness, capture = self.review_harness(
            "review-mismatch",
            behavior="queue-mismatch",
        )
        payload = self.payload("review-mismatch", mode="auto")

        result = harness.run(
            payload,
            review_decisions=review_plan(payload["jobId"]),
        )

        self.assertEqual(result.status, "harness-failed")
        self.assertEqual(result.error["code"], "REVIEW_QUEUE_MISMATCH")
        commands = json.loads(capture.read_text(encoding="utf-8"))
        self.assertEqual(commands[-1]["type"], "review.queue")
        self.assertNotIn(
            "review.submit",
            [command["type"] for command in commands],
        )

    def test_review_plan_is_rejected_when_job_completes_without_review(
        self,
    ) -> None:
        harness, _capture = self.review_harness(
            "review-unused",
            behavior="direct-completed",
        )
        payload = self.payload("review-unused", mode="auto")

        result = harness.run(
            payload,
            review_decisions=review_plan(payload["jobId"]),
        )

        self.assertEqual(result.status, "harness-failed")
        self.assertEqual(result.error["code"], "REVIEW_DECISIONS_UNUSED")
        self.assertEqual(result.terminal_type, "job.completed")

    def test_start_rejection_still_gets_orderly_worker_shutdown(self) -> None:
        harness, _capture = self.harness("reject", "reject")
        result = harness.run(self.payload("reject"))
        self.assertEqual(result.status, "job-failed")
        self.assertEqual(result.terminal_type, "command.rejected")
        self.assertTrue(result.shutdown_acknowledged)
        self.assertEqual(result.exit_code, 0)

    def test_builds_manual_auto_and_hybrid_payloads_without_cross_mode_fields(
        self,
    ) -> None:
        manual = self.payload("manual", mode="manual")
        auto = self.payload("auto", mode="auto")
        hybrid = self.payload("hybrid", mode="hybrid")

        self.assertEqual(manual["speakerCount"], 5)
        self.assertEqual(len(manual["speakerRoles"]), 5)
        self.assertNotIn("speakerCountBounds", manual)
        self.assertNotIn("speakerCount", auto)
        self.assertNotIn("speakerRoles", auto)
        self.assertNotIn("speakerCountBounds", auto)
        self.assertEqual(
            hybrid["speakerCountBounds"],
            {"min": 4, "max": 7},
        )
        self.assertEqual(hybrid["speakerCountPrior"], 5)
        self.assertNotIn("speakerCount", hybrid)
        self.assertNotIn("speakerRoles", hybrid)
        self.assertFalse(manual["renderPdf"])
        self.assertFalse(auto["renderPdf"])
        self.assertFalse(hybrid["renderPdf"])
        self.assertEqual(auto["language"], "auto")
        self.assertEqual(auto["localLlmMode"], "disabled")
        self.assertEqual(auto["translationTargets"], [])
        self.assertNotIn("polish", auto)
        self.assertFalse(auto["summary"])
        self.assertFalse(auto["localLlmAutoApply"])
        self.assertEqual(
            auto["localLlmEndpoint"],
            "http://127.0.0.1:11434",
        )
        self.assertEqual(auto["localLlmEndpointPolicy"], "loopback-only")
        self.assertEqual(auto["outputLocale"], "en")
        self.assertEqual(auto["businessPromptVersion"], "business-v3")

    def test_builds_business_local_llm_acceptance_payload(self) -> None:
        payload = build_start_payload(
            job_id="smoke-business",
            source_path=self.source,
            output_directory=self.root / "business-output",
            speaker_count_mode="auto",
            render_pdf=True,
            language="auto",
            local_llm_mode="business",
            local_llm_model="qwen3.5:9b",
            local_llm_endpoint="http://127.0.0.1:11434",
            local_llm_endpoint_policy="loopback-only",
            translation_targets=("en-US", "ja-JP"),
            summary=True,
            output_locale="zh-Hans",
            business_prompt_version="business-v2",
        )

        self.assertTrue(payload["renderPdf"])
        self.assertEqual(payload["language"], "auto")
        self.assertEqual(payload["localLlmMode"], "business")
        self.assertEqual(payload["localLlmModel"], "qwen3.5:9b")
        self.assertEqual(
            payload["translationTargets"],
            ["en-US", "ja-JP"],
        )
        self.assertNotIn("polish", payload)
        self.assertTrue(payload["summary"])
        self.assertEqual(payload["outputLocale"], "zh-Hans")
        self.assertEqual(payload["businessPromptVersion"], "business-v2")
        self.assertFalse(payload["localLlmAutoApply"])

    def test_builds_canonical_full_output_recipe_without_legacy_conflict(
        self,
    ) -> None:
        recipe_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "product-e2e-output-recipe.v1.json"
        )
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))

        payload = build_start_payload(
            job_id="smoke-complete-product",
            source_path=self.source,
            output_directory=self.root / "complete-product-output",
            speaker_count_mode="auto",
            local_llm_mode="suggestion-only",
            output_customization=recipe,
        )

        self.assertNotIn("renderPdf", payload)
        from backend.output_recipe import parse_output_recipe

        self.assertEqual(
            payload["outputCustomization"],
            parse_output_recipe(recipe).canonical_dict(),
        )
        self.assertEqual(
            payload["outputCustomization"]["delivery"]["subtitleModes"],
            ["sidecar", "soft-mux", "burn-in"],
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_start_payload(
                job_id="smoke-conflicting-product",
                source_path=self.source,
                output_directory=self.root / "conflicting-product-output",
                speaker_count_mode="auto",
                render_pdf=True,
                output_customization=recipe,
            )

    def test_business_payload_validation_fails_closed(self) -> None:
        base = {
            "job_id": "smoke-invalid-business",
            "source_path": self.source,
            "output_directory": self.root / "invalid-business-output",
            "speaker_count_mode": "auto",
        }
        cases = (
            {"translation_targets": ("en-US",), "local_llm_mode": "disabled"},
            {
                "translation_targets": ("en-US", "en-US"),
                "local_llm_mode": "business",
            },
            {
                "translation_targets": ("",),
                "local_llm_mode": "business",
            },
            {
                "local_llm_endpoint_policy": "allow-remote",
                "local_llm_mode": "business",
                "summary": True,
            },
            {
                "local_llm_model": " ",
                "local_llm_mode": "business",
                "summary": True,
            },
        )

        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    build_start_payload(**base, **overrides)

    def test_timeout_captures_diagnostics_and_cleans_only_worker_process_tree(
        self,
    ) -> None:
        harness, _capture = self.harness(
            "hang",
            "hang",
            settings=HarnessSettings(
                timeout_seconds=0.5,
                shutdown_timeout_seconds=0.2,
                cleanup_timeout_seconds=2.0,
            ),
        )
        result = harness.run(self.payload("hang"))

        self.assertEqual(result.status, "harness-failed")
        self.assertEqual(result.error["code"], "JOB_TIMEOUT")
        self.assertIn(result.worker_pid, result.forced_cleanup_pids)
        events = self.read_events(self.paths("hang").event_log)
        child_pid = next(
            event["payload"]["childPid"]
            for event in events
            if event["type"] == "stage.started"
        )
        self.assertIn(child_pid, result.forced_cleanup_pids)
        self.assertTrue(self.wait_pid_gone(result.worker_pid))
        self.assertTrue(self.wait_pid_gone(child_pid))
        stderr = self.paths("hang").stderr_log.read_text(encoding="utf-8")
        self.assertIn("阻塞诊断", stderr)
        self.assertIn(str(child_pid), stderr)
        self.assertIn(
            child_pid,
            result.error["details"]["knownProcessTreePids"],
        )

    def test_idle_timeout_fires_when_current_job_stops_emitting_progress(
        self,
    ) -> None:
        harness, _capture = self.harness(
            "silent",
            "idle-timeout",
            settings=HarnessSettings(
                timeout_seconds=2.0,
                idle_timeout_seconds=0.12,
                hard_timeout_seconds=1.0,
                shutdown_timeout_seconds=0.2,
                cleanup_timeout_seconds=1.0,
            ),
        )

        result = harness.run(self.payload("idle-timeout"))

        self.assertEqual(result.status, "harness-failed")
        self.assertEqual(result.error["code"], "JOB_IDLE_TIMEOUT")
        self.assertEqual(result.idle_timeout_seconds, 0.12)
        self.assertEqual(result.hard_timeout_seconds, 1.0)
        self.assertLess(result.elapsed_seconds, 0.8)
        self.assertGreaterEqual(result.progress_event_count, 2)
        self.assertTrue(self.wait_pid_gone(result.worker_pid))

    def test_heartbeat_refreshes_idle_timeout_but_never_extends_hard_deadline(
        self,
    ) -> None:
        harness, _capture = self.harness(
            "heartbeat-hard",
            "hard-timeout",
            settings=HarnessSettings(
                timeout_seconds=2.0,
                idle_timeout_seconds=0.08,
                hard_timeout_seconds=0.24,
                shutdown_timeout_seconds=0.2,
                cleanup_timeout_seconds=1.0,
            ),
        )

        result = harness.run(self.payload("hard-timeout"))

        self.assertEqual(result.status, "harness-failed")
        self.assertEqual(
            result.error["code"],
            "JOB_HARD_DEADLINE_EXCEEDED",
        )
        # The hard deadline is deliberately very short; Windows scheduling
        # can vary the number of heartbeats observed before it expires.
        self.assertGreater(result.progress_event_count, 0)
        self.assertEqual(result.last_progress_event_type, "stage.progress")
        self.assertGreaterEqual(result.elapsed_seconds, 0.20)
        self.assertLess(result.elapsed_seconds, 0.8)
        self.assertEqual(
            result.error["details"]["hardTimeoutSeconds"],
            0.24,
        )
        self.assertTrue(self.wait_pid_gone(result.worker_pid))

    def test_invalid_utf8_jsonl_fails_closed_and_cleans_exact_worker(self) -> None:
        harness, _capture = self.harness(
            "invalid-json",
            "invalid-json",
            settings=HarnessSettings(
                timeout_seconds=2.0,
                shutdown_timeout_seconds=0.2,
                cleanup_timeout_seconds=2.0,
            ),
        )
        result = harness.run(self.payload("invalid-json"))
        self.assertEqual(result.status, "harness-failed")
        self.assertEqual(result.error["code"], "INVALID_WORKER_JSONL")
        self.assertIn(result.worker_pid, result.forced_cleanup_pids)
        self.assertTrue(self.wait_pid_gone(result.worker_pid))

    def test_worker_exit_before_terminal_event_reports_stderr_and_exit_code(
        self,
    ) -> None:
        harness, _capture = self.harness("early", "early")
        result = harness.run(self.payload("early"))
        self.assertEqual(result.status, "harness-failed")
        self.assertIn(
            result.error["code"],
            {"WORKER_STDOUT_CLOSED", "WORKER_EXITED_EARLY"},
        )
        self.assertEqual(result.exit_code, 7)
        self.assertIn(
            "提前退出",
            self.paths("early").stderr_log.read_text(encoding="utf-8"),
        )

    def test_batch_harness_reuses_one_worker_and_isolates_job_event_logs(
        self,
    ) -> None:
        batch_worker = self.root / "batch_worker.py"
        batch_worker.write_text(BATCH_FAKE_WORKER, encoding="utf-8")
        capture = self.root / "batch-commands.json"
        first_paths = self.paths("batch-first")
        second_paths = self.paths("batch-second")
        harness = ProductionBatchSmokeHarness(
            worker_command=(
                sys.executable,
                "-u",
                str(batch_worker),
                str(capture),
            ),
            cwd=self.root,
            session_event_log=self.root / "batch-session-events.jsonl",
            session_stderr_log=self.root / "batch-session-stderr.log",
            settings=HarnessSettings(
                timeout_seconds=5.0,
                shutdown_timeout_seconds=2.0,
                cleanup_timeout_seconds=2.0,
            ),
            environment=os.environ.copy(),
        )

        results = harness.run(
            (
                BatchSmokeJob(
                    start_payload=self.payload("batch-first", mode="auto"),
                    paths=first_paths,
                ),
                BatchSmokeJob(
                    start_payload=self.payload("batch-second", mode="auto"),
                    paths=second_paths,
                ),
            )
        )

        self.assertEqual(len(results), 2)
        self.assertEqual({result.status for result in results}, {"observed"})
        self.assertEqual(
            {result.worker_pid for result in results},
            {results[0].worker_pid},
        )
        self.assertEqual(
            {result.worker_session_id for result in results},
            {results[0].worker_session_id},
        )
        self.assertTrue(all(result.worker_reused for result in results))
        self.assertTrue(
            all(result.shutdown_acknowledged for result in results)
        )
        self.assertTrue(all(result.exit_code == 0 for result in results))
        self.assertTrue(
            all(result.forced_cleanup_pids == () for result in results)
        )

        commands = json.loads(capture.read_text(encoding="utf-8"))
        self.assertEqual(
            [command["type"] for command in commands],
            [
                "job.start",
                "worker.health",
                "job.start",
                "worker.health",
                "worker.shutdown",
            ],
        )
        first_events = self.read_events(first_paths.event_log)
        second_events = self.read_events(second_paths.event_log)
        self.assertEqual(
            {
                event["jobId"]
                for event in first_events
                if "jobId" in event
            },
            {"smoke-batch-first"},
        )
        self.assertEqual(
            {
                event["jobId"]
                for event in second_events
                if "jobId" in event
            },
            {"smoke-batch-second"},
        )
        self.assertEqual(
            first_paths.stderr_log.read_text(encoding="utf-8").count(
                "model-preloaded-once"
            ),
            1,
        )
        self.assertEqual(
            second_paths.stderr_log.read_text(encoding="utf-8").count(
                "model-preloaded-once"
            ),
            1,
        )

    def test_batch_harness_completes_review_plans_in_the_same_worker(
        self,
    ) -> None:
        worker = self.root / "batch-review-worker.py"
        worker.write_text(REVIEW_FLOW_FAKE_WORKER, encoding="utf-8")
        capture = self.root / "batch-review-commands.json"
        first_paths = self.paths("batch-review-first")
        second_paths = self.paths("batch-review-second")
        first_payload = self.payload("batch-review-first", mode="auto")
        second_payload = self.payload("batch-review-second", mode="auto")
        harness = ProductionBatchSmokeHarness(
            worker_command=(
                sys.executable,
                "-u",
                str(worker),
                str(capture),
                "normal",
            ),
            cwd=self.root,
            session_event_log=self.root / "batch-review-session-events.jsonl",
            session_stderr_log=self.root / "batch-review-session-stderr.log",
            settings=HarnessSettings(
                timeout_seconds=5.0,
                shutdown_timeout_seconds=2.0,
                cleanup_timeout_seconds=2.0,
            ),
            environment=os.environ.copy(),
        )

        results = harness.run(
            (
                BatchSmokeJob(
                    start_payload=first_payload,
                    paths=first_paths,
                    review_decisions=review_plan(first_payload["jobId"]),
                ),
                BatchSmokeJob(
                    start_payload=second_payload,
                    paths=second_paths,
                    review_decisions=review_plan(second_payload["jobId"]),
                ),
            )
        )

        self.assertEqual(
            [result.terminal_type for result in results],
            ["job.completed", "job.completed"],
        )
        self.assertTrue(all(result.status == "observed" for result in results))
        self.assertEqual(
            {result.worker_pid for result in results},
            {results[0].worker_pid},
        )
        commands = json.loads(capture.read_text(encoding="utf-8"))
        self.assertEqual(
            [command["type"] for command in commands].count("job.resume"),
            2,
        )
        self.assertEqual(
            [command["type"] for command in commands].count("speaker.merge"),
            4,
        )
        for paths in (first_paths, second_paths):
            events = self.read_events(paths.event_log)
            self.assertIn("job.completed", {event["type"] for event in events})
            self.assertGreaterEqual(
                sum(
                    event["type"] == "command.completed"
                    for event in events
                ),
                6,
            )

    def test_batch_failure_does_not_invalidate_prior_terminal_job(self) -> None:
        batch_worker = self.root / "batch_worker_fail_second.py"
        batch_worker.write_text(BATCH_FAKE_WORKER, encoding="utf-8")
        capture = self.root / "batch-fail-commands.json"
        first_paths = self.paths("batch-fail-first")
        second_paths = self.paths("batch-fail-second")
        harness = ProductionBatchSmokeHarness(
            worker_command=(
                sys.executable,
                "-u",
                str(batch_worker),
                str(capture),
                "fail-second",
            ),
            cwd=self.root,
            session_event_log=self.root / "batch-fail-session-events.jsonl",
            session_stderr_log=self.root / "batch-fail-session-stderr.log",
            settings=HarnessSettings(
                timeout_seconds=5.0,
                shutdown_timeout_seconds=2.0,
                cleanup_timeout_seconds=2.0,
            ),
            environment=os.environ.copy(),
        )

        results = harness.run(
            (
                BatchSmokeJob(
                    start_payload=self.payload("batch-fail-first", mode="auto"),
                    paths=first_paths,
                ),
                BatchSmokeJob(
                    start_payload=self.payload("batch-fail-second", mode="auto"),
                    paths=second_paths,
                ),
            )
        )

        self.assertEqual(results[0].status, "observed")
        self.assertEqual(results[0].terminal_type, "review.required")
        self.assertIsNone(results[0].error)
        self.assertFalse(results[0].shutdown_acknowledged)
        self.assertEqual(results[1].status, "harness-failed")
        self.assertIsNotNone(results[1].error)
        self.assertIn(
            results[1].error["code"],
            {"WORKER_EXITED_EARLY", "WORKER_STDOUT_CLOSED"},
        )


if __name__ == "__main__":
    unittest.main()
