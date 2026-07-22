from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from reporting.java_pdf_client import JavaPdfClient, PdfRenderError


FACETS = [
    "AESTHETIC-COHERENCE",
    "AESTHETIC-DISTINCTION",
    "AESTHETIC-REFINEMENT",
    "AESTHETIC-PROPORTION",
    "AESTHETIC-HIERARCHY",
    "AESTHETIC-TYPOGRAPHY",
    "AESTHETIC-COLOR-RELATIONSHIPS",
    "AESTHETIC-RHYTHM",
    "AESTHETIC-DENSITY",
    "AESTHETIC-RESTRAINT",
    "AESTHETIC-REAL-CONTENT-STRESS",
    "AESTHETIC-FONT-FAILURE",
    "AESTHETIC-IMAGE-FAILURE",
    "AESTHETIC-SCRIPT-FAILURE",
]
HARD_GATES = [
    "PDF-OPENABLE",
    "PDF-PAGE-COUNT",
    "PDF-PAGE-SIZE",
    "PDF-TRANSCRIPT-TEXT-INTEGRITY",
    "PDF-SEGMENT-COUNT",
    "PDF-TIMESTAMP-INTEGRITY",
    "PDF-SPEAKER-SET-INTEGRITY",
    "PDF-FONT-EMBEDDED",
    "PDF-NO-BLANK-PAGES",
    "PDF-NO-CONTENT-OVERFLOW",
    "PDF-OFFLINE-ASSETS",
    "PDF-PAGE-EVIDENCE",
    "PDF-IMMUTABLE-CONTENT-HASH",
]


def report_document(speaker_count: int = 5, *, mode: str = "auto") -> dict:
    if speaker_count < 1:
        raise ValueError("speaker_count must be positive")
    if mode not in {"auto", "manual", "hybrid"}:
        raise ValueError("mode must be auto, manual, or hybrid")

    speakers = []
    segments = []
    for index in range(1, speaker_count + 1):
        speaker_id = f"speaker-{index}"
        speakers.append(
            {
                "id": speaker_id,
                "order": index,
                "displayName": f"角色{index}",
                "shortLabel": str(index),
                "colorToken": f"speaker.{index}",
            }
        )
        scores = [
            {
                "speakerId": f"speaker-{candidate}",
                "score": 0.9 if candidate == index else 0.1,
            }
            for candidate in range(1, speaker_count + 1)
        ]
        segments.append(
            {
                "id": f"segment-{index:03d}",
                "startMs": (index - 1) * 1000,
                "endMs": index * 1000,
                "speakerId": speaker_id,
                "rawText": f"合成原文{index}。",
                "normalizedText": f"合成原文{index}。",
                "displayText": f"合成原文{index}。",
                "language": "zh-CN",
                "confidence": 0.9,
                "reviewStatus": "accepted",
                "evidence": {
                    "asr": {
                        "provider": "qwen-asr",
                        "model": "synthetic",
                        "confidence": 0.9,
                    },
                    "boundary": {
                        "provider": "funasr",
                        "confidence": 0.9,
                        "overlapDetected": False,
                    },
                    "speaker": {
                        "provider": "camp-plus",
                        "assignment": speaker_id,
                        "locked": False,
                        "margin": 0.8,
                        "scores": scores,
                    },
                },
                "revisions": [],
            }
        )
    speaker_ids = [
        f"speaker-{index}" for index in range(1, speaker_count + 1)
    ]
    speaker_policy = {
        "mode": mode,
        "resolvedCount": speaker_count,
        "requireExactSet": True,
        "speakerIds": speaker_ids,
        "unknownSpeakerAllowed": False,
        "speakerChangeRequiresEvidence": True,
    }
    if mode in {"auto", "hybrid"}:
        speaker_policy["detection"] = {
            "provider": "synthetic-diarization-consensus",
            "estimatedCount": speaker_count,
            "confidence": 0.96,
            "candidates": [
                {"count": speaker_count, "confidence": 0.96}
            ],
        }
    if mode in {"manual", "hybrid"}:
        speaker_policy["requestedCount"] = speaker_count
    if mode == "hybrid":
        speaker_policy["minimumCount"] = max(1, speaker_count - 1)
        speaker_policy["maximumCount"] = speaker_count + 1

    return {
        "schemaVersion": "1.0.0",
        "documentId": f"synthetic-render-document-{speaker_count}-{mode}",
        "generatedAt": "2026-07-21T00:00:00Z",
        "language": "zh-CN",
        "source": {
            "fileName": "synthetic.wav",
            "mediaType": "audio/wav",
            "durationMs": speaker_count * 1000,
        },
        "speakerPolicy": speaker_policy,
        "speakers": speakers,
        "segments": segments,
        "provenance": {
            "pipelineVersion": "test",
            "models": [{"role": "asr", "name": "synthetic"}],
            "offline": True,
        },
    }


FAKE_RENDERER = r'''
import hashlib
import json
import os
import sys
import time
from pathlib import Path

FACETS = %s
HARD_GATES = %s
mode = os.environ.get("FAKE_RENDERER_MODE", "success")
if mode == "nonzero":
    print("synthetic renderer failure", file=sys.stderr)
    raise SystemExit(7)
if mode == "timeout":
    time.sleep(10)
if mode == "malformed":
    print("{not-json")
    raise SystemExit(0)

request_path = Path(sys.argv[-1])
request = json.loads(request_path.read_text(encoding="utf-8"))
root = Path(request["outputDirectory"])
report_path = Path(request["reportDocumentPath"])
document = json.loads(report_path.read_text(encoding="utf-8"))
(root / "render").mkdir(parents=True, exist_ok=True)
(root / "artifacts" / "screens").mkdir(parents=True, exist_ok=True)

files = {
    "report-document": report_path,
    "canonical-xhtml": root / "render" / "report.xhtml",
    "pdf": root / "render" / "report.pdf",
    "page-image": root / "artifacts" / "screens" / "page-1.png",
    "contact-sheet": root / "artifacts" / "contact-sheet.png",
    "repair-queue": root / "artifacts" / "repair-queue.json",
}
files["canonical-xhtml"].write_text("<html><body>synthetic</body></html>", encoding="utf-8")
files["pdf"].write_bytes(b"%%PDF-1.4\nsynthetic\n%%%%EOF\n")
files["page-image"].write_bytes(b"synthetic-page")
files["contact-sheet"].write_bytes(b"synthetic-contact")
files["repair-queue"].write_text("[]\n", encoding="utf-8")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

quality_path = root / "artifacts" / "quality-report.json"
hard_pass = mode != "hard_gate_failure"
quality = {
    "schemaVersion": "1.0.0",
    "documentId": document["documentId"],
    "round": 1,
    "status": "passed",
    "minimumScore": request["qualityPolicy"]["minimumScore"],
    "score": 96,
    "hardGatesPassed": hard_pass,
    "hardGates": [{
        "id": gate,
        "status": "passed" if hard_pass else "failed",
        "message": "synthetic gate",
        "evidenceIds": ["pdf"],
    } for gate in HARD_GATES],
    "facets": [{
        "id": facet,
        "status": "passed",
        "score": 96,
        "weight": 1 / len(FACETS),
        "message": "synthetic facet",
        "evidenceIds": ["pdf"],
    } for facet in FACETS],
    "evidence": [{
        "id": "pdf",
        "type": "pdf",
        "relativePath": "render/report.pdf",
        "sha256": digest(files["pdf"]),
        "verified": True,
    }],
    "repairQueue": [],
}
quality_path.write_text(json.dumps(quality, ensure_ascii=False), encoding="utf-8")
files["quality-report"] = quality_path

manifest_artifacts = []
for index, (kind, path) in enumerate(files.items(), start=1):
    relative = path.relative_to(root).as_posix()
    value_hash = digest(path)
    if mode == "hash_mismatch" and kind == "pdf":
        value_hash = "0" * 64
    manifest_artifacts.append({
        "artifactId": f"artifact-{index}",
        "type": kind,
        "relativePath": relative,
        "mimeType": "application/octet-stream",
        "sha256": value_hash,
        "bytes": path.stat().st_size,
        "verified": True,
    })
manifest = {
    "schemaVersion": "1.0.0",
    "jobId": request["jobId"],
    "documentId": document["documentId"],
    "rendererVersion": "synthetic-1.0",
    "createdAt": "2026-07-21T00:00:00Z",
    "artifacts": manifest_artifacts,
}
manifest_path = root / "artifacts" / "manifest.json"
manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

pdf_path = "render/report.pdf"
if mode == "path_escape":
    pdf_path = "../escaped.pdf"
result = {
    "schemaVersion": "1.0.0",
    "requestId": request["requestId"],
    "jobId": request["jobId"],
    "status": "passed",
    "rendererVersion": "synthetic-1.0",
    "roundsCompleted": 1,
    "artifacts": {
        "reportDocumentPath": report_path.relative_to(root).as_posix(),
        "htmlPath": "render/report.xhtml",
        "pdfPath": pdf_path,
        "manifestPath": "artifacts/manifest.json",
        "qualityReportPath": "artifacts/quality-report.json",
        "repairQueuePath": "artifacts/repair-queue.json",
        "screenshotsDirectory": "artifacts/screens",
        "contactSheetPath": "artifacts/contact-sheet.png",
    },
    "quality": {
        "status": "passed",
        "hardGatesPassed": hard_pass,
        "score": 96,
    },
}
if mode == "unknown_field":
    result["unexpected"] = True
print(json.dumps(result, ensure_ascii=False))
''' % (repr(FACETS), repr(HARD_GATES))


class JavaPdfClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.renderer = self.root / "fake_renderer.py"
        self.renderer.write_text(
            textwrap.dedent(FAKE_RENDERER),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def client(self, mode: str = "success", **kwargs) -> JavaPdfClient:
        return JavaPdfClient(
            [sys.executable, str(self.renderer)],
            allowed_output_root=self.root / "outputs",
            timeout_seconds=kwargs.pop("timeout_seconds", 3.0),
            environment={"FAKE_RENDERER_MODE": mode},
            **kwargs,
        )

    def test_render_request_is_offline_a4_and_contains_all_facets(self) -> None:
        client = self.client()
        output = client.allowed_output_root / "job"
        report_path = output / "input" / "report-document.json"
        request = client.create_request(
            report_document_path=report_path,
            output_directory=output,
            job_id="job-001",
        )
        self.assertTrue(request["renderer"]["offline"])
        self.assertEqual(request["renderer"]["page"]["size"], "A4")
        self.assertFalse(
            request["renderer"]["fontPolicy"]["allowSystemFallback"]
        )
        self.assertEqual(request["qualityPolicy"]["facetIds"], FACETS)

    def test_success_verifies_document_manifest_quality_and_pdf(self) -> None:
        outcome = self.client().render(
            report_document(),
            job_id="job-success",
            output_directory="job-success",
        )
        self.assertEqual(outcome.result["status"], "passed")
        self.assertEqual(len(outcome.report_document_sha256), 64)
        self.assertTrue(outcome.artifact_paths["pdfPath"].is_file())

    def test_dynamic_speaker_cardinalities_are_not_capped_at_five(self) -> None:
        for speaker_count in (1, 2, 5, 8, 13):
            with self.subTest(speaker_count=speaker_count):
                outcome = self.client().render(
                    report_document(speaker_count),
                    job_id=f"job-speakers-{speaker_count}",
                    output_directory=f"job-speakers-{speaker_count}",
                )
                persisted = json.loads(
                    outcome.report_document_path.read_text(encoding="utf-8")
                )
                self.assertEqual(outcome.result["status"], "passed")
                self.assertEqual(
                    len(persisted["speakers"]),
                    speaker_count,
                )

    def test_manual_and_hybrid_policies_pass_the_client_contract(self) -> None:
        for mode in ("manual", "hybrid"):
            with self.subTest(mode=mode):
                outcome = self.client().render(
                    report_document(8, mode=mode),
                    job_id=f"job-{mode}",
                    output_directory=f"job-{mode}",
                )
                persisted = json.loads(
                    outcome.report_document_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    persisted["speakerPolicy"]["mode"],
                    mode,
                )

    def test_nonzero_exit_fails_closed(self) -> None:
        with self.assertRaisesRegex(PdfRenderError, "exited with code 7"):
            self.client("nonzero").render(
                report_document(),
                job_id="job-nonzero",
                output_directory="job-nonzero",
            )

    def test_timeout_terminates_renderer(self) -> None:
        with self.assertRaisesRegex(PdfRenderError, "exceeded timeout"):
            self.client("timeout", timeout_seconds=0.2).render(
                report_document(),
                job_id="job-timeout",
                output_directory="job-timeout",
            )

    def test_malformed_stdout_fails_closed(self) -> None:
        with self.assertRaisesRegex(PdfRenderError, "not a single JSON"):
            self.client("malformed").render(
                report_document(),
                job_id="job-malformed",
                output_directory="job-malformed",
            )

    def test_unknown_result_field_fails_closed(self) -> None:
        with self.assertRaisesRegex(PdfRenderError, "unsupported fields"):
            self.client("unknown_field").render(
                report_document(),
                job_id="job-unknown-field",
                output_directory="job-unknown-field",
            )

    def test_from_jar_requires_existing_jar_and_builds_request_cli(self) -> None:
        missing = self.root / "missing.jar"
        with self.assertRaisesRegex(PdfRenderError, "JAR is missing"):
            JavaPdfClient.from_jar(
                missing,
                allowed_output_root=self.root / "outputs",
                java_executable=sys.executable,
            )
        jar = self.root / "pdf-renderer.jar"
        jar.write_bytes(b"synthetic-jar")
        client = JavaPdfClient.from_jar(
            jar,
            allowed_output_root=self.root / "outputs",
            java_executable=sys.executable,
        )
        self.assertEqual(
            client.renderer_command[-2:],
            ("--request", "{request}"),
        )
        self.assertEqual(client.renderer_command[1:3], ("-jar", str(jar.resolve())))

    def test_path_escape_is_rejected(self) -> None:
        with self.assertRaisesRegex(PdfRenderError, "safe relative path"):
            self.client("path_escape").render(
                report_document(),
                job_id="job-path",
                output_directory="job-path",
            )

    def test_manifest_hash_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(PdfRenderError, "SHA-256 mismatch"):
            self.client("hash_mismatch").render(
                report_document(),
                job_id="job-hash",
                output_directory="job-hash",
            )

    def test_hard_gate_failure_cannot_be_offset_by_score(self) -> None:
        with self.assertRaisesRegex(PdfRenderError, "hard gates"):
            self.client("hard_gate_failure").render(
                report_document(),
                job_id="job-gate",
                output_directory="job-gate",
            )

    def test_output_directory_escape_is_rejected(self) -> None:
        with self.assertRaisesRegex(PdfRenderError, "escapes allowed root"):
            self.client().render(
                report_document(),
                job_id="job-escape",
                output_directory=self.root / "outside",
            )


if __name__ == "__main__":
    unittest.main()
