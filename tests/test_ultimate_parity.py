from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tools import check_ultimate_parity as parity


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "tools" / "check_ultimate_parity.py"
MANIFEST_PATH = REPO_ROOT / "docs" / "refactor" / "ultimate-parity.json"
DOCUMENT_PATH = REPO_ROOT / "docs" / "refactor" / "ULTIMATE_PARITY.md"
NOW = datetime(2026, 7, 22, 8, 0, tzinfo=timezone.utc)

REQUIRED_CAPABILITY_IDS = {
    "transcription",
    "speaker-diarization",
    "translation",
    "polishing",
    "summary",
    "export",
    "dynamic-n",
    "java-pdf",
    "desktop-react-typescript-tauri",
    "design-pack",
    "globalization",
    "real-media-auto",
    "real-media-manual-5",
    "offline-security",
    "packaging-rollback",
    "legacy-cutover",
}

REQUIRED_GATE_IDS = {
    "ULT-TRANSCRIPTION-001",
    "ULT-SPEAKER-001",
    "ULT-TRANSLATION-001",
    "ULT-POLISH-001",
    "ULT-SUMMARY-001",
    "ULT-EXPORT-001",
    "ULT-DYNAMIC-N-001",
    "ULT-JAVA-PDF-001",
    "ULT-DESKTOP-001",
    "ULT-DESIGN-PACK-001",
    "ULT-GLOBALIZATION-001",
    "ULT-REAL-MOV-AUTO-001",
    "ULT-REAL-MOV-MANUAL-5-001",
    "ULT-OFFLINE-SECURITY-001",
    "ULT-PACKAGING-ROLLBACK-001",
    "ULT-LEGACY-CUTOVER-001",
}


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _artifact_for_role(
    attestation: dict[str, Any], role: str
) -> dict[str, Any]:
    return next(
        artifact
        for artifact in attestation["artifacts"]
        if artifact["role"] == role
    )


def _rewrite_json_artifact(
    evidence_root: Path,
    attestation: dict[str, Any],
    role: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    artifact = _artifact_for_role(attestation, role)
    raw = _json_bytes(payload)
    path = evidence_root / artifact["path"]
    path.write_bytes(raw)
    artifact["bytes"] = len(raw)
    artifact["sha256"] = hashlib.sha256(raw).hexdigest()
    return artifact


def _sync_internal_manifest_binding(
    evidence_root: Path,
    attestation: dict[str, Any],
    role: str,
) -> None:
    artifact = _artifact_for_role(attestation, role)
    manifest_artifact = _artifact_for_role(attestation, "artifact-manifest")
    manifest_path = evidence_root / manifest_artifact["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = next(
        item for item in manifest["artifacts"] if item["type"] == role
    )
    item["bytes"] = artifact["bytes"]
    item["sha256"] = artifact["sha256"]
    _rewrite_json_artifact(
        evidence_root,
        attestation,
        "artifact-manifest",
        manifest,
    )


def _run(*argv: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )


def _git(*argv: str, cwd: Path) -> str:
    completed = _run("git", *argv, cwd=cwd)
    if completed.returncode != 0:
        raise AssertionError(
            f"git {' '.join(argv)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _create_protected_paths(repo: Path, manifest: dict[str, Any]) -> None:
    for relative in manifest["policy"]["protectedLegacyPaths"]:
        path = repo / relative
        if Path(relative).suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"protected fixture: {relative}\n", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / ".keep").write_text("protected fixture\n", encoding="utf-8")


def _initialize_clean_repository(repo: Path, manifest: dict[str, Any]) -> str:
    repo.mkdir(parents=True)
    _git("init", cwd=repo)
    _git("config", "user.email", "parity-tests@example.invalid", cwd=repo)
    _git("config", "user.name", "Ultimate Parity Tests", cwd=repo)
    _create_protected_paths(repo, manifest)
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "Create protected parity fixture", cwd=repo)
    _git("branch", "-M", "main", cwd=repo)
    head = _git("rev-parse", "HEAD", cwd=repo)
    _git("update-ref", "refs/remotes/origin/main", head, cwd=repo)
    return head


def _artifact_entry(
    evidence_root: Path,
    relative: str,
    role: str,
    payload: bytes,
) -> dict[str, Any]:
    path = evidence_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    value = {
        "role": role,
        "path": relative.replace("\\", "/"),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    contract = parity.REAL_MEDIA_ARTIFACT_CONTRACTS.get(role)
    if contract:
        value["mediaType"] = contract["mediaType"]
        value["format"] = contract["format"]
    return value


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _real_media_artifacts(
    *,
    manifest: dict[str, Any],
    evidence_root: Path,
    gate_id: str,
    run_name: str,
    job_id: str,
    document_id: str,
    source_sha: str,
    count: int,
    mode: str,
) -> tuple[list[dict[str, Any]], int, float]:
    speaker_ids = [f"speaker-{index}" for index in range(1, count + 1)]
    segments = [
        {
            "id": f"segment-{index}",
            "startMs": (index - 1) * 1_000,
            "endMs": index * 1_000,
            "speakerId": speaker_id,
            "displayText": f"Fixture segment {index}.",
        }
        for index, speaker_id in enumerate(speaker_ids, start=1)
    ]
    speaker_policy = {
        "mode": mode,
        "resolvedCount": count,
        "requireExactSet": True,
        "speakerIds": speaker_ids,
    }
    source = {
        "fileName": manifest["realMedia"]["sourceFileName"],
        "sha256": source_sha,
        "durationMs": count * 1_000,
        "mediaType": "video/quicktime",
    }
    transcript = {
        "schemaVersion": "2.0.0",
        "documentId": document_id,
        "jobId": job_id,
        "source": source,
        "speakerPolicy": speaker_policy,
        "speakers": [{"id": item} for item in speaker_ids],
        "segments": segments,
    }
    report = {
        "schemaVersion": "1.0.0",
        "documentId": document_id,
        "source": source,
        "speakerPolicy": speaker_policy,
        "speakers": [
            {
                "id": item,
                "order": index,
                "displayName": f"Speaker {index}",
                "shortLabel": f"S{index}",
            }
            for index, item in enumerate(speaker_ids, start=1)
        ],
        "segments": segments,
        "provenance": {"offline": True},
    }
    pdf_bytes = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    base = f"runs/{run_name}"
    artifacts: dict[str, dict[str, Any]] = {}
    artifacts["transcript"] = _artifact_entry(
        evidence_root,
        f"{base}/transcript.json",
        "transcript",
        _json_bytes(transcript),
    )
    artifacts["report-document"] = _artifact_entry(
        evidence_root,
        f"{base}/report-document.json",
        "report-document",
        _json_bytes(report),
    )
    artifacts["pdf"] = _artifact_entry(
        evidence_root,
        f"{base}/report.pdf",
        "pdf",
        pdf_bytes,
    )
    page_count = 1
    inspection = {
        "schemaVersion": "1.0.0",
        "validator": "PDFBox",
        "validatorVersion": "2.0.30",
        "documentId": document_id,
        "pdfSha256": artifacts["pdf"]["sha256"],
        "openable": True,
        "pageCount": page_count,
        "allPagesA4": True,
        "transcriptTextIntegrity": True,
        "segmentCountIntegrity": True,
        "timestampIntegrity": True,
        "speakerSetIntegrity": True,
        "allFontsEmbedded": True,
        "searchableText": True,
        "noReplacementCharacters": True,
        "pages": [{"pageNumber": 1}],
        "missingSegmentIds": [],
        "duplicateSegmentIds": [],
        "missingTimestamps": [],
        "missingSpeakerIds": [],
    }
    artifacts["pdf-inspection"] = _artifact_entry(
        evidence_root,
        f"{base}/pdf-inspection.json",
        "pdf-inspection",
        _json_bytes(inspection),
    )
    quality_score = float(manifest["javaPdf"]["minimumQualityScore"])
    quality = {
        "schemaVersion": "1.0.0",
        "documentId": document_id,
        "round": 1,
        "status": "passed",
        "minimumScore": quality_score,
        "score": quality_score,
        "hardGatesPassed": True,
        "hardGates": [
            {"id": item, "status": "passed"}
            for item in manifest["javaPdf"]["hardGateIds"]
        ],
        "facets": [
            {"id": item, "status": "passed", "weight": 1 / 14}
            for item in manifest["javaPdf"]["designPackFacetIds"]
        ],
        "evidence": [
            {
                "id": "evidence-pdf",
                "verified": True,
                "sha256": artifacts["pdf"]["sha256"],
            }
        ],
        "repairQueue": [],
    }
    artifacts["quality-report"] = _artifact_entry(
        evidence_root,
        f"{base}/quality-report.json",
        "quality-report",
        _json_bytes(quality),
    )
    artifacts["contact-sheet"] = _artifact_entry(
        evidence_root,
        f"{base}/contact-sheet.png",
        "contact-sheet",
        png_bytes,
    )
    internal_manifest = {
        "schemaVersion": "1.0.0",
        "jobId": job_id,
        "documentId": document_id,
        "rendererVersion": "3.0.0",
        "artifacts": [
            {
                "artifactId": role,
                "type": role,
                "relativePath": value["path"],
                "mimeType": value["mediaType"],
                "sha256": value["sha256"],
                "bytes": value["bytes"],
                "verified": True,
            }
            for role, value in artifacts.items()
            if role in {"report-document", "pdf", "quality-report", "contact-sheet"}
        ],
    }
    artifacts["artifact-manifest"] = _artifact_entry(
        evidence_root,
        f"{base}/artifact-manifest.json",
        "artifact-manifest",
        _json_bytes(internal_manifest),
    )
    return (
        [artifacts[role] for role in manifest["realMedia"]["requiredArtifactRoles"]],
        len(segments),
        quality_score,
    )


def _real_media_attestation(
    *,
    manifest: dict[str, Any],
    evidence_root: Path,
    gate_id: str,
    kind: str,
    head: str,
    profile: str,
    source_sha: str = "a" * 64,
    persisted_language: str = "zh-CN",
    include_auto_override: bool = False,
    resolved_count: int | None = None,
) -> dict[str, Any]:
    manual = profile == "realMediaManualFive"
    count = resolved_count if resolved_count is not None else (5 if manual else 4)
    run_name = "manual-five" if manual else "auto"
    job_id = f"job-{run_name}"
    document_id = f"document-{run_name}"
    artifacts, segment_count, quality_score = _real_media_artifacts(
        manifest=manifest,
        evidence_root=evidence_root,
        gate_id=gate_id,
        run_name=run_name,
        job_id=job_id,
        document_id=document_id,
        source_sha=source_sha,
        count=count,
        mode="manual" if manual else "auto",
    )
    value: dict[str, Any] = {
        "schemaVersion": parity.SCHEMA_VERSION,
        "kind": kind,
        "gateId": gate_id,
        "status": "passed",
        "commitSha": head,
        "generatedAt": NOW.isoformat(),
        "runId": f"run-{run_name}",
        "jobId": job_id,
        "documentId": document_id,
        "sourceFileName": manifest["realMedia"]["sourceFileName"],
        "sourceSha256": source_sha,
        "terminalState": "completed",
        "persistedLanguage": persisted_language,
        "speakerCountMode": "manual" if manual else "auto",
        "detectedSpeakerCount": count,
        "resolvedSpeakerCount": count,
        "speakerIds": [
            f"speaker-{index}" for index in range(1, count + 1)
        ],
        "artifacts": artifacts,
        "quality": {
            "hardGatesPassed": True,
            "hardGates": [
                {"id": item, "status": "passed"}
                for item in manifest["javaPdf"]["hardGateIds"]
            ],
            "facets": [
                {"id": item, "status": "passed"}
                for item in manifest["javaPdf"]["designPackFacetIds"]
            ],
            "score": quality_score,
            "missingSegments": [],
            "missingTimestamps": [],
            "missingSpeakerIds": [],
            "fontFailures": [],
            "blankPages": [],
            "overflowFindings": [],
            "remoteAssetFindings": [],
        },
        "metrics": {
            "speakerCount": {
                "status": "passed",
                "expectedCount": count,
                "detectedCount": count,
                "resolvedCount": count,
                "absoluteError": 0,
                "reviewRequired": False,
            },
            "diarization": {
                "status": "passed",
                "segmentCount": segment_count,
                "auditedSegmentCount": segment_count,
                "speakerAssignmentErrors": 0,
                "unresolvedSpeakerSegments": 0,
                "humanAuditCompleted": True,
            },
            "boundary": {
                "status": "passed",
                "segmentCount": segment_count,
                "invalidIntervals": 0,
                "nonMonotonicIntervals": 0,
                "outOfBoundsIntervals": 0,
            },
            "asr": {
                "status": "passed",
                "segmentCount": segment_count,
                "auditedSegmentCount": segment_count,
                "emptySegments": 0,
                "unresolvedTextSegments": 0,
                "sourceLanguagePreserved": True,
            },
            "semantic": {
                "status": "passed",
                "reviewedRevisionCount": 0,
                "unresolvedRevisionCount": 0,
                "rawTranscriptImmutable": True,
                "speakerLocksPreserved": True,
                "humanApproved": True,
            },
            "efficiency": {
                "status": "passed",
                "mediaDurationSeconds": float(segment_count),
                "wallClockSeconds": float(segment_count),
                "realTimeFactor": 1.0,
                "peakRamMb": 1.0,
                "peakVramMb": 0.0,
            },
            "pdf": {
                "status": "passed",
                "pageCount": 1,
                "qualityScore": quality_score,
                "hardGateFailureCount": 0,
                "facetFailureCount": 0,
                "pdfBoxValidated": True,
            },
        },
    }
    if manual:
        value["requestedSpeakerCount"] = 5
        value["manualSpeakerCount"] = 5
    elif include_auto_override:
        value["manualSpeakerCount"] = 5
        value["requestedSpeakerCount"] = 5
    return value


def _write_complete_evidence(
    *,
    manifest: dict[str, Any],
    evidence_root: Path,
    head: str,
    manual_source_sha: str = "a" * 64,
) -> None:
    for gate in manifest["gates"]:
        for spec in gate["evidence"]["attestations"]:
            path = evidence_root / spec["path"]
            profile = spec.get("profile")
            if profile:
                value = _real_media_attestation(
                    manifest=manifest,
                    evidence_root=evidence_root,
                    gate_id=gate["id"],
                    kind=spec["kind"],
                    head=head,
                    profile=profile,
                    source_sha=(
                        manual_source_sha
                        if profile == "realMediaManualFive"
                        else "a" * 64
                    ),
                )
            else:
                value = {
                    "schemaVersion": parity.SCHEMA_VERSION,
                    "kind": spec["kind"],
                    "gateId": gate["id"],
                    "status": "passed",
                    "commitSha": head,
                    "generatedAt": NOW.isoformat(),
                }
            _write_json(path, value)


def _synthetic_manifest(
    *,
    head: str,
    approvals: bool,
) -> dict[str, Any]:
    manifest = _load_manifest()
    manifest["policy"]["protectedMainRefs"] = {
        "refs/heads/main": head,
        "refs/remotes/origin/main": head,
    }
    approval_time = NOW.isoformat()
    for key in ("legacyRemoval", "mainReplacement"):
        authorization = manifest["policy"]["authorization"][key]
        authorization.update(
            {
                "approved": approvals,
                "approvedBy": "release-owner" if approvals else None,
                "approvedAt": approval_time if approvals else None,
                "changeTicket": "MTS-ULTIMATE-1" if approvals else None,
            }
        )
    for gate in manifest["gates"]:
        gate["state"] = "passed"
        gate["repoChecks"] = []
        gate["commandChecks"] = []
        for spec in gate["evidence"]["attestations"]:
            if not spec.get("profile"):
                spec["minimumArtifacts"] = 0
    return manifest


class UltimateParityManifestTests(unittest.TestCase):
    def test_committed_manifest_is_structurally_valid(self) -> None:
        self.assertEqual(parity.validate_manifest(_load_manifest()), [])

    def test_required_capabilities_and_gates_map_bidirectionally(self) -> None:
        manifest = _load_manifest()
        capabilities = {
            item["id"]: set(item["gateIds"]) for item in manifest["capabilities"]
        }
        gates = {
            item["id"]: set(item["capabilityIds"]) for item in manifest["gates"]
        }
        self.assertEqual(set(capabilities), REQUIRED_CAPABILITY_IDS)
        self.assertEqual(set(gates), REQUIRED_GATE_IDS)
        for capability_id, gate_ids in capabilities.items():
            for gate_id in gate_ids:
                self.assertIn(capability_id, gates[gate_id])
        for gate_id, capability_ids in gates.items():
            for capability_id in capability_ids:
                self.assertIn(gate_id, capabilities[capability_id])

    def test_manifest_and_document_are_english_only(self) -> None:
        for path in (MANIFEST_PATH, DOCUMENT_PATH):
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                parity.CJK_RE.search(text),
                f"CJK text is forbidden in {path.relative_to(REPO_ROOT)}",
            )

    def test_cli_manifest_validation_succeeds_from_repository_root(self) -> None:
        completed = _run(
            sys.executable,
            str(CHECKER),
            "--validate-manifest",
            "--format",
            "json",
            cwd=REPO_ROOT,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["manifestValid"])
        self.assertEqual(payload["configurationErrors"], [])

    def test_default_cli_is_intentionally_release_blocked(self) -> None:
        env = os.environ.copy()
        env.pop("MTS_ULTIMATE_EVIDENCE_ROOT", None)
        completed = subprocess.run(
            [sys.executable, str(CHECKER), "--format", "json"],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["manifestValid"])
        self.assertFalse(payload["releaseEligible"])
        self.assertFalse(payload["parityEligible"])
        self.assertFalse(payload["legacyRemovalAllowed"])
        self.assertFalse(payload["mainReplacementAllowed"])

    def test_missing_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = parity.evaluate(
                repo_root=REPO_ROOT,
                manifest_path=MANIFEST_PATH,
                evidence_root=Path(directory),
                now=NOW,
            )
        self.assertFalse(result["releaseEligible"])
        self.assertEqual(result["gateSummary"]["passed"], 0)
        first_gate = result["gates"][0]
        evidence_messages = [
            message
            for item in first_gate["attestations"]
            for message in item["messages"]
        ]
        self.assertTrue(
            any("does not exist" in message for message in evidence_messages)
        )

    def test_missing_required_gate_invalidates_manifest(self) -> None:
        manifest = _load_manifest()
        manifest["gates"] = manifest["gates"][:-1]
        errors = parity.validate_manifest(manifest)
        self.assertIn(
            "gates must exactly cover policy.requiredGateIds",
            errors,
        )


class UltimateParityEvidenceTests(unittest.TestCase):
    def _generic_spec(self) -> dict[str, Any]:
        return {
            "path": "attestations/example.json",
            "kind": "capability-attestation",
            "minimumArtifacts": 0,
        }

    def _generic_attestation(
        self,
        *,
        head: str = "a" * 40,
        generated_at: datetime = NOW,
    ) -> dict[str, Any]:
        return {
            "schemaVersion": parity.SCHEMA_VERSION,
            "kind": "capability-attestation",
            "gateId": "ULT-TRANSCRIPTION-001",
            "status": "passed",
            "commitSha": head,
            "generatedAt": generated_at.isoformat(),
        }

    def test_stale_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_json(
                root / self._generic_spec()["path"],
                self._generic_attestation(
                    generated_at=NOW - timedelta(hours=337)
                ),
            )
            result = parity.verify_attestation(
                evidence_root=root,
                spec=self._generic_spec(),
                gate_id="ULT-TRANSCRIPTION-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
            )
        self.assertFalse(result["passed"])
        self.assertIn("attestation is stale", result["messages"])

    def test_head_mismatched_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_json(
                root / self._generic_spec()["path"],
                self._generic_attestation(head="b" * 40),
            )
            result = parity.verify_attestation(
                evidence_root=root,
                spec=self._generic_spec(),
                gate_id="ULT-TRANSCRIPTION-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
            )
        self.assertFalse(result["passed"])
        self.assertIn(
            "commitSha does not match the current HEAD",
            result["messages"],
        )

    def test_artifact_sha_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _artifact_entry(
                root,
                "artifacts/result.bin",
                "result",
                b"trusted bytes",
            )
            artifact["sha256"] = "0" * 64
            attestation = self._generic_attestation()
            attestation["artifacts"] = [artifact]
            spec = self._generic_spec()
            spec["minimumArtifacts"] = 1
            _write_json(root / spec["path"], attestation)
            result = parity.verify_attestation(
                evidence_root=root,
                spec=spec,
                gate_id="ULT-TRANSCRIPTION-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
            )
        self.assertFalse(result["passed"])
        self.assertIn("artifacts[0].sha256 mismatch", result["messages"])

    def test_real_media_bin_cannot_satisfy_a_canonical_role(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "path": "real-media/auto.json",
                "kind": "real-media-attestation",
                "profile": "realMediaAuto",
                "minimumArtifacts": 7,
            }
            value = _real_media_attestation(
                manifest=manifest,
                evidence_root=root,
                gate_id="ULT-REAL-MOV-AUTO-001",
                kind=spec["kind"],
                head="a" * 40,
                profile=spec["profile"],
            )
            transcript = _artifact_for_role(value, "transcript")
            source = root / transcript["path"]
            destination = source.with_suffix(".bin")
            source.rename(destination)
            transcript["path"] = destination.relative_to(root).as_posix()
            _write_json(root / spec["path"], value)
            result = parity.verify_attestation(
                evidence_root=root,
                spec=spec,
                gate_id="ULT-REAL-MOV-AUTO-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
                real_media=manifest["realMedia"],
                java_pdf=manifest["javaPdf"],
            )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any(
                "role 'transcript' must use .json" in message
                for message in result["messages"]
            )
        )

    def test_real_media_duplicate_artifact_roles_fail(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "path": "real-media/auto.json",
                "kind": "real-media-attestation",
                "profile": "realMediaAuto",
                "minimumArtifacts": 7,
            }
            value = _real_media_attestation(
                manifest=manifest,
                evidence_root=root,
                gate_id="ULT-REAL-MOV-AUTO-001",
                kind=spec["kind"],
                head="a" * 40,
                profile=spec["profile"],
            )
            _artifact_for_role(value, "contact-sheet")["role"] = "pdf"
            _write_json(root / spec["path"], value)
            result = parity.verify_attestation(
                evidence_root=root,
                spec=spec,
                gate_id="ULT-REAL-MOV-AUTO-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
                real_media=manifest["realMedia"],
                java_pdf=manifest["javaPdf"],
            )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any(".role is duplicated" in message for message in result["messages"])
        )
        self.assertTrue(
            any(
                "artifacts are missing canonical roles: contact-sheet" in message
                for message in result["messages"]
            )
        )

    def test_real_media_status_only_metrics_fail_closed(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "path": "real-media/auto.json",
                "kind": "real-media-attestation",
                "profile": "realMediaAuto",
                "minimumArtifacts": 7,
            }
            value = _real_media_attestation(
                manifest=manifest,
                evidence_root=root,
                gate_id="ULT-REAL-MOV-AUTO-001",
                kind=spec["kind"],
                head="a" * 40,
                profile=spec["profile"],
            )
            value["metrics"] = {
                domain: {"status": "passed"}
                for domain in manifest["realMedia"]["requiredMetricDomains"]
            }
            _write_json(root / spec["path"], value)
            result = parity.verify_attestation(
                evidence_root=root,
                spec=spec,
                gate_id="ULT-REAL-MOV-AUTO-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
                real_media=manifest["realMedia"],
                java_pdf=manifest["javaPdf"],
            )
        self.assertFalse(result["passed"])
        self.assertIn(
            "metrics.speakerCount.expectedCount is required",
            result["messages"],
        )
        self.assertIn(
            "metrics.pdf.pdfBoxValidated is required",
            result["messages"],
        )

    def test_transcript_report_segment_identity_mismatch_fails(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "path": "real-media/auto.json",
                "kind": "real-media-attestation",
                "profile": "realMediaAuto",
                "minimumArtifacts": 7,
            }
            value = _real_media_attestation(
                manifest=manifest,
                evidence_root=root,
                gate_id="ULT-REAL-MOV-AUTO-001",
                kind=spec["kind"],
                head="a" * 40,
                profile=spec["profile"],
            )
            report_artifact = _artifact_for_role(value, "report-document")
            report = json.loads(
                (root / report_artifact["path"]).read_text(encoding="utf-8")
            )
            report["segments"][0]["speakerId"] = "speaker-2"
            _rewrite_json_artifact(
                root,
                value,
                "report-document",
                report,
            )
            _sync_internal_manifest_binding(root, value, "report-document")
            _write_json(root / spec["path"], value)
            result = parity.verify_attestation(
                evidence_root=root,
                spec=spec,
                gate_id="ULT-REAL-MOV-AUTO-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
                real_media=manifest["realMedia"],
                java_pdf=manifest["javaPdf"],
            )
        self.assertFalse(result["passed"])
        self.assertIn(
            "transcript and ReportDocument segment identity/timing/speaker data differ",
            result["messages"],
        )

    def test_pdf_inspection_must_bind_the_exact_pdf_hash(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "path": "real-media/auto.json",
                "kind": "real-media-attestation",
                "profile": "realMediaAuto",
                "minimumArtifacts": 7,
            }
            value = _real_media_attestation(
                manifest=manifest,
                evidence_root=root,
                gate_id="ULT-REAL-MOV-AUTO-001",
                kind=spec["kind"],
                head="a" * 40,
                profile=spec["profile"],
            )
            inspection_artifact = _artifact_for_role(value, "pdf-inspection")
            inspection = json.loads(
                (root / inspection_artifact["path"]).read_text(encoding="utf-8")
            )
            inspection["pdfSha256"] = "b" * 64
            _rewrite_json_artifact(
                root,
                value,
                "pdf-inspection",
                inspection,
            )
            _write_json(root / spec["path"], value)
            result = parity.verify_attestation(
                evidence_root=root,
                spec=spec,
                gate_id="ULT-REAL-MOV-AUTO-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
                real_media=manifest["realMedia"],
                java_pdf=manifest["javaPdf"],
            )
        self.assertFalse(result["passed"])
        self.assertIn(
            "PDF inspection is not bound to the attested PDF hash",
            result["messages"],
        )

    def test_quality_and_artifact_manifest_identity_mismatches_fail(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "path": "real-media/auto.json",
                "kind": "real-media-attestation",
                "profile": "realMediaAuto",
                "minimumArtifacts": 7,
            }
            value = _real_media_attestation(
                manifest=manifest,
                evidence_root=root,
                gate_id="ULT-REAL-MOV-AUTO-001",
                kind=spec["kind"],
                head="a" * 40,
                profile=spec["profile"],
            )
            quality_artifact = _artifact_for_role(value, "quality-report")
            quality = json.loads(
                (root / quality_artifact["path"]).read_text(encoding="utf-8")
            )
            quality["documentId"] = "wrong-quality-document"
            _rewrite_json_artifact(root, value, "quality-report", quality)
            _sync_internal_manifest_binding(root, value, "quality-report")

            manifest_artifact = _artifact_for_role(value, "artifact-manifest")
            artifact_manifest = json.loads(
                (root / manifest_artifact["path"]).read_text(encoding="utf-8")
            )
            artifact_manifest["jobId"] = "wrong-manifest-job"
            artifact_manifest["documentId"] = "wrong-manifest-document"
            _rewrite_json_artifact(
                root,
                value,
                "artifact-manifest",
                artifact_manifest,
            )
            _write_json(root / spec["path"], value)
            result = parity.verify_attestation(
                evidence_root=root,
                spec=spec,
                gate_id="ULT-REAL-MOV-AUTO-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
                real_media=manifest["realMedia"],
                java_pdf=manifest["javaPdf"],
            )
        self.assertFalse(result["passed"])
        self.assertIn(
            "quality report documentId does not match",
            result["messages"],
        )
        self.assertIn(
            "artifact manifest jobId does not match",
            result["messages"],
        )
        self.assertIn(
            "artifact manifest documentId does not match",
            result["messages"],
        )

    def test_manual_real_media_must_resolve_exactly_five_speakers(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "path": "real-media/manual-5.json",
                "kind": "real-media-attestation",
                "profile": "realMediaManualFive",
                "minimumArtifacts": 7,
            }
            value = _real_media_attestation(
                manifest=manifest,
                evidence_root=root,
                gate_id="ULT-REAL-MOV-MANUAL-5-001",
                kind=spec["kind"],
                head="a" * 40,
                profile=spec["profile"],
                resolved_count=4,
            )
            _write_json(root / spec["path"], value)
            result = parity.verify_attestation(
                evidence_root=root,
                spec=spec,
                gate_id="ULT-REAL-MOV-MANUAL-5-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
                real_media=manifest["realMedia"],
                java_pdf=manifest["javaPdf"],
            )
        self.assertFalse(result["passed"])
        self.assertIn(
            "manual evidence must detect and resolve exactly five speakers",
            result["messages"],
        )

    def test_auto_real_media_rejects_manual_overrides_and_auto_language(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "path": "real-media/auto.json",
                "kind": "real-media-attestation",
                "profile": "realMediaAuto",
                "minimumArtifacts": 7,
            }
            value = _real_media_attestation(
                manifest=manifest,
                evidence_root=root,
                gate_id="ULT-REAL-MOV-AUTO-001",
                kind=spec["kind"],
                head="a" * 40,
                profile=spec["profile"],
                persisted_language="auto",
                include_auto_override=True,
            )
            _write_json(root / spec["path"], value)
            result = parity.verify_attestation(
                evidence_root=root,
                spec=spec,
                gate_id="ULT-REAL-MOV-AUTO-001",
                head_commit="a" * 40,
                max_age_hours=336,
                now=NOW,
                real_media=manifest["realMedia"],
                java_pdf=manifest["javaPdf"],
            )
        self.assertFalse(result["passed"])
        self.assertIn(
            "persistedLanguage must be concrete and must not be 'auto'",
            result["messages"],
        )
        self.assertIn(
            "auto evidence must not contain a manualSpeakerCount override",
            result["messages"],
        )
        self.assertIn(
            "auto evidence must not contain a requestedSpeakerCount override",
            result["messages"],
        )


class UltimateParityGuardrailTests(unittest.TestCase):
    def test_premature_legacy_deletion_is_reported(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _create_protected_paths(repo, manifest)
            removed = repo / manifest["policy"]["protectedLegacyPaths"][0]
            removed.unlink()
            result = parity.evaluate_guardrails(
                repo_root=repo,
                manifest=manifest,
                parity_eligible=False,
                release_gates_passed=False,
                head_commit="a" * 40,
                refs=manifest["policy"]["protectedMainRefs"],
                now=NOW,
            )
        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertIn(
            "protected legacy path was removed before authorization: main.py",
            result["legacyProtection"]["violations"],
        )

    def test_premature_protected_main_change_is_reported(self) -> None:
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _create_protected_paths(repo, manifest)
            refs = dict(manifest["policy"]["protectedMainRefs"])
            refs["refs/heads/main"] = "f" * 40
            result = parity.evaluate_guardrails(
                repo_root=repo,
                manifest=manifest,
                parity_eligible=False,
                release_gates_passed=False,
                head_commit="a" * 40,
                refs=refs,
                now=NOW,
            )
        self.assertFalse(result["mainProtection"]["passed"])
        self.assertTrue(
            any(
                "protected ref refs/heads/main changed before authorization"
                in message
                for message in result["mainProtection"]["violations"]
            )
        )

    def test_real_mov_auto_and_manual_source_hashes_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            seed_manifest = _load_manifest()
            head = _initialize_clean_repository(repo, seed_manifest)
            manifest = _synthetic_manifest(head=head, approvals=True)
            evidence = root / "evidence"
            _write_complete_evidence(
                manifest=manifest,
                evidence_root=evidence,
                head=head,
                manual_source_sha="b" * 64,
            )
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, manifest)
            result = parity.evaluate(
                repo_root=repo,
                manifest_path=manifest_path,
                evidence_root=evidence,
                now=NOW,
            )
        self.assertFalse(result["releaseEligible"])
        real_gates = {
            gate["id"]: gate
            for gate in result["gates"]
            if gate["id"]
            in {
                "ULT-REAL-MOV-AUTO-001",
                "ULT-REAL-MOV-MANUAL-5-001",
            }
        }
        for gate in real_gates.values():
            self.assertFalse(gate["passed"])
            self.assertIn(
                "real MOV auto and manual=5 attestations must share sourceSha256",
                gate["blockingReasons"],
            )

    def test_release_requires_all_evidence_clean_git_and_both_approvals(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            seed_manifest = _load_manifest()
            head = _initialize_clean_repository(repo, seed_manifest)
            evidence = root / "evidence"

            blocked_manifest = _synthetic_manifest(
                head=head,
                approvals=False,
            )
            _write_complete_evidence(
                manifest=blocked_manifest,
                evidence_root=evidence,
                head=head,
            )
            blocked_path = root / "manifest-blocked.json"
            _write_json(blocked_path, blocked_manifest)
            blocked = parity.evaluate(
                repo_root=repo,
                manifest_path=blocked_path,
                evidence_root=evidence,
                now=NOW,
            )
            self.assertTrue(blocked["parityEligible"])
            self.assertFalse(blocked["legacyRemovalAllowed"])
            self.assertFalse(blocked["mainReplacementAllowed"])
            self.assertFalse(blocked["releaseEligible"])

            approved_manifest = _synthetic_manifest(
                head=head,
                approvals=True,
            )
            approved_path = root / "manifest-approved.json"
            _write_json(approved_path, approved_manifest)
            approved = parity.evaluate(
                repo_root=repo,
                manifest_path=approved_path,
                evidence_root=evidence,
                now=NOW,
            )
        self.assertTrue(approved["manifestValid"])
        self.assertEqual(approved["gateSummary"]["failed"], 0)
        self.assertTrue(approved["workingTreeClean"])
        self.assertTrue(approved["parityEligible"])
        self.assertTrue(approved["legacyRemovalAllowed"])
        self.assertTrue(approved["mainReplacementAllowed"])
        self.assertTrue(approved["releaseEligible"])


if __name__ == "__main__":
    unittest.main()
