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
from unittest import mock

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


def _bind_legacy_baseline(
    repo: Path,
    manifest: dict[str, Any],
    head: str,
) -> None:
    entries: dict[str, dict[str, str]] = {}
    for relative in manifest["policy"]["protectedLegacyPaths"]:
        line = _git("ls-tree", head, "--", relative, cwd=repo)
        metadata, resolved_path = line.split("\t", 1)
        self_mode, git_type, object_id = metadata.split()
        if resolved_path != relative:
            raise AssertionError(
                f"git ls-tree returned {resolved_path!r} for {relative!r}"
            )
        entries[relative] = {
            "gitType": git_type,
            "gitMode": self_mode,
            "gitObjectId": object_id,
        }
    manifest["policy"]["protectedLegacyBaseline"] = {
        "sourceCommit": head,
        "entries": entries,
    }


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
    repo: Path,
    head: str,
    approvals: bool,
) -> dict[str, Any]:
    manifest = _load_manifest()
    _bind_legacy_baseline(repo, manifest, head)
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


def _commit_manifest(
    repo: Path,
    manifest: dict[str, Any],
    *,
    message: str,
) -> tuple[Path, str]:
    if _git("branch", "--show-current", cwd=repo) == "main":
        _git("switch", "-c", "codex/parity-test", cwd=repo)
    manifest_path = repo / "docs" / "refactor" / "ultimate-parity.json"
    _write_json(manifest_path, manifest)
    _git("add", manifest_path.relative_to(repo).as_posix(), cwd=repo)
    _git("commit", "-m", message, cwd=repo)
    return manifest_path, _git("rev-parse", "HEAD", cwd=repo)


class UltimateParityManifestTests(unittest.TestCase):
    def test_committed_manifest_is_structurally_valid(self) -> None:
        self.assertEqual(parity.validate_manifest(_load_manifest()), [])

    def test_protected_legacy_baseline_exactly_matches_approved_git_objects(
        self,
    ) -> None:
        manifest = _load_manifest()
        baseline = manifest["policy"]["protectedLegacyBaseline"]
        self.assertEqual(
            set(baseline["entries"]),
            set(manifest["policy"]["protectedLegacyPaths"]),
        )
        for relative, expected in baseline["entries"].items():
            self.assertEqual(
                parity._git_tree_entry(
                    REPO_ROOT,
                    baseline["sourceCommit"],
                    relative,
                ),
                expected,
            )

    def test_protected_legacy_baseline_coverage_is_fail_closed(self) -> None:
        manifest = _load_manifest()
        del manifest["policy"]["protectedLegacyBaseline"]["entries"]["main.py"]
        self.assertIn(
            "policy.protectedLegacyBaseline.entries must exactly cover "
            "policy.protectedLegacyPaths",
            parity.validate_manifest(manifest),
        )

    def test_authorization_timestamps_require_rfc3339_timezones(self) -> None:
        invalid_values = (
            "2026-07-22T08:00:00",
            "2026-07-22 08:00:00Z",
            "2026-02-30T08:00:00Z",
            "not-a-timestamp",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                manifest = _load_manifest()
                manifest["policy"]["authorization"]["legacyRemoval"][
                    "approvedAt"
                ] = value
                self.assertIn(
                    "policy.authorization.legacyRemoval.approvedAt must be an "
                    "RFC3339 timestamp with an explicit timezone",
                    parity.validate_manifest(manifest),
                )

    def test_manifest_rejects_reversed_authorization_timestamps(self) -> None:
        manifest = _load_manifest()
        manifest["policy"]["authorization"]["legacyRemoval"]["approvedAt"] = (
            "2026-07-22T08:00:00Z"
        )
        manifest["policy"]["authorization"]["mainReplacement"]["approvedAt"] = (
            "2026-07-22T07:59:59Z"
        )
        self.assertIn(
            "policy.authorization.mainReplacement.approvedAt must be greater "
            "than or equal to "
            "policy.authorization.legacyRemoval.approvedAt",
            parity.validate_manifest(manifest),
        )

    def test_manifest_allows_equal_authorization_timestamps(self) -> None:
        manifest = _load_manifest()
        for authorization in manifest["policy"]["authorization"].values():
            authorization["approvedAt"] = "2026-07-22T08:00:00Z"
        self.assertFalse(
            any(
                "approvedAt" in error
                for error in parity.validate_manifest(manifest)
            )
        )

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

    def test_naive_evaluation_time_fails_closed(self) -> None:
        result = parity.evaluate(
            repo_root=REPO_ROOT,
            manifest_path=MANIFEST_PATH,
            now=NOW.replace(tzinfo=None),
        )
        self.assertFalse(result["manifestValid"])
        self.assertFalse(result["releaseEligible"])
        self.assertEqual(
            result["configurationErrors"],
            ["evaluation time must be a timezone-aware datetime"],
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
    def _fixture(
        self,
        directory: str,
        *,
        approvals: bool = False,
    ) -> tuple[Path, dict[str, Any], str]:
        repo = Path(directory) / "repo"
        seed_manifest = _load_manifest()
        head = _initialize_clean_repository(repo, seed_manifest)
        manifest = _synthetic_manifest(
            repo=repo,
            head=head,
            approvals=approvals,
        )
        return repo, manifest, head

    def _evaluate(
        self,
        *,
        repo: Path,
        manifest: dict[str, Any],
        head: str,
        parity_eligible: bool = False,
        release_gates_passed: bool = False,
        now: datetime = NOW,
    ) -> dict[str, Any]:
        refs = {
            ref_name: _git("rev-parse", "--verify", ref_name, cwd=repo)
            for ref_name in manifest["policy"]["protectedMainRefs"]
        }
        return parity.evaluate_guardrails(
            repo_root=repo,
            manifest=manifest,
            parity_eligible=parity_eligible,
            release_gates_passed=release_gates_passed,
            head_commit=head,
            refs=refs,
            now=now,
        )

    def test_premature_legacy_deletion_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            removed = repo / manifest["policy"]["protectedLegacyPaths"][0]
            removed.unlink()
            result = self._evaluate(repo=repo, manifest=manifest, head=head)
        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertIn(
            "protected legacy path was removed before authorization: main.py",
            result["legacyProtection"]["violations"],
        )

    def test_premature_legacy_rename_is_reported_as_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            protected = repo / "main.py"
            protected.rename(repo / "main-renamed.py")
            result = self._evaluate(repo=repo, manifest=manifest, head=head)
        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertIn(
            "protected legacy path was removed before authorization: main.py",
            result["legacyProtection"]["violations"],
        )

    def test_in_place_legacy_modification_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            protected = repo / "main.py"
            protected.write_text(
                protected.read_text(encoding="utf-8") + "modified in place\n",
                encoding="utf-8",
            )
            result = self._evaluate(repo=repo, manifest=manifest, head=head)
        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertTrue(
            any(
                "protected legacy content changed before authorization: main.py"
                in message
                for message in result["legacyProtection"]["violations"]
            )
        )

    def test_guard_compares_raw_bytes_and_rejects_a_malicious_clean_filter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            baseline_object_id = manifest["policy"]["protectedLegacyBaseline"][
                "entries"
            ]["main.py"]["gitObjectId"]
            (repo / ".gitattributes").write_text(
                "main.py filter=parity-evil\n",
                encoding="utf-8",
            )
            _git(
                "config",
                "filter.parity-evil.clean",
                f"git cat-file blob {baseline_object_id}",
                cwd=repo,
            )
            (repo / "main.py").write_bytes(b"malicious replacement\n")

            filtered_object_id = _git(
                "hash-object",
                "--path=main.py",
                "--",
                "main.py",
                cwd=repo,
            )
            self.assertEqual(filtered_object_id, baseline_object_id)

            result = self._evaluate(repo=repo, manifest=manifest, head=head)

        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertIn(
            "protected legacy path has a content-transforming Git attribute "
            "before authorization: main.py (filter=parity-evil)",
            result["legacyProtection"]["violations"],
        )
        self.assertTrue(
            any(
                "protected legacy content changed before authorization: main.py"
                in message
                for message in result["legacyProtection"]["violations"]
            )
        )

    def test_guard_rejects_other_content_transforming_git_attributes(self) -> None:
        cases = (
            (
                "main.py working-tree-encoding=UTF-16\n",
                "working-tree-encoding=UTF-16",
            ),
            ("main.py ident\n", "ident=set"),
            ("main.py filter=unspecified\n", "filter=unspecified"),
            ("main.py filter=unset\n", "filter=unset"),
        )
        for attributes, expected in cases:
            with self.subTest(attributes=attributes):
                with tempfile.TemporaryDirectory() as directory:
                    repo, manifest, head = self._fixture(directory)
                    (repo / ".gitattributes").write_text(
                        attributes,
                        encoding="utf-8",
                    )
                    result = self._evaluate(
                        repo=repo,
                        manifest=manifest,
                        head=head,
                    )
                self.assertFalse(result["legacyProtection"]["passed"])
                self.assertIn(
                    "protected legacy path has a content-transforming Git "
                    f"attribute before authorization: main.py ({expected})",
                    result["legacyProtection"]["violations"],
                )

    def test_guard_allows_only_its_own_utf8_lf_crlf_equivalence_rule(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            baseline_object_id = manifest["policy"]["protectedLegacyBaseline"][
                "entries"
            ]["main.py"]["gitObjectId"]
            baseline_bytes = parity._git_blob_bytes(repo, baseline_object_id)
            lf_bytes = baseline_bytes.replace(b"\r\n", b"\n")
            crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
            alternate_bytes = (
                crlf_bytes if crlf_bytes != baseline_bytes else lf_bytes
            )
            self.assertNotEqual(alternate_bytes, baseline_bytes)
            (repo / "main.py").write_bytes(alternate_bytes)

            result = self._evaluate(repo=repo, manifest=manifest, head=head)

        self.assertTrue(result["legacyProtection"]["passed"])
        self.assertTrue(
            parity._guard_content_equivalent(b"text\n", b"text\r\n")
        )
        self.assertFalse(
            parity._guard_content_equivalent(
                b"\x00binary\n",
                b"\x00binary\r\n",
            )
        )

    def test_empty_legacy_shell_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            (repo / "main.py").write_bytes(b"")
            result = self._evaluate(repo=repo, manifest=manifest, head=head)
        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertTrue(
            any(
                "protected legacy content changed before authorization: main.py"
                in message
                for message in result["legacyProtection"]["violations"]
            )
        )

    def test_file_and_directory_type_replacements_are_reported(self) -> None:
        cases = (("main.py", "tree"), ("mts_ui", "blob"))
        for relative, replacement_kind in cases:
            with self.subTest(relative=relative, replacement_kind=replacement_kind):
                with tempfile.TemporaryDirectory() as directory:
                    repo, manifest, head = self._fixture(directory)
                    protected = repo / relative
                    if protected.is_dir():
                        for child in sorted(
                            protected.rglob("*"),
                            key=lambda item: len(item.parts),
                            reverse=True,
                        ):
                            if child.is_file():
                                child.unlink()
                            else:
                                child.rmdir()
                        protected.rmdir()
                    else:
                        protected.unlink()
                    if replacement_kind == "tree":
                        protected.mkdir()
                        (protected / "shim.py").write_text(
                            "pass\n",
                            encoding="utf-8",
                        )
                    else:
                        protected.write_text("pass\n", encoding="utf-8")
                    result = self._evaluate(
                        repo=repo,
                        manifest=manifest,
                        head=head,
                    )
                self.assertFalse(result["legacyProtection"]["passed"])
                self.assertTrue(
                    any(
                        f"protected legacy path type changed before authorization: "
                        f"{relative}"
                        in message
                        for message in result["legacyProtection"]["violations"]
                    )
                )

    def test_actual_symlink_or_junction_reparse_point_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            if os.name == "nt":
                relative = "mts_ui"
                protected = repo / relative
                (protected / ".keep").unlink()
                protected.rmdir()
                target = Path(directory) / "junction-target"
                target.mkdir()
                (target / ".keep").write_text("replacement\n", encoding="utf-8")
                completed = _run(
                    "cmd.exe",
                    "/d",
                    "/c",
                    "mklink",
                    "/J",
                    str(protected),
                    str(target),
                    cwd=repo,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            else:
                relative = "main.py"
                protected = repo / relative
                protected.unlink()
                target = Path(directory) / "symlink-target.py"
                target.write_text("replacement\n", encoding="utf-8")
                protected.symlink_to(target)
            result = self._evaluate(repo=repo, manifest=manifest, head=head)
        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertIn(
            "protected legacy path uses a symlink or junction before "
            f"authorization: {relative}",
            result["legacyProtection"]["violations"],
        )

    def test_windows_reparse_attribute_is_classified_without_following(self) -> None:
        metadata = mock.Mock()
        metadata.st_mode = parity.stat_module.S_IFDIR
        metadata.st_file_attributes = parity.FILE_ATTRIBUTE_REPARSE_POINT
        with mock.patch.object(parity.os, "lstat", return_value=metadata):
            self.assertEqual(
                parity._filesystem_entry_kind(Path("junction-fixture")),
                "reparse",
            )

    def test_committed_legacy_identity_change_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            (repo / "main.py").write_text("replacement launcher\n", encoding="utf-8")
            _git("add", "main.py", cwd=repo)
            _git("commit", "-m", "Replace protected launcher", cwd=repo)
            changed_head = _git("rev-parse", "HEAD", cwd=repo)
            result = self._evaluate(
                repo=repo,
                manifest=manifest,
                head=changed_head,
            )
        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertTrue(
            any(
                "protected legacy Git identity changed before authorization: main.py"
                in message
                for message in result["legacyProtection"]["violations"]
            )
        )

    def test_manifest_baseline_identity_tamper_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory, approvals=True)
            manifest["policy"]["protectedLegacyBaseline"]["entries"]["main.py"][
                "gitObjectId"
            ] = "f" * 40
            result = self._evaluate(
                repo=repo,
                manifest=manifest,
                head=head,
                parity_eligible=True,
                release_gates_passed=True,
            )
        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertFalse(result["legacyRemovalAllowed"])
        self.assertFalse(result["mainReplacementAllowed"])
        self.assertTrue(
            any(
                "approved legacy baseline identity does not match its source commit: "
                "main.py"
                in message
                for message in result["legacyProtection"]["violations"]
            )
        )

    def test_main_replacement_timestamp_cannot_precede_legacy_removal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory, approvals=True)
            manifest["policy"]["authorization"]["legacyRemoval"]["approvedAt"] = (
                (NOW - timedelta(minutes=1)).isoformat()
            )
            manifest["policy"]["authorization"]["mainReplacement"]["approvedAt"] = (
                (NOW - timedelta(minutes=2)).isoformat()
            )
            result = self._evaluate(
                repo=repo,
                manifest=manifest,
                head=head,
                parity_eligible=True,
                release_gates_passed=True,
            )

        self.assertTrue(result["legacyRemovalAllowed"])
        self.assertFalse(result["mainReplacementAllowed"])
        self.assertFalse(result["authorizationOrder"]["passed"])
        self.assertIn(
            "main replacement approvedAt must be greater than or equal to "
            "legacy removal approvedAt",
            result["authorizationOrder"]["violations"],
        )

    def test_equal_authorization_timestamps_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory, approvals=True)
            approved_at = (NOW - timedelta(minutes=1)).isoformat()
            for authorization in manifest["policy"]["authorization"].values():
                authorization["approvedAt"] = approved_at
            result = self._evaluate(
                repo=repo,
                manifest=manifest,
                head=head,
                parity_eligible=True,
                release_gates_passed=True,
            )

        self.assertTrue(result["authorizationOrder"]["passed"])
        self.assertTrue(result["legacyRemovalAllowed"])
        self.assertTrue(result["mainReplacementAllowed"])

    def test_authorization_timestamp_validation_fails_closed(self) -> None:
        cases = (
            (
                "missing timezone",
                "2026-07-22T07:59:00",
                "approvedAt is not a valid timezone-qualified RFC3339 timestamp",
            ),
            (
                "invalid timestamp",
                "2026-02-30T07:59:00Z",
                "approvedAt is not a valid timezone-qualified RFC3339 timestamp",
            ),
            (
                "future timestamp",
                (NOW + timedelta(seconds=1)).isoformat(),
                "approvedAt is in the future",
            ),
        )
        for label, approved_at, expected_message in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    repo, manifest, head = self._fixture(
                        directory,
                        approvals=True,
                    )
                    manifest["policy"]["authorization"]["legacyRemoval"][
                        "approvedAt"
                    ] = (NOW - timedelta(minutes=1)).isoformat()
                    manifest["policy"]["authorization"]["mainReplacement"][
                        "approvedAt"
                    ] = approved_at
                    result = self._evaluate(
                        repo=repo,
                        manifest=manifest,
                        head=head,
                        parity_eligible=True,
                        release_gates_passed=True,
                    )
                self.assertFalse(
                    result["mainReplacementAuthorization"]["passed"]
                )
                self.assertEqual(
                    result["mainReplacementAuthorization"]["message"],
                    expected_message,
                )
                self.assertFalse(result["mainReplacementAllowed"])

    def test_premature_protected_main_change_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            tree = _git("rev-parse", "HEAD^{tree}", cwd=repo)
            moved_ref = _git(
                "commit-tree",
                tree,
                "-p",
                head,
                "-m",
                "Move protected remote ref",
                cwd=repo,
            )
            _git(
                "update-ref",
                "refs/remotes/origin/main",
                moved_ref,
                cwd=repo,
            )
            result = self._evaluate(repo=repo, manifest=manifest, head=head)
        self.assertFalse(result["mainProtection"]["passed"])
        self.assertTrue(
            any(
                "protected ref refs/remotes/origin/main changed before cutover execution"
                in message
                for message in result["mainProtection"]["violations"]
            )
        )

    def test_approved_cutover_still_requires_protected_ref_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory, approvals=True)
            tree = _git("rev-parse", "HEAD^{tree}", cwd=repo)
            moved_ref = _git(
                "commit-tree",
                tree,
                "-p",
                head,
                "-m",
                "Move protected remote ref after approval",
                cwd=repo,
            )
            _git(
                "update-ref",
                "refs/remotes/origin/main",
                moved_ref,
                cwd=repo,
            )
            result = self._evaluate(
                repo=repo,
                manifest=manifest,
                head=head,
                parity_eligible=True,
                release_gates_passed=True,
            )
        self.assertFalse(result["mainProtection"]["passed"])
        self.assertFalse(result["mainReplacementAllowed"])
        self.assertTrue(
            any(
                "protected ref refs/remotes/origin/main changed before cutover "
                "execution"
                in message
                for message in result["mainProtection"]["violations"]
            )
        )

    def test_approved_cutover_still_requires_protected_head_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory, approvals=True)
            (repo / "main.py").write_text(
                "committed replacement launcher\n",
                encoding="utf-8",
            )
            _git("add", "main.py", cwd=repo)
            _git("commit", "-m", "Replace protected launcher", cwd=repo)
            changed_head = _git("rev-parse", "HEAD", cwd=repo)
            result = self._evaluate(
                repo=repo,
                manifest=manifest,
                head=changed_head,
                parity_eligible=True,
                release_gates_passed=True,
            )
        self.assertFalse(result["legacyProtection"]["passed"])
        self.assertFalse(result["legacyRemovalAllowed"])
        self.assertFalse(result["mainReplacementAllowed"])

    def test_guardrail_rejects_a_head_snapshot_that_is_not_current_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory)
            result = self._evaluate(
                repo=repo,
                manifest=manifest,
                head="f" * 40,
            )
        self.assertFalse(result["headProtection"]["passed"])
        self.assertEqual(result["headProtection"]["actualHead"], head)
        self.assertIn(
            "guardrail HEAD snapshot mismatch: "
            f"expected current HEAD {head}, got {'f' * 40}",
            result["headProtection"]["violations"],
        )

    def test_git_state_rejects_hidden_index_flags_and_manifest_tamper(self) -> None:
        for flag, clear_flag, expected in (
            ("--assume-unchanged", "--no-assume-unchanged", "assume-unchanged"),
            ("--skip-worktree", "--no-skip-worktree", "skip-worktree"),
        ):
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as directory:
                    repo, manifest, head = self._fixture(directory)
                    manifest_path = repo / "main.py"
                    _git("update-index", flag, "main.py", cwd=repo)
                    manifest_path.write_text(
                        "hidden release-manifest replacement\n",
                        encoding="utf-8",
                    )
                    state = parity._git_state(
                        repo,
                        manifest["policy"]["protectedMainRefs"].keys(),
                        manifest_path,
                        manifest_path.read_bytes(),
                    )
                    _git("update-index", clear_flag, "main.py", cwd=repo)

                self.assertFalse(state["workingTreeClean"])
                self.assertFalse(state["indexFlagsClean"])
                self.assertIn(
                    {"flag": expected, "path": "main.py"},
                    state["indexFlaggedPaths"],
                )
                self.assertFalse(state["manifestIntegrity"]["passed"])
                self.assertTrue(
                    any(
                        "release manifest raw bytes differ from current HEAD"
                        in message
                        for message in state["manifestIntegrity"]["violations"]
                    )
                )

    def test_guardrail_rejects_naive_evaluation_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, manifest, head = self._fixture(directory, approvals=True)
            result = self._evaluate(
                repo=repo,
                manifest=manifest,
                head=head,
                parity_eligible=True,
                release_gates_passed=True,
                now=NOW.replace(tzinfo=None),
            )
        self.assertFalse(result["legacyRemovalAuthorization"]["passed"])
        self.assertFalse(result["mainReplacementAuthorization"]["passed"])
        self.assertEqual(
            result["legacyRemovalAuthorization"]["message"],
            "evaluation time must be a timezone-aware datetime",
        )
        self.assertFalse(result["legacyRemovalAllowed"])
        self.assertFalse(result["mainReplacementAllowed"])

    def test_real_mov_auto_and_manual_source_hashes_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            seed_manifest = _load_manifest()
            head = _initialize_clean_repository(repo, seed_manifest)
            manifest = _synthetic_manifest(repo=repo, head=head, approvals=True)
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
                repo=repo,
                head=head,
                approvals=False,
            )
            blocked_path, blocked_head = _commit_manifest(
                repo,
                blocked_manifest,
                message="Add blocked parity manifest",
            )
            _write_complete_evidence(
                manifest=blocked_manifest,
                evidence_root=evidence,
                head=blocked_head,
            )
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
                repo=repo,
                head=head,
                approvals=True,
            )
            approved_path, approved_head = _commit_manifest(
                repo,
                approved_manifest,
                message="Approve parity cutover",
            )
            _write_complete_evidence(
                manifest=approved_manifest,
                evidence_root=evidence,
                head=approved_head,
            )
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

    def test_full_evaluate_rejects_hidden_manifest_tamper(self) -> None:
        for flag, clear_flag in (
            ("--assume-unchanged", "--no-assume-unchanged"),
            ("--skip-worktree", "--no-skip-worktree"),
        ):
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    repo = root / "repo"
                    seed_manifest = _load_manifest()
                    baseline_head = _initialize_clean_repository(
                        repo,
                        seed_manifest,
                    )
                    manifest = _synthetic_manifest(
                        repo=repo,
                        head=baseline_head,
                        approvals=True,
                    )
                    manifest_path, head = _commit_manifest(
                        repo,
                        manifest,
                        message="Approve parity cutover",
                    )
                    evidence = root / "evidence"
                    _write_complete_evidence(
                        manifest=manifest,
                        evidence_root=evidence,
                        head=head,
                    )
                    _git(
                        "update-index",
                        flag,
                        manifest_path.relative_to(repo).as_posix(),
                        cwd=repo,
                    )
                    hidden_manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    hidden_manifest["title"] = "Hidden approval rewrite"
                    _write_json(manifest_path, hidden_manifest)
                    result = parity.evaluate(
                        repo_root=repo,
                        manifest_path=manifest_path,
                        evidence_root=evidence,
                        now=NOW,
                    )
                    _git(
                        "update-index",
                        clear_flag,
                        manifest_path.relative_to(repo).as_posix(),
                        cwd=repo,
                    )

                self.assertFalse(result["indexFlagsClean"])
                self.assertFalse(result["manifestIntegrity"]["passed"])
                self.assertFalse(result["workingTreeClean"])
                self.assertFalse(result["legacyRemovalAllowed"])
                self.assertFalse(result["mainReplacementAllowed"])
                self.assertFalse(result["releaseEligible"])

    def test_manifest_decision_bytes_are_bound_against_toctou_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            seed_manifest = _load_manifest()
            baseline_head = _initialize_clean_repository(repo, seed_manifest)
            manifest = _synthetic_manifest(
                repo=repo,
                head=baseline_head,
                approvals=True,
            )
            manifest_path, head = _commit_manifest(
                repo,
                manifest,
                message="Approve parity cutover",
            )
            evidence = root / "evidence"
            _write_complete_evidence(
                manifest=manifest,
                evidence_root=evidence,
                head=head,
            )
            malicious_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            malicious_manifest["title"] = "TOCTOU decision payload"
            _write_json(manifest_path, malicious_manifest)
            original_git_state = parity._git_state

            def restore_then_check(
                repo_root: Path,
                protected_refs: Any,
                checked_manifest_path: Path,
                decision_bytes: bytes,
            ) -> dict[str, Any]:
                _git(
                    "checkout",
                    "--",
                    checked_manifest_path.relative_to(repo_root).as_posix(),
                    cwd=repo_root,
                )
                return original_git_state(
                    repo_root,
                    protected_refs,
                    checked_manifest_path,
                    decision_bytes,
                )

            with mock.patch.object(
                parity,
                "_git_state",
                side_effect=restore_then_check,
            ):
                result = parity.evaluate(
                    repo_root=repo,
                    manifest_path=manifest_path,
                    evidence_root=evidence,
                    now=NOW,
                )

        self.assertFalse(result["manifestIntegrity"]["passed"])
        self.assertTrue(
            any(
                "release manifest decision bytes differ from current HEAD"
                in message
                for message in result["manifestIntegrity"]["violations"]
            )
        )
        self.assertFalse(result["workingTreeClean"])
        self.assertFalse(result["legacyRemovalAllowed"])
        self.assertFalse(result["mainReplacementAllowed"])
        self.assertFalse(result["releaseEligible"])


if __name__ == "__main__":
    unittest.main()
