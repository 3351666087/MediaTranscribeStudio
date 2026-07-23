from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
from typing import Any

from jsonschema import Draft202012Validator
import pytest

from tools.design_quality.evidence import EvidenceValidationError
from tools.design_quality.validator import (
    audit_project,
    validate_ocr_evidence,
    validate_screenshot_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
SCHEMAS = (
    "design-quality-screenshot-evidence.schema.json",
    "design-quality-ocr-result.schema.json",
    "design-quality-report.schema.json",
)


def _copy_evidence_schemas(project_root: Path) -> None:
    destination = project_root / "contracts"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in SCHEMAS[:2]:
        shutil.copyfile(CONTRACTS / filename, destination / filename)


def _write_test_png(path: Path, width: int = 800, height: int = 600) -> str:
    # The evidence reader only needs the required PNG signature and IHDR
    # dimensions; the contract tests do not pretend this is a real screenshot.
    data = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _screenshot_manifest(relative_path: str, sha256: str) -> dict[str, Any]:
    coverage = {
        "landingLight": "landing-light",
        "landingDark": "landing-dark",
        "speakerStudio": "speaker-studio",
        "qualityLab": "quality-lab",
        "pipelineObservatory": "pipeline-observatory",
        "preferences": "preferences",
        "dragOverlay": "drag-overlay",
    }
    views = {
        "landing-light": ("light", "overview-landing"),
        "landing-dark": ("dark", "overview-landing"),
        "speaker-studio": ("light", "speaker-studio"),
        "quality-lab": ("light", "quality-lab"),
        "pipeline-observatory": ("dark", "pipeline-observatory"),
        "preferences": ("light", "preferences"),
        "drag-overlay": ("dark", "drag-overlay"),
    }
    screenshots = []
    for evidence_id, (theme, view) in views.items():
        screenshots.append(
            {
                "id": evidence_id,
                "path": relative_path,
                "sha256": sha256,
                "width": 800,
                "height": 600,
                "theme": theme,
                "view": view,
                "locale": "en",
                "captureSurface": "tauri-native-webview",
                "realCapture": True,
                "synthetic": False,
                "capturedAt": "2026-07-22T10:00:00Z",
                "notes": "Contract fixture only; not repository release evidence.",
            }
        )
    return {
        "schemaVersion": "1.0.0",
        "kind": "media-transcribe-studio/native-screenshot-evidence",
        "capturedAt": "2026-07-22T10:00:00Z",
        "buildId": "contract-test",
        "sourceRevision": "abcdef0",
        "capturePlatform": {
            "os": "Windows",
            "architecture": "x86_64",
            "displayScale": 1.0,
            "windowWidth": 800,
            "windowHeight": 600,
        },
        "coverage": coverage,
        "screenshots": screenshots,
    }


def _ocr_manifest(relative_path: str, sha256: str) -> dict[str, Any]:
    roles = {
        "day": ("day-asset", "day-background"),
        "night": ("night-asset", "night-background"),
        "scene": ("scene-asset", "scene-background"),
    }
    return {
        "schemaVersion": "1.0.0",
        "kind": "media-transcribe-studio/background-ocr-evidence",
        "createdAt": "2026-07-22T10:00:00Z",
        "realOcrRun": True,
        "synthetic": False,
        "coverage": {
            coverage_key: evidence_id
            for coverage_key, (evidence_id, _role) in roles.items()
        },
        "assets": [
            {
                "id": evidence_id,
                "role": role,
                "path": relative_path,
                "sha256": sha256,
                "width": 800,
                "height": 600,
                "engine": "contract-fixture-ocr",
                "engineVersion": "1.0.0",
                "runAt": "2026-07-22T10:00:00Z",
                "status": "completed",
                "noTextDetected": True,
                "textRegions": [],
                "placementAssessment": {
                    "recommendedBackgroundPosition": "center 48%",
                    "primarySubjectBox": None,
                    "occlusionRisk": "low",
                    "safeControlRegions": [],
                    "notes": "Contract fixture only; not repository OCR evidence.",
                },
            }
            for evidence_id, role in roles.values()
        ],
    }


@pytest.mark.parametrize("filename", SCHEMAS)
def test_design_quality_schemas_are_valid_draft_2020_12(filename: str) -> None:
    schema = json.loads((CONTRACTS / filename).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


def test_native_screenshot_evidence_checks_schema_hash_dimensions_and_coverage(
    tmp_path: Path,
) -> None:
    _copy_evidence_schemas(tmp_path)
    image = tmp_path / "evidence" / "native.png"
    sha256 = _write_test_png(image)
    manifest = _screenshot_manifest("evidence/native.png", sha256)
    manifest_path = tmp_path / "screenshots.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    validated = validate_screenshot_evidence(tmp_path, manifest_path)
    assert validated["coverage"]["dragOverlay"] == "drag-overlay"


def test_native_screenshot_evidence_rejects_browser_only_capture(
    tmp_path: Path,
) -> None:
    _copy_evidence_schemas(tmp_path)
    image = tmp_path / "evidence" / "native.png"
    sha256 = _write_test_png(image)
    manifest = _screenshot_manifest("evidence/native.png", sha256)
    manifest["screenshots"][0]["captureSurface"] = "browser"
    manifest_path = tmp_path / "screenshots.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EvidenceValidationError, match="captureSurface"):
        validate_screenshot_evidence(tmp_path, manifest_path)


def test_native_screenshot_evidence_rejects_stale_hash(tmp_path: Path) -> None:
    _copy_evidence_schemas(tmp_path)
    image = tmp_path / "evidence" / "native.png"
    sha256 = _write_test_png(image)
    manifest = _screenshot_manifest("evidence/native.png", sha256)
    manifest["screenshots"][0]["sha256"] = "0" * 64
    manifest_path = tmp_path / "screenshots.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EvidenceValidationError, match="SHA-256 mismatch"):
        validate_screenshot_evidence(tmp_path, manifest_path)


def test_background_ocr_evidence_checks_real_assets_and_roles(
    tmp_path: Path,
) -> None:
    _copy_evidence_schemas(tmp_path)
    image = tmp_path / "evidence" / "background.png"
    sha256 = _write_test_png(image)
    manifest = _ocr_manifest("evidence/background.png", sha256)
    manifest_path = tmp_path / "ocr.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validated = validate_ocr_evidence(tmp_path, manifest_path)
    assert validated["realOcrRun"] is True
    assert validated["synthetic"] is False


def test_background_ocr_evidence_rejects_inconsistent_no_text_result(
    tmp_path: Path,
) -> None:
    _copy_evidence_schemas(tmp_path)
    image = tmp_path / "evidence" / "background.png"
    sha256 = _write_test_png(image)
    manifest = _ocr_manifest("evidence/background.png", sha256)
    broken = copy.deepcopy(manifest)
    broken["assets"][0]["noTextDetected"] = False
    manifest_path = tmp_path / "ocr.json"
    manifest_path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(EvidenceValidationError, match="textRegions"):
        validate_ocr_evidence(tmp_path, manifest_path)


def test_live_audit_report_matches_the_published_schema() -> None:
    report = audit_project(ROOT).as_dict()
    schema = json.loads(
        (CONTRACTS / "design-quality-report.schema.json").read_text(
            encoding="utf-8"
        )
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    assert errors == []
