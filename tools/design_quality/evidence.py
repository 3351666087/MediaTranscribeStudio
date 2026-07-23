"""Validation for real native screenshots and background-asset OCR evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class EvidenceValidationError(ValueError):
    """Raised when evidence is missing, synthetic, stale, or malformed."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"{path} must contain a JSON object")
    return value


def _validator(schema_path: Path) -> Draft202012Validator:
    schema = _load_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise EvidenceValidationError(
            f"invalid Draft 2020-12 schema {schema_path}: {exc.message}"
        ) from exc
    return Draft202012Validator(schema)


def _validate_schema(document: Mapping[str, Any], schema_path: Path) -> None:
    errors = sorted(
        _validator(schema_path).iter_errors(document),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if not errors:
        return
    details = "; ".join(
        f"{'/'.join(str(item) for item in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in errors[:12]
    )
    suffix = "" if len(errors) <= 12 else f"; +{len(errors) - 12} more"
    raise EvidenceValidationError(f"schema validation failed: {details}{suffix}")


def _safe_evidence_path(project_root: Path, relative_path: str) -> Path:
    if "\x00" in relative_path:
        raise EvidenceValidationError("evidence path contains a NUL byte")
    path = Path(relative_path)
    if path.is_absolute():
        raise EvidenceValidationError(
            f"evidence path must be project-relative: {relative_path!r}"
        )
    resolved_root = project_root.resolve(strict=True)
    resolved = (resolved_root / path).resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise EvidenceValidationError(
            f"evidence path escapes project root: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise EvidenceValidationError(f"evidence path is not a file: {relative_path!r}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise EvidenceValidationError("file is not a valid PNG")
    if data[12:16] != b"IHDR":
        raise EvidenceValidationError("PNG is missing its leading IHDR chunk")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        raise EvidenceValidationError("PNG has invalid dimensions")
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise EvidenceValidationError("file is not a valid JPEG")
    index = 2
    while index + 4 <= len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            break
        length = int.from_bytes(data[index : index + 2], "big")
        if length < 2 or index + length > len(data):
            break
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if length < 7:
                break
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            if width <= 0 or height <= 0:
                raise EvidenceValidationError("JPEG has invalid dimensions")
            return width, height
        index += length
    raise EvidenceValidationError("JPEG dimensions could not be decoded")


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise EvidenceValidationError("file is not a valid WebP")
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8L":
        if data[20] != 0x2F:
            raise EvidenceValidationError("WebP VP8L signature is invalid")
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    if chunk == b"VP8 ":
        start = data.find(b"\x9d\x01\x2a", 20)
        if start < 0 or start + 7 > len(data):
            raise EvidenceValidationError("WebP VP8 frame header is missing")
        width = int.from_bytes(data[start + 3 : start + 5], "little") & 0x3FFF
        height = int.from_bytes(data[start + 5 : start + 7], "little") & 0x3FFF
        if width <= 0 or height <= 0:
            raise EvidenceValidationError("WebP has invalid dimensions")
        return width, height
    raise EvidenceValidationError(f"unsupported WebP chunk type {chunk!r}")


def image_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    suffix = path.suffix.casefold()
    if suffix == ".png":
        return _png_dimensions(data)
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_dimensions(data)
    if suffix == ".webp":
        return _webp_dimensions(data)
    raise EvidenceValidationError(f"unsupported evidence image format: {path.suffix}")


def _verify_image_record(
    project_root: Path,
    record: Mapping[str, Any],
) -> Path:
    path = _safe_evidence_path(project_root, str(record["path"]))
    actual_sha = _sha256(path)
    if actual_sha != record["sha256"]:
        raise EvidenceValidationError(
            f"SHA-256 mismatch for {record['path']}: "
            f"declared {record['sha256']}, actual {actual_sha}"
        )
    width, height = image_dimensions(path)
    if (width, height) != (record["width"], record["height"]):
        raise EvidenceValidationError(
            f"dimension mismatch for {record['path']}: "
            f"declared {record['width']}x{record['height']}, "
            f"actual {width}x{height}"
        )
    return path


def validate_screenshot_document(
    *,
    project_root: Path,
    manifest_path: Path,
    schema_path: Path,
) -> Mapping[str, Any]:
    document = _load_json(manifest_path)
    _validate_schema(document, schema_path)

    screenshots = document["screenshots"]
    ids = [record["id"] for record in screenshots]
    if len(ids) != len(set(ids)):
        raise EvidenceValidationError("screenshot ids must be unique")
    by_id = {record["id"]: record for record in screenshots}
    coverage = document["coverage"]
    missing = sorted(set(coverage.values()) - set(by_id))
    if missing:
        raise EvidenceValidationError(
            "coverage references missing screenshot ids: " + ", ".join(missing)
        )

    for record in screenshots:
        _verify_image_record(project_root, record)

    if by_id[coverage["landingLight"]]["theme"] != "light":
        raise EvidenceValidationError("landingLight must reference a light screenshot")
    if by_id[coverage["landingDark"]]["theme"] != "dark":
        raise EvidenceValidationError("landingDark must reference a dark screenshot")

    expected_views = {
        "landingLight": "overview-landing",
        "landingDark": "overview-landing",
        "speakerStudio": "speaker-studio",
        "qualityLab": "quality-lab",
        "pipelineObservatory": "pipeline-observatory",
        "preferences": "preferences",
        "dragOverlay": "drag-overlay",
    }
    for coverage_key, expected_view in expected_views.items():
        record = by_id[coverage[coverage_key]]
        if record["view"] != expected_view:
            raise EvidenceValidationError(
                f"{coverage_key} must reference view {expected_view!r}"
            )
        if record["captureSurface"] != "tauri-native-webview":
            raise EvidenceValidationError(
                f"{coverage_key} is browser-only or not a Tauri native capture"
            )
    return document


def validate_ocr_document(
    *,
    project_root: Path,
    manifest_path: Path,
    schema_path: Path,
) -> Mapping[str, Any]:
    document = _load_json(manifest_path)
    _validate_schema(document, schema_path)

    assets = document["assets"]
    ids = [record["id"] for record in assets]
    if len(ids) != len(set(ids)):
        raise EvidenceValidationError("OCR asset ids must be unique")
    by_id = {record["id"]: record for record in assets}
    coverage = document["coverage"]
    missing = sorted(set(coverage.values()) - set(by_id))
    if missing:
        raise EvidenceValidationError(
            "OCR coverage references missing asset ids: " + ", ".join(missing)
        )

    expected_roles = {
        "day": "day-background",
        "night": "night-background",
        "scene": "scene-background",
    }
    for coverage_key, expected_role in expected_roles.items():
        record = by_id[coverage[coverage_key]]
        if record["role"] != expected_role:
            raise EvidenceValidationError(
                f"{coverage_key} must reference role {expected_role!r}"
            )

    for record in assets:
        _verify_image_record(project_root, record)
        regions = record["textRegions"]
        if record["noTextDetected"] and regions:
            raise EvidenceValidationError(
                f"{record['id']} says noTextDetected but contains text regions"
            )
        if not record["noTextDetected"] and not regions:
            raise EvidenceValidationError(
                f"{record['id']} must contain text regions or set noTextDetected=true"
            )
    return document
