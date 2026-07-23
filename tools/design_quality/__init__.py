"""Fail-closed Design Pack quality gates for MediaTranscribeStudio."""

from .validator import (
    AuditReport,
    Finding,
    audit_project,
    validate_ocr_evidence,
    validate_screenshot_evidence,
)

__all__ = [
    "AuditReport",
    "Finding",
    "audit_project",
    "validate_ocr_evidence",
    "validate_screenshot_evidence",
]
