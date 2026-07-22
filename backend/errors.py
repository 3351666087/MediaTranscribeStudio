"""Structured, fail-closed errors for the offline worker boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class WorkerError(Exception):
    """An expected worker failure that is safe to serialize to JSON."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})
        self.retryable = bool(retryable)

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class JobCancelled(WorkerError):
    """Raised cooperatively when a cancellation token is observed."""

    def __init__(self) -> None:
        super().__init__("JOB_CANCELLED", "job cancellation was requested")


def invalid_request(message: str, **details: Any) -> WorkerError:
    return WorkerError("INVALID_REQUEST", message, details=details)
