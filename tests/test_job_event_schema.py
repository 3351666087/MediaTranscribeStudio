from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts" / "job-event.schema.json"


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _event(event_type: str) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "eventId": f"event-{event_type}",
        "jobId": "job-contract-parity",
        "sequence": 0,
        "timestamp": "2026-07-22T00:00:00Z",
        "type": event_type,
        "payload": {},
    }


@pytest.mark.parametrize(
    "event_type",
    (
        "review.required",
        "review.decision.persisted",
    ),
)
def test_review_events_are_part_of_the_public_worker_contract(
    event_type: str,
) -> None:
    _validator().validate(_event(event_type))


def test_unknown_review_event_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _validator().validate(_event("review.decision.saved"))
