"""Runtime validation for the public business-processing output contracts.

The JSON Schemas in ``contracts/`` are the single source of truth.  Loading
them here prevents the durable artifacts written by the worker from drifting
away from the contracts consumed by other processes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class BusinessOutputContractError(ValueError):
    """Raised when an output artifact does not satisfy its public contract."""


_CONTRACTS_DIRECTORY = Path(__file__).resolve().parents[1] / "contracts"
_OUTPUT_SCHEMA_BY_VARIANT = {
    "polish": "polish-output.schema.json",
    "summary": "summary-output.schema.json",
}


def _schema_filename(variant: str) -> str:
    if variant.startswith("translation:"):
        return "translation-output.schema.json"
    try:
        return _OUTPUT_SCHEMA_BY_VARIANT[variant]
    except KeyError as exc:
        raise BusinessOutputContractError(
            f"unsupported business output variant {variant!r}"
        ) from exc


@lru_cache(maxsize=3)
def _validator(filename: str) -> Draft202012Validator:
    path = _CONTRACTS_DIRECTORY / filename
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaError) as exc:
        raise BusinessOutputContractError(
            f"business output contract {filename!r} is unavailable or invalid"
        ) from exc
    return Draft202012Validator(schema)


def validate_business_output_contract(
    value: Mapping[str, Any],
    *,
    variant: str,
) -> None:
    """Validate one artifact against the exact published Draft 2020-12 schema."""

    filename = _schema_filename(variant)
    errors = sorted(
        _validator(filename).iter_errors(dict(value)),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    location = "$"
    for part in error.absolute_path:
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    raise BusinessOutputContractError(
        f"{filename} rejected {location}: {error.message}"
    )


__all__ = [
    "BusinessOutputContractError",
    "validate_business_output_contract",
]
