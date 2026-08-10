"""Validation and coverage helpers for the global speech sample matrix."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_LANGUAGE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})+$")
_SPLITS = frozenset({"development", "regression", "held-out"})
_ACQUISITION_KINDS = frozenset(
    {"hf-viewer-row", "hf-viewer-search-row", "hf-streaming-row"}
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schemaVersion",
        "libraryId",
        "maxDurationSeconds",
        "generatedRoot",
        "randomSeed",
        "sources",
        "cases",
        "derivedMatrices",
        "plannedRealDiarizationSources",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "id",
        "provider",
        "dataset",
        "revision",
        "license",
        "homepage",
        "attribution",
    }
)
_CASE_FIELDS = frozenset(
    {
        "id",
        "sourceId",
        "acquisition",
        "language",
        "region",
        "evaluationSplit",
        "scenario",
        "expectedSpeakerCount",
    }
)
_ACQUISITION_FIELDS = frozenset(
    {
        "kind",
        "config",
        "split",
        "rowIndex",
        "transcriptField",
        "rawTranscriptField",
        "pathField",
        "speakerField",
        "speakerId",
        "recordingField",
        "recordingId",
        "rowIdField",
        "rowId",
        "searchQuery",
    }
)
_REQUIRED_ACQUISITION_FIELDS = frozenset({"kind", "config", "split", "rowIndex"})
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class GlobalSampleLibraryError(ValueError):
    """Raised when the checked-in global sample definition is unsafe."""


@dataclass(frozen=True)
class GlobalSampleSource:
    source_id: str
    provider: str
    dataset: str
    revision: str
    license: str
    homepage: str
    attribution: str


@dataclass(frozen=True)
class GlobalSampleCase:
    case_id: str
    source_id: str
    acquisition: dict[str, Any]
    language: str
    region: str
    evaluation_split: str
    scenarios: tuple[str, ...]
    expected_speaker_count: int


@dataclass(frozen=True)
class GlobalSampleManifest:
    schema_version: str
    library_id: str
    max_duration_seconds: float
    generated_root: str
    random_seed: int
    sources: tuple[GlobalSampleSource, ...]
    cases: tuple[GlobalSampleCase, ...]
    derived_matrices: tuple[dict[str, Any], ...]
    planned_real_diarization_sources: tuple[dict[str, Any], ...]


def _exact_fields(value: dict[str, Any], allowed: frozenset[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise GlobalSampleLibraryError(f"{field} fields are invalid: {'; '.join(details)}")


def _required_and_allowed_fields(
    value: dict[str, Any],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    field: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise GlobalSampleLibraryError(f"{field} fields are invalid: {'; '.join(details)}")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GlobalSampleLibraryError(f"{field} must be non-empty text")
    text = value.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise GlobalSampleLibraryError(f"{field} contains control characters")
    return text


def _identifier(value: Any, field: str) -> str:
    text = _text(value, field)
    if not _ID.fullmatch(text):
        raise GlobalSampleLibraryError(f"{field} is not a safe identifier")
    return text


def _source(value: Any, index: int) -> GlobalSampleSource:
    field = f"sources[{index}]"
    if not isinstance(value, dict):
        raise GlobalSampleLibraryError(f"{field} must be an object")
    _exact_fields(value, _SOURCE_FIELDS, field)
    source_id = _identifier(value["id"], f"{field}.id")
    provider = _text(value["provider"], f"{field}.provider")
    if provider != "huggingface":
        raise GlobalSampleLibraryError(f"{field}.provider must be huggingface")
    dataset = _text(value["dataset"], f"{field}.dataset")
    if dataset.count("/") != 1 or dataset.startswith("/") or dataset.endswith("/"):
        raise GlobalSampleLibraryError(f"{field}.dataset must be namespace/repository")
    revision = _text(value["revision"], f"{field}.revision")
    if not _REVISION.fullmatch(revision):
        raise GlobalSampleLibraryError(f"{field}.revision must be a 40-character commit")
    license_id = _text(value["license"], f"{field}.license").casefold()
    if license_id not in {
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "mit",
        "apache-2.0",
    }:
        raise GlobalSampleLibraryError(f"{field}.license is not approved")
    homepage = _text(value["homepage"], f"{field}.homepage")
    if homepage != f"https://huggingface.co/datasets/{dataset}":
        raise GlobalSampleLibraryError(f"{field}.homepage does not match dataset")
    return GlobalSampleSource(
        source_id=source_id,
        provider=provider,
        dataset=dataset,
        revision=revision,
        license=license_id,
        homepage=homepage,
        attribution=_text(value["attribution"], f"{field}.attribution"),
    )


def _case(value: Any, index: int, source_ids: set[str]) -> GlobalSampleCase:
    field = f"cases[{index}]"
    if not isinstance(value, dict):
        raise GlobalSampleLibraryError(f"{field} must be an object")
    _exact_fields(value, _CASE_FIELDS, field)
    case_id = _identifier(value["id"], f"{field}.id")
    source_id = _identifier(value["sourceId"], f"{field}.sourceId")
    if source_id not in source_ids:
        raise GlobalSampleLibraryError(f"{field}.sourceId is not declared")
    acquisition = value["acquisition"]
    if not isinstance(acquisition, dict):
        raise GlobalSampleLibraryError(f"{field}.acquisition must be an object")
    _required_and_allowed_fields(
        acquisition,
        required=_REQUIRED_ACQUISITION_FIELDS,
        allowed=_ACQUISITION_FIELDS,
        field=f"{field}.acquisition",
    )
    kind = _text(acquisition["kind"], f"{field}.acquisition.kind")
    if kind not in _ACQUISITION_KINDS:
        raise GlobalSampleLibraryError(f"{field}.acquisition.kind is unsupported")
    config = _text(acquisition["config"], f"{field}.acquisition.config")
    split = _text(acquisition["split"], f"{field}.acquisition.split")
    row_index = acquisition["rowIndex"]
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise GlobalSampleLibraryError(
            f"{field}.acquisition.rowIndex must be a non-negative integer"
        )
    language = _text(value["language"], f"{field}.language")
    if not _LANGUAGE.fullmatch(language):
        raise GlobalSampleLibraryError(f"{field}.language must be a concrete BCP-47 tag")
    evaluation_split = _text(value["evaluationSplit"], f"{field}.evaluationSplit")
    if evaluation_split not in _SPLITS:
        raise GlobalSampleLibraryError(f"{field}.evaluationSplit is unsupported")
    scenarios = value["scenario"]
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or any(not isinstance(item, str) or not _ID.fullmatch(item) for item in scenarios)
        or len(scenarios) != len(set(scenarios))
    ):
        raise GlobalSampleLibraryError(f"{field}.scenario must be unique safe IDs")
    expected = value["expectedSpeakerCount"]
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise GlobalSampleLibraryError(
            f"{field}.expectedSpeakerCount must be a positive integer"
        )
    optional_acquisition: dict[str, str] = {}
    for optional_field in (
        "transcriptField",
        "rawTranscriptField",
        "pathField",
        "speakerField",
        "speakerId",
        "recordingField",
        "recordingId",
        "rowIdField",
        "rowId",
        "searchQuery",
    ):
        if optional_field not in acquisition:
            continue
        optional_value = _text(
            acquisition[optional_field],
            f"{field}.acquisition.{optional_field}",
        )
        if optional_field.endswith("Field") and not _FIELD_NAME.fullmatch(
            optional_value
        ):
            raise GlobalSampleLibraryError(
                f"{field}.acquisition.{optional_field} is not a safe field name"
            )
        optional_acquisition[optional_field] = optional_value
    if ("speakerField" in optional_acquisition) != (
        "speakerId" in optional_acquisition
    ):
        raise GlobalSampleLibraryError(
            f"{field}.acquisition speakerField and speakerId must be paired"
        )
    if ("recordingField" in optional_acquisition) != (
        "recordingId" in optional_acquisition
    ):
        raise GlobalSampleLibraryError(
            f"{field}.acquisition recordingField and recordingId must be paired"
        )
    if ("rowIdField" in optional_acquisition) != (
        "rowId" in optional_acquisition
    ):
        raise GlobalSampleLibraryError(
            f"{field}.acquisition rowIdField and rowId must be paired"
        )
    if kind == "hf-viewer-search-row":
        required_search_fields = {"searchQuery", "rowIdField", "rowId"}
        if not required_search_fields <= set(optional_acquisition):
            raise GlobalSampleLibraryError(
                f"{field}.acquisition search rows require searchQuery, "
                "rowIdField, and rowId"
            )
    elif "searchQuery" in optional_acquisition:
        raise GlobalSampleLibraryError(
            f"{field}.acquisition searchQuery requires hf-viewer-search-row"
        )
    return GlobalSampleCase(
        case_id=case_id,
        source_id=source_id,
        acquisition={
            "kind": kind,
            "config": config,
            "split": split,
            "rowIndex": row_index,
            **optional_acquisition,
        },
        language=language,
        region=_text(value["region"], f"{field}.region"),
        evaluation_split=evaluation_split,
        scenarios=tuple(scenarios),
        expected_speaker_count=expected,
    )


def load_global_manifest(path: str | Path) -> GlobalSampleManifest:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalSampleLibraryError(f"cannot read global sample manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise GlobalSampleLibraryError("global sample manifest must be an object")
    _exact_fields(raw, _TOP_LEVEL_FIELDS, "manifest")
    if raw["schemaVersion"] != SCHEMA_VERSION:
        raise GlobalSampleLibraryError("unsupported global sample schemaVersion")
    max_duration = raw["maxDurationSeconds"]
    if (
        isinstance(max_duration, bool)
        or not isinstance(max_duration, (int, float))
        or not 10 <= float(max_duration) <= 90
    ):
        raise GlobalSampleLibraryError("maxDurationSeconds must be between 10 and 90")
    generated_root = _text(raw["generatedRoot"], "generatedRoot")
    generated_path = Path(generated_root)
    if generated_path.is_absolute() or ".." in generated_path.parts:
        raise GlobalSampleLibraryError("generatedRoot must be a safe relative path")
    random_seed = raw["randomSeed"]
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise GlobalSampleLibraryError("randomSeed must be a non-negative integer")
    sources = tuple(_source(value, index) for index, value in enumerate(raw["sources"]))
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise GlobalSampleLibraryError("source IDs must be unique")
    cases = tuple(
        _case(value, index, set(source_ids)) for index, value in enumerate(raw["cases"])
    )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise GlobalSampleLibraryError("case IDs must be unique")
    group_splits: dict[tuple[str, str, str], set[str]] = {}
    for case in cases:
        for group_kind, acquisition_field in (
            ("speaker", "speakerId"),
            ("recording", "recordingId"),
        ):
            group_id = case.acquisition.get(acquisition_field)
            if isinstance(group_id, str):
                group_splits.setdefault(
                    (case.source_id, group_kind, group_id),
                    set(),
                ).add(case.evaluation_split)
    leaked_groups = sorted(
        group
        for group, splits in group_splits.items()
        if len(splits) > 1
    )
    if leaked_groups:
        raise GlobalSampleLibraryError(
            "source speaker/recording groups cross evaluation splits: "
            + ", ".join("/".join(group) for group in leaked_groups)
        )
    if len({case.language for case in cases}) < 12:
        raise GlobalSampleLibraryError("global matrix must cover at least 12 languages")
    if len({case.region for case in cases}) < 8:
        raise GlobalSampleLibraryError("global matrix must cover at least 8 regions")
    if {case.evaluation_split for case in cases} != _SPLITS:
        raise GlobalSampleLibraryError("global matrix must cover all evaluation splits")
    derived = raw["derivedMatrices"]
    planned = raw["plannedRealDiarizationSources"]
    if not isinstance(derived, list) or not derived:
        raise GlobalSampleLibraryError("derivedMatrices must be a non-empty array")
    if not isinstance(planned, list) or not planned:
        raise GlobalSampleLibraryError(
            "plannedRealDiarizationSources must be a non-empty array"
        )
    return GlobalSampleManifest(
        schema_version=SCHEMA_VERSION,
        library_id=_identifier(raw["libraryId"], "libraryId"),
        max_duration_seconds=float(max_duration),
        generated_root=generated_root,
        random_seed=random_seed,
        sources=sources,
        cases=cases,
        derived_matrices=tuple(derived),
        planned_real_diarization_sources=tuple(planned),
    )


def coverage_summary(manifest: GlobalSampleManifest) -> dict[str, Any]:
    return {
        "caseCount": len(manifest.cases),
        "languages": sorted({case.language for case in manifest.cases}),
        "regions": sorted({case.region for case in manifest.cases}),
        "evaluationSplits": sorted(
            {case.evaluation_split for case in manifest.cases}
        ),
        "scenarios": sorted(
            {scenario for case in manifest.cases for scenario in case.scenarios}
        ),
        "sourceIds": sorted({case.source_id for case in manifest.cases}),
    }
