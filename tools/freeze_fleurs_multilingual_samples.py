"""Freeze a truth-bearing multilingual FLEURS evaluation matrix on local disk."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from tools.build_global_sample_library import (  # noqa: E402
    _build_case,
    _request_json,
    _sha256,
)
from tools.global_sample_library import (  # noqa: E402
    GlobalSampleCase,
    GlobalSampleLibraryError,
    GlobalSampleManifest,
    GlobalSampleSource,
    load_global_manifest,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "sample_library" / "global-manifest.v1.json"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / ".runtime_cache" / "sample-library" / "global"
)
DEFAULT_REFERENCE = DEFAULT_OUTPUT_ROOT / "fleurs-multilingual-frozen.v1.json"
TARGET_LANGUAGES = {
    "ar_eg": "ar-EG",
    "es_419": "es-419",
    "hi_in": "hi-IN",
    "ja_jp": "ja-JP",
    "ko_kr": "ko-KR",
    "yue_hant_hk": "yue-Hant-HK",
}
SOURCE_SPLITS = {"development": "validation", "held-out": "test"}
CASES_PER_BUCKET = 3


def _fleurs_source(manifest: GlobalSampleManifest) -> GlobalSampleSource:
    matches = [source for source in manifest.sources if source.source_id == "fleurs"]
    if len(matches) != 1:
        raise GlobalSampleLibraryError(
            "manifest must declare exactly one FLEURS source"
        )
    source = matches[0]
    if (
        source.dataset != "google/fleurs"
        or source.license != "cc-by-4.0"
        or len(source.revision) != 40
    ):
        raise GlobalSampleLibraryError("FLEURS source provenance is not pinned")
    return source


def select_fleurs_cases(
    manifest: GlobalSampleManifest,
) -> tuple[GlobalSampleSource, list[GlobalSampleCase]]:
    source = _fleurs_source(manifest)
    selected = [
        case
        for case in manifest.cases
        if case.source_id == source.source_id
        and case.acquisition["config"] in TARGET_LANGUAGES
        and case.evaluation_split in SOURCE_SPLITS
    ]
    counts = Counter(
        (case.acquisition["config"], case.evaluation_split) for case in selected
    )
    expected = {
        (config, split): CASES_PER_BUCKET
        for config in TARGET_LANGUAGES
        for split in SOURCE_SPLITS
    }
    if counts != expected:
        raise GlobalSampleLibraryError(
            "FLEURS matrix must contain exactly three cases per language and split"
        )
    source_rows: set[tuple[str, str, int]] = set()
    for case in selected:
        config = case.acquisition["config"]
        if case.language != TARGET_LANGUAGES[config]:
            raise GlobalSampleLibraryError(
                f"{case.case_id} language does not match FLEURS config"
            )
        expected_source_split = SOURCE_SPLITS[case.evaluation_split]
        if case.acquisition["split"] != expected_source_split:
            raise GlobalSampleLibraryError(
                f"{case.case_id} is assigned to the wrong FLEURS source split"
            )
        if case.acquisition["kind"] not in {
            "hf-viewer-row",
            "hf-viewer-search-row",
        }:
            raise GlobalSampleLibraryError(
                f"{case.case_id} must be retrievable through Dataset Viewer"
            )
        locator = (
            config,
            case.acquisition["split"],
            case.acquisition["rowIndex"],
        )
        if locator in source_rows:
            raise GlobalSampleLibraryError("FLEURS source rows must be unique")
        source_rows.add(locator)
    return source, sorted(
        selected,
        key=lambda case: (
            case.acquisition["config"],
            case.evaluation_split,
            case.acquisition["rowIndex"],
        ),
    )


def viewer_evidence(
    source: GlobalSampleSource,
    cases: Sequence[GlobalSampleCase],
) -> dict[str, Any]:
    dataset = source.dataset
    valid_url = (
        "https://datasets-server.huggingface.co/is-valid?dataset="
        "google%2Ffleurs"
    )
    valid = _request_json(valid_url)
    if valid.get("viewer") is not True or valid.get("search") is not True:
        raise GlobalSampleLibraryError("FLEURS Dataset Viewer is not fully available")
    splits_url = (
        "https://datasets-server.huggingface.co/splits?dataset="
        "google%2Ffleurs"
    )
    split_body = _request_json(splits_url)
    split_rows = split_body.get("splits")
    if not isinstance(split_rows, list):
        raise GlobalSampleLibraryError("FLEURS Dataset Viewer splits are invalid")
    available = {
        (row.get("config"), row.get("split"))
        for row in split_rows
        if isinstance(row, dict) and row.get("dataset") == dataset
    }
    selected_pairs = sorted(
        {
            (case.acquisition["config"], case.acquisition["split"])
            for case in cases
        }
    )
    missing = [pair for pair in selected_pairs if pair not in available]
    if missing:
        raise GlobalSampleLibraryError(
            "FLEURS Dataset Viewer is missing selected splits: "
            + ", ".join("/".join(pair) for pair in missing)
        )
    return {
        "baseUrl": "https://datasets-server.huggingface.co",
        "dataset": dataset,
        "isValid": {
            key: valid.get(key)
            for key in ("preview", "viewer", "search", "filter", "statistics")
        },
        "selectedConfigSplits": [
            {"config": config, "split": split}
            for config, split in selected_pairs
        ],
        "assetRevisionValidation": "required-per-row",
    }


def _installed_case(
    *,
    manifest: GlobalSampleManifest,
    source: GlobalSampleSource,
    case: GlobalSampleCase,
    output_root: Path,
    staging_root: Path,
) -> dict[str, Any]:
    row = _build_case(
        manifest=manifest,
        source=source,
        case=case,
        output_root=staging_root,
    )
    staged = staging_root / row["path"]
    target = output_root / "audio" / f"{case.case_id}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if _sha256(target) != row["sha256"]:
            raise GlobalSampleLibraryError(
                f"existing media differs from the pinned Viewer row: {case.case_id}"
            )
        staged.unlink()
        disposition = "verified-existing"
    else:
        os.replace(staged, target)
        disposition = "installed"
    row.update(
        {
            "path": target.relative_to(output_root).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
            "durationSeconds": round(row["durationSeconds"], 6),
            "downloadedAt": datetime.fromtimestamp(
                target.stat().st_mtime,
                UTC,
            ).isoformat(),
            "localDisposition": disposition,
            "tuningEligible": case.evaluation_split == "development",
        }
    )
    return row


def reference_body(
    *,
    manifest_path: Path,
    source: GlobalSampleSource,
    viewer: dict[str, Any],
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not cases:
        raise GlobalSampleLibraryError("FLEURS frozen matrix contains no cases")
    recording_paths = [str(case["sourceRow"]["path"]) for case in cases]
    if len(recording_paths) != len(set(recording_paths)):
        raise GlobalSampleLibraryError("FLEURS recording paths are not unique")
    cross_split_hashes: dict[str, set[str]] = {}
    for case in cases:
        cross_split_hashes.setdefault(case["sha256"], set()).add(
            case["evaluationSplit"]
        )
    leaked_hashes = sorted(
        digest for digest, splits in cross_split_hashes.items() if len(splits) > 1
    )
    if leaked_hashes:
        raise GlobalSampleLibraryError("FLEURS audio crosses evaluation splits")
    by_language_split = {
        language: {
            split: sum(
                case["language"] == language
                and case["evaluationSplit"] == split
                for case in cases
            )
            for split in SOURCE_SPLITS
        }
        for language in TARGET_LANGUAGES.values()
    }
    body = {
        "schemaVersion": "1.0.0",
        "artifactType": "fleurs-multilingual-frozen-reference",
        "sourceManifest": {
            "path": str(manifest_path.resolve()),
            "fileSha256": sha256_file(manifest_path),
        },
        "source": {
            "id": source.source_id,
            "dataset": source.dataset,
            "revision": source.revision,
            "license": source.license,
            "homepage": source.homepage,
            "attribution": source.attribution,
        },
        "viewerEvidence": viewer,
        "splitPolicy": {
            "development": {
                "sourceSplit": "validation",
                "tuningEligible": True,
            },
            "held-out": {
                "sourceSplit": "test",
                "tuningEligible": False,
                "unlockRequiredForProductRun": True,
            },
        },
        "isolation": {
            "recordingIdentityField": "sourceRow.path",
            "recordingPathsUnique": True,
            "crossEvaluationAudioShaLeakage": False,
            "speakerIdentityAvailable": False,
            "speakerIsolationClaimed": False,
            "limitation": (
                "The public FLEURS Viewer schema exposes recording path and row ID "
                "but no speaker ID; official validation/test splits and unique "
                "recordings are frozen, but speaker-disjointness is not claimed."
            ),
        },
        "truthPersistencePolicy": {
            "referenceTruthPersisted": True,
            "blindProductManifestSource": "truth-free checked-in global manifest",
            "referenceManifestMayEnterBlindPacket": False,
        },
        "counts": {
            "cases": len(cases),
            "languages": len(by_language_split),
            "byLanguageAndSplit": by_language_split,
            "durationSeconds": round(
                sum(float(case["durationSeconds"]) for case in cases),
                6,
            ),
        },
        "cases": list(cases),
    }
    return {**body, "canonicalSha256": canonical_json_sha256(body)}


def freeze(
    *,
    manifest_path: Path,
    output_root: Path,
    reference_path: Path,
) -> dict[str, Any]:
    manifest = load_global_manifest(manifest_path)
    source, selected = select_fleurs_cases(manifest)
    viewer = viewer_evidence(source, selected)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=".fleurs-freeze-",
        dir=output_root,
    ) as temporary:
        staging_root = Path(temporary)
        for case in selected:
            print(f"freeze {case.case_id}", flush=True)
            rows.append(
                _installed_case(
                    manifest=manifest,
                    source=source,
                    case=case,
                    output_root=output_root,
                    staging_root=staging_root,
                )
            )
    value = reference_body(
        manifest_path=manifest_path,
        source=source,
        viewer=viewer,
        cases=rows,
    )
    atomic_write_json(reference_path, value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    value = freeze(
        manifest_path=args.manifest.resolve(strict=True),
        output_root=args.output_root.resolve(),
        reference_path=args.reference.resolve(),
    )
    print(
        json.dumps(
            {
                "reference": str(args.reference.resolve()),
                "fileSha256": sha256_file(args.reference.resolve()),
                "canonicalSha256": value["canonicalSha256"],
                "cases": value["counts"]["cases"],
                "durationSeconds": value["counts"]["durationSeconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
