#!/usr/bin/env python3
"""Prepare a truth-isolated, architecture-independent speech audit batch.

This module deliberately imports no production backend code. It normalizes the
existing sample assets into three artifacts:

* a blind media manifest for processing;
* a sealed reference manifest for later scoring; and
* an empty audit ledger for recording tools, evidence, uncertainty, and output.

The reference is "sealed" by workflow separation and a SHA-256 binding. It is
not encrypted; reviewers must keep it away from the processing context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / ".runtime_cache"
    / "sample-library"
    / "first-principles"
)
SCHEMA_VERSION = "1.1.0"


@dataclass(frozen=True)
class SourceSpec:
    collection: str
    kind: str
    path: Path
    partition_override: str | None = None


DEFAULT_SOURCES = (
    SourceSpec(
        "global-single-speaker",
        "global-single-speaker",
        PROJECT_ROOT
        / ".runtime_cache/sample-library/global/global-sample-library.resolved.v1.json",
    ),
    SourceSpec(
        "global-real-diarization",
        "real-diarization",
        PROJECT_ROOT
        / ".runtime_cache/sample-library/global/real/global-real-diarization.resolved.v1.json",
    ),
    SourceSpec(
        "aishell4-real-diarization",
        "real-diarization",
        PROJECT_ROOT
        / ".runtime_cache/sample-library/global/real-aishell4-v1/global-real-diarization.resolved.v1.json",
        "development-reused",
    ),
    SourceSpec(
        "global-derived-diarization",
        "derived-diarization",
        PROJECT_ROOT
        / ".runtime_cache/sample-library/global/derived/global-derived-diarization.resolved.v1.json",
    ),
    SourceSpec(
        "voice-activity",
        "voice-activity",
        PROJECT_ROOT
        / ".runtime_cache/sample-library/voice-activity/voice-activity-samples.resolved.v1.json",
    ),
    SourceSpec(
        "code-switch",
        "code-switch",
        PROJECT_ROOT
        / ".runtime_cache/sample-library/code-switch-short-v2/code-switch-sample-library.resolved.v1.json",
    ),
)


class FirstPrinciplesBatchError(ValueError):
    """Raised when an input cannot be made into a trustworthy audit case."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FirstPrinciplesBatchError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise FirstPrinciplesBatchError(f"manifest {path} must contain a cases array")
    return value


def _nonempty_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _truth_flag(value: Any, field: str) -> bool:
    return isinstance(value, Mapping) and value.get(field) is True


def _duration(case: Mapping[str, Any]) -> float:
    direct = case.get("durationSeconds")
    audio = case.get("audio")
    nested = audio.get("durationSeconds") if isinstance(audio, Mapping) else None
    selection = case.get("windowSelection")
    selected = (
        selection.get("durationSeconds") if isinstance(selection, Mapping) else None
    )
    for value in (direct, nested, selected):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    raise FirstPrinciplesBatchError(f"{case.get('id')} has no positive duration")


def _partition(case: Mapping[str, Any]) -> str:
    value = case.get("evaluationSplit") or case.get("evaluationRole")
    if value in {"development", "regression", "held-out"}:
        return str(value)
    # These cases were already used during development and can never be blind
    # promotion evidence, even when their upstream dataset called the split test.
    return "development-reused"


def _origin(case: Mapping[str, Any], kind: str) -> str:
    value = case.get("realOrSynthetic")
    if value in {"real-recording", "synthetic-mixture", "derived-mixture"}:
        return str(value)
    if kind == "derived-diarization":
        return "derived-mixture"
    if kind == "voice-activity" and case.get("sourceId") == "generated-silence":
        return "synthetic-signal"
    return "real-recording"


def _language_list(case: Mapping[str, Any], *, eligible: bool) -> list[str] | None:
    if not eligible:
        return None
    expected = case.get("expectedLanguages")
    if isinstance(expected, list) and expected and all(
        isinstance(item, str) and item for item in expected
    ):
        return list(dict.fromkeys(expected))
    language = _nonempty_text(case.get("language"))
    if language:
        return [language]
    turns = case.get("turns")
    if isinstance(turns, list):
        languages = [
            turn.get("language")
            for turn in turns
            if isinstance(turn, Mapping) and _nonempty_text(turn.get("language"))
        ]
        if languages:
            return list(dict.fromkeys(str(item) for item in languages))
    return None


def _normalized_eligibility(case: Mapping[str, Any], kind: str) -> dict[str, bool]:
    truth = case.get("truthEligibility")
    if kind == "global-single-speaker":
        return {
            "lexicalSpeechPresence": True,
            "speakerCount": True,
            "speakerTimeline": False,
            "overlap": False,
            "transcript": True,
            "languageDocument": True,
            "languageTimeline": False,
            "translation": False,
        }
    if kind == "voice-activity":
        return {
            "lexicalSpeechPresence": _truth_flag(truth, "lexicalSpeech"),
            "speakerCount": False,
            "speakerTimeline": False,
            "overlap": False,
            "transcript": False,
            "languageDocument": False,
            "languageTimeline": False,
            "translation": False,
        }
    return {
        "lexicalSpeechPresence": True,
        "speakerCount": _truth_flag(truth, "speakerCount"),
        "speakerTimeline": _truth_flag(truth, "turnBoundaries"),
        "overlap": _truth_flag(truth, "overlap"),
        "transcript": _truth_flag(truth, "asr"),
        "languageDocument": (
            _truth_flag(truth, "language")
            or _truth_flag(truth, "languageDocumentPair")
        ),
        "languageTimeline": (
            _truth_flag(truth, "languageTiming")
            or _truth_flag(truth, "languageWords")
        ),
        "translation": False,
    }


def _reference_turns(case: Mapping[str, Any], eligible: bool) -> list[Any] | None:
    if not eligible:
        return None
    turns = case.get("referenceTranscriptTurns") or case.get("turns")
    return list(turns) if isinstance(turns, list) else None


def _reference_transcript(case: Mapping[str, Any], eligible: bool) -> str | None:
    if not eligible:
        return None
    for field in ("scoringTranscript", "expectedTranscript", "transcript"):
        value = _nonempty_text(case.get(field))
        if value:
            return value
    turns = _reference_turns(case, True)
    if turns:
        text = " ".join(
            str(turn.get("transcript")).strip()
            for turn in turns
            if isinstance(turn, Mapping) and _nonempty_text(turn.get("transcript"))
        )
        return text or None
    return None


def _case_limitations(
    *,
    case: Mapping[str, Any],
    origin: str,
    partition: str,
    eligibility: Mapping[str, bool],
) -> list[str]:
    limitations: list[str] = []
    if partition == "development-reused":
        limitations.append("previously-used-development-case-not-promotion-evidence")
    if origin != "real-recording":
        limitations.append("non-real-distribution-diagnostic-only")
    if not eligibility["transcript"]:
        limitations.append("no-qualified-reference-transcript")
    if not eligibility["speakerTimeline"]:
        limitations.append("no-qualified-speaker-timeline")
    if not eligibility["languageTimeline"]:
        limitations.append("no-qualified-word-or-time-language-reference")
    if not eligibility["translation"]:
        limitations.append("no-independent-reference-translation")
    issues = case.get("sourceAnnotationIssues")
    if isinstance(issues, list) and issues:
        limitations.append("source-annotation-issues-present")
    return limitations


def _normalized_case(spec: SourceSpec, manifest_path: Path, case: Any) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        raise FirstPrinciplesBatchError(f"{spec.collection} contains a non-object case")
    raw_id = _nonempty_text(case.get("id"))
    relative_path = _nonempty_text(case.get("path"))
    media_sha = _nonempty_text(case.get("sha256"))
    if not raw_id or not relative_path or not media_sha or len(media_sha) != 64:
        raise FirstPrinciplesBatchError(
            f"{spec.collection}/{raw_id or '<unknown>'} lacks id/path/SHA-256"
        )
    media_path = (manifest_path.parent / relative_path).resolve()
    if not media_path.is_file():
        raise FirstPrinciplesBatchError(f"media is missing: {media_path}")

    eligibility = _normalized_eligibility(case, spec.kind)
    origin = _origin(case, spec.kind)
    partition = spec.partition_override or _partition(case)
    lexical_speech = (
        bool(case.get("expectedLexicalSpeech"))
        if spec.kind == "voice-activity"
        else True
    )
    speaker_count = (
        case.get("expectedSpeakerCount") if eligibility["speakerCount"] else None
    )
    if speaker_count is not None and (
        isinstance(speaker_count, bool)
        or not isinstance(speaker_count, int)
        or speaker_count < 1
    ):
        raise FirstPrinciplesBatchError(f"{spec.collection}/{raw_id} has bad speaker count")
    languages = _language_list(case, eligible=eligibility["languageDocument"])
    transcript = _reference_transcript(case, eligibility["transcript"])
    turns = _reference_turns(case, eligibility["speakerTimeline"])
    overlap = case.get("overlapIntervals") if eligibility["overlap"] else None
    if overlap is not None and not isinstance(overlap, list):
        raise FirstPrinciplesBatchError(f"{spec.collection}/{raw_id} has bad overlap truth")

    raw_scenarios = case.get("scenario")
    scenarios = list(raw_scenarios) if isinstance(raw_scenarios, list) else []
    if spec.kind == "voice-activity":
        for value in (case.get("signalClass"), case.get("category")):
            text = _nonempty_text(value)
            if text:
                scenarios.append(text)
    if origin != "real-recording":
        scenarios.append(origin)
    scenarios = sorted(set(str(item) for item in scenarios if _nonempty_text(item)))
    limitations = _case_limitations(
        case=case,
        origin=origin,
        partition=partition,
        eligibility=eligibility,
    )

    if not lexical_speech:
        reference_class = "negative-lexical-speech-oracle"
    elif origin != "real-recording":
        reference_class = "synthetic-diagnostic"
    elif all(
        eligibility[field]
        for field in (
            "speakerCount",
            "speakerTimeline",
            "transcript",
            "languageDocument",
        )
    ):
        reference_class = "full-final-state-oracle"
    else:
        reference_class = "partial-oracle"

    return {
        "caseId": f"{spec.collection}/{raw_id}",
        "sourceCaseId": raw_id,
        "collection": spec.collection,
        "partition": partition,
        "origin": origin,
        "durationSeconds": _duration(case),
        "media": {
            "path": str(media_path),
            "sha256": media_sha,
            "declaredBytes": case.get("bytes"),
        },
        "scenarios": scenarios,
        "referenceClass": reference_class,
        "eligibility": eligibility,
        "reference": {
            "lexicalSpeechPresent": lexical_speech,
            "humanVocalizationPresent": (
                case.get("humanVocalization")
                if spec.kind == "voice-activity"
                else None
            ),
            "speakerCount": speaker_count,
            "languages": languages,
            "transcript": transcript,
            "speakerTurns": turns,
            "overlapIntervals": list(overlap) if isinstance(overlap, list) else None,
            "languageTruth": (
                case.get("languageTruth")
                if eligibility["languageDocument"]
                else None
            ),
            "translation": None,
        },
        "provenance": {
            "resolvedManifest": str(manifest_path.resolve()),
            "resolvedManifestSha256": _sha256_file(manifest_path),
            "sourceId": case.get("sourceId"),
            "sourceDataset": case.get("sourceDataset"),
            "sourceRevision": case.get("sourceRevision"),
            "sourceArtifactSha256": case.get("sourceArtifactSha256"),
            "realOrSynthetic": case.get("realOrSynthetic"),
        },
        "limitations": limitations,
    }


def load_cases(sources: Sequence[SourceSpec]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in sources:
        manifest = _load_object(spec.path)
        for raw_case in manifest["cases"]:
            case = _normalized_case(spec, spec.path, raw_case)
            if case["caseId"] in seen:
                raise FirstPrinciplesBatchError(f"duplicate case ID {case['caseId']}")
            seen.add(case["caseId"])
            cases.append(case)
    return cases


def _speaker_bucket(value: Any) -> str:
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if isinstance(value, int) and 3 <= value <= 5:
        return "3-5"
    if isinstance(value, int) and value >= 6:
        return "6+"
    return "unknown"


def _duration_bucket(value: float) -> str:
    if value <= 10:
        return "short"
    if value <= 30:
        return "medium"
    return "long"


def case_features(case: Mapping[str, Any]) -> frozenset[str]:
    reference = case["reference"]
    eligibility = case["eligibility"]
    features = {
        f"collection:{case['collection']}",
        f"origin:{case['origin']}",
        f"speech:{str(reference['lexicalSpeechPresent']).lower()}",
        f"speaker:{_speaker_bucket(reference['speakerCount'])}",
        f"duration:{_duration_bucket(case['durationSeconds'])}",
        f"reference:{case['referenceClass']}",
    }
    features.update(f"scenario:{value}" for value in case["scenarios"])
    features.update(
        f"language:{value}" for value in (reference.get("languages") or ["unknown"])
    )
    features.update(
        f"eligible:{field}" for field, enabled in eligibility.items() if enabled
    )
    return frozenset(features)


def _feature_weight(feature: str) -> float:
    prefix = feature.split(":", 1)[0]
    return {
        "collection": 80.0,
        "origin": 50.0,
        "speech": 50.0,
        "speaker": 60.0,
        "reference": 40.0,
        "duration": 20.0,
        "eligible": 10.0,
        "scenario": 2.0,
        "language": 1.0,
    }[prefix]


def select_diverse_cases(
    cases: Sequence[dict[str, Any]],
    *,
    maximum: int,
    include_held_out: bool = False,
) -> list[dict[str, Any]]:
    if maximum < 1:
        raise FirstPrinciplesBatchError("maximum must be positive")
    eligible = [
        case
        for case in cases
        if include_held_out or case["partition"] != "held-out"
    ]
    if not eligible:
        raise FirstPrinciplesBatchError("no cases remain after partition filtering")
    feature_sets = {case["caseId"]: case_features(case) for case in eligible}
    frequencies = Counter(feature for values in feature_sets.values() for feature in values)
    selected: list[dict[str, Any]] = []
    uncovered = set(frequencies)
    remaining = {case["caseId"]: case for case in eligible}

    while remaining and len(selected) < maximum:
        def score(case_id: str) -> tuple[float, int, str]:
            values = feature_sets[case_id]
            novelty = sum(
                _feature_weight(value) / frequencies[value]
                for value in values & uncovered
            )
            # Prefer cases that cover more truth domains when diversity is tied.
            truth_width = sum(
                bool(value) for value in remaining[case_id]["eligibility"].values()
            )
            return (novelty, truth_width, case_id)

        winner_id = max(remaining, key=score)
        winner = remaining.pop(winner_id)
        selected.append(winner)
        uncovered.difference_update(feature_sets[winner_id])
    return selected


def _coverage(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(cases)
    return {
        "caseCount": len(rows),
        "collections": sorted({str(row["collection"]) for row in rows}),
        "origins": dict(sorted(Counter(str(row["origin"]) for row in rows).items())),
        "partitions": dict(
            sorted(Counter(str(row["partition"]) for row in rows).items())
        ),
        "lexicalSpeechPresence": dict(
            sorted(
                Counter(
                    str(bool(row["reference"]["lexicalSpeechPresent"])).lower()
                    for row in rows
                ).items()
            )
        ),
        "speakerCounts": dict(
            sorted(
                Counter(
                    str(row["reference"]["speakerCount"])
                    if row["reference"]["speakerCount"] is not None
                    else "unknown"
                    for row in rows
                ).items()
            )
        ),
        "languages": sorted(
            {
                language
                for row in rows
                for language in (row["reference"].get("languages") or [])
            }
        ),
        "scenarios": sorted(
            {scenario for row in rows for scenario in row.get("scenarios", [])}
        ),
        "referenceClasses": dict(
            sorted(Counter(str(row["referenceClass"]) for row in rows).items())
        ),
    }


def prepare_batch(
    *,
    sources: Sequence[SourceSpec],
    output_root: Path,
    maximum: int,
    include_held_out: bool = False,
    verify_media_hashes: bool = False,
) -> dict[str, Any]:
    all_cases = load_cases(sources)
    selected = select_diverse_cases(
        all_cases,
        maximum=maximum,
        include_held_out=include_held_out,
    )
    if verify_media_hashes:
        for case in selected:
            actual = _sha256_file(Path(case["media"]["path"]))
            if actual != case["media"]["sha256"]:
                raise FirstPrinciplesBatchError(
                    f"media SHA-256 mismatch for {case['caseId']}: {actual}"
                )

    source_bindings = [
        {
            "collection": source.collection,
            "kind": source.kind,
            "path": str(source.path.resolve()),
            "sha256": _sha256_file(source.path),
            "partitionOverride": source.partition_override,
        }
        for source in sources
    ]
    identity = {
        "schemaVersion": SCHEMA_VERSION,
        "sourceBindings": source_bindings,
        "selectedCaseIds": [case["caseId"] for case in selected],
    }
    batch_id = "fp-" + _sha256_bytes(_json_bytes(identity))[:16]
    batch_dir = output_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    opaque_cases: list[dict[str, Any]] = []
    used_audit_ids: set[str] = set()
    media_dir = batch_dir / "media"
    media_dir.mkdir(exist_ok=True)
    for case in selected:
        audit_case_id = "case-" + hashlib.sha256(
            f"{batch_id}:{case['media']['sha256']}".encode("ascii")
        ).hexdigest()[:16]
        if audit_case_id in used_audit_ids:
            raise FirstPrinciplesBatchError("opaque audit case ID collision")
        used_audit_ids.add(audit_case_id)
        source_media = Path(case["media"]["path"])
        suffix = source_media.suffix.lower()
        if not suffix or len(suffix) > 8 or not suffix[1:].isalnum():
            suffix = ".media"
        blind_media = media_dir / f"{audit_case_id}{suffix}"
        if blind_media.exists():
            if not blind_media.is_file():
                raise FirstPrinciplesBatchError(
                    f"blind media alias is not a file: {blind_media}"
                )
        else:
            try:
                os.link(source_media, blind_media)
            except OSError:
                shutil.copy2(source_media, blind_media)
        opaque_case = dict(case)
        opaque_case["auditCaseId"] = audit_case_id
        opaque_case["blindMediaPath"] = str(blind_media.resolve())
        opaque_cases.append(opaque_case)

    oracle = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "first-principles-sealed-reference",
        "batchId": batch_id,
        "warning": "Keep this file out of the processing context until predictions are frozen.",
        "sourceBindings": source_bindings,
        "coverage": _coverage(selected),
        "cases": opaque_cases,
    }
    oracle_bytes = _json_bytes(oracle)
    oracle_sha = _sha256_bytes(oracle_bytes)
    oracle_path = batch_dir / "sealed-reference.v1.json"
    oracle_path.write_bytes(oracle_bytes)

    blind_cases = [
        {
            "auditCaseId": case["auditCaseId"],
            "media": {
                "path": case["blindMediaPath"],
                "sha256": case["media"]["sha256"],
                "declaredBytes": case["media"]["declaredBytes"],
            },
            "durationSeconds": case["durationSeconds"],
        }
        for case in opaque_cases
    ]
    blind = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "first-principles-blind-media-batch",
        "batchId": batch_id,
        "selection": {
            "method": "greedy-inverse-frequency-feature-coverage-v1",
            "heldOutIncluded": include_held_out,
            "candidateCount": len(all_cases),
            "selectedCount": len(selected),
        },
        "sealedReference": {
            "relativePath": oracle_path.name,
            "sha256": oracle_sha,
        },
        "processingRules": [
            "Do not read the sealed reference before all predictions are frozen.",
            "Use raw media evidence; do not call the production MediaTranscribeStudio worker.",
            "Record every tool, model, parameter, retry, judgment, uncertainty, and failure.",
            "Do not invent speech, speaker, language, transcript, timing, or translation evidence.",
        ],
        "cases": blind_cases,
    }
    blind_path = batch_dir / "blind-media-batch.v1.json"
    blind_path.write_bytes(_json_bytes(blind))

    audit = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "first-principles-processing-ledger",
        "batchId": batch_id,
        "blindManifestSha256": _sha256_file(blind_path),
        "sealedReferenceSha256": oracle_sha,
        "truthAccessed": False,
        "processingMethod": {
            "assumptions": [],
            "globalToolInventory": [],
            "decisionProcedure": [],
        },
        "cases": [
            {
                "auditCaseId": case["auditCaseId"],
                "mediaSha256": case["media"]["sha256"],
                "status": "pending",
                "toolCalls": [],
                "evidence": [],
                "prediction": {
                    "lexicalSpeechPresent": None,
                    "speakerCount": None,
                    "documentLanguages": [],
                    "segments": [],
                    "translations": [],
                },
                "rationale": [],
                "uncertainties": [],
                "failureReasons": [],
                "wallSeconds": None,
            }
            for case in opaque_cases
        ],
    }
    audit_path = batch_dir / "processing-ledger.v1.json"
    audit_path.write_bytes(_json_bytes(audit))

    summary = {
        "batchId": batch_id,
        "batchDirectory": str(batch_dir.resolve()),
        "blindManifest": str(blind_path.resolve()),
        "sealedReference": str(oracle_path.resolve()),
        "processingLedger": str(audit_path.resolve()),
        "sealedReferenceSha256": oracle_sha,
        "candidateCoverage": _coverage(all_cases),
        "selectedCoverage": _coverage(selected),
    }
    (batch_dir / "batch-summary.v1.json").write_bytes(_json_bytes(summary))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum", type=int, default=20)
    parser.add_argument(
        "--include-held-out",
        action="store_true",
        help="include held-out cases; never use this during method development",
    )
    parser.add_argument(
        "--verify-media-hashes",
        action="store_true",
        help="read selected media and verify hashes (may download File Provider data)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = prepare_batch(
        sources=DEFAULT_SOURCES,
        output_root=args.output_root,
        maximum=args.maximum,
        include_held_out=args.include_held_out,
        verify_media_hashes=args.verify_media_hashes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
