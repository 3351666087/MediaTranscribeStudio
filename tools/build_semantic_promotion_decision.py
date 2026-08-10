"""Build a no-replace semantic-model promotion decision from sealed blind review.

This command only reads the live configuration, model registry, frozen
comparison, blind-package identity vault, and sealed Codex review.  It writes a
promotion decision artifact; it never edits the production configuration and it
never selects a winner from automatic benchmark metrics.

The human review ranking is deterministic and fail-closed.  Candidates are
ordered by (fewest blockers, fewest majors, fewest minors, most passes, most
equal-share weighted preference credit).  A non-unique result is an error, as
is a result that names the incumbent rather than a challenger.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.production_config import ProductionConfig  # noqa: E402
from tools.model_registry import (  # noqa: E402
    RegistryValidationError,
    load_registry,
    validate_registry,
)
from tools.promote_semantic_model import (  # noqa: E402
    ADAPTER_ID,
    DECISION_ARTIFACT_TYPE,
    DECISION_SCHEMA_VERSION,
    DEPLOYMENT_SLOT,
    REVIEW_KIND,
)


class SemanticPromotionDecisionBuildError(ValueError):
    """Raised when sealed promotion evidence cannot be bound safely."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
_SEVERITIES = ("blocker", "major", "minor", "pass")
_MAX_CANDIDATE_COUNT = 99
_REVIEW_CASE_KEYS = frozenset(
    {"assessmentGroups", "caseAlias", "preferredCandidateAliases", "tie"}
)
_GROUP_KEYS = frozenset(
    {"candidateAliases", "evidence", "reason", "recommendedAction", "severity"}
)


def _error(message: str) -> SemanticPromotionDecisionBuildError:
    return SemanticPromotionDecisionBuildError(message)


def _text(value: Any, *, field: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise _error(f"{field} must be trimmed non-empty text")
    return value


def _sha(value: Any, *, field: str) -> str:
    text = _text(value, field=field, maximum=80).casefold()
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    if _SHA256_RE.fullmatch(text) is None:
        raise _error(f"{field} must be a SHA-256 digest")
    return text


def _identifier(value: Any, *, field: str) -> str:
    text = _text(value, field=field, maximum=300)
    if _ID_RE.fullmatch(text) is None:
        raise _error(f"{field} has an invalid identifier")
    return text


def _candidate_count(value: Any, *, field: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 2
        or value > _MAX_CANDIDATE_COUNT
    ):
        raise _error(f"{field} must be between 2 and 99")
    return value


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be an object")
    return dict(value)


def _keys(value: Mapping[str, Any], *, required: frozenset[str], field: str) -> None:
    actual = frozenset(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        raise _error(f"{field} keys are invalid: missing={missing}, unknown={unknown}")


def _regular_file(path: Path, *, field: str) -> Path:
    if path.is_symlink():
        raise _error(f"{field} must be a regular file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _error(f"{field} must be a regular file") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise _error(f"{field} must be a regular file")
    return resolved


def _load(path: Path, *, field: str) -> tuple[Path, dict[str, Any], str]:
    resolved = _regular_file(path, field=field)
    try:
        value = read_json_strict(resolved)
    except Exception as exc:
        raise _error(f"{field} is not strict JSON") from exc
    return resolved, value, sha256_file(resolved)


def _declared_canonical(value: Mapping[str, Any], *, field: str) -> str:
    body = dict(value)
    declared = _sha(body.pop("canonicalSha256", None), field=f"{field}.canonicalSha256")
    if canonical_json_sha256(body) != declared:
        raise _error(f"{field} canonical SHA-256 does not match")
    return declared


def _registry_models(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    try:
        validate_registry(registry, verify_local=False)
    except (RegistryValidationError, OSError) as exc:
        raise _error(f"model registry is invalid: {exc}") from exc
    raw_models = registry.get("models")
    assert isinstance(raw_models, list)
    result: dict[str, Mapping[str, Any]] = {}
    for raw in raw_models:
        assert isinstance(raw, Mapping)
        model_id = _identifier(raw.get("id"), field="registry model id")
        if model_id in result:
            raise _error("model registry contains duplicate model IDs")
        result[model_id] = raw
    return result


def _registry_binding(
    model: Mapping[str, Any],
    *,
    field: str,
) -> tuple[str, str, str]:
    source = _object(model.get("source"), field=f"{field}.source")
    usage = _object(model.get("usage"), field=f"{field}.usage")
    runtime = _object(model.get("runtime"), field=f"{field}.runtime")
    local = _object(model.get("local"), field=f"{field}.local")
    if source.get("provider") != "ollama" or runtime.get("engine") != "ollama":
        raise _error(f"{field} is not an Ollama model")
    roles = usage.get("roles")
    if not isinstance(roles, list) or DEPLOYMENT_SLOT not in roles:
        raise _error(f"{field} is incompatible with {DEPLOYMENT_SLOT}")
    repository = _text(source.get("repository"), field=f"{field}.source.repository")
    tag = _text(source.get("tag"), field=f"{field}.source.tag")
    digest = _sha(source.get("digest"), field=f"{field}.source.digest")
    manifest = _object(local.get("manifest"), field=f"{field}.local.manifest")
    if _sha(manifest.get("sha256"), field=f"{field}.local.manifest.sha256") != digest:
        raise _error(f"{field} source and local manifest identities differ")
    model_name = f"{repository.removeprefix('library/')}:{tag}"
    return model_name, f"sha256:{digest}", digest


def _validate_blind_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_file_sha: str,
    vault_file_sha: str,
    vault_canonical_sha: str,
    comparison: Mapping[str, Any],
    review: Mapping[str, Any],
) -> int:
    required = frozenset(
        {
            "artifactType",
            "candidatesPerPacket",
            "canonicalSha256",
            "files",
            "manifestWrittenLast",
            "packetCount",
            "publicationPolicy",
            "schemaVersion",
        }
    )
    _keys(manifest, required=required, field="blind manifest")
    if manifest["artifactType"] != "anonymous-semantic-blind-review-manifest":
        raise _error("blind manifest artifactType is unsupported")
    if manifest["schemaVersion"] != "1.0.0":
        raise _error("blind manifest schemaVersion is unsupported")
    if manifest["publicationPolicy"] != "atomic-directory-no-replace" or manifest["manifestWrittenLast"] is not True:
        raise _error("blind manifest publication policy is invalid")
    candidate_count = _candidate_count(
        manifest["candidatesPerPacket"],
        field="blind manifest candidatesPerPacket",
    )
    if manifest["packetCount"] != 22:
        raise _error("blind manifest case/candidate counts are invalid")
    review_package = _object(review.get("blindPackage"), field="sealed review.blindPackage")
    if (
        review_package.get("packetCount") != 22
        or review_package.get("candidatesPerPacket") != candidate_count
    ):
        raise _error("sealed review case/candidate counts do not match the manifest")
    declared = _declared_canonical(manifest, field="blind manifest")
    if declared != _sha(review["blindPackage"]["manifestCanonicalSha256"], field="review blindPackage.manifestCanonicalSha256"):
        raise _error("sealed review is bound to another blind manifest canonical SHA")
    if manifest_file_sha != _sha(review["blindPackage"]["manifestFileSha256"], field="review blindPackage.manifestFileSha256"):
        raise _error("sealed review is bound to another blind manifest file SHA")
    evidence = _object(comparison.get("evidence"), field="comparison.evidence")
    if evidence.get("manifestCanonicalSha256") != declared or evidence.get("manifestFileSha256") != manifest_file_sha:
        raise _error("comparison is bound to another blind manifest")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise _error("blind manifest files must be an array")
    vault_rows = [row for row in files if isinstance(row, Mapping) and row.get("relativePath") == "identity-vault.json"]
    if len(vault_rows) != 1:
        raise _error("blind manifest must contain exactly one identity vault")
    vault_row = vault_rows[0]
    if vault_row.get("fileSha256") != vault_file_sha or vault_row.get("canonicalSha256") != vault_canonical_sha:
        raise _error("blind manifest identity-vault evidence does not match")
    return candidate_count


def _validate_vault(
    vault: Mapping[str, Any],
    *,
    candidate_count: int,
) -> dict[str, dict[str, Any]]:
    required = frozenset(
        {
            "artifactType",
            "canonicalSha256",
            "cases",
            "inputSetCanonicalSha256",
            "modelSetCanonicalSha256",
            "schemaVersion",
            "seedSha256",
        }
    )
    _keys(vault, required=required, field="identity vault")
    if vault["artifactType"] != "anonymous-semantic-blind-review-identity-vault" or vault["schemaVersion"] != "1.0.0":
        raise _error("identity vault contract is unsupported")
    _declared_canonical(vault, field="identity vault")
    cases = vault.get("cases")
    if not isinstance(cases, list) or len(cases) != 22:
        raise _error("identity vault case count is invalid")
    by_alias: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(cases):
        case = _object(raw, field=f"identity vault cases[{index}]")
        _keys(case, required=frozenset({"caseAlias", "caseId", "caseSha256", "candidates"}), field=f"identity vault cases[{index}]")
        alias = _identifier(case["caseAlias"], field=f"identity vault cases[{index}].caseAlias")
        if alias in by_alias:
            raise _error("identity vault contains duplicate case aliases")
        case_id = _text(case["caseId"], field=f"identity vault cases[{index}].caseId")
        case_sha = _sha(case["caseSha256"], field=f"identity vault cases[{index}].caseSha256")
        candidates = case["candidates"]
        if not isinstance(candidates, list) or len(candidates) != candidate_count:
            raise _error(f"identity vault {alias} candidate count is invalid")
        seen_aliases: set[str] = set()
        normalized_candidates: list[dict[str, Any]] = []
        for cindex, raw_candidate in enumerate(candidates):
            candidate = _object(raw_candidate, field=f"identity vault {alias}.candidates[{cindex}]")
            _keys(candidate, required=frozenset({"candidateAlias", "digest", "model", "modelId", "resultCanonicalSha256"}), field=f"identity vault {alias}.candidates[{cindex}")
            candidate_alias = _identifier(candidate["candidateAlias"], field=f"identity vault {alias}.candidateAlias")
            if candidate_alias in seen_aliases:
                raise _error(f"identity vault {alias} contains duplicate candidate aliases")
            seen_aliases.add(candidate_alias)
            normalized = dict(candidate)
            normalized["digest"] = f"sha256:{_sha(candidate['digest'], field=f'{alias}.{candidate_alias}.digest')}"
            normalized["model"] = _text(candidate["model"], field=f"{alias}.{candidate_alias}.model")
            normalized["modelId"] = _text(candidate["modelId"], field=f"{alias}.{candidate_alias}.modelId")
            normalized["resultCanonicalSha256"] = _sha(candidate["resultCanonicalSha256"], field=f"{alias}.{candidate_alias}.resultCanonicalSha256")
            normalized_candidates.append(normalized)
        by_alias[alias] = {"caseAlias": alias, "caseId": case_id, "caseSha256": case_sha, "candidates": normalized_candidates}
    # These commitments include the shared-input and model identities that are
    # deliberately absent from the public review packet.
    _sha(vault["inputSetCanonicalSha256"], field="identity vault.inputSetCanonicalSha256")
    _sha(vault["modelSetCanonicalSha256"], field="identity vault.modelSetCanonicalSha256")
    _sha(vault["seedSha256"], field="identity vault.seedSha256")
    return by_alias


def _validate_review(
    review: Mapping[str, Any],
    *,
    vault: Mapping[str, dict[str, Any]],
    candidate_count: int,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, int]]:
    required = frozenset(
        {
            "artifactType",
            "blindPackage",
            "bodySourceFileSha256",
            "canonicalSha256",
            "cases",
            "reviewId",
            "reviewPolicy",
            "reviewer",
            "schemaVersion",
            "validation",
        }
    )
    _keys(review, required=required, field="sealed review")
    if review["artifactType"] != "codex-semantic-blind-review" or review["schemaVersion"] != "1.0.0":
        raise _error("sealed review contract is unsupported")
    _declared_canonical(review, field="sealed review")
    policy = _object(review["reviewPolicy"], field="sealed review.reviewPolicy")
    if policy.get("automaticScoringUsed") is not False or policy.get("referenceTranscriptUsed") is not False:
        raise _error("sealed review is not an independent manual blind decision")
    validation = _object(review["validation"], field="sealed review.validation")
    if validation.get("allCandidatesAssessedExactlyOnce") is not True or validation.get("identityVaultRead") is not False or validation.get("sealedBeforeUnblind") is not True:
        raise _error("sealed review validation flags are unsafe")
    if (
        validation.get("reviewedCaseCount") != len(vault)
        or validation.get("assessedCandidateCount")
        != len(vault) * candidate_count
    ):
        raise _error("sealed review validation counts are inconsistent")
    reviewer = _object(review["reviewer"], field="sealed review.reviewer")
    reviewer_source = _text(reviewer.get("source"), field="sealed review.reviewer.source")
    if reviewer_source not in {"codex-agent", "human"}:
        raise _error("sealed review reviewer source is unsupported")
    review_id = _identifier(review["reviewId"], field="sealed review.reviewId")
    raw_cases = review["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != len(vault):
        raise _error("sealed review case count does not match identity vault")
    seen_cases: set[str] = set()
    severity: dict[str, Counter[str]] = {}
    preference_cases: Counter[str] = Counter()
    unique_preferences: Counter[str] = Counter()
    preferred_credit: dict[str, Fraction] = {}
    for index, raw in enumerate(raw_cases):
        case = _object(raw, field=f"sealed review.cases[{index}]")
        _keys(case, required=_REVIEW_CASE_KEYS, field=f"sealed review.cases[{index}]")
        alias = _identifier(case["caseAlias"], field=f"sealed review.cases[{index}].caseAlias")
        if alias in seen_cases or alias not in vault:
            raise _error("sealed review case alias is not bound to the identity vault")
        seen_cases.add(alias)
        vault_case = vault[alias]
        candidate_map = {c["candidateAlias"]: c for c in vault_case["candidates"]}
        preferred = case["preferredCandidateAliases"]
        if not isinstance(preferred, list) or not preferred:
            raise _error(f"sealed review {alias} has no preferred candidate")
        if len(set(preferred)) != len(preferred) or any(x not in candidate_map for x in preferred):
            raise _error(f"sealed review {alias} preferred aliases are invalid")
        if case["tie"] is not (len(preferred) > 1):
            raise _error(f"sealed review {alias} tie flag is inconsistent")
        assessed: list[str] = []
        for gindex, raw_group in enumerate(case["assessmentGroups"]):
            group = _object(raw_group, field=f"sealed review {alias}.assessmentGroups[{gindex}]")
            _keys(group, required=_GROUP_KEYS, field=f"sealed review {alias}.assessmentGroups[{gindex}]")
            aliases = group["candidateAliases"]
            if not isinstance(aliases, list) or not aliases or len(set(aliases)) != len(aliases) or any(x not in candidate_map for x in aliases):
                raise _error(f"sealed review {alias} assessment aliases are invalid")
            severity_name = group["severity"]
            if severity_name not in _SEVERITIES:
                raise _error(f"sealed review {alias} has invalid severity")
            _text(group["reason"], field=f"sealed review {alias}.reason", maximum=10000)
            _text(group["recommendedAction"], field=f"sealed review {alias}.recommendedAction", maximum=10000)
            evidence = _object(group["evidence"], field=f"sealed review {alias}.evidence")
            _keys(
                evidence,
                required=frozenset({"startMs", "endMs", "observation"}),
                field=f"sealed review {alias}.evidence",
            )
            start_ms = evidence["startMs"]
            end_ms = evidence["endMs"]
            if (
                not isinstance(start_ms, int)
                or isinstance(start_ms, bool)
                or not isinstance(end_ms, int)
                or isinstance(end_ms, bool)
                or start_ms < 0
                or end_ms <= start_ms
            ):
                raise _error(f"sealed review {alias} evidence time range is invalid")
            _text(
                evidence["observation"],
                field=f"sealed review {alias}.evidence.observation",
                maximum=10000,
            )
            for candidate_alias in aliases:
                assessed.append(candidate_alias)
                model = candidate_map[candidate_alias]["model"]
                severity.setdefault(model, Counter())[severity_name] += 1
        if Counter(assessed) != Counter(candidate_map.keys()):
            raise _error(f"sealed review {alias} does not assess every candidate exactly once")
        for candidate_alias in preferred:
            model = candidate_map[candidate_alias]["model"]
            preference_cases[model] += 1
            preferred_credit[model] = preferred_credit.get(model, Fraction()) + Fraction(
                1, len(preferred)
            )
        if len(preferred) == 1:
            unique_preferences[candidate_map[preferred[0]]["model"]] += 1
    if seen_cases != set(vault):
        raise _error("sealed review omitted one or more blind cases")
    # Do not use any benchmark score to select the winner. Severity and the
    # equal-share human preference credit are the only ranking inputs.
    models = sorted(severity)
    ranking = sorted(
        models,
        key=lambda model: (
            severity[model]["blocker"],
            severity[model]["major"],
            severity[model]["minor"],
            -severity[model]["pass"],
            -preferred_credit.get(model, Fraction()),
        ),
    )
    if len(ranking) < 2:
        raise _error("sealed review has fewer than two candidates")
    best_key = (
        severity[ranking[0]]["blocker"],
        severity[ranking[0]]["major"],
        severity[ranking[0]]["minor"],
        -severity[ranking[0]]["pass"],
        -preferred_credit.get(ranking[0], Fraction()),
    )
    tied = [
        model
        for model in ranking
        if (
            severity[model]["blocker"],
            severity[model]["major"],
            severity[model]["minor"],
            -severity[model]["pass"],
            -preferred_credit.get(model, Fraction()),
        )
        == best_key
    ]
    if len(tied) != 1:
        raise _error(f"sealed review winner is not unique: {tied}")
    summary = {
        model: {
            **{severity_name: severity[model][severity_name] for severity_name in _SEVERITIES},
            "preferredCaseCount": preference_cases[model],
            "uniquePreferredCaseCount": unique_preferences[model],
            "preferredCaseCredit": {
                "numerator": preferred_credit.get(model, Fraction()).numerator,
                "denominator": preferred_credit.get(model, Fraction()).denominator,
            },
        }
        for model in models
    }
    return ranking[0], summary, {"reviewedCaseCount": len(raw_cases), "candidateCount": len(models)}


def _comparison_bindings(
    comparison: Mapping[str, Any],
    *,
    registry_models: Mapping[str, Mapping[str, Any]],
    expected_canonical_sha256: str,
    vault_document: Mapping[str, Any],
    vault: Mapping[str, dict[str, Any]],
    vault_file_sha: str,
    vault_canonical_sha: str,
    manifest_file_sha: str,
    manifest_canonical_sha: str,
    review: Mapping[str, Any],
    review_file_sha: str,
    review_canonical_sha: str,
    review_winner_model: str,
    review_summary: Mapping[str, Mapping[str, Any]],
    candidate_count: int,
) -> tuple[str, str, dict[str, tuple[str, str, str]], str, str]:
    required = frozenset(
        {
            "aggregate",
            "artifactType",
            "automaticScoringUsed",
            "blindBatchId",
            "candidateCount",
            "candidateIdentitiesHiddenDuringReview",
            "canonicalSha256",
            "caseCount",
            "caseSetSha256",
            "comparisonId",
            "evidence",
            "modelSetSha256",
            "models",
            "outcome",
            "rankingPolicy",
            "reviewId",
            "schemaVersion",
            "sealedBeforeUnblind",
            "tiedRegistryModelIds",
            "uniqueWinnerRegistryModelId",
        }
    )
    _keys(comparison, required=required, field="comparison")
    if (
        comparison["artifactType"] != "semantic-model-blind-review-comparison"
        or comparison["schemaVersion"] != "1.0.0"
    ):
        raise _error("comparison artifact contract is unsupported")
    comparison_canonical_sha = _declared_canonical(comparison, field="comparison")
    if comparison_canonical_sha != _sha(
        expected_canonical_sha256,
        field="expected comparison canonical SHA-256",
    ):
        raise _error("comparison canonical SHA-256 is not the authorized digest")
    if (
        comparison["automaticScoringUsed"] is not False
        or comparison["candidateIdentitiesHiddenDuringReview"] is not True
        or comparison["sealedBeforeUnblind"] is not True
        or comparison["outcome"] != "unique-winner"
    ):
        raise _error("comparison does not preserve the sealed human-only decision")
    if (
        comparison["candidateCount"] != candidate_count
        or comparison["caseCount"] != len(vault)
    ):
        raise _error("comparison case/candidate counts do not match the vault")
    review_id = _identifier(review["reviewId"], field="sealed review.reviewId")
    if comparison["reviewId"] != review_id:
        raise _error("comparison is bound to another sealed review ID")
    blind_batch_id = _identifier(
        comparison["blindBatchId"],
        field="comparison.blindBatchId",
    )
    case_set_sha = _sha(comparison["caseSetSha256"], field="comparison.caseSetSha256")
    model_set_sha = _sha(comparison["modelSetSha256"], field="comparison.modelSetSha256")
    if case_set_sha != _sha(
        vault_document["inputSetCanonicalSha256"],
        field="identity vault.inputSetCanonicalSha256",
    ):
        raise _error("comparison case-set SHA does not match the identity vault")
    if model_set_sha != _sha(
        vault_document["modelSetCanonicalSha256"],
        field="identity vault.modelSetCanonicalSha256",
    ):
        raise _error("comparison model-set SHA does not match the identity vault")

    evidence = _object(comparison["evidence"], field="comparison.evidence")
    _keys(
        evidence,
        required=frozenset(
            {
                "identityVaultCanonicalSha256",
                "identityVaultFileSha256",
                "manifestCanonicalSha256",
                "manifestFileSha256",
                "reviewerPacketSetSha256",
                "sealedReviewCanonicalSha256",
                "sealedReviewFileSha256",
                "sealedReviewPath",
            }
        ),
        field="comparison.evidence",
    )
    expected_evidence = {
        "identityVaultCanonicalSha256": vault_canonical_sha,
        "identityVaultFileSha256": vault_file_sha,
        "manifestCanonicalSha256": manifest_canonical_sha,
        "manifestFileSha256": manifest_file_sha,
        "reviewerPacketSetSha256": _sha(
            review["blindPackage"]["reviewerPacketSetSha256"],
            field="sealed review blindPackage.reviewerPacketSetSha256",
        ),
        "sealedReviewCanonicalSha256": review_canonical_sha,
        "sealedReviewFileSha256": review_file_sha,
    }
    for field, expected in expected_evidence.items():
        if _sha(evidence.get(field), field=f"comparison.evidence.{field}") != expected:
            raise _error(f"comparison evidence {field} does not match")
    _text(evidence["sealedReviewPath"], field="comparison.evidence.sealedReviewPath")

    ranking_policy = _object(comparison["rankingPolicy"], field="comparison.rankingPolicy")
    _keys(
        ranking_policy,
        required=frozenset({"automaticMetricsUsed", "order", "tiePreferenceWeight"}),
        field="comparison.rankingPolicy",
    )
    if ranking_policy["automaticMetricsUsed"] is not False or ranking_policy["order"] != [
        "blocker-count-ascending",
        "major-count-ascending",
        "minor-count-ascending",
        "pass-count-descending",
        "weighted-preferred-share-descending",
    ]:
        raise _error("comparison ranking policy is not the sealed human-only policy")
    if ranking_policy["tiePreferenceWeight"] != "one case credit divided equally among preferred candidates":
        raise _error("comparison tie preference policy is unsupported")

    raw_models = comparison["models"]
    if not isinstance(raw_models, list) or len(raw_models) != candidate_count:
        raise _error("comparison models are invalid")
    model_bindings: dict[str, tuple[str, str, str]] = {}
    for index, raw_model in enumerate(raw_models):
        model = _object(raw_model, field=f"comparison.models[{index}]")
        _keys(
            model,
            required=frozenset({"digest", "model", "registryModelId"}),
            field=f"comparison.models[{index}]",
        )
        model_name = _text(model["model"], field=f"comparison.models[{index}].model")
        digest = f"sha256:{_sha(model['digest'], field=f'comparison.models[{index}].digest')}"
        registry_model_id = _normalize_registry_binding(
            model["registryModelId"],
            model_name=model_name,
            digest=digest,
            registry_models=registry_models,
            field=f"comparison.models[{index}].registryModelId",
        )
        if model_name in model_bindings:
            raise _error("comparison contains duplicate model identities")
        model_bindings[model_name] = (model_name, digest, registry_model_id)

    vault_model_sets = {
        frozenset(
            (
                candidate["model"],
                candidate["digest"],
                _normalize_registry_binding(
                    candidate["modelId"],
                    model_name=candidate["model"],
                    digest=candidate["digest"],
                    registry_models=registry_models,
                    field="identity vault candidate.modelId",
                ),
            )
            for candidate in case["candidates"]
        )
        for case in vault.values()
    }
    if len(vault_model_sets) != 1:
        raise _error("identity vault candidate model sets differ by case")
    comparison_model_set = frozenset(model_bindings.values())
    if comparison_model_set != next(iter(vault_model_sets)):
        raise _error("comparison and identity-vault model identities differ")

    raw_aggregate = comparison["aggregate"]
    if not isinstance(raw_aggregate, list) or len(raw_aggregate) != candidate_count:
        raise _error("comparison aggregate is invalid")
    aggregate_by_model: dict[str, Mapping[str, Any]] = {}
    ranks: set[int] = set()
    for index, raw_row in enumerate(raw_aggregate):
        row = _object(raw_row, field=f"comparison.aggregate[{index}]")
        _keys(
            row,
            required=frozenset(
                {
                    "caseCount",
                    "digest",
                    "model",
                    "preferredCaseCredit",
                    "rank",
                    "registryModelId",
                    "severityCounts",
                    "weightedPreferredShare",
                }
            ),
            field=f"comparison.aggregate[{index}]",
        )
        model_name = _text(row["model"], field=f"comparison.aggregate[{index}].model")
        if model_name in aggregate_by_model or model_name not in model_bindings:
            raise _error("comparison aggregate model identity is invalid")
        aggregate_by_model[model_name] = row
        rank = row["rank"]
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 1
            or rank > candidate_count
        ):
            raise _error("comparison aggregate rank is invalid")
        ranks.add(rank)
        if row["caseCount"] != len(vault):
            raise _error("comparison aggregate case count is invalid")
        _, expected_digest, expected_registry_id = model_bindings[model_name]
        row_digest = f"sha256:{_sha(row['digest'], field=f'comparison.aggregate[{index}].digest')}"
        row_registry_id = _normalize_registry_binding(
            row["registryModelId"],
            model_name=model_name,
            digest=row_digest,
            registry_models=registry_models,
            field=f"comparison.aggregate[{index}].registryModelId",
        )
        if row_digest != expected_digest or row_registry_id != expected_registry_id:
            raise _error("comparison aggregate model binding does not match")
        expected_summary = review_summary.get(model_name)
        if expected_summary is None:
            raise _error("comparison aggregate model is absent from sealed review")
        counts = _object(row["severityCounts"], field=f"comparison aggregate {model_name}.severityCounts")
        _keys(counts, required=frozenset(_SEVERITIES), field=f"comparison aggregate {model_name}.severityCounts")
        if any(counts[name] != expected_summary[name] for name in _SEVERITIES):
            raise _error("comparison aggregate severity counts differ from sealed review")
        credit = _object(row["preferredCaseCredit"], field=f"comparison aggregate {model_name}.preferredCaseCredit")
        _keys(credit, required=frozenset({"numerator", "denominator"}), field=f"comparison aggregate {model_name}.preferredCaseCredit")
        expected_credit = expected_summary["preferredCaseCredit"]
        if credit != expected_credit:
            raise _error("comparison preferred-case credit differs from sealed review")
        expected_share = credit["numerator"] / credit["denominator"] / len(vault)
        actual_share = row["weightedPreferredShare"]
        if (
            isinstance(actual_share, bool)
            or not isinstance(actual_share, (int, float))
            or not math.isclose(float(actual_share), expected_share, rel_tol=0.0, abs_tol=5e-13)
        ):
            raise _error("comparison weighted preferred share differs from sealed review")
    if set(aggregate_by_model) != set(model_bindings):
        raise _error("comparison aggregate rank coverage is invalid")
    expected_ranks: dict[str, int] = {}
    previous_key: tuple[Any, ...] | None = None
    expected_rank = 0
    ordered_models = sorted(
        model_bindings,
        key=lambda model_name: (
            review_summary[model_name]["blocker"],
            review_summary[model_name]["major"],
            review_summary[model_name]["minor"],
            -review_summary[model_name]["pass"],
            -Fraction(
                review_summary[model_name]["preferredCaseCredit"]["numerator"],
                review_summary[model_name]["preferredCaseCredit"]["denominator"],
            ),
            model_name,
        ),
    )
    for position, model_name in enumerate(ordered_models, start=1):
        summary = review_summary[model_name]
        credit = summary["preferredCaseCredit"]
        key = (
            summary["blocker"],
            summary["major"],
            summary["minor"],
            -summary["pass"],
            -Fraction(credit["numerator"], credit["denominator"]),
        )
        if key != previous_key:
            expected_rank = position
            previous_key = key
        expected_ranks[model_name] = expected_rank
    if any(
        aggregate_by_model[model_name]["rank"] != expected_rank
        for model_name, expected_rank in expected_ranks.items()
    ):
        raise _error("comparison aggregate ranks differ from the sealed review")

    raw_winner_id = _text(
        comparison["uniqueWinnerRegistryModelId"],
        field="comparison.uniqueWinnerRegistryModelId",
        maximum=300,
    )
    tied = comparison["tiedRegistryModelIds"]
    if tied != [raw_winner_id]:
        raise _error("comparison unique-winner set is inconsistent")
    winner_matches = [
        (model_name, binding)
        for model_name, binding in model_bindings.items()
        if raw_winner_id in {model_name, binding[2]}
    ]
    if len(winner_matches) != 1 or winner_matches[0][0] != review_winner_model:
        raise _error("comparison winner differs from the sealed human review")
    winner_binding = winner_matches[0][1]
    winner_registry_id = winner_binding[2]
    if aggregate_by_model[review_winner_model]["rank"] != 1:
        raise _error("comparison winner is not ranked first")
    return (
        case_set_sha,
        blind_batch_id,
        model_bindings,
        winner_registry_id,
        comparison_canonical_sha,
    )


def _match_registry_model(
    registry_models: Mapping[str, Mapping[str, Any]],
    *,
    model_name: str,
    digest: str,
    field: str,
) -> str:
    matches = []
    for model_id, raw in registry_models.items():
        source = raw.get("source")
        usage = raw.get("usage")
        runtime = raw.get("runtime")
        if (
            not isinstance(source, Mapping)
            or source.get("provider") != "ollama"
            or not isinstance(runtime, Mapping)
            or runtime.get("engine") != "ollama"
            or not isinstance(usage, Mapping)
            or not isinstance(usage.get("roles"), list)
            or DEPLOYMENT_SLOT not in usage["roles"]
        ):
            continue
        bound_name, bound_digest, _ = _registry_binding(raw, field=f"registry.models[{model_id}]")
        if bound_name == model_name and bound_digest == digest:
            matches.append(model_id)
    if len(matches) != 1:
        raise _error(f"{field} does not map to exactly one registry model: {matches}")
    return matches[0]


def _normalize_registry_binding(
    value: Any,
    *,
    model_name: str,
    digest: str,
    registry_models: Mapping[str, Mapping[str, Any]],
    field: str,
) -> str:
    """Return the canonical registry ID for one exact model/digest binding.

    Early benchmark packages used the Ollama runtime reference as ``modelId``.
    Accept that legacy spelling only when it is exactly the bound model name
    and the model name plus digest resolves to one compatible registry entry.
    Arbitrary aliases and ambiguous registry matches remain invalid.
    """

    raw = _text(value, field=field, maximum=300)
    registry_id = _match_registry_model(
        registry_models,
        model_name=model_name,
        digest=digest,
        field=field,
    )
    if raw not in {registry_id, model_name}:
        raise _error(
            f"{field} must be the canonical registry ID or exact model reference"
        )
    return registry_id


def build_semantic_promotion_decision(
    *,
    config_path: Path,
    registry_path: Path,
    comparison_path: Path,
    blind_manifest_path: Path,
    identity_vault_path: Path,
    sealed_review_path: Path,
    output_path: Path,
    expected_sealed_review_sha256: str,
    expected_comparison_canonical_sha256: str,
    decision_id: str | None = None,
) -> dict[str, Any]:
    config_file = _regular_file(config_path, field="production config")
    config_sha = sha256_file(config_file)
    try:
        config = ProductionConfig.load(config_file)
    except Exception as exc:
        raise _error(f"production config is invalid: {exc}") from exc
    registry_file, registry, _ = _load(registry_path, field="model registry")
    del registry_file
    registry_models = _registry_models(registry)
    _, comparison, comparison_file_sha = _load(
        comparison_path,
        field="comparison artifact",
    )
    _, vault, vault_file_sha = _load(identity_vault_path, field="identity vault")
    _, manifest, manifest_file_sha = _load(blind_manifest_path, field="blind manifest")
    _, review, review_file_sha = _load(sealed_review_path, field="sealed review")
    expected_review_sha = _sha(expected_sealed_review_sha256, field="expected sealed review SHA-256")
    if review_file_sha != expected_review_sha:
        raise _error("sealed review file SHA-256 does not match the authorized digest")
    vault_canonical_sha = _declared_canonical(vault, field="identity vault")
    manifest_canonical_sha = _declared_canonical(manifest, field="blind manifest")
    review_canonical_sha = _declared_canonical(review, field="sealed review")
    candidate_count = _validate_blind_manifest(
        manifest,
        manifest_file_sha=manifest_file_sha,
        vault_file_sha=vault_file_sha,
        vault_canonical_sha=vault_canonical_sha,
        comparison=comparison,
        review=review,
    )
    vault_by_alias = _validate_vault(
        vault,
        candidate_count=candidate_count,
    )
    winner_model, review_summary, review_counts = _validate_review(
        review=review,
        vault=vault_by_alias,
        candidate_count=candidate_count,
    )
    (
        case_set_sha,
        blind_batch_id,
        comparison_models,
        comparison_winner_registry_id,
        comparison_canonical_sha,
    ) = _comparison_bindings(
        comparison,
        registry_models=registry_models,
        expected_canonical_sha256=expected_comparison_canonical_sha256,
        vault_document=vault,
        vault=vault_by_alias,
        vault_file_sha=vault_file_sha,
        vault_canonical_sha=vault_canonical_sha,
        manifest_file_sha=manifest_file_sha,
        manifest_canonical_sha=manifest_canonical_sha,
        review=review,
        review_file_sha=review_file_sha,
        review_canonical_sha=review_canonical_sha,
        review_winner_model=winner_model,
        review_summary=review_summary,
        candidate_count=candidate_count,
    )
    # Bind every unblinded candidate to the exact comparison digest and registry
    # identity before allowing the selected candidate into a decision.
    registry_by_model: dict[str, str] = {}
    for model_name, (_, digest, declared_registry_id) in comparison_models.items():
        actual_registry_id = _match_registry_model(
            registry_models,
            model_name=model_name,
            digest=digest,
            field=f"comparison model {model_name}",
        )
        if actual_registry_id != declared_registry_id:
            raise _error(f"comparison model {model_name} registry ID does not match")
        registry_by_model[model_name] = actual_registry_id
    winner_registry_id = registry_by_model.get(winner_model)
    if winner_registry_id is None:
        raise _error("sealed review winner is absent from comparison models")
    if winner_registry_id != comparison_winner_registry_id:
        raise _error("comparison and sealed review winners differ")
    incumbent_model = config.speaker.local_llm_model
    incumbent_digest = config.speaker.local_llm_model_digest
    incumbent_registry_id = _match_registry_model(
        registry_models,
        model_name=incumbent_model,
        digest=incumbent_digest,
        field="active production model",
    )
    if winner_registry_id == incumbent_registry_id:
        raise _error("sealed review winner is already the active production model")
    if winner_model == incumbent_model and winner_registry_id != incumbent_registry_id:
        raise _error("active model identity is inconsistent with registry")
    review_id = _identifier(review["reviewId"], field="sealed review.reviewId")
    final_decision_id = _identifier(
        decision_id or f"{review_id}.promote.{winner_registry_id}",
        field="decisionId",
    )
    decision: dict[str, Any] = {
        "schemaVersion": DECISION_SCHEMA_VERSION,
        "artifactType": DECISION_ARTIFACT_TYPE,
        "decisionId": final_decision_id,
        "deploymentSlot": DEPLOYMENT_SLOT,
        "adapterId": ADAPTER_ID,
        "automaticScoring": False,
        "expectedConfigSha256": config_sha,
        "incumbentRegistryModelId": incumbent_registry_id,
        "challengerRegistryModelId": winner_registry_id,
        "outcome": "promote-challenger",
        "blindReview": {
            "kind": REVIEW_KIND,
            "blindBatchId": blind_batch_id,
            "caseSetSha256": case_set_sha,
            "sameCaseBatch": True,
            "candidateIdentitiesHidden": True,
            "winnerRegistryModelId": winner_registry_id,
        },
        "evidence": {
            "blindReviewArtifactSha256": review_file_sha,
            "comparisonArtifactSha256": comparison_file_sha,
        },
        "audit": {
            "source": _text(review["reviewer"].get("source"), field="sealed review reviewer source"),
            "reviewer": _text(review["reviewer"].get("actor"), field="sealed review reviewer actor"),
            "decidedAt": _text(review["reviewer"].get("reviewedAt"), field="sealed review reviewer reviewedAt"),
        },
    }
    decision["canonicalSha256"] = canonical_json_sha256(decision)
    # Validate the exact contract used by the mutating CAS tool without calling
    # that tool or changing the active configuration.
    from tools.promote_semantic_model import _validate_decision  # noqa: PLC0415

    _validate_decision(decision, file_sha256=canonical_json_sha256(decision))
    atomic_write_json_no_replace(output_path, decision)
    return {
        "decision": decision,
        "decisionFileSha256": sha256_file(output_path.resolve(strict=True)),
        "productionConfigSha256": config_sha,
        "comparisonArtifactSha256": comparison_file_sha,
        "comparisonCanonicalSha256": comparison_canonical_sha,
        "sealedReviewFileSha256": review_file_sha,
        "identityVaultFileSha256": vault_file_sha,
        "blindManifestFileSha256": manifest_file_sha,
        "winnerModel": winner_model,
        "winnerRegistryModelId": winner_registry_id,
        "incumbentModel": incumbent_model,
        "incumbentRegistryModelId": incumbent_registry_id,
        "humanReviewSummary": review_summary,
        "humanReviewCounts": review_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--identity-vault", type=Path, required=True)
    parser.add_argument("--sealed-review", type=Path, required=True)
    parser.add_argument("--expected-sealed-review-sha256", required=True)
    parser.add_argument("--expected-comparison-canonical-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = build_semantic_promotion_decision(
            config_path=arguments.config,
            registry_path=arguments.registry,
            comparison_path=arguments.comparison,
            blind_manifest_path=arguments.blind_manifest,
            identity_vault_path=arguments.identity_vault,
            sealed_review_path=arguments.sealed_review,
            output_path=arguments.output,
            expected_sealed_review_sha256=arguments.expected_sealed_review_sha256,
            expected_comparison_canonical_sha256=(
                arguments.expected_comparison_canonical_sha256
            ),
            decision_id=arguments.decision_id,
        )
    except (OSError, SemanticPromotionDecisionBuildError) as exc:
        print(f"semantic promotion decision build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
