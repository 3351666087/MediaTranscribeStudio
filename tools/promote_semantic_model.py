"""Atomically promote a blind-reviewed local semantic model.

The active semantic model is the ``speaker.localLlmModel`` plus
``speaker.localLlmModelDigest`` binding in the production configuration.
Registry status is inventory history only: any compatible model can replace
the incumbent immediately after winning a same-batch blind product review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402
from backend.production_config import ProductionConfig  # noqa: E402
from tools.model_registry import (  # noqa: E402
    RegistryValidationError,
    load_registry,
    validate_registry,
)


DECISION_SCHEMA_VERSION = "1.0.0"
DECISION_ARTIFACT_TYPE = "semantic-model-blind-promotion-decision"
RECEIPT_ARTIFACT_TYPE = "semantic-model-production-promotion-receipt"
DEPLOYMENT_SLOT = "semantic-arbitration"
ADAPTER_ID = "ollama-loopback"
REVIEW_KIND = "same-batch-blind-product-review"
_MAX_JSON_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
_AUDIT_SOURCES = frozenset({"human", "codex-agent"})


class SemanticModelPromotionError(ValueError):
    """Raised when semantic promotion evidence or its CAS is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticModelPromotionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise SemanticModelPromotionError(f"non-finite JSON number: {value}")


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise SemanticModelPromotionError(f"{label} must be a regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SemanticModelPromotionError(f"{label} must be a regular file")
    raw = resolved.read_bytes()
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise SemanticModelPromotionError(f"{label} has an invalid size")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticModelPromotionError(
            f"{label} must be valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SemanticModelPromotionError(f"{label} must be an object")
    return value, raw


def _object(
    value: Any,
    *,
    field: str,
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticModelPromotionError(f"{field} must be an object")
    output = {str(key): item for key, item in value.items()}
    missing = sorted(required - set(output))
    unknown = sorted(set(output) - required)
    if missing or unknown:
        raise SemanticModelPromotionError(
            f"{field} keys are invalid: missing={missing}, unknown={unknown}"
        )
    return output


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SemanticModelPromotionError(
            f"{field} must be trimmed non-empty text"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SemanticModelPromotionError(f"{field} contains control characters")
    return value


def _identifier(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if _ID_RE.fullmatch(text) is None:
        raise SemanticModelPromotionError(f"{field} has an invalid identifier")
    return text


def _sha256(value: Any, *, field: str) -> str:
    text = _text(value, field=field).casefold().removeprefix("sha256:")
    if _SHA256_RE.fullmatch(text) is None:
        raise SemanticModelPromotionError(f"{field} must be a SHA-256 digest")
    return text


def _validate_decision(
    value: Mapping[str, Any],
    *,
    file_sha256: str,
) -> dict[str, str]:
    decision = _object(
        value,
        field="decision",
        required=frozenset(
            {
                "schemaVersion",
                "artifactType",
                "decisionId",
                "deploymentSlot",
                "adapterId",
                "automaticScoring",
                "expectedConfigSha256",
                "incumbentRegistryModelId",
                "challengerRegistryModelId",
                "outcome",
                "blindReview",
                "evidence",
                "audit",
                "canonicalSha256",
            }
        ),
    )
    if decision["schemaVersion"] != DECISION_SCHEMA_VERSION:
        raise SemanticModelPromotionError("decision schemaVersion is unsupported")
    if decision["artifactType"] != DECISION_ARTIFACT_TYPE:
        raise SemanticModelPromotionError("decision artifactType is unsupported")
    if decision["deploymentSlot"] != DEPLOYMENT_SLOT:
        raise SemanticModelPromotionError("decision deploymentSlot is unsupported")
    if decision["adapterId"] != ADAPTER_ID:
        raise SemanticModelPromotionError("decision adapterId is unsupported")
    if decision["automaticScoring"] is not False:
        raise SemanticModelPromotionError(
            "promotion requires an explicit blind semantic review decision"
        )
    if decision["outcome"] != "promote-challenger":
        raise SemanticModelPromotionError("decision does not promote challenger")

    canonical = dict(decision)
    declared_canonical = _sha256(
        canonical.pop("canonicalSha256"),
        field="decision.canonicalSha256",
    )
    if canonical_json_sha256(canonical) != declared_canonical:
        raise SemanticModelPromotionError(
            "decision canonical SHA-256 does not match"
        )

    incumbent = _identifier(
        decision["incumbentRegistryModelId"],
        field="decision.incumbentRegistryModelId",
    )
    challenger = _identifier(
        decision["challengerRegistryModelId"],
        field="decision.challengerRegistryModelId",
    )
    if incumbent == challenger:
        raise SemanticModelPromotionError(
            "decision challenger must differ from incumbent"
        )

    blind_review = _object(
        decision["blindReview"],
        field="decision.blindReview",
        required=frozenset(
            {
                "kind",
                "blindBatchId",
                "caseSetSha256",
                "sameCaseBatch",
                "candidateIdentitiesHidden",
                "winnerRegistryModelId",
            }
        ),
    )
    if blind_review["kind"] != REVIEW_KIND:
        raise SemanticModelPromotionError(
            "decision did not use the required blind product review"
        )
    if blind_review["sameCaseBatch"] is not True:
        raise SemanticModelPromotionError(
            "incumbent and challenger must use the same review case batch"
        )
    if blind_review["candidateIdentitiesHidden"] is not True:
        raise SemanticModelPromotionError(
            "candidate identities must remain hidden during review"
        )
    if blind_review["winnerRegistryModelId"] != challenger:
        raise SemanticModelPromotionError(
            "blind review winner does not match the challenger"
        )

    evidence = _object(
        decision["evidence"],
        field="decision.evidence",
        required=frozenset(
            {"blindReviewArtifactSha256", "comparisonArtifactSha256"}
        ),
    )
    audit = _object(
        decision["audit"],
        field="decision.audit",
        required=frozenset({"source", "reviewer", "decidedAt"}),
    )
    source = _identifier(audit["source"], field="decision.audit.source")
    if source not in _AUDIT_SOURCES:
        raise SemanticModelPromotionError(
            "decision.audit.source must be human or codex-agent"
        )
    _text(audit["reviewer"], field="decision.audit.reviewer")
    _text(audit["decidedAt"], field="decision.audit.decidedAt")
    return {
        "decisionId": _identifier(
            decision["decisionId"], field="decision.decisionId"
        ),
        "decisionArtifactSha256": file_sha256,
        "decisionCanonicalSha256": declared_canonical,
        "expectedConfigSha256": _sha256(
            decision["expectedConfigSha256"],
            field="decision.expectedConfigSha256",
        ),
        "incumbentRegistryModelId": incumbent,
        "challengerRegistryModelId": challenger,
        "blindBatchId": _identifier(
            blind_review["blindBatchId"],
            field="decision.blindReview.blindBatchId",
        ),
        "caseSetSha256": _sha256(
            blind_review["caseSetSha256"],
            field="decision.blindReview.caseSetSha256",
        ),
        "blindReviewArtifactSha256": _sha256(
            evidence["blindReviewArtifactSha256"],
            field="decision.evidence.blindReviewArtifactSha256",
        ),
        "comparisonArtifactSha256": _sha256(
            evidence["comparisonArtifactSha256"],
            field="decision.evidence.comparisonArtifactSha256",
        ),
        "reviewSource": source,
        "reviewer": str(audit["reviewer"]),
        "decidedAt": str(audit["decidedAt"]),
    }


def _registry_model(
    registry: Mapping[str, Any],
    *,
    registry_model_id: str,
) -> Mapping[str, Any]:
    models = registry.get("models")
    if not isinstance(models, list):
        raise SemanticModelPromotionError("registry models are invalid")
    selected = next(
        (
            model
            for model in models
            if isinstance(model, Mapping)
            and model.get("id") == registry_model_id
        ),
        None,
    )
    if selected is None:
        raise SemanticModelPromotionError(
            f"registry model is missing: {registry_model_id}"
        )
    return selected


def _registry_binding(
    registry: Mapping[str, Any],
    *,
    registry_model_id: str,
) -> dict[str, str]:
    selected = _registry_model(
        registry,
        registry_model_id=registry_model_id,
    )
    source = selected.get("source")
    usage = selected.get("usage")
    runtime = selected.get("runtime")
    local = selected.get("local")
    if not all(
        isinstance(item, Mapping)
        for item in (source, usage, runtime, local)
    ):
        raise SemanticModelPromotionError("registry model binding is invalid")
    assert isinstance(source, Mapping)
    assert isinstance(usage, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(local, Mapping)
    roles = usage.get("roles")
    if (
        source.get("provider") != "ollama"
        or runtime.get("engine") != "ollama"
        or not isinstance(roles, list)
        or DEPLOYMENT_SLOT not in roles
    ):
        raise SemanticModelPromotionError(
            "registry model is incompatible with semantic arbitration"
        )
    repository = _text(
        source.get("repository"), field="registry.model.source.repository"
    )
    tag = _text(source.get("tag"), field="registry.model.source.tag")
    digest = _sha256(
        source.get("digest"), field="registry.model.source.digest"
    )
    if not repository.startswith("library/"):
        raise SemanticModelPromotionError(
            "registry semantic model has an invalid Ollama repository"
        )
    manifest = local.get("manifest")
    if not isinstance(manifest, Mapping):
        raise SemanticModelPromotionError(
            "registry semantic model manifest identity is invalid"
        )
    manifest_sha256 = _sha256(
        manifest.get("sha256"),
        field="registry.model.local.manifest.sha256",
    )
    if manifest_sha256 != digest:
        raise SemanticModelPromotionError(
            "registry semantic model source and manifest identities differ"
        )
    return {
        "registryModelId": registry_model_id,
        "model": f"{repository.removeprefix('library/')}:{tag}",
        "digest": f"sha256:{digest}",
        "manifestSha256": manifest_sha256,
    }


def _serialize_config(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_evidence_artifact(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    if path.is_symlink():
        raise SemanticModelPromotionError(f"{label} must be a regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SemanticModelPromotionError(f"{label} must be a regular file")
    if sha256_file(resolved) != expected_sha256:
        raise SemanticModelPromotionError(f"{label} SHA-256 does not match")
    return resolved


def _publish_rollback_snapshot(
    *,
    rollback_root: Path,
    decision_id: str,
    config_sha256: str,
    config_bytes: bytes,
) -> Path:
    if rollback_root.exists() and rollback_root.is_symlink():
        raise SemanticModelPromotionError(
            "rollback root must not be a symbolic link"
        )
    rollback_root.mkdir(parents=True, exist_ok=True)
    resolved_root = rollback_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise SemanticModelPromotionError("rollback root must be a directory")
    snapshot = resolved_root / (
        f"{decision_id}.{config_sha256}.production-config.rollback.json"
    )
    if snapshot.exists():
        if snapshot.is_symlink() or sha256_file(snapshot) != config_sha256:
            raise SemanticModelPromotionError(
                "existing rollback snapshot differs from the incumbent config"
            )
        return snapshot
    temporary = snapshot.with_name(f".{snapshot.name}.{uuid.uuid4().hex}.tmp")
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(config_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, snapshot)
        published = True
        _fsync_directory(resolved_root)
    except FileExistsError:
        if snapshot.is_symlink() or sha256_file(snapshot) != config_sha256:
            raise SemanticModelPromotionError(
                "concurrent rollback snapshot differs from the incumbent config"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    if not published and not snapshot.is_file():
        raise SemanticModelPromotionError("rollback snapshot was not published")
    return snapshot


@contextmanager
def _promotion_lock(path: Path) -> Iterator[None]:
    """Share the production-config promotion lock with every model slot."""

    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise SemanticModelPromotionError(
                    "another production model promotion is in progress"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise SemanticModelPromotionError(
                    "another production model promotion is in progress"
                ) from exc
        locked = True
        metadata = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "acquiredUnixSeconds": time.time(),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, metadata)
        os.ftruncate(descriptor, len(metadata))
        os.fsync(descriptor)
        yield
    finally:
        if locked:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def promote_semantic_model(
    *,
    config_path: Path,
    registry_path: Path,
    decision_path: Path,
    blind_review_artifact_path: Path,
    comparison_artifact_path: Path,
    rollback_root: Path,
    expected_config_sha256: str,
    _before_replace: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """CAS one same-batch blind-review winner into production."""

    if config_path.is_symlink():
        raise SemanticModelPromotionError(
            "production config must be a regular file"
        )
    config_file = config_path.resolve(strict=True)
    if not config_file.is_file():
        raise SemanticModelPromotionError(
            "production config must be a regular file"
        )
    expected_sha256 = _sha256(
        expected_config_sha256,
        field="expectedConfigSha256",
    )
    lock_path = config_file.with_name(f".{config_file.name}.promotion.lock")
    temporary = config_file.with_name(
        f".{config_file.name}.{uuid.uuid4().hex}.promotion.tmp"
    )
    with ExitStack() as stack:
        stack.enter_context(_promotion_lock(lock_path))
        stack.callback(temporary.unlink, missing_ok=True)
        config, config_raw = _load_json(config_file, label="production config")
        current_sha256 = hashlib.sha256(config_raw).hexdigest()
        if current_sha256 != expected_sha256:
            raise SemanticModelPromotionError(
                "production config compare-and-swap is stale"
            )
        active_config = ProductionConfig.load(config_file)

        decision_raw, decision_bytes = _load_json(
            decision_path,
            label="promotion decision",
        )
        decision = _validate_decision(
            decision_raw,
            file_sha256=hashlib.sha256(decision_bytes).hexdigest(),
        )
        if decision["expectedConfigSha256"] != expected_sha256:
            raise SemanticModelPromotionError(
                "decision is not bound to the expected production config"
            )
        blind_review_path = _verify_evidence_artifact(
            blind_review_artifact_path,
            expected_sha256=decision["blindReviewArtifactSha256"],
            label="blind-review artifact",
        )
        comparison_path = _verify_evidence_artifact(
            comparison_artifact_path,
            expected_sha256=decision["comparisonArtifactSha256"],
            label="comparison artifact",
        )
        if blind_review_path == comparison_path:
            raise SemanticModelPromotionError(
                "blind-review and comparison artifacts must be distinct"
            )

        try:
            registry = load_registry(registry_path)
            validate_registry(
                registry,
                verify_local=True,
                model_ids=[decision["challengerRegistryModelId"]],
            )
        except (OSError, RegistryValidationError) as exc:
            raise SemanticModelPromotionError(
                f"model registry is invalid: {exc}"
            ) from exc
        incumbent = _registry_binding(
            registry,
            registry_model_id=decision["incumbentRegistryModelId"],
        )
        challenger = _registry_binding(
            registry,
            registry_model_id=decision["challengerRegistryModelId"],
        )
        if (
            active_config.speaker.local_llm_model != incumbent["model"]
            or active_config.speaker.local_llm_model_digest
            != incumbent["digest"]
        ):
            raise SemanticModelPromotionError(
                "decision incumbent does not match active production"
            )

        speaker = config.get("speaker")
        if not isinstance(speaker, dict):
            raise SemanticModelPromotionError(
                "production config speaker binding is invalid"
            )
        rollback_snapshot = _publish_rollback_snapshot(
            rollback_root=rollback_root,
            decision_id=decision["decisionId"],
            config_sha256=expected_sha256,
            config_bytes=config_raw,
        )
        speaker["localLlmModel"] = challenger["model"]
        speaker["localLlmModelDigest"] = challenger["digest"]
        payload = _serialize_config(config)
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        ProductionConfig.load(temporary)
        if _before_replace is not None:
            _before_replace(temporary)
        if sha256_file(config_file) != expected_sha256:
            raise SemanticModelPromotionError(
                "production config changed before atomic replacement"
            )
        os.replace(temporary, config_file)
        _fsync_directory(config_file.parent)
        promoted_sha256 = sha256_file(config_file)
        return {
            "schemaVersion": "1.0.0",
            "artifactType": RECEIPT_ARTIFACT_TYPE,
            "deploymentSlot": DEPLOYMENT_SLOT,
            "adapterId": ADAPTER_ID,
            "decisionId": decision["decisionId"],
            "decisionArtifactSha256": decision["decisionArtifactSha256"],
            "decisionCanonicalSha256": decision[
                "decisionCanonicalSha256"
            ],
            "blindReviewArtifactSha256": decision[
                "blindReviewArtifactSha256"
            ],
            "comparisonArtifactSha256": decision[
                "comparisonArtifactSha256"
            ],
            "blindBatchId": decision["blindBatchId"],
            "caseSetSha256": decision["caseSetSha256"],
            "reviewSource": decision["reviewSource"],
            "reviewer": decision["reviewer"],
            "decidedAt": decision["decidedAt"],
            "previousConfigSha256": expected_sha256,
            "promotedConfigSha256": promoted_sha256,
            "incumbentRegistryModelId": decision[
                "incumbentRegistryModelId"
            ],
            "activeRegistryModelId": decision[
                "challengerRegistryModelId"
            ],
            "activeModel": challenger["model"],
            "activeModelDigest": challenger["digest"],
            "activeManifestSha256": challenger["manifestSha256"],
            "rollbackConfigPath": str(rollback_snapshot),
            "rollbackConfigSha256": expected_sha256,
            "promotionPolicy": {
                "registryStatusUsedAsGate": False,
                "incumbentMarginRequired": False,
                "protectionPeriodApplied": False,
                "immediateReplacementAllowed": True,
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--blind-review-artifact", type=Path, required=True)
    parser.add_argument("--comparison-artifact", type=Path, required=True)
    parser.add_argument("--rollback-root", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        receipt = promote_semantic_model(
            config_path=arguments.config,
            registry_path=arguments.registry,
            decision_path=arguments.decision,
            blind_review_artifact_path=arguments.blind_review_artifact,
            comparison_artifact_path=arguments.comparison_artifact,
            rollback_root=arguments.rollback_root,
            expected_config_sha256=arguments.expected_config_sha256,
        )
    except (OSError, SemanticModelPromotionError) as exc:
        print(f"semantic-model promotion failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
