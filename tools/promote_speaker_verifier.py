"""Atomically replace the active secondary speaker-verifier deployment.

The production configuration is the active source of truth.  Registry status
labels are deliberately ignored: the registry only proves model identity and
declares which deployment slot and adapter combinations are compatible.
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
    _host_path,
    load_registry,
    validate_registry,
)


DECISION_SCHEMA_VERSION = "1.0.0"
DECISION_ARTIFACT_TYPE = "speaker-verifier-blind-promotion-decision"
DEPLOYMENT_SLOT = "secondary-speaker-verification"
_MANUAL_AUDIT_SOURCES = frozenset({"human", "codex-manual"})
_MAX_JSON_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")


class SpeakerVerifierPromotionError(ValueError):
    """Raised when promotion evidence or its compare-and-swap is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpeakerVerifierPromotionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise SpeakerVerifierPromotionError(f"{label} must be a regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SpeakerVerifierPromotionError(f"{label} must be a regular file")
    raw = resolved.read_bytes()
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise SpeakerVerifierPromotionError(f"{label} has an invalid size")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeakerVerifierPromotionError(
            f"{label} must be valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SpeakerVerifierPromotionError(f"{label} must be an object")
    return value, raw


def _object(
    value: Any,
    *,
    field: str,
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpeakerVerifierPromotionError(f"{field} must be an object")
    output = {str(key): item for key, item in value.items()}
    missing = sorted(required - set(output))
    unknown = sorted(set(output) - required)
    if missing or unknown:
        raise SpeakerVerifierPromotionError(
            f"{field} keys are invalid: missing={missing}, unknown={unknown}"
        )
    return output


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SpeakerVerifierPromotionError(
            f"{field} must be trimmed non-empty text"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SpeakerVerifierPromotionError(f"{field} contains control characters")
    return value


def _identifier(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if _ID_RE.fullmatch(text) is None:
        raise SpeakerVerifierPromotionError(f"{field} has an invalid identifier")
    return text


def _sha256(value: Any, *, field: str) -> str:
    text = _text(value, field=field).casefold().removeprefix("sha256:")
    if _SHA256_RE.fullmatch(text) is None:
        raise SpeakerVerifierPromotionError(f"{field} must be a SHA-256 digest")
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
                "automaticScoring",
                "expectedConfigSha256",
                "incumbentRegistryModelId",
                "challengerRegistryModelId",
                "adapterId",
                "outcome",
                "evidence",
                "audit",
                "canonicalSha256",
            }
        ),
    )
    if decision["schemaVersion"] != DECISION_SCHEMA_VERSION:
        raise SpeakerVerifierPromotionError("decision schemaVersion is unsupported")
    if decision["artifactType"] != DECISION_ARTIFACT_TYPE:
        raise SpeakerVerifierPromotionError("decision artifactType is unsupported")
    if decision["deploymentSlot"] != DEPLOYMENT_SLOT:
        raise SpeakerVerifierPromotionError("decision deploymentSlot is unsupported")
    if decision["automaticScoring"] is not False:
        raise SpeakerVerifierPromotionError(
            "promotion requires an explicit manual blind-review decision"
        )
    if decision["outcome"] != "promote-challenger":
        raise SpeakerVerifierPromotionError("decision does not promote challenger")
    canonical = dict(decision)
    declared_canonical = _sha256(
        canonical.pop("canonicalSha256"),
        field="decision.canonicalSha256",
    )
    if canonical_json_sha256(canonical) != declared_canonical:
        raise SpeakerVerifierPromotionError(
            "decision canonical SHA-256 does not match"
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
    audit_source = _text(audit["source"], field="decision.audit.source")
    if audit_source not in _MANUAL_AUDIT_SOURCES:
        raise SpeakerVerifierPromotionError(
            "decision.audit.source must identify a manual reviewer"
        )
    _text(audit["reviewer"], field="decision.audit.reviewer")
    _text(audit["decidedAt"], field="decision.audit.decidedAt")
    incumbent = _identifier(
        decision["incumbentRegistryModelId"],
        field="decision.incumbentRegistryModelId",
    )
    challenger = _identifier(
        decision["challengerRegistryModelId"],
        field="decision.challengerRegistryModelId",
    )
    if incumbent == challenger:
        raise SpeakerVerifierPromotionError(
            "decision challenger must differ from incumbent"
        )
    return {
        "decisionId": _identifier(
            decision["decisionId"], field="decision.decisionId"
        ),
        "decisionArtifactSha256": file_sha256,
        "decisionCanonicalSha256": declared_canonical,
        "auditSource": audit_source,
        "blindReviewArtifactSha256": _sha256(
            evidence["blindReviewArtifactSha256"],
            field="decision.evidence.blindReviewArtifactSha256",
        ),
        "comparisonArtifactSha256": _sha256(
            evidence["comparisonArtifactSha256"],
            field="decision.evidence.comparisonArtifactSha256",
        ),
        "expectedConfigSha256": _sha256(
            decision["expectedConfigSha256"],
            field="decision.expectedConfigSha256",
        ),
        "incumbentRegistryModelId": incumbent,
        "challengerRegistryModelId": challenger,
        "adapterId": _identifier(
            decision["adapterId"], field="decision.adapterId"
        ),
    }


def _registry_binding(
    registry: Mapping[str, Any],
    *,
    registry_model_id: str,
    deployment_slot: str,
    adapter_id: str,
) -> dict[str, str]:
    models = registry.get("models")
    if not isinstance(models, list):
        raise SpeakerVerifierPromotionError("registry models are invalid")
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
        raise SpeakerVerifierPromotionError(
            f"registry model is missing: {registry_model_id}"
        )
    compatible = any(
        isinstance(slot, Mapping)
        and slot.get("id") == deployment_slot
        and isinstance(slot.get("adapterIds"), list)
        and adapter_id in slot["adapterIds"]
        for slot in selected.get("deploymentSlots", [])
    )
    if not compatible:
        raise SpeakerVerifierPromotionError(
            "registry model is incompatible with the deployment slot or adapter"
        )
    local = selected.get("local")
    if not isinstance(local, Mapping):
        raise SpeakerVerifierPromotionError("registry model local identity is invalid")
    manifest_identity = local.get("manifest")
    if not isinstance(manifest_identity, Mapping):
        raise SpeakerVerifierPromotionError(
            "registry model manifest identity is invalid"
        )
    raw_path = _text(local.get("path"), field="registry.model.local.path")
    relative_manifest = _text(
        manifest_identity.get("path"),
        field="registry.model.local.manifest.path",
    )
    manifest_sha256 = _sha256(
        manifest_identity.get("sha256"),
        field="registry.model.local.manifest.sha256",
    )
    manifest_path = _host_path(raw_path) / relative_manifest
    try:
        actual_manifest_sha256 = sha256_file(manifest_path.resolve(strict=True))
        manifest_document, _ = _load_json(
            manifest_path,
            label="challenger model manifest",
        )
    except OSError as exc:
        raise SpeakerVerifierPromotionError(
            "challenger model manifest is unavailable"
        ) from exc
    if actual_manifest_sha256 != manifest_sha256:
        raise SpeakerVerifierPromotionError(
            "challenger model manifest differs from the registry identity"
        )
    manifest_model_key = _text(
        manifest_document.get("modelKey"),
        field="challenger model manifest.modelKey",
    )
    return {
        "path": raw_path,
        "deploymentSlot": deployment_slot,
        "registryModelId": registry_model_id,
        "manifestModelKey": manifest_model_key,
        "manifestSha256": f"sha256:{manifest_sha256}",
        "adapterId": adapter_id,
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
        raise SpeakerVerifierPromotionError(f"{label} must be a regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SpeakerVerifierPromotionError(f"{label} must be a regular file")
    if sha256_file(resolved) != expected_sha256:
        raise SpeakerVerifierPromotionError(f"{label} SHA-256 does not match")
    return resolved


def _publish_rollback_snapshot(
    *,
    rollback_root: Path,
    decision_id: str,
    config_sha256: str,
    config_bytes: bytes,
) -> Path:
    if rollback_root.exists() and rollback_root.is_symlink():
        raise SpeakerVerifierPromotionError(
            "rollback root must not be a symbolic link"
        )
    rollback_root.mkdir(parents=True, exist_ok=True)
    resolved_root = rollback_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise SpeakerVerifierPromotionError("rollback root must be a directory")
    snapshot = resolved_root / (
        f"{decision_id}.{config_sha256}.production-config.rollback.json"
    )
    if snapshot.exists():
        if snapshot.is_symlink() or sha256_file(snapshot) != config_sha256:
            raise SpeakerVerifierPromotionError(
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
            raise SpeakerVerifierPromotionError(
                "concurrent rollback snapshot differs from the incumbent config"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    if not published and not snapshot.is_file():
        raise SpeakerVerifierPromotionError("rollback snapshot was not published")
    return snapshot


@contextmanager
def _promotion_lock(path: Path) -> Iterator[None]:
    """Take a crash-released OS lock while retaining one stable lock inode."""

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
                raise SpeakerVerifierPromotionError(
                    "another speaker-verifier promotion is in progress"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise SpeakerVerifierPromotionError(
                    "another speaker-verifier promotion is in progress"
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


def promote_speaker_verifier(
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
    """CAS one reviewed challenger into the active production config."""

    if config_path.is_symlink():
        raise SpeakerVerifierPromotionError(
            "production config must be a regular file"
        )
    config_file = config_path.resolve(strict=True)
    if not config_file.is_file():
        raise SpeakerVerifierPromotionError(
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
        config, config_raw = _load_json(
            config_file,
            label="production config",
        )
        current_sha256 = hashlib.sha256(config_raw).hexdigest()
        if current_sha256 != expected_sha256:
            raise SpeakerVerifierPromotionError(
                "production config compare-and-swap is stale"
            )
        ProductionConfig.load(config_file)

        decision_raw, decision_bytes = _load_json(
            decision_path,
            label="promotion decision",
        )
        decision = _validate_decision(
            decision_raw,
            file_sha256=hashlib.sha256(decision_bytes).hexdigest(),
        )
        if decision["expectedConfigSha256"] != expected_sha256:
            raise SpeakerVerifierPromotionError(
                "decision is not bound to the expected production config"
            )
        _verify_evidence_artifact(
            blind_review_artifact_path,
            expected_sha256=decision["blindReviewArtifactSha256"],
            label="blind-review artifact",
        )
        _verify_evidence_artifact(
            comparison_artifact_path,
            expected_sha256=decision["comparisonArtifactSha256"],
            label="comparison artifact",
        )
        models = config.get("models")
        if not isinstance(models, dict):
            raise SpeakerVerifierPromotionError(
                "production config models are invalid"
            )
        active = (
            ProductionConfig.load(config_file)
            .models.secondary_speaker_verifier
        )
        if active.registry_model_id != decision["incumbentRegistryModelId"]:
            raise SpeakerVerifierPromotionError(
                "decision incumbent does not match active production"
            )

        try:
            registry = load_registry(registry_path)
            validate_registry(
                registry,
                verify_local=True,
                model_ids=[decision["challengerRegistryModelId"]],
            )
        except (OSError, RegistryValidationError) as exc:
            raise SpeakerVerifierPromotionError(
                f"model registry is invalid: {exc}"
            ) from exc
        binding = _registry_binding(
            registry,
            registry_model_id=decision["challengerRegistryModelId"],
            deployment_slot=DEPLOYMENT_SLOT,
            adapter_id=decision["adapterId"],
        )
        rollback_snapshot = _publish_rollback_snapshot(
            rollback_root=rollback_root,
            decision_id=decision["decisionId"],
            config_sha256=expected_sha256,
            config_bytes=config_raw,
        )
        binding["promotionEvidence"] = {
            "decisionId": decision["decisionId"],
            "decisionArtifactSha256": (
                f"sha256:{decision['decisionArtifactSha256']}"
            ),
            "decisionCanonicalSha256": (
                f"sha256:{decision['decisionCanonicalSha256']}"
            ),
            "blindReviewArtifactSha256": (
                f"sha256:{decision['blindReviewArtifactSha256']}"
            ),
            "comparisonArtifactSha256": (
                f"sha256:{decision['comparisonArtifactSha256']}"
            ),
            "previousConfigSha256": f"sha256:{expected_sha256}",
            "rollbackConfigPath": str(rollback_snapshot),
            "rollbackConfigSha256": f"sha256:{expected_sha256}",
        }
        models.pop("eres2netV2", None)
        models["secondarySpeakerVerifier"] = binding
        payload = _serialize_config(config)
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        ProductionConfig.load(temporary)
        if _before_replace is not None:
            _before_replace(temporary)
        if sha256_file(config_file) != expected_sha256:
            raise SpeakerVerifierPromotionError(
                "production config changed before atomic replacement"
            )
        os.replace(temporary, config_file)
        _fsync_directory(config_file.parent)
        promoted_sha256 = sha256_file(config_file)
        return {
            "schemaVersion": "1.0.0",
            "artifactType": "speaker-verifier-production-promotion-receipt",
            "deploymentSlot": DEPLOYMENT_SLOT,
            "promotionPolicy": {
                "registryStatusUsedAsGate": False,
                "incumbentMarginRequired": False,
                "protectionPeriodApplied": False,
                "immediateReplacementAllowed": True,
            },
            "decisionId": decision["decisionId"],
            "decisionArtifactSha256": decision["decisionArtifactSha256"],
            "auditSource": decision["auditSource"],
            "previousConfigSha256": expected_sha256,
            "promotedConfigSha256": promoted_sha256,
            "incumbentRegistryModelId": decision[
                "incumbentRegistryModelId"
            ],
            "activeRegistryModelId": decision["challengerRegistryModelId"],
            "activeAdapterId": decision["adapterId"],
            "rollbackConfigPath": str(rollback_snapshot),
            "rollbackConfigSha256": expected_sha256,
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
        receipt = promote_speaker_verifier(
            config_path=arguments.config,
            registry_path=arguments.registry,
            decision_path=arguments.decision,
            blind_review_artifact_path=arguments.blind_review_artifact,
            comparison_artifact_path=arguments.comparison_artifact,
            rollback_root=arguments.rollback_root,
            expected_config_sha256=arguments.expected_config_sha256,
        )
    except (OSError, SpeakerVerifierPromotionError) as exc:
        print(f"speaker-verifier promotion failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
