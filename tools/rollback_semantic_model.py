"""Atomically restore a verified semantic-model production configuration.

This is an emergency recovery path for a promoted model that cannot serve the
product. It restores the immutable incumbent snapshot recorded by a promotion
receipt. Rollback does not protect the restored model: any later same-batch
blind-review winner may replace it immediately through the normal promotion
CAS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
from tools.promote_semantic_model import (  # noqa: E402
    ADAPTER_ID,
    DEPLOYMENT_SLOT,
    RECEIPT_ARTIFACT_TYPE as PROMOTION_RECEIPT_ARTIFACT_TYPE,
    SemanticModelPromotionError,
    _fsync_directory,
    _identifier,
    _load_json,
    _promotion_lock,
    _registry_binding,
    _sha256,
    _text,
)


ROLLBACK_RECEIPT_SCHEMA_VERSION = "1.0.0"
ROLLBACK_RECEIPT_ARTIFACT_TYPE = "semantic-model-production-rollback-receipt"
RECOVERY_REASONS = frozenset(
    {
        "active-model-unavailable",
        "artifact-integrity-failure",
        "operator-drill",
        "resource-exhaustion",
        "startup-failure",
    }
)
_PROMOTION_RECEIPT_FIELDS = frozenset(
    {
        "schemaVersion",
        "artifactType",
        "deploymentSlot",
        "adapterId",
        "decisionId",
        "decisionArtifactSha256",
        "decisionCanonicalSha256",
        "blindReviewArtifactSha256",
        "comparisonArtifactSha256",
        "blindBatchId",
        "caseSetSha256",
        "reviewSource",
        "reviewer",
        "decidedAt",
        "previousConfigSha256",
        "promotedConfigSha256",
        "incumbentRegistryModelId",
        "activeRegistryModelId",
        "activeModel",
        "activeModelDigest",
        "activeManifestSha256",
        "rollbackConfigPath",
        "rollbackConfigSha256",
        "promotionPolicy",
    }
)


class SemanticModelRollbackError(ValueError):
    """Raised when rollback evidence or its production-config CAS is invalid."""


def _validate_promotion_receipt(
    value: Mapping[str, Any],
    *,
    file_sha256: str,
) -> dict[str, str]:
    if set(value) != _PROMOTION_RECEIPT_FIELDS:
        missing = sorted(_PROMOTION_RECEIPT_FIELDS - set(value))
        unknown = sorted(set(value) - _PROMOTION_RECEIPT_FIELDS)
        raise SemanticModelRollbackError(
            "promotion receipt fields are invalid: "
            f"missing={missing}, unknown={unknown}"
        )
    if value.get("schemaVersion") != "1.0.0":
        raise SemanticModelRollbackError("promotion receipt schemaVersion is unsupported")
    if value.get("artifactType") != PROMOTION_RECEIPT_ARTIFACT_TYPE:
        raise SemanticModelRollbackError("promotion receipt artifactType is unsupported")
    if value.get("deploymentSlot") != DEPLOYMENT_SLOT:
        raise SemanticModelRollbackError("promotion receipt deploymentSlot is unsupported")
    if value.get("adapterId") != ADAPTER_ID:
        raise SemanticModelRollbackError("promotion receipt adapterId is unsupported")
    policy = value.get("promotionPolicy")
    if policy != {
        "registryStatusUsedAsGate": False,
        "incumbentMarginRequired": False,
        "protectionPeriodApplied": False,
        "immediateReplacementAllowed": True,
    }:
        raise SemanticModelRollbackError("promotion receipt policy is invalid")

    decision_id = _identifier(value.get("decisionId"), field="receipt.decisionId")
    incumbent_id = _identifier(
        value.get("incumbentRegistryModelId"),
        field="receipt.incumbentRegistryModelId",
    )
    active_id = _identifier(
        value.get("activeRegistryModelId"),
        field="receipt.activeRegistryModelId",
    )
    if incumbent_id == active_id:
        raise SemanticModelRollbackError(
            "promotion receipt incumbent and active model must differ"
        )
    previous_sha256 = _sha256(
        value.get("previousConfigSha256"),
        field="receipt.previousConfigSha256",
    )
    rollback_sha256 = _sha256(
        value.get("rollbackConfigSha256"),
        field="receipt.rollbackConfigSha256",
    )
    if rollback_sha256 != previous_sha256:
        raise SemanticModelRollbackError(
            "promotion receipt rollback snapshot is not the previous config"
        )
    digest_fields = (
        "decisionArtifactSha256",
        "decisionCanonicalSha256",
        "blindReviewArtifactSha256",
        "comparisonArtifactSha256",
        "caseSetSha256",
        "promotedConfigSha256",
        "activeModelDigest",
        "activeManifestSha256",
    )
    digests = {
        field: _sha256(value.get(field), field=f"receipt.{field}")
        for field in digest_fields
    }
    return {
        "receiptFileSha256": file_sha256,
        "decisionId": decision_id,
        "incumbentRegistryModelId": incumbent_id,
        "activeRegistryModelId": active_id,
        "activeModel": _text(value.get("activeModel"), field="receipt.activeModel"),
        "activeModelDigest": f"sha256:{digests['activeModelDigest']}",
        "previousConfigSha256": previous_sha256,
        "promotedConfigSha256": digests["promotedConfigSha256"],
        "rollbackConfigPath": _text(
            value.get("rollbackConfigPath"),
            field="receipt.rollbackConfigPath",
        ),
        "rollbackConfigSha256": rollback_sha256,
    }


def _resolve_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise SemanticModelRollbackError(f"{label} must be a regular file")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SemanticModelRollbackError(f"{label} must be a regular file") from exc
    if not resolved.is_file():
        raise SemanticModelRollbackError(f"{label} must be a regular file")
    return resolved


def _publish_failed_active_snapshot(
    *,
    recovery_root: Path,
    decision_id: str,
    config_sha256: str,
    config_bytes: bytes,
) -> Path:
    if recovery_root.exists() and recovery_root.is_symlink():
        raise SemanticModelRollbackError("recovery root must not be a symbolic link")
    recovery_root.mkdir(parents=True, exist_ok=True)
    resolved_root = recovery_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise SemanticModelRollbackError("recovery root must be a directory")
    snapshot = resolved_root / (
        f"{decision_id}.{config_sha256}.failed-active.production-config.json"
    )
    if snapshot.exists():
        if snapshot.is_symlink() or sha256_file(snapshot) != config_sha256:
            raise SemanticModelRollbackError(
                "existing failed-active snapshot differs from the active config"
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
            raise SemanticModelRollbackError(
                "concurrent failed-active snapshot differs from the active config"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    if not published and not snapshot.is_file():
        raise SemanticModelRollbackError("failed-active snapshot was not published")
    return snapshot


def rollback_semantic_model(
    *,
    config_path: Path,
    registry_path: Path,
    promotion_receipt_path: Path,
    rollback_config_path: Path,
    recovery_root: Path,
    expected_active_config_sha256: str,
    reason: str,
    _before_replace: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """CAS the receipt-bound incumbent config back into the active pointer."""

    if reason not in RECOVERY_REASONS:
        raise SemanticModelRollbackError(
            f"rollback reason is unsupported: {reason!r}"
        )
    config_file = _resolve_regular_file(config_path, label="production config")
    receipt_file = _resolve_regular_file(
        promotion_receipt_path,
        label="promotion receipt",
    )
    rollback_file = _resolve_regular_file(
        rollback_config_path,
        label="rollback config",
    )
    expected_active_sha256 = _sha256(
        expected_active_config_sha256,
        field="expectedActiveConfigSha256",
    )
    lock_path = config_file.with_name(f".{config_file.name}.promotion.lock")
    temporary = config_file.with_name(
        f".{config_file.name}.{uuid.uuid4().hex}.rollback.tmp"
    )
    with ExitStack() as stack:
        stack.enter_context(_promotion_lock(lock_path))
        stack.callback(temporary.unlink, missing_ok=True)
        current, current_raw = _load_json(config_file, label="production config")
        current_sha256 = hashlib.sha256(current_raw).hexdigest()
        if current_sha256 != expected_active_sha256:
            raise SemanticModelRollbackError(
                "production config rollback compare-and-swap is stale"
            )
        current_config = ProductionConfig.load(config_file)

        receipt_value, receipt_raw = _load_json(
            receipt_file,
            label="promotion receipt",
        )
        receipt = _validate_promotion_receipt(
            receipt_value,
            file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        )
        if receipt["promotedConfigSha256"] != expected_active_sha256:
            raise SemanticModelRollbackError(
                "promotion receipt is not bound to the active production config"
            )
        declared_rollback = Path(receipt["rollbackConfigPath"]).resolve(strict=False)
        if declared_rollback != rollback_file:
            raise SemanticModelRollbackError(
                "rollback config path does not match the promotion receipt"
            )
        if sha256_file(rollback_file) != receipt["rollbackConfigSha256"]:
            raise SemanticModelRollbackError("rollback config SHA-256 does not match")
        rollback_value, rollback_raw = _load_json(
            rollback_file,
            label="rollback config",
        )
        if hashlib.sha256(rollback_raw).hexdigest() != receipt["previousConfigSha256"]:
            raise SemanticModelRollbackError(
                "rollback config is not the receipt-bound previous config"
            )
        rollback_config = ProductionConfig.load(rollback_file)

        if (
            current_config.speaker.local_llm_model != receipt["activeModel"]
            or current_config.speaker.local_llm_model_digest
            != receipt["activeModelDigest"]
        ):
            raise SemanticModelRollbackError(
                "promotion receipt active model does not match production"
            )
        try:
            registry = load_registry(registry_path)
            validate_registry(
                registry,
                verify_local=True,
                model_ids=[receipt["incumbentRegistryModelId"]],
            )
            rollback_binding = _registry_binding(
                registry,
                registry_model_id=receipt["incumbentRegistryModelId"],
            )
        except (OSError, RegistryValidationError, SemanticModelPromotionError) as exc:
            raise SemanticModelRollbackError(
                f"rollback model is not locally available: {exc}"
            ) from exc
        if (
            rollback_config.speaker.local_llm_model != rollback_binding["model"]
            or rollback_config.speaker.local_llm_model_digest
            != rollback_binding["digest"]
        ):
            raise SemanticModelRollbackError(
                "rollback config does not bind the receipt incumbent model"
            )

        failed_snapshot = _publish_failed_active_snapshot(
            recovery_root=recovery_root,
            decision_id=receipt["decisionId"],
            config_sha256=current_sha256,
            config_bytes=current_raw,
        )
        with temporary.open("xb") as handle:
            handle.write(rollback_raw)
            handle.flush()
            os.fsync(handle.fileno())
        ProductionConfig.load(temporary)
        if _before_replace is not None:
            _before_replace(temporary)
        if sha256_file(config_file) != expected_active_sha256:
            raise SemanticModelRollbackError(
                "production config changed before atomic rollback replacement"
            )
        os.replace(temporary, config_file)
        _fsync_directory(config_file.parent)
        restored_sha256 = sha256_file(config_file)
        if restored_sha256 != receipt["previousConfigSha256"]:
            raise SemanticModelRollbackError(
                "restored production config digest is inconsistent"
            )
        body: dict[str, Any] = {
            "schemaVersion": ROLLBACK_RECEIPT_SCHEMA_VERSION,
            "artifactType": ROLLBACK_RECEIPT_ARTIFACT_TYPE,
            "deploymentSlot": DEPLOYMENT_SLOT,
            "adapterId": ADAPTER_ID,
            "reason": reason,
            "decisionId": receipt["decisionId"],
            "promotionReceiptFileSha256": receipt["receiptFileSha256"],
            "failedActiveRegistryModelId": receipt["activeRegistryModelId"],
            "failedActiveConfigSha256": current_sha256,
            "failedActiveConfigSnapshotPath": str(failed_snapshot),
            "failedActiveConfigSnapshotSha256": current_sha256,
            "restoredRegistryModelId": receipt["incumbentRegistryModelId"],
            "restoredModel": rollback_binding["model"],
            "restoredModelDigest": rollback_binding["digest"],
            "restoredManifestSha256": rollback_binding["manifestSha256"],
            "restoredConfigSha256": restored_sha256,
            "completedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "host": socket.gethostname(),
            "processId": os.getpid(),
            "rollbackPolicy": {
                "blindReviewRequiredForEmergencyRecovery": False,
                "registryStatusUsedAsGate": False,
                "restoredModelProtectionApplied": False,
                "immediateChallengerReplacementAllowed": True,
            },
        }
        body["canonicalSha256"] = canonical_json_sha256(body)
        return body


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--promotion-receipt", type=Path, required=True)
    parser.add_argument("--rollback-config", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--expected-active-config-sha256", required=True)
    parser.add_argument("--reason", choices=sorted(RECOVERY_REASONS), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        receipt = rollback_semantic_model(
            config_path=arguments.config,
            registry_path=arguments.registry,
            promotion_receipt_path=arguments.promotion_receipt,
            rollback_config_path=arguments.rollback_config,
            recovery_root=arguments.recovery_root,
            expected_active_config_sha256=(
                arguments.expected_active_config_sha256
            ),
            reason=arguments.reason,
        )
    except (OSError, SemanticModelPromotionError, SemanticModelRollbackError) as exc:
        print(f"semantic-model rollback failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
