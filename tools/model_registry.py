"""Validate the auditable registry of locally installed model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "local-model-registry.json"
)

MODEL_STATUSES = frozenset(
    {"production", "challenger", "development", "research", "retired"}
)
DECISION_POLICIES = frozenset(
    {"direct", "evidence-only", "fallback-only", "challenger-only", "suggestion-only"}
)
MANIFEST_FORMATS = frozenset(
    {"mts-model-manifest-v1", "ollama-oci-manifest-v2"}
)
ARTIFACT_FORMATS = frozenset(
    {"safetensors", "pytorch-checkpoint", "gguf"}
)
QUANTIZATION_SCHEMES = frozenset({"not-declared", "Q4_K_M"})

_ID_RE = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
_ROLE_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MODELSCOPE_REVISION_RE = re.compile(r"^(?:master|v[0-9]+\.[0-9]+\.[0-9]+)$")
_SPDX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_ACCELERATOR_RE = re.compile(r"^(?:cpu|gpu|cuda(?::[0-9]+)?)$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^([A-Za-z]):/(.+)$")


class RegistryValidationError(ValueError):
    """Raised when registry evidence cannot be accepted safely."""


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistryValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RegistryValidationError(f"cannot read {label}: {path}") from error
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except UnicodeDecodeError as error:
        raise RegistryValidationError(f"{label} is not UTF-8: {path}") from error
    except json.JSONDecodeError as error:
        raise RegistryValidationError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{label} root must be an object")
    return value


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> Mapping[str, Any]:
    """Load a registry with duplicate-key and UTF-8 checks."""

    return _read_json(path.resolve(strict=True), label="model registry")


def _expect_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    field: str,
) -> None:
    keys = frozenset(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing or unknown:
        raise RegistryValidationError(
            f"{field} keys are invalid: missing={missing}, unknown={unknown}"
        )


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{field} must be an object")
    return value


def _list(value: Any, *, field: str, nonempty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty" if nonempty else "an"
        raise RegistryValidationError(f"{field} must be {qualifier} array")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RegistryValidationError(f"{field} must be a trimmed non-empty string")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RegistryValidationError(f"{field} must be a positive integer")
    return value


def _sha256(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if not _SHA256_RE.fullmatch(text):
        raise RegistryValidationError(f"{field} must be lowercase SHA-256 hex")
    return text


def _digest(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if not _SHA256_DIGEST_RE.fullmatch(text):
        raise RegistryValidationError(
            f"{field} must have the form sha256:<64 lowercase hex>"
        )
    return text


def _relative_path(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if "\\" in text or text.startswith("/"):
        raise RegistryValidationError(
            f"{field} must be a normalized relative POSIX path"
        )
    raw_parts = text.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise RegistryValidationError(f"{field} contains an unsafe path segment")
    normalized = PurePosixPath(text)
    if normalized.is_absolute() or normalized.as_posix() != text:
        raise RegistryValidationError(
            f"{field} must be a normalized relative POSIX path"
        )
    return text


def _absolute_local_path(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if "\\" in text:
        raise RegistryValidationError(
            f"{field} must use forward slashes for deterministic identity"
        )
    windows = _WINDOWS_ABSOLUTE_RE.fullmatch(text)
    if windows is not None:
        if any(part in {"", ".", ".."} for part in windows.group(2).split("/")):
            raise RegistryValidationError(f"{field} contains an unsafe path segment")
        return text
    if not text.startswith("/"):
        raise RegistryValidationError(f"{field} must be an absolute local path")
    if any(part in {".", ".."} for part in text.split("/")):
        raise RegistryValidationError(f"{field} contains an unsafe path segment")
    return text


def _host_path(value: str) -> Path:
    windows = _WINDOWS_ABSOLUTE_RE.fullmatch(value)
    if windows is None:
        return Path(value)
    if os.name == "nt":
        return Path(value)
    drive = windows.group(1).lower()
    return Path("/mnt") / drive / Path(*windows.group(2).split("/"))


def _validate_hardware_profile(value: Any, *, index: int) -> str:
    field = f"hardwareProfiles[{index}]"
    profile = _mapping(value, field=field)
    _expect_keys(
        profile,
        required=frozenset({"id", "platform", "cpu", "ramBytes", "gpu"}),
        field=field,
    )
    profile_id = _text(profile["id"], field=f"{field}.id")
    if not _ID_RE.fullmatch(profile_id):
        raise RegistryValidationError(f"{field}.id has an invalid identifier")
    _text(profile["platform"], field=f"{field}.platform")
    _positive_int(profile["ramBytes"], field=f"{field}.ramBytes")

    cpu = _mapping(profile["cpu"], field=f"{field}.cpu")
    _expect_keys(
        cpu,
        required=frozenset({"model", "physicalCores", "logicalProcessors"}),
        field=f"{field}.cpu",
    )
    _text(cpu["model"], field=f"{field}.cpu.model")
    physical = _positive_int(
        cpu["physicalCores"], field=f"{field}.cpu.physicalCores"
    )
    logical = _positive_int(
        cpu["logicalProcessors"], field=f"{field}.cpu.logicalProcessors"
    )
    if logical < physical:
        raise RegistryValidationError(
            f"{field}.cpu.logicalProcessors must be >= physicalCores"
        )

    gpu = _mapping(profile["gpu"], field=f"{field}.gpu")
    _expect_keys(
        gpu,
        required=frozenset(
            {"model", "vramBytes", "computeCapability", "driverVersion"}
        ),
        field=f"{field}.gpu",
    )
    _text(gpu["model"], field=f"{field}.gpu.model")
    _positive_int(gpu["vramBytes"], field=f"{field}.gpu.vramBytes")
    if not re.fullmatch(
        r"[0-9]+\.[0-9]+",
        _text(gpu["computeCapability"], field=f"{field}.gpu.computeCapability"),
    ):
        raise RegistryValidationError(
            f"{field}.gpu.computeCapability must be major.minor"
        )
    _text(gpu["driverVersion"], field=f"{field}.gpu.driverVersion")
    return profile_id


def _validate_source(value: Any, *, field: str) -> Mapping[str, Any]:
    source = _mapping(value, field=field)
    provider = _text(source.get("provider"), field=f"{field}.provider")
    if provider in {"modelscope", "huggingface"}:
        _expect_keys(
            source,
            required=frozenset({"provider", "repository", "revision"}),
            field=field,
        )
        _text(source["repository"], field=f"{field}.repository")
        revision = _text(source["revision"], field=f"{field}.revision")
        if provider == "huggingface" and not _GIT_COMMIT_RE.fullmatch(revision):
            raise RegistryValidationError(
                f"{field}.revision must be a pinned 40-character commit for Hugging Face"
            )
        if provider == "modelscope" and not _MODELSCOPE_REVISION_RE.fullmatch(
            revision
        ):
            raise RegistryValidationError(
                f"{field}.revision must be master or a vMAJOR.MINOR.PATCH tag"
            )
        return source
    if provider == "ollama":
        _expect_keys(
            source,
            required=frozenset({"provider", "repository", "tag", "digest"}),
            field=field,
        )
        repository = _text(source["repository"], field=f"{field}.repository")
        if not repository.startswith("library/"):
            raise RegistryValidationError(
                f"{field}.repository must include the Ollama library namespace"
            )
        _text(source["tag"], field=f"{field}.tag")
        _digest(source["digest"], field=f"{field}.digest")
        return source
    raise RegistryValidationError(f"{field}.provider is unsupported: {provider!r}")


def _validate_model(
    value: Any,
    *,
    index: int,
    hardware_ids: frozenset[str],
) -> Mapping[str, Any]:
    field = f"models[{index}]"
    model = _mapping(value, field=field)
    _expect_keys(
        model,
        required=frozenset(
            {
                "id",
                "displayName",
                "source",
                "license",
                "usage",
                "runtime",
                "quantization",
                "hardwareProfileIds",
                "local",
            }
        ),
        optional=frozenset({"deploymentSlots"}),
        field=field,
    )
    model_id = _text(model["id"], field=f"{field}.id")
    if not _ID_RE.fullmatch(model_id):
        raise RegistryValidationError(f"{field}.id has an invalid identifier")
    _text(model["displayName"], field=f"{field}.displayName")
    source = _validate_source(model["source"], field=f"{field}.source")

    license_value = _mapping(model["license"], field=f"{field}.license")
    _expect_keys(
        license_value,
        required=frozenset({"spdx", "evidencePath", "evidenceSha256"}),
        field=f"{field}.license",
    )
    spdx = _text(license_value["spdx"], field=f"{field}.license.spdx")
    if spdx.upper() in {"UNKNOWN", "NOASSERTION"} or not _SPDX_RE.fullmatch(spdx):
        raise RegistryValidationError(
            f"{field}.license.spdx must be a concrete SPDX identifier"
        )
    _relative_path(
        license_value["evidencePath"], field=f"{field}.license.evidencePath"
    )
    _sha256(
        license_value["evidenceSha256"],
        field=f"{field}.license.evidenceSha256",
    )

    usage = _mapping(model["usage"], field=f"{field}.usage")
    _expect_keys(
        usage,
        required=frozenset({"status", "roles", "decisionPolicy"}),
        field=f"{field}.usage",
    )
    status = _text(usage["status"], field=f"{field}.usage.status")
    if status not in MODEL_STATUSES:
        raise RegistryValidationError(
            f"{field}.usage.status is unsupported: {status!r}"
        )
    decision_policy = _text(
        usage["decisionPolicy"], field=f"{field}.usage.decisionPolicy"
    )
    if decision_policy not in DECISION_POLICIES:
        raise RegistryValidationError(
            f"{field}.usage.decisionPolicy is unsupported: {decision_policy!r}"
        )
    roles = _list(usage["roles"], field=f"{field}.usage.roles")
    checked_roles: list[str] = []
    for role_index, raw_role in enumerate(roles):
        role = _text(raw_role, field=f"{field}.usage.roles[{role_index}]")
        if not _ROLE_RE.fullmatch(role):
            raise RegistryValidationError(
                f"{field}.usage.roles[{role_index}] has an invalid role"
            )
        checked_roles.append(role)
    if len({role.casefold() for role in checked_roles}) != len(checked_roles):
        raise RegistryValidationError(f"{field}.usage.roles contains duplicates")

    deployment_slots = _list(
        model.get("deploymentSlots", []),
        field=f"{field}.deploymentSlots",
        nonempty=False,
    )
    checked_slots: list[str] = []
    for slot_index, raw_slot in enumerate(deployment_slots):
        slot_field = f"{field}.deploymentSlots[{slot_index}]"
        slot = _mapping(raw_slot, field=slot_field)
        _expect_keys(
            slot,
            required=frozenset({"id", "adapterIds"}),
            field=slot_field,
        )
        slot_id = _text(slot["id"], field=f"{slot_field}.id")
        if not _ROLE_RE.fullmatch(slot_id):
            raise RegistryValidationError(
                f"{slot_field}.id has an invalid deployment slot"
            )
        adapter_ids = _list(
            slot["adapterIds"], field=f"{slot_field}.adapterIds"
        )
        checked_adapters: list[str] = []
        for adapter_index, raw_adapter in enumerate(adapter_ids):
            adapter_id = _text(
                raw_adapter,
                field=f"{slot_field}.adapterIds[{adapter_index}]",
            )
            if not _ID_RE.fullmatch(adapter_id):
                raise RegistryValidationError(
                    f"{slot_field}.adapterIds[{adapter_index}] has an invalid identifier"
                )
            checked_adapters.append(adapter_id)
        if len(set(checked_adapters)) != len(checked_adapters):
            raise RegistryValidationError(
                f"{slot_field}.adapterIds contains duplicates"
            )
        checked_slots.append(slot_id)
    if len(set(checked_slots)) != len(checked_slots):
        raise RegistryValidationError(
            f"{field}.deploymentSlots contains duplicate slots"
        )

    runtime = _mapping(model["runtime"], field=f"{field}.runtime")
    _expect_keys(
        runtime,
        required=frozenset(
            {"engine", "version", "executablePath", "accelerator", "artifactFormat"}
        ),
        field=f"{field}.runtime",
    )
    _text(runtime["engine"], field=f"{field}.runtime.engine")
    _text(runtime["version"], field=f"{field}.runtime.version")
    _absolute_local_path(
        runtime["executablePath"], field=f"{field}.runtime.executablePath"
    )
    accelerator = _text(
        runtime["accelerator"], field=f"{field}.runtime.accelerator"
    )
    if not _ACCELERATOR_RE.fullmatch(accelerator):
        raise RegistryValidationError(
            f"{field}.runtime.accelerator is unsupported: {accelerator!r}"
        )
    artifact_format = _text(
        runtime["artifactFormat"], field=f"{field}.runtime.artifactFormat"
    )
    if artifact_format not in ARTIFACT_FORMATS:
        raise RegistryValidationError(
            f"{field}.runtime.artifactFormat is unsupported: {artifact_format!r}"
        )

    quantization = _mapping(
        model["quantization"], field=f"{field}.quantization"
    )
    _expect_keys(
        quantization,
        required=frozenset({"scheme", "evidencePath"}),
        field=f"{field}.quantization",
    )
    scheme = _text(
        quantization["scheme"], field=f"{field}.quantization.scheme"
    )
    if scheme not in QUANTIZATION_SCHEMES:
        raise RegistryValidationError(
            f"{field}.quantization.scheme is unsupported: {scheme!r}"
        )
    _relative_path(
        quantization["evidencePath"],
        field=f"{field}.quantization.evidencePath",
    )
    if scheme != "not-declared" and artifact_format != "gguf":
        raise RegistryValidationError(
            f"{field} declares quantization for a non-GGUF artifact"
        )

    model_hardware = _list(
        model["hardwareProfileIds"], field=f"{field}.hardwareProfileIds"
    )
    checked_hardware: list[str] = []
    for hardware_index, raw_id in enumerate(model_hardware):
        hardware_id = _text(
            raw_id, field=f"{field}.hardwareProfileIds[{hardware_index}]"
        )
        if hardware_id not in hardware_ids:
            raise RegistryValidationError(
                f"{field}.hardwareProfileIds references unknown profile {hardware_id!r}"
            )
        checked_hardware.append(hardware_id)
    if len(set(checked_hardware)) != len(checked_hardware):
        raise RegistryValidationError(
            f"{field}.hardwareProfileIds contains duplicates"
        )

    local = _mapping(model["local"], field=f"{field}.local")
    _expect_keys(
        local,
        required=frozenset({"path", "manifest"}),
        field=f"{field}.local",
    )
    _absolute_local_path(local["path"], field=f"{field}.local.path")
    manifest = _mapping(local["manifest"], field=f"{field}.local.manifest")
    _expect_keys(
        manifest,
        required=frozenset(
            {"format", "path", "sha256", "fileCount", "totalBytes"}
        ),
        field=f"{field}.local.manifest",
    )
    manifest_format = _text(
        manifest["format"], field=f"{field}.local.manifest.format"
    )
    if manifest_format not in MANIFEST_FORMATS:
        raise RegistryValidationError(
            f"{field}.local.manifest.format is unsupported: {manifest_format!r}"
        )
    manifest_path = _relative_path(
        manifest["path"], field=f"{field}.local.manifest.path"
    )
    manifest_sha = _sha256(
        manifest["sha256"], field=f"{field}.local.manifest.sha256"
    )
    _positive_int(
        manifest["fileCount"], field=f"{field}.local.manifest.fileCount"
    )
    _positive_int(
        manifest["totalBytes"], field=f"{field}.local.manifest.totalBytes"
    )
    if source["provider"] == "ollama":
        if manifest_format != "ollama-oci-manifest-v2":
            raise RegistryValidationError(
                f"{field} Ollama source requires an Ollama OCI manifest"
            )
        expected_manifest_path = (
            "manifests/registry.ollama.ai/"
            f"{source['repository']}/{source['tag']}"
        )
        if manifest_path != expected_manifest_path:
            raise RegistryValidationError(
                f"{field}.local.manifest.path does not match the Ollama tag"
            )
        if source["digest"] != f"sha256:{manifest_sha}":
            raise RegistryValidationError(
                f"{field}.source.digest does not match local manifest SHA-256"
            )
    elif manifest_format != "mts-model-manifest-v1":
        raise RegistryValidationError(
            f"{field} non-Ollama source requires an MTS model manifest"
        )
    return model


def _hash_file(path: Path, cache: dict[Path, str]) -> str:
    cached = cache.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RegistryValidationError(f"cannot hash local file: {path}") from error
    value = digest.hexdigest()
    cache[path] = value
    return value


def _member_path(root: Path, relative: str, *, field: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RegistryValidationError(f"{field} does not exist: {candidate}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RegistryValidationError(f"{field} resolves outside the model root") from error
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise RegistryValidationError(f"{field} traverses a symbolic link")
    if not resolved.is_file():
        raise RegistryValidationError(f"{field} must be a file: {resolved}")
    return resolved


def _verify_inventory(
    *,
    model_id: str,
    root: Path,
    raw_files: Any,
    expected_count: int,
    expected_total: int,
    cache: dict[Path, str],
) -> None:
    files = _list(raw_files, field=f"{model_id}.manifest.files")
    seen: set[str] = set()
    total = 0
    checked: list[tuple[str, int, str, Path]] = []
    for index, raw_entry in enumerate(files):
        field = f"{model_id}.manifest.files[{index}]"
        entry = _mapping(raw_entry, field=field)
        relative = _relative_path(entry.get("path"), field=f"{field}.path")
        folded = relative.casefold()
        if folded in seen:
            raise RegistryValidationError(
                f"{model_id} manifest repeats file path: {relative}"
            )
        seen.add(folded)
        size = _positive_int(entry.get("size"), field=f"{field}.size")
        sha = _sha256(entry.get("sha256"), field=f"{field}.sha256")
        path = _member_path(root, relative, field=f"{field}.path")
        checked.append((relative, size, sha, path))
        total += size
    if len(checked) != expected_count:
        raise RegistryValidationError(
            f"{model_id} fileCount mismatch: expected {expected_count}, got {len(checked)}"
        )
    if total != expected_total:
        raise RegistryValidationError(
            f"{model_id} totalBytes mismatch: expected {expected_total}, got {total}"
        )
    for relative, expected_size, expected_sha, path in checked:
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RegistryValidationError(
                f"{model_id} size mismatch for {relative}: "
                f"expected {expected_size}, got {actual_size}"
            )
        actual_sha = _hash_file(path, cache)
        if actual_sha != expected_sha:
            raise RegistryValidationError(
                f"{model_id} SHA-256 mismatch for {relative}: "
                f"expected {expected_sha}, got {actual_sha}"
            )


def _verify_mts_manifest(
    *,
    model: Mapping[str, Any],
    root: Path,
    manifest: Mapping[str, Any],
    cache: dict[Path, str],
) -> None:
    model_id = str(model["id"])
    source = _mapping(model["source"], field=f"{model_id}.source")
    schema_version = manifest.get("schemaVersion")
    if schema_version not in {"1.0.0", "1.1.0"}:
        raise RegistryValidationError(
            f"{model_id} local manifest has unsupported schemaVersion"
        )
    for manifest_key, source_key in (
        ("provider", "provider"),
        ("repoId", "repository"),
        ("revision", "revision"),
    ):
        if manifest.get(manifest_key) != source[source_key]:
            raise RegistryValidationError(
                f"{model_id} local manifest {manifest_key} does not match registry source"
            )
    if schema_version == "1.1.0":
        if manifest.get("kind") != "derived-safetensors-reshard":
            raise RegistryValidationError(
                f"{model_id} schemaVersion 1.1.0 manifest is not a safetensors reshard"
            )
        reshard = _mapping(
            manifest.get("reshard"), field=f"{model_id}.manifest.reshard"
        )
        if (
            reshard.get("exactTensorBytesPreserved") is not True
            or reshard.get("modelQualityChanged") is not False
        ):
            raise RegistryValidationError(
                f"{model_id} reshard does not prove byte-preserving model quality"
            )
        source_evidence = _mapping(
            reshard.get("source"),
            field=f"{model_id}.manifest.reshard.source",
        )
        for manifest_key, source_key in (
            ("provider", "provider"),
            ("repoId", "repository"),
            ("revision", "revision"),
        ):
            if source_evidence.get(manifest_key) != source[source_key]:
                raise RegistryValidationError(
                    f"{model_id} reshard source {manifest_key} does not match registry"
                )
    identity = _mapping(
        _mapping(model["local"], field=f"{model_id}.local")["manifest"],
        field=f"{model_id}.local.manifest",
    )
    expected_count = int(identity["fileCount"])
    expected_total = int(identity["totalBytes"])
    if manifest.get("fileCount", expected_count) != expected_count:
        raise RegistryValidationError(
            f"{model_id} local manifest fileCount does not match registry"
        )
    if manifest.get("totalBytes") != expected_total:
        raise RegistryValidationError(
            f"{model_id} local manifest totalBytes does not match registry"
        )
    _verify_inventory(
        model_id=model_id,
        root=root,
        raw_files=manifest.get("files"),
        expected_count=expected_count,
        expected_total=expected_total,
        cache=cache,
    )


def _ollama_blob_path(root: Path, digest: str, *, field: str) -> Path:
    checked = _digest(digest, field=field)
    return _member_path(
        root,
        f"blobs/sha256-{checked.removeprefix('sha256:')}",
        field=field,
    )


def _verify_ollama_manifest(
    *,
    model: Mapping[str, Any],
    root: Path,
    manifest: Mapping[str, Any],
    cache: dict[Path, str],
) -> None:
    model_id = str(model["id"])
    if manifest.get("schemaVersion") != 2:
        raise RegistryValidationError(
            f"{model_id} Ollama manifest must use schemaVersion 2"
        )
    config = _mapping(manifest.get("config"), field=f"{model_id}.manifest.config")
    layers = _list(manifest.get("layers"), field=f"{model_id}.manifest.layers")
    descriptors = [config, *layers]
    identity = _mapping(
        _mapping(model["local"], field=f"{model_id}.local")["manifest"],
        field=f"{model_id}.local.manifest",
    )
    if len(descriptors) != identity["fileCount"]:
        raise RegistryValidationError(
            f"{model_id} Ollama descriptor count does not match registry"
        )
    total = 0
    config_path: Path | None = None
    license_paths: set[str] = set()
    for index, descriptor in enumerate(descriptors):
        field = f"{model_id}.manifest.descriptors[{index}]"
        digest = _digest(descriptor.get("digest"), field=f"{field}.digest")
        size = _positive_int(descriptor.get("size"), field=f"{field}.size")
        path = _ollama_blob_path(root, digest, field=f"{field}.digest")
        actual_size = path.stat().st_size
        if actual_size != size:
            raise RegistryValidationError(
                f"{model_id} Ollama blob size mismatch for {digest}: "
                f"expected {size}, got {actual_size}"
            )
        if _hash_file(path, cache) != digest.removeprefix("sha256:"):
            raise RegistryValidationError(
                f"{model_id} Ollama blob SHA-256 mismatch for {digest}"
            )
        total += size
        if index == 0:
            config_path = path
        if descriptor.get("mediaType") == "application/vnd.ollama.image.license":
            license_paths.add(
                f"blobs/sha256-{digest.removeprefix('sha256:')}"
            )
    if total != identity["totalBytes"]:
        raise RegistryValidationError(
            f"{model_id} Ollama totalBytes mismatch: "
            f"expected {identity['totalBytes']}, got {total}"
        )
    license_value = _mapping(model["license"], field=f"{model_id}.license")
    if license_value["evidencePath"] not in license_paths:
        raise RegistryValidationError(
            f"{model_id} license evidence is not an Ollama license layer"
        )
    assert config_path is not None
    config_document = _read_json(config_path, label=f"{model_id} Ollama config")
    quantization = _mapping(
        model["quantization"], field=f"{model_id}.quantization"
    )
    runtime = _mapping(model["runtime"], field=f"{model_id}.runtime")
    if config_document.get("file_type") != quantization["scheme"]:
        raise RegistryValidationError(
            f"{model_id} quantization does not match the Ollama config"
        )
    if config_document.get("model_format") != runtime["artifactFormat"]:
        raise RegistryValidationError(
            f"{model_id} artifact format does not match the Ollama config"
        )


def _verify_local_model(
    model: Mapping[str, Any], *, cache: dict[Path, str]
) -> None:
    model_id = str(model["id"])
    local = _mapping(model["local"], field=f"{model_id}.local")
    local_path = _host_path(str(local["path"]))
    if local_path.is_symlink():
        raise RegistryValidationError(f"{model_id} local root must not be a symlink")
    try:
        root = local_path.resolve(strict=True)
    except OSError as error:
        raise RegistryValidationError(
            f"{model_id} local root does not exist: {local_path}"
        ) from error
    if not root.is_dir():
        raise RegistryValidationError(f"{model_id} local root is not a directory")

    runtime = _mapping(model["runtime"], field=f"{model_id}.runtime")
    executable = _host_path(str(runtime["executablePath"]))
    if executable.is_symlink() or not executable.is_file():
        raise RegistryValidationError(
            f"{model_id} runtime executable is missing or symbolic: {executable}"
        )

    identity = _mapping(local["manifest"], field=f"{model_id}.local.manifest")
    manifest_path = _member_path(
        root,
        str(identity["path"]),
        field=f"{model_id}.local.manifest.path",
    )
    actual_manifest_sha = _hash_file(manifest_path, cache)
    if actual_manifest_sha != identity["sha256"]:
        raise RegistryValidationError(
            f"{model_id} local manifest SHA-256 mismatch: "
            f"expected {identity['sha256']}, got {actual_manifest_sha}"
        )
    manifest = _read_json(manifest_path, label=f"{model_id} local manifest")

    license_value = _mapping(model["license"], field=f"{model_id}.license")
    license_path = _member_path(
        root,
        str(license_value["evidencePath"]),
        field=f"{model_id}.license.evidencePath",
    )
    actual_license_sha = _hash_file(license_path, cache)
    if actual_license_sha != license_value["evidenceSha256"]:
        raise RegistryValidationError(
            f"{model_id} license evidence SHA-256 mismatch"
        )
    quantization = _mapping(
        model["quantization"], field=f"{model_id}.quantization"
    )
    _member_path(
        root,
        str(quantization["evidencePath"]),
        field=f"{model_id}.quantization.evidencePath",
    )

    if identity["format"] == "mts-model-manifest-v1":
        _verify_mts_manifest(
            model=model,
            root=root,
            manifest=manifest,
            cache=cache,
        )
    else:
        _verify_ollama_manifest(
            model=model,
            root=root,
            manifest=manifest,
            cache=cache,
        )


def validate_registry(
    document: Mapping[str, Any],
    *,
    verify_local: bool = False,
    model_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate schema and optionally verify all selected local artifacts."""

    _expect_keys(
        document,
        required=frozenset(
            {"schemaVersion", "registryId", "hardwareProfiles", "models"}
        ),
        field="registry",
    )
    if document.get("schemaVersion") != SCHEMA_VERSION:
        raise RegistryValidationError(
            f"unsupported registry schemaVersion={document.get('schemaVersion')!r}"
        )
    registry_id = _text(document["registryId"], field="registry.registryId")
    if not _ID_RE.fullmatch(registry_id):
        raise RegistryValidationError("registry.registryId has an invalid identifier")

    raw_profiles = _list(
        document["hardwareProfiles"], field="registry.hardwareProfiles"
    )
    profile_ids = [
        _validate_hardware_profile(value, index=index)
        for index, value in enumerate(raw_profiles)
    ]
    if len({value.casefold() for value in profile_ids}) != len(profile_ids):
        raise RegistryValidationError("registry contains duplicate hardware profile IDs")
    known_hardware = frozenset(profile_ids)

    raw_models = _list(document["models"], field="registry.models")
    models = [
        _validate_model(value, index=index, hardware_ids=known_hardware)
        for index, value in enumerate(raw_models)
    ]
    ids = [str(model["id"]) for model in models]
    if len({value.casefold() for value in ids}) != len(ids):
        raise RegistryValidationError("registry contains duplicate model IDs")
    sources = [
        json.dumps(model["source"], sort_keys=True, separators=(",", ":"))
        for model in models
    ]
    if len(set(sources)) != len(sources):
        raise RegistryValidationError("registry contains duplicate source identities")
    local_identities: list[tuple[str, str]] = []
    models_by_local_path: dict[str, list[Mapping[str, Any]]] = {}
    for model in models:
        local = _mapping(model["local"], field="model.local")
        manifest = _mapping(local["manifest"], field="model.local.manifest")
        local_path = str(local["path"]).casefold()
        manifest_path = str(manifest["path"]).casefold()
        local_identities.append((local_path, manifest_path))
        models_by_local_path.setdefault(local_path, []).append(model)
    if len(set(local_identities)) != len(local_identities):
        raise RegistryValidationError(
            "registry contains duplicate local manifest identities"
        )
    for shared_models in models_by_local_path.values():
        if len(shared_models) < 2:
            continue
        providers = {
            str(_mapping(model["source"], field="model.source")["provider"])
            for model in shared_models
        }
        if providers != {"ollama"}:
            raise RegistryValidationError(
                "registry contains duplicate non-Ollama local model paths"
            )

    requested = tuple(dict.fromkeys(model_ids or ids))
    unknown = sorted(set(requested) - set(ids))
    if unknown:
        raise RegistryValidationError(
            "unknown model ID(s): "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(sorted(ids))
        )
    if verify_local:
        by_id = {str(model["id"]): model for model in models}
        hash_cache: dict[Path, str] = {}
        for model_id in requested:
            _verify_local_model(by_id[model_id], cache=hash_cache)
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="registry JSON (default: repository local-model-registry.json)",
    )
    parser.add_argument(
        "--verify-local",
        action="store_true",
        help="hash manifests, license evidence, model files, and Ollama blobs",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="verify one model ID; repeat to select multiple (default: all)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        document = load_registry(arguments.registry)
        selected = validate_registry(
            document,
            verify_local=arguments.verify_local,
            model_ids=arguments.models,
        )
    except (OSError, RegistryValidationError) as error:
        print(f"model registry validation failed: {error}", file=sys.stderr)
        return 2
    models = _list(document["models"], field="registry.models")
    status_counts: dict[str, int] = {}
    for model in models:
        status = str(
            _mapping(
                _mapping(model, field="model")["usage"], field="model.usage"
            )["status"]
        )
        status_counts[status] = status_counts.get(status, 0) + 1
    print(
        json.dumps(
            {
                "registry": str(arguments.registry.resolve()),
                "schemaVersion": document["schemaVersion"],
                "modelCount": len(models),
                "selectedModelCount": len(selected),
                "verifyLocal": arguments.verify_local,
                "statuses": dict(sorted(status_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
