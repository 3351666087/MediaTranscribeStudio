"""Derive pinned product configs for every selected local LLM candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_registry import load_registry, validate_registry  # noqa: E402


class ProductConfigSetError(ValueError):
    """Raised when a registry entry cannot produce a product config."""


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProductConfigSetError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProductConfigSetError(f"{label} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductConfigSetError(f"{field} must be an object")
    return value


def _model_binding(model: Mapping[str, Any]) -> tuple[str, str]:
    model_id = str(model.get("id") or "")
    source = _mapping(model.get("source"), field=f"{model_id}.source")
    usage = _mapping(model.get("usage"), field=f"{model_id}.usage")
    roles = usage.get("roles")
    if source.get("provider") != "ollama":
        raise ProductConfigSetError(f"{model_id} is not an Ollama model")
    if not isinstance(roles, list) or "semantic-arbitration" not in roles:
        raise ProductConfigSetError(
            f"{model_id} is not registered for semantic arbitration"
        )
    repository = str(source.get("repository") or "")
    tag = str(source.get("tag") or "")
    digest = str(source.get("digest") or "")
    if not repository.startswith("library/") or not tag:
        raise ProductConfigSetError(f"{model_id} has no local Ollama tag")
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ProductConfigSetError(f"{model_id} has an invalid Ollama digest")
    model_name = f"{repository.removeprefix('library/')}:{tag}"
    return model_name, digest


def build_config_set(
    *,
    source_config_path: Path,
    registry_path: Path,
    output_directory: Path,
    model_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    source_path = source_config_path.resolve(strict=True)
    registry_file = registry_path.resolve(strict=True)
    source_config = _json_object(source_path, label="source config")
    registry = load_registry(registry_file)
    selected_ids = validate_registry(
        registry,
        verify_local=False,
        model_ids=model_ids,
    )
    models = {
        str(model["id"]): model
        for model in registry["models"]
        if isinstance(model, Mapping)
    }
    if model_ids is None:
        selected_ids = tuple(
            model_id
            for model_id in selected_ids
            if (
                isinstance(models[model_id].get("source"), Mapping)
                and models[model_id]["source"].get("provider") == "ollama"
                and "semantic-arbitration"
                in models[model_id].get("usage", {}).get("roles", [])
            )
        )
    if not selected_ids:
        raise ProductConfigSetError("no semantic Ollama models were selected")

    speaker = source_config.get("speaker")
    if not isinstance(speaker, dict):
        raise ProductConfigSetError("source config speaker must be an object")

    output_root = output_directory.resolve(strict=False)
    records: list[dict[str, Any]] = []
    for model_id in selected_ids:
        model_name, digest = _model_binding(models[model_id])
        generated = copy.deepcopy(source_config)
        generated_speaker = generated["speaker"]
        generated_speaker["localLlmModel"] = model_name
        generated_speaker["localLlmModelDigest"] = digest
        output_path = output_root / f"{model_id}.json"
        _write_json_atomic(output_path, generated)
        records.append(
            {
                "modelId": model_id,
                "model": model_name,
                "digest": digest,
                "configPath": output_path.name,
                "configFileSha256": _sha256_file(output_path),
                "configCanonicalSha256": _canonical_sha256(generated),
            }
        )

    manifest: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "artifactType": "local-llm-product-config-set",
        "sourceConfig": {
            "path": str(source_path),
            "fileSha256": _sha256_file(source_path),
            "canonicalSha256": _canonical_sha256(source_config),
        },
        "registry": {
            "path": str(registry_file),
            "fileSha256": _sha256_file(registry_file),
            "canonicalSha256": _canonical_sha256(registry),
        },
        "configs": records,
    }
    manifest["canonicalSha256"] = _canonical_sha256(manifest)
    _write_json_atomic(output_root / "config-set.manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--model", action="append", dest="models")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = build_config_set(
        source_config_path=arguments.source_config,
        registry_path=arguments.registry,
        output_directory=arguments.output_directory,
        model_ids=arguments.models,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
