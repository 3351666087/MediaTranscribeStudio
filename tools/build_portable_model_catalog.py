"""Build a machine-independent model catalog from the audited local registry.

The local registry intentionally records the operator's installed paths and
file inventories. Those values are useful evidence but must not be copied into
an installer. This tool emits a portable catalog containing source identities,
roles, licenses, and relative download destinations only. It never copies
model artifacts and never reads secrets.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "local-model-registry.json"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "configs" / "model-catalog.v1.json"
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


class PortableCatalogError(ValueError):
    """Raised when the source registry cannot produce a safe catalog."""


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PortableCatalogError(f"cannot read model registry: {path}") from exc
    if not isinstance(value, Mapping):
        raise PortableCatalogError("model registry root must be an object")
    return value


def _relative_destination(local_path: Any, model_id: str) -> str:
    if not isinstance(local_path, str) or not local_path.strip():
        return f"models/{model_id}"
    value = local_path.strip().replace("\\", "/")
    marker = "/models/"
    marker_index = value.casefold().find(marker)
    if marker_index >= 0:
        suffix = value[marker_index + len(marker) :].strip("/")
        if suffix and all(part not in {".", ".."} for part in suffix.split("/")):
            return f"models/{suffix}"
    return f"models/{model_id}"


def _portable_model(raw: Mapping[str, Any]) -> dict[str, Any]:
    model_id = raw.get("id")
    display_name = raw.get("displayName")
    source = raw.get("source")
    usage = raw.get("usage")
    runtime = raw.get("runtime")
    if not isinstance(model_id, str) or not model_id.strip():
        raise PortableCatalogError("model id must be a non-empty string")
    if not isinstance(display_name, str) or not display_name.strip():
        raise PortableCatalogError(f"{model_id}: displayName is missing")
    if not isinstance(source, Mapping) or not isinstance(usage, Mapping):
        raise PortableCatalogError(f"{model_id}: source and usage are required")
    if not isinstance(runtime, Mapping):
        raise PortableCatalogError(f"{model_id}: runtime is required")

    # Keep only fields that identify and describe a download. ``local`` is the
    # deliberate boundary: it contains absolute paths and installed-file hashes.
    source_copy = copy.deepcopy(dict(source))
    runtime_copy = copy.deepcopy(dict(runtime))
    runtime_copy["executablePath"] = "runtime/media-asr/python"
    result: dict[str, Any] = {
        "id": model_id,
        "displayName": display_name,
        "source": source_copy,
        "license": copy.deepcopy(raw.get("license", {})),
        "usage": copy.deepcopy(dict(usage)),
        "runtime": runtime_copy,
        "quantization": copy.deepcopy(raw.get("quantization", {})),
        "hardwareProfileIds": list(raw.get("hardwareProfileIds", [])),
        "download": {
            "destinationRelative": _relative_destination(
                (raw.get("local") or {}).get("path")
                if isinstance(raw.get("local"), Mapping)
                else None,
                model_id,
            ),
            "requiresModelManager": True,
        },
    }
    if "deploymentSlots" in raw:
        result["deploymentSlots"] = copy.deepcopy(raw["deploymentSlots"])
    return result


def _assert_portable(value: Any, *, path: str = "$") -> None:
    if isinstance(value, str):
        if _WINDOWS_ABSOLUTE.match(value) or value.startswith(("/", "\\\\")):
            raise PortableCatalogError(f"portable catalog contains an absolute path at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_portable(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_portable(item, path=f"{path}[{index}]")


def build_catalog(registry: Mapping[str, Any]) -> dict[str, Any]:
    raw_models = registry.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise PortableCatalogError("model registry must contain a non-empty models array")
    models = [_portable_model(model) for model in raw_models if isinstance(model, Mapping)]
    if not models:
        raise PortableCatalogError("model registry contains no object models")

    raw_profiles = registry.get("hardwareProfiles", [])
    profiles: list[dict[str, Any]] = []
    if isinstance(raw_profiles, list):
        for profile in raw_profiles:
            if isinstance(profile, Mapping) and isinstance(profile.get("id"), str):
                # Platform is useful for filtering; CPU/GPU details are
                # workstation evidence and do not belong in an installer.
                profiles.append(
                    {
                        "id": profile["id"],
                        "platform": profile.get("platform", "unknown"),
                    }
                )

    production = [
        model
        for model in models
        if model.get("usage", {}).get("status") == "production"
    ]
    semantic = next(
        (
            model
            for model in production
            if "semantic-arbitration" in model.get("usage", {}).get("roles", [])
        ),
        None,
    )
    speaker = next(
        (
            model
            for model in production
            if "secondary-speaker-verification"
            in model.get("usage", {}).get("roles", [])
        ),
        None,
    )
    catalog: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "catalogId": "mediatranscribestudio.portable-model-catalog",
        "sourceRegistryId": registry.get("registryId", "unknown"),
        "artifactPolicy": {
            "bundledModelArtifacts": False,
            "absolutePathsIncluded": False,
        },
        "defaults": {
            "semanticModelId": semantic.get("id") if semantic else None,
            "speakerVerifierModelId": speaker.get("id") if speaker else None,
        },
        "hardwareProfiles": profiles,
        "models": models,
    }
    _assert_portable(catalog)
    return catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        catalog = build_catalog(_read_object(args.registry))
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "dryRun": True,
                        "output": str(args.output.resolve()),
                        "modelCount": len(catalog["models"]),
                        "bundledModelArtifacts": False,
                        "absolutePathsIncluded": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "output": str(args.output.resolve()),
                    "modelCount": len(catalog["models"]),
                    "bundledModelArtifacts": False,
                    "absolutePathsIncluded": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except PortableCatalogError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
