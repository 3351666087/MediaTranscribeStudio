from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

from jsonschema import Draft202012Validator

from tools.build_portable_model_catalog import build_catalog


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "local-model-registry.json"
SCHEMA = ROOT / "configs" / "model-catalog.v1.schema.json"
CATALOG = ROOT / "configs" / "model-catalog.v1.json"
ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]|^/|^\\\\")


def _assert_no_absolute(value: object) -> None:
    if isinstance(value, str):
        assert ABSOLUTE.match(value) is None, value
    elif isinstance(value, dict):
        for item in value.values():
            _assert_no_absolute(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_absolute(item)


def test_generated_catalog_is_schema_valid_portable_and_weight_free() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(json.loads(SCHEMA.read_text(encoding="utf-8")))
    Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(catalog)
    _assert_no_absolute(catalog)
    assert catalog["artifactPolicy"] == {
        "bundledModelArtifacts": False,
        "absolutePathsIncluded": False,
    }
    assert all("local" not in model for model in catalog["models"])
    assert all("apiKey" not in json.dumps(model) for model in catalog["models"])


def test_catalog_generation_is_deterministic_and_strips_machine_identity() -> None:
    source = json.loads(REGISTRY.read_text(encoding="utf-8"))
    first = build_catalog(source)
    second = build_catalog(source)
    assert first == second
    assert first["models"]
    assert all(
        model["runtime"]["executablePath"] == "runtime/media-asr/python"
        for model in first["models"]
    )
    assert all(
        model["download"]["destinationRelative"].startswith("models/")
        for model in first["models"]
    )


def test_catalog_cli_dry_run_does_not_write(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_portable_model_catalog.py"),
            "--registry",
            str(REGISTRY),
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["bundledModelArtifacts"] is False
    assert not output.exists()
