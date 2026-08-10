from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from backend.persistence import canonical_json_sha256, sha256_file
from tools import build_product_baseline_report as baseline


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_source_snapshot_binds_tracked_and_untracked_source_files(
    tmp_path: Path,
) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "config", "user.email", "baseline@example.invalid"),
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Baseline Fixture"),
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True
    )
    untracked = tmp_path / "tools" / "new.py"
    untracked.parent.mkdir()
    untracked.write_text("VALUE = 1\n", encoding="utf-8")

    snapshot = baseline.build_source_snapshot(
        tmp_path, source_roots=("tools",)
    )

    assert snapshot["fileCount"] == 2
    assert [row["path"] for row in snapshot["files"]] == [
        "tools/new.py",
        "tracked.txt",
    ]
    assert snapshot["filesCanonicalSha256"] == canonical_json_sha256(
        snapshot["files"]
    )


def test_artifact_evidence_records_file_and_declared_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    value = {
        "schemaVersion": "1.0.0",
        "libraryId": "fixture-library",
        "cases": [{"id": "a"}, {"id": "b"}],
        "counts": {"development": 1, "heldOut": 1},
    }
    _write_json(path, value)

    evidence = baseline._artifact_evidence(path)

    assert evidence["fileSha256"] == sha256_file(path)
    assert evidence["canonicalSha256"] == canonical_json_sha256(value)
    assert evidence["identity"]["libraryId"] == "fixture-library"
    assert evidence["counts"] == {
        "cases": 2,
        "declared.development": 1,
        "declared.heldOut": 1,
    }


def test_main_writes_non_replacing_report_and_sha256_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "baseline.json"
    fixture = {
        "schemaVersion": "1.0.0",
        "artifactType": "product-development-baseline",
        "models": {"modelCount": 3},
        "repository": {"sourceSnapshot": {"fileCount": 7}},
    }
    fixture["canonicalSha256"] = canonical_json_sha256(fixture)
    monkeypatch.setattr(baseline, "build_report", lambda **_kwargs: fixture)

    arguments = [
        "--production-config",
        str(tmp_path / "config.json"),
        "--output",
        str(output),
    ]
    assert baseline.main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == fixture
    sidecar = output.with_name(output.name + ".sha256")
    assert sidecar.read_text(encoding="ascii") == (
        f"{sha256_file(output)}  baseline.json\n"
    )
    assert baseline.main(arguments) == 2


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"counts": {}, "counts": {}}', encoding="utf-8")

    try:
        baseline._artifact_evidence(path)
    except baseline.BaselineCaptureError as error:
        assert "duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate JSON key was accepted")
