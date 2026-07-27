from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.maintain_local_storage import (
    MINIMUM_ALLOWED_FREE_BYTES,
    StoragePolicyError,
    load_policy,
    maintain_storage,
    parse_ollama_list,
    scratch_candidates,
)


def _write_policy(root: Path, *, config_model: str = "qwen3.5:9b") -> Path:
    (root / "production.config.json").write_text(
        json.dumps({"speaker": {"localLlmModel": config_model}}),
        encoding="utf-8",
    )
    policy_path = root / "model-lifecycle.v1.json"
    policy_path.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "minimumFreeBytes": MINIMUM_ALLOWED_FREE_BYTES,
                "productionConfigPaths": [
                    {"path": "production.config.json", "required": True}
                ],
                "ollamaModels": [
                    {"name": "qwen3.5:9b", "status": "active"},
                    {"name": "qwen3.5:4b", "status": "retired"},
                ],
                "localArtifacts": [],
                "scratchPolicies": [{"path": ".runtime_cache/tmp", "maxAgeDays": 7}],
            }
        ),
        encoding="utf-8",
    )
    return policy_path


def _runner(installed: list[str], calls: list[list[str]]):
    def run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(command)
        if command == ["ollama", "list"]:
            rows = ["NAME ID SIZE MODIFIED"]
            rows.extend(f"{name} id 1 GB now" for name in installed)
            return subprocess.CompletedProcess(command, 0, "\n".join(rows), "")
        if command[:2] == ["ollama", "rm"]:
            installed.remove(command[2])
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    return run


def test_parse_ollama_list_preserves_exact_model_names() -> None:
    assert parse_ollama_list(
        "\nNAME          ID        SIZE\n"
        "qwen3.5:9b    abc       6.6 GB\n"
        "model:tag     def       1.0 GB\n"
    ) == ("qwen3.5:9b", "model:tag")


def test_policy_rejects_broad_or_under_age_scratch_roots(tmp_path: Path) -> None:
    policy_path = _write_policy(tmp_path)
    value = json.loads(policy_path.read_text(encoding="utf-8"))
    value["scratchPolicies"] = [{"path": ".", "maxAgeDays": 0}]
    policy_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(StoragePolicyError, match="safe project-relative"):
        load_policy(policy_path, project_root=tmp_path)


def test_retired_model_still_referenced_fails_closed(tmp_path: Path) -> None:
    policy_path = _write_policy(tmp_path, config_model="qwen3.5:4b")
    policy = load_policy(policy_path, project_root=tmp_path)
    calls: list[list[str]] = []

    with pytest.raises(StoragePolicyError, match="not active"):
        maintain_storage(
            policy,
            apply=True,
            runner=_runner(["qwen3.5:4b"], calls),
        )

    assert calls == []


def test_audit_never_removes_models_or_scratch(tmp_path: Path) -> None:
    policy = load_policy(_write_policy(tmp_path), project_root=tmp_path)
    scratch = tmp_path / ".runtime_cache" / "tmp" / "expired"
    scratch.mkdir(parents=True)
    payload = scratch / "cache.bin"
    payload.write_bytes(b"cache")
    old_ns = 1_000_000_000
    os.utime(payload, ns=(old_ns, old_ns))
    os.utime(scratch, ns=(old_ns, old_ns))
    calls: list[list[str]] = []

    report = maintain_storage(
        policy,
        apply=False,
        now_ns=old_ns + 8 * 86_400 * 1_000_000_000,
        runner=_runner(["qwen3.5:9b", "qwen3.5:4b"], calls),
    )

    assert report["retiredInstalledModelsBefore"] == ["qwen3.5:4b"]
    assert report["retiredInstalledModels"] == ["qwen3.5:4b"]
    assert len(report["scratchCandidatesBefore"]) == 1
    assert len(report["scratchCandidates"]) == 1
    assert scratch.is_dir()
    assert calls == [["ollama", "list"]]


def test_apply_removes_only_registered_retired_and_expired_scratch(
    tmp_path: Path,
) -> None:
    policy = load_policy(_write_policy(tmp_path), project_root=tmp_path)
    scratch_root = tmp_path / ".runtime_cache" / "tmp"
    expired = scratch_root / "expired"
    recent = scratch_root / "recent"
    expired.mkdir(parents=True)
    recent.mkdir()
    expired_file = expired / "cache.bin"
    recent_file = recent / "cache.bin"
    expired_file.write_bytes(b"old")
    recent_file.write_bytes(b"new")
    old_ns = 1_000_000_000
    now_ns = old_ns + 8 * 86_400 * 1_000_000_000
    for path in (expired_file, expired):
        os.utime(path, ns=(old_ns, old_ns))
    for path in (recent_file, recent, scratch_root):
        os.utime(path, ns=(now_ns, now_ns))
    installed = ["qwen3.5:9b", "qwen3.5:4b", "unmanaged:latest"]
    calls: list[list[str]] = []

    report = maintain_storage(
        policy,
        apply=True,
        now_ns=now_ns,
        runner=_runner(installed, calls),
    )

    assert report["removedOllamaModels"] == ["qwen3.5:4b"]
    assert report["removedScratchPaths"] == [str(expired)]
    assert report["retiredInstalledModelsBefore"] == ["qwen3.5:4b"]
    assert report["retiredInstalledModels"] == []
    assert report["scratchCandidates"] == []
    assert report["actionRequired"] is False
    assert report["unmanagedInstalledOllamaModels"] == ["unmanaged:latest"]
    assert installed == ["qwen3.5:9b", "unmanaged:latest"]
    assert not expired.exists()
    assert recent_file.read_bytes() == b"new"
    assert calls == [
        ["ollama", "list"],
        ["ollama", "rm", "qwen3.5:4b"],
    ]


def test_newest_descendant_prevents_early_directory_removal(tmp_path: Path) -> None:
    policy = load_policy(_write_policy(tmp_path), project_root=tmp_path)
    candidate = tmp_path / ".runtime_cache" / "tmp" / "mixed"
    candidate.mkdir(parents=True)
    child = candidate / "recent.bin"
    child.write_bytes(b"x")
    old_ns = 1_000_000_000
    now_ns = old_ns + 8 * 86_400 * 1_000_000_000
    os.utime(candidate, ns=(old_ns, old_ns))
    os.utime(child, ns=(now_ns, now_ns))

    assert scratch_candidates(policy.scratch_policies, now_ns=now_ns) == ()


def test_apply_removes_only_registered_retired_local_artifacts(
    tmp_path: Path,
) -> None:
    policy_path = _write_policy(tmp_path)
    app_support = tmp_path / "application-support"
    active = app_support / "venvs" / "active-runtime"
    retired = tmp_path / ".runtime_cache" / "venvs" / "retired-runtime"
    active.mkdir(parents=True)
    retired.mkdir(parents=True)
    (active / "keep.bin").write_bytes(b"active")
    (retired / "remove.bin").write_bytes(b"retired")
    value = json.loads(policy_path.read_text(encoding="utf-8"))
    value["localArtifacts"] = [
        {
            "id": "active-runtime",
            "root": "applicationSupport",
            "path": "venvs/active-runtime",
            "status": "active",
        },
        {
            "id": "retired-runtime",
            "root": "project",
            "path": ".runtime_cache/venvs/retired-runtime",
            "status": "retired",
        },
    ]
    policy_path.write_text(json.dumps(value), encoding="utf-8")
    installed = ["qwen3.5:9b"]
    calls: list[list[str]] = []

    report = maintain_storage(
        load_policy(
            policy_path,
            project_root=tmp_path,
            application_support_root=app_support,
        ),
        apply=True,
        runner=_runner(installed, calls),
    )

    assert report["retiredLocalArtifactsBefore"] == ["retired-runtime"]
    assert report["retiredLocalArtifacts"] == []
    assert report["removedLocalArtifacts"] == ["retired-runtime"]
    assert report["missingActiveLocalArtifacts"] == []
    assert report["plannedLocalArtifactReclaimBytes"] > 0
    assert report["actionRequired"] is False
    assert active.is_dir()
    assert not retired.exists()
