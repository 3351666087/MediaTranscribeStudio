from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.audit_hf_model_snapshot import audit_snapshot


def _git_blob_oid(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _tree(*, config: bytes, weights: bytes) -> list[dict[str, object]]:
    return [
        {
            "type": "file",
            "path": "config.json",
            "size": len(config),
            "oid": _git_blob_oid(config),
            "securityFileStatus": {"status": "safe"},
        },
        {
            "type": "file",
            "path": "model.safetensors",
            "size": len(weights),
            "oid": "1" * 40,
            "lfs": {
                "oid": hashlib.sha256(weights).hexdigest(),
                "size": len(weights),
            },
            "securityFileStatus": {"status": "safe"},
        },
    ]


def test_audit_verifies_git_and_lfs_files(tmp_path: Path) -> None:
    config = b'{"model_type":"fixture"}\n'
    weights = b"safe-tensors-fixture"
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "model.safetensors").write_bytes(weights)
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "download.metadata").write_text("ignored", encoding="utf-8")

    report = audit_snapshot(
        repo_id="org/model",
        revision="a" * 40,
        local_dir=tmp_path,
        tree=_tree(config=config, weights=weights),
    )

    assert report["fileCount"] == 2
    assert report["totalBytes"] == len(config) + len(weights)
    assert len(report["manifestSha256"]) == 64
    by_path = {row["path"]: row for row in report["files"]}
    assert by_path["config.json"]["verification"] == "git-blob-sha1"
    assert (
        by_path["model.safetensors"]["verification"] == "lfs-sha256"
    )


def test_audit_rejects_extra_file(tmp_path: Path) -> None:
    config = b"config"
    weights = b"weights"
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "model.safetensors").write_bytes(weights)
    (tmp_path / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="extra=.*unexpected.txt"):
        audit_snapshot(
            repo_id="org/model",
            revision="a" * 40,
            local_dir=tmp_path,
            tree=_tree(config=config, weights=weights),
        )


def test_audit_rejects_modified_lfs_file(tmp_path: Path) -> None:
    config = b"config"
    weights = b"weights"
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        audit_snapshot(
            repo_id="org/model",
            revision="a" * 40,
            local_dir=tmp_path,
            tree=_tree(config=config, weights=weights),
        )


def test_audit_rejects_modified_git_blob(tmp_path: Path) -> None:
    config = b"config"
    weights = b"weights"
    (tmp_path / "config.json").write_bytes(b"CONFIG")
    (tmp_path / "model.safetensors").write_bytes(weights)

    with pytest.raises(ValueError, match="Git blob mismatch"):
        audit_snapshot(
            repo_id="org/model",
            revision="a" * 40,
            local_dir=tmp_path,
            tree=_tree(config=config, weights=weights),
        )


def test_audit_rejects_snapshot_symlink(tmp_path: Path) -> None:
    config = b"config"
    weights = b"weights"
    external = tmp_path.parent / "external-config.json"
    external.write_bytes(config)
    (tmp_path / "config.json").symlink_to(external)
    (tmp_path / "model.safetensors").write_bytes(weights)

    with pytest.raises(ValueError, match="symbolic link"):
        audit_snapshot(
            repo_id="org/model",
            revision="a" * 40,
            local_dir=tmp_path,
            tree=_tree(config=config, weights=weights),
        )
