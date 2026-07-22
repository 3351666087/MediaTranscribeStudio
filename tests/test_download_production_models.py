from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.download_production_models import (
    MANIFEST_NAME,
    ModelInstallError,
    assert_source_matches_lock,
    install_model,
    load_lock,
    resolve_target,
    verify_model,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_lock(path: Path, target: Path, files: dict[str, bytes]) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "provider": "modelscope",
                "models": [
                    {
                        "key": "demo",
                        "repoId": "example/demo",
                        "revision": "v1",
                        "target": str(target),
                        "totalBytes": sum(len(value) for value in files.values()),
                        "fileCount": len(files),
                        "sourceRevisions": ["fixture"],
                        "files": [
                            {
                                "path": name,
                                "size": len(payload),
                                "sha256": _sha256(payload),
                            }
                            for name, payload in files.items()
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class _FakeApi:
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    def get_model_files(self, *_args, **_kwargs):
        return [
            {
                "Type": "blob",
                "Path": path,
                "Size": len(payload),
                "Sha256": _sha256(payload),
            }
            for path, payload in self._files.items()
        ]


def test_load_lock_rejects_path_traversal(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, tmp_path / "target", {"../escape.bin": b"x"})
    with pytest.raises(ModelInstallError, match="unsafe locked file path"):
        load_lock(lock_path)


def test_source_inventory_drift_fails_closed(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, tmp_path / "target", {"model.bin": b"locked"})
    model = load_lock(lock_path).models[0]
    with pytest.raises(ModelInstallError, match="source drift"):
        assert_source_matches_lock(model, _FakeApi({"model.bin": b"changed"}))


def test_install_downloads_to_partial_then_publishes(tmp_path: Path) -> None:
    files = {"model.bin": b"weights", "nested/config.json": b"{}"}
    lock_path = tmp_path / "lock.json"
    target = tmp_path / "published"
    _write_lock(lock_path, target, files)
    lock = load_lock(lock_path)
    model = lock.models[0]
    observed_local_dirs: list[Path] = []

    def fake_download(_repo_id: str, **kwargs) -> str:
        local_dir = Path(kwargs["local_dir"])
        observed_local_dirs.append(local_dir)
        for name, payload in files.items():
            destination = local_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        assert not target.exists()
        return str(local_dir)

    outcome = install_model(
        model,
        target,
        lock_sha256=lock.sha256,
        workers=2,
        api=_FakeApi(files),
        downloader=fake_download,
        verify_source=True,
    )

    assert outcome == "installed"
    assert target.is_dir()
    assert observed_local_dirs == [tmp_path / "published.mts-download.partial"]
    assert not verify_model(model, target, workers=2)
    manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["modelKey"] == "demo"
    assert manifest["lockSha256"] == lock.sha256


def test_existing_corrupt_target_is_never_replaced(tmp_path: Path) -> None:
    files = {"model.bin": b"expected"}
    lock_path = tmp_path / "lock.json"
    target = tmp_path / "published"
    _write_lock(lock_path, target, files)
    target.mkdir()
    (target / "model.bin").write_bytes(b"corrupt")
    model = load_lock(lock_path).models[0]

    with pytest.raises(ModelInstallError, match="refusing destructive replacement"):
        install_model(
            model,
            target,
            lock_sha256="0" * 64,
            workers=1,
            api=_FakeApi(files),
            downloader=lambda *_args, **_kwargs: pytest.fail("must not download"),
            verify_source=True,
        )
    assert (target / "model.bin").read_bytes() == b"corrupt"


def test_target_root_override_is_deterministic(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, tmp_path / "ignored", {"model.bin": b"x"})
    model = load_lock(lock_path).models[0]
    assert resolve_target(model, tmp_path / "portable") == (
        tmp_path / "portable" / "demo"
    ).resolve()
