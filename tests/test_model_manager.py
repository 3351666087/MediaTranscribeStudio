from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.model_manager import (
    DownloadRequest,
    ModelManagerError,
    build_download_command,
    run_download,
)


def test_huggingface_download_is_revision_pinned_and_secret_free(tmp_path: Path) -> None:
    request = DownloadRequest(
        provider="huggingface",
        model="Qwen/Qwen3-ASR-1.7B",
        revision="a" * 40,
        destination=tmp_path / "qwen",
        token_env="MTS_HF_TOKEN",
    )

    command, _environment = build_download_command(request)
    receipt = run_download(request, dry_run=True)

    assert command == [
        "hf",
        "download",
        "Qwen/Qwen3-ASR-1.7B",
        "--revision",
        "a" * 40,
        "--local-dir",
        str((tmp_path / "qwen").resolve()),
    ]
    assert receipt["tokenEnvironmentVariable"] == "MTS_HF_TOKEN"
    assert "super-secret" not in json.dumps(receipt)


def test_huggingface_download_requires_a_pinned_revision(tmp_path: Path) -> None:
    with pytest.raises(ModelManagerError, match="pinned revision"):
        DownloadRequest(
            provider="huggingface",
            model="org/model",
            destination=tmp_path,
        )


def test_ollama_download_uses_configured_data_drive_and_no_shell(tmp_path: Path) -> None:
    request = DownloadRequest(
        provider="ollama",
        model="qwen3.5:27b-q4_K_M",
        destination=tmp_path / "ollama",
        endpoint="http://127.0.0.1:11434",
    )
    command, environment = build_download_command(request)

    assert command == ["ollama", "pull", "qwen3.5:27b-q4_K_M"]
    assert environment == {
        "OLLAMA_MODELS": str((tmp_path / "ollama").resolve()),
        "OLLAMA_HOST": "http://127.0.0.1:11434",
    }


def test_secret_alias_is_copied_only_to_child_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MTS_HF_TOKEN", "super-secret")
    request = DownloadRequest(
        provider="huggingface",
        model="org/model",
        revision="b" * 40,
        destination=tmp_path / "model",
        token_env="MTS_HF_TOKEN",
        executable=os.fspath(tmp_path / "hf"),
    )
    Path(request.executable or "").write_text("fixture", encoding="utf-8")
    captured = {}

    class Completed:
        returncode = 0

    def runner(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return Completed()

    receipt = run_download(request, runner=runner)

    assert captured["env"]["HF_TOKEN"] == "super-secret"
    assert "super-secret" not in json.dumps(receipt)
    assert receipt["status"] == "completed"


def test_rejects_traversal_bearing_model_name(tmp_path: Path) -> None:
    with pytest.raises(ModelManagerError, match="unsupported characters"):
        DownloadRequest(
            provider="ollama",
            model="../escape",
            destination=tmp_path,
        )
