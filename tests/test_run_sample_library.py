from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_sample_library


def _manifest(root: Path) -> Path:
    source = root / "sample.wav"
    source.write_bytes(b"not-used-by-the-mocked-runner")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "libraryId": "unknown-speaker-count",
                "cases": [
                    {
                        "id": "unknown",
                        "path": source.name,
                        "language": "auto",
                        "expectedSpeakerCount": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_auto_mode_accepts_case_without_reference_speaker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_case(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(run_sample_library, "_run_case", fake_run_case)
    exit_code = run_sample_library.main(
        [
            "--manifest",
            str(_manifest(tmp_path)),
            "--results-root",
            str(tmp_path / "results"),
            "--worker-output-root",
            str(tmp_path / "outputs"),
            "--case",
            "unknown",
            "--no-reuse-worker",
        ]
    )

    assert exit_code == 0
    assert captured["expected_speaker_count"] is None


def test_auto_language_mode_does_not_pass_reference_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["cases"][0]["language"] = "fr-FR"
    manifest.write_text(json.dumps(value), encoding="utf-8")

    def fake_run_case(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(run_sample_library, "_run_case", fake_run_case)
    exit_code = run_sample_library.main(
        [
            "--manifest",
            str(manifest),
            "--results-root",
            str(tmp_path / "results"),
            "--worker-output-root",
            str(tmp_path / "outputs"),
            "--language-mode",
            "auto",
            "--no-reuse-worker",
        ]
    )

    assert exit_code == 0
    assert captured["language"] == "auto"


@pytest.mark.parametrize("mode", ["manual", "hybrid"])
def test_reference_modes_reject_unknown_speaker_count(
    tmp_path: Path,
    mode: str,
) -> None:
    with pytest.raises(SystemExit, match="no reference expectedSpeakerCount"):
        run_sample_library.main(
            [
                "--manifest",
                str(_manifest(tmp_path)),
                "--results-root",
                str(tmp_path / "results"),
                "--worker-output-root",
                str(tmp_path / "outputs"),
                "--case",
                "unknown",
                "--speaker-count-mode",
                mode,
                "--no-reuse-worker",
            ]
        )
