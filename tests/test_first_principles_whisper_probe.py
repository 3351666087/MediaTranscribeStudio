from __future__ import annotations

from pathlib import Path

from tools.run_first_principles_whisper_probe import _initial_report


def test_whisper_report_records_pinned_revision_and_blind_contract(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    manifest = {"batchId": "batch"}
    blind = tmp_path / "blind.json"
    blind.write_text("{}", encoding="utf-8")
    report = _initial_report(
        blind_path=blind,
        blind_manifest=manifest,
        model_path=model,
        revision="a" * 40,
        maximum=3,
    )

    assert report["truthAccessed"] is False
    assert report["productionBackendImported"] is False
    assert report["probe"]["modelRevision"] == "a" * 40
    assert report["probe"]["maximumCases"] == 3
