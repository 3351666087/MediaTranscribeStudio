from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.stage_production_diarization_inputs import (
    DiarizationInputStageError,
    stage_inputs,
)


def _manifest(
    root: Path,
    *,
    name: str = "manifest.json",
    case_id: str = "case-1",
    digest: str | None = None,
) -> Path:
    audio = root / "audio"
    audio.mkdir(exist_ok=True)
    media = audio / f"{case_id}.wav"
    if not media.exists():
        media.write_bytes((case_id.encode("ascii") + b"\0") * 128)
    path = root / name
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": case_id,
                        "path": f"audio\\{case_id}.wav",
                        "bytes": media.stat().st_size,
                        "sha256": digest or sha256_file(media),
                        "evaluationSplit": "held-out",
                        "language": "en",
                        "expectedSpeakerCount": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_stage_inputs_creates_verified_hard_links_and_hash_bound_receipt(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    first = _manifest(source_root, case_id="case-1")
    second = _manifest(
        source_root,
        name="second.json",
        case_id="case-2",
    )
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    staging = allowed / "speaker-eval-r1"

    receipt = stage_inputs(
        manifests=[first, second],
        case_ids=["case-2", "case-1"],
        allowed_root=allowed,
        staging_directory=staging,
    )

    assert [row["caseId"] for row in receipt["cases"]] == [
        "case-2",
        "case-1",
    ]
    assert receipt["additionalMediaBytesAllocated"] == 0
    assert receipt["sourceBytes"] > 0
    for row in receipt["cases"]:
        source = Path(row["source"]["path"])
        staged = Path(row["staged"]["path"])
        assert os.path.samefile(source, staged)
        assert row["staged"]["hardLinkVerified"] is True
        assert sha256_file(staged) == row["source"]["sha256"]
    persisted = json.loads(
        (staging / "staging-receipt.v1.json").read_text(encoding="utf-8")
    )
    assert persisted == receipt
    body = {
        key: value for key, value in receipt.items() if key != "canonicalSha256"
    }
    assert receipt["canonicalSha256"] == canonical_json_sha256(body)


def test_digest_mismatch_fails_before_staging(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = _manifest(source, digest="0" * 64)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    staging = allowed / "speaker-eval-r1"

    with pytest.raises(DiarizationInputStageError, match="SHA-256 mismatch"):
        stage_inputs(
            manifests=[manifest],
            case_ids=["case-1"],
            allowed_root=allowed,
            staging_directory=staging,
        )

    assert not staging.exists()


def test_staging_must_be_new_and_inside_allowed_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = _manifest(source)
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    with pytest.raises(DiarizationInputStageError, match="descendant"):
        stage_inputs(
            manifests=[manifest],
            case_ids=["case-1"],
            allowed_root=allowed,
            staging_directory=tmp_path / "outside",
        )

    existing = allowed / "existing"
    existing.mkdir()
    with pytest.raises(DiarizationInputStageError, match="already exists"):
        stage_inputs(
            manifests=[manifest],
            case_ids=["case-1"],
            allowed_root=allowed,
            staging_directory=existing,
        )


def test_staging_rejects_parent_alias_outside_allowed_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = _manifest(source)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = allowed / "redirect"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable: {exc}")
        junction = subprocess.run(
            (
                "cmd.exe",
                "/d",
                "/c",
                "mklink",
                "/j",
                str(alias),
                str(outside),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if junction.returncode != 0:
            pytest.skip(
                "directory aliases are unavailable: "
                + junction.stderr.strip()
            )

    try:
        with pytest.raises(DiarizationInputStageError, match="descendant"):
            stage_inputs(
                manifests=[manifest],
                case_ids=["case-1"],
                allowed_root=allowed,
                staging_directory=alias / "speaker-eval-r1",
            )

        assert not (outside / "speaker-eval-r1").exists()
    finally:
        if os.path.lexists(alias):
            alias.rmdir()


def test_hard_link_failure_rolls_back_only_new_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    first = _manifest(source_root, case_id="case-1")
    second = _manifest(
        source_root,
        name="second.json",
        case_id="case-2",
    )
    source_paths = sorted((source_root / "audio").glob("*.wav"))
    source_hashes = {path: sha256_file(path) for path in source_paths}
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    staging = allowed / "speaker-eval-r1"
    original_link = os.link
    call_count = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated hard-link failure")
        original_link(source, destination)

    monkeypatch.setattr(os, "link", fail_second_link)

    with pytest.raises(OSError, match="simulated hard-link failure"):
        stage_inputs(
            manifests=[first, second],
            case_ids=["case-1", "case-2"],
            allowed_root=allowed,
            staging_directory=staging,
        )

    assert not staging.exists()
    assert source_hashes == {path: sha256_file(path) for path in source_paths}


def test_duplicate_case_across_manifests_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = _manifest(source, name="first.json")
    second = _manifest(source, name="second.json")
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    with pytest.raises(DiarizationInputStageError, match="duplicated"):
        stage_inputs(
            manifests=[first, second],
            case_ids=["case-1"],
            allowed_root=allowed,
            staging_directory=allowed / "speaker-eval-r1",
        )


def test_case_ids_are_path_safe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = _manifest(source)
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    with pytest.raises(DiarizationInputStageError, match="path-safe"):
        stage_inputs(
            manifests=[manifest],
            case_ids=["../case-1"],
            allowed_root=allowed,
            staging_directory=allowed / "speaker-eval-r1",
        )
