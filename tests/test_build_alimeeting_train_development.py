from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from tools.build_alimeeting_train_development import (
    DEFAULT_ARCHIVE_CRC64,
    TrainFreezeError,
    _extract_selected_members,
    _normalize_train_textgrid_speaker_ids,
    _safe_member_name,
    _validate_held_out_isolation,
)


def _add_file(handle: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    handle.addfile(member, io.BytesIO(payload))


def test_official_archive_crc64_is_frozen() -> None:
    assert DEFAULT_ARCHIVE_CRC64 == "7416259116226311466"


def test_train_textgrid_speaker_ids_are_strictly_normalized() -> None:
    grid = {
        "tiers": [
            {"name": "R0003_M0046_F_SPK0093", "intervals": []},
            {"name": "R0003_M0046_M_SPK0094", "intervals": []},
        ]
    }

    normalized, mapping = _normalize_train_textgrid_speaker_ids(
        grid,
        session_id="R0003_M0046",
    )

    assert [tier["name"] for tier in normalized["tiers"]] == [
        "N_SPK0093",
        "N_SPK0094",
    ]
    assert mapping == {
        "R0003_M0046_F_SPK0093": "N_SPK0093",
        "R0003_M0046_M_SPK0094": "N_SPK0094",
    }
    with pytest.raises(TrainFreezeError, match="tier speaker ID is invalid"):
        _normalize_train_textgrid_speaker_ids(
            {"tiers": [{"name": "R0003_M0047_F_SPK0093"}]},
            session_id="R0003_M0046",
        )


def test_safe_member_name_rejects_links_and_path_traversal() -> None:
    traversal = tarfile.TarInfo("Train_Ali_far/../outside.wav")
    with pytest.raises(TrainFreezeError, match="unsafe archive member"):
        _safe_member_name(traversal)

    symlink = tarfile.TarInfo("Train_Ali_far/audio_dir/link.wav")
    symlink.type = tarfile.SYMTYPE
    symlink.linkname = "/tmp/outside.wav"
    with pytest.raises(TrainFreezeError, match="unsafe archive member"):
        _safe_member_name(symlink)


def test_extract_selected_members_streams_only_requested_session(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "Train_Ali_far.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as handle:
        _add_file(
            handle,
            "Train_Ali_far/audio_dir/R0003_M0046_MS002.wav",
            b"selected-audio",
        )
        _add_file(
            handle,
            "Train_Ali_far/audio_dir/R0003_M0047_MS006.wav",
            b"other-audio",
        )
        _add_file(
            handle,
            "Train_Ali_far/textgrid_dir/R0003_M0046.TextGrid",
            b"selected-grid",
        )

    audio, textgrid, evidence = _extract_selected_members(
        archive,
        session_id="R0003_M0046",
        destination=tmp_path / "sources",
    )

    assert audio.read_bytes() == b"selected-audio"
    assert textgrid.read_bytes() == b"selected-grid"
    assert evidence == {
        "membersScanned": 3,
        "audioMember": "Train_Ali_far/audio_dir/R0003_M0046_MS002.wav",
        "textgridMember": "Train_Ali_far/textgrid_dir/R0003_M0046.TextGrid",
    }
    assert not (audio.parent / "R0003_M0047_MS006.wav").exists()


def test_extract_selected_members_can_resume_only_explicit_clean_extraction(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "unused-after-safe-extraction.tar.gz"
    archive.write_bytes(b"placeholder")
    final = tmp_path / "sources" / "R0003_M0046"
    (final / "audio_dir").mkdir(parents=True)
    (final / "textgrid_dir").mkdir(parents=True)
    audio = final / "audio_dir" / "R0003_M0046_MS002.wav"
    textgrid = final / "textgrid_dir" / "R0003_M0046.TextGrid"
    audio.write_bytes(b"selected-audio")
    textgrid.write_bytes(b"selected-grid")

    with pytest.raises(TrainFreezeError, match="refusing to overwrite"):
        _extract_selected_members(
            archive,
            session_id="R0003_M0046",
            destination=tmp_path / "sources",
        )
    resumed_audio, resumed_grid, evidence = _extract_selected_members(
        archive,
        session_id="R0003_M0046",
        destination=tmp_path / "sources",
        reuse_existing=True,
    )

    assert resumed_audio == audio
    assert resumed_grid == textgrid
    assert evidence["reusedExistingExtraction"] is True

    (final / "unexpected.bin").write_bytes(b"reject")
    with pytest.raises(TrainFreezeError, match="unexpected member"):
        _extract_selected_members(
            archive,
            session_id="R0003_M0046",
            destination=tmp_path / "sources",
            reuse_existing=True,
        )


def test_extract_selected_members_cleans_staging_after_unsafe_member(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        _add_file(
            handle,
            "Train_Ali_far/audio_dir/R0003_M0046_MS002.wav",
            b"selected-audio",
        )
        _add_file(handle, "Train_Ali_far/../outside", b"unsafe")

    destination = tmp_path / "sources"
    with pytest.raises(TrainFreezeError, match="unsafe archive member"):
        _extract_selected_members(
            archive,
            session_id="R0003_M0046",
            destination=destination,
        )

    assert not (destination / ".R0003_M0046.part").exists()
    assert not (destination / "R0003_M0046").exists()


def test_held_out_isolation_binds_archive_session_source_and_media(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "held-out.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [{"archive": {"sha256": "a" * 64}}],
                "cases": [
                    {
                        "sourceSessionId": "R8001_M8004",
                        "sha256": "b" * 64,
                        "sourceAudioArtifacts": [{"sha256": "c" * 64}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    evidence = _validate_held_out_isolation(
        manifest,
        archive_sha256="d" * 64,
        session_id="R0003_M0046",
        source_audio_sha256="e" * 64,
        media_sha256="f" * 64,
    )

    assert evidence["heldOutInventory"] == {
        "archiveSha256Count": 1,
        "sessionIdCount": 1,
        "sourceAudioSha256Count": 1,
        "mediaSha256Count": 1,
    }
    assert all(evidence["checks"].values())

    with pytest.raises(TrainFreezeError, match="sourceAudioSha256Disjoint"):
        _validate_held_out_isolation(
            manifest,
            archive_sha256="d" * 64,
            session_id="R0003_M0046",
            source_audio_sha256="c" * 64,
            media_sha256="f" * 64,
        )
