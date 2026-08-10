from __future__ import annotations

import io
import json
import os
import re
import wave
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import tools.freeze_ascend_code_switch_samples as freezer
from backend.persistence import canonical_json_sha256
from tools.freeze_ascend_code_switch_samples import (
    AscendCodeSwitchFreezeError,
    assert_truth_redacted,
    freeze,
    load_selection_lock,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "sample_library" / "ascend-code-switch-selection.v1.json"
ATTRIBUTION = ROOT / "sample_library" / "ascend-code-switch" / "ATTRIBUTION.md"
HELD_OUT_SENTINEL = "SEALED_HELD_OUT_REFERENCE_"
DEVELOPMENT_SENTINEL = "DEVELOPMENT_REFERENCE_"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _wav(duration_seconds: float) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * round(duration_seconds * 16_000))
    return payload.getvalue()


def _install_viewer_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    selection = load_selection_lock(SELECTION)
    by_locator = {
        (case["stableKey"]["split"], case["stableKey"]["rowIndex"]): case
        for case in selection["cases"]
    }

    def request_json(
        url: str,
        *,
        label: str,
        timeout: float = 120.0,
    ) -> dict[str, object]:
        assert label.startswith("ASCEND")
        parsed = urlparse(url)
        if parsed.path.endswith("/is-valid"):
            return {"viewer": True}
        if parsed.path.endswith("/splits"):
            return {
                "splits": [
                    {
                        "dataset": freezer.DATASET,
                        "config": freezer.CONFIG,
                        "split": "validation",
                    },
                    {
                        "dataset": freezer.DATASET,
                        "config": freezer.CONFIG,
                        "split": "test",
                    },
                ]
            }
        assert parsed.path.endswith("/rows")
        query = parse_qs(parsed.query)
        split = query["split"][0]
        row_index = int(query["offset"][0])
        case = by_locator[(split, row_index)]
        stable = case["stableKey"]
        prefix = (
            DEVELOPMENT_SENTINEL
            if case["evaluationSplit"] == "development"
            else HELD_OUT_SENTINEL
        )
        return {
            "rows": [
                {
                    "row_idx": row_index,
                    "row": {
                        "id": stable["id"],
                        "language": "mixed",
                        "original_speaker_id": stable["speaker"],
                        "session_id": stable["session"],
                        "topic": case["topic"],
                        "duration": case["durationSeconds"],
                        "transcription": prefix + case["id"],
                        "audio": [
                            {
                                "src": (
                                    "https://asset.invalid/"
                                    f"{freezer.REVISION}/--/{freezer.CONFIG}/"
                                    f"{split}/{row_index}/audio/audio.wav"
                                )
                            }
                        ],
                    },
                }
            ]
        }

    def request_bytes(
        url: str,
        *,
        label: str,
        timeout: float = 240.0,
        attempts: int = 5,
    ) -> bytes:
        assert label.startswith("ASCEND audio")
        assert timeout == 300.0
        assert attempts == 5
        match = re.search(r"/(validation|test)/(\d+)/audio/", url)
        assert match is not None
        split, raw_index = match.groups()
        case = by_locator[(split, int(raw_index))]
        return _wav(float(case["durationSeconds"]))

    monkeypatch.setattr(freezer, "_request_json", request_json)
    monkeypatch.setattr(freezer, "_request_bytes", request_bytes)


def test_checked_in_selection_pins_raw_ascend_and_required_coverage() -> None:
    value = load_selection_lock(SELECTION)

    assert value["source"] == {
        "dataset": "CAiRE/ASCEND",
        "revision": "737e9800ae31be9932ba8464c80366559bd28424",
        "config": "main",
        "license": "cc-by-sa-4.0",
        "homepage": "https://huggingface.co/datasets/CAiRE/ASCEND",
        "attributionPath": "ascend-code-switch/ATTRIBUTION.md",
    }
    assert Counter(case["evaluationSplit"] for case in value["cases"]) == {
        "development": 6,
        "held-out": 6,
    }
    speakers: set[int] = set()
    for evaluation_split, source_split in (
        ("development", "validation"),
        ("held-out", "test"),
    ):
        cases = [
            case
            for case in value["cases"]
            if case["evaluationSplit"] == evaluation_split
        ]
        assert {case["stableKey"]["split"] for case in cases} == {source_split}
        assert Counter(case["durationBucket"] for case in cases) == {
            "short": 2,
            "medium": 2,
            "long": 2,
        }
        assert len({case["topic"] for case in cases}) >= 3
        assert (
            len(
                {
                    (case["stableKey"]["speaker"], case["stableKey"]["session"])
                    for case in cases
                }
            )
            >= 3
        )
        speakers.update(case["stableKey"]["speaker"] for case in cases)
        for case in cases:
            assert set(case["stableKey"]) == {
                "revision",
                "config",
                "split",
                "id",
                "rowIndex",
                "speaker",
                "session",
            }
    assert len(speakers) >= 3
    assert value["selectionPolicy"]["heldOutSpeakerLimitation"] == {
        "availableDistinctSpeakersInTestMixed": 2,
        "selectedDistinctSpeakers": 2,
        "note": (
            "The pinned test/mixed source rows contain only two speaker "
            "identities, so a three-speaker held-out claim is impossible. "
            "Coverage is reported without extrapolation."
        ),
    }
    assert_truth_redacted(value)
    serialized = SELECTION.read_text(encoding="utf-8")
    assert "transcription" not in serialized.casefold()
    assert "referenceTranscript" not in serialized


def test_attribution_pins_revision_config_license_and_truth_policy() -> None:
    text = ATTRIBUTION.read_text(encoding="utf-8")

    assert "CAiRE/ASCEND" in text
    assert freezer.REVISION in text
    assert "`main`" in text
    assert "CC BY-SA 4.0" in text
    assert "isolated scorer vault" in text


def test_freeze_keeps_held_out_truth_only_in_separate_scorer_vault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_viewer_fixture(monkeypatch)
    output_root = tmp_path / "ordinary" / "ascend"
    vault_root = tmp_path / "sealed" / "ascend"

    result = freeze(
        selection_path=SELECTION,
        output_root=output_root,
        scorer_vault_root=vault_root,
    )

    manifest_path = output_root / freezer.PUBLIC_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "canonicalSha256"}
    assert manifest["canonicalSha256"] == canonical_json_sha256(body)
    assert result["manifest"] == str(manifest_path.resolve())
    assert result["developmentCases"] == 6
    assert result["heldOutCases"] == 6
    assert manifest["coverage"]["cases"] == 12
    assert manifest["coverage"]["distinctSpeakersOverall"] == 5
    assert manifest["coverage"]["speakerDisjointEvaluationSplits"] is True
    assert manifest["truthPersistencePolicy"] == {
        "ordinaryManifestContainsTranscript": False,
        "developmentReferencePersistedSeparately": True,
        "developmentReferenceMayEnterReviewerPacket": False,
        "heldOutTruthPersistedOnlyInIsolatedScorerVault": True,
        "heldOutTruthInOrdinaryManifest": False,
        "heldOutTruthInReviewerPacket": False,
        "scorerVaultPublishedBeforeOrdinaryManifest": True,
    }
    assert_truth_redacted(manifest)

    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in output_root.rglob("*")
        if path.is_file() and path.suffix != ".wav"
    )
    assert HELD_OUT_SENTINEL not in public_text
    assert DEVELOPMENT_SENTINEL in public_text

    vault_path = vault_root / freezer.SCORER_VAULT_NAME
    vault = json.loads(vault_path.read_text(encoding="utf-8"))
    vault_text = vault_path.read_text(encoding="utf-8")
    assert len(vault["cases"]) == 6
    assert {case["sourceKey"]["split"] for case in vault["cases"]} == {"test"}
    assert HELD_OUT_SENTINEL in vault_text
    assert DEVELOPMENT_SENTINEL not in vault_text
    assert not vault_root.is_relative_to(output_root)
    if os.name != "nt":
        assert vault_root.stat().st_mode & 0o777 == 0o700
        assert vault_path.stat().st_mode & 0o777 == 0o600


def test_viewer_row_rejects_asset_outside_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_selection_lock(SELECTION)["cases"][0]
    stable = case["stableKey"]

    def request_json(
        url: str,
        *,
        label: str,
        timeout: float = 120.0,
    ) -> dict[str, object]:
        return {
            "rows": [
                {
                    "row_idx": stable["rowIndex"],
                    "row": {
                        "id": stable["id"],
                        "duration": case["durationSeconds"],
                        "language": "mixed",
                        "original_speaker_id": stable["speaker"],
                        "session_id": stable["session"],
                        "topic": case["topic"],
                        "transcription": "development fixture",
                        "audio": [
                            {
                                "src": (
                                    "https://asset.invalid/"
                                    f"{'0' * 40}/--/main/validation/"
                                    f"{stable['rowIndex']}/audio/audio.wav"
                                )
                            }
                        ],
                    },
                }
            ]
        }

    monkeypatch.setattr(freezer, "_request_json", request_json)

    with pytest.raises(AscendCodeSwitchFreezeError, match="pinned revision"):
        freezer._viewer_row(case)


def test_selection_rejects_coverage_tampering(tmp_path: Path) -> None:
    value = json.loads(SELECTION.read_text(encoding="utf-8"))
    development = [
        case for case in value["cases"] if case["evaluationSplit"] == "development"
    ]
    development[0]["topic"] = development[1]["topic"]
    development[2]["topic"] = development[1]["topic"]
    development[4]["topic"] = development[1]["topic"]
    development[5]["topic"] = development[1]["topic"]
    path = _write_json(tmp_path / "invalid-selection.json", value)

    with pytest.raises(AscendCodeSwitchFreezeError, match="three topics"):
        load_selection_lock(path)


def test_truth_redactor_rejects_nested_reviewer_leak() -> None:
    with pytest.raises(AscendCodeSwitchFreezeError, match="transcript fields"):
        assert_truth_redacted(
            {"reviewer": {"cases": [{"referenceTranscript": "sealed truth"}]}}
        )


def test_freeze_refuses_existing_public_or_vault_root(tmp_path: Path) -> None:
    output_root = tmp_path / "ordinary"
    vault_root = tmp_path / "vault"
    output_root.mkdir()

    with pytest.raises(FileExistsError):
        freeze(
            selection_path=SELECTION,
            output_root=output_root,
            scorer_vault_root=vault_root,
        )

    output_root.rmdir()
    vault_root.mkdir()
    with pytest.raises(FileExistsError):
        freeze(
            selection_path=SELECTION,
            output_root=output_root,
            scorer_vault_root=vault_root,
        )


def test_scorer_vault_must_be_outside_public_root(tmp_path: Path) -> None:
    with pytest.raises(AscendCodeSwitchFreezeError, match="outside"):
        freeze(
            selection_path=SELECTION,
            output_root=tmp_path / "ordinary",
            scorer_vault_root=tmp_path / "ordinary" / "vault",
        )
