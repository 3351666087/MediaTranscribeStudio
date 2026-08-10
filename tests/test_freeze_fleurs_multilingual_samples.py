from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import tools.build_global_sample_library as global_builder
from backend.persistence import canonical_json_sha256
from tools.freeze_fleurs_multilingual_samples import (
    TARGET_LANGUAGES,
    reference_body,
    select_fleurs_cases,
)
from tools.global_sample_library import (
    GlobalSampleCase,
    GlobalSampleLibraryError,
    GlobalSampleSource,
    load_global_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sample_library" / "global-manifest.v1.json"


def _source() -> GlobalSampleSource:
    return GlobalSampleSource(
        source_id="fleurs",
        provider="huggingface",
        dataset="google/fleurs",
        revision="a" * 40,
        license="cc-by-4.0",
        homepage="https://huggingface.co/datasets/google/fleurs",
        attribution="FLEURS, Google; CC BY 4.0.",
    )


def _search_case(*, row_id: str = "42") -> GlobalSampleCase:
    return GlobalSampleCase(
        case_id="fleurs-search-fixture",
        source_id="fleurs",
        acquisition={
            "kind": "hf-viewer-search-row",
            "config": "ja_jp",
            "split": "test",
            "rowIndex": 7,
            "rowIdField": "id",
            "rowId": row_id,
            "searchQuery": "Japanese",
        },
        language="ja-JP",
        region="East Asia",
        evaluation_split="held-out",
        scenarios=("real-recording", "single-speaker"),
        expected_speaker_count=1,
    )


def test_manifest_freezes_three_development_and_held_out_rows_per_language() -> None:
    manifest = load_global_manifest(MANIFEST)
    source, cases = select_fleurs_cases(manifest)

    assert source.dataset == "google/fleurs"
    assert source.license == "cc-by-4.0"
    assert len(cases) == 36
    for config, language in TARGET_LANGUAGES.items():
        for split in ("development", "held-out"):
            bucket = [
                case
                for case in cases
                if case.acquisition["config"] == config
                and case.evaluation_split == split
            ]
            assert len(bucket) == 3
            assert {case.language for case in bucket} == {language}
            assert len({case.acquisition["rowIndex"] for case in bucket}) == 3
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected_ids = {case.case_id for case in cases}
    truth_keys = {
        "expectedTranscript",
        "scoringTranscript",
        "rawTranscript",
        "referenceTranscript",
    }
    assert all(
        not truth_keys.intersection(row)
        for row in raw["cases"]
        if row["id"] in selected_ids
    )


def test_search_acquisition_requires_query_and_pinned_source_row_id(
    tmp_path: Path,
) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    target = next(
        row
        for row in value["cases"]
        if row["acquisition"]["kind"] == "hf-viewer-search-row"
    )
    target["acquisition"].pop("rowId")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(GlobalSampleLibraryError, match="rowIdField and rowId"):
        load_global_manifest(path)


def test_viewer_search_row_uses_pinned_row_index_and_verifies_row_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    case = _search_case()
    requested_urls: list[str] = []

    def request_json(url: str, *, timeout: float = 120.0) -> dict[str, object]:
        assert timeout == 180.0
        parsed = urlparse(url)
        requested_urls.append(url)
        assert parsed.path.endswith("/rows")
        query = parse_qs(parsed.query)
        assert query["offset"] == ["7"]
        assert query["length"] == ["1"]
        assert "query" not in query
        return {
            "rows": [
                {
                    "row_idx": 7,
                    "row": {
                        "id": 42,
                        "path": "fixture.wav",
                        "audio": [
                            {
                                "src": (
                                    "https://datasets-server.huggingface.co/"
                                    "cached-assets/google/fleurs/"
                                    f"{source.revision}/audio.wav"
                                )
                            }
                        ],
                        "transcription": "fixture words",
                        "raw_transcription": "Fixture words.",
                        "gender": 0,
                    },
                }
            ],
            "num_rows_total": 300,
        }

    monkeypatch.setattr(global_builder, "_request_json", request_json)
    monkeypatch.setattr(
        global_builder,
        "_request_bytes",
        lambda url, timeout=120.0, attempts=5: b"audio",
    )

    audio, metadata = global_builder._viewer_row(source, case)

    assert audio == b"audio"
    assert len(requested_urls) == 1
    assert metadata["sourceRowId"] == {"field": "id", "id": "42"}
    assert metadata["viewerAccess"] == {
        "kind": "hf-viewer-search-row",
        "searchQuery": "Japanese",
        "selectionEndpoint": "/search",
        "fetchEndpoint": "/rows",
        "fetchRowIndex": 7,
    }


def test_viewer_search_row_falls_back_to_immutable_parquet_after_rows_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    case = _search_case()
    requested_urls: list[str] = []
    parquet_access = {
        "url": (
            "https://huggingface.co/datasets/google/fleurs/resolve/"
            + "b" * 40
            + "/ja_jp/test/0000.parquet"
        ),
        "revision": "b" * 40,
        "filename": "ja_jp/test/0000.parquet",
        "contentSha256": "c" * 64,
        "size": 123,
        "sourceRevision": source.revision,
        "rowIndex": 7,
        "audioSha256": hashlib.sha256(b"audio").hexdigest(),
    }
    row = {
        "id": 42,
        "path": "fixture.wav",
        "audio": {"bytes": b"audio", "path": "fixture.wav"},
        "transcription": "fixture words",
        "raw_transcription": "Fixture words.",
    }

    def request_json(url: str, *, timeout: float = 120.0) -> dict[str, object]:
        requested_urls.append(url)
        raise GlobalSampleLibraryError("/rows returned HTTP 500")

    monkeypatch.setattr(global_builder, "_request_json", request_json)
    monkeypatch.setattr(
        global_builder,
        "_parquet_row",
        lambda source, case: (b"audio", row, parquet_access),
    )

    audio, metadata = global_builder._viewer_row(source, case)

    assert audio == b"audio"
    assert len(requested_urls) == 1
    assert metadata["sourceRowId"] == {"field": "id", "id": "42"}
    assert metadata["viewerAccess"] == {
        "kind": "hf-viewer-search-row",
        "searchQuery": "Japanese",
        "selectionEndpoint": "/search",
        "fetchEndpoint": "/parquet",
        "fetchRowIndex": 7,
        "fallbackFrom": "/rows",
        "parquet": parquet_access,
    }


def test_viewer_search_row_parquet_fallback_rejects_changed_row_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    case = _search_case(row_id="99")
    row = {
        "id": 42,
        "path": "fixture.wav",
        "audio": {"bytes": b"audio", "path": "fixture.wav"},
        "transcription": "fixture words",
    }
    monkeypatch.setattr(
        global_builder,
        "_request_json",
        lambda url, timeout=120.0: (_ for _ in ()).throw(
            GlobalSampleLibraryError("HTTP 500")
        ),
    )
    monkeypatch.setattr(
        global_builder,
        "_parquet_row",
        lambda source, case: (
            b"audio",
            row,
            {
                "url": (
                    "https://huggingface.co/datasets/google/fleurs/resolve/"
                    + "b" * 40
                    + "/ja_jp/test/0000.parquet"
                ),
                "revision": "b" * 40,
                "filename": "0000.parquet",
                "contentSha256": "c" * 64,
                "size": 1,
                "sourceRevision": source.revision,
                "rowIndex": 7,
                "audioSha256": hashlib.sha256(b"audio").hexdigest(),
            },
        ),
    )
    monkeypatch.setattr(
        global_builder,
        "_request_bytes",
        lambda *args, **kwargs: pytest.fail("mismatched fallback row must not download"),
    )

    with pytest.raises(GlobalSampleLibraryError, match="source row identity mismatch"):
        global_builder._viewer_row(source, case)


def test_viewer_search_row_rejects_changed_source_row_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    case = _search_case(row_id="99")
    monkeypatch.setattr(
        global_builder,
        "_request_json",
        lambda url, timeout=120.0: {
            "rows": [
                {
                    "row_idx": 7,
                    "row": {
                        "id": 42,
                        "path": "fixture.wav",
                        "audio": [
                            {
                                "src": (
                                    "https://datasets-server.huggingface.co/"
                                    "cached-assets/google/fleurs/"
                                    f"{source.revision}/audio.wav"
                                )
                            }
                        ],
                        "transcription": "fixture words",
                        "raw_transcription": "Fixture words.",
                    },
                }
            ],
            "num_rows_total": 1,
        },
    )

    with pytest.raises(GlobalSampleLibraryError, match="source row identity mismatch"):
        global_builder._viewer_row(source, case)


def test_viewer_search_row_rejects_a_different_row_at_the_pinned_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    case = _search_case()
    monkeypatch.setattr(
        global_builder,
        "_request_json",
        lambda url, timeout=120.0: {
            "rows": [
                {
                    "row_idx": 8,
                    "row": {
                        "id": 42,
                        "path": "different.wav",
                        "audio": [
                            {
                                "src": (
                                    "https://datasets-server.huggingface.co/"
                                    "cached-assets/google/fleurs/"
                                    f"{source.revision}/different.wav"
                                )
                            }
                        ],
                        "transcription": "different words",
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        global_builder,
        "_request_bytes",
        lambda *args, **kwargs: pytest.fail("mismatched row must not be downloaded"),
    )

    with pytest.raises(GlobalSampleLibraryError, match="row identity is invalid"):
        global_builder._viewer_row(source, case)


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        ([], "exactly one audio asset"),
        (
            [
                {
                    "src": (
                        "https://datasets-server.huggingface.co/"
                        f"cached-assets/google/fleurs/{'b' * 40}/different.wav"
                    )
                }
            ],
            "not bound to the pinned dataset revision",
        ),
    ],
)
def test_viewer_search_row_rejects_unpinned_audio_assets(
    monkeypatch: pytest.MonkeyPatch,
    audio: list[dict[str, str]],
    message: str,
) -> None:
    source = _source()
    case = _search_case()
    monkeypatch.setattr(
        global_builder,
        "_request_json",
        lambda url, timeout=120.0: {
            "rows": [
                {
                    "row_idx": 7,
                    "row": {
                        "id": 42,
                        "path": "fixture.wav",
                        "audio": audio,
                        "transcription": "fixture words",
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        global_builder,
        "_request_bytes",
        lambda *args, **kwargs: pytest.fail("unpinned asset must not be downloaded"),
    )

    with pytest.raises(GlobalSampleLibraryError, match=message):
        global_builder._viewer_row(source, case)


def test_reference_marks_held_out_as_not_tuning_eligible(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    cases = [
        {
            "id": "dev",
            "language": "ar-EG",
            "evaluationSplit": "development",
            "durationSeconds": 1.0,
            "sha256": "a" * 64,
            "sourceRow": {"path": "dev.wav"},
        },
        {
            "id": "held",
            "language": "ar-EG",
            "evaluationSplit": "held-out",
            "durationSeconds": 2.0,
            "sha256": "b" * 64,
            "sourceRow": {"path": "held.wav"},
        },
    ]

    value = reference_body(
        manifest_path=manifest,
        source=_source(),
        viewer={"viewer": True},
        cases=cases,
    )

    assert value["splitPolicy"]["held-out"]["tuningEligible"] is False
    assert (
        value["truthPersistencePolicy"]["referenceManifestMayEnterBlindPacket"]
        is False
    )
    body = dict(value)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)
