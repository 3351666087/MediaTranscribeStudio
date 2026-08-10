from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from types import SimpleNamespace
from pathlib import Path

import pytest

from tools.global_sample_library import (
    GlobalSampleCase,
    GlobalSampleLibraryError,
    GlobalSampleSource,
    coverage_summary,
    load_global_manifest,
)
from tools.build_global_sample_library import (
    _cached_case_matches_acquisition,
    _case_row_metadata,
    _refresh_existing_case,
    _streaming_row,
)
from tools.build_global_derived_matrix import overlap_intervals
import tools.build_global_real_diarization as real_diarization_builder
from tools.build_global_real_diarization import (
    _alimeeting_tree_evidence,
    _clip_reference_transcript,
    _validated_alimeeting_archive_members,
    align_diarization_window_to_stm,
    alimeeting_textgrid_turns,
    parse_alimeeting_textgrid,
    parse_aishell4_rttm,
    parse_aishell4_stm,
    parse_aishell4_textgrid_audio_tier,
    select_alimeeting_textgrid_window,
    select_diarization_window,
    _voxconverse_shard_plans,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sample_library" / "global-manifest.v1.json"


def test_global_manifest_covers_regions_languages_and_splits() -> None:
    manifest = load_global_manifest(MANIFEST)
    coverage = coverage_summary(manifest)

    assert coverage["caseCount"] == 93
    assert len(coverage["languages"]) == 34
    assert len(coverage["regions"]) >= 8
    assert coverage["evaluationSplits"] == [
        "development",
        "held-out",
        "regression",
    ]
    assert {"real-recording", "single-speaker"} <= set(coverage["scenarios"])
    assert {
        "audiobook",
        "challenging-read-speech",
        "far-field",
        "meeting-speech",
        "telephone-band",
    } <= set(coverage["scenarios"])
    assert {source.license for source in manifest.sources} == {
        "cc-by-4.0",
        "cc-by-sa-4.0",
    }
    split_counts = {
        split: sum(case.evaluation_split == split for case in manifest.cases)
        for split in coverage["evaluationSplits"]
    }
    assert split_counts == {
        "development": 38,
        "held-out": 35,
        "regression": 20,
    }


def test_dataset_specific_row_fields_and_groups_are_verified() -> None:
    source = GlobalSampleSource(
        source_id="fixture",
        provider="huggingface",
        dataset="example/dataset",
        revision="a" * 40,
        license="cc-by-4.0",
        homepage="https://huggingface.co/datasets/example/dataset",
        attribution="Fixture.",
    )
    case = GlobalSampleCase(
        case_id="fixture-case",
        source_id="fixture",
        acquisition={
            "kind": "hf-viewer-row",
            "config": "default",
            "split": "test",
            "rowIndex": 3,
            "transcriptField": "text",
            "pathField": "file",
            "speakerField": "speaker_id",
            "speakerId": "42",
            "recordingField": "chapter_id",
            "recordingId": "9",
        },
        language="en-US",
        region="North America",
        evaluation_split="held-out",
        scenarios=("real-recording", "single-speaker"),
        expected_speaker_count=1,
    )
    row = {
        "text": "expected words",
        "file": "sample.flac",
        "speaker_id": 42,
        "chapter_id": 9,
    }

    metadata = _case_row_metadata(row, source=source, case=case)

    assert metadata["transcript"] == "expected words"
    assert metadata["path"] == "sample.flac"
    assert metadata["verifiedSourceGroups"] == {
        "speaker": {"field": "speaker_id", "id": "42"},
        "recording": {"field": "chapter_id", "id": "9"},
    }
    row["chapter_id"] = 10
    with pytest.raises(GlobalSampleLibraryError, match="recording identity mismatch"):
        _case_row_metadata(row, source=source, case=case)

    cached = {
        "sourceLocator": {
            "url": f"https://huggingface.co/datasets/{source.dataset}/tree/{source.revision}",
            "dataset": source.dataset,
            "revision": source.revision,
            "config": "default",
            "split": "test",
            "rowIndex": 3,
        },
        "sourceRow": {
            "fieldMapping": metadata["fieldMapping"],
            "verifiedSourceGroups": metadata["verifiedSourceGroups"],
        },
    }
    assert _cached_case_matches_acquisition(cached, source=source, case=case)
    cached["sourceRow"]["verifiedSourceGroups"]["recording"]["id"] = "10"
    assert not _cached_case_matches_acquisition(cached, source=source, case=case)


def test_cached_default_field_mapping_is_migrated(
    tmp_path: Path,
) -> None:
    source = GlobalSampleSource(
        source_id="fixture",
        provider="huggingface",
        dataset="example/dataset",
        revision="a" * 40,
        license="cc-by-4.0",
        homepage="https://huggingface.co/datasets/example/dataset",
        attribution="Fixture.",
    )
    case = GlobalSampleCase(
        case_id="fixture-case",
        source_id="fixture",
        acquisition={
            "kind": "hf-viewer-row",
            "config": "default",
            "split": "test",
            "rowIndex": 3,
        },
        language="en-US",
        region="North America",
        evaluation_split="held-out",
        scenarios=("real-recording", "single-speaker"),
        expected_speaker_count=1,
    )
    output = tmp_path / "fixture.wav"
    output.write_bytes(b"audio")
    cached = {
        "expectedTranscript": "expected words",
        "rawTranscript": "expected words",
        "sourceRow": {"path": "fixture.wav"},
    }

    _refresh_existing_case(
        cached,
        source=source,
        case=case,
        output_path=output,
    )

    assert cached["sourceRow"]["fieldMapping"] == {
        "transcript": "transcription",
        "rawTranscript": "raw_transcription",
        "path": "path",
    }
    assert cached["sourceRow"]["verifiedSourceGroups"] == {}


def test_streaming_runtime_failure_is_isolated_as_sample_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = GlobalSampleSource(
        source_id="fixture",
        provider="huggingface",
        dataset="example/dataset",
        revision="a" * 40,
        license="cc-by-4.0",
        homepage="https://huggingface.co/datasets/example/dataset",
        attribution="Fixture.",
    )
    case = GlobalSampleCase(
        case_id="fixture-streaming",
        source_id="fixture",
        acquisition={
            "kind": "hf-streaming-row",
            "config": "default",
            "split": "test",
            "rowIndex": 0,
        },
        language="en-US",
        region="North America",
        evaluation_split="held-out",
        scenarios=("real-recording", "single-speaker"),
        expected_speaker_count=1,
    )

    def fail_load(*args: object, **kwargs: object) -> object:
        raise RuntimeError("local runtime path must not escape")

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Audio=object, load_dataset=fail_load),
    )
    with pytest.raises(
        GlobalSampleLibraryError,
        match="Hugging Face streaming failed: RuntimeError",
    ) as captured:
        _streaming_row(source, case)
    assert "local runtime path" not in str(captured.value)


def test_global_manifest_requires_pinned_dataset_revision(
    tmp_path: Path,
) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["sources"][0]["revision"] = "main"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(GlobalSampleLibraryError, match="40-character commit"):
        load_global_manifest(path)


def test_global_manifest_rejects_unapproved_or_missing_license(
    tmp_path: Path,
) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["sources"][0]["license"] = "unknown"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(GlobalSampleLibraryError, match="license is not approved"):
        load_global_manifest(path)


def test_global_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["cases"][1]["id"] = value["cases"][0]["id"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(GlobalSampleLibraryError, match="case IDs must be unique"):
        load_global_manifest(path)


def test_global_manifest_rejects_source_group_leakage(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    first = value["cases"][0]
    second = value["cases"][1]
    first["acquisition"].update(
        {"speakerField": "speaker_id", "speakerId": "shared-speaker"}
    )
    second["acquisition"].update(
        {"speakerField": "speaker_id", "speakerId": "shared-speaker"}
    )
    first["evaluationSplit"] = "development"
    second["evaluationSplit"] = "held-out"
    path = tmp_path / "leaked.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(GlobalSampleLibraryError, match="cross evaluation splits"):
        load_global_manifest(path)


def test_global_manifest_dynamic_n_matrices_cover_stress_counts() -> None:
    manifest = load_global_manifest(MANIFEST)
    matrices = {
        str(matrix["kind"]): matrix for matrix in manifest.derived_matrices
    }

    assert matrices["sequential-speaker-mixture"]["speakerCounts"] == [2, 3, 5, 8]
    assert matrices["overlap-speaker-mixture"]["speakerCounts"] == [
        2,
        3,
        5,
        8,
        13,
    ]
    assert len(matrices["overlap-speaker-mixture"]["sourceCaseIds"]) >= 13


def test_global_manifest_pins_real_diarization_sources() -> None:
    manifest = load_global_manifest(MANIFEST)
    sources = {source.source_id: source for source in manifest.sources}
    plans = {
        str(plan["sourceId"]): plan
        for plan in manifest.planned_real_diarization_sources
    }

    assert sources["aishell4"].revision == (
        "df062e4993eeb9873605f8c74d6fac1db0560799"
    )
    assert sources["aishell4"].license == "cc-by-sa-4.0"
    assert plans["aishell4"]["sessionId"] == "L_R003S01C02"
    assert plans["aishell4"]["targetSpeakerCounts"] == [5]
    assert plans["aishell4"]["evaluationSplit"] == "regression"
    assert plans["aishell4"]["officialEvaluationRevision"] == (
        "bad82b77c3753df1b232c5c6491cd3e2f2e32d24"
    )
    assert plans["voxconverse"]["targetSpeakerCounts"] == [1, 2, 3, 5, 8, 13]
    assert plans["voxconverse"]["parquetBytes"] == 485283393
    assert plans["voxconverse"]["parquetSha256"] == (
        "f36c54412f0ac9cfe7ec2682e27f70f3"
        "e3df63d8e2b8f288d4f62341a595917d"
    )
    shards = plans["voxconverse"]["parquetShards"]
    assert [shard["parquetPath"] for shard in shards] == [
        "data/dev-00000-of-00005.parquet",
        "data/dev-00002-of-00005.parquet",
        "data/test-00000-of-00011.parquet",
    ]
    assert [shard["parquetSha256"] for shard in shards[1:]] == [
        "77800f7d5fa37116e7f7d7c6b482f9a9ea654827b7af0e4d61c43facc5a7125e",
        "487d66b75a2edc808407a6aa3b344bdd492b326f6a5b78b580d250e203f0860f",
    ]
    n13_targets = [
        (shard["parquetPath"], target)
        for shard in shards
        for target in shard["rowTargets"]
        if target["targetSpeakerCount"] == 13
    ]
    assert n13_targets == [
        (
            "data/dev-00002-of-00005.parquet",
            {
                "rowIndex": 38,
                "targetSpeakerCount": 13,
                "evaluationSplit": "development",
                "maximumDurationSeconds": 300,
            },
        ),
        (
            "data/test-00000-of-00011.parquet",
            {
                "rowIndex": 8,
                "targetSpeakerCount": 13,
                "evaluationSplit": "held-out",
                "maximumDurationSeconds": 120,
            },
        ),
    ]
    assert {
        target["evaluationSplit"]
        for target in plans["voxconverse"]["rowTargets"]
    } == {"development", "held-out", "regression"}
    assert plans["ami"]["evaluationSplit"] == "development"
    assert {
        (target["rowIndex"], target["targetSpeakerCount"])
        for target in plans["voxconverse"]["rowTargets"]
    } >= {
        (35, 1),
        (13, 5),
        (15, 5),
        (23, 8),
        (2, 8),
    }
    held_out_rows = {
        target["rowIndex"]
        for target in plans["voxconverse"]["rowTargets"]
        if target["evaluationSplit"] == "held-out"
    }
    development_rows = {
        target["rowIndex"]
        for target in plans["voxconverse"]["rowTargets"]
        if target["evaluationSplit"] == "development"
    }
    assert held_out_rows.isdisjoint(development_rows)
    alimeeting = plans["alimeeting"]
    assert alimeeting["provider"] == "openslr"
    assert alimeeting["officialHomepage"] == "https://www.openslr.org/119/"
    assert alimeeting["license"] == "cc-by-sa-4.0"
    assert alimeeting["evaluationSplit"] == "held-out"
    assert alimeeting["targetSpeakerCounts"] == [2, 3, 4]
    assert alimeeting["unsupportedTargetSpeakerCounts"] == [5]
    assert alimeeting["archiveBytes"] == 3673718355
    assert alimeeting["archiveSha256"] == (
        "dc47343b2474b5ebcf458927e878155f6"
        "ddeb59c85e685b3645c32a1f9578d92"
    )
    assert alimeeting["extractedTreeSha256"] == (
        "3e39ace217a8a7c98707d9742e55332c"
        "bdb9349ea47ef18e05f4d60e9a12b629"
    )
    assert alimeeting["selectionUsesModelScores"] is False
    assert {
        target["speakerCount"] for target in alimeeting["sessionTargets"]
    } == {2, 3, 4}


def test_overlap_intervals_tracks_distinct_active_speakers() -> None:
    turns = [
        {"speakerId": "a", "startSeconds": 0.0, "endSeconds": 4.0},
        {"speakerId": "b", "startSeconds": 1.0, "endSeconds": 3.0},
        {"speakerId": "c", "startSeconds": 2.0, "endSeconds": 5.0},
    ]

    assert overlap_intervals(turns) == [
        {"startSeconds": 1.0, "endSeconds": 2.0, "speakerIds": ["a", "b"]},
        {
            "startSeconds": 2.0,
            "endSeconds": 3.0,
            "speakerIds": ["a", "b", "c"],
        },
        {"startSeconds": 3.0, "endSeconds": 4.0, "speakerIds": ["a", "c"]},
    ]


def test_real_diarization_window_is_exact_bounded_and_deterministic() -> None:
    turns = [
        {"speakerId": "a", "startSeconds": 0.0, "endSeconds": 12.0},
        {"speakerId": "b", "startSeconds": 4.0, "endSeconds": 10.0},
        {"speakerId": "c", "startSeconds": 30.0, "endSeconds": 35.0},
    ]

    first = select_diarization_window(turns, 2)
    second = select_diarization_window(turns, 2)

    assert first == second
    assert first["algorithm"] == "event-boundary-shortest-coverage-v2"
    assert first["speakerSet"] == ["a", "b"]
    assert first["durationSeconds"] == 10.0
    assert first["annotatedOverlapSeconds"] == 6.0
    assert {turn["transcript"] for turn in first["turns"]} == {None}


def test_real_diarization_window_supports_pinned_high_n_duration() -> None:
    turns = [
        {
            "speakerId": f"speaker-{index:02d}",
            "startSeconds": float(index * 9),
            "endSeconds": float(index * 9 + 1),
        }
        for index in range(13)
    ]
    turns[-1]["endSeconds"] = 120.0

    with pytest.raises(GlobalSampleLibraryError, match=r"no <=90s window"):
        select_diarization_window(turns, 13)

    window = select_diarization_window(
        turns,
        13,
        maximum_duration_seconds=120.0,
    )

    assert window["sourceStartSeconds"] == 0.0
    assert window["sourceEndSeconds"] == 120.0
    assert window["durationSeconds"] == 120.0
    assert window["selectionMaximumDurationSeconds"] == 120.0
    assert len(window["speakerSet"]) == 13


def test_voxconverse_multi_shard_plan_is_split_and_path_bound() -> None:
    plan = {
        "parquetShards": [
            {
                "config": "default",
                "split": "dev",
                "parquetPath": "data/dev-00002-of-00005.parquet",
                "parquetBytes": 123,
                "parquetSha256": "a" * 64,
                "rowTargets": [
                    {
                        "rowIndex": 38,
                        "targetSpeakerCount": 13,
                        "evaluationSplit": "development",
                        "maximumDurationSeconds": 300,
                    }
                ],
            },
            {
                "config": "default",
                "split": "test",
                "parquetPath": "data/test-00000-of-00011.parquet",
                "parquetBytes": 456,
                "parquetSha256": "b" * 64,
                "rowTargets": [
                    {
                        "rowIndex": 8,
                        "targetSpeakerCount": 13,
                        "evaluationSplit": "held-out",
                        "maximumDurationSeconds": 120,
                    }
                ],
            },
        ]
    }

    shards = _voxconverse_shard_plans(plan)

    assert [shard["split"] for shard in shards] == ["dev", "test"]
    assert [shard["legacyNames"] for shard in shards] == [False, False]
    assert [shard["parquetPath"] for shard in shards] == [
        "data/dev-00002-of-00005.parquet",
        "data/test-00000-of-00011.parquet",
    ]

    plan["parquetShards"][1]["parquetPath"] = (
        "data/dev-00002-of-00005.parquet"
    )
    with pytest.raises(GlobalSampleLibraryError, match="shard is invalid"):
        _voxconverse_shard_plans(plan)


def test_alimeeting_textgrid_window_preserves_joint_truth() -> None:
    textgrid = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 12
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "N_SPK0001"
        xmin = 0
        xmax = 12
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 6
            text = "甲""乙"
        intervals [2]:
            xmin = 9
            xmax = 11
            text = "收尾"
    item [2]:
        class = "IntervalTier"
        name = "N_SPK0002"
        xmin = 0
        xmax = 12
        intervals: size = 1
        intervals [1]:
            xmin = 4
            xmax = 7
            text = "回答"
"""

    grid = parse_alimeeting_textgrid(textgrid)
    turns = alimeeting_textgrid_turns(grid)
    first = select_alimeeting_textgrid_window(turns, 2)
    second = select_alimeeting_textgrid_window(turns, 2)

    assert first == second
    assert first["algorithm"] == "textgrid-component-shortest-coverage-v1"
    assert first["sourceStartSeconds"] == 0.0
    assert first["sourceEndSeconds"] == 11.0
    assert first["durationSeconds"] == 11.0
    assert first["speakerSet"] == ["N_SPK0001", "N_SPK0002"]
    assert first["annotatedOverlapSeconds"] == 2.0
    assert [turn["transcript"] for turn in turns] == ['甲"乙', "回答", "收尾"]
    assert [
        turn["transcript"]
        for turn in _clip_reference_transcript(turns, first)
    ] == ['甲"乙', "回答", "收尾"]


def test_alimeeting_case_records_distinguish_far_from_near_mix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = [
        {
            "speakerId": "N_SPK0001",
            "startSeconds": 0.0,
            "endSeconds": 6.0,
            "transcript": "你好",
        },
        {
            "speakerId": "N_SPK0002",
            "startSeconds": 4.0,
            "endSeconds": 11.0,
            "transcript": "世界",
        },
    ]
    window = select_alimeeting_textgrid_window(turns, 2)
    session = {
        "sessionId": "R0001_M0001",
        "speakerCount": 2,
        "turns": turns,
        "farAudio": tmp_path / "far.wav",
        "farAudioEvidence": {"path": "far.wav", "sha256": "a" * 64},
        "nearAudio": [tmp_path / "near-a.wav", tmp_path / "near-b.wav"],
        "nearAudioEvidence": [
            {"path": "near-a.wav", "sha256": "b" * 64},
            {"path": "near-b.wav", "sha256": "c" * 64},
        ],
        "farTextGridEvidence": {
            "path": "truth.TextGrid",
            "sha256": "d" * 64,
        },
    }
    plan = {
        "dataset": "SLR119/AliMeeting",
        "revision": "sha256:" + "e" * 64,
        "evaluationSplit": "held-out",
    }

    def write_fixture_audio(
        _sources: object,
        output: Path,
        _window: dict[str, object],
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fixture-wave")

    monkeypatch.setattr(real_diarization_builder, "_clip_audio", write_fixture_audio)
    monkeypatch.setattr(
        real_diarization_builder,
        "_clip_synchronized_near_audio",
        write_fixture_audio,
    )
    monkeypatch.setattr(
        real_diarization_builder,
        "_probe_audio",
        lambda _path: {
            "codec": "pcm_s16le",
            "sampleRate": 16000,
            "channels": 1,
            "durationSeconds": 11.0,
        },
    )

    far = real_diarization_builder._alimeeting_case_record(
        output_root=tmp_path,
        plan=plan,
        session=session,
        window=window,
        modality="far-field-array",
    )
    near = real_diarization_builder._alimeeting_case_record(
        output_root=tmp_path,
        plan=plan,
        session=session,
        window=window,
        modality="synchronized-near-field-mixture",
    )

    assert far["realOrSynthetic"] == "real-recording"
    assert near["realOrSynthetic"] == "synthetic-mixture"
    assert far["truthEligibility"]["derJer"] is True
    assert far["truthEligibility"]["asr"] is True
    assert near["truthEligibility"] == far["truthEligibility"]
    assert far["scoringTranscript"] == near["scoringTranscript"] == "你好世界"
    assert len(far["sourceAudioArtifacts"]) == 1
    assert len(near["sourceAudioArtifacts"]) == 2
    assert far["windowSelection"]["selectionUsesModelScores"] is False


def test_alimeeting_tree_fingerprint_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_bytes(b"alpha")
    (tmp_path / "nested" / "b.txt").write_bytes(b"beta")
    first = _alimeeting_tree_evidence(tmp_path)
    second = _alimeeting_tree_evidence(tmp_path)
    alpha_sha = hashlib.sha256(b"alpha").hexdigest()
    beta_sha = hashlib.sha256(b"beta").hexdigest()
    expected = hashlib.sha256(
        (
            f"a.txt\t5\t{alpha_sha}\n"
            f"nested/b.txt\t4\t{beta_sha}\n"
        ).encode("utf-8")
    ).hexdigest()

    assert first["fileCount"] == 2
    assert first["bytes"] == 9
    assert first["treeSha256"] == expected
    assert second["treeSha256"] == first["treeSha256"]


def test_alimeeting_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    payload = b"escape"
    with tarfile.open(archive, mode="w:gz") as handle:
        member = tarfile.TarInfo("Eval_Ali/../escape.txt")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    plan = {
        "archiveBytes": archive.stat().st_size,
        "archiveSha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "extractedRootName": "Eval_Ali",
        "extractedFileCount": 1,
        "extractedBytes": len(payload),
    }

    with pytest.raises(GlobalSampleLibraryError, match="unsafe path or entry"):
        _validated_alimeeting_archive_members(archive, plan)


def test_aishell4_annotation_parsers_preserve_joint_truth() -> None:
    recording = "meeting"
    rttm = "\n".join(
        [
            "SPEAKER meeting 1 3.0000 2.0000 <NA> <NA> spk-b <NA> <NA>",
            "SPEAKER meeting 1 1.0000 3.0000 <NA> <NA> spk-a <NA> <NA>",
        ]
    )
    stm = "\n".join(
        [
            "meeting 0 spk-b 3.0 5.0 世 界",
            "meeting 0 spk-a 1.0 4.0 你 好",
        ]
    )
    textgrid = """File type = "ooTextFile"
Object class = "TextGrid"

item []:
    item [1]:
        class = "IntervalTier"
        name = "spk-a"
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 2.5
            text = "你好"
        intervals [2]:
            xmin = 2.5
            xmax = 6
            text = ""
    item [2]:
        class = "IntervalTier"
        name = "spk-b"
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 6
            text = "世界"
"""

    turns = parse_aishell4_rttm(rttm, recording)
    references = parse_aishell4_stm(stm, recording)
    audio_tier = parse_aishell4_textgrid_audio_tier(textgrid)
    window = {
        "sourceStartSeconds": 2.0,
        "sourceEndSeconds": 4.0,
    }

    assert [turn["speakerId"] for turn in turns] == ["spk-a", "spk-b"]
    assert [turn["transcript"] for turn in references] == ["你好", "世界"]
    assert audio_tier == [
        {
            "index": 1,
            "startSeconds": 0.0,
            "endSeconds": 2.5,
            "text": "你好",
        },
        {
            "index": 2,
            "startSeconds": 2.5,
            "endSeconds": 6.0,
            "text": "",
        },
    ]
    assert _clip_reference_transcript(references, window) == [
        {
            "speakerId": "spk-a",
            "sourceStartSeconds": 2.0,
            "sourceEndSeconds": 4.0,
            "startSeconds": 0.0,
            "endSeconds": 2.0,
            "transcript": "你好",
        },
        {
            "speakerId": "spk-b",
            "sourceStartSeconds": 3.0,
            "sourceEndSeconds": 4.0,
            "startSeconds": 1.0,
            "endSeconds": 2.0,
            "transcript": "世界",
        },
    ]

    aligned = align_diarization_window_to_stm(
        {
            "sourceStartSeconds": 2.0,
            "sourceEndSeconds": 4.0,
            "durationSeconds": 2.0,
        },
        turns,
        references,
        2,
    )
    assert aligned["sourceStartSeconds"] == 1.0
    assert aligned["sourceEndSeconds"] == 5.0
    assert aligned["durationSeconds"] == 4.0
    assert aligned["speakerSet"] == ["spk-a", "spk-b"]
