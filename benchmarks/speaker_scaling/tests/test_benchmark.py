from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.speaker_scaling import cli, runner
from benchmarks.speaker_scaling.synthetic import (
    NEAR_VOICE_COSINE,
    SCENARIO_NAMES,
    generate_scenario,
)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def _truth_assignments(case) -> tuple[int, ...]:
    return tuple(
        0
        if case.truth_by_window_id[window.window_id] is None
        else int(case.truth_by_window_id[window.window_id])
        for window in case.windows
    )


def _fake_result(case, *, assignments=None, count=None):
    predicted_count = count or case.persistent_speaker_count
    candidate = SimpleNamespace(
        count=predicted_count,
        work_items=17,
        as_dict=lambda: {
            "count": predicted_count,
            "objective": 1.0,
            "workItems": 17,
        },
    )
    return SimpleNamespace(
        count=predicted_count,
        confidence=0.875,
        candidate_min=predicted_count,
        candidate_max=predicted_count,
        assignments=assignments or _truth_assignments(case),
        count_candidates=(candidate,),
        count_search_truncated=False,
        leader_estimate=predicted_count,
        leader_count_work_items=11,
        total_work_items=28,
    )


def test_defaults_cover_requested_high_cardinalities() -> None:
    assert runner.DEFAULT_SPEAKER_COUNTS == (1, 2, 3, 5, 8, 13, 32, 64, 129)
    assert runner.DEFAULT_MODES == ("manual", "auto", "hybrid")


def test_csv_parsing_preserves_order_and_rejects_invalid_values() -> None:
    assert cli.parse_speaker_counts("5, 1,129") == (5, 1, 129)
    assert cli.parse_modes("hybrid,manual") == ("hybrid", "manual")
    with pytest.raises(cli.CliUsageError):
        cli.parse_speaker_counts("1,0")
    with pytest.raises(cli.CliUsageError):
        cli.parse_speaker_counts("1,1")
    with pytest.raises(cli.CliUsageError):
        cli.parse_modes("auto,unknown")


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_synthetic_scenarios_have_one_embedding_per_window(scenario: str) -> None:
    case = generate_scenario(
        scenario,
        speaker_count=5,
        samples_per_speaker=3,
    )
    assert len(case.windows) == len(case.embeddings)
    assert case.persistent_speaker_count == 5
    assert case.embedding_dimension >= 5
    assert all(
        window.window_id == embedding.window_id
        for window, embedding in zip(case.windows, case.embeddings)
    )


def test_near_voice_reference_cosine_is_explicit() -> None:
    case = generate_scenario(
        "near-voices",
        speaker_count=5,
        samples_per_speaker=3,
    )
    assert case.near_voice_cosine == NEAR_VOICE_COSINE
    assert _cosine(case.reference_centers[0], case.reference_centers[1]) == (
        pytest.approx(NEAR_VOICE_COSINE, abs=1e-12)
    )
    assert _cosine(case.reference_centers[0], case.reference_centers[2]) == (
        pytest.approx(0.0, abs=1e-12)
    )


def test_singleton_is_excluded_from_persistent_truth() -> None:
    case = generate_scenario(
        "singleton-outlier",
        speaker_count=3,
        samples_per_speaker=4,
    )
    assert len(case.windows) == 13
    assert len(case.outlier_window_ids) == 1
    outlier_id = next(iter(case.outlier_window_ids))
    assert case.truth_by_window_id[outlier_id] is None


def test_shuffled_input_is_not_in_temporal_order() -> None:
    case = generate_scenario(
        "shuffled-orthogonal",
        speaker_count=3,
        samples_per_speaker=2,
    )
    assert not case.input_is_time_ordered
    assert sorted(window.start_ms for window in case.windows) != [
        window.start_ms for window in case.windows
    ]


def test_partition_check_is_label_permutation_invariant() -> None:
    case = generate_scenario(
        "orthogonal",
        speaker_count=3,
        samples_per_speaker=2,
    )
    remap = {0: 2, 1: 0, 2: 1}
    assignments = tuple(
        remap[int(case.truth_by_window_id[window.window_id])]
        for window in case.windows
    )
    assert runner.partition_is_correct(case, assignments, 3)


def test_partition_check_ignores_singleton_but_rejects_persistent_merge() -> None:
    case = generate_scenario(
        "singleton-outlier",
        speaker_count=3,
        samples_per_speaker=2,
    )
    assignments = list(_truth_assignments(case))
    assignments[-1] = 2
    assert runner.partition_is_correct(case, assignments, 3)
    assignments[0] = 1
    assert not runner.partition_is_correct(case, assignments, 3)


def test_runner_reuses_backend_cluster_entry_and_reports_required_fields(
    monkeypatch,
) -> None:
    case = generate_scenario(
        "orthogonal",
        speaker_count=3,
        samples_per_speaker=2,
    )
    calls = []

    def fake_cluster(embeddings, windows, request, config):
        calls.append((embeddings, windows, request, config))
        return _fake_result(case)

    monkeypatch.setattr(runner.speaker_pipeline, "_cluster", fake_cluster)
    report = runner.run_case(case, mode="manual", repeat=2)

    assert len(calls) == 2
    assert all(call[0] is case.embeddings for call in calls)
    assert all(call[1] is case.windows for call in calls)
    assert report["summary"]["allPartitionsCorrect"] is True
    first = report["runs"][0]
    assert first["predictedSpeakerCount"] == 3
    assert first["candidateRange"] == {"min": 3, "max": 3}
    assert first["confidence"] == 0.875
    assert first["workItems"] == {
        "leaderCount": 11,
        "candidateSearch": 17,
        "total": 28,
    }
    assert first["partitionCorrect"] is True
    assert first["windowsPerSecond"] > 0.0


@pytest.mark.parametrize("mode", ("manual", "auto", "hybrid"))
def test_real_backend_entry_smoke_for_each_mode(mode: str) -> None:
    case = generate_scenario(
        "orthogonal",
        speaker_count=3,
        samples_per_speaker=3,
    )
    report = runner.run_case(case, mode=mode, repeat=1)
    run = report["runs"][0]
    assert run["predictedSpeakerCount"] == 3
    assert run["partitionCorrect"] is True
    assert run["candidateRange"]["min"] <= 3 <= run["candidateRange"]["max"]
    assert 0.0 <= run["confidence"] <= 1.0


def test_small_report_is_strict_json_and_has_explicit_disclaimers() -> None:
    report = runner.run_benchmark(
        speaker_counts=(1,),
        samples_per_speaker=2,
        modes=("manual",),
        repeat=1,
        scenario_names=("orthogonal",),
    )
    encoded = cli.strict_json_dumps(report)
    decoded = json.loads(encoded)
    assert decoded["disclaimer"] == {
        "isRealDer": False,
        "loadsMedia": False,
        "loadsModels": False,
        "metric": "exactSyntheticPartition",
        "timingScope": "backend.speaker_pipeline._cluster only",
    }
    assert decoded["algorithm"]["entryPoint"] == (
        "backend.speaker_pipeline._cluster"
    )
    assert decoded["algorithm"]["method"]
    assert decoded["summary"]["caseCount"] == 1


def test_cli_output_file_matches_stdout_bytes(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    payload = {
        "schemaVersion": "speaker-scaling-benchmark/v1",
        "strict": True,
    }
    monkeypatch.setattr(cli, "run_benchmark", lambda **_: payload)
    output = tmp_path / "result.benchmark.json"

    exit_code = cli.main(
        [
            "--speaker-counts",
            "1,3",
            "--samples-per-speaker",
            "2",
            "--modes",
            "manual,auto",
            "--repeat",
            "2",
            "--output-json",
            str(output),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == stdout
    assert json.loads(stdout) == payload


def test_cli_validation_failure_is_also_one_json_document(capsys) -> None:
    exit_code = cli.main(["--speaker-counts", "1,1"])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["error"]["kind"] == "usage"
