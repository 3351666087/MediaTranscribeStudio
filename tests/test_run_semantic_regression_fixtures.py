from __future__ import annotations

import sys

import pytest

from tools import run_semantic_regression_fixtures as runner


@pytest.mark.parametrize(
    ("all_cases_completed", "all_request_sets_exact", "expected_exit_code"),
    [
        (True, True, 0),
        (True, False, 1),
        (False, True, 1),
        (False, False, 1),
    ],
)
def test_main_requires_completed_cases_and_exact_request_sets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    all_cases_completed: bool,
    all_request_sets_exact: bool,
    expected_exit_code: int,
) -> None:
    monkeypatch.setattr(
        runner,
        "run",
        lambda **_kwargs: {
            "allCasesCompleted": all_cases_completed,
            "allRequestSetsExact": all_request_sets_exact,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_semantic_regression_fixtures.py", "--output", str(tmp_path)],
    )

    assert runner.main() == expected_exit_code


def test_main_forwards_pinned_model_and_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"allCasesCompleted": True, "allRequestSetsExact": True}

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_semantic_regression_fixtures.py",
            "--output",
            str(tmp_path),
            "--model",
            "fixture:26b",
            "--model-digest",
            "a" * 64,
            "--batch-size",
            "7",
        ],
    )

    assert runner.main() == 0
    assert captured["model"] == "fixture:26b"
    assert captured["model_digest"] == "a" * 64
    assert captured["batch_size"] == 7
