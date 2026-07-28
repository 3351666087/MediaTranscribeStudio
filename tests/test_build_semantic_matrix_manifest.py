from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_semantic_matrix_manifest import main


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, *, mode: str = "candidate-composition") -> tuple:
    output_root = tmp_path / "outputs"
    case_output = output_root / "case-1"
    semantic_root = case_output / "semantic"
    transcript = semantic_root / "input-transcript.v2.json"
    lattice = semantic_root / "lattice.json"
    arbitration = semantic_root / "arbitration.json"
    composition = semantic_root / "composition.json"
    for path in (transcript, lattice, arbitration, composition):
        _write(path, {})
    _write(case_output / "pipeline-metrics.v1.json", {})
    _write(
        case_output / "checkpoint.v2.json",
        {
            "semantic": {
                "mode": mode,
                "status": "completed",
                "autoApply": mode == "candidate-composition",
                "inputLatticePath": str(lattice),
                "arbitrationPath": str(arbitration),
                "artifactPath": str(composition),
            }
        },
    )
    sample_manifest = tmp_path / "samples.json"
    _write(
        sample_manifest,
        {
            "cases": [
                {
                    "id": "case-1",
                    "language": "mul",
                    "expectedLanguages": ["en", "es"],
                    "expectedSpeakerCount": None,
                    "scoringTranscript": "hello hola",
                    "scenario": ["single-speaker", "code-switch"],
                    "truthEligibility": {
                        "speakerCount": False,
                        "asr": True,
                    },
                }
            ]
        },
    )
    return sample_manifest, output_root, tmp_path / "matrix.json"


def test_builds_truth_qualified_pending_review_manifest(
    tmp_path: Path,
) -> None:
    sample_manifest, output_root, output = _fixture(tmp_path)

    assert main(
        [
            "--sample-manifest",
            str(sample_manifest),
            "--worker-output-root",
            str(output_root),
            "--matrix-id",
            "fixture-matrix",
            "--case",
            "case-1",
            "--output",
            str(output),
        ]
    ) == 0

    value = json.loads(output.read_text(encoding="utf-8"))
    case = value["cases"][0]
    assert case["expected"] == {
        "speakerCount": None,
        "languages": ["en", "es"],
        "transcript": "hello hola",
    }
    assert case["agentSemanticReview"]["status"] == "pending"
    assert case["artifacts"]["pipelineMetrics"].endswith(
        "pipeline-metrics.v1.json"
    )


def test_rejects_legacy_semantic_output(tmp_path: Path) -> None:
    sample_manifest, output_root, output = _fixture(
        tmp_path,
        mode="legacy-suggestions",
    )

    with pytest.raises(ValueError, match="mandatory candidate composition"):
        main(
            [
                "--sample-manifest",
                str(sample_manifest),
                "--worker-output-root",
                str(output_root),
                "--matrix-id",
                "fixture-matrix",
                "--output",
                str(output),
            ]
        )
