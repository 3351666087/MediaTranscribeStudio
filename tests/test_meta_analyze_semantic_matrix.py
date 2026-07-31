from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256
from backend.semantic_composition import (
    build_semantic_composition,
    build_semantic_job_arbitration,
)
from test_semantic_composition import (
    _document,
    _full_lattice,
    _ready_response,
)
from tools.meta_analyze_semantic_matrix import main
from tools.meta_analyze_semantic_matrix import _text_error


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    document = _document()
    lattice = _full_lattice(document)
    response = _ready_response(lattice)
    arbitration = build_semantic_job_arbitration(
        job_id=document["jobId"],
        lattice=lattice,
        response=response,
        model="fixture-9b",
        provider={
            "id": "fixture-loopback",
            "version": "1",
            "networkPolicy": "loopback-only",
        },
    )
    composition = build_semantic_composition(document, lattice, arbitration)
    transcript = tmp_path / "transcript.json"
    lattice_path = tmp_path / "lattice.json"
    arbitration_path = tmp_path / "arbitration.json"
    composition_path = tmp_path / "composition.json"
    _write(transcript, document)
    _write(lattice_path, lattice)
    _write(arbitration_path, arbitration)
    _write(composition_path, composition)
    manifest = {
        "schemaVersion": "1.0.0",
        "matrixId": "fixture-matrix",
        "cases": [
            {
                "id": "fixture-case",
                "scenarios": ["single-speaker", "multilingual"],
                "expected": {
                    "speakerCount": 2,
                    "languages": ["en", "es"],
                    "transcript": "Hello world",
                },
                "artifacts": {
                    "transcript": str(transcript),
                    "lattice": str(lattice_path),
                    "arbitration": str(arbitration_path),
                    "composition": str(composition_path),
                },
                "agentSemanticReview": {
                    "status": "completed",
                    "meaningPreservation": "partial",
                    "speakerCoherence": "pass",
                    "languageCoherence": "pass",
                    "translationCompleteness": "not-applicable",
                    "notes": ["Fixture text is intentionally incomplete."],
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    _write(manifest_path, manifest)
    return manifest_path, output


def test_meta_analysis_preserves_final_text_and_model_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, output = _fixture(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "meta_analyze_semantic_matrix.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    assert main() == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    case = report["cases"][0]
    assert case["final"]["text"] == "Hola World"
    assert case["baseline"]["textQuality"]["metric"] == "wer"
    assert case["final"]["textQuality"]["metric"] == "wer"
    assert case["final"]["textChangeFromBaseline"] in {
        "improved",
        "regressed",
        "unchanged",
    }
    assert case["semanticExecution"]["model"] == "fixture-9b"
    assert case["semanticExecution"]["sourceDurationSeconds"] == 2.0
    assert case["semanticExecution"]["realtimeFactor"] is None
    assert case["semanticExecution"]["selectedProducerSystemCounts"]
    assert case["agentSemanticReview"]["reviewerType"] == (
        "agent-semantic-audit"
    )
    assert case["agentSemanticReview"]["humanApproval"] is False
    assert report["policy"]["acceptanceUnit"] == (
        "speaker-language-time-finalText"
    )
    assert report["releaseApproved"] is False


def test_text_error_uses_language_appropriate_units() -> None:
    english = _text_error(
        "the quick fox",
        "the fox",
        languages=["en-US"],
    )
    chinese = _text_error(
        "这个挺普遍",
        "这个普遍",
        languages=["zh-CN"],
    )

    assert english == {
        "metric": "wer",
        "errors": 1,
        "referenceUnits": 3,
        "hypothesisUnits": 2,
        "errorRate": 0.333333333,
    }
    assert chinese == {
        "metric": "cer",
        "errors": 1,
        "referenceUnits": 5,
        "hypothesisUnits": 4,
        "errorRate": 0.2,
    }


def test_meta_analysis_rejects_rebound_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, output = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    composition_path = Path(
        manifest["cases"][0]["artifacts"]["composition"]
    )
    composition = json.loads(composition_path.read_text(encoding="utf-8"))
    composition["segments"][0]["finalText"] = "tampered"
    composition["compositionSha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in composition.items()
            if key != "compositionSha256"
        }
    )
    _write(composition_path, composition)
    monkeypatch.setattr(
        "sys.argv",
        [
            "meta_analyze_semantic_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(Exception):
        main()


def test_meta_analysis_never_promotes_agent_review_to_human_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, output = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = copy.deepcopy(
        manifest["cases"][0]["agentSemanticReview"]
    )
    review["humanApproval"] = True
    manifest["cases"][0]["agentSemanticReview"] = review
    _write(manifest_path, manifest)
    monkeypatch.setattr(
        "sys.argv",
        [
            "meta_analyze_semantic_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match="fields do not match"):
        main()
