from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from backend.persistence import canonical_json_sha256
from tools.unblind_semantic_model_review import (
    ARTIFACT_TYPE,
    SemanticModelUnblindError,
    main,
    unblind_semantic_model_review,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_canonical(value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["canonicalSha256"] = canonical_json_sha256(value)
    return body


def _fixture(
    tmp_path: Path,
    *,
    tie: bool = False,
    preferred_model_indexes: tuple[int, ...] = (1,),
    candidate_count: int = 6,
    ollama_model_refs: bool = False,
) -> dict[str, Path | str]:
    package = tmp_path / "blind-package"
    reviewer_root = package / "reviewer"
    model_rows = [
        {
            "model": (
                f"qwen3.5:{index + 1}b-q4_K_M"
                if ollama_model_refs
                else f"model-runtime-{index + 1}"
            ),
            "modelId": (
                f"qwen3.5:{index + 1}b-q4_K_M"
                if ollama_model_refs
                else f"model-{index + 1}"
            ),
            "digest": f"sha256:{hashlib.sha256(f'model-{index + 1}'.encode()).hexdigest()}",
        }
        for index in range(candidate_count)
    ]
    packet_values: dict[str, dict[str, Any]] = {}
    vault_cases: list[dict[str, Any]] = []
    review_cases: list[dict[str, Any]] = []
    for case_number in range(1, 23):
        case_alias = f"case-{case_number:03d}"
        shared_input = {
            "language": "en-US",
            "rawAsrSegments": [
                {
                    "segmentId": f"segment-{case_number}",
                    "startMs": 0,
                    "endMs": 1_000,
                    "speakerId": "speaker-1",
                    "language": "en-US",
                    "rawText": f"asr-{case_number}",
                    "overlapping": False,
                }
            ],
            "speakerTimeline": [
                {
                    "startMs": 0,
                    "endMs": 1_000,
                    "speakerId": "speaker-1",
                    "overlapping": False,
                }
            ],
        }
        # Rotate aliases in half the cases so unblinding must use the vault
        # mapping instead of assuming candidate-01 is the same model.
        alias_to_model_index = {
            f"candidate-{alias_index:02d}": (
                (alias_index + case_number - 1) % candidate_count
            )
            for alias_index in range(1, candidate_count + 1)
        }
        candidates = []
        vault_candidates = []
        for alias, model_index in alias_to_model_index.items():
            result = {
                "status": "composition-complete",
                "caseNumber": case_number,
                "candidate": alias,
            }
            result_sha = canonical_json_sha256(result)
            candidates.append(
                {
                    "candidateAlias": alias,
                    "resultCanonicalSha256": result_sha,
                    "result": result,
                }
            )
            model = model_rows[model_index]
            vault_candidates.append(
                {
                    "candidateAlias": alias,
                    "model": model["model"],
                    "modelId": model["modelId"],
                    "digest": model["digest"],
                    "resultCanonicalSha256": result_sha,
                }
            )
        packet = _with_canonical(
            {
                "schemaVersion": "1.0.0",
                "artifactType": "anonymous-semantic-blind-review-packet",
                "caseAlias": case_alias,
                "sharedInputCanonicalSha256": canonical_json_sha256(shared_input),
                "sharedInput": shared_input,
                "candidates": candidates,
            }
        )
        packet_path = reviewer_root / f"{case_alias}.json"
        _write_json(packet_path, packet)
        packet_values[case_alias] = packet
        case_sha = hashlib.sha256(f"case-{case_number}".encode()).hexdigest()
        vault_cases.append(
            {
                "caseAlias": case_alias,
                "caseId": f"source-case-{case_number:03d}",
                "caseSha256": case_sha,
                "candidates": vault_candidates,
            }
        )

        preferred_aliases = [
            alias
            for alias, model_index in alias_to_model_index.items()
            if model_index in preferred_model_indexes
        ]
        if not tie and len(preferred_aliases) != 1:
            raise AssertionError("fixture requires exactly one preferred model")
        groups: list[dict[str, Any]] = []
        for alias, model_index in alias_to_model_index.items():
            if model_index in preferred_model_indexes:
                severity = "pass"
            elif model_index == 0:
                severity = "major"
            else:
                severity = "minor"
            groups.append(
                {
                    "candidateAliases": [alias],
                    "severity": severity,
                    "reason": "Fixture Codex review observation.",
                    "evidence": {
                        "startMs": 0,
                        "endMs": 1_000,
                        "observation": "Fixture evidence.",
                    },
                    "recommendedAction": "Preserve or challenge according to the review.",
                }
            )
        review_cases.append(
            {
                "caseAlias": case_alias,
                "preferredCandidateAliases": preferred_aliases,
                "tie": tie,
                "assessmentGroups": groups,
            }
        )

    vault = _with_canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "anonymous-semantic-blind-review-identity-vault",
            "seedSha256": "a" * 64,
            "inputSetCanonicalSha256": canonical_json_sha256(
                [
                    {
                        "caseId": row["caseId"],
                        "caseSha256": row["caseSha256"],
                        "sharedInputCanonicalSha256": packet_values[
                            row["caseAlias"]
                        ]["sharedInputCanonicalSha256"],
                    }
                    for row in vault_cases
                ]
            ),
            "modelSetCanonicalSha256": canonical_json_sha256(model_rows),
            "cases": vault_cases,
        }
    )
    vault_path = package / "identity-vault.json"
    _write_json(vault_path, vault)

    manifest_files = [
        {
            "role": "identity-vault",
            "relativePath": "identity-vault.json",
            "canonicalSha256": vault["canonicalSha256"],
            "fileSha256": _file_sha(vault_path),
            "sizeBytes": vault_path.stat().st_size,
        }
    ]
    for case_number in range(1, 23):
        case_alias = f"case-{case_number:03d}"
        packet_path = reviewer_root / f"{case_alias}.json"
        packet = packet_values[case_alias]
        manifest_files.append(
            {
                "role": "reviewer-packet",
                "relativePath": f"reviewer/{case_alias}.json",
                "canonicalSha256": packet["canonicalSha256"],
                "fileSha256": _file_sha(packet_path),
                "sizeBytes": packet_path.stat().st_size,
            }
        )
    manifest = _with_canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "anonymous-semantic-blind-review-manifest",
            "publicationPolicy": "atomic-directory-no-replace",
            "manifestWrittenLast": True,
            "packetCount": 22,
            "candidatesPerPacket": candidate_count,
            "files": manifest_files,
        }
    )
    manifest_path = package / "manifest.json"
    _write_json(manifest_path, manifest)
    packet_set_sha = canonical_json_sha256(
        [
            {
                "caseAlias": f"case-{case_number:03d}",
                "fileSha256": manifest_files[case_number]["fileSha256"],
                "canonicalSha256": manifest_files[case_number]["canonicalSha256"],
            }
            for case_number in range(1, 23)
        ]
    )
    review = _with_canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "codex-semantic-blind-review",
            "reviewId": "fixture-semantic-blind-review-r1",
            "blindPackage": {
                "manifestFileSha256": _file_sha(manifest_path),
                "packetCount": 22,
                "candidatesPerPacket": candidate_count,
                "manifestCanonicalSha256": manifest["canonicalSha256"],
                "reviewerPacketSetSha256": packet_set_sha,
            },
            "reviewPolicy": {
                "automaticScoringUsed": False,
                "referenceTranscriptUsed": False,
                "identityVaultReadBeforeSealing": False,
                "allowedSeverities": ["blocker", "major", "minor", "pass"],
                "preferenceRule": "Fixture manual preference rule.",
            },
            "reviewer": {
                "source": "codex-agent",
                "actor": "fixture-reviewer",
                "reviewedAt": "2026-08-10T00:00:00Z",
            },
            "cases": review_cases,
            "bodySourceFileSha256": "b" * 64,
            "validation": {
                "reviewedCaseCount": 22,
                "assessedCandidateCount": 22 * candidate_count,
                "allCandidatesAssessedExactlyOnce": True,
                "identityVaultRead": False,
                "sealedBeforeUnblind": True,
            },
        }
    )
    review_path = tmp_path / "sealed-review.json"
    _write_json(review_path, review)
    return {
        "package": package,
        "review": review_path,
        "output": tmp_path / "comparison.json",
        "manifest": manifest_path,
        "vault": vault_path,
    }


def test_unblind_selects_unique_winner_and_binds_all_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = unblind_semantic_model_review(
        package_root=fixture["package"],
        sealed_review_path=fixture["review"],
        output_path=fixture["output"],
    )
    assert value["artifactType"] == ARTIFACT_TYPE
    assert value["blindBatchId"].startswith("semantic-blind-")
    assert value["caseCount"] == 22
    assert value["candidateCount"] == 6
    assert value["uniqueWinnerRegistryModelId"] == "model-2"
    assert value["outcome"] == "unique-winner"
    assert value["tiedRegistryModelIds"] == ["model-2"]
    assert len(value["models"]) == 6
    assert len(value["aggregate"]) == 6
    assert value["aggregate"][0]["registryModelId"] == "model-2"
    assert value["evidence"]["sealedReviewFileSha256"] == _file_sha(fixture["review"])
    assert value["evidence"]["manifestFileSha256"] == _file_sha(fixture["manifest"])
    assert value["evidence"]["identityVaultFileSha256"] == _file_sha(fixture["vault"])
    persisted = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    assert persisted == value


def test_unblind_reports_tie_when_weighted_preference_is_equal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, tie=True, preferred_model_indexes=(1, 2))
    value = unblind_semantic_model_review(
        package_root=fixture["package"],
        sealed_review_path=fixture["review"],
        output_path=fixture["output"],
    )
    assert value["outcome"] == "tie"
    assert value["uniqueWinnerRegistryModelId"] is None
    assert value["tiedRegistryModelIds"] == ["model-2", "model-3"]
    assert value["aggregate"][0]["weightedPreferredShare"] == value["aggregate"][1][
        "weightedPreferredShare"
    ]


def test_unblind_supports_four_candidate_challenger_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, candidate_count=4)
    value = unblind_semantic_model_review(
        package_root=fixture["package"],
        sealed_review_path=fixture["review"],
        output_path=fixture["output"],
    )
    assert value["candidateCount"] == 4
    assert len(value["models"]) == 4
    assert len(value["aggregate"]) == 4
    assert value["uniqueWinnerRegistryModelId"] == "model-2"


def test_unblind_accepts_ollama_quantized_model_reference(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, candidate_count=4, ollama_model_refs=True)
    value = unblind_semantic_model_review(
        package_root=fixture["package"],
        sealed_review_path=fixture["review"],
        output_path=fixture["output"],
    )
    assert value["candidateCount"] == 4
    assert value["models"][0]["registryModelId"].startswith("qwen3.5:")


def test_review_seal_is_validated_before_identity_vault_is_opened(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    review = json.loads(Path(fixture["review"]).read_text(encoding="utf-8"))
    review["validation"]["sealedBeforeUnblind"] = False
    review.pop("canonicalSha256")
    review["canonicalSha256"] = canonical_json_sha256(review)
    _write_json(Path(fixture["review"]), review)
    Path(fixture["vault"]).write_text("not-json\n", encoding="utf-8")

    with pytest.raises(SemanticModelUnblindError, match="sealedBeforeUnblind"):
        unblind_semantic_model_review(
            package_root=fixture["package"],
            sealed_review_path=fixture["review"],
            output_path=fixture["output"],
        )


def test_manifest_file_sha_is_validated_before_identity_vault_is_opened(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    review = json.loads(Path(fixture["review"]).read_text(encoding="utf-8"))
    review["blindPackage"]["manifestFileSha256"] = "f" * 64
    review.pop("canonicalSha256")
    review["canonicalSha256"] = canonical_json_sha256(review)
    _write_json(Path(fixture["review"]), review)
    Path(fixture["vault"]).write_text("not-json\n", encoding="utf-8")

    with pytest.raises(SemanticModelUnblindError, match="manifest file SHA"):
        unblind_semantic_model_review(
            package_root=fixture["package"],
            sealed_review_path=fixture["review"],
            output_path=fixture["output"],
        )


def test_packet_result_sha_mismatch_fails_before_identity_vault_is_opened(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    packet_path = Path(fixture["package"]) / "reviewer" / "case-001.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["candidates"][0]["result"]["tampered"] = True
    packet.pop("canonicalSha256")
    packet["canonicalSha256"] = canonical_json_sha256(packet)
    _write_json(packet_path, packet)
    manifest_path = Path(fixture["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["files"] if row["relativePath"] == "reviewer/case-001.json"
    )
    packet_row["canonicalSha256"] = packet["canonicalSha256"]
    packet_row["fileSha256"] = _file_sha(packet_path)
    packet_row["sizeBytes"] = packet_path.stat().st_size
    manifest.pop("canonicalSha256")
    manifest["canonicalSha256"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    review_path = Path(fixture["review"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["blindPackage"]["manifestFileSha256"] = _file_sha(manifest_path)
    review["blindPackage"]["manifestCanonicalSha256"] = manifest["canonicalSha256"]
    review["blindPackage"]["reviewerPacketSetSha256"] = canonical_json_sha256(
        [
            {
                "caseAlias": f"case-{case_number:03d}",
                "fileSha256": next(
                    row for row in manifest["files"]
                    if row["relativePath"] == f"reviewer/case-{case_number:03d}.json"
                )["fileSha256"],
                "canonicalSha256": next(
                    row for row in manifest["files"]
                    if row["relativePath"] == f"reviewer/case-{case_number:03d}.json"
                )["canonicalSha256"],
            }
            for case_number in range(1, 23)
        ]
    )
    review.pop("canonicalSha256")
    review["canonicalSha256"] = canonical_json_sha256(review)
    _write_json(review_path, review)
    Path(fixture["vault"]).write_text("not-json\n", encoding="utf-8")

    with pytest.raises(SemanticModelUnblindError, match="result SHA"):
        unblind_semantic_model_review(
            package_root=fixture["package"],
            sealed_review_path=fixture["review"],
            output_path=fixture["output"],
        )


def test_identity_vault_alias_result_binding_is_checked_after_packets(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    vault_path = Path(fixture["vault"])
    vault = json.loads(vault_path.read_text(encoding="utf-8"))
    vault["cases"][0]["candidates"][0]["resultCanonicalSha256"] = "e" * 64
    vault.pop("canonicalSha256")
    vault["canonicalSha256"] = canonical_json_sha256(vault)
    _write_json(vault_path, vault)
    manifest_path = Path(fixture["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["files"][0]
    row["canonicalSha256"] = vault["canonicalSha256"]
    row["fileSha256"] = _file_sha(vault_path)
    row["sizeBytes"] = vault_path.stat().st_size
    manifest.pop("canonicalSha256")
    manifest["canonicalSha256"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    review_path = Path(fixture["review"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["blindPackage"]["manifestFileSha256"] = _file_sha(manifest_path)
    review["blindPackage"]["manifestCanonicalSha256"] = manifest["canonicalSha256"]
    review.pop("canonicalSha256")
    review["canonicalSha256"] = canonical_json_sha256(review)
    _write_json(review_path, review)

    with pytest.raises(SemanticModelUnblindError, match="alias/result SHA"):
        unblind_semantic_model_review(
            package_root=fixture["package"],
            sealed_review_path=review_path,
            output_path=fixture["output"],
        )


def test_output_is_no_replace(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = Path(fixture["output"])
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        unblind_semantic_model_review(
            package_root=fixture["package"],
            sealed_review_path=fixture["review"],
            output_path=output,
        )


def test_cli_writes_comparison(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    assert (
        main(
            [
                "--package-root",
                str(fixture["package"]),
                "--sealed-review",
                str(fixture["review"]),
                "--output",
                str(fixture["output"]),
            ]
        )
        == 0
    )
    value = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    assert value["artifactType"] == ARTIFACT_TYPE
