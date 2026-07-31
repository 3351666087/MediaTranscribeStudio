#!/usr/bin/env python3
"""Adjudicate frozen blind probes against their sealed reference.

This is intentionally a small, deterministic evaluator. It does not import
the production backend and never turns a metric into release approval. The
result is an evidence matrix for human/LLM semantic review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from rapidfuzz.distance import Levenshtein


SCHEMA_VERSION = "2.0.0"
_TOKEN = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[\w]+")


class FirstPrinciplesAuditError(ValueError):
    """Raised when frozen probe evidence cannot be safely compared."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FirstPrinciplesAuditError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FirstPrinciplesAuditError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _tokens(text: str | None) -> list[str]:
    if not isinstance(text, str):
        return []
    return [item.casefold() for item in _TOKEN.findall(text)]


def _text_error(reference: str | None, hypothesis: str | None) -> dict[str, Any] | None:
    truth = _tokens(reference)
    predicted = _tokens(hypothesis)
    if not truth:
        return None
    distance = Levenshtein.distance(truth, predicted)
    return {
        "referenceTokens": len(truth),
        "hypothesisTokens": len(predicted),
        "editDistance": distance,
        "normalizedError": round(distance / max(1, len(truth)), 6),
    }


def _target_chinese_script(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    han = 0
    foreign_letters = 0
    for character in text:
        codepoint = ord(character)
        if 0x3400 <= codepoint <= 0x9FFF:
            han += 1
        elif character.isalpha():
            foreign_letters += 1
    scored = han + foreign_letters
    score = han / scored if scored else 0.0
    return {
        "hanCharacters": han,
        "foreignLetters": foreign_letters,
        "hanFraction": round(score, 6),
        "targetScriptPass": han > 0 and score >= 0.5,
    }


def _mean(values: Sequence[float | int | None]) -> float | None:
    scored = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(scored) / len(scored), 6) if scored else None


def _language_code(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.casefold().replace("_", "-")
    aliases = {
        "english": "en",
        "en": "en",
        "chinese": "zh",
        "zh": "zh",
        "japanese": "ja",
        "ja": "ja",
        "korean": "ko",
        "ko": "ko",
        "spanish": "es",
        "es": "es",
        "portuguese": "pt",
        "pt": "pt",
        "german": "de",
        "de": "de",
        "french": "fr",
        "fr": "fr",
        "swahili": "sw",
        "sw": "sw",
        "filipino": "tl",
        "tagalog": "tl",
        "tl": "tl",
        "hindi": "hi",
        "hi": "hi",
    }
    return aliases.get(normalized, normalized.split("-", 1)[0])


def _interval_duration(intervals: Any) -> int:
    if not isinstance(intervals, list):
        return 0
    total = 0
    for item in intervals:
        if not isinstance(item, Mapping):
            continue
        start = item.get("startMs")
        end = item.get("endMs")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            total += end - start
    return total


def _probe_cases(path: Path, artifact_type: str) -> dict[str, Any]:
    report = _read(path)
    if report.get("artifactType") != artifact_type:
        raise FirstPrinciplesAuditError(f"{path} has unexpected artifact type")
    if report.get("truthAccessed") is not False:
        raise FirstPrinciplesAuditError(f"{path} is not a frozen blind report")
    return report


def _by_case(report: Mapping[str, Any], field: str) -> dict[str, dict[str, Any]]:
    rows = report.get("cases")
    if not isinstance(rows, list):
        raise FirstPrinciplesAuditError("probe cases must be an array")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get(field), str):
            raise FirstPrinciplesAuditError("probe case identity is invalid")
        key = str(row[field])
        if key in output:
            raise FirstPrinciplesAuditError(f"duplicate probe case {key}")
        output[key] = dict(row)
    return output


def adjudicate(
    *,
    blind_path: Path,
    oracle_path: Path,
    qwen_path: Path,
    whisper_path: Path,
    vad_path: Path,
    diarization_path: Path,
    output_path: Path,
    llm_path: Path | None = None,
) -> dict[str, Any]:
    blind = _read(blind_path)
    oracle = _read(oracle_path)
    if blind.get("artifactType") != "first-principles-blind-media-batch":
        raise FirstPrinciplesAuditError("blind artifact type is invalid")
    if oracle.get("artifactType") != "first-principles-sealed-reference":
        raise FirstPrinciplesAuditError("sealed reference artifact type is invalid")
    if blind.get("batchId") != oracle.get("batchId"):
        raise FirstPrinciplesAuditError("blind and sealed reference batches differ")
    sealed_sha = _sha256(oracle_path)
    if blind.get("sealedReference", {}).get("sha256") != sealed_sha:
        raise FirstPrinciplesAuditError("sealed reference hash binding failed")
    qwen = _probe_cases(qwen_path, "first-principles-asr-probe")
    whisper = _probe_cases(whisper_path, "first-principles-whisper-probe")
    vad = _probe_cases(vad_path, "first-principles-vad-probe")
    diarization = _probe_cases(
        diarization_path, "first-principles-diarization-probe"
    )
    reports = {
        "qwen": _by_case(qwen, "auditCaseId"),
        "whisper": _by_case(whisper, "auditCaseId"),
        "vad": _by_case(vad, "auditCaseId"),
        "diarization": _by_case(diarization, "auditCaseId"),
    }
    llm_report: dict[str, Any] | None = None
    if llm_path is not None:
        llm_report = _probe_cases(
            llm_path, "first-principles-local-llm-arbitration"
        )
        if llm_report.get("batchId") != blind.get("batchId"):
            raise FirstPrinciplesAuditError("LLM and blind batches differ")
        expected_llm_inputs = {
            "blind": _sha256(blind_path),
            "qwen": _sha256(qwen_path),
            "whisper": _sha256(whisper_path),
            "vad": _sha256(vad_path),
            "diarization": _sha256(diarization_path),
        }
        bound_inputs = llm_report.get("inputs")
        if not isinstance(bound_inputs, Mapping) or any(
            not isinstance(bound_inputs.get(name), Mapping)
            or bound_inputs[name].get("sha256") != sha256
            for name, sha256 in expected_llm_inputs.items()
        ):
            raise FirstPrinciplesAuditError("LLM frozen input binding failed")
        reports["semantic"] = _by_case(llm_report, "auditCaseId")
    oracle_rows = oracle.get("cases")
    blind_rows = blind.get("cases")
    if not isinstance(oracle_rows, list) or not isinstance(blind_rows, list):
        raise FirstPrinciplesAuditError("reference and blind cases must be arrays")
    blind_ids = {
        row.get("auditCaseId")
        for row in blind_rows
        if isinstance(row, Mapping)
    }
    if len(blind_ids) != len(blind_rows):
        raise FirstPrinciplesAuditError("blind case IDs are not unique")

    cases: list[dict[str, Any]] = []
    for reference in oracle_rows:
        if not isinstance(reference, Mapping):
            raise FirstPrinciplesAuditError("sealed reference case is invalid")
        audit_id = reference.get("auditCaseId")
        if not isinstance(audit_id, str) or audit_id not in blind_ids:
            raise FirstPrinciplesAuditError("reference case is not in blind batch")
        for name in ("qwen", "whisper", "vad", "diarization"):
            rows = reports[name]
            if audit_id not in rows:
                raise FirstPrinciplesAuditError(f"{name} is missing {audit_id}")
            if rows[audit_id].get("status") != "completed":
                raise FirstPrinciplesAuditError(f"{name} did not complete {audit_id}")
            expected_sha = next(
                row["media"]["sha256"]
                for row in blind_rows
                if row.get("auditCaseId") == audit_id
            )
            if rows[audit_id].get("mediaSha256") != expected_sha:
                raise FirstPrinciplesAuditError(f"{name} media binding failed for {audit_id}")

        truth = reference.get("reference")
        eligibility = reference.get("eligibility")
        if not isinstance(truth, Mapping) or not isinstance(eligibility, Mapping):
            raise FirstPrinciplesAuditError(f"{audit_id} truth shape is invalid")
        qwen_row = reports["qwen"][audit_id]
        whisper_row = reports["whisper"][audit_id]
        vad_row = reports["vad"][audit_id]
        diarization_row = reports["diarization"][audit_id]
        semantic_row = reports.get("semantic", {}).get(audit_id)
        semantic_proposal = (
            semantic_row.get("proposal")
            if isinstance(semantic_row, Mapping)
            and semantic_row.get("status") == "completed"
            and isinstance(semantic_row.get("proposal"), Mapping)
            else None
        )
        qwen_text = qwen_row.get("text") if isinstance(qwen_row.get("text"), str) else ""
        whisper_text = (
            whisper_row.get("text") if isinstance(whisper_row.get("text"), str) else ""
        )
        vad_ms = _interval_duration(vad_row.get("intervals"))
        qwen_lang = _language_code(qwen_row.get("language"))
        whisper_lang = _language_code(whisper_row.get("language"))
        reference_languages = [
            _language_code(value) for value in (truth.get("languages") or [])
        ]
        reference_languages = [value for value in reference_languages if value]
        lexical_truth = truth.get("lexicalSpeechPresent")
        observed_nonempty = bool(qwen_text or whisper_text)
        machine_findings: list[str] = []
        if vad_ms == 0 and observed_nonempty:
            machine_findings.append("asr-nonempty-with-zero-vad")
        if vad_ms > 0 and not observed_nonempty:
            machine_findings.append("vad-present-with-empty-asr")
        if qwen_text and whisper_text:
            similarity = Levenshtein.normalized_similarity(
                _tokens(qwen_text), _tokens(whisper_text)
            )
        else:
            similarity = None
        if similarity is not None and similarity < 0.5:
            machine_findings.append("cross-asr-text-divergence")
        if (
            reference_languages
            and qwen_lang
            and qwen_lang not in reference_languages
        ):
            machine_findings.append("qwen-document-language-disagrees")
        if (
            reference_languages
            and whisper_lang
            and whisper_lang not in reference_languages
        ):
            machine_findings.append("whisper-document-language-disagrees")
        speaker_truth = truth.get("speakerCount")
        speaker_observed = diarization_row.get("speakerCount")
        if (
            eligibility.get("speakerCount") is True
            and isinstance(speaker_truth, int)
            and speaker_observed != speaker_truth
        ):
            machine_findings.append("speaker-count-mismatch")
        if lexical_truth is False and observed_nonempty:
            machine_findings.append("non-lexical-or-no-speech-asr-hallucination-candidate")
        if lexical_truth is True and vad_ms == 0:
            machine_findings.append("lexical-speech-without-vad")
        qwen_error = (
            _text_error(truth.get("transcript"), qwen_text)
            if eligibility.get("transcript") is True
            else None
        )
        whisper_error = (
            _text_error(truth.get("transcript"), whisper_text)
            if eligibility.get("transcript") is True
            else None
        )
        semantic_text = (
            semantic_proposal.get("transcriptText")
            if isinstance(semantic_proposal, Mapping)
            and isinstance(semantic_proposal.get("transcriptText"), str)
            else ""
        )
        semantic_error = (
            _text_error(truth.get("transcript"), semantic_text)
            if eligibility.get("transcript") is True
            and isinstance(semantic_proposal, Mapping)
            else None
        )
        base_error_values = [
            error["normalizedError"]
            for error in (qwen_error, whisper_error)
            if isinstance(error, Mapping)
        ]
        best_base_error = min(base_error_values) if base_error_values else None
        semantic_normalized_error = (
            semantic_error.get("normalizedError")
            if isinstance(semantic_error, Mapping)
            else None
        )
        semantic_delta = (
            round(float(semantic_normalized_error) - float(best_base_error), 6)
            if isinstance(semantic_normalized_error, (int, float))
            and isinstance(best_base_error, (int, float))
            else None
        )
        semantic_languages = [
            _language_code(value)
            for value in (
                semantic_proposal.get("languages", [])
                if isinstance(semantic_proposal, Mapping)
                else []
            )
        ]
        semantic_languages = sorted(
            {value for value in semantic_languages if value}
        )
        semantic_speech = (
            semantic_proposal.get("lexicalSpeechPresent")
            if isinstance(semantic_proposal, Mapping)
            else None
        )
        semantic_count = (
            semantic_proposal.get("speakerCount")
            if isinstance(semantic_proposal, Mapping)
            else None
        )
        semantic_translation = (
            semantic_proposal.get("translationText")
            if isinstance(semantic_proposal, Mapping)
            else None
        )
        translation_script = (
            _target_chinese_script(semantic_translation)
            if semantic_speech is True
            else None
        )
        post_semantic_findings: list[str] = []
        if semantic_row is None:
            post_semantic_findings.append("semantic-result-missing")
        elif semantic_proposal is None:
            post_semantic_findings.append("semantic-structured-output-failed")
        if (
            isinstance(lexical_truth, bool)
            and isinstance(semantic_speech, bool)
            and semantic_speech != lexical_truth
        ):
            post_semantic_findings.append("semantic-speech-presence-mismatch")
        if (
            eligibility.get("speakerCount") is True
            and isinstance(speaker_truth, int)
            and semantic_count != speaker_truth
        ):
            post_semantic_findings.append("semantic-speaker-count-mismatch")
        if semantic_delta is not None and semantic_delta > 0:
            post_semantic_findings.append("semantic-transcript-worse-than-best-base")
        if (
            eligibility.get("languageDocument") is True
            and reference_languages
            and set(semantic_languages) != set(reference_languages)
        ):
            post_semantic_findings.append("semantic-language-set-mismatch")
        if (
            isinstance(translation_script, Mapping)
            and translation_script.get("targetScriptPass") is not True
        ):
            post_semantic_findings.append("semantic-translation-not-target-chinese")
        cases.append(
            {
                "auditCaseId": audit_id,
                "sourceCaseId": reference.get("sourceCaseId"),
                "collection": reference.get("collection"),
                "partition": reference.get("partition"),
                "referenceClass": reference.get("referenceClass"),
                "truth": {
                    "lexicalSpeechPresent": lexical_truth,
                    "speakerCount": speaker_truth,
                    "languages": reference_languages,
                    "transcript": truth.get("transcript"),
                    "eligibility": dict(eligibility),
                },
                "observed": {
                    "qwen": {
                        "language": qwen_row.get("language"),
                        "text": qwen_text,
                    },
                    "whisper": {
                        "language": whisper_row.get("language"),
                        "text": whisper_text,
                        "segments": whisper_row.get("segments", []),
                    },
                    "vad": {
                        "intervals": vad_row.get("intervals", []),
                        "speechMilliseconds": vad_ms,
                    },
                    "diarization": {
                        "speakerCount": speaker_observed,
                        "regularTurns": diarization_row.get("regularTurns", []),
                        "exclusiveTurns": diarization_row.get("exclusiveTurns", []),
                    },
                    "semantic": {
                        "status": (
                            semantic_row.get("status")
                            if isinstance(semantic_row, Mapping)
                            else "missing"
                        ),
                        "proposal": dict(semantic_proposal)
                        if isinstance(semantic_proposal, Mapping)
                        else None,
                        "failure": (
                            semantic_row.get("failure")
                            if isinstance(semantic_row, Mapping)
                            else None
                        ),
                        "constraintCorrections": (
                            semantic_row.get("constraintCorrections", [])
                            if isinstance(semantic_row, Mapping)
                            else []
                        ),
                        "inference": (
                            semantic_row.get("inference")
                            if isinstance(semantic_row, Mapping)
                            else None
                        ),
                    },
                },
                "metrics": {
                    "crossAsrNormalizedSimilarity": (
                        round(similarity, 6) if similarity is not None else None
                    ),
                    "qwenTranscriptError": qwen_error,
                    "whisperTranscriptError": whisper_error,
                    "bestBaseTranscriptError": best_base_error,
                    "semanticTranscriptError": semantic_error,
                    "semanticTranscriptDeltaVsBestBase": semantic_delta,
                    "speechPresenceMatch": (
                        bool(lexical_truth) == (vad_ms > 0 and observed_nonempty)
                        if isinstance(lexical_truth, bool)
                        else None
                    ),
                    "speakerCountError": (
                        abs(speaker_observed - speaker_truth)
                        if isinstance(speaker_observed, int)
                        and isinstance(speaker_truth, int)
                        else None
                    ),
                    "qwenDocumentLanguageMatch": (
                        qwen_lang in reference_languages
                        if reference_languages and qwen_lang
                        else None
                    ),
                    "whisperDocumentLanguageMatch": (
                        whisper_lang in reference_languages
                        if reference_languages and whisper_lang
                        else None
                    ),
                    "semanticStructuredCompleted": semantic_proposal is not None,
                    "semanticSpeechPresenceMatch": (
                        semantic_speech == lexical_truth
                        if isinstance(semantic_speech, bool)
                        and isinstance(lexical_truth, bool)
                        else None
                    ),
                    "semanticSpeakerCountError": (
                        abs(semantic_count - speaker_truth)
                        if eligibility.get("speakerCount") is True
                        and isinstance(semantic_count, int)
                        and isinstance(speaker_truth, int)
                        else None
                    ),
                    "semanticLanguageSetExact": (
                        set(semantic_languages) == set(reference_languages)
                        if eligibility.get("languageDocument") is True
                        and reference_languages
                        and isinstance(semantic_proposal, Mapping)
                        else None
                    ),
                    "semanticTranslationTargetScript": translation_script,
                },
                "machineFindings": machine_findings,
                "postSemanticFindings": post_semantic_findings,
                "manualReview": {
                    "status": "required",
                    "humanApproval": False,
                    "notes": [],
                    "translationReviewed": False,
                },
            }
        )

    result = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "first-principles-adjudication-matrix",
        "batchId": blind.get("batchId"),
        "truthAccessed": True,
        "releaseApproved": False,
        "promotionEligible": False,
        "method": {
            "principle": "freeze independent evidence before opening reference",
            "automaticMetricsAreDiagnosticOnly": True,
            "translationReferenceAvailable": False,
            "semanticArbitrationMandatory": llm_path is not None,
            "humanApprovalRequired": True,
        },
        "inputs": {
            "blind": {"path": str(blind_path.resolve()), "sha256": _sha256(blind_path)},
            "sealedReference": {
                "path": str(oracle_path.resolve()),
                "sha256": sealed_sha,
            },
            "qwen": {"path": str(qwen_path.resolve()), "sha256": _sha256(qwen_path)},
            "whisper": {
                "path": str(whisper_path.resolve()),
                "sha256": _sha256(whisper_path),
            },
            "vad": {"path": str(vad_path.resolve()), "sha256": _sha256(vad_path)},
            "diarization": {
                "path": str(diarization_path.resolve()),
                "sha256": _sha256(diarization_path),
            },
            "semantic": (
                {"path": str(llm_path.resolve()), "sha256": _sha256(llm_path)}
                if llm_path is not None
                else None
            ),
        },
        "coverage": {
            "caseCount": len(cases),
            "speechPresenceMatches": sum(
                row["metrics"]["speechPresenceMatch"] is True for row in cases
            ),
            "speakerCountScored": sum(
                row["metrics"]["speakerCountError"] is not None for row in cases
            ),
            "speakerCountExact": sum(
                row["metrics"]["speakerCountError"] == 0 for row in cases
            ),
            "transcriptScored": sum(
                row["metrics"]["qwenTranscriptError"] is not None for row in cases
            ),
            "manualReviewCases": len(cases),
            "semanticStructuredCompleted": sum(
                row["metrics"]["semanticStructuredCompleted"] is True
                for row in cases
            ),
            "semanticSpeechPresenceScored": sum(
                row["metrics"]["semanticSpeechPresenceMatch"] is not None
                for row in cases
            ),
            "semanticSpeechPresenceMatches": sum(
                row["metrics"]["semanticSpeechPresenceMatch"] is True
                for row in cases
            ),
            "semanticSpeakerCountScored": sum(
                row["metrics"]["semanticSpeakerCountError"] is not None
                for row in cases
            ),
            "semanticSpeakerCountExact": sum(
                row["metrics"]["semanticSpeakerCountError"] == 0
                for row in cases
            ),
            "semanticTranscriptScored": sum(
                row["metrics"]["semanticTranscriptError"] is not None
                for row in cases
            ),
            "meanQwenTranscriptError": _mean(
                row["metrics"]["qwenTranscriptError"]["normalizedError"]
                if isinstance(row["metrics"]["qwenTranscriptError"], Mapping)
                else None
                for row in cases
            ),
            "meanWhisperTranscriptError": _mean(
                row["metrics"]["whisperTranscriptError"]["normalizedError"]
                if isinstance(row["metrics"]["whisperTranscriptError"], Mapping)
                else None
                for row in cases
            ),
            "meanBestBaseTranscriptError": _mean(
                row["metrics"]["bestBaseTranscriptError"] for row in cases
            ),
            "meanSemanticTranscriptError": _mean(
                row["metrics"]["semanticTranscriptError"]["normalizedError"]
                if isinstance(row["metrics"]["semanticTranscriptError"], Mapping)
                else None
                for row in cases
            ),
            "semanticTranscriptBetterThanBestBase": sum(
                isinstance(row["metrics"]["semanticTranscriptDeltaVsBestBase"], (int, float))
                and row["metrics"]["semanticTranscriptDeltaVsBestBase"] < 0
                for row in cases
            ),
            "semanticTranscriptEqualToBestBase": sum(
                row["metrics"]["semanticTranscriptDeltaVsBestBase"] == 0
                for row in cases
            ),
            "semanticTranscriptWorseThanBestBase": sum(
                isinstance(row["metrics"]["semanticTranscriptDeltaVsBestBase"], (int, float))
                and row["metrics"]["semanticTranscriptDeltaVsBestBase"] > 0
                for row in cases
            ),
            "semanticLanguageSetScored": sum(
                row["metrics"]["semanticLanguageSetExact"] is not None
                for row in cases
            ),
            "semanticLanguageSetExact": sum(
                row["metrics"]["semanticLanguageSetExact"] is True
                for row in cases
            ),
            "semanticTranslationTargetScriptScored": sum(
                row["metrics"]["semanticTranslationTargetScript"] is not None
                for row in cases
            ),
            "semanticTranslationTargetScriptPassed": sum(
                isinstance(row["metrics"]["semanticTranslationTargetScript"], Mapping)
                and row["metrics"]["semanticTranslationTargetScript"].get(
                    "targetScriptPass"
                )
                is True
                for row in cases
            ),
            "semanticConstraintCorrectedCases": sum(
                bool(row["observed"]["semantic"]["constraintCorrections"])
                for row in cases
            ),
        },
        "cases": cases,
    }
    _write(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blind", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--vad", type=Path, required=True)
    parser.add_argument("--diarization", type=Path, required=True)
    parser.add_argument("--llm", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = adjudicate(
        blind_path=args.blind,
        oracle_path=args.oracle,
        qwen_path=args.qwen,
        whisper_path=args.whisper,
        vad_path=args.vad,
        diarization_path=args.diarization,
        output_path=args.output,
        llm_path=args.llm,
    )
    print(json.dumps(result["coverage"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
