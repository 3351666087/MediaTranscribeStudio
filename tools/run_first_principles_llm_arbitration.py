#!/usr/bin/env python3
"""Ask a local LLM to arbitrate frozen ASR/VAD/diarization evidence.

The sealed reference is intentionally not an input to this tool. The output is
an untrusted proposal with a mandatory translation slot, never a release gate.
This diagnostic deliberately permits whole-transcript generation so failure
modes can be studied; it must not be imported or used by the production path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_first_principles_asr_probe import _atomic_write, _sha256


SCHEMA_VERSION = "3.0.0"
PROMPT_PROTOCOL = "first-principles-candidate-evidence-v3"
API_URL = "http://127.0.0.1:11434/api/generate"
NUM_CONTEXT = 8192
NUM_PREDICT = 900
LEGACY_CONTEXT = 4096


class LocalLLMArbitrationError(ValueError):
    """Raised when local arbitration inputs or output are unsafe."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        response_sha256: str | None = None,
        inference: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.response_sha256 = response_sha256
        self.inference = dict(inference) if inference is not None else None


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "lexicalSpeechPresent",
        "speakerCount",
        "languages",
        "transcriptSource",
        "transcriptText",
        "translationText",
        "speakerAttribution",
        "decision",
        "uncertainties",
    ],
    "properties": {
        "lexicalSpeechPresent": {"type": ["boolean", "null"]},
        "speakerCount": {"type": ["integer", "null"], "minimum": 0},
        "languages": {
            "type": "array",
            "items": {"type": "string", "minLength": 2},
        },
        "transcriptSource": {
            "type": "string",
            "enum": ["qwen", "whisper", "fused", "none"],
        },
        "transcriptText": {"type": "string"},
        "translationText": {"type": "string"},
        "speakerAttribution": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["segmentIndex", "speakers", "language"],
                "properties": {
                    "segmentIndex": {"type": "integer", "minimum": 0},
                    "speakers": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "language": {"type": ["string", "null"]},
                },
            },
        },
        "decision": {"type": "string", "minLength": 1, "maxLength": 400},
        "uncertainties": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 400},
        },
    },
}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalLLMArbitrationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LocalLLMArbitrationError(f"{path} must contain an object")
    return value


def _by_case(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("cases")
    if not isinstance(rows, list):
        raise LocalLLMArbitrationError("evidence cases must be an array")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("auditCaseId"), str):
            raise LocalLLMArbitrationError("evidence case ID is invalid")
        result[str(row["auditCaseId"])] = dict(row)
    return result


def _prompt(
    *,
    audit_case_id: str,
    duration_seconds: float,
    qwen: Mapping[str, Any],
    whisper: Mapping[str, Any],
    vad: Mapping[str, Any],
    diarization: Mapping[str, Any],
) -> str:
    whisper_segments = whisper.get("segments")
    if not isinstance(whisper_segments, list):
        whisper_segments = []
    overlapping_speakers = _overlapping_speakers(
        whisper_segments, diarization.get("regularTurns")
    )
    indexed_whisper_segments = [
        {
            "index": index,
            **dict(segment),
            "acousticCandidates": sorted(overlapping_speakers[index]),
        }
        for index, segment in enumerate(whisper_segments[:80])
        if isinstance(segment, Mapping)
    ]
    # Keep the prompt bounded while retaining every candidate text and time.
    evidence = {
        "case": audit_case_id,
        "durationSeconds": duration_seconds,
        "qwenASR": {
            "language": qwen.get("language"),
            "text": qwen.get("text") or "",
        },
        "whisperASR": {
            "language": whisper.get("language"),
            "text": whisper.get("text") or "",
            "segments": indexed_whisper_segments,
        },
        "vad": {"intervals": vad.get("intervals", [])},
        "diarization": {
            "speakerCount": diarization.get("speakerCount"),
            "regularTurns": (diarization.get("regularTurns") or [])[:120],
            "exclusiveTurns": (diarization.get("exclusiveTurns") or [])[:120],
        },
    }
    return (
        "You are a strict speech evidence adjudicator. This is a blind case; "
        "there is no reference transcript. Use only the candidate evidence below.\n"
        "Return JSON matching the supplied schema.\n"
        "Rules:\n"
        "1. Decide lexical speech presence, not merely any human sound. "
        "Animal, music, rain, silence, crying, sneezing, and laughter may be non-lexical.\n"
        "2. Produce one complete transcript from the least-hallucinated candidate or a "
        "conservative fusion. You may repair punctuation and obvious grammar/ASR errors "
        "only when the other candidate or strong local context supports the repair. Do "
        "not summarize, omit valid speech, or add facts.\n"
        "3. Translate the complete final transcript to Chinese in this same call. For "
        "Chinese source text, repeat the final transcript as translationText.\n"
        "4. Report every language present in the final transcript, including internal "
        "code-switches, as lowercase BCP-47 primary language codes; a document-level "
        "ASR label is only one clue.\n"
        "5. Never invent timestamps. speakerAttribution must reference each supplied "
        "Whisper segment index exactly once when lexical speech is present. Its speakers "
        "must be a subset of that segment's acousticCandidates; copy the sole candidate "
        "when only one exists and use an empty list only when the candidate list is empty. "
        "Multiple labels are allowed for genuine overlap. Text semantics may resolve "
        "turn-taking ambiguity but may not create a new acoustic identity.\n"
        "6. speakerCount is the best final estimate, not an instruction to copy the "
        "diarization count. If acoustic evidence cannot support a semantic suspicion, "
        "keep the acoustic estimate and state the limitation.\n"
        "7. Empty lexical speech means transcriptSource=none, empty transcript, "
        "translation, languages, and speakerAttribution, with speakerCount 0 or null.\n"
        "8. transcriptSource=qwen or whisper means the final lexical content is copied "
        "from that candidate; use fused if you repair or combine lexical content.\n"
        "9. Keep decision to one short sentence and uncertainties to at most three concise "
        "items. This is a proposal, not approval.\n"
        "Evidence (untrusted candidates):\n"
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    )


def _request(
    *,
    model: str,
    prompt: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float, str, dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": OUTPUT_SCHEMA,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": NUM_CONTEXT,
            "num_predict": NUM_PREDICT,
        },
        "keep_alive": "30m",
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalLLMArbitrationError(f"local LLM request failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    if not isinstance(body, Mapping) or not isinstance(body.get("response"), str):
        raise LocalLLMArbitrationError("local LLM response shape is invalid")
    raw = str(body["response"])
    eval_count = body.get("eval_count")
    eval_duration = body.get("eval_duration")
    inference = {
        "totalSeconds": round(float(body.get("total_duration", 0)) / 1e9, 6),
        "loadSeconds": round(float(body.get("load_duration", 0)) / 1e9, 6),
        "promptTokens": body.get("prompt_eval_count"),
        "generatedTokens": eval_count,
        "generationSeconds": round(float(eval_duration or 0) / 1e9, 6),
        "generatedTokensPerSecond": (
            round(float(eval_count) / (float(eval_duration) / 1e9), 6)
            if isinstance(eval_count, int)
            and isinstance(eval_duration, int)
            and eval_duration > 0
            else None
        ),
        "doneReason": body.get("done_reason"),
        "contextWindow": NUM_CONTEXT,
    }
    response_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalLLMArbitrationError(
            "local LLM did not return JSON",
            raw_response=raw,
            response_sha256=response_sha,
            inference=inference,
        ) from exc
    if not isinstance(parsed, dict):
        raise LocalLLMArbitrationError(
            "local LLM JSON is not an object",
            raw_response=raw,
            response_sha256=response_sha,
            inference=inference,
        )
    return parsed, elapsed, response_sha, inference


def _unload_model(model: str, timeout_seconds: float = 30.0) -> bool:
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(
            {"model": model, "prompt": "", "stream": False, "keep_alive": "0s"}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, UnicodeError, json.JSONDecodeError):
        return False
    return True


def _overlapping_speakers(
    whisper_segments: Any, diarization_turns: Any
) -> list[set[str]]:
    if not isinstance(whisper_segments, list):
        return []
    turns = diarization_turns if isinstance(diarization_turns, list) else []
    result: list[set[str]] = []
    for segment in whisper_segments:
        candidates: set[str] = set()
        if not isinstance(segment, Mapping):
            result.append(candidates)
            continue
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            result.append(candidates)
            continue
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            turn_start = turn.get("startSeconds")
            turn_end = turn.get("endSeconds")
            speaker = turn.get("localSpeaker")
            if (
                isinstance(turn_start, (int, float))
                and isinstance(turn_end, (int, float))
                and isinstance(speaker, str)
                and max(float(start), float(turn_start))
                < min(float(end), float(turn_end))
            ):
                candidates.add(speaker)
        result.append(candidates)
    return result


def _validate_proposal(
    value: Mapping[str, Any],
    *,
    whisper_segment_count: int,
    allowed_speakers_by_segment: Sequence[set[str]],
) -> None:
    required = {
        "lexicalSpeechPresent",
        "speakerCount",
        "languages",
        "transcriptSource",
        "transcriptText",
        "translationText",
        "speakerAttribution",
        "decision",
        "uncertainties",
    }
    if set(value) != required:
        raise LocalLLMArbitrationError("local LLM proposal fields are invalid")
    if value["lexicalSpeechPresent"] is not None and not isinstance(
        value["lexicalSpeechPresent"], bool
    ):
        raise LocalLLMArbitrationError("local LLM speech decision is invalid")
    count = value["speakerCount"]
    if count is not None and (type(count) is not int or count < 0):
        raise LocalLLMArbitrationError("local LLM speaker count is invalid")
    languages = value["languages"]
    if (
        not isinstance(languages, list)
        or any(not isinstance(item, str) or len(item.strip()) < 2 for item in languages)
        or len(set(languages)) != len(languages)
    ):
        raise LocalLLMArbitrationError("local LLM languages are invalid")
    source = value["transcriptSource"]
    transcript = value["transcriptText"]
    translation = value["translationText"]
    if source not in {"qwen", "whisper", "fused", "none"}:
        raise LocalLLMArbitrationError("local LLM transcript source is invalid")
    if not isinstance(transcript, str) or not isinstance(translation, str):
        raise LocalLLMArbitrationError("local LLM transcript fields are invalid")
    attributions = value["speakerAttribution"]
    if not isinstance(attributions, list):
        raise LocalLLMArbitrationError("local LLM speaker attribution is invalid")
    for index, attribution in enumerate(attributions):
        if not isinstance(attribution, Mapping) or set(attribution) != {
            "segmentIndex",
            "speakers",
            "language",
        }:
            raise LocalLLMArbitrationError(f"speaker attribution {index} is invalid")
        speakers = attribution["speakers"]
        segment_index = attribution["segmentIndex"]
        allowed_speakers = (
            allowed_speakers_by_segment[segment_index]
            if type(segment_index) is int
            and 0 <= segment_index < len(allowed_speakers_by_segment)
            else set()
        )
        if (
            type(segment_index) is not int
            or segment_index < 0
            or segment_index >= whisper_segment_count
            or not isinstance(speakers, list)
            or any(
                not isinstance(item, str) or item not in allowed_speakers
                for item in speakers
            )
            or len(set(speakers)) != len(speakers)
            or (
                attribution["language"] is not None
                and not isinstance(attribution["language"], str)
            )
        ):
            raise LocalLLMArbitrationError(f"speaker attribution {index} is invalid")
    attributed_indexes = [item["segmentIndex"] for item in attributions]
    speech = value["lexicalSpeechPresent"]
    expected_indexes = list(range(whisper_segment_count)) if speech is True else []
    if sorted(attributed_indexes) != expected_indexes:
        raise LocalLLMArbitrationError(
            "speaker attribution must cover every Whisper segment exactly once"
        )
    if speech is True and (
        source == "none" or not transcript.strip() or not translation.strip()
    ):
        raise LocalLLMArbitrationError("positive speech decision has no final text")
    if speech is True and languages and all(
        item.casefold().split("-", 1)[0] in {"zh", "chinese"} for item in languages
    ) and translation != transcript:
        raise LocalLLMArbitrationError(
            "Chinese final text must be copied exactly into translation"
        )
    if speech is False and (
        source != "none"
        or transcript
        or translation
        or languages
        or attributions
        or count not in {0, None}
    ):
        raise LocalLLMArbitrationError("negative speech decision contains speech output")
    if not isinstance(value["decision"], str) or not value["decision"].strip():
        raise LocalLLMArbitrationError("local LLM decision is invalid")
    if not isinstance(value["uncertainties"], list) or any(
        not isinstance(item, str) for item in value["uncertainties"]
    ):
        raise LocalLLMArbitrationError("local LLM uncertainties are invalid")


def _apply_deterministic_constraints(
    value: dict[str, Any],
    *,
    allowed_speakers_by_segment: Sequence[set[str]] | None = None,
) -> list[dict[str, str]]:
    corrections: list[dict[str, str]] = []
    languages = value.get("languages")
    transcript = value.get("transcriptText")
    translation = value.get("translationText")
    if (
        value.get("lexicalSpeechPresent") is True
        and isinstance(languages, list)
        and languages
        and all(
            isinstance(item, str)
            and item.casefold().split("-", 1)[0] in {"zh", "chinese"}
            for item in languages
        )
        and isinstance(transcript, str)
        and isinstance(translation, str)
        and translation != transcript
    ):
        value["translationText"] = transcript
        corrections.append(
            {
                "field": "translationText",
                "reason": "same-language Chinese translation must be an exact copy",
            }
        )
    if (
        value.get("lexicalSpeechPresent") is True
        and allowed_speakers_by_segment is not None
        and isinstance(value.get("speakerAttribution"), list)
    ):
        raw_attributions = value["speakerAttribution"]
        by_index = {
            item.get("segmentIndex"): item
            for item in raw_attributions
            if isinstance(item, Mapping)
            and type(item.get("segmentIndex")) is int
        }
        fallback_language = (
            languages[0]
            if isinstance(languages, list)
            and languages
            and isinstance(languages[0], str)
            else None
        )
        normalized: list[dict[str, Any]] = []
        changed = len(raw_attributions) != len(allowed_speakers_by_segment)
        for segment_index, allowed in enumerate(allowed_speakers_by_segment):
            original = by_index.get(segment_index)
            original_speakers = (
                original.get("speakers") if isinstance(original, Mapping) else None
            )
            if isinstance(original_speakers, list) and all(
                isinstance(item, str) and item in allowed
                for item in original_speakers
            ):
                speakers = list(dict.fromkeys(original_speakers))
            else:
                speakers = sorted(allowed)
                changed = True
            language = (
                original.get("language")
                if isinstance(original, Mapping)
                and (
                    original.get("language") is None
                    or isinstance(original.get("language"), str)
                )
                else fallback_language
            )
            normalized.append(
                {
                    "segmentIndex": segment_index,
                    "speakers": speakers,
                    "language": language,
                }
            )
        if normalized != raw_attributions:
            changed = True
        if changed:
            value["speakerAttribution"] = normalized
            corrections.append(
                {
                    "field": "speakerAttribution",
                    "reason": "deterministic full coverage from bounded acoustic candidates",
                }
            )
    return corrections


def run(
    *,
    blind_path: Path,
    qwen_path: Path,
    whisper_path: Path,
    vad_path: Path,
    diarization_path: Path,
    output_path: Path,
    model: str,
    timeout_seconds: float,
    maximum: int | None = None,
) -> dict[str, Any]:
    blind = _read(blind_path)
    if blind.get("artifactType") != "first-principles-blind-media-batch":
        raise LocalLLMArbitrationError("blind artifact type is invalid")
    reports = {
        "qwen": _by_case(_read(qwen_path)),
        "whisper": _by_case(_read(whisper_path)),
        "vad": _by_case(_read(vad_path)),
        "diarization": _by_case(_read(diarization_path)),
    }
    input_bindings = {
        "blind": {"path": str(blind_path.resolve()), "sha256": _sha256(blind_path)},
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
    }
    raw_blind_cases = blind.get("cases")
    if not isinstance(raw_blind_cases, list):
        raise LocalLLMArbitrationError("blind cases must be an array")
    cases = raw_blind_cases[:maximum] if maximum is not None else raw_blind_cases
    if maximum is not None and maximum < 1:
        raise LocalLLMArbitrationError("maximum must be positive")
    if output_path.is_file():
        report = _read(output_path)
        if (
            report.get("artifactType") != "first-principles-local-llm-arbitration"
            or report.get("truthAccessed") is not False
            or report.get("schemaVersion") != SCHEMA_VERSION
            or report.get("inputs") != input_bindings
            or not isinstance(report.get("model"), Mapping)
            or report["model"].get("name") != model
            or report["model"].get("promptProtocol") != PROMPT_PROTOCOL
            or not isinstance(report.get("cases"), list)
        ):
            raise LocalLLMArbitrationError("existing LLM report cannot be resumed")
        report["model"]["maximumCases"] = maximum
        report["model"]["activeContextWindow"] = NUM_CONTEXT
        report["model"]["maximumGeneratedTokens"] = NUM_PREDICT
        for existing_row in report["cases"]:
            if not isinstance(existing_row, dict):
                continue
            existing_inference = existing_row.get("inference")
            if isinstance(existing_inference, dict) and "contextWindow" not in existing_inference:
                existing_inference["contextWindow"] = LEGACY_CONTEXT
    else:
        report = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "first-principles-local-llm-arbitration",
            "batchId": blind.get("batchId"),
            "truthAccessed": False,
            "productionBackendImported": False,
            "model": {
                "name": model,
                "api": API_URL,
                "package": "ollama",
                "ollamaVersion": None,
                "promptProtocol": PROMPT_PROTOCOL,
                "targetLanguage": "Chinese",
                "maximumCases": maximum,
                "activeContextWindow": NUM_CONTEXT,
                "maximumGeneratedTokens": NUM_PREDICT,
            },
            "inputs": input_bindings,
            "cases": [],
        }
        _atomic_write(output_path, report)

    completed = {
        row.get("auditCaseId")
        for row in report.get("cases", [])
        if isinstance(row, Mapping) and row.get("status") == "completed"
    }
    for case in cases:
        audit_id = case.get("auditCaseId")
        if not isinstance(audit_id, str) or audit_id in completed:
            continue
        if any(audit_id not in rows for rows in reports.values()):
            raise LocalLLMArbitrationError(f"evidence is missing {audit_id}")
        prompt = _prompt(
            audit_case_id=audit_id,
            duration_seconds=float(case.get("durationSeconds", 0)),
            qwen=reports["qwen"][audit_id],
            whisper=reports["whisper"][audit_id],
            vad=reports["vad"][audit_id],
            diarization=reports["diarization"][audit_id],
        )
        prior_failures = [
            {
                "failure": row.get("failure"),
                "wallSeconds": row.get("wallSeconds"),
                "promptSha256": row.get("promptSha256"),
            }
            for row in report.get("cases", [])
            if isinstance(row, Mapping)
            and row.get("auditCaseId") == audit_id
            and row.get("status") == "failed"
        ]
        report["cases"] = [
            row
            for row in report.get("cases", [])
            if not (
                isinstance(row, Mapping) and row.get("auditCaseId") == audit_id
            )
        ]
        row: dict[str, Any] = {
            "auditCaseId": audit_id,
            "status": "failed",
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "proposal": None,
            "rawProposal": None,
            "rawResponse": None,
            "constraintCorrections": [],
            "wallSeconds": None,
            "responseSha256": None,
            "inference": None,
            "failure": None,
            "priorFailures": prior_failures,
        }
        started = time.perf_counter()
        try:
            raw_proposal, elapsed, response_sha, inference = _request(
                model=model,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
            )
            whisper_segments = reports["whisper"][audit_id].get("segments")
            whisper_segment_count = (
                len(whisper_segments) if isinstance(whisper_segments, list) else 0
            )
            diarization_turns = reports["diarization"][audit_id].get("regularTurns")
            allowed_speakers_by_segment = _overlapping_speakers(
                whisper_segments, diarization_turns
            )
            proposal = copy.deepcopy(raw_proposal)
            corrections = _apply_deterministic_constraints(
                proposal,
                allowed_speakers_by_segment=allowed_speakers_by_segment,
            )
            row.update(
                {
                    "proposal": proposal,
                    "rawProposal": raw_proposal,
                    "constraintCorrections": corrections,
                    "wallSeconds": round(elapsed, 6),
                    "responseSha256": response_sha,
                    "inference": inference,
                }
            )
            _validate_proposal(
                proposal,
                whisper_segment_count=whisper_segment_count,
                allowed_speakers_by_segment=allowed_speakers_by_segment,
            )
            row.update(
                {
                    "status": "completed",
                }
            )
        except Exception as exc:  # keep later cases observable
            if isinstance(exc, LocalLLMArbitrationError):
                row["rawResponse"] = exc.raw_response
                row["responseSha256"] = (
                    exc.response_sha256 or row["responseSha256"]
                )
                row["inference"] = exc.inference or row["inference"]
            row["failure"] = f"{type(exc).__name__}: {exc}"
            row["wallSeconds"] = round(time.perf_counter() - started, 6)
        report["cases"].append(row)
        _atomic_write(output_path, report)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blind", type=Path)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--vad", type=Path, required=True)
    parser.add_argument("--diarization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:27b-q4_K_M")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--maximum", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(
            blind_path=args.blind,
            qwen_path=args.qwen,
            whisper_path=args.whisper,
            vad_path=args.vad,
            diarization_path=args.diarization,
            output_path=args.output,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            maximum=args.maximum,
        )
    finally:
        if not _unload_model(args.model):
            print(f"warning: could not unload Ollama model {args.model}", file=sys.stderr)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "observedCases": len(result["cases"]),
                "truthAccessed": result["truthAccessed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
