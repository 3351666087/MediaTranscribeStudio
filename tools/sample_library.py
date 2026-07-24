"""Validation and small quality helpers for the short sample library."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
MAX_ID_LENGTH = 80
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_SCENARIOS = {
    "clean-single",
    "telephony-noise",
    "two-speaker-turns",
    "overlap",
    "video-scene-cuts",
    "reverb",
    "three-speaker-meeting",
    "downloaded-reference",
}


class SampleLibraryError(ValueError):
    """Raised when the sample specification is malformed."""


@dataclass(frozen=True)
class SampleCase:
    case_id: str
    language: str
    scenario: str
    kind: str
    output: str
    utterances: tuple[dict[str, Any], ...]
    effects: tuple[str, ...]
    remote: dict[str, Any] | None
    expected_transcript: str | None

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(item["speaker"]) for item in self.utterances))


@dataclass(frozen=True)
class SampleManifest:
    schema_version: str
    library_id: str
    max_duration_seconds: float
    generated_root: str
    cases: tuple[SampleCase, ...]


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SampleLibraryError(f"{field} must be non-empty text")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise SampleLibraryError(f"{field} contains control characters")
    return value.strip()


def _relative_output(value: Any, field: str) -> str:
    text = _require_text(value, field).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or text.startswith("/"):
        raise SampleLibraryError(f"{field} must be a relative path")
    if path.suffix.casefold() not in {".wav", ".mp4"}:
        raise SampleLibraryError(f"{field} must end in .wav or .mp4")
    return text


def _validate_utterance(value: Any, field: str, *, offset_allowed: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SampleLibraryError(f"{field} must be an object")
    speaker = _require_text(value.get("speaker"), f"{field}.speaker")
    voice = value.get("voice")
    if voice is not None:
        voice = _require_text(voice, f"{field}.voice")
    text = _require_text(value.get("text"), f"{field}.text")
    offset_ms = value.get("offsetMs", 0)
    if isinstance(offset_ms, bool) or not isinstance(offset_ms, int) or offset_ms < 0:
        raise SampleLibraryError(f"{field}.offsetMs must be a non-negative integer")
    if not offset_allowed and offset_ms:
        raise SampleLibraryError(f"{field}.offsetMs is only allowed for overlap samples")
    return {
        "speaker": speaker,
        **({"voice": voice} if voice is not None else {}),
        "text": text,
        "offsetMs": offset_ms,
    }


def load_manifest(path: str | Path) -> SampleManifest:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SampleLibraryError(f"unable to read sample manifest: {source}") from exc
    if not isinstance(raw, dict):
        raise SampleLibraryError("sample manifest must be an object")
    if raw.get("schemaVersion") != SCHEMA_VERSION:
        raise SampleLibraryError("unsupported sample manifest schemaVersion")
    library_id = _require_text(raw.get("libraryId"), "libraryId")
    if not _ID_PATTERN.fullmatch(library_id):
        raise SampleLibraryError("libraryId contains unsupported characters")
    max_duration = raw.get("maxDurationSeconds")
    if isinstance(max_duration, bool) or not isinstance(max_duration, (int, float)):
        raise SampleLibraryError("maxDurationSeconds must be a number")
    if max_duration <= 0 or max_duration > 30:
        raise SampleLibraryError("maxDurationSeconds must be in (0, 30]")
    generated_root = _require_text(raw.get("generatedRoot"), "generatedRoot")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SampleLibraryError("cases must be a non-empty array")
    cases: list[SampleCase] = []
    seen: set[str] = set()
    seen_outputs: set[str] = set()
    for index, item in enumerate(raw_cases):
        field = f"cases[{index}]"
        if not isinstance(item, dict):
            raise SampleLibraryError(f"{field} must be an object")
        case_id = _require_text(item.get("id"), f"{field}.id")
        if not _ID_PATTERN.fullmatch(case_id) or len(case_id) > MAX_ID_LENGTH:
            raise SampleLibraryError(f"{field}.id contains unsupported characters")
        if case_id in seen:
            raise SampleLibraryError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        language = _require_text(item.get("language"), f"{field}.language")
        if not _LANGUAGE_PATTERN.fullmatch(language):
            raise SampleLibraryError(f"{field}.language must be a BCP-47-like tag")
        scenario = _require_text(item.get("scenario"), f"{field}.scenario")
        if scenario not in _SCENARIOS:
            raise SampleLibraryError(f"{field}.scenario is unsupported")
        kind = _require_text(item.get("kind"), f"{field}.kind")
        if kind not in {"audio", "video"}:
            raise SampleLibraryError(f"{field}.kind must be audio or video")
        output = _relative_output(item.get("output"), f"{field}.output")
        if output in seen_outputs:
            raise SampleLibraryError(f"duplicate output path: {output}")
        seen_outputs.add(output)
        if kind == "audio" and not output.endswith(".wav"):
            raise SampleLibraryError(f"{field}.audio output must be .wav")
        if kind == "video" and not output.endswith(".mp4"):
            raise SampleLibraryError(f"{field}.video output must be .mp4")
        raw_utterances = item.get("utterances")
        if not isinstance(raw_utterances, list) or not raw_utterances:
            raise SampleLibraryError(f"{field}.utterances must be non-empty")
        utterances = tuple(
            _validate_utterance(
                utterance,
                f"{field}.utterances[{utterance_index}]",
                offset_allowed=scenario == "overlap",
            )
            for utterance_index, utterance in enumerate(raw_utterances)
        )
        if scenario == "overlap" and len(utterances) < 2:
            raise SampleLibraryError(f"{field}.overlap needs at least two utterances")
        if scenario == "three-speaker-meeting" and len(
            {item["speaker"] for item in utterances}
        ) < 3:
            raise SampleLibraryError(f"{field}.meeting needs at least three speakers")
        raw_effects = item.get("effects", [])
        if not isinstance(raw_effects, list) or any(
            not isinstance(effect, str) or not effect.strip() for effect in raw_effects
        ):
            raise SampleLibraryError(f"{field}.effects must be text values")
        remote = item.get("remote")
        if remote is not None:
            if not isinstance(remote, dict):
                raise SampleLibraryError(f"{field}.remote must be an object")
            _require_text(remote.get("url"), f"{field}.remote.url")
            digest = _require_text(remote.get("sha256"), f"{field}.remote.sha256")
            if not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise SampleLibraryError(f"{field}.remote.sha256 must be SHA-256")
        expected_transcript = item.get("expectedTranscript")
        if expected_transcript is not None:
            expected_transcript = _require_text(
                expected_transcript,
                f"{field}.expectedTranscript",
            )
        cases.append(
            SampleCase(
                case_id=case_id,
                language=language,
                scenario=scenario,
                kind=kind,
                output=output,
                utterances=utterances,
                effects=tuple(str(effect).strip() for effect in raw_effects),
                remote=dict(remote) if remote is not None else None,
                expected_transcript=expected_transcript,
            )
        )
    return SampleManifest(
        schema_version=SCHEMA_VERSION,
        library_id=library_id,
        max_duration_seconds=float(max_duration),
        generated_root=generated_root,
        cases=tuple(cases),
    )


def normalize_text(text: str) -> str:
    """Normalize text for a conservative, language-agnostic comparison."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        char
        if char.isalnum() or char.isspace()
        else " "
        for char in normalized
    )
    return " ".join(normalized.split())


def tokenize_for_score(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if any("\u3400" <= char <= "\u9fff" for char in normalized):
        return [char for char in normalized if not char.isspace()]
    return normalized.split()


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    reference_tokens = tokenize_for_score(reference)
    if not reference_tokens:
        return None
    return edit_distance(reference_tokens, tokenize_for_score(hypothesis)) / len(
        reference_tokens
    )
