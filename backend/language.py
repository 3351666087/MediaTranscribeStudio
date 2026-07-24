"""Central language contracts for requests, ASR, and persisted artifacts.

The worker accepts ``auto`` only at request boundaries. Persisted transcript,
business, and report artifacts always carry a concrete practical BCP-47 tag,
``und`` when language evidence is unavailable, or ``mul`` for genuinely
multilingual material.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


AUTO_LANGUAGE = "auto"
UNDETERMINED_LANGUAGE = "und"
MULTIPLE_LANGUAGES = "mul"

_PRIMARY = re.compile(r"^[A-Za-z]{2,8}$")
_EXTLANG = re.compile(r"^[A-Za-z]{3}$")
_SCRIPT = re.compile(r"^[A-Za-z]{4}$")
_REGION = re.compile(r"^(?:[A-Za-z]{2}|[0-9]{3})$")
_VARIANT = re.compile(r"^(?:[A-Za-z0-9]{5,8}|[0-9][A-Za-z0-9]{3})$")
_EXTENSION_SINGLETON = re.compile(r"^[0-9A-WY-Za-wy-z]$")
_EXTENSION_SUBTAG = re.compile(r"^[A-Za-z0-9]{2,8}$")
_PRIVATE_SUBTAG = re.compile(r"^[A-Za-z0-9]{1,8}$")
_QWEN_SPLIT = re.compile(r"[,;/|+]+")
_UNKNOWN_QWEN_LANGUAGE_VALUES = frozenset(
    {"", "auto", "none", "null", "unknown", "undetermined", "n/a"}
)

_QWEN_LANGUAGE_BY_PRIMARY = {
    "zh": "Chinese",
    "cmn": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}
_BCP47_BY_QWEN_LANGUAGE = {
    "chinese": "zh",
    "cantonese": "yue",
    "english": "en",
    "arabic": "ar",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "portuguese": "pt",
    "indonesian": "id",
    "italian": "it",
    "korean": "ko",
    "russian": "ru",
    "thai": "th",
    "vietnamese": "vi",
    "japanese": "ja",
    "turkish": "tr",
    "hindi": "hi",
    "malay": "ms",
    "dutch": "nl",
    "swedish": "sv",
    "danish": "da",
    "finnish": "fi",
    "polish": "pl",
    "czech": "cs",
    "filipino": "fil",
    "persian": "fa",
    "greek": "el",
    "romanian": "ro",
    "hungarian": "hu",
    "macedonian": "mk",
    "swahili": "sw",
    "oromo": "om",
}


def qwen_supported_primary_language_tags() -> tuple[str, ...]:
    """Return the explicit primary-language tags accepted by Qwen3-ASR."""

    return tuple(sorted(_QWEN_LANGUAGE_BY_PRIMARY))


def normalize_language_tag(value: Any, *, allow_auto: bool = False) -> str:
    """Validate and canonicalize a practical BCP-47 language tag.

    Supported forms include language/extlang/script/region/variant/extension
    sequences, private-use suffixes, and private-use-only tags. Grandfathered
    tags are intentionally excluded because the product contract requires a
    structurally parseable 2-8 letter primary language.
    """

    if not isinstance(value, str):
        raise ValueError("language tag must be a string")
    if not value or value != value.strip():
        raise ValueError("language tag must not be empty or padded")
    if not value.isascii():
        raise ValueError("language tag must contain only ASCII characters")

    text = value.replace("_", "-")
    if len(text) > 255:
        raise ValueError("language tag must not exceed 255 characters")
    if text.startswith("-") or text.endswith("-") or "--" in text:
        raise ValueError("language tag contains an empty subtag")

    if text.casefold() == AUTO_LANGUAGE:
        if allow_auto:
            return AUTO_LANGUAGE
        raise ValueError("auto is allowed only for request input")

    subtags = text.split("-")
    if subtags[0].casefold() == "x":
        if len(subtags) == 1 or any(
            not _PRIVATE_SUBTAG.fullmatch(subtag) for subtag in subtags[1:]
        ):
            raise ValueError("private-use language tag is malformed")
        return "-".join(["x", *(subtag.lower() for subtag in subtags[1:])])

    primary = subtags[0]
    if not _PRIMARY.fullmatch(primary):
        raise ValueError("language tag has an invalid primary language subtag")

    output = [primary.lower()]
    index = 1

    if 2 <= len(primary) <= 3:
        extlang_count = 0
        while (
            index < len(subtags)
            and extlang_count < 3
            and _EXTLANG.fullmatch(subtags[index])
        ):
            output.append(subtags[index].lower())
            index += 1
            extlang_count += 1

    if index < len(subtags) and _SCRIPT.fullmatch(subtags[index]):
        output.append(subtags[index].title())
        index += 1

    if index < len(subtags) and _REGION.fullmatch(subtags[index]):
        region = subtags[index]
        output.append(region.upper() if region.isalpha() else region)
        index += 1

    seen_variants: set[str] = set()
    while index < len(subtags) and _VARIANT.fullmatch(subtags[index]):
        variant = subtags[index].lower()
        if variant in seen_variants:
            raise ValueError("language tag contains a duplicate variant")
        seen_variants.add(variant)
        output.append(variant)
        index += 1

    seen_extensions: set[str] = set()
    while index < len(subtags) and _EXTENSION_SINGLETON.fullmatch(
        subtags[index]
    ):
        singleton = subtags[index].lower()
        if singleton in seen_extensions:
            raise ValueError("language tag contains a duplicate extension singleton")
        seen_extensions.add(singleton)
        output.append(singleton)
        index += 1

        extension_start = index
        while index < len(subtags) and _EXTENSION_SUBTAG.fullmatch(
            subtags[index]
        ):
            output.append(subtags[index].lower())
            index += 1
        if index == extension_start:
            raise ValueError("language extension requires at least one subtag")

    if index < len(subtags) and subtags[index].casefold() == "x":
        output.append("x")
        index += 1
        private_start = index
        while index < len(subtags) and _PRIVATE_SUBTAG.fullmatch(subtags[index]):
            output.append(subtags[index].lower())
            index += 1
        if index == private_start:
            raise ValueError("private-use marker requires at least one subtag")

    if index != len(subtags):
        raise ValueError("language tag contains a malformed subtag sequence")
    return "-".join(output)


def qwen_language_for_request(requested_language: Any) -> str | None:
    """Map a request language to the canonical Qwen3-ASR language name.

    ``auto`` becomes ``None`` so the installed Qwen3-ASR runtime performs
    language identification. The runtime rejects the literal value ``Auto``.
    An explicit but unsupported language fails closed rather than silently
    transcribing with the wrong language prompt.
    """

    normalized = normalize_language_tag(requested_language, allow_auto=True)
    if normalized == AUTO_LANGUAGE:
        return None
    if normalized in {UNDETERMINED_LANGUAGE, MULTIPLE_LANGUAGES} or normalized.startswith(
        "x-"
    ):
        raise ValueError(
            f"Qwen3-ASR does not accept explicit language tag {normalized!r}"
        )
    primary = normalized.split("-", 1)[0]
    qwen_language = _QWEN_LANGUAGE_BY_PRIMARY.get(primary)
    if qwen_language is None:
        raise ValueError(
            f"Qwen3-ASR does not support explicit language tag {normalized!r}"
        )
    return qwen_language


def normalize_qwen_language_candidates(value: Any) -> tuple[str, ...]:
    """Convert Qwen language output into unique canonical BCP-47 candidates."""

    if value is None:
        return ()
    raw_items: list[Any]
    if isinstance(value, str):
        raw_items = [
            item.strip()
            for item in _QWEN_SPLIT.split(value)
            if item.strip()
        ]
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        raw_items = list(value)
    else:
        raw_items = [value]

    candidates: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip()
        if text.casefold() in _UNKNOWN_QWEN_LANGUAGE_VALUES:
            continue
        normalized = _BCP47_BY_QWEN_LANGUAGE.get(text.casefold())
        if normalized is None:
            try:
                normalized = normalize_language_tag(text, allow_auto=False)
            except ValueError:
                continue
        if normalized in {UNDETERMINED_LANGUAGE, MULTIPLE_LANGUAGES}:
            continue
        if normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)
    return tuple(candidates)


def _finite_positive_weight(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.0
    return weight if math.isfinite(weight) and weight > 0.0 else 0.0


def _detection_candidates_and_weight(detection: Any) -> tuple[tuple[str, ...], float]:
    value = detection
    weight = 1.0
    if isinstance(detection, Mapping):
        value = detection.get(
            "languageCandidates",
            detection.get(
                "language_candidates",
                detection.get(
                    "languages",
                    detection.get(
                        "language",
                        detection.get("rawLanguage", detection.get("raw_language")),
                    ),
                ),
            ),
        )
        duration = detection.get(
            "speechDurationMs",
            detection.get(
                "speech_duration_ms",
                detection.get("durationMs", detection.get("duration_ms")),
            ),
        )
        if duration is None:
            start = detection.get("startMs", detection.get("start_ms"))
            end = detection.get("endMs", detection.get("end_ms"))
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                duration = float(end) - float(start)
        parsed_weight = _finite_positive_weight(duration)
        if parsed_weight:
            weight = parsed_weight
    elif (
        isinstance(detection, Sequence)
        and not isinstance(detection, (str, bytes, bytearray))
        and len(detection) == 2
    ):
        value = detection[0]
        parsed_weight = _finite_positive_weight(detection[1])
        if parsed_weight:
            weight = parsed_weight
    return normalize_qwen_language_candidates(value), weight


def reconcile_detected_languages(
    detections: Iterable[Any],
    *,
    requested_language: Any = AUTO_LANGUAGE,
    dominant_ratio: float = 0.8,
    dominant_margin: float = 0.2,
) -> str:
    """Resolve duration-weighted language detections to BCP-47/``und``/``mul``."""

    request = normalize_language_tag(requested_language, allow_auto=True)
    if request != AUTO_LANGUAGE:
        return request
    if not 0.5 <= dominant_ratio <= 1.0:
        raise ValueError("dominant_ratio must be between 0.5 and 1.0")
    if not 0.0 <= dominant_margin <= 1.0:
        raise ValueError("dominant_margin must be between 0.0 and 1.0")

    weights: dict[str, float] = {}
    for detection in detections:
        candidates, weight = _detection_candidates_and_weight(detection)
        if not candidates:
            continue
        split_weight = weight / len(candidates)
        for language in candidates:
            weights[language] = weights.get(language, 0.0) + split_weight

    if not weights:
        return UNDETERMINED_LANGUAGE
    if len(weights) == 1:
        return next(iter(weights))

    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    total = sum(weight for _, weight in ranked)
    first_language, first_weight = ranked[0]
    second_weight = ranked[1][1]
    first_ratio = first_weight / total
    margin = (first_weight - second_weight) / total
    if first_ratio >= dominant_ratio and margin >= dominant_margin:
        return first_language
    return MULTIPLE_LANGUAGES
