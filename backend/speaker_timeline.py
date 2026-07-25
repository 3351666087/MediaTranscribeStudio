"""Validated, first-class speaker timeline contracts.

The regular timeline preserves overlapping speakers for diarization scoring.
The exclusive timeline is the model-native, single-speaker view intended for
coarse ASR-segment ownership.  Neither timeline changes or duplicates ASR text.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import WorkerError


SPEAKER_TIMELINE_SCHEMA_VERSION = "1.0.0"
SPEAKER_TIMELINE_AUTHORITY = (
    "pyannote-community-1-native-regular-exclusive-v1"
)
_CANONICAL_SPEAKER = re.compile(r"^speaker-([1-9][0-9]*)$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _fail(message: str) -> None:
    raise WorkerError("SPEAKER_TIMELINE_INVALID", message)


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_canonical_ids(values: Sequence[str]) -> tuple[str, ...]:
    canonical = tuple(values)
    if (
        not canonical
        or any(
            not isinstance(item, str)
            or _CANONICAL_SPEAKER.fullmatch(item) is None
            for item in canonical
        )
        or len(set(canonical)) != len(canonical)
    ):
        _fail("canonical speaker IDs must be a non-empty unique speaker-N set")
    expected = tuple(
        sorted(canonical, key=lambda item: int(item.removeprefix("speaker-")))
    )
    if canonical != expected:
        _fail("canonical speaker IDs must use deterministic numeric order")
    return canonical


def _validate_turns(
    raw_turns: Any,
    *,
    path: str,
    duration_ms: int,
    canonical_ids: set[str],
    local_to_canonical: Mapping[str, str],
    exclusive: bool,
) -> list[dict[str, Any]]:
    if (
        not isinstance(raw_turns, Sequence)
        or isinstance(raw_turns, (str, bytes, bytearray))
        or not raw_turns
    ):
        _fail(f"{path}.turns must be a non-empty array")
    turns: list[dict[str, Any]] = []
    previous_key: tuple[int, int, str, str] | None = None
    previous_end = -1
    for index, raw in enumerate(raw_turns):
        if not isinstance(raw, Mapping):
            _fail(f"{path}.turns[{index}] must be an object")
        start_ms = raw.get("startMs")
        end_ms = raw.get("endMs")
        speaker_id = raw.get("speakerId")
        local_speaker = raw.get("localSpeaker")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < 0
            or end_ms <= start_ms
            or end_ms > duration_ms
        ):
            _fail(f"{path}.turns[{index}] has invalid or out-of-range boundaries")
        if not isinstance(speaker_id, str) or speaker_id not in canonical_ids:
            _fail(f"{path}.turns[{index}].speakerId is outside the canonical set")
        if not isinstance(local_speaker, str) or not local_speaker.strip():
            _fail(f"{path}.turns[{index}].localSpeaker must be non-empty text")
        local_speaker = local_speaker.strip()
        if local_to_canonical.get(local_speaker) != speaker_id:
            _fail(f"{path}.turns[{index}] conflicts with the accepted mapping")
        key = (start_ms, end_ms, speaker_id, local_speaker)
        if previous_key is not None and key < previous_key:
            _fail(f"{path}.turns must use deterministic timeline order")
        if exclusive and start_ms < previous_end:
            _fail(f"{path}.turns must not overlap")
        previous_key = key
        previous_end = max(previous_end, end_ms)
        turns.append(
            {
                "startMs": start_ms,
                "endMs": end_ms,
                "speakerId": speaker_id,
                "localSpeaker": local_speaker,
            }
        )
    return turns


def validate_speaker_timeline(
    value: Any,
    *,
    duration_ms: int,
    canonical_speaker_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate and copy an authoritative regular/exclusive timeline."""

    if not isinstance(value, Mapping):
        _fail("speakerTimeline must be an object")
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 1
    ):
        _fail("speakerTimeline duration must be a positive integer")
    canonical = _normalized_canonical_ids(canonical_speaker_ids)
    canonical_set = set(canonical)
    if value.get("schemaVersion") != SPEAKER_TIMELINE_SCHEMA_VERSION:
        _fail("speakerTimeline schemaVersion is unsupported")
    if value.get("authority") != SPEAKER_TIMELINE_AUTHORITY:
        _fail("speakerTimeline authority is unsupported")

    provider = value.get("provider")
    if (
        not isinstance(provider, Mapping)
        or provider.get("id") != "pyannote-community-1"
        or not isinstance(provider.get("version"), str)
        or not str(provider["version"]).strip()
    ):
        _fail("speakerTimeline provider identity is invalid")

    mapping = value.get("mapping")
    if not isinstance(mapping, Mapping) or mapping.get("accepted") is not True:
        _fail("speakerTimeline requires an accepted canonical mapping")
    if (
        mapping.get("method")
        != "global-duration-weighted-acoustic-hungarian-v1"
    ):
        _fail("speakerTimeline mapping method is unsupported")
    margin = mapping.get("margin")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        _fail("speakerTimeline mapping margin is invalid")
    raw_local_to_canonical = mapping.get("localToCanonical")
    if not isinstance(raw_local_to_canonical, Mapping):
        _fail("speakerTimeline mapping.localToCanonical must be an object")
    local_to_canonical = {
        str(local): str(speaker)
        for local, speaker in raw_local_to_canonical.items()
        if isinstance(local, str)
        and local.strip()
        and isinstance(speaker, str)
    }
    if (
        len(local_to_canonical) != len(raw_local_to_canonical)
        or set(local_to_canonical.values()) != canonical_set
        or len(local_to_canonical) != len(canonical)
    ):
        _fail("speakerTimeline mapping must be a complete canonical bijection")

    normalized: dict[str, Any] = {
        "schemaVersion": SPEAKER_TIMELINE_SCHEMA_VERSION,
        "authority": SPEAKER_TIMELINE_AUTHORITY,
        "provider": {
            "id": "pyannote-community-1",
            "version": str(provider["version"]).strip(),
        },
        "mapping": {
            "method": mapping["method"],
            "accepted": True,
            "margin": float(margin),
            "localToCanonical": dict(sorted(local_to_canonical.items())),
        },
    }
    observed_regular: set[str] = set()
    observed_regular_local: set[str] = set()
    for name, semantics, exclusive in (
        ("regular", "overlap-preserving", False),
        ("exclusive", "single-speaker", True),
    ):
        timeline = value.get(name)
        if (
            not isinstance(timeline, Mapping)
            or timeline.get("semantics") != semantics
            or timeline.get("native") is not True
        ):
            _fail(f"speakerTimeline.{name} must be a native {semantics} timeline")
        turns = _validate_turns(
            timeline.get("turns"),
            path=f"speakerTimeline.{name}",
            duration_ms=duration_ms,
            canonical_ids=canonical_set,
            local_to_canonical=local_to_canonical,
            exclusive=exclusive,
        )
        digest = timeline.get("sha256")
        if (
            not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or digest != _stable_digest(turns)
        ):
            _fail(f"speakerTimeline.{name}.sha256 does not bind its turns")
        normalized[name] = {
            "semantics": semantics,
            "native": True,
            "turns": turns,
            "sha256": digest,
        }
        if name == "regular":
            observed_regular.update(turn["speakerId"] for turn in turns)
            observed_regular_local.update(turn["localSpeaker"] for turn in turns)
    if observed_regular != canonical_set:
        _fail("speakerTimeline regular turns must observe every canonical speaker")
    if observed_regular_local != set(local_to_canonical):
        _fail("speakerTimeline regular turns must observe every mapped local speaker")

    text_alignment = value.get("textAlignment")
    if (
        not isinstance(text_alignment, Mapping)
        or text_alignment.get("status") != "segment-level-exclusive-dominance"
        or text_alignment.get("wordTimestampsAvailable") is not False
        or text_alignment.get("sourceTextMutable") is not False
    ):
        _fail("speakerTimeline textAlignment policy is invalid")
    normalized["textAlignment"] = {
        "status": "segment-level-exclusive-dominance",
        "wordTimestampsAvailable": False,
        "sourceTextMutable": False,
    }
    return normalized


def _merge_regular_turns(
    turns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for turn in turns:
        key = (str(turn["speakerId"]), str(turn["localSpeaker"]))
        grouped.setdefault(key, []).append(
            (int(turn["startMs"]), int(turn["endMs"]))
        )
    merged: list[dict[str, Any]] = []
    for (speaker_id, local_speaker), intervals in grouped.items():
        combined: list[list[int]] = []
        for start_ms, end_ms in sorted(intervals):
            if combined and start_ms <= combined[-1][1]:
                combined[-1][1] = max(combined[-1][1], end_ms)
            else:
                combined.append([start_ms, end_ms])
        merged.extend(
            {
                "startMs": start_ms,
                "endMs": end_ms,
                "speakerId": speaker_id,
                "localSpeaker": local_speaker,
            }
            for start_ms, end_ms in combined
        )
    return sorted(
        merged,
        key=lambda item: (
            item["startMs"],
            item["endMs"],
            item["speakerId"],
            item["localSpeaker"],
        ),
    )


def _merge_exclusive_turns(
    turns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for raw in sorted(
        turns,
        key=lambda item: (
            int(item["startMs"]),
            int(item["endMs"]),
            str(item["speakerId"]),
            str(item["localSpeaker"]),
        ),
    ):
        turn = {
            "startMs": int(raw["startMs"]),
            "endMs": int(raw["endMs"]),
            "speakerId": str(raw["speakerId"]),
            "localSpeaker": str(raw["localSpeaker"]),
        }
        if (
            merged
            and turn["startMs"] == merged[-1]["endMs"]
            and turn["speakerId"] == merged[-1]["speakerId"]
            and turn["localSpeaker"] == merged[-1]["localSpeaker"]
        ):
            merged[-1]["endMs"] = turn["endMs"]
        else:
            merged.append(turn)
    return merged


def build_speaker_timeline(
    *,
    provider_version: str,
    local_to_canonical: Mapping[str, str],
    mapping_margin: float,
    regular_turns: Sequence[Mapping[str, Any]],
    exclusive_turns: Sequence[Mapping[str, Any]],
    duration_ms: int,
    canonical_speaker_ids: Sequence[str],
) -> dict[str, Any]:
    """Build and validate a native Community-1 authoritative timeline."""

    regular = _merge_regular_turns(regular_turns)
    exclusive = _merge_exclusive_turns(exclusive_turns)
    value = {
        "schemaVersion": SPEAKER_TIMELINE_SCHEMA_VERSION,
        "authority": SPEAKER_TIMELINE_AUTHORITY,
        "provider": {
            "id": "pyannote-community-1",
            "version": str(provider_version),
        },
        "mapping": {
            "method": "global-duration-weighted-acoustic-hungarian-v1",
            "accepted": True,
            "margin": float(mapping_margin),
            "localToCanonical": dict(sorted(local_to_canonical.items())),
        },
        "regular": {
            "semantics": "overlap-preserving",
            "native": True,
            "turns": regular,
            "sha256": _stable_digest(regular),
        },
        "exclusive": {
            "semantics": "single-speaker",
            "native": True,
            "turns": exclusive,
            "sha256": _stable_digest(exclusive),
        },
        "textAlignment": {
            "status": "segment-level-exclusive-dominance",
            "wordTimestampsAvailable": False,
            "sourceTextMutable": False,
        },
    }
    return validate_speaker_timeline(
        value,
        duration_ms=duration_ms,
        canonical_speaker_ids=canonical_speaker_ids,
    )


def speaker_timeline_turns(
    value: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[Mapping[str, Any], ...]:
    if mode not in {"regular", "exclusive"}:
        raise ValueError("speaker timeline mode must be regular or exclusive")
    timeline = value.get(mode)
    if not isinstance(timeline, Mapping):
        return ()
    turns = timeline.get("turns")
    if not isinstance(turns, list):
        return ()
    return tuple(turn for turn in turns if isinstance(turn, Mapping))
