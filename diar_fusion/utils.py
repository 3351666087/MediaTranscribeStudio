from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def count_diar_speakers(diar_segments: List[Dict[str, Any]]) -> int:
    return len(
        {
            str(item.get("speaker", "") or "").strip()
            for item in diar_segments or []
            if str(item.get("speaker", "") or "").strip()
        }
    )


def normalize_speaker_id(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return "0"
    digits = ""
    for char in reversed(value):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    if digits:
        try:
            return str(int(digits))
        except Exception:
            return digits
    return value


def merge_adjacent_speaker_turns(
    diar_segments: List[Dict[str, Any]],
    merge_gap_sec: float,
) -> List[Dict[str, Any]]:
    if not diar_segments:
        return []

    merged: List[Dict[str, Any]] = []
    for item in sorted(
        diar_segments,
        key=lambda value: (
            float(value.get("start", 0.0)),
            float(value.get("end", 0.0)),
            str(value.get("speaker", "")),
        ),
    ):
        start = float(item.get("start", 0.0))
        end = max(start, float(item.get("end", start)))
        speaker = str(item.get("speaker", "0"))
        if (
            merged
            and str(merged[-1].get("speaker", "")) == speaker
            and start <= float(merged[-1].get("end", 0.0)) + max(0.0, merge_gap_sec)
        ):
            merged[-1]["end"] = max(float(merged[-1]["end"]), end)
            continue
        merged.append({"start": start, "end": end, "speaker": speaker})
    return merged


def merge_time_regions(
    regions: List[Dict[str, Any]],
    max_gap_sec: float = 0.0,
    min_duration_sec: float = 0.0,
    max_end_sec: Optional[float] = None,
) -> List[Dict[str, float]]:
    valid: List[Tuple[float, float]] = []
    for region in regions or []:
        try:
            start = float(region.get("start", 0.0))
            end = float(region.get("end", start))
        except Exception:
            continue
        if max_end_sec is not None:
            start = max(0.0, min(start, max_end_sec))
            end = max(0.0, min(end, max_end_sec))
        if end <= start:
            continue
        valid.append((start, end))

    if not valid:
        return []

    valid.sort(key=lambda item: (item[0], item[1]))
    merge_gap = max(0.0, float(max_gap_sec or 0.0))
    min_duration = max(0.0, float(min_duration_sec or 0.0))

    merged: List[Dict[str, float]] = []
    cur_start, cur_end = valid[0]
    for start, end in valid[1:]:
        if start <= cur_end + merge_gap:
            cur_end = max(cur_end, end)
        else:
            if cur_end - cur_start >= min_duration:
                merged.append({"start": cur_start, "end": cur_end})
            cur_start, cur_end = start, end
    if cur_end - cur_start >= min_duration:
        merged.append({"start": cur_start, "end": cur_end})
    return merged


def segment_overlap_seconds(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))


def derive_overlap_regions_from_diar_segments(
    diar_segments: List[Dict[str, Any]],
    min_duration_sec: float = 0.2,
    max_gap_sec: float = 0.08,
) -> List[Dict[str, float]]:
    if len(diar_segments or []) < 2:
        return []

    ordered = sorted(
        (
            {
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", item.get("start", 0.0))),
                "speaker": str(item.get("speaker", "")),
            }
            for item in diar_segments
        ),
        key=lambda x: (x["start"], x["end"]),
    )

    overlaps: List[Dict[str, float]] = []
    for idx, left in enumerate(ordered):
        left_start = float(left["start"])
        left_end = float(left["end"])
        left_speaker = str(left["speaker"])
        if left_end <= left_start:
            continue
        for right in ordered[idx + 1 :]:
            right_start = float(right["start"])
            if right_start >= left_end:
                break
            right_end = float(right["end"])
            if right_end <= right_start:
                continue
            if left_speaker and right.get("speaker") and left_speaker == str(right["speaker"]):
                continue
            overlap_start = max(left_start, right_start)
            overlap_end = min(left_end, right_end)
            if overlap_end - overlap_start <= 0:
                continue
            overlaps.append({"start": overlap_start, "end": overlap_end})

    return merge_time_regions(
        overlaps,
        max_gap_sec=max_gap_sec,
        min_duration_sec=min_duration_sec,
    )


def normalize_diar_segments_for_fusion(
    diar_segments: List[Dict[str, Any]],
    *,
    min_turn_sec: float = 0.05,
    merge_gap_sec: float = 0.08,
    normalize_ids: bool = True,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    min_turn = max(0.01, float(min_turn_sec))
    for item in diar_segments or []:
        try:
            start = max(0.0, float(item.get("start", 0.0) or 0.0))
            end = max(start, float(item.get("end", start) or start))
        except Exception:
            continue
        if end - start < min_turn:
            continue
        speaker = str(item.get("speaker", "0") or "").strip() or "0"
        if normalize_ids:
            speaker = normalize_speaker_id(speaker)
        normalized.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
            }
        )
    return merge_adjacent_speaker_turns(
        normalized,
        merge_gap_sec=max(0.0, float(merge_gap_sec)),
    )


def diar_speaker_duration_map(
    diar_segments: List[Dict[str, Any]],
) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    for item in diar_segments or []:
        speaker = str(item.get("speaker", "") or "").strip()
        if not speaker:
            continue
        start = float(item.get("start", 0.0) or 0.0)
        end = max(start, float(item.get("end", start) or start))
        durations[speaker] = durations.get(speaker, 0.0) + max(0.0, end - start)
    return durations


def diar_speaker_overlap_matrix(
    source_segments: List[Dict[str, Any]],
    reference_segments: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    matrix: Dict[str, Dict[str, float]] = {}
    for source in source_segments or []:
        source_speaker = str(source.get("speaker", "") or "").strip()
        if not source_speaker:
            continue
        source_start = float(source.get("start", 0.0) or 0.0)
        source_end = max(source_start, float(source.get("end", source_start) or source_start))
        for reference in reference_segments or []:
            reference_speaker = str(reference.get("speaker", "") or "").strip()
            if not reference_speaker:
                continue
            overlap = segment_overlap_seconds(
                source_start,
                source_end,
                float(reference.get("start", 0.0) or 0.0),
                float(reference.get("end", reference.get("start", 0.0) or 0.0) or 0.0),
            )
            if overlap <= 0.0:
                continue
            bucket = matrix.setdefault(source_speaker, {})
            bucket[reference_speaker] = bucket.get(reference_speaker, 0.0) + overlap
    return matrix


def build_strong_reference_speaker_map(
    source_segments: List[Dict[str, Any]],
    reference_segments: List[Dict[str, Any]],
    *,
    min_overlap_ratio: float,
    min_overlap_sec: float,
    min_margin_ratio: float,
) -> Dict[str, str]:
    if not source_segments or not reference_segments:
        return {}

    source_durations = diar_speaker_duration_map(source_segments)
    reference_durations = diar_speaker_duration_map(reference_segments)
    overlap_matrix = diar_speaker_overlap_matrix(
        source_segments,
        reference_segments,
    )
    reverse_matrix = diar_speaker_overlap_matrix(
        reference_segments,
        source_segments,
    )
    mapping: Dict[str, str] = {}
    for source_speaker, overlaps in overlap_matrix.items():
        ranked = sorted(
            overlaps.items(),
            key=lambda item: (float(item[1]), str(item[0])),
            reverse=True,
        )
        if not ranked:
            continue
        best_reference, best_overlap = ranked[0]
        second_overlap = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        source_duration = max(1e-6, float(source_durations.get(source_speaker, 0.0) or 0.0))
        overlap_ratio = float(best_overlap) / source_duration
        margin_ratio = float(best_overlap) / max(1e-6, second_overlap)
        if float(best_overlap) < max(0.25, float(min_overlap_sec)):
            continue
        if overlap_ratio < float(min_overlap_ratio):
            continue
        if margin_ratio < float(min_margin_ratio):
            continue

        reverse_ranked = sorted(
            (reverse_matrix.get(str(best_reference), {}) or {}).items(),
            key=lambda item: (float(item[1]), str(item[0])),
            reverse=True,
        )
        if not reverse_ranked:
            continue
        reverse_best_source, reverse_best_overlap = reverse_ranked[0]
        if str(reverse_best_source) != str(source_speaker):
            continue
        reverse_second_overlap = float(reverse_ranked[1][1]) if len(reverse_ranked) > 1 else 0.0
        reverse_margin_ratio = float(reverse_best_overlap) / max(1e-6, reverse_second_overlap)
        reference_duration = max(1e-6, float(reference_durations.get(str(best_reference), 0.0) or 0.0))
        reverse_overlap_ratio = float(reverse_best_overlap) / reference_duration
        if reverse_overlap_ratio < max(0.35, float(min_overlap_ratio) * 0.75):
            continue
        if reverse_margin_ratio < float(min_margin_ratio):
            continue

        mapping[source_speaker] = str(best_reference)
    return mapping


def build_overlap_seed_regions(
    *,
    msdd_segments: List[Dict[str, Any]],
    pyannote_segments: List[Dict[str, Any]],
    sortformer_segments: List[Dict[str, Any]],
    hybrid_cfg: Dict[str, Any],
    overlap_cfg: Dict[str, Any],
) -> List[Dict[str, float]]:
    overlap_union_cfg = hybrid_cfg.get("overlap_union", {}) or {}
    if not safe_bool(overlap_union_cfg.get("enabled", True), True):
        return []

    min_region_sec = max(
        0.05,
        safe_float(
            overlap_union_cfg.get("min_region_sec", overlap_cfg.get("min_region_sec", 0.25)),
            safe_float(overlap_cfg.get("min_region_sec", 0.25), 0.25),
        ),
    )
    merge_gap_sec = max(
        0.0,
        safe_float(
            overlap_union_cfg.get(
                "merge_gap_sec",
                ((overlap_cfg.get("osd", {}) or {}).get("merge_gap_sec", 0.08)),
            ),
            safe_float(
                (overlap_cfg.get("osd", {}) or {}).get("merge_gap_sec", 0.08),
                0.08,
            ),
        ),
    )
    merged_inputs: List[Dict[str, float]] = []
    for diar_segments in (msdd_segments, pyannote_segments, sortformer_segments):
        if len(diar_segments or []) < 2:
            continue
        merged_inputs.extend(
            derive_overlap_regions_from_diar_segments(
                diar_segments=diar_segments,
                min_duration_sec=min_region_sec,
                max_gap_sec=merge_gap_sec,
            )
        )

    return merge_time_regions(
        merged_inputs,
        max_gap_sec=merge_gap_sec,
        min_duration_sec=min_region_sec,
    )


def score_hybrid_candidate(
    candidate: Dict[str, Any],
    *,
    audio_duration: float,
    max_speakers: int,
    speaker_count_votes: Counter,
    requested_num_speakers: int = 0,
) -> float:
    backend = str(candidate.get("backend", "") or "")
    speaker_count = max(0, int(candidate.get("speaker_count", 0) or 0))
    turn_count = max(0, int(candidate.get("turn_count", 0) or 0))

    score_map = {
        "pyannote": 100.0,
        "msdd": 94.0,
        "sortformer": 82.0,
    }
    score = score_map.get(backend, 0.0)

    if backend == "pyannote":
        if float(audio_duration) >= 5 * 60:
            score += 8.0
        if speaker_count > 4:
            score += 4.0
    elif backend == "msdd":
        if speaker_count > 4:
            score += 14.0
        if float(audio_duration) >= 5 * 60:
            score += 6.0
    elif backend == "sortformer":
        if speaker_count <= 2 and float(audio_duration) <= 3 * 60:
            score += 8.0
        if speaker_count >= min(4, max(1, max_speakers)):
            score -= 18.0

    if speaker_count > max_speakers:
        score -= 12.0
    if speaker_count_votes.get(speaker_count, 0) > 1:
        score += 6.0
    if turn_count > 0:
        score += min(6.0, math.log1p(turn_count))

    fixed_num = max(0, int(requested_num_speakers or 0))
    if fixed_num > 0:
        distance = abs(int(speaker_count) - fixed_num)
        if distance == 0:
            manual_bonus = {
                "msdd": 64.0,
                "pyannote": 60.0,
                "sortformer": 56.0,
            }.get(backend, 58.0)
            score += manual_bonus
        else:
            manual_penalty = 28.0 * float(distance)
            if backend == "sortformer":
                manual_penalty += 6.0
            score -= manual_penalty
    return score


def select_hybrid_candidate(
    candidates: List[Dict[str, Any]],
    *,
    audio_duration: float,
    max_speakers: int,
    requested_num_speakers: int = 0,
) -> Tuple[List[Dict[str, Any]], str]:
    if not candidates:
        return [], ""

    speaker_count_votes = Counter(
        max(0, int(item.get("speaker_count", 0) or 0))
        for item in candidates
    )
    scored: List[Dict[str, Any]] = []
    for item in candidates:
        score = score_hybrid_candidate(
            item,
            audio_duration=audio_duration,
            max_speakers=max_speakers,
            speaker_count_votes=speaker_count_votes,
            requested_num_speakers=requested_num_speakers,
        )
        enriched = dict(item)
        enriched["_score"] = float(score)
        scored.append(enriched)

    scored.sort(
        key=lambda item: (
            float(item.get("_score", 0.0)),
            int(item.get("speaker_count", 0) or 0),
            int(item.get("turn_count", 0) or 0),
            str(item.get("route", "")),
        ),
        reverse=True,
    )
    best = scored[0]
    return list(best.get("segments") or []), str(best.get("route", "") or "")


def should_probe_pyannote_hybrid(
    candidates: List[Dict[str, Any]],
    *,
    audio_duration: float,
    max_speakers: int,
    requested_num_speakers: int = 0,
) -> bool:
    if int(requested_num_speakers or 0) > 0:
        return True
    if not candidates:
        return True

    speaker_counts = {
        max(0, int(item.get("speaker_count", 0) or 0))
        for item in candidates
    }
    if len(speaker_counts) > 1:
        return True
    if float(audio_duration) >= 8 * 60:
        return True

    capped_sortformer = [
        item
        for item in candidates
        if str(item.get("backend", "") or "") == "sortformer"
        and int(item.get("speaker_count", 0) or 0) >= min(4, max(1, max_speakers))
    ]
    return bool(capped_sortformer)
