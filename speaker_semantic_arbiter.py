from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional


AsyncLLMCaller = Callable[..., Awaitable[str]]


@dataclass
class SpeakerProfile:
    speaker: str
    role: str = ""
    style: str = ""
    evidence: List[str] = field(default_factory=list)


@dataclass
class SegmentDecision:
    index: int
    speaker: str
    confidence: float = 0.0
    highlight_terms: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class SegmentMergeDecision:
    indices: List[int] = field(default_factory=list)
    speaker: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass
class WindowArbitrationResponse:
    decisions: Dict[int, SegmentDecision] = field(default_factory=dict)
    merge_groups: List[SegmentMergeDecision] = field(default_factory=list)


@dataclass
class SpeakerArbitrationResult:
    decisions: Dict[int, SegmentDecision] = field(default_factory=dict)
    merge_groups: List[SegmentMergeDecision] = field(default_factory=list)
    speaker_profiles: Dict[str, SpeakerProfile] = field(default_factory=dict)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpeakerSemanticArbiter:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        llm_call: AsyncLLMCaller,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        llm_cfg: Dict[str, Any] = {}
        if isinstance(config, dict):
            raw_llm_cfg = config.get("llm", {})
            if isinstance(raw_llm_cfg, dict):
                llm_cfg = dict(raw_llm_cfg)
        else:
            getter = getattr(config, "get", None)
            if callable(getter):
                try:
                    raw_llm_cfg = getter("llm", {})
                except TypeError:
                    raw_llm_cfg = getter("llm")
                except Exception:
                    raw_llm_cfg = {}
                if isinstance(raw_llm_cfg, dict):
                    llm_cfg = dict(raw_llm_cfg)

        raw_cfg = llm_cfg.get("speaker_arbitration", {}) if isinstance(llm_cfg, dict) else {}
        self.cfg = dict(raw_cfg or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.window_chars = self._safe_int(self.cfg.get("window_chars", 9000), 9000, minimum=1500)
        self.max_segments_per_window = self._safe_int(
            self.cfg.get("max_segments_per_window", 36),
            36,
            minimum=6,
        )
        self.window_overlap_segments = self._safe_int(
            self.cfg.get("window_overlap_segments", 4),
            4,
            minimum=0,
        )
        self.max_highlight_terms = self._safe_int(
            self.cfg.get("max_highlight_terms", 3),
            3,
            minimum=1,
        )
        self.allow_segment_merge = bool(self.cfg.get("enable_segment_merge", True))
        self.max_merge_group_size = self._safe_int(
            self.cfg.get("max_merge_group_size", 4),
            4,
            minimum=2,
        )
        self.max_merge_gap_sec = self._safe_float(
            self.cfg.get("max_merge_gap_sec", 0.90),
            0.90,
            minimum=0.0,
        )
        self.max_merged_duration_sec = self._safe_float(
            self.cfg.get("max_merged_duration_sec", 10.0),
            10.0,
            minimum=0.5,
        )
        self.llm_call = llm_call
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _safe_int(value: Any, default: int, *, minimum: int = 0) -> int:
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return max(minimum, int(default))

    @staticmethod
    def _safe_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
        try:
            return max(minimum, float(value))
        except (TypeError, ValueError):
            return max(minimum, float(default))

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _compact_text(value: Any) -> str:
        return re.sub(r"[\W_]+", "", str(value or ""), flags=re.UNICODE)

    @staticmethod
    def _format_ts(seconds: Any) -> str:
        try:
            total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
        except (TypeError, ValueError):
            total_ms = 0
        hours, rem = divmod(total_ms, 3600000)
        minutes, rem = divmod(rem, 60000)
        secs, millis = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @staticmethod
    def _extract_json_payload(text: str) -> Any:
        source = str(text or "").strip()
        if not source:
            return None
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", source, flags=re.S | re.I)
        candidates = list(fenced)
        candidates.append(source)
        for candidate in candidates:
            payload = str(candidate or "").strip()
            if not payload:
                continue
            for raw in (payload, SpeakerSemanticArbiter._trim_json_container(payload)):
                if not raw:
                    continue
                try:
                    return json.loads(raw)
                except Exception:
                    continue
        return None

    @staticmethod
    def _trim_json_container(text: str) -> str:
        source = str(text or "").strip()
        if not source:
            return ""
        first_obj = source.find("{")
        last_obj = source.rfind("}")
        if 0 <= first_obj < last_obj:
            return source[first_obj : last_obj + 1]
        first_arr = source.find("[")
        last_arr = source.rfind("]")
        if 0 <= first_arr < last_arr:
            return source[first_arr : last_arr + 1]
        return source

    @staticmethod
    def _speaker_constraints_text(speaker_constraints: Dict[str, Any]) -> str:
        if not isinstance(speaker_constraints, dict):
            return "- No additional hard constraints."
        lines: List[str] = []
        num_speakers = int(speaker_constraints.get("num_speakers", 0) or 0)
        min_speakers = int(speaker_constraints.get("min_speakers", 0) or 0)
        max_speakers = int(speaker_constraints.get("max_speakers", 0) or 0)
        if num_speakers > 0:
            lines.append(f"- Exact speaker count constraint: {num_speakers}")
        else:
            if min_speakers > 0:
                lines.append(f"- Minimum speaker count constraint: {min_speakers}")
            if max_speakers > 0:
                lines.append(f"- Maximum speaker count constraint: {max_speakers}")
        ui_mode = str(speaker_constraints.get("mode", "") or "").strip()
        if ui_mode:
            lines.append(f"- UI speaker mode: {ui_mode}")
        return "\n".join(lines) if lines else "- No additional hard constraints."

    def _build_speaker_context(self, segments: Iterable[Any]) -> List[Dict[str, Any]]:
        speaker_rows: Dict[str, Dict[str, Any]] = {}
        for seg in segments or []:
            speaker = str(getattr(seg, "speaker", "") or "").strip()
            if not speaker:
                continue
            row = speaker_rows.setdefault(
                speaker,
                {
                    "speaker": speaker,
                    "duration": 0.0,
                    "count": 0,
                    "samples": [],
                },
            )
            start = self._safe_float(getattr(seg, "start", 0.0), 0.0)
            end = self._safe_float(getattr(seg, "end", start), start)
            row["duration"] += max(0.0, end - start)
            row["count"] += 1
            text = self._normalize_text(getattr(seg, "text", ""))
            if text and len(row["samples"]) < 3:
                row["samples"].append(text[:140])
        return sorted(
            speaker_rows.values(),
            key=lambda item: (-float(item["duration"]), -int(item["count"]), str(item["speaker"])),
        )

    def _build_windows(self, segments: List[Any]) -> List[List[Dict[str, Any]]]:
        windows: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_chars = 0

        normalized: List[Dict[str, Any]] = []
        for index, seg in enumerate(segments):
            text = self._normalize_text(getattr(seg, "text", ""))
            if not text:
                continue
            normalized.append(
                {
                    "index": index,
                    "speaker": str(getattr(seg, "speaker", "") or "").strip() or "UNKNOWN",
                    "start": self._safe_float(getattr(seg, "start", 0.0), 0.0),
                    "end": self._safe_float(getattr(seg, "end", 0.0), 0.0),
                    "text": text[:220],
                }
            )

        for idx, item in enumerate(normalized):
            start = float(item.get("start", 0.0) or 0.0)
            end = max(start, float(item.get("end", start) or start))
            item["duration"] = max(0.0, end - start)
            prev_item = normalized[idx - 1] if idx > 0 else None
            next_item = normalized[idx + 1] if idx + 1 < len(normalized) else None
            item["prev_speaker"] = str(prev_item.get("speaker", "")) if prev_item else ""
            item["next_speaker"] = str(next_item.get("speaker", "")) if next_item else ""
            item["gap_prev"] = (
                max(0.0, start - float(prev_item.get("end", start) or start))
                if prev_item
                else 0.0
            )
            item["gap_next"] = (
                max(0.0, float(next_item.get("start", end) or end) - end)
                if next_item
                else 0.0
            )
            compact = self._compact_text(item.get("text", ""))
            flags: List[str] = []
            if not compact:
                flags.append("punct")
            if len(compact) <= 6 or float(item["duration"]) <= 1.0:
                flags.append("short")
            if len(compact) <= 3 and float(item["duration"]) <= 0.8:
                flags.append("backchannel")
            item["flags"] = flags

        run_start = 0
        while run_start < len(normalized):
            run_end = run_start
            run_speaker = str(normalized[run_start].get("speaker", "") or "")
            while (
                run_end + 1 < len(normalized)
                and str(normalized[run_end + 1].get("speaker", "") or "") == run_speaker
            ):
                run_end += 1
            run_size = (run_end - run_start) + 1
            for run_idx in range(run_start, run_end + 1):
                normalized[run_idx]["run_pos"] = (run_idx - run_start) + 1
                normalized[run_idx]["run_size"] = run_size
            run_start = run_end + 1

        for item in normalized:
            rendered = self._window_line(item)
            extra = len(rendered) + (1 if current else 0)
            if (
                current
                and (
                    len(current) >= self.max_segments_per_window
                    or current_chars + extra > self.window_chars
                )
            ):
                windows.append(list(current))
                overlap = min(self.window_overlap_segments, len(current))
                current = list(current[-overlap:]) if overlap > 0 else []
                current_chars = sum(len(self._window_line(one)) for one in current)
            current.append(item)
            current_chars += extra

        if current:
            windows.append(list(current))
        return windows

    def _window_line(self, item: Dict[str, Any]) -> str:
        flags = ",".join(str(flag) for flag in list(item.get("flags") or []) if str(flag).strip())
        flag_suffix = f" | flags={flags}" if flags else ""
        return (
            f"#{int(item['index'])} "
            f"[{self._format_ts(item['start'])} - {self._format_ts(item['end'])}"
            f" | dur={float(item.get('duration', 0.0) or 0.0):.2f}s"
            f" | gap_prev={float(item.get('gap_prev', 0.0) or 0.0):.2f}s"
            f" | prev={str(item.get('prev_speaker', '') or '-')}"
            f" | next={str(item.get('next_speaker', '') or '-')}"
            f" | run={int(item.get('run_pos', 1) or 1)}/{int(item.get('run_size', 1) or 1)}"
            f"{flag_suffix}] "
            f"{item['speaker']}: {item['text']}"
        )

    async def _request_profiles(
        self,
        *,
        analysis_summary: str,
        analysis_points: List[str],
        speaker_context: List[Dict[str, Any]],
        allowed_speakers: List[str],
        speaker_constraints: Dict[str, Any],
        response_language: str,
    ) -> Dict[str, SpeakerProfile]:
        if not allowed_speakers:
            return {}

        context_blocks: List[str] = []
        for row in speaker_context:
            samples = list(row.get("samples") or [])
            sample_text = "\n".join(f"- {sample}" for sample in samples[:3]) or "- None"
            context_blocks.append(
                "\n".join(
                    [
                        f"Speaker {row['speaker']}",
                        f"Duration: {float(row.get('duration', 0.0)):.1f}s",
                        f"Turns: {int(row.get('count', 0) or 0)}",
                        "Samples:",
                        sample_text,
                    ]
                )
            )

        summary_lines = [self._normalize_text(analysis_summary)]
        summary_lines.extend(
            f"- {self._normalize_text(item)}"
            for item in list(analysis_points or [])[:6]
            if self._normalize_text(item)
        )
        summary_text = "\n".join(line for line in summary_lines if line)

        prompt = (
            "Build concise semantic role hints for a diarized transcript.\n"
            f"Allowed speakers: {', '.join(allowed_speakers)}\n"
            "Hard constraints:\n"
            f"{self._speaker_constraints_text(speaker_constraints)}\n\n"
            "Global summary:\n"
            f"{summary_text or '(none)'}\n\n"
            "Speaker evidence:\n"
            f"{chr(10).join(context_blocks)}\n\n"
            "Return JSON only with this schema:\n"
            "{\"speaker_profiles\":[{\"speaker\":\"A\",\"role\":\"...\",\"style\":\"...\",\"evidence\":[\"...\"]}]}\n\n"
            "Rules:\n"
            "- Only use allowed speaker labels.\n"
            "- Do not invent people outside the allowed speaker list.\n"
            "- Keep role/style concise.\n"
        )
        if response_language:
            prompt += (
                f"- Write role and style in language `{response_language}` when possible.\n"
            )

        result = await self.llm_call(
            [
                {
                    "role": "system",
                    "content": (
                        "You resolve speaker roles for transcript diarization. "
                        "Return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            enable_thinking=False,
            request_label="speaker_profiles",
        )
        payload = self._extract_json_payload(result)
        if not isinstance(payload, dict):
            return {}

        profiles: Dict[str, SpeakerProfile] = {}
        for raw in list(payload.get("speaker_profiles") or []):
            if not isinstance(raw, dict):
                continue
            speaker = str(raw.get("speaker", "") or "").strip()
            if speaker not in allowed_speakers:
                continue
            evidence = [
                self._normalize_text(item)
                for item in list(raw.get("evidence") or [])
                if self._normalize_text(item)
            ][:3]
            profiles[speaker] = SpeakerProfile(
                speaker=speaker,
                role=self._normalize_text(raw.get("role", ""))[:80],
                style=self._normalize_text(raw.get("style", ""))[:80],
                evidence=evidence,
            )
        return profiles

    async def _request_window_decisions(
        self,
        *,
        window_index: int,
        total_windows: int,
        window: List[Dict[str, Any]],
        analysis_summary: str,
        analysis_points: List[str],
        allowed_speakers: List[str],
        speaker_constraints: Dict[str, Any],
        speaker_profiles: Dict[str, SpeakerProfile],
    ) -> WindowArbitrationResponse:
        if not window:
            return WindowArbitrationResponse()

        summary_lines = [self._normalize_text(analysis_summary)]
        summary_lines.extend(
            f"- {self._normalize_text(item)}"
            for item in list(analysis_points or [])[:8]
            if self._normalize_text(item)
        )
        summary_text = "\n".join(line for line in summary_lines if line)
        profile_lines = []
        for speaker in allowed_speakers:
            profile = speaker_profiles.get(speaker)
            if profile is None:
                profile_lines.append(f"- {speaker}: role unknown")
                continue
            role = profile.role or "role unknown"
            style = f"; style={profile.style}" if profile.style else ""
            evidence = f"; evidence={', '.join(profile.evidence[:2])}" if profile.evidence else ""
            profile_lines.append(f"- {speaker}: {role}{style}{evidence}")

        prompt = (
            "Perform final semantic speaker arbitration for this transcript window.\n"
            f"Window {window_index}/{total_windows}\n"
            f"Allowed speakers: {', '.join(allowed_speakers)}\n"
            "Hard constraints:\n"
            f"{self._speaker_constraints_text(speaker_constraints)}\n\n"
            "Global summary:\n"
            f"{summary_text or '(none)'}\n\n"
            "Speaker profiles:\n"
            f"{chr(10).join(profile_lines)}\n\n"
            "Transcript window:\n"
            f"{chr(10).join(self._window_line(item) for item in window)}\n\n"
            "Return JSON only with this schema:\n"
            "{\"decisions\":[{\"index\":12,\"speaker\":\"A\",\"confidence\":0.92,"
            "\"highlight_terms\":[\"term1\",\"term2\"],\"reason\":\"...\"}],"
            "\"merge_groups\":[{\"indices\":[12,13],\"speaker\":\"A\",\"confidence\":0.88,"
            "\"reason\":\"continuous utterance\"}]}\n\n"
            "Rules:\n"
            "- Return one decision for every listed index.\n"
            "- speaker must be one of the allowed speaker labels.\n"
            "- Treat the current speaker labels as noisy hints, not ground truth.\n"
            "- If uncertain, keep the current speaker label from the transcript line.\n"
            "- confidence must be between 0 and 1.\n"
            f"- highlight_terms must contain 0 to {self.max_highlight_terms} short phrases copied verbatim from that line only.\n"
            "- Do not invent highlight terms or paraphrase them.\n"
            "- Use semantics, unfinished-thought continuity, turn-taking, question-answer flow, self-reference, and context continuity.\n"
            "- Very short acknowledgements/backchannels should be reassigned only when the surrounding context strongly shows they belong to the same real speaker thread.\n"
            "- Use merge_groups only for adjacent indices that become one continuous utterance after arbitration, with small internal gaps and no real speaker handoff.\n"
            f"- Each merge group must contain 2 to {self.max_merge_group_size} adjacent indices only.\n"
            f"- Do not merge across a clear question-answer exchange, explicit interruption, or total merged duration above about {self.max_merged_duration_sec:.1f}s.\n"
            f"- Prefer not to merge when any internal gap is clearly larger than about {self.max_merge_gap_sec:.2f}s.\n"
        )

        result = await self.llm_call(
            [
                {
                    "role": "system",
                    "content": (
                        "You do transcript speaker arbitration. "
                        "Return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            enable_thinking=False,
            request_label=f"speaker_window_{window_index}_{total_windows}",
        )
        payload = self._extract_json_payload(result)
        if not isinstance(payload, dict):
            return WindowArbitrationResponse()

        current_speakers = {int(item["index"]): str(item["speaker"]) for item in window}
        decisions: Dict[int, SegmentDecision] = {}
        for raw in list(payload.get("decisions") or []):
            if not isinstance(raw, dict):
                continue
            try:
                index = int(raw.get("index"))
            except (TypeError, ValueError):
                continue
            if index not in current_speakers:
                continue
            speaker = str(raw.get("speaker", "") or "").strip()
            if speaker not in allowed_speakers:
                speaker = current_speakers[index]
            confidence = self._safe_float(raw.get("confidence", 0.0), 0.0, minimum=0.0)
            confidence = max(0.0, min(1.0, confidence))
            highlights: List[str] = []
            line_text = next((item["text"] for item in window if int(item["index"]) == index), "")
            normalized_line = self._normalize_text(line_text)
            for item in list(raw.get("highlight_terms") or [])[: self.max_highlight_terms]:
                text = self._normalize_text(item)
                if not text or text not in normalized_line:
                    continue
                highlights.append(text)
            decisions[index] = SegmentDecision(
                index=index,
                speaker=speaker,
                confidence=confidence,
                highlight_terms=highlights,
                reason=self._normalize_text(raw.get("reason", ""))[:180],
            )
        merge_groups: List[SegmentMergeDecision] = []
        if self.allow_segment_merge:
            window_indices = {int(item["index"]) for item in window}
            for raw in list(payload.get("merge_groups") or []):
                if not isinstance(raw, dict):
                    continue
                raw_indices = list(raw.get("indices") or [])
                indices: List[int] = []
                for value in raw_indices:
                    try:
                        indices.append(int(value))
                    except (TypeError, ValueError):
                        continue
                if len(indices) < 2:
                    continue
                indices = sorted(dict.fromkeys(indices))
                if len(indices) < 2 or len(indices) > self.max_merge_group_size:
                    continue
                if any(index not in window_indices for index in indices):
                    continue
                if any((left + 1) != right for left, right in zip(indices, indices[1:])):
                    continue
                speaker = str(raw.get("speaker", "") or "").strip()
                if speaker and speaker not in allowed_speakers:
                    speaker = ""
                confidence = self._safe_float(raw.get("confidence", 0.0), 0.0, minimum=0.0)
                confidence = max(0.0, min(1.0, confidence))
                merge_groups.append(
                    SegmentMergeDecision(
                        indices=indices,
                        speaker=speaker,
                        confidence=confidence,
                        reason=self._normalize_text(raw.get("reason", ""))[:180],
                    )
                )
        return WindowArbitrationResponse(decisions=decisions, merge_groups=merge_groups)

    async def arbitrate(
        self,
        segments: List[Any],
        *,
        analysis_summary: str,
        analysis_points: Optional[List[str]] = None,
        response_language: str = "",
        speaker_constraints: Optional[Dict[str, Any]] = None,
    ) -> SpeakerArbitrationResult:
        result = SpeakerArbitrationResult()
        if not self.enabled:
            result.reason = "disabled"
            return result
        if not segments:
            result.reason = "no_segments"
            return result

        allowed_speakers = sorted(
            {
                str(getattr(seg, "speaker", "") or "").strip()
                for seg in segments
                if str(getattr(seg, "speaker", "") or "").strip()
            }
        )
        if len(allowed_speakers) <= 1:
            result.reason = "single_speaker"
            return result

        analysis_summary = self._normalize_text(analysis_summary)
        analysis_points = [
            self._normalize_text(item)
            for item in list(analysis_points or [])
            if self._normalize_text(item)
        ]
        if not analysis_summary and not analysis_points:
            result.reason = "missing_summary"
            return result

        speaker_constraints = dict(speaker_constraints or {})
        speaker_context = self._build_speaker_context(segments)
        windows = self._build_windows(segments)
        if not windows:
            result.reason = "no_window"
            return result

        speaker_profiles = await self._request_profiles(
            analysis_summary=analysis_summary,
            analysis_points=analysis_points,
            speaker_context=speaker_context,
            allowed_speakers=allowed_speakers,
            speaker_constraints=speaker_constraints,
            response_language=response_language,
        )
        merged: Dict[int, SegmentDecision] = {}
        merged_groups: Dict[tuple[int, ...], SegmentMergeDecision] = {}
        for window_index, window in enumerate(windows, start=1):
            window_response = await self._request_window_decisions(
                window_index=window_index,
                total_windows=len(windows),
                window=window,
                analysis_summary=analysis_summary,
                analysis_points=analysis_points,
                allowed_speakers=allowed_speakers,
                speaker_constraints=speaker_constraints,
                speaker_profiles=speaker_profiles,
            )
            for index, decision in window_response.decisions.items():
                prev = merged.get(index)
                if prev is None or float(decision.confidence) >= float(prev.confidence):
                    merged[index] = decision
            for merge_group in window_response.merge_groups:
                key = tuple(int(index) for index in list(merge_group.indices or []))
                if len(key) < 2:
                    continue
                prev_group = merged_groups.get(key)
                if prev_group is None or float(merge_group.confidence) >= float(prev_group.confidence):
                    merged_groups[key] = merge_group

        result.decisions = merged
        result.merge_groups = list(merged_groups.values())
        result.speaker_profiles = speaker_profiles
        result.reason = "ok" if (merged or merged_groups) else "no_decisions"
        result.metadata = {
            "window_count": len(windows),
            "allowed_speakers": allowed_speakers,
            "speaker_context_count": len(speaker_context),
            "profile_count": len(speaker_profiles),
            "decision_count": len(merged),
            "merge_group_count": len(merged_groups),
        }
        return result
