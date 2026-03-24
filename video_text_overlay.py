"""
video_text_overlay.py - Burn transcription text back into source videos.

Primary strategy:
FFmpeg burn-in with UTF-8 SRT/ASS subtitles and container-aware render plans.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import VIDEO_EXTENSIONS
from runtime_paths import find_tool_executable
from utils import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)


class VideoTextOverlayEngine:
    """Render ASR segments into video subtitles with an FFmpeg-based pipeline."""

    STYLE_PRESET_DEFAULT = "modern_box"
    STYLE_PRESET_CUSTOM = "custom"
    STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
        "modern_box": {
            "font_name": "Microsoft YaHei",
            "font_size": 28,
            "font_color": "#FFFFFF",
            "outline_color": "#000000",
            "outline_px": 1,
            "box_color": "#000000",
            "box_opacity": 70,
            "font_bold": True,
            "bottom_margin_px": 56,
            "side_margin_px": 72,
            "max_line_chars": 18,
            "max_lines_per_caption": 2,
        },
        "minimal_box": {
            "font_name": "Microsoft YaHei",
            "font_size": 24,
            "font_color": "#FFFFFF",
            "outline_color": "#000000",
            "outline_px": 1,
            "box_color": "#000000",
            "box_opacity": 62,
            "font_bold": False,
            "bottom_margin_px": 46,
            "side_margin_px": 82,
            "max_line_chars": 20,
            "max_lines_per_caption": 2,
        },
        "short_video": {
            "font_name": "Microsoft YaHei",
            "font_size": 30,
            "font_color": "#FFFFFF",
            "outline_color": "#000000",
            "outline_px": 1,
            "box_color": "#000000",
            "box_opacity": 74,
            "font_bold": True,
            "bottom_margin_px": 88,
            "side_margin_px": 44,
            "max_line_chars": 15,
            "max_lines_per_caption": 2,
        },
    }

    def __init__(self, config):
        self.config = config
        raw_cfg = config.get("video_text_overlay", {}) if hasattr(config, "get") else {}
        self.overlay_cfg: Dict[str, Any] = dict(raw_cfg or {})

        self.enabled = bool(self.overlay_cfg.get("enabled", False))
        renderer = str(self.overlay_cfg.get("renderer", "ffmpeg") or "ffmpeg").strip().lower()
        self.renderer = "ffmpeg" if renderer != "ffmpeg" else renderer
        self.copy_audio = bool(self.overlay_cfg.get("copy_audio", True))
        self.embed_subtitle_stream = bool(
            self.overlay_cfg.get("embed_subtitle_stream", True)
        )
        self.ffmpeg_video_codec = str(
            self.overlay_cfg.get("ffmpeg_video_codec", "h264_nvenc") or "h264_nvenc"
        ).strip()
        self.ffmpeg_preset = str(self.overlay_cfg.get("ffmpeg_preset", "p4") or "p4").strip()
        self.ffmpeg_crf = self._safe_int(self.overlay_cfg.get("ffmpeg_crf", 20), 20)
        self.webm_video_codec = str(
            self.overlay_cfg.get("webm_video_codec", "libvpx-vp9") or "libvpx-vp9"
        ).strip()
        self.webm_audio_codec = str(
            self.overlay_cfg.get("webm_audio_codec", "libopus") or "libopus"
        ).strip()
        self.webm_crf = self._safe_int(self.overlay_cfg.get("webm_crf", 32), 32)
        self.webm_cpu_used = self._safe_int(self.overlay_cfg.get("webm_cpu_used", 2), 2)
        self.webm_audio_bitrate = str(
            self.overlay_cfg.get("webm_audio_bitrate", "160k") or "160k"
        ).strip()

        style_cfg = self._resolve_style_config()
        self.style_preset = str(
            style_cfg.get("style_preset", self.STYLE_PRESET_DEFAULT)
            or self.STYLE_PRESET_DEFAULT
        ).strip().lower()
        self.font_name = str(
            style_cfg.get("font_name", "Microsoft YaHei") or "Microsoft YaHei"
        ).strip()
        self.font_size = self._safe_int(style_cfg.get("font_size", 26), 26)
        self.font_color = str(style_cfg.get("font_color", "#FFFFFF") or "#FFFFFF").strip()
        effect = str(self.overlay_cfg.get("style_effect", "auto") or "auto").strip().lower()
        if effect not in {"auto", "karaoke", "keyword", "plain"}:
            effect = "auto"
        self.style_effect = effect
        self.highlight_color = str(
            self.overlay_cfg.get("highlight_color", "#FFD54A") or "#FFD54A"
        ).strip()
        self.outline_color = str(
            style_cfg.get("outline_color", "#000000") or "#000000"
        ).strip()
        self.outline_px = self._safe_int(style_cfg.get("outline_px", 0), 0)
        self.box_color = str(style_cfg.get("box_color", "#000000") or "#000000").strip()
        self.box_opacity = max(
            0,
            min(100, self._safe_int(style_cfg.get("box_opacity", 78), 78)),
        )
        self.font_bold = bool(style_cfg.get("font_bold", True))
        self.bottom_margin_px = self._safe_int(style_cfg.get("bottom_margin_px", 42), 42)
        self.side_margin_px = self._safe_int(style_cfg.get("side_margin_px", 54), 54)
        self.max_line_chars = self._safe_int(style_cfg.get("max_line_chars", 26), 26)
        self.max_lines_per_caption = max(
            1,
            self._safe_int(style_cfg.get("max_lines_per_caption", 2), 2),
        )
        self.output_suffix = str(
            self.overlay_cfg.get("output_suffix", ".captioned") or ".captioned"
        ).strip()
        if self.output_suffix and not self.output_suffix.startswith("."):
            self.output_suffix = f".{self.output_suffix}"

        self.ffmpeg_binary = self._resolve_ffmpeg_binary()
        self.ffprobe_binary = self._resolve_ffprobe_binary()
        self.media_helper_binary = self._resolve_media_helper_binary()
        self._ffmpeg_encoder_cache: Optional[set[str]] = None
        self._videotoolbox_disabled = False
        self._videotoolbox_disabled_reason = ""
        self._active_temp_dir: Optional[Path] = None

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    def _resolve_style_config(self) -> Dict[str, Any]:
        requested = str(
            self.overlay_cfg.get("style_preset", self.STYLE_PRESET_DEFAULT)
            or self.STYLE_PRESET_DEFAULT
        ).strip().lower()

        if requested == self.STYLE_PRESET_CUSTOM:
            return {
                "style_preset": requested,
                "font_name": self.overlay_cfg.get("font_name", "Microsoft YaHei"),
                "font_size": self.overlay_cfg.get("font_size", 26),
                "font_color": self.overlay_cfg.get("font_color", "#FFFFFF"),
                "outline_color": self.overlay_cfg.get("outline_color", "#000000"),
                "outline_px": self.overlay_cfg.get("outline_px", 0),
                "box_color": self.overlay_cfg.get("box_color", "#000000"),
                "box_opacity": self.overlay_cfg.get("box_opacity", 78),
                "font_bold": self.overlay_cfg.get("font_bold", True),
                "bottom_margin_px": self.overlay_cfg.get("bottom_margin_px", 42),
                "side_margin_px": self.overlay_cfg.get("side_margin_px", 54),
                "max_line_chars": self.overlay_cfg.get("max_line_chars", 26),
                "max_lines_per_caption": self.overlay_cfg.get("max_lines_per_caption", 2),
            }

        preset = self.STYLE_PRESETS.get(requested)
        if preset is None:
            logger.warning(
                "Unknown video_text_overlay.style_preset=%s, fallback to %s",
                requested,
                self.STYLE_PRESET_DEFAULT,
            )
            requested = self.STYLE_PRESET_DEFAULT
            preset = self.STYLE_PRESETS[requested]

        resolved = dict(preset)
        # Presets provide the base look, while explicit config fields remain free to
        # fine-tune size, box opacity, and margins from the UI.
        for key in (
            "font_name",
            "font_size",
            "font_color",
            "outline_color",
            "outline_px",
            "box_color",
            "box_opacity",
            "font_bold",
            "bottom_margin_px",
            "side_margin_px",
            "max_line_chars",
            "max_lines_per_caption",
        ):
            value = self.overlay_cfg.get(key)
            if value not in (None, ""):
                resolved[key] = value
        resolved["style_preset"] = requested
        return resolved

    def _resolve_ffmpeg_binary(self) -> Optional[str]:
        configured = str(self.overlay_cfg.get("ffmpeg_path", "") or "").strip()
        resolved = find_tool_executable("ffmpeg", configured=configured)
        if resolved:
            return resolved

        audio_cfg_ffmpeg = str(self.config.get("audio.ffmpeg_path", "") or "").strip()
        resolved = find_tool_executable("ffmpeg", configured=audio_cfg_ffmpeg)
        if resolved:
            return resolved

        return find_tool_executable("ffmpeg")

    def _resolve_ffprobe_binary(self) -> Optional[str]:
        configured = str(self.overlay_cfg.get("ffprobe_path", "") or "").strip()
        resolved = find_tool_executable("ffprobe", configured=configured)
        if resolved:
            return resolved
        return find_tool_executable("ffprobe")

    def _resolve_media_helper_binary(self) -> Optional[str]:
        configured = str(self.overlay_cfg.get("native_media_helper_path", "") or "").strip()
        resolved = find_tool_executable("mts_media_helper", configured=configured)
        if resolved:
            return resolved
        return find_tool_executable("mts_media_helper")

    def is_enabled_for(self, source_file: Path) -> bool:
        return bool(self.enabled and source_file.suffix.lower() in VIDEO_EXTENSIONS)

    def _subtitle_paths(self, source_file: Path, output_dir: Path) -> Tuple[Path, Path, Path]:
        suffix = self.output_suffix or ".captioned"
        base_name = f"{source_file.stem}{suffix}"
        return (
            output_dir / f"{base_name}{source_file.suffix.lower()}",
            output_dir / f"{base_name}.srt",
            output_dir / f"{base_name}.ass",
        )

    def _resolve_render_temp_path(
        self,
        output_video: Path,
        *,
        suffix_tag: str,
    ) -> Path:
        temp_root = self._active_temp_dir or (output_video.parent / "temp")
        temp_root.mkdir(parents=True, exist_ok=True)
        return temp_root / f"{output_video.stem}.{suffix_tag}{output_video.suffix}"

    def _prefer_native_media_helper(self) -> bool:
        return bool(self.media_helper_binary and os.name == "nt")

    @staticmethod
    def _hex_rgb(value: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
        named = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "green": (0, 255, 0),
            "red": (255, 0, 0),
        }
        text = str(value or "").strip().lower()
        if text in named:
            return named[text]
        if text.startswith("#"):
            text = text[1:]
        if len(text) == 6:
            try:
                r = int(text[0:2], 16)
                g = int(text[2:4], 16)
                b = int(text[4:6], 16)
                return r, g, b
            except ValueError:
                return default
        return default

    @classmethod
    def _to_ass_color(cls, value: str, default: Tuple[int, int, int]) -> str:
        r, g, b = cls._hex_rgb(value, default=default)
        return f"&H00{b:02X}{g:02X}{r:02X}"

    @classmethod
    def _to_ass_box_color(
        cls,
        value: str,
        default: Tuple[int, int, int],
        *,
        opacity_percent: int,
    ) -> str:
        r, g, b = cls._hex_rgb(value, default=default)
        opacity = max(0, min(100, int(opacity_percent)))
        alpha = int(round(255.0 * (100.0 - float(opacity)) / 100.0))
        return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"

    @staticmethod
    def _format_srt_ts(seconds: float) -> str:
        total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
        hours, rem = divmod(total_ms, 3600000)
        minutes, rem = divmod(rem, 60000)
        secs, millis = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_ass_ts(seconds: float) -> str:
        total_cs = int(round(max(0.0, float(seconds)) * 100.0))
        hours, rem = divmod(total_cs, 360000)
        minutes, rem = divmod(rem, 6000)
        secs, centis = divmod(rem, 100)
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"

    @staticmethod
    def _char_display_width(ch: str) -> float:
        if not ch:
            return 0.0
        if ch.isspace():
            return 0.35
        if ch.isdigit():
            return 0.62
        if "a" <= ch <= "z":
            return 0.56
        if "A" <= ch <= "Z":
            return 0.68
        if ch in ",.;:!?，。！？；：、":
            return 0.45
        if ch in "([{<（【《「『":
            return 0.55
        if ch in ")]}>）】》」』":
            return 0.48
        if unicodedata.east_asian_width(ch) in {"W", "F"}:
            return 1.0
        if unicodedata.east_asian_width(ch) == "A":
            return 0.85
        return 0.72

    @classmethod
    def _text_display_width(cls, text: str) -> float:
        return sum(cls._char_display_width(ch) for ch in str(text or ""))

    @staticmethod
    def _last_break_index(buffer: List[str]) -> int:
        break_chars = set(" \t,.;:!?，。！？；：、)]}>）】》」』")
        for idx in range(len(buffer), 0, -1):
            if buffer[idx - 1] in break_chars:
                return idx
        return -1

    def _wrap_text(self, text: str) -> str:
        plain = str(text or "").strip()
        if not plain:
            return ""

        target_width = max(8.0, float(self.max_line_chars))
        wrapped_lines: List[str] = []
        for raw_line in plain.splitlines() or [plain]:
            line = str(raw_line or "").strip()
            if not line:
                continue

            if " " in line and self._text_display_width(line) <= target_width:
                wrapped_lines.append(line)
                continue

            buffer: List[str] = []
            line_width = 0.0

            for ch in line:
                if ch.isspace():
                    if not buffer or buffer[-1].isspace():
                        continue
                    ch = " "

                ch_width = self._char_display_width(ch)
                projected_width = line_width + ch_width

                if buffer and projected_width > target_width:
                    break_idx = self._last_break_index(buffer)
                    if break_idx > 0:
                        current = "".join(buffer[:break_idx]).strip()
                        if current:
                            wrapped_lines.append(current)
                        remainder = "".join(buffer[break_idx:]).lstrip()
                        buffer = list(remainder) if remainder else []
                        line_width = self._text_display_width(remainder)
                    else:
                        current = "".join(buffer).strip()
                        if current:
                            wrapped_lines.append(current)
                        buffer = []
                        line_width = 0.0

                if ch == " " and not buffer:
                    continue

                buffer.append(ch)
                line_width += ch_width

            tail = "".join(buffer).strip()
            if tail:
                wrapped_lines.append(tail)

        if not wrapped_lines:
            wrapped_lines = textwrap.wrap(plain, width=max(1, int(self.max_line_chars)))
        return "\n".join(wrapped_lines).strip()

    @staticmethod
    def _compose_caption_text(text: str, speaker: str) -> str:
        content = str(text or "").strip()
        speaker_name = str(speaker or "").strip()
        if speaker_name and speaker_name.upper() != "UNKNOWN":
            return f"{speaker_name}: {content}"
        return content

    def _effective_outline_px(self) -> int:
        outline_px = max(0, int(self.outline_px))
        if int(self.box_opacity) > 0 and outline_px <= 0:
            return 1
        return outline_px

    @staticmethod
    def _speaker_prefix(speaker: str) -> str:
        speaker_name = str(speaker or "").strip()
        if speaker_name and speaker_name.upper() != "UNKNOWN":
            return f"{speaker_name}: "
        return ""

    @staticmethod
    def _ass_escape_fragment(text: str) -> str:
        value = str(text or "")
        value = value.replace("\\", r"\\")
        value = value.replace("{", r"\{").replace("}", r"\}")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        return value.replace("\n", r"\N")

    def _normalize_segment_words(
        self,
        words: Any,
        *,
        seg_start: float,
        seg_end: float,
    ) -> List[Dict[str, Any]]:
        if not isinstance(words, (list, tuple)):
            return []

        seg_start_f = float(seg_start)
        seg_end_f = max(seg_start_f + 0.2, float(seg_end))
        raw_items: List[Dict[str, Any]] = []
        for raw in words:
            if isinstance(raw, dict):
                text = str(raw.get("text", raw.get("word", "")) or "")
                start = raw.get("start")
                end = raw.get("end")
            else:
                text = str(getattr(raw, "text", getattr(raw, "word", "")) or "")
                start = getattr(raw, "start", None)
                end = getattr(raw, "end", None)

            if not text.strip():
                continue

            item: Dict[str, Any] = {"text": text}
            try:
                if start is not None:
                    item["start"] = float(start)
            except Exception:
                pass
            try:
                if end is not None:
                    item["end"] = float(end)
            except Exception:
                pass

            raw_items.append(item)

        if not raw_items:
            return []

        raw_starts = [
            float(item["start"])
            for item in raw_items
            if item.get("start") is not None
        ]
        raw_ends = [
            float(item["end"])
            for item in raw_items
            if item.get("end") is not None
        ]
        should_reanchor = False
        if raw_starts and raw_ends:
            raw_min_start = min(raw_starts)
            raw_max_end = max(raw_ends)
            overlap_left = max(seg_start_f, raw_min_start)
            overlap_right = min(seg_end_f, raw_max_end)
            overlap = max(0.0, overlap_right - overlap_left)
            raw_span = max(0.0, raw_max_end - raw_min_start)
            seg_span = max(0.2, seg_end_f - seg_start_f)
            if (
                raw_max_end < seg_start_f - 0.5
                or raw_min_start > seg_end_f + 0.5
                or (
                    raw_span > 0.0
                    and overlap < min(0.15, raw_span * 0.1)
                    and abs(raw_min_start - seg_start_f) > 1.0
                    and abs(raw_max_end - seg_end_f) > 1.0
                )
            ):
                should_reanchor = True

        if should_reanchor and raw_starts:
            offset = seg_start_f - min(raw_starts)
            for item in raw_items:
                try:
                    if item.get("start") is not None:
                        item["start"] = float(item["start"]) + offset
                except Exception:
                    pass
                try:
                    if item.get("end") is not None:
                        item["end"] = float(item["end"]) + offset
                except Exception:
                    pass

        normalized: List[Dict[str, Any]] = []
        for item in raw_items:
            normalized_item: Dict[str, Any] = {"text": item["text"]}
            start_value = item.get("start")
            end_value = item.get("end")
            try:
                if start_value is not None:
                    normalized_item["start"] = max(seg_start_f, float(start_value))
            except Exception:
                pass
            try:
                if end_value is not None:
                    normalized_item["end"] = min(seg_end_f, float(end_value))
            except Exception:
                pass
            if (
                normalized_item.get("start") is not None
                and normalized_item.get("end") is not None
                and float(normalized_item["end"]) <= float(normalized_item["start"])
            ):
                normalized_item["end"] = min(
                    seg_end_f,
                    float(normalized_item["start"]) + 0.08,
                )
            normalized.append(normalized_item)

        return normalized

    @staticmethod
    def _words_support_karaoke(words: List[Dict[str, Any]]) -> bool:
        if not words:
            return False
        prev_end = None
        for item in words:
            start = item.get("start")
            end = item.get("end")
            if start is None or end is None:
                return False
            try:
                start_f = float(start)
                end_f = float(end)
            except Exception:
                return False
            if end_f <= start_f:
                return False
            if prev_end is not None and start_f + 0.12 < prev_end:
                return False
            prev_end = end_f
        return True

    def _token_display_text(self, text: str, *, is_line_start: bool) -> str:
        token = str(text or "").replace("\r", " ").replace("\n", " ")
        if is_line_start:
            return token.lstrip()
        return re.sub(r"^\s+", " ", token)

    def _chunk_plain_lines(
        self,
        lines: List[str],
        *,
        start: float,
        end: float,
    ) -> List[Dict[str, Any]]:
        chunk_size = max(1, int(self.max_lines_per_caption))
        if len(lines) <= chunk_size:
            return [{"start": float(start), "end": float(end), "lines": list(lines)}]

        grouped = [lines[pos : pos + chunk_size] for pos in range(0, len(lines), chunk_size)]
        duration = max(0.2, float(end) - float(start))
        if duration < 0.7 * len(grouped):
            return [{"start": float(start), "end": float(end), "lines": list(lines)}]

        weights = [
            max(1.0, self._text_display_width(" ".join(chunk)))
            for chunk in grouped
        ]
        total_weight = sum(weights) or float(len(grouped))
        cursor = float(start)
        consumed_weight = 0.0
        chunks: List[Dict[str, Any]] = []
        for pos, chunk in enumerate(grouped):
            if pos == len(grouped) - 1:
                chunk_end = float(end)
            else:
                consumed_weight += weights[pos]
                chunk_end = float(start) + duration * (consumed_weight / total_weight)
                chunk_end = max(cursor + 0.35, min(float(end), chunk_end))
            chunks.append({"start": cursor, "end": chunk_end, "lines": chunk})
            cursor = chunk_end
        return chunks

    def _wrap_word_lines(
        self,
        words: List[Dict[str, Any]],
        *,
        speaker_prefix: str,
    ) -> List[List[Dict[str, Any]]]:
        if not words:
            return []

        limit = max(8.0, float(self.max_line_chars))
        prefix_width = self._text_display_width(speaker_prefix)
        lines: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_width = prefix_width

        for raw in words:
            token = dict(raw)
            display_text = self._token_display_text(
                token.get("text", ""),
                is_line_start=not current,
            )
            if not display_text.strip():
                continue
            token_width = self._text_display_width(display_text)
            if current and current_width + token_width > limit:
                lines.append(current)
                current = []
                current_width = 0.0
                display_text = self._token_display_text(token.get("text", ""), is_line_start=True)
                token_width = self._text_display_width(display_text)
            token["display_text"] = display_text
            current.append(token)
            current_width += token_width

        if current:
            lines.append(current)
        return lines

    def _build_karaoke_ass_text(
        self,
        chunk_lines: List[List[Dict[str, Any]]],
        *,
        speaker_prefix: str,
    ) -> str:
        if not chunk_lines:
            return ""

        accent = self._to_ass_color(self.highlight_color, default=(255, 213, 74))
        base = self._to_ass_color(self.font_color, default=(255, 255, 255))
        pieces: List[str] = []
        if speaker_prefix:
            pieces.append(self._ass_escape_fragment(speaker_prefix))
        pieces.append(f"{{\\1c{accent}\\2c{base}}}")
        for line_index, tokens in enumerate(chunk_lines):
            if line_index > 0:
                pieces.append(r"\N")
            for token in tokens:
                start = token.get("start")
                end = token.get("end")
                try:
                    duration_cs = max(1, int(round((float(end) - float(start)) * 100.0)))
                except Exception:
                    duration_cs = 10
                pieces.append(f"{{\\kf{duration_cs}}}")
                pieces.append(
                    self._ass_escape_fragment(
                        token.get("display_text", token.get("text", ""))
                    )
                )
        pieces.append(r"{\rModernBox}")
        return "".join(pieces)

    @staticmethod
    def _segment_highlight_terms(meta: Any) -> List[str]:
        if isinstance(meta, dict):
            values = meta.get("semantic_highlights", meta.get("highlight_terms", []))
        else:
            values = getattr(meta, "semantic_highlights", [])
        if not isinstance(values, (list, tuple)):
            return []
        out: List[str] = []
        seen: set[str] = set()
        for item in values:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    @staticmethod
    def _non_overlapping_top_spans(
        candidates: List[Tuple[float, int, int]],
        *,
        limit: int,
    ) -> List[Tuple[int, int]]:
        chosen: List[Tuple[int, int]] = []
        for _score, start, end in sorted(candidates, reverse=True):
            if any(not (end <= left or start >= right) for left, right in chosen):
                continue
            chosen.append((start, end))
            if len(chosen) >= max(1, int(limit)):
                break
        return sorted(chosen)

    def _semantic_highlight_spans(
        self,
        text: str,
        *,
        highlight_terms: Optional[List[str]] = None,
        prefix_len: int = 0,
    ) -> List[Tuple[int, int]]:
        source = str(text or "")
        if not source.strip():
            return []
        spans: List[Tuple[float, int, int]] = []
        lowered = source.casefold()
        for term in highlight_terms or []:
            normalized = str(term or "").strip()
            if not normalized:
                continue
            start_at = max(0, int(prefix_len))
            needle = normalized.casefold()
            pos = lowered.find(needle, start_at)
            if pos < 0:
                continue
            spans.append((float(len(normalized)) + 6.0, pos, pos + len(normalized)))
        if not spans:
            return []
        return self._non_overlapping_top_spans(spans, limit=min(2, len(spans)))

    def _keyword_highlight_spans(
        self,
        text: str,
        *,
        highlight_terms: Optional[List[str]] = None,
        prefix_len: int = 0,
    ) -> List[Tuple[int, int]]:
        source = str(text or "")
        if not source.strip():
            return []

        semantic_spans = self._semantic_highlight_spans(
            source,
            highlight_terms=highlight_terms,
            prefix_len=prefix_len,
        )
        if semantic_spans:
            return semantic_spans

        stopwords = {
            "the", "and", "but", "for", "with", "this", "that", "from", "into",
            "your", "have", "will", "just", "than", "then", "they", "them",
        }
        candidates: List[Tuple[float, int, int]] = []
        pattern = re.compile(
            r"[A-Za-z][A-Za-z'-]{2,}|[0-9]+(?:[./:%-][0-9A-Za-z]+)*|[\u4e00-\u9fff]{2,8}"
        )
        for match in pattern.finditer(source):
            token = match.group(0)
            start, end = match.span()
            if end <= max(0, int(prefix_len)):
                continue
            lowered = token.lower()
            if re.fullmatch(r"[A-Za-z][A-Za-z'-]{2,}", token):
                if lowered in stopwords:
                    continue
                score = float(len(token))
                if token[:1].isupper():
                    score += 1.8
                if "-" in token or "'" in token:
                    score += 0.6
            elif re.fullmatch(r"[0-9]+(?:[./:%-][0-9A-Za-z]+)*", token):
                score = float(len(token)) + 2.4
            else:
                keep = min(max(2, len(token)), 6)
                start = max(start, end - keep)
                score = float(keep) + 1.8
                if keep >= 4:
                    score += 0.8
            if any(marker in token.lower() for marker in ("todo", "plan", "risk", "bug")):
                score += 1.5
            if re.search(r"(请|要|将|会|需|必须|完成|确认|安排|明天|今天|下周)", token):
                score += 1.2
            candidates.append((score + (end / max(1, len(source))), start, end))

        if not candidates:
            return []
        return self._non_overlapping_top_spans(candidates, limit=2)

    def _build_keyword_ass_text(
        self,
        text: str,
        *,
        highlight_terms: Optional[List[str]] = None,
        prefix_len: int = 0,
    ) -> str:
        source = str(text or "")
        spans = self._keyword_highlight_spans(
            source,
            highlight_terms=highlight_terms,
            prefix_len=prefix_len,
        )
        if not spans:
            return self._ass_escape_text(source)

        accent = self._to_ass_color(self.highlight_color, default=(255, 213, 74))
        out: List[str] = []
        cursor = 0
        for start, end in sorted(spans):
            out.append(self._ass_escape_fragment(source[cursor:start]))
            out.append(f"{{\\1c{accent}}}")
            out.append(self._ass_escape_fragment(source[start:end]))
            out.append(r"{\rModernBox}")
            cursor = end
        out.append(self._ass_escape_fragment(source[cursor:]))
        return "".join(out)

    def _iter_render_segments(
        self,
        segments: Iterable[Any],
    ) -> Iterable[Dict[str, Any]]:
        rendered_index = 0
        for _idx, start, end, text, speaker, words, meta in self._iter_segments(segments):
            speaker_prefix = self._speaker_prefix(speaker)
            normalized_words = self._normalize_segment_words(
                words,
                seg_start=start,
                seg_end=end,
            )
            highlight_terms = self._segment_highlight_terms(meta)
            if (
                self.style_effect in {"auto", "karaoke"}
                and self._words_support_karaoke(normalized_words)
            ):
                wrapped_word_lines = self._wrap_word_lines(
                    normalized_words,
                    speaker_prefix=speaker_prefix,
                )
                if wrapped_word_lines:
                    chunk_size = max(1, int(self.max_lines_per_caption))
                    for chunk_pos in range(0, len(wrapped_word_lines), chunk_size):
                        chunk_lines = wrapped_word_lines[chunk_pos : chunk_pos + chunk_size]
                        chunk_words = [token for line in chunk_lines for token in line]
                        if not chunk_words:
                            continue
                        rendered_index += 1
                        chunk_prefix = speaker_prefix if chunk_pos == 0 else ""
                        plain_lines: List[str] = []
                        for line_idx, line_tokens in enumerate(chunk_lines):
                            line_text = "".join(
                                str(token.get("display_text", token.get("text", "")))
                                for token in line_tokens
                            ).strip()
                            if line_idx == 0 and chunk_prefix:
                                line_text = (
                                    f"{chunk_prefix}{line_text}"
                                    if line_text
                                    else chunk_prefix.rstrip()
                                )
                            plain_lines.append(line_text)
                        yield {
                            "index": rendered_index,
                            "start": float(chunk_words[0].get("start", start) or start),
                            "end": float(chunk_words[-1].get("end", end) or end),
                            "text": "\n".join(line for line in plain_lines if line.strip()),
                            "ass_text": self._build_karaoke_ass_text(
                                chunk_lines,
                                speaker_prefix=chunk_prefix,
                            ),
                        }
                    continue

            wrapped = self._wrap_text(self._compose_caption_text(text, speaker))
            lines = [line.strip() for line in wrapped.splitlines() if line.strip()]
            if not lines:
                continue
            for chunk in self._chunk_plain_lines(lines, start=start, end=end):
                rendered_index += 1
                plain_text = "\n".join(chunk["lines"])
                ass_text = self._ass_escape_text(plain_text)
                if self.style_effect in {"auto", "keyword"}:
                    ass_text = self._build_keyword_ass_text(
                        plain_text,
                        highlight_terms=highlight_terms,
                        prefix_len=len(speaker_prefix),
                    )
                yield {
                    "index": rendered_index,
                    "start": float(chunk["start"]),
                    "end": float(chunk["end"]),
                    "text": plain_text,
                    "ass_text": ass_text,
                }

    @staticmethod
    def _ass_escape_text(text: str) -> str:
        value = str(text or "")
        value = value.replace("\\", r"\\")
        value = value.replace("{", r"\{").replace("}", r"\}")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        return value.replace("\n", r"\N")

    @staticmethod
    def _iter_segments(
        segments: Iterable[Any],
    ) -> Iterable[Tuple[int, float, float, str, str, List[Dict[str, Any]], Dict[str, Any]]]:
        index = 0
        for seg in segments or []:
            if isinstance(seg, tuple) and len(seg) in {5, 6, 7}:
                idx_raw, start, end, text, speaker = seg[:5]
                words = list(seg[5]) if len(seg) >= 6 and isinstance(seg[5], (list, tuple)) else []
                meta = dict(seg[6]) if len(seg) >= 7 and isinstance(seg[6], dict) else {}
                try:
                    idx_val = int(idx_raw)
                except (TypeError, ValueError):
                    idx_val = index + 1
                index = max(index + 1, idx_val)
                yield index, float(start), float(end), str(text), str(speaker), words, meta
                continue
            start = float(getattr(seg, "start", 0.0) or 0.0)
            end = float(getattr(seg, "end", 0.0) or 0.0)
            text = str(getattr(seg, "text", "") or "").strip()
            speaker = str(getattr(seg, "speaker", "") or "").strip()
            words = list(getattr(seg, "words", []) or [])
            meta = {
                "semantic_highlights": list(getattr(seg, "semantic_highlights", []) or []),
                "speaker_role": str(getattr(seg, "speaker_role", "") or "").strip(),
                "arbitration_confidence": float(
                    getattr(seg, "arbitration_confidence", 0.0) or 0.0
                ),
            }
            if not text:
                continue
            if end <= start:
                end = start + 0.2
            index += 1
            yield index, start, end, text, speaker, words, meta

    def _write_srt(self, segments: Iterable[Any], srt_path: Path) -> None:
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(srt_path, "w", encoding="utf-8") as f:
            for item in self._iter_render_segments(segments):
                f.write(f"{item['index']}\n")
                f.write(
                    f"{self._format_srt_ts(item['start'])} --> {self._format_srt_ts(item['end'])}\n"
                )
                f.write(f"{item['text']}\n\n")

    def _write_ass(
        self,
        segments: Iterable[Any],
        ass_path: Path,
        *,
        width: int = 0,
        height: int = 0,
    ) -> None:
        ass_path.parent.mkdir(parents=True, exist_ok=True)

        play_res_x = max(640, int(width or 1920))
        play_res_y = max(360, int(height or 1080))
        safe_font = self.font_name.replace(",", " ").strip() or "Microsoft YaHei"
        primary = self._to_ass_color(self.font_color, default=(255, 255, 255))
        outline = self._to_ass_color(self.outline_color, default=(0, 0, 0))
        back = self._to_ass_box_color(
            self.box_color,
            default=(0, 0, 0),
            opacity_percent=self.box_opacity,
        )
        bold = -1 if self.font_bold else 0
        outline_px = self._effective_outline_px()
        style_line = ",".join(
            [
                "ModernBox",
                safe_font,
                str(max(8, self.font_size)),
                primary,
                primary,
                outline,
                back,
                str(bold),
                "0",
                "0",
                "0",
                "100",
                "100",
                "0.2",
                "0",
                "3",
                str(outline_px),
                "0",
                "2",
                str(max(0, self.side_margin_px)),
                str(max(0, self.side_margin_px)),
                str(max(0, self.bottom_margin_px)),
                "1",
            ]
        )

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write("[Script Info]\n")
            f.write("ScriptType: v4.00+\n")
            f.write("WrapStyle: 2\n")
            f.write("ScaledBorderAndShadow: yes\n")
            f.write(f"PlayResX: {play_res_x}\n")
            f.write(f"PlayResY: {play_res_y}\n")
            f.write("\n[V4+ Styles]\n")
            f.write(
                "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
                "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,"
                "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
                "Alignment,MarginL,MarginR,MarginV,Encoding\n"
            )
            f.write(f"Style: {style_line}\n")
            f.write("\n[Events]\n")
            f.write(
                "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
            )
            for item in self._iter_render_segments(segments):
                f.write(
                    "Dialogue: 0,"
                    f"{self._format_ass_ts(item['start'])},"
                    f"{self._format_ass_ts(item['end'])},"
                    f"ModernBox,,0,0,0,,{item['ass_text']}\n"
                )

    @staticmethod
    def _ffmpeg_escape_filter_path(path: Path) -> str:
        safe_path = path.resolve().as_posix()
        safe_path = safe_path.replace(":", r"\:")
        safe_path = safe_path.replace("'", r"\'")
        return safe_path

    def _subtitle_force_style(self) -> str:
        safe_font = self.font_name.replace(",", " ").replace("'", r"\'").strip()
        if not safe_font:
            safe_font = "Microsoft YaHei"

        border_style = 3 if int(self.box_opacity) > 0 else 1
        outline_px = self._effective_outline_px()

        style_items = [
            ("FontName", safe_font),
            ("FontSize", max(1, int(self.font_size))),
            ("PrimaryColour", self._to_ass_color(self.font_color, default=(255, 255, 255))),
            ("OutlineColour", self._to_ass_color(self.outline_color, default=(0, 0, 0))),
            (
                "BackColour",
                self._to_ass_box_color(
                    self.box_color,
                    default=(0, 0, 0),
                    opacity_percent=self.box_opacity,
                ),
            ),
            ("BorderStyle", border_style),
            ("Outline", outline_px),
            ("Shadow", 0),
            ("Alignment", 2),
            ("MarginL", max(0, int(self.side_margin_px))),
            ("MarginR", max(0, int(self.side_margin_px))),
            ("MarginV", max(0, int(self.bottom_margin_px))),
            ("Bold", -1 if self.font_bold else 0),
            ("WrapStyle", 2),
            ("Spacing", 0.2),
        ]
        return ",".join(f"{key}={value}" for key, value in style_items)

    def _ffmpeg_subtitles_filter(self, srt_path: Path, source_info: Dict[str, Any]) -> str:
        parts = [f"'{self._ffmpeg_escape_filter_path(srt_path)}'", "charenc=UTF-8"]
        width = int(source_info.get("width") or 0)
        height = int(source_info.get("height") or 0)
        if width > 0 and height > 0:
            parts.append(f"original_size={width}x{height}")

        fonts_dir = str(self.overlay_cfg.get("fonts_dir", "") or "").strip()
        if fonts_dir:
            font_path = Path(fonts_dir).expanduser()
            if font_path.exists() and font_path.is_dir():
                parts.append(f"fontsdir='{self._ffmpeg_escape_filter_path(font_path)}'")

        force_style = self._subtitle_force_style()
        if force_style:
            parts.append(f"force_style='{force_style}'")
        return "subtitles=" + ":".join(parts)

    def _ffmpeg_ass_filter(self, ass_path: Path, source_info: Dict[str, Any]) -> str:
        parts = [f"'{self._ffmpeg_escape_filter_path(ass_path)}'"]
        width = int(source_info.get("width") or 0)
        height = int(source_info.get("height") or 0)
        if width > 0 and height > 0:
            parts.append(f"original_size={width}x{height}")

        fonts_dir = str(self.overlay_cfg.get("fonts_dir", "") or "").strip()
        if fonts_dir:
            font_path = Path(fonts_dir).expanduser()
            if font_path.exists() and font_path.is_dir():
                parts.append(f"fontsdir='{self._ffmpeg_escape_filter_path(font_path)}'")
        parts.append("shaping=auto")
        return "ass=" + ":".join(parts)

    def _ffmpeg_burn_filter_plans(
        self,
        *,
        srt_path: Path,
        ass_path: Path,
        source_info: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        plans: List[Tuple[str, str]] = []
        if ass_path.exists():
            plans.append(("ass", self._ffmpeg_ass_filter(ass_path, source_info)))
        if srt_path.exists():
            plans.append(("subtitles", self._ffmpeg_subtitles_filter(srt_path, source_info)))
        return plans

    @staticmethod
    def _helper_error_text(stdout: str, stderr: str) -> str:
        candidates = [str(stderr or "").strip(), str(stdout or "").strip()]
        for raw in candidates:
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                message = str(payload.get("error", "") or "").strip()
                if message:
                    return message
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if lines:
                return lines[-1]
        return "native media helper failed"

    def _run_media_helper(self, args: List[str], timeout: int = 7200) -> subprocess.CompletedProcess[str]:
        if not self.media_helper_binary:
            raise RuntimeError("native media helper is not available")

        cmd = [str(self.media_helper_binary), *args]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **windows_hidden_subprocess_kwargs(),
        )
        if result.returncode == 0:
            helper_stdout = str(result.stdout or "").strip()
            if helper_stdout.startswith('{"ok":true') or helper_stdout.startswith('{"ok": true'):
                logger.info("Native media helper success: %s", helper_stdout)
            return result
        raise RuntimeError(self._helper_error_text(result.stdout, result.stderr))

    def _helper_fonts_dir(self) -> str:
        fonts_dir = str(self.overlay_cfg.get("fonts_dir", "") or "").strip()
        if not fonts_dir:
            return ""
        font_path = Path(fonts_dir).expanduser()
        if font_path.exists() and font_path.is_dir():
            return str(font_path)
        return ""

    def _probe_media_info(self, media_path: Path) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "width": 0,
            "height": 0,
            "video_codec": "",
            "audio_codec": "",
            "subtitle_codec": "",
            "audio_stream_count": 0,
            "subtitle_stream_count": 0,
            "has_audio": False,
        }
        if not self.ffprobe_binary:
            return info

        try:
            if self.media_helper_binary:
                result = self._run_media_helper(
                    [
                        "probe",
                        "--ffprobe",
                        str(self.ffprobe_binary),
                        "--input",
                        str(media_path),
                        "--timeout",
                        "30",
                    ],
                    timeout=35,
                )
            else:
                result = subprocess.run(
                    [
                        str(self.ffprobe_binary),
                        "-v",
                        "error",
                        "-print_format",
                        "json",
                        "-show_streams",
                        "-show_format",
                        str(media_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    **windows_hidden_subprocess_kwargs(),
                )
        except Exception as exc:
            logger.debug("ffprobe media probe failed for %s: %s", media_path, exc)
            return info

        if result.returncode != 0 or not str(result.stdout or "").strip():
            return info

        try:
            payload = json.loads(result.stdout)
        except Exception as exc:
            logger.debug("ffprobe JSON parse failed for %s: %s", media_path, exc)
            return info

        streams = payload.get("streams") or []
        if not isinstance(streams, list):
            return info

        video_stream = next(
            (one for one in streams if str(one.get("codec_type") or "").lower() == "video"),
            {},
        )
        audio_stream = next(
            (one for one in streams if str(one.get("codec_type") or "").lower() == "audio"),
            {},
        )
        subtitle_stream = next(
            (one for one in streams if str(one.get("codec_type") or "").lower() == "subtitle"),
            {},
        )
        audio_count = sum(
            1 for one in streams if str(one.get("codec_type") or "").lower() == "audio"
        )
        subtitle_count = sum(
            1 for one in streams if str(one.get("codec_type") or "").lower() == "subtitle"
        )
        info["width"] = int(video_stream.get("width") or 0)
        info["height"] = int(video_stream.get("height") or 0)
        info["video_codec"] = str(video_stream.get("codec_name") or "").strip().lower()
        info["audio_codec"] = str(audio_stream.get("codec_name") or "").strip().lower()
        info["subtitle_codec"] = str(subtitle_stream.get("codec_name") or "").strip().lower()
        info["audio_stream_count"] = int(audio_count)
        info["subtitle_stream_count"] = int(subtitle_count)
        info["has_audio"] = bool(audio_count > 0)
        return info

    def describe_embedded_subtitle_stream(self, media_path: Path) -> Dict[str, Any]:
        info = self._probe_media_info(Path(media_path))
        return {
            "subtitle_stream_count": int(info.get("subtitle_stream_count", 0) or 0),
            "subtitle_codec": str(info.get("subtitle_codec", "") or "").strip().lower(),
        }

    def can_embed_subtitle_stream(
        self,
        output_video: Path,
        *,
        srt_path: Path,
        ass_path: Path,
    ) -> bool:
        subtitle_input, subtitle_codec = self._subtitle_stream_plan(
            Path(output_video),
            srt_path=Path(srt_path),
            ass_path=Path(ass_path),
        )
        return subtitle_input is not None and bool(subtitle_codec)

    def _load_ffmpeg_encoder_cache(self) -> set[str]:
        if self._ffmpeg_encoder_cache is not None:
            return self._ffmpeg_encoder_cache
        if not self.ffmpeg_binary:
            self._ffmpeg_encoder_cache = set()
            return self._ffmpeg_encoder_cache

        try:
            if self.media_helper_binary:
                result = self._run_media_helper(
                    ["encoders", "--ffmpeg", str(self.ffmpeg_binary), "--timeout", "30"],
                    timeout=35,
                )
            else:
                result = subprocess.run(
                    [str(self.ffmpeg_binary), "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    **windows_hidden_subprocess_kwargs(),
                )
        except Exception as exc:
            logger.debug("ffmpeg encoder probe failed: %s", exc)
            self._ffmpeg_encoder_cache = set()
            return self._ffmpeg_encoder_cache

        encoder_names: set[str] = set()
        text = f"{result.stdout or ''}\n{result.stderr or ''}"
        for line in text.splitlines():
            match = re.match(r"^\s*[VAS.]{6}\s+([^\s]+)\s+", line)
            if match:
                encoder_names.add(str(match.group(1)).strip())
        self._ffmpeg_encoder_cache = encoder_names
        return self._ffmpeg_encoder_cache

    @staticmethod
    def _unique_values(values: Iterable[Optional[str]]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def _filter_available_encoders(self, candidates: Iterable[str]) -> List[str]:
        normalized = self._unique_values(candidates)
        available = self._load_ffmpeg_encoder_cache()
        if not available:
            return normalized
        preferred = [name for name in normalized if name in available]
        return preferred or normalized

    def _normalize_preset(self, video_codec: str) -> str:
        preset = str(self.ffmpeg_preset or "").strip().lower()
        if not preset:
            return ""

        if video_codec.endswith("_nvenc"):
            nvenc_aliases = {
                "default": "p4",
                "slow": "p6",
                "medium": "p4",
                "fast": "p2",
                "hp": "p1",
                "hq": "p5",
                "bd": "p4",
                "ll": "p2",
                "llhq": "p4",
                "llhp": "p1",
                "lossless": "p7",
                "losslesshp": "p7",
            }
            allowed = {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}
            return nvenc_aliases.get(preset, preset if preset in allowed else "p4")

        if video_codec in {"libx264", "libx265"}:
            allowed = {
                "ultrafast",
                "superfast",
                "veryfast",
                "faster",
                "fast",
                "medium",
                "slow",
                "slower",
                "veryslow",
                "placebo",
            }
            return preset if preset in allowed else "medium"

        return ""

    @staticmethod
    def _is_videotoolbox_codec(video_codec: str) -> bool:
        return str(video_codec or "").strip().lower() in {
            "h264_videotoolbox",
            "hevc_videotoolbox",
        }

    @staticmethod
    def _is_videotoolbox_init_error(exc: Exception | str) -> bool:
        text = str(exc or "").strip().lower()
        if not text:
            return False
        markers = (
            "cannot create compression session",
            "hardware encoder may be busy",
            "error while opening encoder",
            "-12908",
            "compression session",
            "vtencoderxpcservice",
        )
        return any(marker in text for marker in markers)

    def _disable_videotoolbox_for_session(self, reason: str) -> None:
        text = str(reason or "VideoToolbox unavailable").strip()
        if self._videotoolbox_disabled:
            return
        self._videotoolbox_disabled = True
        self._videotoolbox_disabled_reason = text
        logger.warning("VideoToolbox disabled for current run: %s", text)

    def _standard_video_codecs(self) -> List[str]:
        codecs = self._filter_available_encoders(
            [
                self.ffmpeg_video_codec,
                "h264_videotoolbox",
                "hevc_videotoolbox",
                "libx264",
                "mpeg4",
            ]
        )
        if self._videotoolbox_disabled:
            codecs = [codec for codec in codecs if not self._is_videotoolbox_codec(codec)]
        return codecs

    def _webm_video_codecs(self) -> List[str]:
        return self._filter_available_encoders(
            [
                self.webm_video_codec,
                "libvpx-vp9",
                "libvpx",
            ]
        )

    def _webm_audio_codecs(self, has_audio: bool) -> List[str]:
        if not has_audio:
            return []
        return self._filter_available_encoders(
            [
                self.webm_audio_codec,
                "libopus",
                "opus",
                "libvorbis",
            ]
        )

    def _standard_audio_mode(
        self,
        target_ext: str,
        *,
        has_audio: bool,
        prefer_copy: bool,
    ) -> Optional[str]:
        if not has_audio:
            return None
        if target_ext == ".webm":
            codecs = self._webm_audio_codecs(has_audio=True)
            return codecs[0] if codecs else None
        if prefer_copy and self.copy_audio:
            return "copy"
        return "aac"

    def _video_encode_args(self, video_codec: str) -> List[str]:
        args: List[str] = ["-pix_fmt", "yuv420p"]
        preset = self._normalize_preset(video_codec)
        if video_codec.endswith("_nvenc"):
            if preset:
                args.extend(["-preset", preset])
            args.extend(["-cq", str(max(0, self.ffmpeg_crf))])
        elif video_codec in {"h264_videotoolbox", "hevc_videotoolbox"}:
            if sys.platform == "darwin":
                args.extend(["-allow_sw", "1", "-realtime", "true"])
            args.extend(["-b:v", "0"])
            args.extend(["-q:v", "65" if video_codec == "h264_videotoolbox" else "55"])
        elif video_codec in {"libx264", "libx265"}:
            if preset:
                args.extend(["-preset", preset])
            args.extend(["-crf", str(max(0, self.ffmpeg_crf))])
        elif video_codec == "mpeg4":
            args.extend(["-q:v", "3"])
        elif video_codec == "libvpx-vp9":
            args.extend(
                [
                    "-deadline",
                    "good",
                    "-cpu-used",
                    str(max(0, self.webm_cpu_used)),
                    "-row-mt",
                    "1",
                    "-tile-columns",
                    "1",
                    "-crf",
                    str(max(0, min(63, self.webm_crf))),
                    "-b:v",
                    "0",
                ]
            )
        elif video_codec == "libvpx":
            args.extend(
                [
                    "-deadline",
                    "good",
                    "-cpu-used",
                    str(max(0, self.webm_cpu_used)),
                    "-crf",
                    str(max(4, min(63, self.webm_crf))),
                    "-b:v",
                    "0",
                ]
            )
        return args

    def _audio_encode_args(self, audio_codec: Optional[str]) -> List[str]:
        codec = str(audio_codec or "").strip()
        if not codec or codec == "copy":
            return []
        if codec in {"libopus", "opus"}:
            return ["-b:a", self.webm_audio_bitrate, "-vbr", "on"]
        if codec == "libvorbis":
            return ["-q:a", "5"]
        if codec == "aac":
            return ["-b:a", "192k"]
        return []

    def _ffmpeg_input_variants(self, video_codec: str) -> List[Tuple[str, List[str]]]:
        if sys.platform != "darwin" or not self._is_videotoolbox_codec(video_codec):
            return [("default", [])]
        return [
            ("hwdec", ["-hwaccel", "videotoolbox"]),
            ("default", []),
        ]

    def _subtitle_stream_plan(
        self,
        output_video: Path,
        *,
        srt_path: Path,
        ass_path: Path,
    ) -> Tuple[Optional[Path], str]:
        ext = output_video.suffix.lower()
        if ext in {".mp4", ".mov", ".m4v"}:
            return srt_path, "mov_text"
        if ext == ".mkv":
            if ass_path.exists():
                return ass_path, "ass"
            return srt_path, "srt"
        if ext == ".webm":
            return srt_path, "webvtt"
        return None, ""

    def _embed_subtitle_stream_if_supported(
        self,
        output_video: Path,
        *,
        srt_path: Path,
        ass_path: Path,
    ) -> Dict[str, str]:
        if not self.embed_subtitle_stream or not self.ffmpeg_binary:
            return {}

        subtitle_input, subtitle_codec = self._subtitle_stream_plan(
            output_video,
            srt_path=srt_path,
            ass_path=ass_path,
        )
        if subtitle_input is None or not subtitle_codec:
            logger.info(
                "Skip subtitle stream mux: container %s has no configured subtitle stream plan",
                output_video.suffix.lower() or "<none>",
            )
            return {}
        if not subtitle_input.exists():
            logger.warning("Skip subtitle stream mux: subtitle source missing: %s", subtitle_input)
            return {}

        available_encoders = self._load_ffmpeg_encoder_cache()
        if available_encoders and subtitle_codec not in available_encoders:
            logger.warning(
                "Skip subtitle stream mux: subtitle encoder %s is unavailable in ffmpeg",
                subtitle_codec,
            )
            return {}

        temp_output = self._resolve_render_temp_path(output_video, suffix_tag="muxing")
        self._cleanup_file(temp_output)
        cmd = [
            str(self.ffmpeg_binary),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(output_video),
            "-i",
            str(subtitle_input),
            "-map",
            "0",
            "-map",
            "1:0",
            "-c",
            "copy",
            "-c:s",
            subtitle_codec,
            "-metadata:s:s:0",
            "title=Transcription",
            "-disposition:s:0",
            "default",
            str(temp_output),
        ]
        try:
            logger.info(
                "FFmpeg subtitle stream mux: container=%s codec=%s source=%s",
                output_video.suffix.lower() or "<none>",
                subtitle_codec,
                subtitle_input.name,
            )
            if self.media_helper_binary:
                self._run_native_mux_attempt(
                    output_video=output_video,
                    temp_output=temp_output,
                    subtitle_input=subtitle_input,
                    subtitle_codec=subtitle_codec,
                )
            else:
                self._run_ffmpeg(cmd)
                self._ensure_output(temp_output, label="subtitle_stream_mux")
            stream_info = self.describe_embedded_subtitle_stream(temp_output)
            if int(stream_info.get("subtitle_stream_count", 0) or 0) <= 0:
                raise RuntimeError(
                    "subtitle stream mux finished but ffprobe detected no subtitle stream"
                )
            self._cleanup_file(output_video)
            temp_output.replace(output_video)
            verified_codec = str(stream_info.get("subtitle_codec", "") or "").strip().lower()
            logger.info(
                "FFmpeg subtitle stream verified: container=%s codec=%s",
                output_video.suffix.lower() or "<none>",
                verified_codec or subtitle_codec,
            )
            return {
                "subtitle_stream_embedded": "true",
                "subtitle_stream_codec": verified_codec or subtitle_codec,
            }
        except Exception as exc:
            logger.warning("FFmpeg subtitle stream mux failed: %s", exc)
            self._cleanup_file(temp_output)
            return {}

    def ensure_embedded_subtitle_stream(
        self,
        output_video: Path,
        *,
        srt_path: Path,
        ass_path: Path,
    ) -> Dict[str, str]:
        if not self.embed_subtitle_stream or not self.ffmpeg_binary:
            return {}

        if not self.can_embed_subtitle_stream(output_video, srt_path=srt_path, ass_path=ass_path):
            return {}

        current_stream = self.describe_embedded_subtitle_stream(output_video)
        if int(current_stream.get("subtitle_stream_count", 0) or 0) > 0:
            current_codec = str(current_stream.get("subtitle_codec", "") or "").strip().lower()
            logger.info(
                "FFmpeg subtitle stream already present: container=%s codec=%s",
                output_video.suffix.lower() or "<none>",
                current_codec or "unknown",
            )
            meta = {"subtitle_stream_embedded": "true"}
            if current_codec:
                meta["subtitle_stream_codec"] = current_codec
            return meta

        return self._embed_subtitle_stream_if_supported(
            output_video,
            srt_path=srt_path,
            ass_path=ass_path,
        )

    def _build_ffmpeg_cmd(
        self,
        input_video: Path,
        output_video: Path,
        *,
        input_args: Optional[List[str]] = None,
        filter_arg: str,
        video_codec: str,
        audio_codec: Optional[str],
    ) -> List[str]:
        cmd = [
            str(self.ffmpeg_binary),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if input_args:
            cmd.extend(list(input_args))
        cmd.extend(
            [
                "-i",
                str(input_video),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
                "-sn",
                "-dn",
            ]
        )
        if filter_arg:
            cmd.extend(["-vf", filter_arg])
        cmd.extend(["-c:v", video_codec])
        cmd.extend(self._video_encode_args(video_codec))
        if audio_codec:
            cmd.extend(["-c:a", audio_codec])
            cmd.extend(self._audio_encode_args(audio_codec))
        else:
            cmd.append("-an")
        if output_video.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(str(output_video))
        return cmd

    def _run_ffmpeg(self, cmd: List[str], timeout: int = 7200) -> None:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            **windows_hidden_subprocess_kwargs(),
        )
        if result.returncode == 0:
            return
        stderr = (result.stderr or result.stdout or "").strip()
        tail = stderr.splitlines()[-1] if stderr else "unknown ffmpeg error"
        raise RuntimeError(tail)

    @staticmethod
    def _cleanup_file(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    @staticmethod
    def _ensure_output(path: Path, *, label: str) -> None:
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"{label} finished but output video was not generated")

    def _run_ffmpeg_attempt(self, label: str, cmd: List[str], output_video: Path) -> Path:
        self._cleanup_file(output_video)
        logger.info("FFmpeg subtitle attempt [%s] -> %s", label, output_video.name)
        self._run_ffmpeg(cmd)
        self._ensure_output(output_video, label=label)
        return output_video

    def _run_native_burn_attempt(
        self,
        *,
        label: str,
        filter_kind: str,
        input_video: Path,
        subtitle_path: Path,
        output_video: Path,
        source_info: Dict[str, Any],
        video_codec: str,
        audio_codec: Optional[str],
    ) -> Path:
        if not self.media_helper_binary or not self.ffmpeg_binary:
            raise RuntimeError("native media helper is not available")

        args = [
            "burn",
            "--ffmpeg",
            str(self.ffmpeg_binary),
            "--input",
            str(input_video),
            "--output",
            str(output_video),
            "--subtitle-path",
            str(subtitle_path),
            "--filter-kind",
            str(filter_kind),
            "--video-codec",
            str(video_codec),
            "--preset",
            self._normalize_preset(video_codec),
            "--ffmpeg-crf",
            str(self.ffmpeg_crf),
            "--webm-crf",
            str(self.webm_crf),
            "--webm-cpu-used",
            str(self.webm_cpu_used),
            "--webm-audio-bitrate",
            str(self.webm_audio_bitrate),
            "--width",
            str(int(source_info.get("width") or 0)),
            "--height",
            str(int(source_info.get("height") or 0)),
            "--timeout",
            "7200",
        ]
        fonts_dir = self._helper_fonts_dir()
        if fonts_dir:
            args.extend(["--fonts-dir", fonts_dir])
        if output_video.suffix.lower() == ".mp4":
            args.extend(["--movflags-faststart", "1"])
        if audio_codec:
            args.extend(["--audio-codec", str(audio_codec)])
        if filter_kind == "subtitles":
            force_style = self._subtitle_force_style()
            if force_style:
                args.extend(["--force-style", force_style])

        self._cleanup_file(output_video)
        logger.info("Native subtitle attempt [%s] -> %s", label, output_video.name)
        self._run_media_helper(args, timeout=7210)
        self._ensure_output(output_video, label=label)
        return output_video

    def _run_native_mux_attempt(
        self,
        *,
        output_video: Path,
        temp_output: Path,
        subtitle_input: Path,
        subtitle_codec: str,
    ) -> None:
        if not self.media_helper_binary or not self.ffmpeg_binary:
            raise RuntimeError("native media helper is not available")

        self._cleanup_file(temp_output)
        self._run_media_helper(
            [
                "mux",
                "--ffmpeg",
                str(self.ffmpeg_binary),
                "--input-video",
                str(output_video),
                "--subtitle-input",
                str(subtitle_input),
                "--subtitle-codec",
                str(subtitle_codec),
                "--output",
                str(temp_output),
                "--timeout",
                "7200",
            ],
            timeout=7210,
        )
        self._ensure_output(temp_output, label="subtitle_stream_mux")

    def _render_filtered(
        self,
        *,
        input_video: Path,
        srt_path: Path,
        ass_path: Path,
        output_video: Path,
        source_info: Dict[str, Any],
        video_codec: str,
        audio_codec: Optional[str],
        label: str,
    ) -> Path:
        filter_plans = self._ffmpeg_burn_filter_plans(
            srt_path=srt_path,
            ass_path=ass_path,
            source_info=source_info,
        )
        if not filter_plans:
            raise RuntimeError("no subtitle burn filter inputs were generated")
        if self._is_videotoolbox_codec(video_codec) and self._videotoolbox_disabled:
            raise RuntimeError(
                self._videotoolbox_disabled_reason or "VideoToolbox disabled for current run"
            )

        errors: List[str] = []
        input_variants = self._ffmpeg_input_variants(video_codec)
        for filter_name, filter_arg in filter_plans:
            try:
                subtitle_path = srt_path if filter_name == "subtitles" else ass_path
                if self.media_helper_binary and subtitle_path.exists():
                    return self._run_native_burn_attempt(
                        label=f"{label}:{filter_name}",
                        filter_kind=filter_name,
                        input_video=input_video,
                        subtitle_path=subtitle_path,
                        output_video=output_video,
                        source_info=source_info,
                        video_codec=video_codec,
                        audio_codec=audio_codec,
                    )

                for input_label, input_args in input_variants:
                    try:
                        cmd = self._build_ffmpeg_cmd(
                            input_video,
                            output_video,
                            input_args=input_args,
                            filter_arg=filter_arg,
                            video_codec=video_codec,
                            audio_codec=audio_codec,
                        )
                        return self._run_ffmpeg_attempt(
                            f"{label}:{filter_name}:{input_label}",
                            cmd,
                            output_video,
                        )
                    except Exception as exc:
                        if (
                            self._is_videotoolbox_codec(video_codec)
                            and self._is_videotoolbox_init_error(exc)
                        ):
                            self._disable_videotoolbox_for_session(str(exc))
                        errors.append(f"{filter_name}/{input_label}:{exc}")
            except Exception as exc:
                if (
                    self._is_videotoolbox_codec(video_codec)
                    and self._is_videotoolbox_init_error(exc)
                ):
                    self._disable_videotoolbox_for_session(str(exc))
                errors.append(f"{filter_name}:{exc}")
        raise RuntimeError(" | ".join(errors))

    def _transcode_video(
        self,
        *,
        input_video: Path,
        output_video: Path,
        video_codec: str,
        audio_codec: Optional[str],
        label: str,
    ) -> Path:
        cmd = self._build_ffmpeg_cmd(
            input_video,
            output_video,
            filter_arg="",
            video_codec=video_codec,
            audio_codec=audio_codec,
        )
        return self._run_ffmpeg_attempt(label, cmd, output_video)

    @staticmethod
    def _summarize_errors(errors: List[str]) -> str:
        if not errors:
            return "ffmpeg subtitle burn failed"
        tail = errors[-5:]
        return "ffmpeg subtitle burn failed after multiple attempts: " + " | ".join(tail)

    def _render_standard_with_ffmpeg(
        self,
        input_video: Path,
        srt_path: Path,
        ass_path: Path,
        desired_output: Path,
        source_info: Dict[str, Any],
    ) -> Path:
        has_audio = bool(source_info.get("has_audio"))
        target_ext = desired_output.suffix.lower()
        errors: List[str] = []

        audio_modes = self._unique_values(
            [
                self._standard_audio_mode(target_ext, has_audio=has_audio, prefer_copy=True),
                self._standard_audio_mode(target_ext, has_audio=has_audio, prefer_copy=False),
            ]
        )
        if not audio_modes and not has_audio:
            audio_modes = [""]

        for video_codec in self._standard_video_codecs():
            for audio_codec in audio_modes:
                try:
                    return self._render_filtered(
                        input_video=input_video,
                        srt_path=srt_path,
                        ass_path=ass_path,
                        output_video=desired_output,
                        source_info=source_info,
                        video_codec=video_codec,
                        audio_codec=audio_codec or None,
                        label=f"direct:{target_ext}:{video_codec}:{audio_codec or 'noaudio'}",
                    )
                except Exception as exc:
                    errors.append(f"{target_ext}:{video_codec}:{audio_codec or 'noaudio'}:{exc}")

        if target_ext != ".mkv":
            safe_output = desired_output.with_suffix(".mkv")
            logger.warning(
                "FFmpeg burn-in could not keep container %s; fallback to MKV output %s",
                target_ext,
                safe_output.name,
            )
            mkv_audio_modes = self._unique_values(
                [
                    self._standard_audio_mode(".mkv", has_audio=has_audio, prefer_copy=True),
                    self._standard_audio_mode(".mkv", has_audio=has_audio, prefer_copy=False),
                ]
            )
            if not mkv_audio_modes and not has_audio:
                mkv_audio_modes = [""]
            for video_codec in self._standard_video_codecs():
                for audio_codec in mkv_audio_modes:
                    try:
                        return self._render_filtered(
                            input_video=input_video,
                            srt_path=srt_path,
                            ass_path=ass_path,
                            output_video=safe_output,
                            source_info=source_info,
                            video_codec=video_codec,
                            audio_codec=audio_codec or None,
                            label=f"fallback:mkv:{video_codec}:{audio_codec or 'noaudio'}",
                        )
                    except Exception as exc:
                        errors.append(f"mkv:{video_codec}:{audio_codec or 'noaudio'}:{exc}")

        raise RuntimeError(self._summarize_errors(errors))

    def _render_webm_with_ffmpeg(
        self,
        input_video: Path,
        srt_path: Path,
        ass_path: Path,
        desired_output: Path,
        source_info: Dict[str, Any],
    ) -> Path:
        has_audio = bool(source_info.get("has_audio"))
        errors: List[str] = []
        webm_video_codecs = self._webm_video_codecs()
        webm_audio_codecs = self._webm_audio_codecs(has_audio)

        if webm_video_codecs and (webm_audio_codecs or not has_audio):
            audio_modes = webm_audio_codecs if has_audio else [""]
            for video_codec in webm_video_codecs:
                for audio_codec in audio_modes:
                    try:
                        return self._render_filtered(
                            input_video=input_video,
                            srt_path=srt_path,
                            ass_path=ass_path,
                            output_video=desired_output,
                            source_info=source_info,
                            video_codec=video_codec,
                            audio_codec=audio_codec or None,
                            label=f"webm-direct:{video_codec}:{audio_codec or 'noaudio'}",
                        )
                    except Exception as exc:
                        errors.append(f"webm-direct:{video_codec}:{audio_codec or 'noaudio'}:{exc}")
        else:
            if not webm_video_codecs:
                errors.append("webm-direct:no-video-encoder")
            if has_audio and not webm_audio_codecs:
                errors.append("webm-direct:no-audio-encoder")

        stage1_output = desired_output.with_suffix(".burn-stage1.mkv")
        stage1_audio_modes = self._unique_values(
            [
                self._standard_audio_mode(".mkv", has_audio=has_audio, prefer_copy=True),
                self._standard_audio_mode(".mkv", has_audio=has_audio, prefer_copy=False),
            ]
        )
        if not stage1_audio_modes and not has_audio:
            stage1_audio_modes = [""]

        if webm_video_codecs and (webm_audio_codecs or not has_audio):
            for stage1_video_codec in self._standard_video_codecs():
                for stage1_audio_codec in stage1_audio_modes:
                    try:
                        intermediate = self._render_filtered(
                            input_video=input_video,
                            srt_path=srt_path,
                            ass_path=ass_path,
                            output_video=stage1_output,
                            source_info=source_info,
                            video_codec=stage1_video_codec,
                            audio_codec=stage1_audio_codec or None,
                            label=(
                                f"webm-stage1:{stage1_video_codec}:"
                                f"{stage1_audio_codec or 'noaudio'}"
                            ),
                        )
                    except Exception as exc:
                        errors.append(
                            f"webm-stage1:{stage1_video_codec}:{stage1_audio_codec or 'noaudio'}:{exc}"
                        )
                        continue

                    try:
                        audio_modes = webm_audio_codecs if has_audio else [""]
                        for final_video_codec in webm_video_codecs:
                            for final_audio_codec in audio_modes:
                                try:
                                    return self._transcode_video(
                                        input_video=intermediate,
                                        output_video=desired_output,
                                        video_codec=final_video_codec,
                                        audio_codec=final_audio_codec or None,
                                        label=(
                                            f"webm-stage2:{final_video_codec}:"
                                            f"{final_audio_codec or 'noaudio'}"
                                        ),
                                    )
                                except Exception as exc:
                                    errors.append(
                                        f"webm-stage2:{final_video_codec}:"
                                        f"{final_audio_codec or 'noaudio'}:{exc}"
                                    )
                    finally:
                        self._cleanup_file(intermediate)

        safe_output = desired_output.with_suffix(".mkv")
        logger.warning(
            "FFmpeg could not produce a compatible WebM reliably; fallback to MKV output %s",
            safe_output.name,
        )
        try:
            return self._render_standard_with_ffmpeg(
                input_video=input_video,
                srt_path=srt_path,
                ass_path=ass_path,
                desired_output=safe_output,
                source_info=source_info,
            )
        except Exception as exc:
            errors.append(f"webm-safe-mkv:{exc}")
            raise RuntimeError(self._summarize_errors(errors))

    def _render_with_ffmpeg(
        self,
        input_video: Path,
        srt_path: Path,
        ass_path: Path,
        output_video: Path,
    ) -> Path:
        if not self.ffmpeg_binary:
            raise RuntimeError("ffmpeg is not available")

        source_info = self._probe_media_info(input_video)
        if output_video.suffix.lower() == ".webm":
            return self._render_webm_with_ffmpeg(
                input_video=input_video,
                srt_path=srt_path,
                ass_path=ass_path,
                desired_output=output_video,
                source_info=source_info,
            )
        return self._render_standard_with_ffmpeg(
            input_video=input_video,
            srt_path=srt_path,
            ass_path=ass_path,
            desired_output=output_video,
            source_info=source_info,
        )

    def render(
        self,
        source_file: Path,
        segments: Iterable[Any],
        output_dir: Path,
        temp_dir: Optional[Path] = None,
    ) -> Dict[str, str]:
        if not self.is_enabled_for(source_file):
            return {}

        valid_segments = list(self._iter_segments(segments))
        if not valid_segments:
            raise RuntimeError("No valid text segments available for video subtitle burn-in")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        previous_temp_dir = self._active_temp_dir
        self._active_temp_dir = Path(temp_dir) if temp_dir is not None else output_dir / "temp"
        self._active_temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_video, srt_path, ass_path = self._subtitle_paths(source_file, output_dir)

            source_info = self._probe_media_info(source_file)
            self._write_srt(valid_segments, srt_path)
            self._write_ass(
                valid_segments,
                ass_path,
                width=int(source_info.get("width") or 0),
                height=int(source_info.get("height") or 0),
            )

            actual_output = self._render_with_ffmpeg(source_file, srt_path, ass_path, output_video)
            backend = "ffmpeg"

            mux_meta = self.ensure_embedded_subtitle_stream(
                actual_output,
                srt_path=srt_path,
                ass_path=ass_path,
            )

            result = {
                "video_burned": str(actual_output),
                "subtitle_srt": str(srt_path),
                "subtitle_ass": str(ass_path),
                "subtitle_backend": backend,
            }
            result.update(mux_meta)
            return result
        finally:
            self._active_temp_dir = previous_temp_dir
