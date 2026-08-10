"""Production adapter for deterministic rendered-subtitle visual evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .media_probe import MediaProbe, MediaProbeResult
from .output_orchestration import OutputExecutionPlan
from .persistence import canonical_json_sha256
from .subtitle_visual_evidence import (
    SUBTITLE_RENDER_EVIDENCE_REQUEST_KIND,
    SUBTITLE_RENDER_EVIDENCE_SCHEMA_VERSION,
    SubtitleVisualEvidenceCollector,
    default_subtitle_visual_evidence_sampling,
)
from .subtitle_visual_qa import (
    default_subtitle_visual_qa_policy,
    evaluate_subtitle_visual_qa,
)
from .subtitles import SubtitleArrangement, SubtitleOutputMode, SubtitleStyle
from .windows_font_evidence import WindowsFontEvidenceProvider


class ProductionSubtitleVisualQAHook:
    """Collect real frame pixels and return fail-closed visual-QA evidence."""

    def __init__(
        self,
        *,
        ffmpeg_path: str | Path,
        ffprobe_path: str | Path,
        probe: MediaProbe | None = None,
        collector: Any | None = None,
        evaluator: Callable[[Mapping[str, Any]], Any] = (
            evaluate_subtitle_visual_qa
        ),
    ) -> None:
        self.probe = probe or MediaProbe(
            ffprobe_command=(str(ffprobe_path),),
            ffmpeg_command=(str(ffmpeg_path),),
        )
        self.collector = collector or SubtitleVisualEvidenceCollector(
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            font_evidence_provider=WindowsFontEvidenceProvider(),
        )
        self.evaluator = evaluator

    def __call__(
        self,
        *,
        source_path: Path,
        rendered_path: Path,
        arrangement: SubtitleArrangement,
        delivery_receipt: Any,
        execution_plan: OutputExecutionPlan | None = None,
    ) -> dict[str, Any]:
        if not isinstance(arrangement, SubtitleArrangement):
            raise TypeError("visual QA requires a SubtitleArrangement")
        if not isinstance(execution_plan, OutputExecutionPlan):
            raise TypeError("visual QA requires the resolved output execution plan")
        if not arrangement.cues:
            raise ValueError("visual QA requires at least one subtitle cue")
        style = execution_plan.subtitle_style
        if not isinstance(style, SubtitleStyle):
            raise ValueError("visual QA requires a resolved subtitle style")

        source = Path(source_path).resolve(strict=True)
        rendered = Path(rendered_path).resolve(strict=True)
        receipt = self._receipt_mapping(delivery_receipt)
        try:
            mode = SubtitleOutputMode(str(receipt["mode"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("visual QA received an invalid delivery mode") from exc
        if mode not in {SubtitleOutputMode.SOFT_MUX, SubtitleOutputMode.BURN_IN}:
            raise ValueError("visual QA only supports soft-mux and burn-in")

        probe = self.probe.probe(source)
        width, height = self._video_dimensions(probe)
        speaker_rows, style_by_speaker = self._speaker_rows(
            arrangement,
            style=style,
        )
        policy = self._policy(execution_plan, has_speaker_colors=bool(
            arrangement.speaker_colors
        ))
        request = {
            "kind": SUBTITLE_RENDER_EVIDENCE_REQUEST_KIND,
            "schemaVersion": SUBTITLE_RENDER_EVIDENCE_SCHEMA_VERSION,
            "collectionId": (
                f"production-{mode.value}-"
                f"{execution_plan.customization_sha256[:16]}"
            ),
            "sourceMediaPath": str(source),
            "renderedMediaPath": str(rendered),
            "renderArtifact": {
                "renderer": "ffmpeg-libass",
                "rendererVersion": "locally-hash-bound-ffmpeg",
                "renderConfigurationSha256": (
                    execution_plan.deterministic_hash()
                ),
            },
            "policy": policy,
            "sampling": default_subtitle_visual_evidence_sampling(),
            "speakers": speaker_rows,
            "cues": [
                self._cue_row(
                    cue,
                    index=index,
                    width=width,
                    height=height,
                    style=style,
                    style_by_speaker=style_by_speaker,
                )
                for index, cue in enumerate(arrangement.cues, start=1)
            ],
        }

        collect_kwargs: dict[str, Any] = {"delivery_receipt": delivery_receipt}
        subtitle_path = Path(str(receipt.get("subtitlePath", "")))
        subtitle = subtitle_path.resolve(strict=True)
        if subtitle.suffix.casefold() != ".ass":
            raise ValueError(
                "visual QA requires the delivery receipt's canonical ASS carrier"
            )
        collect_kwargs.update(
            canonical_ass_overlay_path=subtitle,
            canonical_ass_overlay_sha256=self._file_sha256(subtitle),
        )
        collected = self.collector.collect(request, **collect_kwargs)
        qa_result = self.evaluator(collected.qa_request)
        qa_payload = self._result_mapping(qa_result)
        evidence_payload = collected.to_dict()
        return {
            **qa_payload,
            "deliveryMode": mode.value,
            "executionPlanSha256": execution_plan.deterministic_hash(),
            "renderEvidence": {
                "artifactType": "subtitle-render-evidence-result",
                "evidenceArtifactSha256": evidence_payload.get(
                    "evidenceArtifactSha256"
                ),
                "qaRequestSha256": evidence_payload.get("qaRequestSha256"),
                "selectionArtifactSha256": evidence_payload.get(
                    "selection", {}
                ).get("selectionArtifactSha256"),
                "sourceMediaSha256": evidence_payload.get(
                    "sourceMedia", {}
                ).get("sha256"),
                "renderArtifactSha256": evidence_payload.get(
                    "renderArtifact", {}
                ).get("sha256"),
            },
        }

    @staticmethod
    def _receipt_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("visual QA requires a structured delivery receipt")
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("delivery receipt serialization must be an object")
        return dict(payload)

    @staticmethod
    def _result_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            payload = dict(value)
        else:
            to_dict = getattr(value, "to_dict", None)
            if not callable(to_dict):
                raise TypeError("visual QA evaluator returned no object")
            payload = to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("visual QA evaluator serialization must be an object")
        result = dict(payload)
        canonical_json_sha256(result)
        return result

    @staticmethod
    def _video_dimensions(probe: MediaProbeResult) -> tuple[int, int]:
        streams = [
            stream
            for stream in probe.streams
            if stream.index in probe.video_stream_indexes
            and not stream.attached_picture
        ]
        if not streams or not streams[0].width or not streams[0].height:
            raise ValueError("visual QA requires a dimensioned video stream")
        return streams[0].width, streams[0].height

    @staticmethod
    def _speaker_id(cue: Any) -> str:
        return str(cue.speaker_id or cue.speaker or "speaker-unassigned")

    @classmethod
    def _speaker_rows(
        cls,
        arrangement: SubtitleArrangement,
        *,
        style: SubtitleStyle,
    ) -> tuple[list[dict[str, str]], dict[str, str]]:
        assignments = {
            item.speaker_id: item for item in arrangement.speaker_colors
        }
        speaker_ids = sorted({cls._speaker_id(cue) for cue in arrangement.cues})
        rows: list[dict[str, str]] = []
        style_by_speaker: dict[str, str] = {}
        for speaker_id in speaker_ids:
            assignment = assignments.get(speaker_id)
            rows.append(
                {
                    "speakerId": speaker_id,
                    "color": (
                        assignment.color if assignment else style.primary_color
                    ).upper(),
                }
            )
            style_by_speaker[speaker_id] = (
                assignment.style_name if assignment else "Default"
            )
        return rows, style_by_speaker

    @staticmethod
    def _policy(
        plan: OutputExecutionPlan,
        *,
        has_speaker_colors: bool,
    ) -> dict[str, Any]:
        policy = default_subtitle_visual_qa_policy()
        cue_policy = plan.cue_policy
        assert cue_policy is not None
        policy["layout"].update(
            maxLines=cue_policy.max_lines,
            maxCharactersPerLine=cue_policy.max_characters_per_line,
            maxReadingSpeed=cue_policy.max_reading_speed,
        )
        policy["contrast"]["minimumRatio"] = 3.0
        policy["contrast"]["requiredBackgroundClasses"] = (
            ProductionSubtitleVisualQAHook._effective_background_classes(
                plan.subtitle_style,
                dark_maximum=policy["contrast"]["darkMaximumLuminance"],
                light_minimum=policy["contrast"]["lightMinimumLuminance"],
            )
        )
        if not has_speaker_colors:
            policy["speakerColor"]["minimumDeltaE2000"] = 0.0
        return policy

    @staticmethod
    def _effective_background_classes(
        style: SubtitleStyle,
        *,
        dark_maximum: float,
        light_minimum: float,
    ) -> list[str]:
        """Derive classes from the rendered box, not from the scene plate."""

        if style.background_opacity <= 0.0:
            return ["dark", "light"]
        color = style.background_color.lstrip("#")
        if len(color) != 6:
            return ["dark", "light"]
        channels = tuple(
            int(color[offset : offset + 2], 16) for offset in (0, 2, 4)
        )
        opacity = style.background_opacity
        composites = []
        for plate in (0, 255):
            composites.append(
                tuple(
                    round(channel * opacity + plate * (1.0 - opacity))
                    for channel in channels
                )
            )
        luminances = [
            ProductionSubtitleVisualQAHook._relative_luminance(color)
            for color in composites
        ]
        if max(luminances) <= dark_maximum:
            return ["dark"]
        if min(luminances) >= light_minimum:
            return ["light"]
        return ["dark", "light"]

    @staticmethod
    def _relative_luminance(color: tuple[int, int, int]) -> float:
        channels = []
        for value in color:
            component = value / 255.0
            channels.append(
                component / 12.92
                if component <= 0.04045
                else ((component + 0.055) / 1.055) ** 2.4
            )
        return (
            0.2126 * channels[0]
            + 0.7152 * channels[1]
            + 0.0722 * channels[2]
        )

    @classmethod
    def _cue_row(
        cls,
        cue: Any,
        *,
        index: int,
        width: int,
        height: int,
        style: SubtitleStyle,
        style_by_speaker: Mapping[str, str],
    ) -> dict[str, Any]:
        speaker_id = cls._speaker_id(cue)
        lines = cue.text.splitlines() or [cue.text]
        return {
            "cueId": f"cue-{index:06d}",
            "startMs": cue.start_ms,
            "endMs": cue.end_ms,
            "text": cue.text,
            "speakerId": speaker_id,
            "styleId": style_by_speaker[speaker_id],
            "renderedLines": lines,
            "karaokeMode": "none",
            "wordTimingEvidence": None,
            "bounds": cls._alignment_bounds(
                width,
                height,
                alignment=style.alignment,
            ),
            "requestedFontFamilies": list(
                dict.fromkeys((style.font_family, *style.font_fallbacks))
            ),
        }

    @staticmethod
    def _alignment_bounds(
        width: int,
        height: int,
        *,
        alignment: int,
    ) -> dict[str, int]:
        x = max(1, round(width * 0.02))
        box_width = max(1, width - 2 * x)
        if alignment in {7, 8, 9}:
            y = max(1, round(height * 0.01))
            bottom = max(y + 1, round(height * 0.60))
        elif alignment in {4, 5, 6}:
            y = max(1, round(height * 0.20))
            bottom = max(y + 1, round(height * 0.80))
        else:
            y = max(1, round(height * 0.40))
            bottom = max(y + 1, height - 1)
        return {
            "x": x,
            "y": y,
            "width": box_width,
            "height": min(height, bottom) - y,
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = ["ProductionSubtitleVisualQAHook"]
