"""Run an acceptance-only subtitle and real-video delivery transaction.

The tool consumes an existing transcript document whose timestamps are relative
to the supplied short video. It generates SRT, WebVTT, and ASS sidecars, then
uses the production quarantine/publication boundary for soft-mux and burn-in
outputs. Publication is allowed only after representative frames and the
embedded subtitle stream pass deterministic checks.

This is deliberately not a review approval tool. Its manifest always records
that linguistic and speaker quality remain unapproved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.media_probe import MediaProbe
from backend.output_orchestration import MediaProbeArtifact, OutputExecutionPlan
from backend.output_publication import publish_output_plans
from backend.output_recipe import parse_output_recipe
from backend.persistence import (
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from backend.subtitle_delivery import SubtitleDeliveryExecutor
from backend.subtitles import (
    CuePolicy,
    SubtitleFormat,
    SubtitleOutputMode,
    SubtitleStyle,
)


_FORMAT_SUFFIX = {
    SubtitleFormat.SRT: ".srt",
    SubtitleFormat.WEBVTT: ".vtt",
    SubtitleFormat.ASS: ".ass",
}


def _run(command: Sequence[str], *, timeout_seconds: float = 180.0) -> bytes:
    result = subprocess.run(  # noqa: S603 - argv-only local tool boundary
        tuple(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-4_000:]
        raise RuntimeError(
            f"local media command failed with exit {result.returncode}: {detail}"
        )
    return result.stdout


def _ass_filter(path: Path) -> str:
    escaped = str(path).replace("\\", "\\\\").replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    return f"ass=filename='{escaped}'"


def _extract_frame(
    *,
    ffmpeg: Path,
    media: Path,
    timestamp_ms: int,
    target: Path,
    overlay: Path | None = None,
) -> None:
    command = [
        str(ffmpeg),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media),
        # Output-side seeking preserves the original filter timeline. Input-side
        # seeking resets ASS time to zero and can falsely render cue 1 at every
        # representative timestamp.
        "-ss",
        f"{timestamp_ms / 1_000:.3f}",
        "-frames:v",
        "1",
    ]
    if overlay is not None:
        command.extend(("-vf", _ass_filter(overlay)))
    command.extend(("-y", str(target)))
    _run(command, timeout_seconds=90.0)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"representative frame was not created: {target}")


def _frame_metrics(
    base_path: Path,
    expected_path: Path,
    actual_path: Path | None,
) -> dict[str, Any]:
    base = np.asarray(Image.open(base_path).convert("RGB"), dtype=np.int16)
    expected = np.asarray(
        Image.open(expected_path).convert("RGB"), dtype=np.int16
    )
    if base.shape != expected.shape:
        raise RuntimeError("expected ASS overlay changed the frame dimensions")

    delta = np.max(np.abs(expected - base), axis=2)
    mask = delta >= 24
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"passed": False, "reason": "no-overlay-pixels"}

    height, width = mask.shape
    bounds = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]
    safe = (
        bounds[0] >= int(width * 0.02)
        and bounds[2] <= int(width * 0.98)
        and bounds[1] >= int(height * 0.40)
        and bounds[3] <= int(height * 0.98)
    )
    overlay_pixels = int(mask.sum())
    result: dict[str, Any] = {
        "overlayPixelCount": overlay_pixels,
        "overlayPixelRatio": round(float(mask.mean()), 9),
        "overlayBoundsPx": bounds,
        "frameSize": {"width": width, "height": height},
        "insideConservativeSafeArea": safe,
        "expectedOverlayVisible": overlay_pixels >= 300,
    }
    passed = safe and overlay_pixels >= 300

    if actual_path is not None:
        actual = np.asarray(
            Image.open(actual_path).convert("RGB"), dtype=np.int16
        )
        if actual.shape != base.shape:
            raise RuntimeError("burn-in output changed the frame dimensions")
        actual_delta = np.max(np.abs(actual - base), axis=2)
        actual_mask = actual_delta >= 24
        coverage = float(np.logical_and(mask, actual_mask).sum()) / max(
            1, overlay_pixels
        )
        expected_error = float(
            np.mean(np.abs(actual[mask] - expected[mask]))
        )
        source_error = float(np.mean(np.abs(base[mask] - expected[mask])))
        actual_delta_mean = float(np.mean(np.abs(actual[mask] - base[mask])))
        burn_visible = (
            coverage >= 0.70
            and actual_delta_mean >= 12
            and expected_error <= source_error * 1.25
        )
        result.update(
            {
                "burnMaskCoverage": round(coverage, 9),
                "burnRegionDeltaMean": round(actual_delta_mean, 6),
                "burnVsExpectedErrorMean": round(expected_error, 6),
                "sourceVsExpectedErrorMean": round(source_error, 6),
                "burnInVisible": burn_visible,
            }
        )
        passed = passed and burn_visible

    result["passed"] = passed
    return result


def _representative_indexes(cue_count: int) -> tuple[int, ...]:
    if cue_count < 1:
        raise ValueError("cue_count must be positive")
    requested = (
        0,
        cue_count // 4,
        cue_count // 2,
        (3 * cue_count) // 4,
        cue_count - 1,
    )
    return tuple(dict.fromkeys(requested))


def _write_contact_sheet(paths: Sequence[Path], target: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        thumbnail_width = 640
        thumbnails: list[Image.Image] = []
        for image in images:
            height = max(1, round(image.height * thumbnail_width / image.width))
            thumbnails.append(image.resize((thumbnail_width, height)))
        label_height = 34
        canvas = Image.new(
            "RGB",
            (
                thumbnail_width,
                sum(image.height + label_height for image in thumbnails),
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        cursor = 0
        for index, (path, image) in enumerate(zip(paths, thumbnails, strict=True)):
            draw.text((8, cursor + 8), f"{index + 1}. {path.stem}", fill="black")
            cursor += label_height
            canvas.paste(image, (0, cursor))
            cursor += image.height
        canvas.save(target, format="PNG")
    finally:
        for image in images:
            image.close()


def _recipe() -> Any:
    return parse_output_recipe(
        {
            "schemaVersion": "1.0.0",
            "report": {
                "template": "soft-glass",
                "font": "system-sans",
                "customFontFamily": "",
                "pageSize": "a4",
                "density": "balanced",
                "accentColor": "#6959d2",
            },
            "subtitles": {
                "enabled": True,
                "theme": "youtube-clean",
                "size": "medium",
                "safeArea": "broadcast",
                "position": "bottom",
                "speakerPalette": "adaptive-spectrum",
                "backgroundOpacity": 72,
                "maximumLines": 2,
                "avoidVisualCollisions": True,
                "wordProgressHighlight": False,
            },
            "delivery": {
                "formats": ["srt", "webvtt", "ass"],
                "subtitleModes": ["sidecar", "soft-mux", "burn-in"],
                "includeMediaMetadata": True,
                "preserveSourceMedia": True,
                "fileNamePattern": "{sourceStem}-{artifact}",
            },
            "finishing": {
                "includeCover": True,
                "includeChapters": True,
                "includeTimestamps": True,
                "includeHeader": True,
                "includeFooter": True,
                "includeSpeakerIndex": True,
                "includeConfidenceNotes": False,
                "chapterStyle": "semantic",
                "timestampStyle": "segment",
                "customTitle": "",
            },
        }
    )


def _plan(
    mode: SubtitleOutputMode,
    *,
    source: Path,
    output_root: Path,
    probe_artifact: MediaProbeArtifact,
    sidecars: Mapping[SubtitleFormat, Path],
) -> OutputExecutionPlan:
    formats = (
        SubtitleFormat.SRT,
        SubtitleFormat.WEBVTT,
        SubtitleFormat.ASS,
    )
    return OutputExecutionPlan(
        customization_sha256=hashlib.sha256(
            f"real-video-acceptance-{mode.value}".encode("ascii")
        ).hexdigest(),
        source_path=source,
        output_directory=output_root,
        media_probe_artifact=probe_artifact,
        report_enabled=False,
        report_config={},
        exports_config={},
        safety_config={
            "preserveSourceMedia": True,
            "overwriteSourceMedia": False,
        },
        reversibility_config={
            "sourceMediaImmutable": True,
            "derivedArtifactOnly": True,
        },
        subtitle_enabled=True,
        subtitle_config={
            "speakerColors": {
                "mode": "automatic",
                "seed": "real-video-acceptance",
                "algorithm": "oklch-hash-v1",
                "overrides": [],
            }
        },
        subtitle_formats=formats,
        subtitle_style=SubtitleStyle(
            font_family="PingFang SC",
            font_fallbacks=("Arial Unicode MS", "Arial"),
            font_size=58,
            font_weight=600,
            outline_width=3.0,
            shadow_depth=1.2,
            margin_horizontal=90,
            margin_vertical=70,
        ),
        cue_policy=CuePolicy(
            max_characters_per_line=42,
            max_lines=2,
            max_reading_speed=17.0,
            min_cue_ms=900,
            max_cue_ms=7_000,
            gap_ms=80,
            include_speaker_labels=True,
        ),
        subtitle_theme="youtube-clean",
        speaker_color_mode="automatic",
        speaker_color_seed="real-video-acceptance",
        speaker_color_overrides={},
        sidecar_paths=tuple((item, sidecars[item]) for item in formats),
        delivery_mode=mode,
        delivery_output_path=(
            None
            if mode is SubtitleOutputMode.SIDECAR
            else output_root
            / (
                "video-soft-mux.mp4"
                if mode is SubtitleOutputMode.SOFT_MUX
                else "video-burn-in.mp4"
            )
        ),
        subtitle_codec=None,
        burn_in_strategy=(
            "h264-high-quality"
            if mode is SubtitleOutputMode.BURN_IN
            else None
        ),
        visual_qa_required=mode is not SubtitleOutputMode.SIDECAR,
        presentation_limitations=(),
    )


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve(strict=True)
    upstream_source = args.upstream_source.expanduser().resolve(strict=True)
    transcript_path = args.transcript.expanduser().resolve(strict=True)
    review_path = args.review_queue.expanduser().resolve(strict=True)
    ffmpeg = args.ffmpeg.expanduser().resolve(strict=True)
    ffprobe = args.ffprobe.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    if not source.is_file() or not upstream_source.is_file():
        raise ValueError("source and upstream source must be regular files")
    selection_reason = args.selection_reason.strip()
    if not selection_reason:
        raise ValueError("selection reason must not be empty")

    document = json.loads(transcript_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("jobId") != document.get("jobId"):
        raise ValueError("review queue and transcript jobId do not match")
    open_count = review.get("openCount")
    if (
        isinstance(open_count, bool)
        or not isinstance(open_count, int)
        or open_count < 0
    ):
        raise ValueError("review queue openCount must be a non-negative integer")
    if sha256_file(upstream_source) != args.upstream_sha256:
        raise ValueError("upstream source SHA-256 does not match the file")
    expected_window_duration = args.window_end_ms - args.window_start_ms
    transcript_duration = document.get("source", {}).get("durationMs")
    if (
        not isinstance(transcript_duration, int)
        or abs(transcript_duration - expected_window_duration) > 750
    ):
        raise ValueError("window bounds do not match the transcript time base")

    probe_engine = MediaProbe(
        ffprobe_command=(str(ffprobe),),
        ffmpeg_command=(str(ffmpeg),),
    )
    probe = probe_engine.probe(source)
    expected_duration = document.get("source", {}).get("durationMs")
    if (
        not isinstance(expected_duration, int)
        or probe.duration_ms is None
        or abs(probe.duration_ms - expected_duration) > 750
    ):
        raise ValueError(
            "source video duration does not match the transcript time base"
        )

    output_root.mkdir(parents=True)
    probe_path = output_root / "media-probe.v1.json"
    atomic_write_json(probe_path, probe.to_dict())
    probe_artifact = MediaProbeArtifact(
        path=probe_path,
        size_bytes=probe_path.stat().st_size,
        sha256=sha256_file(probe_path),
        source_path=source,
        source_size_bytes=source.stat().st_size,
        source_sha256=sha256_file(source),
        probe_fingerprint_sha256=probe.probe_fingerprint_sha256,
    )
    sidecars = {
        item: output_root / f"subtitles{_FORMAT_SUFFIX[item]}"
        for item in SubtitleFormat
    }
    plans = tuple(
        _plan(
            mode,
            source=source,
            output_root=output_root,
            probe_artifact=probe_artifact,
            sidecars=sidecars,
        )
        for mode in (
            SubtitleOutputMode.SIDECAR,
            SubtitleOutputMode.SOFT_MUX,
            SubtitleOutputMode.BURN_IN,
        )
    )

    def visual_qa_hook(
        *,
        source_path: Path,
        rendered_path: Path,
        arrangement: Any,
        delivery_receipt: Any,
        execution_plan: Any | None = None,
    ) -> dict[str, Any]:
        del execution_plan
        mode = delivery_receipt.mode.value
        qa_root = output_root / "representative-frames" / mode
        qa_root.mkdir(parents=True, exist_ok=True)
        ass_path = sidecars[SubtitleFormat.ASS]
        frame_rows: list[dict[str, Any]] = []
        contact_paths: list[Path] = []
        for index in _representative_indexes(len(arrangement.cues)):
            cue = arrangement.cues[index]
            timestamp_ms = (cue.start_ms + cue.end_ms) // 2
            base_path = qa_root / f"cue-{cue.number:03d}-source.png"
            expected_path = qa_root / f"cue-{cue.number:03d}-expected.png"
            actual_path = qa_root / f"cue-{cue.number:03d}-rendered.png"
            _extract_frame(
                ffmpeg=ffmpeg,
                media=Path(source_path),
                timestamp_ms=timestamp_ms,
                target=base_path,
            )
            _extract_frame(
                ffmpeg=ffmpeg,
                media=Path(source_path),
                timestamp_ms=timestamp_ms,
                target=expected_path,
                overlay=ass_path,
            )
            rendered_frame: Path | None = None
            if mode == SubtitleOutputMode.BURN_IN.value:
                _extract_frame(
                    ffmpeg=ffmpeg,
                    media=Path(rendered_path),
                    timestamp_ms=timestamp_ms,
                    target=actual_path,
                )
                rendered_frame = actual_path
                contact_paths.append(actual_path)
            else:
                contact_paths.append(expected_path)
            frame_rows.append(
                {
                    "cueNumber": cue.number,
                    "timestampMs": timestamp_ms,
                    "speakerId": cue.speaker_id,
                    "sourceFrameSha256": sha256_file(base_path),
                    "expectedAssFrameSha256": sha256_file(expected_path),
                    "renderedFrameSha256": (
                        sha256_file(actual_path)
                        if rendered_frame is not None
                        else None
                    ),
                    "metrics": _frame_metrics(
                        base_path,
                        expected_path,
                        rendered_frame,
                    ),
                }
            )

        contact_sheet = qa_root / "contact-sheet.png"
        _write_contact_sheet(contact_paths, contact_sheet)
        stream_evidence: dict[str, Any] | None = None
        if mode == SubtitleOutputMode.SOFT_MUX.value:
            raw_probe = _run(
                (
                    str(ffprobe),
                    "-v",
                    "error",
                    "-select_streams",
                    "s",
                    "-show_entries",
                    "stream=index,codec_name:stream_tags=language,title",
                    "-of",
                    "json",
                    str(rendered_path),
                )
            )
            stream_evidence = json.loads(raw_probe.decode("utf-8"))
            extracted = qa_root / "embedded-subtitles.srt"
            _run(
                (
                    str(ffmpeg),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(rendered_path),
                    "-map",
                    "0:s:0",
                    "-c:s",
                    "srt",
                    "-y",
                    str(extracted),
                )
            )
            text = extracted.read_text(encoding="utf-8")
            extracted_count = len(re.findall(r"(?m)^\d+\s*$", text))
            stream_evidence.update(
                {
                    "extractedSubtitleSha256": sha256_file(extracted),
                    "extractedCueCount": extracted_count,
                    "expectedCueCount": len(arrangement.cues),
                    "speakerLabelsPresent": "[speaker-" in text,
                }
            )
            stream_evidence["passed"] = (
                bool(stream_evidence.get("streams"))
                and extracted_count == len(arrangement.cues)
                and stream_evidence["speakerLabelsPresent"]
            )

        passed = all(
            bool(item["metrics"]["passed"]) for item in frame_rows
        ) and (
            stream_evidence is None or bool(stream_evidence["passed"])
        )
        evidence = {
            "passed": passed,
            "analysisId": f"real-video-{mode}-representative-frames-v1",
            "method": "source-vs-ass-mask-and-burn-coverage-v1",
            "deliveryMode": mode,
            "renderedArtifactSha256": sha256_file(Path(rendered_path)),
            "canonicalAssSha256": sha256_file(ass_path),
            "contactSheet": {
                "path": str(contact_sheet),
                "sha256": sha256_file(contact_sheet),
            },
            "representativeFrames": frame_rows,
            "subtitleStream": stream_evidence,
            "limitations": [
                "model transcript remains unapproved",
                "pixel-mask QA does not replace human linguistic review",
            ],
        }
        atomic_write_json(
            output_root / f"{mode}-visual-qa.v1.json", evidence
        )
        return evidence

    executor = SubtitleDeliveryExecutor(
        probe=probe_engine,
        ffmpeg_command=(str(ffmpeg),),
    )
    manifest = publish_output_plans(
        _recipe(),
        plans,
        document,
        executor=executor,
        visual_qa_hook=visual_qa_hook,
        subtitle_language=document.get("language") or "und",
        subtitle_title="MediaTranscribeStudio real-video acceptance",
    )
    manifest_payload = manifest.to_dict()
    manifest_path = output_root / "output-publication-manifest.v1.json"
    atomic_write_json(manifest_path, manifest_payload)

    acceptance = {
        "schemaVersion": "1.0.0",
        "artifactType": "unapproved-real-video-delivery-acceptance",
        "applicationPolicy": "suggestion-only",
        "requiresHumanApproval": True,
        "releaseApproved": False,
        "sourceWindow": {
            "upstreamSourcePath": str(upstream_source),
            "upstreamSourceSha256": args.upstream_sha256,
            "startMs": args.window_start_ms,
            "endMs": args.window_end_ms,
            "selectionReason": selection_reason,
            "selectionUsesModelScores": False,
            "derivedSourcePath": str(source),
            "derivedSourceSha256": sha256_file(source),
        },
        "transcript": {
            "path": str(transcript_path),
            "sourceDocumentHash": canonical_json_sha256(document),
            "reviewQueueSha256": sha256_file(review_path),
            "openReviewItems": open_count,
        },
        "publicationManifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "manifestSha256": manifest_payload["manifestSha256"],
        },
        "qualityBoundary": {
            "technicalDeliveryPassed": True,
            "linguisticQualityPassed": False,
            "speakerQualityPassed": False,
            "reason": (
                "OPEN_REVIEW_ITEMS_AND_NO_REFERENCE_TRUTH"
                if open_count
                else "NO_REFERENCE_TRUTH_OR_HUMAN_RELEASE_APPROVAL"
            ),
        },
    }
    acceptance_path = output_root / "unapproved-delivery-acceptance.v1.json"
    atomic_write_json(acceptance_path, acceptance)
    return {
        "manifest": str(manifest_path),
        "manifestSha256": manifest_payload["manifestSha256"],
        "acceptance": str(acceptance_path),
        "customerArtifacts": manifest_payload["customerArtifacts"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--upstream-sha256", required=True, metavar="SHA256")
    parser.add_argument("--window-start-ms", type=int, required=True)
    parser.add_argument("--window-end-ms", type=int, required=True)
    parser.add_argument("--selection-reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{64}", args.upstream_sha256) is None:
        raise SystemExit("--upstream-sha256 must be a lowercase SHA-256")
    if args.window_start_ms < 0 or args.window_end_ms <= args.window_start_ms:
        raise SystemExit("window bounds must be positive and increasing")
    result = run_acceptance(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
