"""Deterministic, immutable transcript exports with atomic publication.

The exporter consumes the current immutable transcript-document v2 contract.
It never mutates that document, never overwrites an existing destination, and
publishes only exact bytes that have been durably written and hash-verified.
HTML output is deliberately valid, self-contained XHTML so the same artifact
can be reviewed in assistive technology or consumed by OpenHTMLtoPDF.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import WorkerError
from .language import normalize_language_tag
from .persistence import canonical_json_bytes, sha256_file


TRANSCRIPT_EXPORT_SCHEMA_VERSION = "2.0.0"
_SEGMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$")
_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SPEAKER_ID = re.compile(r"^speaker-([1-9][0-9]*)$")
_PATH_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_UNSAFE_NAME = re.compile(r'[<>:"|?*]')
_WINDOWS_RESERVED_NAME = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)",
    re.IGNORECASE,
)


class TranscriptExportFormat(str, Enum):
    JSON = "json"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    XHTML = "xhtml"


class TranscriptExportError(WorkerError):
    """Structured failure at the transcript-export trust boundary."""


@dataclass(frozen=True)
class TranscriptExportReceipt:
    """Integrity evidence for one immutably published transcript export."""

    format: str
    path: Path
    sha256: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "path": str(self.path),
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True)
class _SegmentView:
    segment_id: str
    start_ms: int
    end_ms: int
    speaker_id: str
    speaker_label: str
    raw_text: str
    normalized_text: str
    display_text: str


@dataclass(frozen=True)
class _TranscriptView:
    document_id: str
    job_id: str
    title: str | None
    language: str
    source_file_name: str
    duration_ms: int
    segments: tuple[_SegmentView, ...]


def export_transcript(
    document: Mapping[str, Any],
    *,
    export_format: str | TranscriptExportFormat,
    output_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> TranscriptExportReceipt:
    """Render and atomically publish one deterministic transcript artifact.

    ``output_path`` may be absolute or relative to ``output_root``.  Its parent
    directory must already exist.  Both lexical traversal and resolved-path
    escape are rejected, including escape through a linked parent directory.
    Existing destinations are never replaced.
    """

    resolved_format = _resolve_format(export_format)
    snapshot, canonical_document = _snapshot_document(document)
    view = _validate_document(snapshot)
    payload = _render_export(
        resolved_format,
        canonical_document=canonical_document,
        view=view,
    )
    target = _resolve_target(output_root=output_root, output_path=output_path)
    return _publish(
        target,
        payload,
        export_format=resolved_format.value,
    )


def render_transcript_export(
    document: Mapping[str, Any],
    *,
    export_format: str | TranscriptExportFormat,
) -> bytes:
    """Return the exact deterministic bytes without touching the filesystem."""

    resolved_format = _resolve_format(export_format)
    snapshot, canonical_document = _snapshot_document(document)
    view = _validate_document(snapshot)
    return _render_export(
        resolved_format,
        canonical_document=canonical_document,
        view=view,
    )


def _resolve_format(
    value: str | TranscriptExportFormat,
) -> TranscriptExportFormat:
    if isinstance(value, TranscriptExportFormat):
        return value
    if not isinstance(value, str) or _PATH_CONTROL.search(value):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_FORMAT_INVALID",
            "export_format must be one exact supported format identifier",
        )
    try:
        return TranscriptExportFormat(value)
    except ValueError as exc:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_FORMAT_INVALID",
            "export_format must be json, txt, markdown, html, or xhtml",
            details={"format": value},
        ) from exc


def _snapshot_document(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(document, Mapping):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "transcript document must be an object",
        )
    try:
        canonical = canonical_json_bytes(document)
        snapshot = json.loads(canonical.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "transcript document must be strict, finite UTF-8 JSON",
            details={"exceptionType": type(exc).__name__},
        ) from exc
    if not isinstance(snapshot, dict):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "transcript document root must be an object",
        )
    return snapshot, canonical


def _validate_document(document: Mapping[str, Any]) -> _TranscriptView:
    if document.get("schemaVersion") != TRANSCRIPT_EXPORT_SCHEMA_VERSION:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "transcript document schemaVersion must be 2.0.0",
        )

    document_id = _required_text(
        document.get("documentId"),
        "$.documentId",
        pattern=_DOCUMENT_ID,
    )
    job_id = _required_text(
        document.get("jobId"),
        "$.jobId",
        pattern=_DOCUMENT_ID,
    )
    language = _required_text(document.get("language"), "$.language")
    try:
        normalized_language = normalize_language_tag(
            language,
            allow_auto=False,
        )
    except ValueError as exc:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "$.language must be a persisted BCP-47 tag, und, or mul",
        ) from exc
    if normalized_language != language:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "$.language must already be in canonical persisted form",
        )

    title: str | None = None
    if "title" in document:
        title = _required_text(document.get("title"), "$.title")

    source = _required_mapping(document.get("source"), "$.source")
    source_file_name = _required_text(
        source.get("fileName"),
        "$.source.fileName",
    )
    duration_ms = _required_integer(
        source.get("durationMs"),
        "$.source.durationMs",
        minimum=1,
    )

    policy = _required_mapping(
        document.get("speakerPolicy"),
        "$.speakerPolicy",
    )
    speaker_count = _required_integer(
        policy.get("resolvedCount"),
        "$.speakerPolicy.resolvedCount",
        minimum=1,
    )
    expected_speaker_ids = tuple(
        f"speaker-{index}" for index in range(1, speaker_count + 1)
    )
    raw_policy_ids = policy.get("speakerIds")
    if (
        not isinstance(raw_policy_ids, list)
        or tuple(raw_policy_ids) != expected_speaker_ids
    ):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "$.speakerPolicy.speakerIds must be the ordered canonical set",
        )
    if policy.get("requireExactSet") is not True:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "$.speakerPolicy.requireExactSet must be true",
        )

    raw_speakers = document.get("speakers")
    if not isinstance(raw_speakers, list) or len(raw_speakers) != speaker_count:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "$.speakers must contain every resolved speaker exactly once",
        )
    speaker_labels: dict[str, str] = {}
    actual_speaker_ids: list[str] = []
    for index, item in enumerate(raw_speakers):
        speaker = _required_mapping(item, f"$.speakers[{index}]")
        speaker_id = _required_text(
            speaker.get("id"),
            f"$.speakers[{index}].id",
            pattern=_SPEAKER_ID,
        )
        actual_speaker_ids.append(speaker_id)
        raw_label = speaker.get("role")
        if raw_label is None:
            raw_label = speaker.get("displayName")
        speaker_labels[speaker_id] = (
            _required_text(
                raw_label,
                f"$.speakers[{index}].role",
            )
            if raw_label is not None
            else speaker_id
        )
    if tuple(actual_speaker_ids) != expected_speaker_ids:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "$.speakers must preserve canonical speaker order",
        )

    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "$.segments must be a non-empty array",
        )

    segments: list[_SegmentView] = []
    seen_segment_ids: set[str] = set()
    observed_speakers: set[str] = set()
    previous_start = -1
    for index, item in enumerate(raw_segments):
        path = f"$.segments[{index}]"
        segment = _required_mapping(item, path)
        segment_id = _required_text(
            segment.get("id"),
            f"{path}.id",
            pattern=_SEGMENT_ID,
        )
        if segment_id in seen_segment_ids:
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
                f"{path}.id duplicates {segment_id!r}",
            )
        seen_segment_ids.add(segment_id)
        start_ms = _required_integer(
            segment.get("startMs"),
            f"{path}.startMs",
            minimum=0,
        )
        end_ms = _required_integer(
            segment.get("endMs"),
            f"{path}.endMs",
            minimum=1,
        )
        if start_ms < previous_start:
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
                f"{path}.startMs is not monotonic",
            )
        if end_ms <= start_ms or end_ms > duration_ms:
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
                f"{path} has an invalid or out-of-range timestamp",
            )
        previous_start = start_ms

        speaker_id = _required_text(
            segment.get("speakerId"),
            f"{path}.speakerId",
            pattern=_SPEAKER_ID,
        )
        if speaker_id not in speaker_labels:
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
                f"{path}.speakerId is outside the resolved speaker set",
            )
        observed_speakers.add(speaker_id)

        raw_text = _required_text(
            segment.get("rawText"),
            f"{path}.rawText",
        )
        normalized_text = _required_text(
            segment.get("normalizedText"),
            f"{path}.normalizedText",
        )
        display_text = _required_text(
            segment.get("displayText"),
            f"{path}.displayText",
        )
        segments.append(
            _SegmentView(
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_id=speaker_id,
                speaker_label=speaker_labels[speaker_id],
                raw_text=raw_text,
                normalized_text=normalized_text,
                display_text=display_text,
            )
        )

    if observed_speakers != set(expected_speaker_ids):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "$.segments must observe every resolved canonical speaker",
        )

    return _TranscriptView(
        document_id=document_id,
        job_id=job_id,
        title=title,
        language=language,
        source_file_name=source_file_name,
        duration_ms=duration_ms,
        segments=tuple(segments),
    )


def _required_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            f"{path} must be an object",
        )
    return value


def _required_integer(
    value: Any,
    path: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            f"{path} must be an integer greater than or equal to {minimum}",
        )
    return value


def _required_text(
    value: Any,
    path: str,
    *,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            f"{path} must be non-empty text",
        )
    if pattern is not None and not pattern.fullmatch(value):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            f"{path} has an invalid identifier",
        )
    if not all(_is_xml_character(character) for character in value):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            f"{path} contains a character forbidden by XML 1.0",
        )
    return value


def _is_xml_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        codepoint in {0x09, 0x0A, 0x0D}
        or 0x20 <= codepoint <= 0xD7FF
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )


def _render_export(
    export_format: TranscriptExportFormat,
    *,
    canonical_document: bytes,
    view: _TranscriptView,
) -> bytes:
    if export_format is TranscriptExportFormat.JSON:
        return canonical_document
    if export_format is TranscriptExportFormat.TXT:
        text = _render_txt(view)
    elif export_format is TranscriptExportFormat.MARKDOWN:
        text = _render_markdown(view)
    else:
        text = _render_xhtml(view)
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
            "transcript export could not be encoded as strict UTF-8",
        ) from exc


def _render_txt(view: _TranscriptView) -> str:
    lines = [
        "MediaTranscribeStudio Transcript",
        "================================",
        f"Document ID: {view.document_id}",
        f"Job ID: {view.job_id}",
        f"Language: {view.language}",
        f"Source: {view.source_file_name}",
        f"Duration ms: {view.duration_ms}",
        f"Segment count: {len(view.segments)}",
    ]
    if view.title is not None:
        lines.extend(("Title:", view.title))
    for index, segment in enumerate(view.segments, start=1):
        lines.extend(
            (
                "",
                f"Segment {index}",
                f"ID: {segment.segment_id}",
                f"Speaker ID: {segment.speaker_id}",
                f"Speaker label: {segment.speaker_label}",
                f"Start ms: {segment.start_ms}",
                f"End ms: {segment.end_ms}",
                (
                    "Timestamp: "
                    f"{_format_timestamp(segment.start_ms)} --> "
                    f"{_format_timestamp(segment.end_ms)}"
                ),
                "Raw text:",
                segment.raw_text,
                "Normalized text:",
                segment.normalized_text,
                "Display text:",
                segment.display_text,
            )
        )
    return "\n".join(lines) + "\n"


def _render_markdown(view: _TranscriptView) -> str:
    lines = [
        "# MediaTranscribeStudio Transcript",
        "",
        f"- **Document ID:** `{view.document_id}`",
        f"- **Job ID:** `{view.job_id}`",
        f"- **Language:** `{view.language}`",
        f"- **Duration ms:** `{view.duration_ms}`",
        f"- **Segment count:** `{len(view.segments)}`",
        "- **Source file:**",
        _markdown_fence(view.source_file_name),
    ]
    if view.title is not None:
        lines.extend(("- **Title:**", _markdown_fence(view.title)))
    for index, segment in enumerate(view.segments, start=1):
        lines.extend(
            (
                "",
                f"## Segment {index}",
                "",
                f"- **Segment ID:** `{segment.segment_id}`",
                f"- **Speaker ID:** `{segment.speaker_id}`",
                f"- **Start ms:** `{segment.start_ms}`",
                f"- **End ms:** `{segment.end_ms}`",
                (
                    "- **Timestamp:** "
                    f"`{_format_timestamp(segment.start_ms)} --> "
                    f"{_format_timestamp(segment.end_ms)}`"
                ),
                "- **Speaker label:**",
                _markdown_fence(segment.speaker_label),
                "",
                "### Raw text",
                "",
                _markdown_fence(segment.raw_text),
                "",
                "### Normalized text",
                "",
                _markdown_fence(segment.normalized_text),
                "",
                "### Display text",
                "",
                _markdown_fence(segment.display_text),
            )
        )
    return "\n".join(lines) + "\n"


def _markdown_fence(value: str) -> str:
    longest = max(
        (len(match.group(0)) for match in re.finditer(r"`+", value)),
        default=0,
    )
    fence = "`" * max(3, longest + 1)
    trailing = "" if value.endswith("\n") else "\n"
    return f"{fence}text\n{value}{trailing}{fence}"


_XHTML_STYLE = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm;
}
html {
  color: #172033;
  background: #ffffff;
  font-family: "Noto Sans", "Noto Sans CJK SC", "Segoe UI", sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
}
body {
  margin: 0;
}
header, main, section, article {
  display: block;
}
header.document-header {
  border-bottom: 2px solid #8b5cf6;
  margin-bottom: 18pt;
  padding-bottom: 10pt;
}
h1, h2, h3, h4 {
  color: #2f2455;
  line-height: 1.25;
  page-break-after: avoid;
}
h1 {
  font-size: 22pt;
  margin: 0 0 8pt;
}
h2 {
  font-size: 16pt;
  margin: 0 0 10pt;
}
h3 {
  font-size: 12.5pt;
  margin: 0 0 4pt;
}
h4 {
  font-size: 9.5pt;
  letter-spacing: 0.02em;
  margin: 8pt 0 3pt;
}
dl.metadata {
  margin: 0;
}
dl.metadata dt {
  color: #655b7f;
  float: left;
  font-weight: 700;
  margin-right: 6pt;
}
dl.metadata dd {
  margin: 0 0 3pt 92pt;
}
ol.segments {
  list-style: none;
  margin: 0;
  padding: 0;
}
li.segment-item {
  margin: 0 0 12pt;
  page-break-inside: avoid;
}
article.segment {
  background: #f8f7ff;
  border: 1px solid #ddd7f7;
  border-left: 4px solid #8b5cf6;
  border-radius: 8px;
  padding: 10pt 12pt;
}
p.timestamp {
  color: #5d5474;
  font-family: "JetBrains Mono", "Consolas", monospace;
  font-size: 8.8pt;
  margin: 0 0 6pt;
}
dl.segment-metadata {
  font-size: 8.8pt;
  margin: 0 0 7pt;
}
dl.segment-metadata dt {
  display: inline;
  font-weight: 700;
}
dl.segment-metadata dd {
  display: inline;
  margin: 0 12pt 0 3pt;
}
pre.transcript-text {
  background: #ffffff;
  border: 1px solid #e7e3f5;
  border-radius: 6px;
  font-family: "Noto Sans", "Noto Sans CJK SC", "Segoe UI", sans-serif;
  margin: 0;
  overflow-wrap: break-word;
  padding: 7pt 8pt;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.raw-text {
  color: #4b5563;
}
.display-text {
  color: #111827;
}
""".strip()


def _render_xhtml(view: _TranscriptView) -> str:
    title = view.title or f"Transcript — {view.document_id}"
    document_heading = _xml_text(title)
    language = _xml_attribute(view.language)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE html>",
        (
            '<html xmlns="http://www.w3.org/1999/xhtml" '
            f"lang={language} xml:lang={language}>"
        ),
        "<head>",
        '<meta charset="UTF-8" />',
        '<meta name="generator" content="MediaTranscribeStudio" />',
        f"<title>{document_heading}</title>",
        f'<style type="text/css">{_XHTML_STYLE}</style>',
        "</head>",
        '<body role="document">',
        '<header class="document-header">',
        f"<h1>{document_heading}</h1>",
        '<dl class="metadata">',
        f"<dt>Document ID</dt><dd>{_xml_text(view.document_id)}</dd>",
        f"<dt>Job ID</dt><dd>{_xml_text(view.job_id)}</dd>",
        f"<dt>Language</dt><dd>{_xml_text(view.language)}</dd>",
        f"<dt>Source</dt><dd>{_xml_text(view.source_file_name)}</dd>",
        f"<dt>Duration ms</dt><dd>{view.duration_ms}</dd>",
        f"<dt>Segment count</dt><dd>{len(view.segments)}</dd>",
        "</dl>",
        "</header>",
        '<main id="main-content">',
        '<section aria-labelledby="transcript-heading">',
        '<h2 id="transcript-heading">Transcript</h2>',
        '<ol class="segments">',
    ]
    for index, segment in enumerate(view.segments, start=1):
        heading_id = f"segment-{index}-heading"
        parts.extend(
            (
                '<li class="segment-item">',
                (
                    '<article class="segment" '
                    f'aria-labelledby="{heading_id}" '
                    f"data-segment-id={_xml_attribute(segment.segment_id)} "
                    f"data-speaker-id={_xml_attribute(segment.speaker_id)} "
                    f'data-start-ms="{segment.start_ms}" '
                    f'data-end-ms="{segment.end_ms}">'
                ),
                (
                    f'<h3 id="{heading_id}">'
                    f"{_xml_text(segment.speaker_label)} "
                    f"<small>({_xml_text(segment.speaker_id)})</small>"
                    "</h3>"
                ),
                '<p class="timestamp">',
                (
                    f'<time datetime="{_iso_duration(segment.start_ms)}">'
                    f"{_format_timestamp(segment.start_ms)}</time>"
                    " &#8594; "
                    f'<time datetime="{_iso_duration(segment.end_ms)}">'
                    f"{_format_timestamp(segment.end_ms)}</time>"
                ),
                "</p>",
                '<dl class="segment-metadata">',
                f"<dt>Segment ID</dt><dd>{_xml_text(segment.segment_id)}</dd>",
                f"<dt>Start ms</dt><dd>{segment.start_ms}</dd>",
                f"<dt>End ms</dt><dd>{segment.end_ms}</dd>",
                "</dl>",
                "<section>",
                "<h4>Raw text</h4>",
                (
                    '<pre class="transcript-text raw-text" dir="auto">'
                    f"{_xml_text(segment.raw_text)}</pre>"
                ),
                "</section>",
                "<section>",
                "<h4>Normalized text</h4>",
                (
                    '<pre class="transcript-text normalized-text" dir="auto">'
                    f"{_xml_text(segment.normalized_text)}</pre>"
                ),
                "</section>",
                "<section>",
                "<h4>Display text</h4>",
                (
                    '<pre class="transcript-text display-text" dir="auto">'
                    f"{_xml_text(segment.display_text)}</pre>"
                ),
                "</section>",
                "</article>",
                "</li>",
            )
        )
    parts.extend(
        (
            "</ol>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        )
    )
    return "\n".join(parts) + "\n"


def _xml_text(value: str) -> str:
    return html.escape(value, quote=True)


def _xml_attribute(value: str) -> str:
    return f'"{html.escape(value, quote=True)}"'


def _format_timestamp(milliseconds: int) -> str:
    total_seconds, millis = divmod(milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _iso_duration(milliseconds: int) -> str:
    seconds, millis = divmod(milliseconds, 1000)
    return f"PT{seconds}.{millis:03d}S" if millis else f"PT{seconds}S"


def _path_text(value: str | os.PathLike[str], field_name: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_INVALID",
            f"{field_name} must be a path string",
        )
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip():
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_INVALID",
            f"{field_name} must not be empty",
        )
    if _PATH_CONTROL.search(raw):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_INVALID",
            f"{field_name} contains forbidden control characters",
        )
    return raw


def _resolve_target(
    *,
    output_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> Path:
    root_text = _path_text(output_root, "output_root")
    target_text = _path_text(output_path, "output_path")
    try:
        root = Path(root_text).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_INVALID",
            "output_root must be an existing directory",
        ) from exc
    if not root.is_dir():
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_INVALID",
            "output_root must be an existing directory",
        )

    requested = Path(target_text).expanduser()
    if ".." in requested.parts:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_OUTSIDE_ROOT",
            "output_path must not contain parent traversal",
        )
    lexical = requested if requested.is_absolute() else root / requested
    if not lexical.name or lexical.name in {".", ".."}:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_INVALID",
            "output_path must name a file",
        )
    if (
        _WINDOWS_UNSAFE_NAME.search(lexical.name)
        or lexical.name.endswith((" ", "."))
        or _WINDOWS_RESERVED_NAME.match(lexical.name)
    ):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_INVALID",
            "output_path contains an unsafe Windows file name",
        )
    try:
        parent = lexical.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_INVALID",
            "output_path parent must be an existing directory",
        ) from exc
    if not parent.is_dir():
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_INVALID",
            "output_path parent must be an existing directory",
        )
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_PATH_OUTSIDE_ROOT",
            "output_path must remain inside output_root",
            details={"path": str(lexical), "outputRoot": str(root)},
        ) from exc

    target = parent / lexical.name
    if os.path.lexists(target):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_OUTPUT_EXISTS",
            "output_path already exists and will not be overwritten",
            details={"path": str(target)},
        )
    return target


def _publish(
    target: Path,
    payload: bytes,
    *,
    export_format: str,
) -> TranscriptExportReceipt:
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    expected_size = len(payload)
    temporary = _write_temporary(target.parent, payload)
    published = False
    try:
        if (
            temporary.stat().st_size != expected_size
            or sha256_file(temporary) != expected_sha256
        ):
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_INTEGRITY_FAILED",
                "temporary transcript export failed integrity verification",
            )
        try:
            os.link(temporary, target, follow_symlinks=False)
            published = True
        except FileExistsError as exc:
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_OUTPUT_EXISTS",
                "output_path appeared concurrently; no file was overwritten",
                details={"path": str(target)},
            ) from exc
        except OSError as exc:
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_PUBLISH_FAILED",
                "atomic no-replace publication is unavailable",
                details={"exceptionType": type(exc).__name__},
            ) from exc

        try:
            temporary.unlink()
        except OSError as exc:
            rollback_error: str | None = None
            try:
                target.unlink()
                published = False
            except OSError as rollback_exc:
                rollback_error = type(rollback_exc).__name__
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_CLEANUP_FAILED",
                "published output was rolled back because temporary cleanup failed",
                details={"rollbackError": rollback_error},
            ) from exc
        try:
            _sync_directory(target.parent)
        except OSError as exc:
            rollback_error: str | None = None
            try:
                target.unlink()
                published = False
            except OSError as rollback_exc:
                rollback_error = type(rollback_exc).__name__
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_DURABILITY_FAILED",
                "published output was rolled back after directory sync failed",
                details={"rollbackError": rollback_error},
            ) from exc
    finally:
        if os.path.lexists(temporary):
            try:
                temporary.unlink()
            except OSError:
                pass

    try:
        resolved = target.resolve(strict=True)
        size = resolved.stat().st_size
        digest = sha256_file(resolved)
    except OSError as exc:
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_INTEGRITY_FAILED",
            "published transcript export could not be verified",
            details={"path": str(target)},
        ) from exc
    if (
        not published
        or not resolved.is_file()
        or size != expected_size
        or digest != expected_sha256
    ):
        raise TranscriptExportError(
            "TRANSCRIPT_EXPORT_INTEGRITY_FAILED",
            "published transcript export differs from rendered bytes",
            details={"path": str(resolved)},
        )
    return TranscriptExportReceipt(
        format=export_format,
        path=resolved,
        sha256=digest,
        size=size,
    )


def _write_temporary(parent: Path, payload: bytes) -> Path:
    for _ in range(32):
        temporary = parent / f".mts-export-{uuid.uuid4().hex}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            return temporary
        except FileExistsError:
            continue
        except OSError as exc:
            if os.path.lexists(temporary):
                try:
                    temporary.unlink()
                except OSError:
                    pass
            raise TranscriptExportError(
                "TRANSCRIPT_EXPORT_WRITE_FAILED",
                "temporary transcript export could not be written",
                details={"exceptionType": type(exc).__name__},
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
    raise TranscriptExportError(
        "TRANSCRIPT_EXPORT_WRITE_FAILED",
        "could not allocate a unique temporary transcript export",
    )


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "TRANSCRIPT_EXPORT_SCHEMA_VERSION",
    "TranscriptExportError",
    "TranscriptExportFormat",
    "TranscriptExportReceipt",
    "export_transcript",
    "render_transcript_export",
]
