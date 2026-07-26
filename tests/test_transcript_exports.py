from __future__ import annotations

import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

import pytest

import backend.transcript_exports as transcript_exports
from backend.transcript_exports import (
    TranscriptExportError,
    TranscriptExportFormat,
    export_transcript,
    render_transcript_export,
)


def _document() -> dict[str, Any]:
    return {
        "schemaVersion": "2.0.0",
        "documentId": "doc-export-fixture-001",
        "jobId": "job-export-fixture-001",
        "generatedAt": "2026-07-23T12:00:00Z",
        "title": '评审 <Alpha> & "Beta" — Café ☕',
        "language": "zh-Hans",
        "source": {
            "fileName": "会议 & review.m4a",
            "sha256": "a" * 64,
            "durationMs": 90_000_000,
        },
        "speakerPolicy": {
            "mode": "auto",
            "resolvedCount": 2,
            "speakerIds": ["speaker-1", "speaker-2"],
            "requireExactSet": True,
            "unknownSpeakerAllowed": False,
            "speakerChangeRequiresEvidence": True,
            "estimate": {
                "estimatedCount": 2,
                "confidence": 0.97,
                "candidateRange": {"min": 2, "max": 2},
                "method": "dynamic-n",
            },
        },
        "speakers": [
            {
                "id": "speaker-1",
                "role": '主持人 "A" & <lead>',
            },
            {
                "id": "speaker-2",
                "role": "ضيف / Guest 🌍",
            },
        ],
        "segments": [
            {
                "id": "segment-001",
                "startMs": 1_234,
                "endMs": 3_456,
                "speakerId": "speaker-1",
                "rawText": "原始 <script>alert('x')</script> & 未修改。",
                "normalizedText": "原始内容 & 已校正。",
                "displayText": "原始内容 & 已校正！\n第二行 `code`。",
                "language": "zh-Hans",
                "confidence": 0.98,
                "speakerScores": [
                    {"speakerId": "speaker-1", "score": 0.91},
                    {"speakerId": "speaker-2", "score": 0.20},
                ],
                "speakerMargin": 0.71,
                "overlapping": False,
                "humanLocked": True,
                "revisions": [],
                "evidence": {"asr": {"provider": "qwen3-asr"}},
            },
            {
                "id": "segment-002",
                "startMs": 86_401_234,
                "endMs": 86_405_678,
                "speakerId": "speaker-2",
                "rawText": "مرحبا بالعالم — naïve façade 😀",
                "normalizedText": "مرحبا بالعالم — naïve façade 😀",
                "displayText": "مرحبا بالعالم — naïve façade 😀",
                "language": "ar",
                "confidence": 0.96,
                "speakerScores": [
                    {"speakerId": "speaker-1", "score": 0.10},
                    {"speakerId": "speaker-2", "score": 0.88},
                ],
                "speakerMargin": 0.78,
                "overlapping": False,
                "humanLocked": False,
                "revisions": [],
                "evidence": {"asr": {"provider": "qwen3-asr"}},
            },
        ],
        "provenance": {
            "offline": True,
            "workerVersion": "2.0.0",
            "transcriptionAdapter": {
                "id": "fixture",
                "version": "1",
            },
            "models": [
                {"role": "asr", "name": "Qwen3-ASR-1.7B"},
                {"role": "speaker", "name": "CAM++"},
            ],
        },
    }


_FORMATS = (
    ("json", ".json"),
    ("txt", ".txt"),
    ("markdown", ".md"),
    ("html", ".html"),
    ("xhtml", ".xhtml"),
)


@pytest.mark.parametrize(("export_format", "suffix"), _FORMATS)
def test_exports_are_byte_deterministic_and_receipted(
    tmp_path: Path,
    export_format: str,
    suffix: str,
) -> None:
    document = _document()
    before = copy.deepcopy(document)
    first = export_transcript(
        document,
        export_format=export_format,
        output_root=tmp_path,
        output_path=f"first{suffix}",
    )
    second = export_transcript(
        document,
        export_format=export_format,
        output_root=tmp_path,
        output_path=f"second{suffix}",
    )

    first_bytes = first.path.read_bytes()
    second_bytes = second.path.read_bytes()
    assert first_bytes == second_bytes
    assert first_bytes == render_transcript_export(
        document,
        export_format=export_format,
    )
    assert first.sha256 == second.sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert first.size == second.size == len(first_bytes)
    assert first.format == export_format
    assert first.path == (tmp_path / f"first{suffix}").resolve()
    assert first.to_dict() == {
        "format": export_format,
        "path": str(first.path),
        "sha256": first.sha256,
        "size": first.size,
    }
    assert not first_bytes.startswith(b"\xef\xbb\xbf")
    first_bytes.decode("utf-8", errors="strict")
    assert document == before
    assert not list(tmp_path.glob(".mts-export-*.tmp"))


def test_json_is_canonical_utf8_and_preserves_the_complete_document(
    tmp_path: Path,
) -> None:
    document = _document()
    receipt = export_transcript(
        document,
        export_format=TranscriptExportFormat.JSON,
        output_root=tmp_path,
        output_path="transcript.json",
    )
    payload = receipt.path.read_bytes()

    assert json.loads(payload.decode("utf-8")) == document
    assert b"\\u8bc4\\u5ba1" not in payload
    assert "评审".encode("utf-8") in payload
    assert payload.endswith(b"\n")
    assert payload == (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def test_txt_and_markdown_preserve_ids_speakers_timestamps_and_all_text() -> None:
    document = _document()
    txt = render_transcript_export(document, export_format="txt").decode("utf-8")
    markdown = render_transcript_export(
        document,
        export_format="markdown",
    ).decode("utf-8")

    for rendered in (txt, markdown):
        assert "segment-001" in rendered
        assert "segment-002" in rendered
        assert "speaker-1" in rendered
        assert "speaker-2" in rendered
        assert "00:00:01.234 --> 00:00:03.456" in rendered
        assert "24:00:01.234 --> 24:00:05.678" in rendered
        assert "原始 <script>alert('x')</script> & 未修改。" in rendered
        assert "原始内容 & 已校正。" in rendered
        assert "原始内容 & 已校正！\n第二行 `code`。" in rendered
        assert "مرحبا بالعالم — naïve façade 😀" in rendered

    assert "Start ms: 1234" in txt
    assert "End ms: 3456" in txt
    assert "**Start ms:** `1234`" in markdown
    assert "**End ms:** `3456`" in markdown
    assert "```text\n原始内容 & 已校正！\n第二行 `code`。\n```" in markdown


@pytest.mark.parametrize("export_format", ("html", "xhtml"))
def test_html_is_self_contained_valid_accessible_xhtml_and_escaped(
    export_format: str,
) -> None:
    payload = render_transcript_export(
        _document(),
        export_format=export_format,
    )
    text = payload.decode("utf-8")
    root = ElementTree.fromstring(text)
    namespace = {"x": "http://www.w3.org/1999/xhtml"}

    assert root.tag == "{http://www.w3.org/1999/xhtml}html"
    assert root.attrib["lang"] == "zh-Hans"
    assert root.find(".//x:main", namespace) is not None
    assert root.find(".//x:h1", namespace) is not None
    articles = root.findall(".//x:article", namespace)
    assert len(articles) == 2
    assert articles[0].attrib == {
        "class": "segment",
        "aria-labelledby": "segment-1-heading",
        "data-segment-id": "segment-001",
        "data-speaker-id": "speaker-1",
        "data-start-ms": "1234",
        "data-end-ms": "3456",
    }
    assert "<script>" not in text
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in text
    assert "主持人 &quot;A&quot; &amp; &lt;lead&gt;" in text
    assert "<link" not in text
    assert "<script" not in text
    assert "<img" not in text
    assert "url(" not in text
    assert "@page {" in text
    assert 'datetime="PT1.234S"' in text
    assert 'datetime="PT86401.234S"' in text
    assert "原始内容 &amp; 已校正！" in text
    assert "مرحبا بالعالم — naïve façade 😀" in text


def test_html_and_xhtml_aliases_have_identical_bytes() -> None:
    document = _document()
    assert render_transcript_export(
        document,
        export_format="html",
    ) == render_transcript_export(
        document,
        export_format="xhtml",
    )


@pytest.mark.parametrize(
    "export_format",
    ("json", "txt", "markdown", "html", "xhtml"),
)
def test_mapping_key_insertion_order_cannot_change_export_bytes(
    export_format: str,
) -> None:
    document = _document()
    reordered = json.loads(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    assert render_transcript_export(
        document,
        export_format=export_format,
    ) == render_transcript_export(
        reordered,
        export_format=export_format,
    )


def test_existing_output_is_rejected_without_overwrite_or_temporary_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "transcript.txt"
    target.write_bytes(b"existing-owner")

    with pytest.raises(TranscriptExportError) as raised:
        export_transcript(
            _document(),
            export_format="txt",
            output_root=tmp_path,
            output_path=target,
        )

    assert raised.value.code == "TRANSCRIPT_EXPORT_OUTPUT_EXISTS"
    assert target.read_bytes() == b"existing-owner"
    assert not list(tmp_path.glob(".mts-export-*.tmp"))


def test_concurrent_target_appearance_is_not_overwritten_or_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "transcript.md"

    def competing_link(
        temporary: str | os.PathLike[str],
        output: str | os.PathLike[str],
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        assert Path(temporary).parent == tmp_path
        Path(output).write_bytes(b"concurrent-owner")
        raise FileExistsError(str(output))

    monkeypatch.setattr(transcript_exports.os, "link", competing_link)
    with pytest.raises(TranscriptExportError) as raised:
        export_transcript(
            _document(),
            export_format="markdown",
            output_root=tmp_path,
            output_path=target,
        )

    assert raised.value.code == "TRANSCRIPT_EXPORT_OUTPUT_EXISTS"
    assert target.read_bytes() == b"concurrent-owner"
    assert not list(tmp_path.glob(".mts-export-*.tmp"))


def test_write_and_publish_failures_remove_private_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fsync(_: int) -> None:
        raise OSError("simulated write durability failure")

    monkeypatch.setattr(transcript_exports.os, "fsync", fail_fsync)
    with pytest.raises(TranscriptExportError) as write_error:
        export_transcript(
            _document(),
            export_format="txt",
            output_root=tmp_path,
            output_path="write-failure.txt",
        )
    assert write_error.value.code == "TRANSCRIPT_EXPORT_WRITE_FAILED"
    assert not (tmp_path / "write-failure.txt").exists()
    assert not list(tmp_path.glob(".mts-export-*.tmp"))

    monkeypatch.undo()

    def fail_link(
        _temporary: str | os.PathLike[str],
        _output: str | os.PathLike[str],
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        raise OSError("simulated hard-link failure")

    monkeypatch.setattr(transcript_exports.os, "link", fail_link)
    with pytest.raises(TranscriptExportError) as publish_error:
        export_transcript(
            _document(),
            export_format="txt",
            output_root=tmp_path,
            output_path="publish-failure.txt",
        )
    assert publish_error.value.code == "TRANSCRIPT_EXPORT_PUBLISH_FAILED"
    assert not (tmp_path / "publish-failure.txt").exists()
    assert not list(tmp_path.glob(".mts-export-*.tmp"))


def test_directory_sync_failure_rolls_back_published_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_sync(_: Path) -> None:
        raise OSError("simulated directory sync failure")

    monkeypatch.setattr(transcript_exports, "_sync_directory", fail_sync)
    target = tmp_path / "durability-failure.txt"
    with pytest.raises(TranscriptExportError) as raised:
        export_transcript(
            _document(),
            export_format="txt",
            output_root=tmp_path,
            output_path=target,
        )

    assert raised.value.code == "TRANSCRIPT_EXPORT_DURABILITY_FAILED"
    assert not target.exists()
    assert not list(tmp_path.glob(".mts-export-*.tmp"))


@pytest.mark.parametrize(
    "output_path",
    (
        "../escaped.txt",
        "nested/../../escaped.txt",
        "bad\nname.txt",
        "bad\x7fname.txt",
        "victim.txt:alternate-stream",
        "CON.txt",
        "trailing.",
        "question?.txt",
    ),
)
def test_path_traversal_and_control_characters_fail_closed(
    tmp_path: Path,
    output_path: str,
) -> None:
    with pytest.raises(TranscriptExportError) as raised:
        export_transcript(
            _document(),
            export_format="txt",
            output_root=tmp_path,
            output_path=output_path,
        )

    assert raised.value.code in {
        "TRANSCRIPT_EXPORT_PATH_INVALID",
        "TRANSCRIPT_EXPORT_PATH_OUTSIDE_ROOT",
    }
    assert not list(tmp_path.glob(".mts-export-*.tmp"))


def test_absolute_path_outside_root_and_missing_parent_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()

    with pytest.raises(TranscriptExportError) as outside_error:
        export_transcript(
            _document(),
            export_format="json",
            output_root=root,
            output_path=outside / "transcript.json",
        )
    assert outside_error.value.code == "TRANSCRIPT_EXPORT_PATH_OUTSIDE_ROOT"

    with pytest.raises(TranscriptExportError) as parent_error:
        export_transcript(
            _document(),
            export_format="json",
            output_root=root,
            output_path="missing/transcript.json",
        )
    assert parent_error.value.code == "TRANSCRIPT_EXPORT_PATH_INVALID"
    assert list(root.iterdir()) == []


def test_linked_parent_cannot_escape_output_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    linked = root / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(TranscriptExportError) as raised:
        export_transcript(
            _document(),
            export_format="txt",
            output_root=root,
            output_path="linked/escaped.txt",
        )

    assert raised.value.code == "TRANSCRIPT_EXPORT_PATH_OUTSIDE_ROOT"
    assert not (outside / "escaped.txt").exists()


def test_nested_existing_directory_inside_root_is_allowed(tmp_path: Path) -> None:
    nested = tmp_path / "exports" / "review"
    nested.mkdir(parents=True)

    receipt = export_transcript(
        _document(),
        export_format="html",
        output_root=tmp_path,
        output_path="exports/review/transcript.xhtml",
    )

    assert receipt.path == (nested / "transcript.xhtml").resolve()
    assert receipt.path.is_file()


@pytest.mark.parametrize(
    "export_format",
    ("JSON", "md", "pdf", "", "json\n"),
)
def test_unsupported_or_noncanonical_formats_fail_closed(
    tmp_path: Path,
    export_format: str,
) -> None:
    with pytest.raises(TranscriptExportError) as raised:
        export_transcript(
            _document(),
            export_format=export_format,
            output_root=tmp_path,
            output_path="never-created.out",
        )

    assert raised.value.code == "TRANSCRIPT_EXPORT_FORMAT_INVALID"
    assert list(tmp_path.iterdir()) == []


def _mutate(
    document: dict[str, Any],
    case: str,
) -> None:
    if case == "schema":
        document["schemaVersion"] = "1.0.0"
    elif case == "missing-job":
        del document["jobId"]
    elif case == "empty-segments":
        document["segments"] = []
    elif case == "duplicate-segment":
        document["segments"][1]["id"] = document["segments"][0]["id"]
    elif case == "bool-timestamp":
        document["segments"][0]["startMs"] = True
    elif case == "reverse-timestamp":
        document["segments"][1]["startMs"] = 1_000
    elif case == "out-of-range":
        document["segments"][1]["endMs"] = document["source"]["durationMs"] + 1
    elif case == "unknown-speaker":
        document["segments"][1]["speakerId"] = "speaker-3"
    elif case == "unobserved-speaker":
        document["segments"][1]["speakerId"] = "speaker-1"
    elif case == "empty-raw":
        document["segments"][0]["rawText"] = " "
    elif case == "xml-control":
        document["segments"][0]["displayText"] = "forbidden\x00text"
    elif case == "nonfinite":
        document["segments"][0]["confidence"] = float("nan")
    elif case == "tuple":
        document["provenance"]["models"] = ("not", "a", "json", "array")
    elif case == "noncanonical-language":
        document["language"] = "ZH_hans"
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    (
        "schema",
        "missing-job",
        "empty-segments",
        "duplicate-segment",
        "bool-timestamp",
        "reverse-timestamp",
        "out-of-range",
        "unknown-speaker",
        "unobserved-speaker",
        "empty-raw",
        "xml-control",
        "nonfinite",
        "tuple",
        "noncanonical-language",
    ),
)
def test_invalid_documents_fail_closed_without_artifacts(
    tmp_path: Path,
    case: str,
) -> None:
    document = _document()
    _mutate(document, case)

    with pytest.raises(TranscriptExportError) as raised:
        export_transcript(
            document,
            export_format="json",
            output_root=tmp_path,
            output_path="transcript.json",
        )

    assert raised.value.code == "TRANSCRIPT_EXPORT_DOCUMENT_INVALID"
    assert list(tmp_path.iterdir()) == []


def test_non_mapping_document_and_invalid_output_root_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(TranscriptExportError) as document_error:
        render_transcript_export([], export_format="json")  # type: ignore[arg-type]
    assert document_error.value.code == "TRANSCRIPT_EXPORT_DOCUMENT_INVALID"

    root_file = tmp_path / "not-a-directory"
    root_file.write_text("root", encoding="utf-8")
    with pytest.raises(TranscriptExportError) as root_error:
        export_transcript(
            _document(),
            export_format="json",
            output_root=root_file,
            output_path="transcript.json",
        )
    assert root_error.value.code == "TRANSCRIPT_EXPORT_PATH_INVALID"
    assert root_file.read_text(encoding="utf-8") == "root"
