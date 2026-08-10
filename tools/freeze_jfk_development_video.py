"""Freeze the public-domain JFK development video and its blind run manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / ".runtime_cache" / "sample-library" / "global"
    / "real-jfk-video-v1"
)
DEFAULT_SOURCE = DEFAULT_OUTPUT_ROOT / "commons-jfk-inauguration-240p.vp9.webm"
DEFAULT_WINDOW = DEFAULT_OUTPUT_ROOT / "jfk-ask-not-development.h264-aac.mp4"
DEFAULT_REFERENCE = DEFAULT_OUTPUT_ROOT / "jfk-development-video-reference.v1.json"
DEFAULT_BLIND_MANIFEST = (
    DEFAULT_OUTPUT_ROOT / "blind-product-jfk-development-video.v1.json"
)

CASE_ID = "wikimedia-jfk-ask-not-video-development"
SOURCE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/7/74/"
    "John_F._Kennedy_Inauguration_Speech.ogv/"
    "John_F._Kennedy_Inauguration_Speech.ogv.240p.vp9.webm"
)
SOURCE_BYTES = 36_029_496
SOURCE_SHA256 = "f67e097905489641d2392651b3022cc150fb6dd2fc620b6e726b7f6d4ca0fec1"
WINDOW_START_MS = 832_500
WINDOW_END_MS = 845_500
WINDOW_BYTES = 800_350
WINDOW_SHA256 = "e42db65b4a3423852395ed2b8c3c484cdb8b68db2cf333d0428df59343be0f99"
HISTORICAL_WINDOW_SHA256 = (
    "573951eb560dcc553cea7bf4fa51ee6cf8fb24d6c39dc90fabb3a42c39d144f6"
)
PAGE_REVISION_API = (
    "https://commons.wikimedia.org/w/api.php?action=query&format=json&"
    "formatversion=2&prop=info%7Crevisions%7Cimageinfo%7Cvideoinfo&"
    "titles=File%3AJohn%20F.%20Kennedy%20Inauguration%20Speech.ogv&"
    "rvprop=ids%7Ctimestamp%7Csha1%7Ccontentmodel%7Ccomment%7Cuser&rvlimit=1&"
    "iiprop=timestamp%7Cuser%7Curl%7Csize%7Csha1%7Cmime%7Cmediatype%7Cextmetadata&"
    "viprop=timestamp%7Cuser%7Curl%7Csize%7Csha1%7Cmime%7Cmediatype%7Cmetadata%7C"
    "commonmetadata%7Cextmetadata%7Cderivatives&redirects=1"
)
TIMED_TEXT_REVISION_API = (
    "https://commons.wikimedia.org/w/api.php?action=query&format=json&"
    "formatversion=2&prop=revisions&titles=TimedText%3AJohn%20F.%20Kennedy%20"
    "Inauguration%20Speech.ogv.en.srt&rvprop=ids%7Ctimestamp%7Csha1%7Ccontent&"
    "rvslots=main&rvstartid=917372138&rvendid=917372138&rvlimit=1"
)
REFERENCE_CUES = (
    {
        "number": 125,
        "sourceStartMs": 833_282,
        "sourceEndMs": 841_655,
        "windowStartMs": 782,
        "windowEndMs": 9_155,
        "text": (
            "And so, my fellow Americans: ask not what your country can do "
            "for you\u2014"
        ),
    },
    {
        "number": 126,
        "sourceStartMs": 841_655,
        "sourceEndMs": 844_985,
        "windowStartMs": 9_155,
        "windowEndMs": 12_485,
        "text": "ask what you can do for your country.",
    },
)
REFERENCE_TRANSCRIPT = " ".join(str(cue["text"]) for cue in REFERENCE_CUES)
_FORBIDDEN_BLIND_KEYS = frozenset(
    {
        "answer",
        "cues",
        "expectedTranscript",
        "nativeTranscript",
        "rawTranscript",
        "referenceTranscript",
        "referenceTruth",
        "scoringTranscript",
        "speakerIdentity",
        "timedText",
    }
)


class JfkVideoFreezeError(ValueError):
    """Raised when the frozen video or its evidence is inconsistent."""


def _tool_path(path: Path, executable: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt" or executable.suffix.casefold() != ".exe":
        return str(resolved)
    completed = subprocess.run(
        ("wslpath", "-w", str(resolved)),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise JfkVideoFreezeError(f"cannot translate path for {executable}")
    return completed.stdout.strip()


def _run(command: Sequence[str], *, label: str) -> str:
    completed = subprocess.run(
        tuple(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr[-4_000:].strip()
        raise JfkVideoFreezeError(f"{label} failed: {detail}")
    return completed.stdout


def _ffmpeg_arguments(source: str, output: str) -> list[str]:
    return [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        "832.500",
        "-i",
        source,
        "-t",
        "13.000",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-preset:v",
        "medium",
        "-crf:v",
        "18",
        "-pix_fmt:v",
        "yuv420p",
        "-threads:v",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar:a",
        "48000",
        "-ac:a",
        "2",
        "-movflags",
        "+faststart",
        output,
    ]


def _encode_window(*, ffmpeg: Path, source: Path, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}"
    )
    try:
        command = [
            str(ffmpeg.resolve(strict=True)),
            *_ffmpeg_arguments(
                _tool_path(source, ffmpeg),
                _tool_path(temporary, ffmpeg),
            ),
        ]
        _run(command, label="JFK window encoding")
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _probe(*, ffprobe: Path, media: Path) -> dict[str, Any]:
    raw = _run(
        (
            str(ffprobe.resolve(strict=True)),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            _tool_path(media, ffprobe),
        ),
        label=f"ffprobe {media.name}",
    )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JfkVideoFreezeError("ffprobe returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise JfkVideoFreezeError("ffprobe root must be an object")
    return value


def _streams(probe: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = probe.get("streams")
    if not isinstance(rows, list):
        raise JfkVideoFreezeError("ffprobe streams are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("codec_type"), str):
            result[str(row["codec_type"])] = row
    return result


def validate_source_probe(probe: Mapping[str, Any]) -> None:
    streams = _streams(probe)
    video = streams.get("video", {})
    audio = streams.get("audio", {})
    if (
        video.get("codec_name") != "vp9"
        or video.get("width") != 320
        or video.get("height") != 240
        or audio.get("codec_name") != "opus"
        or audio.get("sample_rate") != "48000"
        or audio.get("channels") != 2
    ):
        raise JfkVideoFreezeError("Commons source probe does not match 240p VP9/Opus")
    raw_format = probe.get("format")
    if not isinstance(raw_format, Mapping):
        raise JfkVideoFreezeError("Commons source format probe is missing")
    try:
        duration = float(raw_format["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JfkVideoFreezeError("Commons source duration is invalid") from exc
    if duration < WINDOW_END_MS / 1_000:
        raise JfkVideoFreezeError("Commons source is shorter than the frozen window")


def validate_window_probe(probe: Mapping[str, Any]) -> float:
    streams = _streams(probe)
    video = streams.get("video", {})
    audio = streams.get("audio", {})
    if (
        video.get("codec_name") != "h264"
        or video.get("width") != 320
        or video.get("height") != 240
        or video.get("pix_fmt") != "yuv420p"
        or audio.get("codec_name") != "aac"
        or audio.get("sample_rate") != "48000"
        or audio.get("channels") != 2
    ):
        raise JfkVideoFreezeError("derived window probe does not match H.264/AAC")
    raw_format = probe.get("format")
    if not isinstance(raw_format, Mapping):
        raise JfkVideoFreezeError("derived window format probe is missing")
    try:
        duration = float(raw_format["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JfkVideoFreezeError("derived window duration is invalid") from exc
    if not 13.0 <= duration <= 13.02:
        raise JfkVideoFreezeError("derived window duration is outside tolerance")
    return duration


def _relative_path(path: Path, manifest: Path) -> str:
    return Path(os.path.relpath(path, manifest.parent)).as_posix()


def _canonical(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "canonicalSha256": canonical_json_sha256(value)}


def reference_body(
    *,
    source: Path,
    window: Path,
    blind_manifest: Path,
    source_probe: dict[str, Any],
    window_probe: dict[str, Any],
    ffmpeg_version: Sequence[str],
) -> dict[str, Any]:
    duration = validate_window_probe(window_probe)
    validate_source_probe(source_probe)
    body = {
        "schemaVersion": "1.0.0",
        "artifactType": "wikimedia-jfk-development-video-reference",
        "sampleId": CASE_ID,
        "evaluation": {
            "split": "development",
            "tuningEligible": True,
            "heldOut": False,
        },
        "source": {
            "provider": "Wikimedia Commons",
            "page": {
                "title": "File:John F. Kennedy Inauguration Speech.ogv",
                "pageId": 5_744_345,
                "revisionId": 1_141_760_595,
                "revisionTimestamp": "2026-01-04T14:59:23Z",
                "revisionSha1": "6c6ad611fb6403116d000ec5bce510f40912baa1",
                "revisionUrl": (
                    "https://commons.wikimedia.org/w/index.php?title=File%3AJohn_F._"
                    "Kennedy_Inauguration_Speech.ogv&oldid=1141760595"
                ),
                "metadataApi": PAGE_REVISION_API,
            },
            "originalMedia": {
                "uploadTimestamp": "2018-03-05T12:19:31Z",
                "sha1": "a20a2212f6f103eca8b4cb99b9a9d078fb50c622",
                "bytes": 253_165_444,
                "width": 640,
                "height": 480,
                "durationSeconds": 930.0967634300968,
            },
            "derivative": {
                "url": SOURCE_URL,
                "transcodeKey": "240p.vp9.webm",
                "mimeType": 'video/webm; codecs="vp9, opus"',
                "lastModified": "Thu, 30 Aug 2018 18:41:16 GMT",
                "etag": "4d22db66986b3543f54416de1fd99a6c",
                "objectSha1": "7ba599aa7bb4915c7d0dc39f53b4ea1f4ae1c841",
                "path": str(source.resolve()),
                "bytes": SOURCE_BYTES,
                "sha256": SOURCE_SHA256,
                "ffprobe": source_probe,
            },
        },
        "license": {
            "id": "public-domain-us-government",
            "shortName": "Public domain",
            "usageTerms": "Public domain",
            "copyrighted": False,
            "attributionRequired": False,
            "artist": "John F. Kennedy",
            "credit": (
                "John F. Kennedy Presidential Library, USG-17; via Wikimedia Commons"
            ),
            "sourcePageCreditUrl": (
                "https://www.jfklibrary.org/Asset-Viewer/Archives/USG-17.aspx"
            ),
        },
        "window": {
            "selectionPolicy": "official-timed-text-cues-125-126-not-model-output",
            "sourceStartMs": WINDOW_START_MS,
            "sourceEndMs": WINDOW_END_MS,
            "requestedDurationMs": WINDOW_END_MS - WINDOW_START_MS,
            "path": str(window.resolve()),
            "bytes": WINDOW_BYTES,
            "sha256": WINDOW_SHA256,
            "durationSeconds": duration,
            "ffprobe": window_probe,
            "encoding": {
                "ffmpegVersion": list(ffmpeg_version),
                "arguments": _ffmpeg_arguments("{source}", "{output}"),
                "metadataCopied": False,
            },
            "artifactContinuity": {
                "historicalSha256": HISTORICAL_WINDOW_SHA256,
                "historicalByteIdentityMatched": WINDOW_SHA256
                == HISTORICAL_WINDOW_SHA256,
                "samePinnedSourceAndWindow": True,
                "note": (
                    "The prior artifact used a different encoder build or recipe. "
                    "This Windows freeze records a new immutable byte identity."
                ),
            },
        },
        "referenceTruth": {
            "language": "en-US",
            "expectedSpeakerCount": 1,
            "speakerIdentity": "John F. Kennedy",
            "referenceTranscript": REFERENCE_TRANSCRIPT,
            "timedText": {
                "title": (
                    "TimedText:John F. Kennedy Inauguration Speech.ogv.en.srt"
                ),
                "revisionId": 917_372_138,
                "revisionTimestamp": "2024-08-29T22:14:45Z",
                "revisionSha1": "231acff18a9eb3dfef5b3873630ef712d9d51f4a",
                "revisionApi": TIMED_TEXT_REVISION_API,
                "cues": list(REFERENCE_CUES),
                "windowLeadingSilenceMs": 782,
                "windowTrailingSilenceMs": 515,
            },
        },
        "truthPersistencePolicy": {
            "referenceTruthPersisted": True,
            "referenceMayEnterProductRun": False,
            "blindManifest": str(blind_manifest.resolve()),
            "blindManifestContainsTranscriptOrTimeline": False,
        },
    }
    return _canonical(body)


def _assert_truth_redacted(value: Mapping[str, Any]) -> None:
    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            forbidden = _FORBIDDEN_BLIND_KEYS.intersection(node)
            if forbidden:
                raise JfkVideoFreezeError(
                    "blind manifest contains truth keys: "
                    + ", ".join(sorted(forbidden))
                )
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    encoded = json.dumps(value, ensure_ascii=False)
    if any(str(cue["text"]) in encoded for cue in REFERENCE_CUES):
        raise JfkVideoFreezeError("blind manifest contains reference text")


def blind_body(
    *,
    window: Path,
    manifest_path: Path,
    window_probe: dict[str, Any],
) -> dict[str, Any]:
    duration = validate_window_probe(window_probe)
    streams = _streams(window_probe)
    video = streams["video"]
    audio = streams["audio"]
    body = {
        "schemaVersion": "1.0.0",
        "artifactType": "truth-redacted-local-product-matrix",
        "libraryId": "mts-wikimedia-jfk-development-video-v1",
        "selection": {
            "splits": ["development"],
            "heldOutExplicitlyUnlocked": False,
            "tuningEligible": True,
        },
        "truthPersistencePolicy": {
            "referenceTranscriptPersisted": False,
            "referenceTimelinePersisted": False,
            "expectedAnswerPersisted": False,
            "modelIdentityInReviewPacket": False,
        },
        "counts": {"cases": 1, "missingCases": 0},
        "missingCaseIds": [],
        "cases": [
            {
                "id": CASE_ID,
                "sourceId": "wikimedia-commons-jfk-inauguration",
                "language": "en-US",
                "region": "North America",
                "evaluationSplit": "development",
                "scenario": [
                    "real-recording",
                    "single-speaker",
                    "archival-video",
                    "background-crowd",
                ],
                "expectedSpeakerCount": 1,
                "path": _relative_path(window, manifest_path),
                "durationSeconds": duration,
                "bytes": WINDOW_BYTES,
                "sha256": WINDOW_SHA256,
                "container": "mp4",
                "video": {
                    "codec": video["codec_name"],
                    "width": video["width"],
                    "height": video["height"],
                    "pixelFormat": video["pix_fmt"],
                    "frameRate": video.get("avg_frame_rate"),
                },
                "audio": {
                    "codec": audio["codec_name"],
                    "sampleRateHz": int(str(audio["sample_rate"])),
                    "channels": audio["channels"],
                    "durationSeconds": duration,
                },
                "sourceLocator": {
                    "provider": "Wikimedia Commons",
                    "pageId": 5_744_345,
                    "pageRevisionId": 1_141_760_595,
                    "sourceMediaSha256": SOURCE_SHA256,
                    "sourceWindowMs": [WINDOW_START_MS, WINDOW_END_MS],
                    "license": "public-domain-us-government",
                },
            }
        ],
    }
    _assert_truth_redacted(body)
    return _canonical(body)


def _validate_file(path: Path, *, size: int, digest: str, label: str) -> None:
    if not path.is_file():
        raise JfkVideoFreezeError(f"{label} is missing: {path}")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise JfkVideoFreezeError(f"{label} byte identity does not match the freeze")


def freeze(
    *,
    source: Path,
    window: Path,
    reference: Path,
    blind_manifest: Path,
    ffmpeg: Path,
    ffprobe: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_file(
        source.resolve(strict=True),
        size=SOURCE_BYTES,
        digest=SOURCE_SHA256,
        label="Commons 240p source",
    )
    _encode_window(ffmpeg=ffmpeg, source=source, output=window)
    _validate_file(
        window.resolve(strict=True),
        size=WINDOW_BYTES,
        digest=WINDOW_SHA256,
        label="JFK development window",
    )
    source_probe = _probe(ffprobe=ffprobe, media=source)
    window_probe = _probe(ffprobe=ffprobe, media=window)
    version = _run(
        (str(ffmpeg.resolve(strict=True)), "-version"),
        label="ffmpeg version",
    ).splitlines()
    reference_value = reference_body(
        source=source,
        window=window,
        blind_manifest=blind_manifest,
        source_probe=source_probe,
        window_probe=window_probe,
        ffmpeg_version=version,
    )
    blind_value = blind_body(
        window=window,
        manifest_path=blind_manifest,
        window_probe=window_probe,
    )
    atomic_write_json(reference, reference_value)
    atomic_write_json(blind_manifest, blind_value)
    return reference_value, blind_value


def _default_tool(name: str) -> Path:
    located = shutil.which(name) or shutil.which(f"{name}.exe")
    if located:
        return Path(located)
    windows = Path(f"/mnt/c/ffmpeg/bin/{name}.exe")
    if windows.is_file():
        return windows
    return Path(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--window", type=Path, default=DEFAULT_WINDOW)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--blind-manifest",
        type=Path,
        default=DEFAULT_BLIND_MANIFEST,
    )
    parser.add_argument("--ffmpeg", type=Path, default=_default_tool("ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=_default_tool("ffprobe"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference, blind = freeze(
        source=args.source,
        window=args.window,
        reference=args.reference,
        blind_manifest=args.blind_manifest,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    print(
        json.dumps(
            {
                "reference": str(args.reference.resolve()),
                "referenceFileSha256": sha256_file(args.reference.resolve()),
                "referenceCanonicalSha256": reference["canonicalSha256"],
                "blindManifest": str(args.blind_manifest.resolve()),
                "blindManifestFileSha256": sha256_file(
                    args.blind_manifest.resolve()
                ),
                "blindManifestCanonicalSha256": blind["canonicalSha256"],
                "mediaSha256": WINDOW_SHA256,
                "caseId": CASE_ID,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
