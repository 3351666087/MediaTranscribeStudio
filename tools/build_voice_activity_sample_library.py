"""Build a pinned, licensed sample library for speech-presence evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "sample_library" / "voice-activity-manifest.v1.json"
DEFAULT_OUTPUT = ROOT / ".runtime_cache" / "sample-library" / "voice-activity"
APPROVED_LICENSES = frozenset(
    {"cc-by-3.0", "public-domain", "generated-test-fixture"}
)
SPLITS = frozenset({"development", "regression", "held-out"})
USER_AGENT = "MediaTranscribeStudio-voice-activity-samples/1.0"


class VoiceActivitySampleError(ValueError):
    """Raised when sample provenance or generated evidence is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VoiceActivitySampleError(f"{field} must be an object")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VoiceActivitySampleError(f"{field} must be a non-empty string")
    return value


def _sha256_value(value: object, *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise VoiceActivitySampleError(f"{field} must be lowercase SHA-256")
    return text


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VoiceActivitySampleError(f"cannot read manifest: {path}") from exc
    if not isinstance(value, dict):
        raise VoiceActivitySampleError("manifest must contain an object")
    if value.get("schemaVersion") != "1.0.0":
        raise VoiceActivitySampleError("unsupported manifest schemaVersion")
    max_duration = value.get("maxDurationSeconds")
    if (
        not isinstance(max_duration, (int, float))
        or isinstance(max_duration, bool)
        or not math.isfinite(float(max_duration))
        or not 0 < float(max_duration) <= 90
    ):
        raise VoiceActivitySampleError(
            "maxDurationSeconds must be finite and between 0 and 90"
        )

    raw_sources = value.get("sources")
    raw_cases = value.get("cases")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise VoiceActivitySampleError("sources must be a non-empty array")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise VoiceActivitySampleError("cases must be a non-empty array")
    sources: dict[str, Mapping[str, Any]] = {}
    for index, raw_source in enumerate(raw_sources):
        source = _object(raw_source, field=f"sources[{index}]")
        source_id = _nonempty_string(
            source.get("id"),
            field=f"sources[{index}].id",
        )
        if source_id in sources:
            raise VoiceActivitySampleError(f"duplicate source id: {source_id}")
        license_id = _nonempty_string(
            source.get("license"),
            field=f"sources[{index}].license",
        )
        if license_id not in APPROVED_LICENSES:
            raise VoiceActivitySampleError(
                f"source {source_id} license is not approved: {license_id}"
            )
        _nonempty_string(
            source.get("attribution"),
            field=f"sources[{index}].attribution",
        )
        sources[source_id] = source

    seen_cases: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _object(raw_case, field=f"cases[{index}]")
        case_id = _nonempty_string(case.get("id"), field=f"cases[{index}].id")
        if case_id in seen_cases:
            raise VoiceActivitySampleError(f"duplicate case id: {case_id}")
        seen_cases.add(case_id)
        source_id = _nonempty_string(
            case.get("sourceId"),
            field=f"cases[{index}].sourceId",
        )
        if source_id not in sources:
            raise VoiceActivitySampleError(
                f"case {case_id} references unknown source {source_id}"
            )
        if case.get("evaluationSplit") not in SPLITS:
            raise VoiceActivitySampleError(
                f"case {case_id} has an invalid evaluationSplit"
            )
        if case.get("expectedLexicalSpeech") is not False:
            raise VoiceActivitySampleError(
                f"case {case_id} must be a lexical-speech negative"
            )
        selection = _object(
            case.get("selection"),
            field=f"cases[{index}].selection",
        )
        start = selection.get("startSeconds")
        duration = selection.get("durationSeconds")
        for field, number, allow_zero in (
            ("startSeconds", start, True),
            ("durationSeconds", duration, False),
        ):
            if (
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(float(number))
                or (float(number) < 0 if allow_zero else float(number) <= 0)
            ):
                raise VoiceActivitySampleError(
                    f"case {case_id} {field} is invalid"
                )
        if float(duration) > float(max_duration):
            raise VoiceActivitySampleError(
                f"case {case_id} exceeds maxDurationSeconds"
            )
        if sources[source_id].get("kind") != "generated":
            asset = _object(case.get("asset"), field=f"cases[{index}].asset")
            url = _nonempty_string(
                asset.get("url"),
                field=f"cases[{index}].asset.url",
            )
            if not url.startswith("https://"):
                raise VoiceActivitySampleError(
                    f"case {case_id} asset URL must use HTTPS"
                )
            _sha256_value(
                asset.get("sha256"),
                field=f"cases[{index}].asset.sha256",
            )
    return value


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise VoiceActivitySampleError(f"download failed: {url}") from exc
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        actual_sha256 = _sha256(temporary)
        if actual_sha256 != expected_sha256:
            raise VoiceActivitySampleError(
                f"download SHA-256 mismatch for {destination.name}: "
                f"{actual_sha256}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _run_ffmpeg(command: list[str], *, label: str) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise VoiceActivitySampleError(
            f"ffmpeg failed for {label}: {completed.stderr.strip()}"
        )


def _normalize(
    *,
    source: Path,
    output: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start_seconds),
            "-i",
            str(source),
            "-t",
            str(duration_seconds),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        label=output.name,
    )


def _generate_silence(output: Path, *, duration_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            str(duration_seconds),
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        label=output.name,
    )


def _duration_seconds(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    return float(completed.stdout.strip())


def build_library(*, manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    sources = {
        str(source["id"]): source for source in manifest["sources"]
    }
    resolved_cases: list[dict[str, Any]] = []
    for raw_case in manifest["cases"]:
        case = dict(raw_case)
        case_id = str(case["id"])
        source = sources[str(case["sourceId"])]
        selection = case["selection"]
        duration = float(selection["durationSeconds"])
        output = output_root / "audio" / f"{case_id}.wav"
        source_sha256: str | None = None
        source_path: Path | None = None
        if source["kind"] == "generated":
            _generate_silence(output, duration_seconds=duration)
        else:
            asset = case["asset"]
            source_path = output_root / "sources" / str(asset["filename"])
            _download(
                str(asset["url"]),
                source_path,
                str(asset["sha256"]),
            )
            source_sha256 = _sha256(source_path)
            _normalize(
                source=source_path,
                output=output,
                start_seconds=float(selection["startSeconds"]),
                duration_seconds=duration,
            )
        actual_duration = _duration_seconds(output)
        if abs(actual_duration - duration) > 0.05:
            raise VoiceActivitySampleError(
                f"{case_id} duration mismatch: {actual_duration} != {duration}"
            )
        resolved_cases.append(
            {
                "id": case_id,
                "sourceId": case["sourceId"],
                "evaluationSplit": case["evaluationSplit"],
                "signalClass": case["signalClass"],
                "category": case["category"],
                "humanVocalization": case["humanVocalization"],
                "expectedLexicalSpeech": False,
                "truthEligibility": {
                    "voiceActivity": True,
                    "lexicalSpeech": True,
                    "language": False,
                    "asr": False,
                    "speakerCount": False,
                },
                "selection": dict(selection),
                "path": str(output.relative_to(output_root)),
                "bytes": output.stat().st_size,
                "durationSeconds": round(actual_duration, 6),
                "sha256": _sha256(output),
                "sourceAssetSha256": source_sha256,
                "attribution": case["attribution"],
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "schemaVersion": "1.0.0",
        "libraryId": manifest["libraryId"],
        "generatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifestPath": str(manifest_path.resolve()),
        "sources": manifest["sources"],
        "cases": resolved_cases,
    }
    resolved_path = output_root / "voice-activity-samples.resolved.v1.json"
    resolved_path.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    attribution_lines = [
        "# Voice Activity Sample Attribution",
        "",
        "This library contains no meeting or transcript content.",
        "",
    ]
    for source in manifest["sources"]:
        attribution_lines.extend(
            [
                f"## {source['id']}",
                "",
                f"- License: `{source['license']}`",
                f"- Attribution: {source['attribution']}",
                *(
                    [f"- Source: {source['homepage']}"]
                    if source.get("homepage")
                    else []
                ),
                "",
            ]
        )
    attribution_lines.extend(["## Cases", ""])
    attribution_lines.extend(
        f"- `{case['id']}`: {case['attribution']}"
        for case in manifest["cases"]
    )
    (output_root / "ATTRIBUTION.md").write_text(
        "\n".join(attribution_lines) + "\n",
        encoding="utf-8",
    )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = build_library(
        manifest_path=args.manifest.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "libraryId": resolved["libraryId"],
                "caseCount": len(resolved["cases"]),
                "outputRoot": str(args.output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
