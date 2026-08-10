"""Run one reproducible MediaTranscribeStudio product job.

This is the operator-facing wrapper around ``run_production_smoke``.  It
selects a versioned output recipe from a bounded content probe, then delegates
all authoritative media admission, transcription, semantic arbitration, PDF
rendering, and transactional subtitle publication to the production worker.
The wrapper never edits an existing output directory and writes one
no-replace receipt after the worker exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.output_recipe import OutputRecipeError, parse_output_recipe  # noqa: E402


RECEIPT_SCHEMA_VERSION = "1.0.0"
RECEIPT_ARTIFACT_TYPE = "product-e2e-receipt"
REQUIRED_FORMATS = frozenset({"pdf", "txt", "json", "srt", "webvtt", "ass"})
REQUIRED_SUBTITLE_FORMATS = frozenset({"srt", "webvtt", "ass"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_WINDOWS_PATH = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<tail>.*)$")


class ProductE2EError(ValueError):
    """Raised when the fixed product command cannot prove its contract."""


def _host_path(value: str | os.PathLike[str]) -> Path:
    """Map a persisted Windows path when this verifier runs under WSL."""

    raw = os.fspath(value)
    match = _WINDOWS_PATH.fullmatch(raw)
    if os.name != "nt" and match is not None:
        tail = match.group("tail").replace("\\", "/")
        return Path("/mnt") / match.group("drive").lower() / Path(tail)
    return Path(raw)


def _resolve_file(value: str | os.PathLike[str], *, label: str) -> Path:
    path = _host_path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ProductE2EError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _resolve_output(value: str | os.PathLike[str]) -> Path:
    path = _host_path(value).expanduser().absolute()
    if path.exists():
        raise ProductE2EError(
            f"output directory already exists; refusing to overwrite: {path}"
        )
    return path


def _resolve_existing_output(value: str | os.PathLike[str]) -> Path:
    path = _host_path(value).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise ProductE2EError(
            f"existing output must be a regular directory: {path}"
        )
    return path.resolve(strict=True)


def _file_digest(path: Path) -> str:
    return sha256_file(path)


def _canonical_recipe(
    path: Path,
    *,
    media_kind: str,
) -> tuple[dict[str, Any], str, str]:
    """Load and pin one of the two supported product recipes."""

    recipe_path = _resolve_file(path, label="output recipe")
    try:
        recipe = parse_output_recipe(read_json_strict(recipe_path))
    except (OSError, ValueError, OutputRecipeError) as exc:
        raise ProductE2EError(f"output recipe is invalid: {recipe_path}") from exc
    payload = recipe.canonical_dict()
    delivery = payload["delivery"]
    formats = tuple(delivery["formats"])
    modes = tuple(delivery["subtitleModes"])
    if not recipe.render_pdf or not REQUIRED_FORMATS <= set(formats):
        raise ProductE2EError(
            "product recipe must request PDF, TXT, JSON, SRT, WebVTT, and ASS"
        )
    expected_modes = (
        ("sidecar",)
        if media_kind == "audio"
        else ("sidecar", "soft-mux", "burn-in")
    )
    if modes != expected_modes:
        raise ProductE2EError(
            f"{media_kind} product recipe must use subtitleModes={list(expected_modes)!r}"
        )
    return payload, _file_digest(recipe_path), recipe.deterministic_hash()


def classify_probe_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Classify FFprobe JSON without trusting filename extensions."""

    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list) or not raw_streams:
        raise ProductE2EError("content probe did not return a non-empty streams array")
    audio: list[int] = []
    video: list[int] = []
    for index, raw in enumerate(raw_streams):
        if not isinstance(raw, Mapping):
            raise ProductE2EError(f"content probe stream {index} is malformed")
        raw_index = raw.get("index", index)
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            raise ProductE2EError(f"content probe stream {index} has an invalid index")
        codec_type = raw.get("codec_type")
        if codec_type == "audio":
            audio.append(raw_index)
        elif codec_type == "video":
            disposition = raw.get("disposition")
            attached = (
                isinstance(disposition, Mapping)
                and disposition.get("attached_pic") == 1
            )
            if not attached:
                video.append(raw_index)
    if not audio and not video:
        raise ProductE2EError("content probe found neither audio nor video")
    kind = "video" if video else "audio"
    return {
        "kind": kind,
        "audioStreamIndexes": sorted(set(audio)),
        "videoStreamIndexes": sorted(set(video)),
    }


def _configured_ffprobe(config_path: Path) -> str:
    """Resolve a sibling ffprobe from the configured ffmpeg, with PATH fallback."""

    config = read_json_strict(_resolve_file(config_path, label="production config"))
    executables = config.get("executables")
    raw = executables.get("ffmpeg") if isinstance(executables, Mapping) else None
    if isinstance(raw, str) and raw.strip():
        ffmpeg = _host_path(raw.strip())
        if ffmpeg.suffix.casefold() == ".exe":
            sibling = ffmpeg.with_name("ffprobe.exe")
        else:
            sibling = ffmpeg.with_name("ffprobe")
        configured_path = ffmpeg.is_absolute() or ffmpeg.parent != Path(".")
        if configured_path:
            if not sibling.is_file():
                raise ProductE2EError(
                    f"configured ffprobe sibling is missing: {sibling}"
                )
            return str(sibling.resolve(strict=True))
        discovered = shutil.which(sibling.name)
        if discovered is not None:
            return discovered
    return shutil.which("ffprobe") or "ffprobe"


def _validate_cli_contract(args: argparse.Namespace) -> None:
    if args.receipt_only and args.existing_result is None:
        raise ProductE2EError("--receipt-only requires --existing-result")
    if not args.receipt_only and args.existing_result is not None:
        raise ProductE2EError("--existing-result requires --receipt-only")
    if args.receipt_only and args.media_kind == "auto":
        raise ProductE2EError(
            "--receipt-only requires explicit --media-kind audio or video"
        )
    for name in ("idle_timeout_seconds", "hard_timeout_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            option = "--" + name.replace("_", "-")
            raise ProductE2EError(f"{option} must be finite and greater than zero")

    count_fields = {
        "speaker_count": args.speaker_count,
        "speaker_count_min": args.speaker_count_min,
        "speaker_count_max": args.speaker_count_max,
        "speaker_count_prior": args.speaker_count_prior,
    }
    for name, value in count_fields.items():
        if value is not None and value < 1:
            option = "--" + name.replace("_", "-")
            raise ProductE2EError(f"{option} must be a positive integer")
    if args.mode == "manual":
        if args.speaker_count is None:
            raise ProductE2EError("manual mode requires --speaker-count")
        if any(
            count_fields[name] is not None
            for name in (
                "speaker_count_min",
                "speaker_count_max",
                "speaker_count_prior",
            )
        ):
            raise ProductE2EError(
                "manual mode does not accept hybrid speaker count options"
            )
    elif args.mode == "hybrid":
        if args.speaker_count is not None:
            raise ProductE2EError("hybrid mode does not accept --speaker-count")
        if args.speaker_count_min is None or args.speaker_count_max is None:
            raise ProductE2EError("hybrid mode requires speaker count bounds")
        if args.speaker_count_min > args.speaker_count_max:
            raise ProductE2EError(
                "--speaker-count-min must not exceed --speaker-count-max"
            )
        if args.speaker_count_prior is not None and not (
            args.speaker_count_min
            <= args.speaker_count_prior
            <= args.speaker_count_max
        ):
            raise ProductE2EError(
                "--speaker-count-prior must remain within the hybrid bounds"
            )
    elif any(value is not None for value in count_fields.values()):
        raise ProductE2EError("auto mode does not accept speaker count options")


def _validated_job_id(value: str) -> str:
    job_id = value.strip()
    if _JOB_ID.fullmatch(job_id) is None:
        raise ProductE2EError("--job-id is not protocol-safe")
    return job_id


def probe_media_kind(
    source: Path,
    *,
    ffprobe: str,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Perform a bounded selection probe; the worker repeats authoritative admission."""

    before = _file_digest(source)
    command = (
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(source),
    )
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProductE2EError("bounded content probe could not complete") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise ProductE2EError(f"bounded content probe failed: {detail}")
    if len(completed.stdout) > 4 * 1024 * 1024:
        raise ProductE2EError("bounded content probe exceeded its output limit")
    try:
        payload = json.loads(completed.stdout.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductE2EError("bounded content probe returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ProductE2EError("bounded content probe root must be an object")
    selection = classify_probe_payload(payload)
    after = _file_digest(source)
    if before != after:
        raise ProductE2EError("source media changed during recipe selection probe")
    selection.update(
        {
            "method": "content-probe",
            "executable": ffprobe,
            "commandSha256": hashlib.sha256(
                json.dumps(
                    command,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "sourceSha256": after,
        }
    )
    return selection


def build_worker_command(
    *,
    config: Path,
    source: Path,
    output: Path,
    recipe: Path,
    job_id: str,
    mode: str,
    speaker_count: int | None,
    speaker_count_min: int | None,
    speaker_count_max: int | None,
    speaker_count_prior: int | None,
    language: str,
    title: str,
    local_llm_mode: str,
    local_llm_model: str,
    review_decisions: Path | None,
    idle_timeout_seconds: float,
    hard_timeout_seconds: float,
) -> tuple[str, ...]:
    command: list[str] = [
        sys.executable,
        "-m",
        "tools.run_production_smoke",
        "--config",
        str(config),
        "--source",
        str(source),
        "--output-dir",
        str(output),
        "--job-id",
        job_id,
        "--mode",
        mode,
        "--title",
        title,
        "--language",
        language,
        "--local-llm-mode",
        local_llm_mode,
        "--local-llm-model",
        local_llm_model,
        "--output-recipe",
        str(recipe),
        "--idle-timeout-seconds",
        str(idle_timeout_seconds),
        "--hard-timeout-seconds",
        str(hard_timeout_seconds),
    ]
    if mode == "manual":
        if speaker_count is None:
            raise ProductE2EError("manual mode requires --speaker-count")
        command.extend(("--speaker-count", str(speaker_count)))
    elif mode == "hybrid":
        if speaker_count_min is None or speaker_count_max is None:
            raise ProductE2EError("hybrid mode requires speaker count bounds")
        command.extend(
            (
                "--speaker-count-min",
                str(speaker_count_min),
                "--speaker-count-max",
                str(speaker_count_max),
            )
        )
        if speaker_count_prior is not None:
            command.extend(("--speaker-count-prior", str(speaker_count_prior)))
    if review_decisions is not None:
        command.extend(("--review-decisions", str(review_decisions)))
    return tuple(command)


def _output_member(raw: Any, *, output: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ProductE2EError(f"{label} must be a path")
    path = _resolve_file(raw, label=label)
    try:
        path.relative_to(output.resolve())
    except ValueError as exc:
        raise ProductE2EError(f"{label} escapes the output directory") from exc
    return path


def _artifact_record(path: Path, *, label: str, output: Path) -> dict[str, Any]:
    verified = _output_member(str(path), output=output, label=label)
    return {
        "path": str(path),
        "sizeBytes": verified.stat().st_size,
        "sha256": _file_digest(verified),
    }


def validate_completed_run(
    *,
    output: Path,
    result_path: Path,
    expected_job_id: str,
    media_kind: str,
    expected_recipe_hash: str,
    source: Path,
    source_sha256_before: str,
) -> dict[str, Any]:
    """Independently validate the minimum complete product contract."""

    output = output.resolve(strict=True)
    result_path = _resolve_file(result_path, label="worker result")
    result = read_json_strict(result_path)
    checkpoint_path = output / "checkpoint.v2.json"
    checkpoint = read_json_strict(_resolve_file(checkpoint_path, label="checkpoint"))
    if (
        result.get("status") != "observed"
        or result.get("terminal_type") != "job.completed"
        or result.get("job_id") != expected_job_id
        or result.get("exit_code") != 0
        or result.get("shutdown_acknowledged") is not True
        or result.get("forced_cleanup_pids") != []
        or result.get("error") is not None
    ):
        raise ProductE2EError("worker did not end in a clean job.completed state")
    if checkpoint.get("status") != "completed" or checkpoint.get("stage") != "completed":
        raise ProductE2EError("checkpoint is not completed")
    recipe_binding = checkpoint.get("outputCustomization")
    if (
        not isinstance(recipe_binding, Mapping)
        or recipe_binding.get("sha256") != expected_recipe_hash
    ):
        raise ProductE2EError("checkpoint recipe hash does not match the selected recipe")
    semantic = checkpoint.get("semantic")
    if not isinstance(semantic, Mapping) or semantic.get("status") != "completed":
        raise ProductE2EError("semantic arbitration did not complete")
    semantic_path = _output_member(
        semantic.get("artifactPath"),
        output=output,
        label="semantic artifact",
    )
    semantic_artifacts = [
        _artifact_record(
            semantic_path,
            label="semantic artifact",
            output=output,
        )
    ]
    for raw in semantic.get("artifactPaths", []):
        if isinstance(raw, str):
            path = _host_path(raw)
            if path.is_file() and path.resolve() != semantic_path.resolve():
                semantic_artifacts.append(_artifact_record(path, label="semantic evidence", output=output))
    final_path = _resolve_file(
        output / "final-adjudicated-transcript.v1.json",
        label="final transcript",
    )
    transcript_exports: list[dict[str, Any]] = []
    for receipt in checkpoint.get("transcriptExports", []):
        if not isinstance(receipt, Mapping):
            raise ProductE2EError("transcript export receipt is malformed")
        path = _output_member(receipt.get("path"), output=output, label="transcript export")
        record = _artifact_record(path, label="transcript export", output=output)
        if (
            receipt.get("sizeBytes") != record["sizeBytes"]
            or receipt.get("sha256") != record["sha256"]
        ):
            raise ProductE2EError("transcript export receipt hash is stale")
        transcript_exports.append({"format": receipt.get("format"), **record})
    if {str(item.get("format")) for item in transcript_exports} < {"json", "txt"}:
        raise ProductE2EError("JSON and TXT transcript exports are required")
    publication = checkpoint.get("outputPublication")
    if (
        not isinstance(publication, Mapping)
        or publication.get("status") != "published"
    ):
        raise ProductE2EError("output publication did not complete")
    manifest_path = _output_member(
        publication.get("manifestPath"),
        output=output,
        label="publication manifest",
    )
    manifest = read_json_strict(manifest_path)
    if manifest.get("manifestSha256") != publication.get("manifestSha256"):
        raise ProductE2EError("publication manifest hash is stale")
    body = dict(manifest)
    declared_manifest_hash = body.pop("manifestSha256", None)
    if canonical_json_sha256(body) != declared_manifest_hash:
        raise ProductE2EError("publication manifest self hash does not verify")
    source_binding = manifest.get("source")
    if (
        not isinstance(source_binding, Mapping)
        or source_binding.get("unchanged") is not True
    ):
        raise ProductE2EError("publication does not prove immutable source media")
    if source_binding.get("sha256") != source_sha256_before:
        raise ProductE2EError("publication source hash does not match the input")
    customer: list[dict[str, Any]] = []
    for item in manifest.get("customerArtifacts", []):
        if not isinstance(item, Mapping):
            raise ProductE2EError("customer artifact receipt is malformed")
        path = _output_member(
            item.get("path"),
            output=output,
            label="customer artifact",
        )
        record = _artifact_record(path, label="customer artifact", output=output)
        if (
            item.get("sizeBytes") != record["sizeBytes"]
            or item.get("sha256") != record["sha256"]
        ):
            raise ProductE2EError("customer artifact receipt hash is stale")
        integrity = item.get("sourceIntegrity")
        publication_receipt = item.get("publication")
        if (
            not isinstance(integrity, Mapping)
            or integrity.get("unchanged") is not True
            or integrity.get("sourceSha256") != source_sha256_before
            or not isinstance(publication_receipt, Mapping)
            or publication_receipt.get("atomic") is not True
            or publication_receipt.get("noReplace") is not True
            or publication_receipt.get("sourceMediaImmutable") is not True
        ):
            raise ProductE2EError("customer artifact publication guarantees are incomplete")
        customer.append({**dict(item), **record})
    sidecars = {
        str(item.get("subtitleFormat"))
        for item in customer
        if item.get("deliveryMode") == "sidecar"
    }
    if sidecars != REQUIRED_SUBTITLE_FORMATS:
        raise ProductE2EError("SRT, WebVTT, and ASS sidecars are required")
    modes = {str(item.get("deliveryMode")) for item in customer}
    expected_modes = (
        {"sidecar"}
        if media_kind == "audio"
        else {"sidecar", "soft-mux", "burn-in"}
    )
    if modes != expected_modes:
        raise ProductE2EError(
            f"publication delivery modes are invalid: expected={sorted(expected_modes)} "
            f"actual={sorted(modes)}"
        )
    if media_kind == "video":
        video_receipts = [
            item
            for item in customer
            if item.get("deliveryMode") in {"soft-mux", "burn-in"}
        ]
        if any(item.get("artifactType") != "subtitled-media" for item in video_receipts):
            raise ProductE2EError("video delivery receipts are malformed")
        for item in video_receipts:
            visual_hash = item.get("visualQaEvidenceSha256")
            if not isinstance(visual_hash, str) or _SHA256.fullmatch(visual_hash) is None:
                raise ProductE2EError(
                    "video delivery receipts require a verified visual QA hash"
                )
    transaction = manifest.get("transaction")
    if not isinstance(transaction, Mapping) or any(
        value is not True
        for value in (
            transaction.get("allConflictsCheckedBeforeWrites"),
            transaction.get("allMediaQaPassedBeforePublication"),
            transaction.get("rollbackSupported"),
            transaction.get("sourceMediaImmutable"),
        )
    ):
        raise ProductE2EError("publication transaction guarantees are incomplete")
    quality_path = _output_member(
        checkpoint.get("qualityReportPath"),
        output=output,
        label="PDF quality report",
    )
    quality = read_json_strict(quality_path)
    if quality.get("status") != "passed" or checkpoint.get("qualityStatus") != "passed":
        raise ProductE2EError("PDF quality gate did not pass")
    pdf_paths = sorted(output.rglob("*.pdf"))
    if not pdf_paths:
        raise ProductE2EError("published PDF is missing")
    source_sha256_after = _file_digest(source)
    if source_sha256_after != source_sha256_before:
        raise ProductE2EError("source media changed during product execution")
    return {
        "checkpoint": _artifact_record(checkpoint_path, label="checkpoint", output=output),
        "result": {
            "path": str(result_path),
            "sizeBytes": result_path.stat().st_size,
            "sha256": _file_digest(result_path),
        },
        "finalTranscript": _artifact_record(final_path, label="final transcript", output=output),
        "semantic": semantic_artifacts,
        "transcriptExports": transcript_exports,
        "customerArtifacts": customer,
        "publicationManifest": _artifact_record(
            manifest_path,
            label="publication manifest",
            output=output,
        ),
        "pdf": _artifact_record(pdf_paths[0], label="PDF", output=output),
        "qualityReport": _artifact_record(
            quality_path,
            label="PDF quality report",
            output=output,
        ),
        "manifestSha256": declared_manifest_hash,
        "sourceSha256After": source_sha256_after,
    }


def build_receipt(
    *,
    command: Sequence[str],
    job_id: str,
    source: Path,
    source_sha256_before: str,
    media_selection: Mapping[str, Any],
    recipe_path: Path,
    recipe_file_sha256: str,
    recipe_canonical_sha256: str,
    result_path: Path,
    worker_returncode: int | None,
    validation: Mapping[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    source_sha256_after = _file_digest(source) if source.is_file() else None
    body: dict[str, Any] = {
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "artifactType": RECEIPT_ARTIFACT_TYPE,
        "status": "passed" if validation is not None and error is None else "failed",
        "source": {
            "path": str(source),
            "sizeBytes": source.stat().st_size if source.is_file() else None,
            "sha256Before": source_sha256_before,
            "sha256After": source_sha256_after,
            "unchanged": source_sha256_after == source_sha256_before,
        },
        "mediaSelection": dict(media_selection),
        "recipe": {
            "path": str(recipe_path),
            "fileSha256": recipe_file_sha256,
            "canonicalSha256": recipe_canonical_sha256,
        },
        "worker": {
            "command": list(command),
            "jobId": job_id,
            "returnCode": worker_returncode,
            "resultPath": str(result_path),
            "resultSha256": _file_digest(result_path) if result_path.is_file() else None,
        },
        "validation": dict(validation or {}),
        "safety": {
            "sourceMediaImmutable": source_sha256_after == source_sha256_before,
            "authoritativeWorkerProbe": True,
            "failFastRecipeCompilation": True,
            "atomicPublication": validation is not None and error is None,
            "overwriteExistingOutputs": False,
        },
    }
    if error is not None:
        body["error"] = error
    body["canonicalSha256"] = canonical_json_sha256(body)
    return body


def _default_paths() -> tuple[Path, Path]:
    return (
        PROJECT_ROOT / "production.config.json",
        PROJECT_ROOT / "configs" / "product-e2e-output-recipe.v1.json",
    )


def build_parser() -> argparse.ArgumentParser:
    default_config, _ = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--media-kind",
        choices=("auto", "audio", "video"),
        default="auto",
    )
    parser.add_argument(
        "--ffprobe",
        default=None,
        help="bounded selection probe executable",
    )
    parser.add_argument(
        "--audio-recipe",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "product-audio-e2e-output-recipe.v1.json"
        ),
    )
    parser.add_argument(
        "--video-recipe",
        type=Path,
        default=PROJECT_ROOT / "configs" / "product-e2e-output-recipe.v1.json",
    )
    parser.add_argument("--job-id")
    parser.add_argument(
        "--mode",
        choices=("auto", "manual", "hybrid"),
        default="auto",
    )
    parser.add_argument("--speaker-count", type=int)
    parser.add_argument("--speaker-count-min", type=int)
    parser.add_argument("--speaker-count-max", type=int)
    parser.add_argument("--speaker-count-prior", type=int)
    parser.add_argument("--language", default="auto")
    parser.add_argument(
        "--title",
        default="MediaTranscribeStudio product transcript",
    )
    parser.add_argument(
        "--local-llm-mode",
        choices=("suggestion-only", "business", "enabled"),
        default="suggestion-only",
    )
    parser.add_argument("--local-llm-model")
    parser.add_argument("--review-decisions", type=Path)
    parser.add_argument("--idle-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--hard-timeout-seconds", type=float, default=7200.0)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="no-replace receipt path; defaults beside output",
    )
    parser.add_argument(
        "--receipt-only",
        action="store_true",
        help="verify an existing completed run without starting a worker",
    )
    parser.add_argument(
        "--existing-result",
        type=Path,
        help="result JSON for --receipt-only",
    )
    return parser


def _configured_model(config_path: Path) -> str:
    payload = read_json_strict(_resolve_file(config_path, label="production config"))
    speaker = payload.get("speaker")
    if not isinstance(speaker, Mapping):
        raise ProductE2EError("production config has no speaker object")
    model = speaker.get("localLlmModel")
    if not isinstance(model, str) or not model.strip():
        raise ProductE2EError("production config has no speaker.localLlmModel")
    return model.strip()


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli_contract(args)
    source = _resolve_file(args.source, label="source media")
    output = (
        _resolve_existing_output(args.output_dir)
        if args.receipt_only
        else _resolve_output(args.output_dir)
    )
    config = _resolve_file(args.config, label="production config")
    source_sha = _file_digest(source)
    if args.media_kind == "auto":
        selection = probe_media_kind(
            source,
            ffprobe=args.ffprobe or _configured_ffprobe(config),
        )
    else:
        selection = {
            "kind": args.media_kind,
            "method": "explicit",
            "sourceSha256": source_sha,
        }
    recipe_path = (
        args.audio_recipe
        if selection["kind"] == "audio"
        else args.video_recipe
    )
    recipe_path = _resolve_file(recipe_path, label="output recipe")
    recipe, recipe_file_sha, recipe_canonical_sha = _canonical_recipe(
        recipe_path,
        media_kind=selection["kind"],
    )
    del recipe
    job_id = _validated_job_id(args.job_id or f"product-e2e-{source_sha[:24]}")
    model = args.local_llm_model or _configured_model(config)
    if not isinstance(model, str) or not model.strip():
        raise ProductE2EError("local LLM model must be non-empty text")
    model = model.strip()
    review_decisions = (
        _resolve_file(args.review_decisions, label="review decisions")
        if args.review_decisions is not None
        else None
    )
    result_path = output.parent / f"{output.name}-result.json"
    command = build_worker_command(
        config=config.resolve(),
        source=source,
        output=output.resolve(),
        recipe=recipe_path,
        job_id=job_id,
        mode=args.mode,
        speaker_count=args.speaker_count,
        speaker_count_min=args.speaker_count_min,
        speaker_count_max=args.speaker_count_max,
        speaker_count_prior=args.speaker_count_prior,
        language=args.language,
        title=args.title,
        local_llm_mode=args.local_llm_mode,
        local_llm_model=model,
        review_decisions=review_decisions,
        idle_timeout_seconds=args.idle_timeout_seconds,
        hard_timeout_seconds=args.hard_timeout_seconds,
    )
    if args.receipt_only:
        result_path = _resolve_file(args.existing_result, label="worker result")
        validation = validate_completed_run(
            output=output,
            result_path=result_path,
            expected_job_id=job_id,
            media_kind=selection["kind"],
            expected_recipe_hash=recipe_canonical_sha,
            source=source,
            source_sha256_before=source_sha,
        )
        receipt = build_receipt(
            command=command,
            job_id=job_id,
            source=source,
            source_sha256_before=source_sha,
            media_selection=selection,
            recipe_path=recipe_path.resolve(),
            recipe_file_sha256=recipe_file_sha,
            recipe_canonical_sha256=recipe_canonical_sha,
            result_path=result_path,
            worker_returncode=0,
            validation=validation,
        )
    else:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        validation = None
        error: str | None = None
        if completed.returncode == 0:
            try:
                validation = validate_completed_run(
                    output=output,
                    result_path=result_path,
                    expected_job_id=job_id,
                    media_kind=selection["kind"],
                    expected_recipe_hash=recipe_canonical_sha,
                    source=source,
                    source_sha256_before=source_sha,
                )
            except (OSError, ValueError, ProductE2EError) as exc:
                error = str(exc)
        else:
            error = f"production smoke exited with status {completed.returncode}"
        receipt = build_receipt(
            command=command,
            job_id=job_id,
            source=source,
            source_sha256_before=source_sha,
            media_selection=selection,
            recipe_path=recipe_path,
            recipe_file_sha256=recipe_file_sha,
            recipe_canonical_sha256=recipe_canonical_sha,
            result_path=result_path,
            worker_returncode=completed.returncode,
            validation=validation,
            error=error,
        )
    receipt_path = (
        _host_path(args.receipt).expanduser().absolute()
        if args.receipt is not None
        else (output.parent / f"{output.name}-product-e2e-receipt.v1.json")
    )
    atomic_write_json_no_replace(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "passed" else 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except (OSError, ValueError, ProductE2EError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
