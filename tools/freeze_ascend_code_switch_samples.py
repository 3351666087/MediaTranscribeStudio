"""Freeze pinned CAiRE/ASCEND code-switch samples without held-out leakage."""

from __future__ import annotations

import argparse
import ctypes
import errno
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
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


DATASET = "CAiRE/ASCEND"
REVISION = "737e9800ae31be9932ba8464c80366559bd28424"
CONFIG = "main"
LICENSE = "cc-by-sa-4.0"
DEFAULT_SELECTION = (
    PROJECT_ROOT / "sample_library" / "ascend-code-switch-selection.v1.json"
)
PUBLIC_MANIFEST_NAME = "ascend-code-switch-frozen.v1.json"
DEVELOPMENT_REFERENCE_NAME = "development-reference.v1.json"
SCORER_VAULT_NAME = "held-out-truth.v1.json"
USER_AGENT = "MediaTranscribeStudio-ASCEND-freezer/1.0"
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
_STABLE_KEY_FIELDS = {
    "revision",
    "config",
    "split",
    "id",
    "rowIndex",
    "speaker",
    "session",
}
_FORBIDDEN_PUBLIC_TRUTH_KEYS = {
    "expectedtranscript",
    "rawtranscript",
    "referencetranscript",
    "scoringtranscript",
    "text",
    "transcript",
    "transcription",
}
_SOURCE_SPLITS = {"development": "validation", "held-out": "test"}
_DURATION_BUCKETS = {
    "short": (0.0, 3.0),
    "medium": (3.0, 6.0),
    "long": (6.0, None),
}


class AscendCodeSwitchFreezeError(ValueError):
    """Raised when ASCEND evidence cannot be frozen without leakage."""


def _default_eval_root() -> Path:
    configured = os.environ.get("MTS_EVAL_ROOT")
    if configured:
        return Path(configured)
    if os.name == "nt":
        return Path("D:/mts-eval")
    return Path("/mnt/d/mts-eval")


DEFAULT_OUTPUT_ROOT = _default_eval_root() / "ascend-code-switch-v1"
DEFAULT_SCORER_VAULT_ROOT = (
    _default_eval_root() / "scorer-vaults" / "ascend-code-switch-v1"
)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AscendCodeSwitchFreezeError(
                    f"{label} contains duplicate field: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except AscendCodeSwitchFreezeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AscendCodeSwitchFreezeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AscendCodeSwitchFreezeError(f"{label} must contain an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise AscendCodeSwitchFreezeError(
            f"{field} fields are invalid: {'; '.join(details)}"
        )


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AscendCodeSwitchFreezeError(f"{field} must be non-empty text")
    return value.strip()


def _non_negative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AscendCodeSwitchFreezeError(
            f"{field} must be a non-negative integer"
        )
    return value


def _duration_bucket(value: float) -> str:
    if value < 0:
        raise AscendCodeSwitchFreezeError("duration must not be negative")
    if value < 3.0:
        return "short"
    if value < 6.0:
        return "medium"
    return "long"


def assert_truth_redacted(value: Any) -> None:
    """Reject transcript-bearing fields from ordinary/reviewer-safe artifacts."""

    if isinstance(value, Mapping):
        forbidden = sorted(
            key
            for key in value
            if isinstance(key, str)
            and key.casefold() in _FORBIDDEN_PUBLIC_TRUTH_KEYS
        )
        if forbidden:
            raise AscendCodeSwitchFreezeError(
                "truth-redacted artifact contains transcript fields: "
                + ", ".join(forbidden)
            )
        for child in value.values():
            assert_truth_redacted(child)
    elif isinstance(value, list):
        for child in value:
            assert_truth_redacted(child)


def _expected_policy() -> dict[str, Any]:
    return {
        "sourceSplitByEvaluationSplit": dict(_SOURCE_SPLITS),
        "sourceLanguageLabel": "mixed",
        "casesPerEvaluationSplit": 6,
        "durationBuckets": [
            {
                "id": "short",
                "minimumSecondsInclusive": 0.0,
                "maximumSecondsExclusive": 3.0,
                "casesPerEvaluationSplit": 2,
            },
            {
                "id": "medium",
                "minimumSecondsInclusive": 3.0,
                "maximumSecondsExclusive": 6.0,
                "casesPerEvaluationSplit": 2,
            },
            {
                "id": "long",
                "minimumSecondsInclusive": 6.0,
                "maximumSecondsExclusive": None,
                "casesPerEvaluationSplit": 2,
            },
        ],
        "minimumDistinctSpeakersOverall": 3,
        "minimumDistinctSpeakerSessionsPerEvaluationSplit": 3,
        "minimumDistinctTopicsPerEvaluationSplit": 3,
        "speakerDisjointEvaluationSplits": True,
        "heldOutSpeakerLimitation": {
            "availableDistinctSpeakersInTestMixed": 2,
            "selectedDistinctSpeakers": 2,
            "note": (
                "The pinned test/mixed source rows contain only two speaker "
                "identities, so a three-speaker held-out claim is impossible. "
                "Coverage is reported without extrapolation."
            ),
        },
    }


def load_selection_lock(path: Path) -> dict[str, Any]:
    value = _load_json_object(path, label="ASCEND selection lock")
    _exact_fields(
        value,
        {"schemaVersion", "libraryId", "source", "selectionPolicy", "cases"},
        "selection lock",
    )
    if value["schemaVersion"] != "1.0.0":
        raise AscendCodeSwitchFreezeError("selection lock version is unsupported")
    if value["libraryId"] != "mts-caire-ascend-code-switch-v1":
        raise AscendCodeSwitchFreezeError("selection lock libraryId is invalid")
    source = value["source"]
    if not isinstance(source, dict):
        raise AscendCodeSwitchFreezeError("selection lock source must be an object")
    _exact_fields(
        source,
        {"dataset", "revision", "config", "license", "homepage", "attributionPath"},
        "selection lock source",
    )
    expected_source = {
        "dataset": DATASET,
        "revision": REVISION,
        "config": CONFIG,
        "license": LICENSE,
        "homepage": "https://huggingface.co/datasets/CAiRE/ASCEND",
        "attributionPath": "ascend-code-switch/ATTRIBUTION.md",
    }
    if source != expected_source:
        raise AscendCodeSwitchFreezeError("ASCEND source provenance is not pinned")
    if value["selectionPolicy"] != _expected_policy():
        raise AscendCodeSwitchFreezeError("ASCEND selection policy changed")

    rows = value["cases"]
    if not isinstance(rows, list) or len(rows) != 12:
        raise AscendCodeSwitchFreezeError(
            "ASCEND selection must contain exactly twelve cases"
        )
    case_ids: set[str] = set()
    source_rows: set[tuple[str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        field = f"cases[{index}]"
        if not isinstance(row, dict):
            raise AscendCodeSwitchFreezeError(f"{field} must be an object")
        _exact_fields(
            row,
            {
                "id",
                "evaluationSplit",
                "durationBucket",
                "durationSeconds",
                "topic",
                "stableKey",
            },
            field,
        )
        case_id = _non_empty_text(row["id"], f"{field}.id")
        if not _CASE_ID.fullmatch(case_id) or case_id in case_ids:
            raise AscendCodeSwitchFreezeError(f"{field}.id is invalid or duplicate")
        case_ids.add(case_id)
        evaluation_split = row["evaluationSplit"]
        if evaluation_split not in _SOURCE_SPLITS:
            raise AscendCodeSwitchFreezeError(
                f"{field}.evaluationSplit is unsupported"
            )
        stable = row["stableKey"]
        if not isinstance(stable, dict):
            raise AscendCodeSwitchFreezeError(f"{field}.stableKey must be an object")
        _exact_fields(stable, _STABLE_KEY_FIELDS, f"{field}.stableKey")
        if stable["revision"] != REVISION or stable["config"] != CONFIG:
            raise AscendCodeSwitchFreezeError(
                f"{field}.stableKey is not revision/config bound"
            )
        if stable["split"] != _SOURCE_SPLITS[evaluation_split]:
            raise AscendCodeSwitchFreezeError(
                f"{field}.stableKey uses the wrong source split"
            )
        source_id = _non_empty_text(stable["id"], f"{field}.stableKey.id")
        row_index = _non_negative_integer(
            stable["rowIndex"], f"{field}.stableKey.rowIndex"
        )
        _non_negative_integer(stable["speaker"], f"{field}.stableKey.speaker")
        _non_negative_integer(stable["session"], f"{field}.stableKey.session")
        locator = (stable["split"], row_index)
        if locator in source_rows:
            raise AscendCodeSwitchFreezeError("ASCEND source rows must be unique")
        source_rows.add(locator)
        if not source_id.isdigit():
            raise AscendCodeSwitchFreezeError(f"{field}.stableKey.id is invalid")
        duration = row["durationSeconds"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or float(duration) <= 0
        ):
            raise AscendCodeSwitchFreezeError(
                f"{field}.durationSeconds must be positive"
            )
        bucket = _non_empty_text(row["durationBucket"], f"{field}.durationBucket")
        if bucket != _duration_bucket(float(duration)):
            raise AscendCodeSwitchFreezeError(
                f"{field}.durationBucket does not match duration"
            )
        _non_empty_text(row["topic"], f"{field}.topic")
        normalized.append(dict(row))

    counts = Counter(row["evaluationSplit"] for row in normalized)
    if counts != Counter({"development": 6, "held-out": 6}):
        raise AscendCodeSwitchFreezeError(
            "ASCEND selection must contain six cases per evaluation split"
        )
    for evaluation_split in _SOURCE_SPLITS:
        selected = [
            row for row in normalized if row["evaluationSplit"] == evaluation_split
        ]
        bucket_counts = Counter(row["durationBucket"] for row in selected)
        if bucket_counts != Counter({"short": 2, "medium": 2, "long": 2}):
            raise AscendCodeSwitchFreezeError(
                f"{evaluation_split} must contain two short, medium, and long cases"
            )
        if len({row["topic"] for row in selected}) < 3:
            raise AscendCodeSwitchFreezeError(
                f"{evaluation_split} must cover at least three topics"
            )
        if (
            len(
                {
                    (row["stableKey"]["speaker"], row["stableKey"]["session"])
                    for row in selected
                }
            )
            < 3
        ):
            raise AscendCodeSwitchFreezeError(
                f"{evaluation_split} must cover at least three speaker-sessions"
            )
    speakers_by_split = {
        evaluation_split: {
            row["stableKey"]["speaker"]
            for row in normalized
            if row["evaluationSplit"] == evaluation_split
        }
        for evaluation_split in _SOURCE_SPLITS
    }
    if len(set.union(*speakers_by_split.values())) < 3:
        raise AscendCodeSwitchFreezeError(
            "ASCEND selection must cover at least three speakers overall"
        )
    if speakers_by_split["development"] & speakers_by_split["held-out"]:
        raise AscendCodeSwitchFreezeError(
            "ASCEND speaker identities cross evaluation splits"
        )
    if len(speakers_by_split["held-out"]) != 2:
        raise AscendCodeSwitchFreezeError(
            "held-out selection must represent both available test/mixed speakers"
        )
    assert_truth_redacted(value)
    return value


def _request_bytes(
    url: str,
    *,
    label: str,
    timeout: float = 240.0,
    attempts: int = 5,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise AscendCodeSwitchFreezeError(
        f"remote request failed for {label} after {attempts} attempts: "
        f"{type(last_error).__name__}"
    )


def _request_json(url: str, *, label: str, timeout: float = 120.0) -> dict[str, Any]:
    payload = _request_bytes(url, label=label, timeout=timeout)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AscendCodeSwitchFreezeError(
            f"remote JSON is invalid for {label}"
        ) from exc
    if not isinstance(value, dict):
        raise AscendCodeSwitchFreezeError(
            f"remote JSON is not an object for {label}"
        )
    return value


def viewer_evidence() -> dict[str, Any]:
    encoded_dataset = urllib.parse.quote(DATASET, safe="")
    valid = _request_json(
        "https://datasets-server.huggingface.co/is-valid?dataset="
        + encoded_dataset,
        label="ASCEND Dataset Viewer validity",
    )
    if valid.get("viewer") is not True:
        raise AscendCodeSwitchFreezeError("ASCEND Dataset Viewer is unavailable")
    splits = _request_json(
        "https://datasets-server.huggingface.co/splits?dataset="
        + encoded_dataset,
        label="ASCEND Dataset Viewer splits",
    )
    rows = splits.get("splits")
    if not isinstance(rows, list):
        raise AscendCodeSwitchFreezeError("ASCEND Viewer split evidence is invalid")
    available = {
        (row.get("config"), row.get("split"))
        for row in rows
        if isinstance(row, dict) and row.get("dataset") == DATASET
    }
    required = {(CONFIG, "validation"), (CONFIG, "test")}
    if not required <= available:
        raise AscendCodeSwitchFreezeError(
            "ASCEND Viewer is missing validation or test"
        )
    return {
        "baseUrl": "https://datasets-server.huggingface.co",
        "dataset": DATASET,
        "viewer": True,
        "selectedConfigSplits": [
            {"config": CONFIG, "split": "validation"},
            {"config": CONFIG, "split": "test"},
        ],
        "assetRevisionValidation": "required-per-row",
        "pinnedRevision": REVISION,
    }


def _viewer_row(case: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    stable = case["stableKey"]
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": stable["split"],
            "offset": stable["rowIndex"],
            "length": 1,
        }
    )
    response = _request_json(
        f"https://datasets-server.huggingface.co/rows?{query}",
        label=f"ASCEND source row {case['id']}",
        timeout=180.0,
    )
    rows = response.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise AscendCodeSwitchFreezeError(
            f"{case['id']} did not resolve exactly one Viewer row"
        )
    wrapped = rows[0]
    if (
        not isinstance(wrapped, dict)
        or wrapped.get("row_idx") != stable["rowIndex"]
        or not isinstance(wrapped.get("row"), dict)
    ):
        raise AscendCodeSwitchFreezeError(
            f"{case['id']} Viewer row identity is invalid"
        )
    row = wrapped["row"]
    expected_values = {
        "id": stable["id"],
        "language": "mixed",
        "original_speaker_id": stable["speaker"],
        "session_id": stable["session"],
        "topic": case["topic"],
    }
    if any(row.get(key) != expected for key, expected in expected_values.items()):
        raise AscendCodeSwitchFreezeError(
            f"{case['id']} source metadata changed from the selection lock"
        )
    duration = row.get("duration")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or abs(float(duration) - float(case["durationSeconds"])) > 0.001
    ):
        raise AscendCodeSwitchFreezeError(
            f"{case['id']} source duration changed from the selection lock"
        )
    transcription = row.get("transcription")
    if not isinstance(transcription, str) or not transcription.strip():
        raise AscendCodeSwitchFreezeError(
            f"{case['id']} source transcription is unavailable"
        )
    audio = row.get("audio")
    if (
        not isinstance(audio, list)
        or len(audio) != 1
        or not isinstance(audio[0], dict)
        or not isinstance(audio[0].get("src"), str)
    ):
        raise AscendCodeSwitchFreezeError(f"{case['id']} audio asset is unavailable")
    asset_url = str(audio[0]["src"])
    decoded_url = urllib.parse.unquote(asset_url)
    revision_marker = (
        f"/{REVISION}/--/{CONFIG}/{stable['split']}/"
        f"{stable['rowIndex']}/audio/"
    )
    if revision_marker not in decoded_url:
        raise AscendCodeSwitchFreezeError(
            f"{case['id']} audio asset is not bound to the pinned revision"
        )
    return row, transcription, asset_url


def _wav_evidence(payload: bytes, *, case: Mapping[str, Any]) -> dict[str, Any]:
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
    except (EOFError, wave.Error) as exc:
        raise AscendCodeSwitchFreezeError(
            f"{case['id']} downloaded audio is not a valid WAV"
        ) from exc
    if (
        channels < 1
        or sample_width < 1
        or sample_rate <= 0
        or frame_count <= 0
        or compression != "NONE"
    ):
        raise AscendCodeSwitchFreezeError(
            f"{case['id']} WAV format is unsupported"
        )
    duration = frame_count / sample_rate
    if abs(duration - float(case["durationSeconds"])) > 0.05:
        raise AscendCodeSwitchFreezeError(
            f"{case['id']} WAV duration does not match source metadata"
        )
    return {
        "durationSeconds": round(duration, 6),
        "sampleRateHz": sample_rate,
        "channels": channels,
        "sampleWidthBytes": sample_width,
        "frameCount": frame_count,
    }


def _write_new_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        path.chmod(mode)
    except OSError:
        if os.name != "nt":
            raise


def _canonical_artifact(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "canonicalSha256": canonical_json_sha256(body)}


def _coverage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for evaluation_split in _SOURCE_SPLITS:
        selected = [
            case for case in cases if case["evaluationSplit"] == evaluation_split
        ]
        by_split[evaluation_split] = {
            "cases": len(selected),
            "distinctSpeakers": len(
                {case["sourceKey"]["speaker"] for case in selected}
            ),
            "distinctSpeakerSessions": len(
                {
                    (
                        case["sourceKey"]["speaker"],
                        case["sourceKey"]["session"],
                    )
                    for case in selected
                }
            ),
            "distinctTopics": len({case["topic"] for case in selected}),
            "durationBuckets": {
                bucket: sum(case["durationBucket"] == bucket for case in selected)
                for bucket in _DURATION_BUCKETS
            },
        }
    development_speakers = {
        case["sourceKey"]["speaker"]
        for case in cases
        if case["evaluationSplit"] == "development"
    }
    held_out_speakers = {
        case["sourceKey"]["speaker"]
        for case in cases
        if case["evaluationSplit"] == "held-out"
    }
    return {
        "cases": len(cases),
        "distinctSpeakersOverall": len(development_speakers | held_out_speakers),
        "speakerDisjointEvaluationSplits": not bool(
            development_speakers & held_out_speakers
        ),
        "byEvaluationSplit": by_split,
    }


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    if os.name == "nt":
        os.rename(source, target)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise AscendCodeSwitchFreezeError(
                "atomic no-replace directory rename is unavailable"
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, os.strerror(error_number), target)
        raise OSError(error_number, os.strerror(error_number), target)
    raise AscendCodeSwitchFreezeError(
        "atomic no-replace directory rename is unsupported on this platform"
    )


def _assert_separate_roots(output_root: Path, scorer_vault_root: Path) -> None:
    output = output_root.resolve()
    vault = scorer_vault_root.resolve()
    if output == vault or output.is_relative_to(vault) or vault.is_relative_to(output):
        raise AscendCodeSwitchFreezeError(
            "scorer vault must be outside the ordinary freeze root"
        )


def freeze(
    *,
    selection_path: Path,
    output_root: Path,
    scorer_vault_root: Path,
) -> dict[str, Any]:
    selection_path = selection_path.resolve(strict=True)
    output_root = output_root.resolve()
    scorer_vault_root = scorer_vault_root.resolve()
    _assert_separate_roots(output_root, scorer_vault_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    if scorer_vault_root.exists():
        raise FileExistsError(scorer_vault_root)
    selection = load_selection_lock(selection_path)
    attribution_source = (
        selection_path.parent / str(selection["source"]["attributionPath"])
    ).resolve(strict=True)
    evidence = viewer_evidence()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    scorer_vault_root.parent.mkdir(parents=True, exist_ok=True)
    public_stage = Path(
        tempfile.mkdtemp(prefix=".ascend-public-", dir=output_root.parent)
    )
    vault_stage = Path(
        tempfile.mkdtemp(prefix=".ascend-vault-", dir=scorer_vault_root.parent)
    )
    published_vault = False
    try:
        media_root = public_stage / "media"
        media_root.mkdir()
        public_cases: list[dict[str, Any]] = []
        development_truth: list[dict[str, Any]] = []
        held_out_truth: list[dict[str, Any]] = []
        for case in selection["cases"]:
            _, transcription, asset_url = _viewer_row(case)
            audio_payload = _request_bytes(
                asset_url,
                label=f"ASCEND audio {case['id']}",
                timeout=300.0,
            )
            wav = _wav_evidence(audio_payload, case=case)
            media_path = media_root / f"{case['id']}.wav"
            _write_new_bytes(media_path, audio_payload)
            audio_sha256 = sha256_file(media_path)
            stable_key = dict(case["stableKey"])
            public_cases.append(
                {
                    "id": case["id"],
                    "sourceId": "caire-ascend",
                    "evaluationSplit": case["evaluationSplit"],
                    "languageTags": ["zh", "en"],
                    "sourceLanguageLabel": "mixed",
                    "region": "East Asia",
                    "scenarios": [
                        "real-recording",
                        "single-speaker",
                        "zh-en",
                        "intra-utterance-code-switching",
                    ],
                    "expectedSpeakerCount": 1,
                    "topic": case["topic"],
                    "durationBucket": case["durationBucket"],
                    "sourceKey": stable_key,
                    "stableKeySha256": canonical_json_sha256(stable_key),
                    "media": {
                        "path": media_path.relative_to(public_stage).as_posix(),
                        "bytes": len(audio_payload),
                        "sha256": audio_sha256,
                        **wav,
                    },
                    "tuningEligible": case["evaluationSplit"] == "development",
                    "truthAccess": (
                        "development-reference"
                        if case["evaluationSplit"] == "development"
                        else "isolated-scorer-vault-only"
                    ),
                }
            )
            truth_row = {
                "id": case["id"],
                "sourceKey": stable_key,
                "mediaSha256": audio_sha256,
                "referenceTranscript": transcription,
            }
            if case["evaluationSplit"] == "development":
                development_truth.append(truth_row)
            else:
                held_out_truth.append(truth_row)

        if len(development_truth) != 6 or len(held_out_truth) != 6:
            raise AscendCodeSwitchFreezeError("ASCEND truth split count changed")
        generated_at = datetime.now(UTC).isoformat()
        development_body = _canonical_artifact(
            {
                "schemaVersion": "1.0.0",
                "artifactType": "ascend-code-switch-development-reference",
                "generatedAt": generated_at,
                "source": dict(selection["source"]),
                "evaluationSplit": "development",
                "sourceSplit": "validation",
                "tuningEligible": True,
                "cases": development_truth,
            }
        )
        development_path = public_stage / "reference" / DEVELOPMENT_REFERENCE_NAME
        atomic_write_json(development_path, development_body)

        vault_body = _canonical_artifact(
            {
                "schemaVersion": "1.0.0",
                "artifactType": "ascend-code-switch-held-out-scorer-vault",
                "generatedAt": generated_at,
                "source": dict(selection["source"]),
                "evaluationSplit": "held-out",
                "sourceSplit": "test",
                "tuningEligible": False,
                "ordinaryManifestEligible": False,
                "reviewerPacketEligible": False,
                "cases": held_out_truth,
            }
        )
        try:
            vault_stage.chmod(0o700)
        except OSError:
            if os.name != "nt":
                raise
        vault_path = vault_stage / SCORER_VAULT_NAME
        atomic_write_json(vault_path, vault_body)
        try:
            vault_path.chmod(0o600)
        except OSError:
            if os.name != "nt":
                raise

        attribution_path = public_stage / "ATTRIBUTION.md"
        _write_new_bytes(attribution_path, attribution_source.read_bytes())
        manifest_body = {
            "schemaVersion": "1.0.0",
            "artifactType": "ascend-code-switch-truth-redacted-freeze",
            "libraryId": selection["libraryId"],
            "generatedAt": generated_at,
            "sourceSelectionLock": {
                "path": str(selection_path),
                "fileSha256": sha256_file(selection_path),
            },
            "source": dict(selection["source"]),
            "viewerEvidence": evidence,
            "selectionPolicy": dict(selection["selectionPolicy"]),
            "truthPersistencePolicy": {
                "ordinaryManifestContainsTranscript": False,
                "developmentReferencePersistedSeparately": True,
                "developmentReferenceMayEnterReviewerPacket": False,
                "heldOutTruthPersistedOnlyInIsolatedScorerVault": True,
                "heldOutTruthInOrdinaryManifest": False,
                "heldOutTruthInReviewerPacket": False,
                "scorerVaultPublishedBeforeOrdinaryManifest": True,
            },
            "developmentReference": {
                "path": development_path.relative_to(public_stage).as_posix(),
                "fileSha256": sha256_file(development_path),
                "canonicalSha256": development_body["canonicalSha256"],
                "reviewerPacketEligible": False,
            },
            "attribution": {
                "path": attribution_path.relative_to(public_stage).as_posix(),
                "fileSha256": sha256_file(attribution_path),
            },
            "coverage": _coverage(public_cases),
            "cases": public_cases,
            "publication": {
                "policy": "atomic-directory-no-replace",
                "manifestWrittenLast": True,
            },
        }
        assert_truth_redacted(manifest_body)
        public_manifest = _canonical_artifact(manifest_body)
        manifest_path = public_stage / PUBLIC_MANIFEST_NAME
        atomic_write_json(manifest_path, public_manifest)

        _rename_directory_no_replace(vault_stage, scorer_vault_root)
        published_vault = True
        _rename_directory_no_replace(public_stage, output_root)
    finally:
        if public_stage.exists():
            shutil.rmtree(public_stage, ignore_errors=True)
        if vault_stage.exists():
            shutil.rmtree(vault_stage, ignore_errors=True)

    final_manifest = output_root / PUBLIC_MANIFEST_NAME
    final_vault = scorer_vault_root / SCORER_VAULT_NAME
    if not published_vault or not final_manifest.is_file() or not final_vault.is_file():
        raise AscendCodeSwitchFreezeError("ASCEND freeze publication is incomplete")
    return {
        "outputRoot": str(output_root),
        "manifest": str(final_manifest),
        "manifestFileSha256": sha256_file(final_manifest),
        "manifestCanonicalSha256": public_manifest["canonicalSha256"],
        "scorerVaultRoot": str(scorer_vault_root),
        "scorerVaultWritten": True,
        "developmentCases": 6,
        "heldOutCases": 6,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--scorer-vault-root",
        type=Path,
        default=DEFAULT_SCORER_VAULT_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = freeze(
        selection_path=args.selection,
        output_root=args.output_root,
        scorer_vault_root=args.scorer_vault_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
