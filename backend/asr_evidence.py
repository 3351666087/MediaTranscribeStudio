"""Immutable ASR candidate evidence for semantic text arbitration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .language import normalize_language_tag


ASR_CANDIDATE_SET_SCHEMA_VERSION = "1.0.0"
ASR_MODEL_MANIFEST_NAME = ".mts-model-manifest.json"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID = re.compile(r"^asr-[0-9a-f]{24}$")
_SCORE_STATUSES = {
    "available",
    "provider-unavailable",
    "projection-derived",
}
_CANDIDATE_SET_TYPES = {
    "provider-top1-only",
    "provider-nbest",
    "projection-derived-top1",
}
_MODEL_IDENTITY_STATUSES = {
    "manifest-bound",
    "injected-fixture",
    "unverified-local",
}
ASR_CANDIDATE_SET_KEYS = frozenset(
    {
        "candidateSetSchemaVersion",
        "candidateSetType",
        "candidateSetSha256",
        "modelId",
        "modelRevision",
        "modelManifestSha256",
        "modelIdentityStatus",
        "sourceAudioSha256",
        "normalizationProfile",
        "sourceWindowId",
        "sourceStartMs",
        "sourceEndMs",
        "sourceWindowSha256",
        "nBest",
    }
)


class AsrEvidenceError(ValueError):
    """Raised when immutable ASR candidate evidence is malformed."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _non_empty_text(value: Any, field: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AsrEvidenceError(f"{field} must be non-empty text")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise AsrEvidenceError(f"{field} exceeds its maximum length")
    return normalized


def _sha256_text(value: Any, field: str) -> str:
    normalized = _non_empty_text(value, field, maximum=64)
    if _SHA256.fullmatch(normalized) is None:
        raise AsrEvidenceError(f"{field} must be lowercase SHA-256")
    return normalized


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AsrEvidenceError(f"{field} must be an integer >= {minimum}")
    return value


def _optional_score(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AsrEvidenceError(f"{field} must be a finite number or null")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise AsrEvidenceError(f"{field} must be finite")
    return normalized


def model_identity_from_manifest(
    model_path: str | Path,
    *,
    injected_fixture: bool,
) -> dict[str, str]:
    """Bind evidence to a local manifest without hashing model weights per job."""

    path = Path(model_path) / ASR_MODEL_MANIFEST_NAME
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > _MAX_MANIFEST_BYTES:
            raise AsrEvidenceError("ASR model manifest size is invalid")
        document = json.loads(raw.decode("utf-8"))
        if not isinstance(document, Mapping):
            raise AsrEvidenceError("ASR model manifest must be an object")
        return {
            "modelRevision": _non_empty_text(
                document.get("revision"),
                "ASR model manifest revision",
                maximum=160,
            ),
            "modelManifestSha256": _sha256(raw),
            "modelIdentityStatus": "manifest-bound",
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AsrEvidenceError):
        status = "injected-fixture" if injected_fixture else "unverified-local"
        seed = {
            "modelId": "Qwen3-ASR-1.7B",
            "modelRevision": status,
            "modelIdentityStatus": status,
        }
        return {
            "modelRevision": status,
            "modelManifestSha256": _canonical_sha256(seed),
            "modelIdentityStatus": status,
        }


def source_window_sha256(
    *,
    source_audio_sha256: str,
    normalization_profile: str,
    window_id: str,
    start_ms: int,
    end_ms: int,
) -> str:
    source_sha = _sha256_text(source_audio_sha256, "sourceAudioSha256")
    profile = _non_empty_text(
        normalization_profile,
        "normalizationProfile",
        maximum=160,
    )
    identifier = _non_empty_text(window_id, "sourceWindowId", maximum=200)
    start = _integer(start_ms, "sourceStartMs")
    end = _integer(end_ms, "sourceEndMs", minimum=1)
    if end <= start:
        raise AsrEvidenceError("source window end must be after its start")
    return _canonical_sha256(
        {
            "sourceAudioSha256": source_sha,
            "normalizationProfile": profile,
            "sourceWindowId": identifier,
            "startMs": start,
            "endMs": end,
        }
    )


def _token_unit(tokens: Sequence[Mapping[str, Any]]) -> str:
    lexical = [
        str(item.get("text") or "")
        for item in tokens
        if str(item.get("text") or "").strip()
    ]
    if lexical and all(
        len(item) == 1 and re.search(r"[\u3040-\u30ff\u3400-\u9fff]", item)
        for item in lexical
    ):
        return "character"
    return "token"


def _normalize_tokens(
    raw_tokens: Any,
    *,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    if not isinstance(raw_tokens, Sequence) or isinstance(
        raw_tokens,
        (str, bytes, bytearray),
    ):
        raise AsrEvidenceError("ASR candidate tokens must be an array")
    if len(raw_tokens) > 4096:
        raise AsrEvidenceError("ASR candidate token count exceeds 4096")
    tokens: list[dict[str, Any]] = []
    previous_start = start_ms
    for index, raw in enumerate(raw_tokens):
        if not isinstance(raw, Mapping):
            raise AsrEvidenceError(f"ASR token {index} must be an object")
        text = _non_empty_text(raw.get("text"), f"ASR token {index}.text")
        token_start = _integer(raw.get("startMs"), f"ASR token {index}.startMs")
        token_end = _integer(raw.get("endMs"), f"ASR token {index}.endMs")
        if (
            token_start < start_ms
            or token_end > end_ms
            or token_end < token_start
            or token_start < previous_start
        ):
            raise AsrEvidenceError(
                f"ASR token {index} has invalid or non-monotonic timing"
            )
        tokens.append(
            {
                "index": index,
                "text": text,
                "startMs": token_start,
                "endMs": token_end,
            }
        )
        previous_start = token_start
    return tokens


def _score_status(value: float | None, requested: Any, field: str) -> str:
    status = _non_empty_text(requested, field, maximum=40)
    if status not in _SCORE_STATUSES:
        raise AsrEvidenceError(f"{field} is unsupported")
    if (status == "available") != (value is not None):
        raise AsrEvidenceError(
            f"{field} must be available exactly when its score is numeric"
        )
    return status


def _candidate_body(
    *,
    rank: int,
    raw: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    model_manifest_sha256: str,
    model_identity_status: str,
    source_window_id: str,
    source_window_hash: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    text = _non_empty_text(raw.get("text"), f"ASR candidate {rank}.text", maximum=20000)
    try:
        language = normalize_language_tag(
            raw.get("language", "und"),
            allow_auto=False,
        )
    except ValueError as exc:
        raise AsrEvidenceError(
            f"ASR candidate {rank}.language is invalid"
        ) from exc
    tokens = _normalize_tokens(
        raw.get("tokens", []),
        start_ms=start_ms,
        end_ms=end_ms,
    )
    acoustic_score = _optional_score(
        raw.get("acousticScore"),
        f"ASR candidate {rank}.acousticScore",
    )
    decode_score = _optional_score(
        raw.get("decodeScore"),
        f"ASR candidate {rank}.decodeScore",
    )
    acoustic_status = _score_status(
        acoustic_score,
        raw.get("acousticScoreStatus", "provider-unavailable"),
        f"ASR candidate {rank}.acousticScoreStatus",
    )
    decode_status = _score_status(
        decode_score,
        raw.get("decodeScoreStatus", "provider-unavailable"),
        f"ASR candidate {rank}.decodeScoreStatus",
    )
    parent_id = raw.get("parentCandidateId")
    body: dict[str, Any] = {
        "rank": rank,
        "text": text,
        "language": language,
        "modelId": model_id,
        "modelRevision": model_revision,
        "modelManifestSha256": model_manifest_sha256,
        "sourceWindowId": source_window_id,
        "sourceWindowSha256": source_window_hash,
        "tokenUnit": _token_unit(tokens),
        "tokens": tokens,
        "acousticScore": acoustic_score,
        "acousticScoreStatus": acoustic_status,
        "decodeScore": decode_score,
        "decodeScoreStatus": decode_status,
        "lexicalRepairEligible": (
            model_identity_status == "manifest-bound"
            and bool(tokens)
            and acoustic_status == "available"
            and decode_status == "available"
        ),
    }
    if parent_id is not None:
        body["parentCandidateId"] = _non_empty_text(
            parent_id,
            f"ASR candidate {rank}.parentCandidateId",
            maximum=32,
        )
    return body


def build_asr_candidate_set(
    *,
    model_id: str,
    model_revision: str,
    model_manifest_sha256: str,
    model_identity_status: str,
    source_audio_sha256: str,
    normalization_profile: str,
    source_window_id: str,
    start_ms: int,
    end_ms: int,
    hypotheses: Sequence[Mapping[str, Any]],
    candidate_set_type: str | None = None,
) -> dict[str, Any]:
    """Build a hash-bound, ordered ASR candidate set."""

    model = _non_empty_text(model_id, "modelId", maximum=160)
    revision = _non_empty_text(model_revision, "modelRevision", maximum=160)
    manifest_sha = _sha256_text(
        model_manifest_sha256,
        "modelManifestSha256",
    )
    identity_status = _non_empty_text(
        model_identity_status,
        "modelIdentityStatus",
        maximum=40,
    )
    if identity_status not in _MODEL_IDENTITY_STATUSES:
        raise AsrEvidenceError("modelIdentityStatus is unsupported")
    source_sha = _sha256_text(source_audio_sha256, "sourceAudioSha256")
    profile = _non_empty_text(
        normalization_profile,
        "normalizationProfile",
        maximum=160,
    )
    window_id = _non_empty_text(source_window_id, "sourceWindowId", maximum=200)
    start = _integer(start_ms, "sourceStartMs")
    end = _integer(end_ms, "sourceEndMs", minimum=1)
    if end <= start:
        raise AsrEvidenceError("source window end must be after its start")
    if (
        not isinstance(hypotheses, Sequence)
        or isinstance(hypotheses, (str, bytes, bytearray))
        or not hypotheses
        or len(hypotheses) > 8
        or any(not isinstance(item, Mapping) for item in hypotheses)
    ):
        raise AsrEvidenceError("hypotheses must contain between 1 and 8 objects")
    set_type = candidate_set_type or (
        "provider-nbest" if len(hypotheses) > 1 else "provider-top1-only"
    )
    if set_type not in _CANDIDATE_SET_TYPES:
        raise AsrEvidenceError("candidateSetType is unsupported")
    if set_type in {"provider-top1-only", "projection-derived-top1"}:
        if len(hypotheses) != 1:
            raise AsrEvidenceError(f"{set_type} must contain exactly one candidate")
    elif len(hypotheses) < 2:
        raise AsrEvidenceError("provider-nbest must contain at least two candidates")
    for raw in hypotheses:
        has_parent = raw.get("parentCandidateId") is not None
        if set_type == "projection-derived-top1":
            if (
                not has_parent
                or raw.get("acousticScore") is not None
                or raw.get("decodeScore") is not None
                or raw.get("acousticScoreStatus") != "projection-derived"
                or raw.get("decodeScoreStatus") != "projection-derived"
            ):
                raise AsrEvidenceError(
                    "projection-derived-top1 must preserve a parent candidate "
                    "without pretending to rescore it"
                )
        elif has_parent:
            raise AsrEvidenceError(
                "provider candidate sets cannot declare projected parent candidates"
            )
    window_hash = source_window_sha256(
        source_audio_sha256=source_sha,
        normalization_profile=profile,
        window_id=window_id,
        start_ms=start,
        end_ms=end,
    )
    candidates: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for rank, raw in enumerate(hypotheses, start=1):
        body = _candidate_body(
            rank=rank,
            raw=raw,
            model_id=model,
            model_revision=revision,
            model_manifest_sha256=manifest_sha,
            model_identity_status=identity_status,
            source_window_id=window_id,
            source_window_hash=window_hash,
            start_ms=start,
            end_ms=end,
        )
        if body["text"] in seen_text:
            raise AsrEvidenceError("ASR candidate texts must be unique")
        seen_text.add(str(body["text"]))
        candidates.append(
            {
                "candidateId": "asr-" + _canonical_sha256(body)[:24],
                **body,
            }
        )
    body = {
        "candidateSetSchemaVersion": ASR_CANDIDATE_SET_SCHEMA_VERSION,
        "candidateSetType": set_type,
        "modelId": model,
        "modelRevision": revision,
        "modelManifestSha256": manifest_sha,
        "modelIdentityStatus": identity_status,
        "sourceAudioSha256": source_sha,
        "normalizationProfile": profile,
        "sourceWindowId": window_id,
        "sourceStartMs": start,
        "sourceEndMs": end,
        "sourceWindowSha256": window_hash,
        "nBest": candidates,
    }
    return {
        **body,
        "candidateSetSha256": _canonical_sha256(body),
    }


def project_asr_candidate_set(
    source_evidence: Mapping[str, Any],
    *,
    source_text: str,
    target_text: str,
    target_tokens: Sequence[Mapping[str, Any]],
    target_window_id: str,
    target_start_ms: int,
    target_end_ms: int,
) -> dict[str, Any]:
    """Project top-1 evidence without pretending to have rescored N-best output."""

    validated = validate_asr_candidate_set(
        source_evidence,
        expected_text=source_text,
    )
    parent = validated["nBest"][0]
    projected_tokens: list[dict[str, Any]] = []
    for raw in target_tokens:
        if not isinstance(raw, Mapping):
            raise AsrEvidenceError("projected ASR token must be an object")
        start = _integer(raw.get("startMs"), "projected ASR token.startMs")
        end = _integer(raw.get("endMs"), "projected ASR token.endMs")
        clipped_start = max(target_start_ms, start)
        clipped_end = min(target_end_ms, end)
        if clipped_end < clipped_start:
            raise AsrEvidenceError(
                "projected ASR token does not overlap its target window"
            )
        projected_tokens.append(
            {
                "text": raw.get("text"),
                "startMs": clipped_start,
                "endMs": clipped_end,
            }
        )
    return build_asr_candidate_set(
        model_id=validated["modelId"],
        model_revision=validated["modelRevision"],
        model_manifest_sha256=validated["modelManifestSha256"],
        model_identity_status=validated["modelIdentityStatus"],
        source_audio_sha256=validated["sourceAudioSha256"],
        normalization_profile=validated["normalizationProfile"],
        source_window_id=target_window_id,
        start_ms=target_start_ms,
        end_ms=target_end_ms,
        hypotheses=[
            {
                "text": target_text,
                "language": parent["language"],
                "tokens": projected_tokens,
                "acousticScore": None,
                "acousticScoreStatus": "projection-derived",
                "decodeScore": None,
                "decodeScoreStatus": "projection-derived",
                "parentCandidateId": parent["candidateId"],
            }
        ],
        candidate_set_type="projection-derived-top1",
    )


def validate_asr_candidate_set(
    evidence: Mapping[str, Any],
    *,
    expected_text: str | None = None,
    expected_start_ms: int | None = None,
    expected_end_ms: int | None = None,
) -> dict[str, Any]:
    """Validate persisted evidence and recompute every declared identity."""

    if not isinstance(evidence, Mapping):
        raise AsrEvidenceError("ASR evidence must be an object")
    required = set(ASR_CANDIDATE_SET_KEYS)
    missing = sorted(required - set(evidence))
    if missing:
        raise AsrEvidenceError(
            "ASR candidate evidence is missing: " + ", ".join(missing)
        )
    hypotheses: list[dict[str, Any]] = []
    raw_candidates = evidence.get("nBest")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise AsrEvidenceError("ASR nBest must be a non-empty array")
    for rank, raw in enumerate(raw_candidates, start=1):
        if not isinstance(raw, Mapping):
            raise AsrEvidenceError(f"ASR candidate {rank} must be an object")
        if raw.get("rank") != rank:
            raise AsrEvidenceError("ASR candidate ranks must be contiguous")
        hypotheses.append(
            {
                key: raw.get(key)
                for key in (
                    "text",
                    "language",
                    "tokens",
                    "acousticScore",
                    "acousticScoreStatus",
                    "decodeScore",
                    "decodeScoreStatus",
                    "parentCandidateId",
                )
                if key in raw
            }
        )
    rebuilt = build_asr_candidate_set(
        model_id=evidence.get("modelId"),
        model_revision=evidence.get("modelRevision"),
        model_manifest_sha256=evidence.get("modelManifestSha256"),
        model_identity_status=evidence.get("modelIdentityStatus"),
        source_audio_sha256=evidence.get("sourceAudioSha256"),
        normalization_profile=evidence.get("normalizationProfile"),
        source_window_id=evidence.get("sourceWindowId"),
        start_ms=evidence.get("sourceStartMs"),
        end_ms=evidence.get("sourceEndMs"),
        hypotheses=hypotheses,
        candidate_set_type=evidence.get("candidateSetType"),
    )
    for rank, (actual, canonical) in enumerate(
        zip(raw_candidates, rebuilt["nBest"]),
        start=1,
    ):
        if actual != canonical:
            raise AsrEvidenceError(f"ASR candidate {rank} identity is inconsistent")
        if _CANDIDATE_ID.fullmatch(str(actual.get("candidateId") or "")) is None:
            raise AsrEvidenceError(f"ASR candidate {rank} id is invalid")
    for key in required - {"nBest"}:
        if evidence.get(key) != rebuilt.get(key):
            raise AsrEvidenceError(f"ASR candidate set field {key} is inconsistent")
    if expected_text is not None and raw_candidates[0]["text"] != expected_text:
        raise AsrEvidenceError("ASR top-1 candidate must equal immutable rawText")
    if (
        expected_start_ms is not None
        and evidence["sourceStartMs"] != expected_start_ms
    ):
        raise AsrEvidenceError("ASR sourceStartMs does not match the segment")
    if expected_end_ms is not None and evidence["sourceEndMs"] != expected_end_ms:
        raise AsrEvidenceError("ASR sourceEndMs does not match the segment")
    return dict(rebuilt)


__all__ = [
    "ASR_CANDIDATE_SET_KEYS",
    "ASR_CANDIDATE_SET_SCHEMA_VERSION",
    "ASR_MODEL_MANIFEST_NAME",
    "AsrEvidenceError",
    "build_asr_candidate_set",
    "model_identity_from_manifest",
    "project_asr_candidate_set",
    "source_window_sha256",
    "validate_asr_candidate_set",
]
