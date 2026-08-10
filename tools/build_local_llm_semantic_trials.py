"""Build recording-isolated Chinese semantic trials from real TextGrids."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.asr_evidence import (  # noqa: E402
    ASR_CANDIDATE_SET_KEYS,
    build_asr_candidate_set,
    validate_asr_candidate_set,
)
from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)


SCHEMA_VERSION = "1.2.0"
ARTIFACT_TYPE = "local-llm-semantic-trials"
PRAATIO_VERSION = "6.2.0"
GENERATOR_ID = "deterministic-evaluation-generator"
GENERATOR_REVISION = "semantic-chinese-context-v3-symmetric-timeline-candidates"
NORMALIZATION_PROFILE = "textgrid-reference-context-v3"
DEFAULT_EVAL_ROOT = Path("D:/mts-eval")
DEFAULT_OUTPUT = (
    DEFAULT_EVAL_ROOT
    / "semantic-llm-v1"
    / "local-llm-semantic-trials.v2.json"
)
DEFAULT_DEPENDENCY_PATH = PROJECT_ROOT / ".toolchain" / "semantic-eval"
CASE_CATEGORIES = (
    "lexical-protected-preservation",
    "text-correction",
    "speaker-continuity-correction",
    "speaker-boundary-control",
)
CORPORA = ("AliMeeting", "AISHELL-4")
SPLITS = ("development", "held-out")
GENERATOR_CONTRACT = {
    "schemaVersion": "1.0.0",
    "id": GENERATOR_ID,
    "revision": GENERATOR_REVISION,
    "purpose": "semantic-evaluation-only",
    "candidateSource": "deterministic-transform-of-licensed-reference-text",
    "scoreSemantics": "deterministic-evaluation-rank-not-acoustic-confidence",
    "productionModelClaimed": False,
}
GENERATOR_MANIFEST_SHA256 = canonical_json_sha256(GENERATOR_CONTRACT)
_CJK = re.compile(r"[\u3400-\u9fff]")
_ANNOTATION = re.compile(r"<[^>]*>")
_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?")
_NEUTRAL_TURN_ID = re.compile(r"^turn-[0-9]{4}$")
_CURRENT_TIMELINE_PROVIDER_ID = "deterministic-evaluation-current-state"
_ALTERNATE_TIMELINE_PROVIDER_ID = "deterministic-evaluation-top2-timeline"
_ALTERNATE_TIMELINE_PROVIDER_VERSION = "top2-symmetric-v1"
_NEAR_CANDIDATE_HIGH_SCORE = 0.51
_NEAR_CANDIDATE_LOW_SCORE = 0.49
_PROTECTED_LITERALS = (
    "\u4e0d\u8981",
    "\u4e0d\u80fd",
    "\u4e0d\u4f1a",
    "\u4e0d\u662f",
    "\u6ca1\u6709",
    "\u7981\u6b62",
    "\u6ca1",
    "\u65e0",
    "\u672a",
    "\u975e",
    "\u4e0d",
)
_CORRECTIONS = (
    ("\u6211\u4eec", "\u6211\u95e8"),
    ("\u4ea7\u54c1", "\u4ea7\u5e73"),
    ("\u95ee\u9898", "\u95ee\u63d0"),
    ("\u53ef\u4ee5", "\u53ef\u5df2"),
    ("\u7136\u540e", "\u71c3\u540e"),
    ("\u89c9\u5f97", "\u51b3\u5f97"),
    ("\u56e0\u4e3a", "\u5e94\u4e3a"),
    ("\u73b0\u5728", "\u73b0\u518d"),
    ("\u8fd9\u4e2a", "\u8fd9\u5404"),
    ("\u7684", "\u5730"),
    ("\u5728", "\u518d"),
    ("\u662f", "\u4e8b"),
    ("\u6709", "\u53c8"),
)
_ARCHIVE_EVIDENCE = {
    "AliMeeting": {
        "openSlrId": "SLR119",
        "openSlrPage": "https://www.openslr.org/119/",
        "archiveUrl": (
            "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/"
            "AliMeeting/openlr/Eval_Ali.tar.gz"
        ),
        "archiveRelativePath": "source-archives/AliMeeting/Eval_Ali.tar.gz",
        "archiveBytes": 3_673_718_355,
        "archiveSha256": (
            "dc47343b2474b5ebcf458927e878155f6ddeb59c85e685b3645c32a1f9578d92"
        ),
        "retrievedAt": "2026-08-07",
        "license": "CC-BY-SA-4.0",
    },
    "AISHELL-4": {
        "openSlrId": "SLR111",
        "openSlrPage": "https://www.openslr.org/111/",
        "archiveUrl": "https://openslr.trmal.net/resources/111/test.tar.gz",
        "archiveRelativePath": "source-archives/AISHELL-4/test.tar.gz",
        "archiveBytes": 5_241_010_904,
        "archiveSha256": (
            "7e5d306b5f18ab66fcd7e0380c90979b47fd9576bfa8e67e6353bdec7c14a35a"
        ),
        "retrievedAt": "2026-08-07",
        "license": "CC-BY-SA-4.0",
    },
}


class SemanticTrialError(ValueError):
    """Raised when frozen semantic trial evidence is incomplete."""


@dataclass(frozen=True)
class Utterance:
    corpus: str
    recording_id: str
    speaker_label: str
    start_ms: int
    end_ms: int
    text: str

    @property
    def source_key(self) -> str:
        return (
            f"{self.corpus}:{self.recording_id}:{self.speaker_label}:"
            f"{self.start_ms}:{self.end_ms}"
        )


@dataclass
class Recording:
    corpus: str
    recording_id: str
    textgrid_path: Path
    audio_path: Path
    duration_ms: int
    utterances: tuple[Utterance, ...]
    split: str = ""
    textgrid_sha256: str = ""
    audio_sha256: str = ""


@dataclass(frozen=True)
class CaseSpec:
    recording: Recording
    target_index: int
    category: str
    variant_text: str | None
    competitor_label: str | None

    @property
    def anchor_key(self) -> str:
        return self.recording.utterances[self.target_index].source_key


def normalize_label(value: str) -> str:
    text = unicodedata.normalize("NFC", value)
    text = _ANNOTATION.sub("", text)
    return _SPACE.sub(" ", text).strip()


def protected_inventory(text: str) -> dict[str, int]:
    inventory: Counter[str] = Counter()
    occupied: list[tuple[int, int]] = []
    for literal in _PROTECTED_LITERALS:
        start = 0
        while True:
            index = text.find(literal, start)
            if index < 0:
                break
            end = index + len(literal)
            if not any(left < end and index < right for left, right in occupied):
                inventory[literal] += 1
                occupied.append((index, end))
            start = index + len(literal)
    inventory.update(match.group(0) for match in _NUMBER.finditer(text))
    return {key: inventory[key] for key in sorted(inventory)}


def make_text_correction_candidate(text: str) -> str | None:
    protected = protected_inventory(text)
    for source, replacement in _CORRECTIONS:
        if source == replacement or source not in text:
            continue
        candidate = text.replace(source, replacement, 1)
        if candidate != text and protected_inventory(candidate) == protected:
            return candidate
    return None


def make_unsafe_protected_candidate(text: str) -> str | None:
    for literal in _PROTECTED_LITERALS:
        if literal in text:
            candidate = text.replace(literal, "", 1).strip()
            if candidate and candidate != text:
                return candidate
    match = _NUMBER.search(text)
    if match is not None:
        raw = match.group(0)
        digits = re.match(r"\d+", raw)
        if digits is not None:
            replacement = str(int(digits.group(0)) + 1) + raw[len(digits.group(0)) :]
            return text[: match.start()] + replacement + text[match.end() :]
    return None


def assign_recording_splits(
    recording_ids_by_corpus: Mapping[str, Sequence[str]],
) -> dict[tuple[str, str], str]:
    assignments: dict[tuple[str, str], str] = {}
    for corpus in sorted(recording_ids_by_corpus):
        identifiers = sorted(set(recording_ids_by_corpus[corpus]))
        if len(identifiers) < 2:
            raise SemanticTrialError(
                f"{corpus} needs at least two recordings for isolation"
            )
        ordered = sorted(
            identifiers,
            key=lambda recording_id: hashlib.sha256(
                f"semantic-recording-split-v1:{corpus}:{recording_id}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        held_out_count = max(1, len(ordered) // 4)
        held_out = set(ordered[:held_out_count])
        for recording_id in identifiers:
            assignments[(corpus, recording_id)] = (
                "held-out" if recording_id in held_out else "development"
            )
    return assignments


def _load_praatio(dependency_path: Path | None) -> Any:
    if dependency_path is not None:
        resolved = dependency_path.resolve(strict=True)
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
    try:
        from praatio import textgrid

        version = importlib.metadata.version("praatio")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise SemanticTrialError(
            "praatio==6.2.0 is required; install requirements-semantic-eval.txt"
        ) from exc
    if version != PRAATIO_VERSION:
        raise SemanticTrialError(
            f"praatio version must be {PRAATIO_VERSION}, got {version}"
        )
    return textgrid


def _parse_recording(
    *,
    corpus: str,
    recording_id: str,
    textgrid_path: Path,
    audio_path: Path,
    textgrid_module: Any,
) -> Recording:
    grid = textgrid_module.openTextgrid(
        str(textgrid_path.resolve(strict=True)),
        includeEmptyIntervals=False,
        reportingMode="error",
    )
    utterances: list[Utterance] = []
    for tier_name in grid.tierNames:
        tier = grid.getTier(tier_name)
        for entry in tier.entries:
            text = normalize_label(str(entry.label))
            start_ms = round(float(entry.start) * 1000.0)
            end_ms = round(float(entry.end) * 1000.0)
            if (
                not text
                or _CJK.search(text) is None
                or end_ms <= start_ms
                or end_ms - start_ms > 30_000
            ):
                continue
            utterances.append(
                Utterance(
                    corpus=corpus,
                    recording_id=recording_id,
                    speaker_label=str(tier_name),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                )
            )
    utterances.sort(
        key=lambda item: (
            item.start_ms,
            item.end_ms,
            item.speaker_label,
            item.text,
        )
    )
    if len(utterances) < 20:
        raise SemanticTrialError(f"too few Chinese utterances in {textgrid_path}")
    return Recording(
        corpus=corpus,
        recording_id=recording_id,
        textgrid_path=textgrid_path.resolve(),
        audio_path=audio_path.resolve(strict=True),
        duration_ms=round(float(grid.maxTimestamp) * 1000.0),
        utterances=tuple(utterances),
    )


def discover_recordings(
    eval_root: Path,
    *,
    dependency_path: Path | None = DEFAULT_DEPENDENCY_PATH,
) -> list[Recording]:
    textgrid_module = _load_praatio(dependency_path)
    corpus_root = eval_root / "source-corpora"
    alimeeting = (
        corpus_root
        / "AliMeeting"
        / "Eval_Ali"
        / "Eval_Ali_far"
    )
    recordings: list[Recording] = []
    for grid_path in sorted((alimeeting / "textgrid_dir").glob("*.TextGrid")):
        matches = sorted((alimeeting / "audio_dir").glob(f"{grid_path.stem}_*.wav"))
        if len(matches) != 1:
            raise SemanticTrialError(
                f"AliMeeting audio mapping is ambiguous for {grid_path.name}"
            )
        recordings.append(
            _parse_recording(
                corpus="AliMeeting",
                recording_id=grid_path.stem,
                textgrid_path=grid_path,
                audio_path=matches[0],
                textgrid_module=textgrid_module,
            )
        )
    aishell = corpus_root / "AISHELL-4" / "test"
    for grid_path in sorted((aishell / "TextGrid").glob("*.TextGrid")):
        audio_path = aishell / "wav" / f"{grid_path.stem}.flac"
        recordings.append(
            _parse_recording(
                corpus="AISHELL-4",
                recording_id=grid_path.stem,
                textgrid_path=grid_path,
                audio_path=audio_path,
                textgrid_module=textgrid_module,
            )
        )
    counts = Counter(item.corpus for item in recordings)
    if any(counts[corpus] < 2 for corpus in CORPORA):
        raise SemanticTrialError("both corpora need multiple parsed recordings")
    assignments = assign_recording_splits(
        {
            corpus: [
                item.recording_id
                for item in recordings
                if item.corpus == corpus
            ]
            for corpus in CORPORA
        }
    )
    for recording in recordings:
        recording.split = assignments[(recording.corpus, recording.recording_id)]
    return sorted(recordings, key=lambda item: (item.corpus, item.recording_id))


def _context(recording: Recording, target_index: int) -> tuple[Utterance, ...] | None:
    if target_index < 2 or target_index + 2 >= len(recording.utterances):
        return None
    context = recording.utterances[target_index - 2 : target_index + 3]
    if any(
        max(0, right.start_ms - left.end_ms) > 15_000
        for left, right in zip(context, context[1:])
    ):
        return None
    if sum(len(item.text) for item in context) > 500:
        return None
    return context


def candidate_specs(recording: Recording, category: str) -> list[CaseSpec]:
    if category not in CASE_CATEGORIES:
        raise SemanticTrialError(f"unsupported case category: {category}")
    candidates: list[CaseSpec] = []
    for target_index, target in enumerate(recording.utterances):
        context = _context(recording, target_index)
        if context is None:
            continue
        cjk_count = len(_CJK.findall(target.text))
        if cjk_count < 6 or cjk_count > 90:
            continue
        labels = [item.speaker_label for item in context]
        if len(set(labels)) < 2:
            continue
        variant: str | None = None
        competitor: str | None = None
        if category == "text-correction":
            variant = make_text_correction_candidate(target.text)
            if variant is None:
                continue
        elif category == "lexical-protected-preservation":
            variant = make_unsafe_protected_candidate(target.text)
            if variant is None:
                continue
        elif category == "speaker-continuity-correction":
            if (
                context[1].speaker_label != target.speaker_label
                and context[3].speaker_label != target.speaker_label
            ):
                continue
            competitors = sorted(
                {label for label in labels if label != target.speaker_label}
            )
            if not competitors:
                continue
            competitor = competitors[0]
        else:
            adjacent = [context[1].speaker_label, context[3].speaker_label]
            competitors = [
                label for label in adjacent if label != target.speaker_label
            ]
            if not competitors or labels.count(target.speaker_label) < 2:
                continue
            competitor = competitors[0]
        candidates.append(
            CaseSpec(
                recording=recording,
                target_index=target_index,
                category=category,
                variant_text=variant,
                competitor_label=competitor,
            )
        )
    return sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"{category}:{item.anchor_key}".encode("utf-8")
        ).hexdigest(),
    )


def _balanced_select(
    candidates: Sequence[CaseSpec],
    *,
    count: int,
    used_anchors: set[str],
    salt: str,
) -> list[CaseSpec]:
    by_recording: dict[str, deque[CaseSpec]] = {}
    grouped: defaultdict[str, list[CaseSpec]] = defaultdict(list)
    for candidate in candidates:
        if candidate.anchor_key not in used_anchors:
            grouped[candidate.recording.recording_id].append(candidate)
    for recording_id, values in grouped.items():
        by_recording[recording_id] = deque(
            sorted(
                values,
                key=lambda item: hashlib.sha256(
                    f"{salt}:{item.anchor_key}".encode("utf-8")
                ).hexdigest(),
            )
        )
    recording_order = sorted(
        by_recording,
        key=lambda recording_id: hashlib.sha256(
            f"{salt}:{recording_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[CaseSpec] = []
    while len(selected) < count:
        progress = False
        for recording_id in recording_order:
            queue = by_recording[recording_id]
            while queue and queue[0].anchor_key in used_anchors:
                queue.popleft()
            if not queue:
                continue
            item = queue.popleft()
            selected.append(item)
            used_anchors.add(item.anchor_key)
            progress = True
            if len(selected) == count:
                break
        if not progress:
            raise SemanticTrialError(
                f"only {len(selected)} of {count} balanced candidates "
                f"available for {salt}"
            )
    return selected


def select_case_specs(
    recordings: Sequence[Recording],
    *,
    cases_per_category: int,
) -> list[CaseSpec]:
    if (
        isinstance(cases_per_category, bool)
        or cases_per_category < 24
        or cases_per_category > 48
        or cases_per_category % 8 != 0
    ):
        raise SemanticTrialError(
            "cases_per_category must be a multiple of 8 between 24 and 48"
        )
    held_per_corpus = cases_per_category // 8
    development_per_corpus = cases_per_category // 2 - held_per_corpus
    used_anchors: set[str] = set()
    selected: list[CaseSpec] = []
    for category in CASE_CATEGORIES:
        all_candidates = [
            candidate
            for recording in recordings
            for candidate in candidate_specs(recording, category)
        ]
        for corpus in CORPORA:
            for split in SPLITS:
                quota = (
                    development_per_corpus
                    if split == "development"
                    else held_per_corpus
                )
                pool = [
                    item
                    for item in all_candidates
                    if item.recording.corpus == corpus
                    and item.recording.split == split
                ]
                selected.extend(
                    _balanced_select(
                        pool,
                        count=quota,
                        used_anchors=used_anchors,
                        salt=f"{category}:{corpus}:{split}",
                    )
                )
    return sorted(
        selected,
        key=lambda item: (
            item.recording.split,
            item.category,
            item.recording.corpus,
            item.recording.recording_id,
            item.anchor_key,
        ),
    )


def _tokens(text: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    units = [character for character in text if not character.isspace()]
    if not units:
        return []
    duration = end_ms - start_ms
    return [
        {
            "text": unit,
            "startMs": start_ms + math.floor(index * duration / len(units)),
            "endMs": start_ms + math.floor((index + 1) * duration / len(units)),
        }
        for index, unit in enumerate(units)
    ]


def _asr_evidence(
    *,
    source_audio_sha256: str,
    source_window_id: str,
    start_ms: int,
    end_ms: int,
    hypotheses: Sequence[tuple[str, float]],
) -> dict[str, Any]:
    rows = [
        {
            "text": text,
            "language": "zh",
            "tokens": _tokens(text, start_ms, end_ms),
            "acousticScore": score,
            "acousticScoreStatus": "available",
            "decodeScore": score,
            "decodeScoreStatus": "available",
        }
        for text, score in hypotheses
    ]
    candidate_set = build_asr_candidate_set(
        model_id=GENERATOR_ID,
        model_revision=GENERATOR_REVISION,
        model_manifest_sha256=GENERATOR_MANIFEST_SHA256,
        model_identity_status="manifest-bound",
        source_audio_sha256=source_audio_sha256,
        normalization_profile=NORMALIZATION_PROFILE,
        source_window_id=source_window_id,
        start_ms=start_ms,
        end_ms=end_ms,
        hypotheses=rows,
        candidate_set_type=(
            "provider-nbest" if len(rows) > 1 else "provider-top1-only"
        ),
    )
    return {
        **candidate_set,
        "timestamps": candidate_set["nBest"][0]["tokens"],
        "evaluationGenerator": {
            "id": GENERATOR_ID,
            "revision": GENERATOR_REVISION,
            "manifestSha256": GENERATOR_MANIFEST_SHA256,
        },
    }


def _overlap_flags(context: Sequence[Utterance]) -> list[bool]:
    return [
        any(
            index != other_index
            and other.end_ms > item.start_ms
            and other.start_ms < item.end_ms
            for other_index, other in enumerate(context)
        )
        for index, item in enumerate(context)
    ]


def _current_timeline_turns(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "startMs": int(segment["startMs"]),
            "endMs": int(segment["endMs"]),
            "speakerId": str(segment["speakerId"]),
        }
        for segment in segments
    ]


def _speaker_number(speaker_id: str) -> int:
    try:
        return int(speaker_id.removeprefix("speaker-"))
    except ValueError as exc:
        raise SemanticTrialError("speaker ID is not canonical") from exc


def _alternate_timeline_turns(
    segments: Sequence[Mapping[str, Any]],
    *,
    target_index: int,
    alternate_speaker: str,
) -> list[dict[str, Any]]:
    speaker_ids = sorted(
        {str(segment["speakerId"]) for segment in segments},
        key=_speaker_number,
    )
    local_by_speaker = {
        speaker_id: f"evaluation-speaker-{index:04d}"
        for index, speaker_id in enumerate(speaker_ids, start=1)
    }
    target_current = str(segments[target_index]["speakerId"])
    if alternate_speaker == target_current or alternate_speaker not in local_by_speaker:
        raise SemanticTrialError("alternate timeline speaker must be a distinct top-2 candidate")
    current_counts = Counter(str(segment["speakerId"]) for segment in segments)
    turns: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        speaker_choices = [str(segment["speakerId"])]
        if index == target_index:
            speaker_choices = [alternate_speaker]
            if current_counts[target_current] == 1:
                speaker_choices.append(target_current)
        for speaker_id in speaker_choices:
            turns.append(
                {
                    "startMs": int(segment["startMs"]),
                    "endMs": int(segment["endMs"]),
                    "localSpeaker": local_by_speaker[speaker_id],
                }
            )
    return sorted(
        turns,
        key=lambda item: (
            item["startMs"],
            item["endMs"],
            item["localSpeaker"],
        ),
    )


def _alternate_timeline_evidence(
    segments: Sequence[Mapping[str, Any]],
    *,
    target_index: int,
    duration_ms: int,
) -> dict[str, Any]:
    target = segments[target_index]
    ranked = sorted(
        target["speakerScores"],
        key=lambda item: (-float(item["score"]), str(item["speakerId"])),
    )
    if len(ranked) < 2 or str(ranked[0]["speakerId"]) != target["speakerId"]:
        raise SemanticTrialError("target speaker evidence must expose current and alternate")
    turns = _alternate_timeline_turns(
        segments,
        target_index=target_index,
        alternate_speaker=str(ranked[1]["speakerId"]),
    )
    return {
        "provider": {
            "id": _ALTERNATE_TIMELINE_PROVIDER_ID,
            "version": _ALTERNATE_TIMELINE_PROVIDER_VERSION,
        },
        "fullTimelineInference": {
            "scope": "full-normalized-timeline",
            "startMs": 0,
            "endMs": duration_ms,
            "localSpeakerCount": len(
                {str(segment["speakerId"]) for segment in segments}
            ),
            "speakerTurns": turns,
            "speakerTurnsSha256": canonical_json_sha256(turns),
        },
    }


def build_case(spec: CaseSpec) -> dict[str, Any]:
    recording = spec.recording
    if not recording.audio_sha256 or not recording.textgrid_sha256:
        raise SemanticTrialError("recording hashes must be populated before case build")
    context = _context(recording, spec.target_index)
    if context is None:
        raise SemanticTrialError("case context is unavailable")
    target = recording.utterances[spec.target_index]
    labels = sorted({item.speaker_label for item in context})
    speaker_by_label = {
        label: f"speaker-{index}"
        for index, label in enumerate(labels, start=1)
    }
    target_context_index = 2
    true_target_speaker = speaker_by_label[target.speaker_label]
    competitor_speaker = (
        speaker_by_label[spec.competitor_label]
        if spec.competitor_label is not None
        else None
    )
    seed = {
        "schemaVersion": SCHEMA_VERSION,
        "corpus": recording.corpus,
        "recordingId": recording.recording_id,
        "split": recording.split,
        "category": spec.category,
        "targetSourceKey": target.source_key,
    }
    case_id = "semantic-case-" + canonical_json_sha256(seed)[:24]
    overlap_flags = _overlap_flags(context)
    segments: list[dict[str, Any]] = []
    expected_text = target.text
    expected_speaker = true_target_speaker
    target_current_text = target.text
    target_current_speaker = true_target_speaker
    if spec.category == "text-correction":
        assert spec.variant_text is not None
        target_current_text = spec.variant_text
    elif spec.category == "speaker-continuity-correction":
        assert competitor_speaker is not None
        target_current_speaker = competitor_speaker

    for index, utterance in enumerate(context):
        segment_id = f"segment-{index + 1}"
        true_speaker = speaker_by_label[utterance.speaker_label]
        current_speaker = (
            target_current_speaker
            if index == target_context_index
            else true_speaker
        )
        current_text = (
            target_current_text if index == target_context_index else utterance.text
        )
        hypotheses: list[tuple[str, float]] = [(current_text, 0.99)]
        if index == target_context_index and spec.category == "text-correction":
            hypotheses = [
                (current_text, _NEAR_CANDIDATE_HIGH_SCORE),
                (expected_text, _NEAR_CANDIDATE_LOW_SCORE),
            ]
        elif (
            index == target_context_index
            and spec.category == "lexical-protected-preservation"
        ):
            assert spec.variant_text is not None
            hypotheses = [
                (current_text, _NEAR_CANDIDATE_HIGH_SCORE),
                (spec.variant_text, _NEAR_CANDIDATE_LOW_SCORE),
            ]

        if index == target_context_index:
            alternate_speaker = (
                true_target_speaker
                if target_current_speaker != true_target_speaker
                else competitor_speaker
            )
            if alternate_speaker is None:
                alternate_speaker = next(
                    speaker_id
                    for speaker_id in sorted(
                        speaker_by_label.values(),
                        key=_speaker_number,
                    )
                    if speaker_id != target_current_speaker
                )
            leading = [
                (target_current_speaker, _NEAR_CANDIDATE_HIGH_SCORE),
                (alternate_speaker, _NEAR_CANDIDATE_LOW_SCORE),
            ]
            score_pairs = leading + [
                (speaker_id, 0.05)
                for speaker_id in sorted(
                    speaker_by_label.values(),
                    key=_speaker_number,
                )
                if speaker_id not in {item[0] for item in leading}
            ][:1]
        else:
            score_pairs = [(current_speaker, 0.98)] + [
                (speaker_id, 0.01)
                for speaker_id in sorted(speaker_by_label.values())
                if speaker_id != current_speaker
            ][:2]
        asr = _asr_evidence(
            source_audio_sha256=recording.audio_sha256,
            source_window_id=f"{case_id}:{segment_id}",
            start_ms=utterance.start_ms,
            end_ms=utterance.end_ms,
            hypotheses=hypotheses,
        )
        segments.append(
            {
                "id": segment_id,
                "turnId": f"turn-{index + 1:04d}",
                "startMs": utterance.start_ms,
                "endMs": utterance.end_ms,
                "speakerId": current_speaker,
                "rawText": current_text,
                "normalizedText": current_text,
                "displayText": current_text,
                "confidence": 0.8,
                "speakerScores": [
                    {"speakerId": speaker_id, "score": score}
                    for speaker_id, score in score_pairs
                ],
                "speakerMargin": 0.02 if index == target_context_index else 0.97,
                "overlapping": overlap_flags[index],
                "humanLocked": False,
                "revisions": [],
                "language": "zh",
                "evidence": {"asr": asr},
            }
        )

    target_segment = segments[target_context_index]
    target_segment["evidence"]["overlap"] = _alternate_timeline_evidence(
        segments,
        target_index=target_context_index,
        duration_ms=recording.duration_ms,
    )
    current_turns = _current_timeline_turns(segments)
    document = {
        "schemaVersion": "2.0.0",
        "documentId": f"document-{case_id}",
        "jobId": case_id,
        "generatedAt": "2026-08-07T00:00:00Z",
        "language": "zh",
        "source": {
            "fileName": recording.audio_path.name,
            "path": str(recording.audio_path),
            "sha256": recording.audio_sha256,
            "bytes": recording.audio_path.stat().st_size,
            "durationMs": recording.duration_ms,
        },
        "speakerPolicy": {
            "mode": "manual",
            "resolvedCount": len(speaker_by_label),
            "speakerIds": sorted(
                speaker_by_label.values(),
                key=lambda item: int(item.removeprefix("speaker-")),
            ),
        },
        "speakers": [
            {"id": speaker_id}
            for speaker_id in sorted(
                speaker_by_label.values(),
                key=lambda item: int(item.removeprefix("speaker-")),
            )
        ],
        "speakerTimeline": {
            "provider": {
                "id": _CURRENT_TIMELINE_PROVIDER_ID,
                "version": "1",
            },
            "regular": {
                "turns": current_turns,
                "sha256": canonical_json_sha256(current_turns),
            },
        },
        "segments": segments,
        "provenance": {
            "offline": True,
            "models": [],
            "evaluationGenerator": GENERATOR_ID,
        },
    }
    expected_candidate = next(
        item
        for item in target_segment["evidence"]["asr"]["nBest"]
        if item["text"] == expected_text
    )
    body = {
        "caseId": case_id,
        "category": spec.category,
        "split": recording.split,
        "corpus": recording.corpus,
        "recordingId": recording.recording_id,
        "targetSegmentId": target_segment["id"],
        "source": {
            "textGridPath": str(recording.textgrid_path),
            "textGridSha256": recording.textgrid_sha256,
            "audioPath": str(recording.audio_path),
            "audioSha256": recording.audio_sha256,
            "targetTier": target.speaker_label,
            "targetStartMs": target.start_ms,
            "targetEndMs": target.end_ms,
            "contextSourceKeys": [item.source_key for item in context],
        },
        "document": document,
        "documentCanonicalSha256": canonical_json_sha256(document),
        "expected": {
            "normalizedText": expected_text,
            "speakerId": expected_speaker,
            "textCandidateId": expected_candidate["candidateId"],
            "textChangeRequired": spec.category == "text-correction",
            "speakerChangeRequired": (
                spec.category == "speaker-continuity-correction"
            ),
            "protectedInventory": protected_inventory(expected_text),
            "boundaryMustRemain": spec.category == "speaker-boundary-control",
        },
    }
    return {**body, "caseSha256": canonical_json_sha256(body)}


def _verify_archives(eval_root: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for corpus in CORPORA:
        expected = _ARCHIVE_EVIDENCE[corpus]
        archive_path = eval_root / str(expected["archiveRelativePath"])
        resolved = archive_path.resolve(strict=True)
        if not resolved.is_file():
            raise SemanticTrialError(f"archive is not a file: {resolved}")
        if resolved.stat().st_size != expected["archiveBytes"]:
            raise SemanticTrialError(f"archive size mismatches for {corpus}")
        actual_sha = sha256_file(resolved)
        if actual_sha != expected["archiveSha256"]:
            raise SemanticTrialError(f"archive SHA-256 mismatches for {corpus}")
        evidence.append(
            {
                "corpus": corpus,
                "openSlrId": expected["openSlrId"],
                "openSlrPage": expected["openSlrPage"],
                "archiveUrl": expected["archiveUrl"],
                "archivePath": str(resolved),
                "archiveBytes": expected["archiveBytes"],
                "archiveSha256": actual_sha,
                "retrievedAt": expected["retrievedAt"],
                "license": expected["license"],
                "revision": None,
                "revisionStatus": "not-applicable-archive",
                "usagePolicy": "local-evaluation-only",
                "applicationBundlingAllowed": False,
            }
        )
    return evidence


def build_manifest(
    *,
    eval_root: Path,
    dependency_path: Path | None = DEFAULT_DEPENDENCY_PATH,
    cases_per_category: int = 24,
) -> dict[str, Any]:
    resolved_root = eval_root.resolve(strict=True)
    source_corpora = _verify_archives(resolved_root)
    recordings = discover_recordings(
        resolved_root,
        dependency_path=dependency_path,
    )
    specs = select_case_specs(
        recordings,
        cases_per_category=cases_per_category,
    )
    used_recordings = {
        (item.recording.corpus, item.recording.recording_id)
        for item in specs
    }
    for recording in recordings:
        recording.textgrid_sha256 = sha256_file(recording.textgrid_path)
        if (recording.corpus, recording.recording_id) in used_recordings:
            recording.audio_sha256 = sha256_file(recording.audio_path)
    cases = [build_case(spec) for spec in specs]
    category_counts = Counter(item["category"] for item in cases)
    split_counts = Counter(item["split"] for item in cases)
    corpus_counts = Counter(item["corpus"] for item in cases)
    body = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": ARTIFACT_TYPE,
        "createdAt": "2026-08-07T00:00:00Z",
        "purpose": "production-path-local-llm-semantic-evaluation",
        "redistribution": {
            "derivedTextBundledWithApplication": False,
            "localEvaluationOnly": True,
        },
        "parser": {
            "package": "praatio",
            "version": PRAATIO_VERSION,
            "includeEmptyIntervals": False,
        },
        "candidateGenerator": {
            **GENERATOR_CONTRACT,
            "manifestSha256": GENERATOR_MANIFEST_SHA256,
        },
        "sourceCorpora": source_corpora,
        "partition": {
            "method": "recording-isolated-sha256-rank-v1",
            "heldOutFractionPerCorpus": 0.25,
            "recordings": [
                {
                    "corpus": recording.corpus,
                    "recordingId": recording.recording_id,
                    "split": recording.split,
                    "textGridPath": str(recording.textgrid_path),
                    "textGridSha256": recording.textgrid_sha256,
                }
                for recording in recordings
            ],
        },
        "counts": {
            "cases": len(cases),
            "categories": dict(sorted(category_counts.items())),
            "splits": dict(sorted(split_counts.items())),
            "corpora": dict(sorted(corpus_counts.items())),
            "recordings": len(recordings),
            "usedRecordings": len(used_recordings),
        },
        "cases": cases,
    }
    manifest = {**body, "canonicalSha256": canonical_json_sha256(body)}
    return validate_manifest(manifest)


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticTrialError("semantic trial manifest must be an object")
    document = dict(value)
    expected_digest = document.pop("canonicalSha256", None)
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or canonical_json_sha256(document) != expected_digest
    ):
        raise SemanticTrialError("semantic trial manifest canonical digest mismatches")
    if (
        document.get("schemaVersion") != SCHEMA_VERSION
        or document.get("artifactType") != ARTIFACT_TYPE
    ):
        raise SemanticTrialError("semantic trial manifest identity mismatches")
    required_top_level = {
        "schemaVersion",
        "artifactType",
        "createdAt",
        "purpose",
        "redistribution",
        "parser",
        "candidateGenerator",
        "sourceCorpora",
        "partition",
        "counts",
        "cases",
    }
    if set(document) != required_top_level:
        raise SemanticTrialError("semantic trial manifest fields mismatch")
    parser = document.get("parser")
    if (
        not isinstance(parser, Mapping)
        or parser.get("package") != "praatio"
        or parser.get("version") != PRAATIO_VERSION
        or parser.get("includeEmptyIntervals") is not False
    ):
        raise SemanticTrialError("semantic trial parser provenance is invalid")
    generator = document.get("candidateGenerator")
    if not isinstance(generator, Mapping):
        raise SemanticTrialError("semantic trial candidate generator is missing")
    generator_body = dict(generator)
    generator_digest = generator_body.pop("manifestSha256", None)
    if (
        generator_body != GENERATOR_CONTRACT
        or generator_digest != GENERATOR_MANIFEST_SHA256
        or canonical_json_sha256(generator_body) != generator_digest
    ):
        raise SemanticTrialError("semantic trial candidate generator is invalid")
    raw_corpora = document.get("sourceCorpora")
    if not isinstance(raw_corpora, list) or len(raw_corpora) != len(CORPORA):
        raise SemanticTrialError("semantic trial source corpora are invalid")
    by_corpus = {
        str(item.get("corpus")): item
        for item in raw_corpora
        if isinstance(item, Mapping)
    }
    if set(by_corpus) != set(CORPORA):
        raise SemanticTrialError("semantic trial source corpora are incomplete")
    for corpus in CORPORA:
        source = by_corpus[corpus]
        expected = _ARCHIVE_EVIDENCE[corpus]
        if any(
            source.get(key) != expected[key]
            for key in (
                "openSlrId",
                "openSlrPage",
                "archiveUrl",
                "archiveBytes",
                "archiveSha256",
                "retrievedAt",
                "license",
            )
        ) or any(
            (
                source.get("revision") is not None,
                source.get("revisionStatus") != "not-applicable-archive",
                source.get("usagePolicy") != "local-evaluation-only",
                source.get("applicationBundlingAllowed") is not False,
            )
        ):
            raise SemanticTrialError(
                f"semantic trial source evidence is invalid for {corpus}"
            )
    partition = document.get("partition")
    raw_recordings = (
        partition.get("recordings") if isinstance(partition, Mapping) else None
    )
    if (
        not isinstance(partition, Mapping)
        or partition.get("method") != "recording-isolated-sha256-rank-v1"
        or partition.get("heldOutFractionPerCorpus") != 0.25
        or not isinstance(raw_recordings, list)
        or not raw_recordings
    ):
        raise SemanticTrialError("semantic trial partition is invalid")
    partition_by_recording: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_recording in raw_recordings:
        if not isinstance(raw_recording, Mapping):
            raise SemanticTrialError("semantic trial recording entry is invalid")
        key = (
            str(raw_recording.get("corpus")),
            str(raw_recording.get("recordingId")),
        )
        if (
            key in partition_by_recording
            or key[0] not in CORPORA
            or not key[1]
            or raw_recording.get("split") not in SPLITS
            or not isinstance(raw_recording.get("textGridPath"), str)
            or not isinstance(raw_recording.get("textGridSha256"), str)
            or len(str(raw_recording["textGridSha256"])) != 64
        ):
            raise SemanticTrialError("semantic trial recording identity is invalid")
        partition_by_recording[key] = raw_recording
    cases = document.get("cases")
    if not isinstance(cases, list) or not 96 <= len(cases) <= 192:
        raise SemanticTrialError("semantic trial manifest must contain 96-192 cases")
    seen_ids: set[str] = set()
    recording_splits: dict[tuple[str, str], str] = {}
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, Mapping):
            raise SemanticTrialError(f"semantic case {index} is not an object")
        case = dict(raw_case)
        case_digest = case.pop("caseSha256", None)
        if (
            not isinstance(case_digest, str)
            or canonical_json_sha256(case) != case_digest
        ):
            raise SemanticTrialError(f"semantic case {index} digest mismatches")
        case_id = case.get("caseId")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen_ids
            or case.get("category") not in CASE_CATEGORIES
            or case.get("split") not in SPLITS
        ):
            raise SemanticTrialError(f"semantic case {index} identity is invalid")
        seen_ids.add(case_id)
        if case.get("corpus") not in CORPORA:
            raise SemanticTrialError(f"semantic case {index} corpus is invalid")
        key = (str(case.get("corpus")), str(case.get("recordingId")))
        split = str(case["split"])
        prior = recording_splits.setdefault(key, split)
        if prior != split:
            raise SemanticTrialError("recording leaked across evaluation splits")
        partition_row = partition_by_recording.get(key)
        source = case.get("source")
        if (
            partition_row is None
            or partition_row.get("split") != split
            or not isinstance(source, Mapping)
            or source.get("textGridPath") != partition_row.get("textGridPath")
            or source.get("textGridSha256")
            != partition_row.get("textGridSha256")
        ):
            raise SemanticTrialError(
                f"semantic case {index} partition binding is invalid"
            )
        transcript = case.get("document")
        if (
            not isinstance(transcript, Mapping)
            or canonical_json_sha256(transcript)
            != case.get("documentCanonicalSha256")
        ):
            raise SemanticTrialError(
                f"semantic case {index} document digest mismatches"
            )
        if (
            transcript.get("jobId") != case_id
            or transcript.get("documentId") != f"document-{case_id}"
        ):
            raise SemanticTrialError(
                f"semantic case {index} document identity is invalid"
            )
        segments = transcript.get("segments")
        if (
            not isinstance(segments, list)
            or len(segments) != 5
            or any(not isinstance(item, Mapping) for item in segments)
        ):
            raise SemanticTrialError(f"semantic case {index} context is invalid")
        target_id = case.get("targetSegmentId")
        targets = [item for item in segments if item.get("id") == target_id]
        if len(targets) != 1:
            raise SemanticTrialError(f"semantic case {index} target is invalid")
        turn_ids = [item.get("turnId") for item in segments]
        expected_turn_ids = [
            f"turn-{position:04d}" for position in range(1, len(segments) + 1)
        ]
        if (
            turn_ids != expected_turn_ids
            or any(
                not isinstance(turn_id, str)
                or _NEUTRAL_TURN_ID.fullmatch(turn_id) is None
                for turn_id in turn_ids
            )
        ):
            raise SemanticTrialError(
                f"semantic case {index} exposes a non-neutral turn identity"
            )
        timeline = transcript.get("speakerTimeline")
        provider = timeline.get("provider") if isinstance(timeline, Mapping) else None
        regular = timeline.get("regular") if isinstance(timeline, Mapping) else None
        current_turns = _current_timeline_turns(segments)
        if (
            not isinstance(provider, Mapping)
            or provider.get("id") != _CURRENT_TIMELINE_PROVIDER_ID
            or provider.get("version") != "1"
            or not isinstance(regular, Mapping)
            or regular.get("turns") != current_turns
            or regular.get("sha256") != canonical_json_sha256(current_turns)
        ):
            raise SemanticTrialError(
                f"semantic case {index} timeline is not current-state-only"
            )
        context_source_keys = source.get("contextSourceKeys")
        if not isinstance(context_source_keys, list) or any(
            not isinstance(item, str) or not item
            for item in context_source_keys
        ):
            raise SemanticTrialError(
                f"semantic case {index} source context keys are invalid"
            )
        if any(
            source_key in turn_id or turn_id in source_key
            for source_key in context_source_keys
            for turn_id in turn_ids
        ):
            raise SemanticTrialError(
                f"semantic case {index} turn identity leaks a source tier"
            )
        for segment in segments:
            if not isinstance(segment, Mapping):
                raise SemanticTrialError(f"semantic case {index} segment is invalid")
            evidence = segment.get("evidence")
            asr = evidence.get("asr") if isinstance(evidence, Mapping) else None
            if (
                not isinstance(asr, Mapping)
                or not set(ASR_CANDIDATE_SET_KEYS).issubset(asr)
                or not isinstance(asr.get("evaluationGenerator"), Mapping)
                or asr["evaluationGenerator"].get("id") != GENERATOR_ID
            ):
                raise SemanticTrialError(
                    f"semantic case {index} ASR evidence is invalid"
                )
            validate_asr_candidate_set(
                asr,
                expected_text=str(segment["rawText"]),
                expected_start_ms=int(segment["startMs"]),
                expected_end_ms=int(segment["endMs"]),
            )
        expected = case.get("expected")
        target = targets[0]
        target_asr = target["evidence"]["asr"]
        raw_target_scores = target.get("speakerScores")
        target_scores = (
            sorted(
                raw_target_scores,
                key=lambda item: (-float(item["score"]), str(item["speakerId"])),
            )
            if isinstance(raw_target_scores, list)
            and all(isinstance(item, Mapping) for item in raw_target_scores)
            else []
        )
        if (
            len(target_scores) < 2
            or target_scores[0].get("speakerId") != target.get("speakerId")
            or [item.get("score") for item in target_scores[:2]]
            != [_NEAR_CANDIDATE_HIGH_SCORE, _NEAR_CANDIDATE_LOW_SCORE]
            or target_scores[0].get("speakerId")
            == target_scores[1].get("speakerId")
        ):
            raise SemanticTrialError(
                f"semantic case {index} target speaker candidates are not symmetric"
            )
        expected_timeline_evidence = _alternate_timeline_evidence(
            segments,
            target_index=segments.index(target),
            duration_ms=int(transcript["source"]["durationMs"]),
        )
        target_evidence = target.get("evidence")
        if (
            not isinstance(target_evidence, Mapping)
            or target_evidence.get("overlap") != expected_timeline_evidence
        ):
            raise SemanticTrialError(
                f"semantic case {index} alternate timeline is not a deterministic "
                "projection of the public top-2 speaker candidates"
            )
        expected_candidate = next(
            (
                item
                for item in target_asr["nBest"]
                if isinstance(item, Mapping)
                and item.get("candidateId")
                == (
                    expected.get("textCandidateId")
                    if isinstance(expected, Mapping)
                    else None
                )
                and item.get("text")
                == (
                    expected.get("normalizedText")
                    if isinstance(expected, Mapping)
                    else None
                )
            ),
            None,
        )
        if (
            not isinstance(expected, Mapping)
            or expected.get("speakerId")
            not in transcript["speakerPolicy"]["speakerIds"]
            or expected_candidate is None
            or expected_candidate.get("lexicalRepairEligible") is not True
            or expected.get("protectedInventory")
            != protected_inventory(str(expected.get("normalizedText") or ""))
        ):
            raise SemanticTrialError(
                f"semantic case {index} expected state is invalid"
            )
        timeline_target = [
            turn
            for turn in current_turns
            if turn["startMs"] == target["startMs"]
            and turn["endMs"] == target["endMs"]
        ]
        if (
            len(timeline_target) != 1
            or timeline_target[0]["speakerId"] != target["speakerId"]
        ):
            raise SemanticTrialError(
                f"semantic case {index} target timeline leaks a reference speaker"
            )
        category = case.get("category")
        if category == "speaker-continuity-correction" and (
            target.get("speakerId") == expected.get("speakerId")
            or target_scores[1].get("speakerId") != expected.get("speakerId")
            or expected.get("speakerChangeRequired") is not True
        ):
            raise SemanticTrialError(
                f"semantic case {index} speaker correction lacks a blind current state"
            )
        if category == "speaker-boundary-control" and (
            target.get("speakerId") != expected.get("speakerId")
            or target_scores[1].get("speakerId") == expected.get("speakerId")
            or expected.get("boundaryMustRemain") is not True
        ):
            raise SemanticTrialError(
                f"semantic case {index} speaker boundary current state is invalid"
            )
        if category in {
            "text-correction",
            "lexical-protected-preservation",
        } and (
            target.get("speakerId") != expected.get("speakerId")
            or expected.get("speakerChangeRequired") is not False
        ):
            raise SemanticTrialError(
                f"semantic case {index} text case speaker state is invalid"
            )
        if category in {
            "text-correction",
            "lexical-protected-preservation",
        }:
            n_best = target_asr["nBest"]
            score_fields = ("acousticScore", "decodeScore")
            if len(n_best) != 2 or any(
                [item.get(field) for item in n_best]
                != [_NEAR_CANDIDATE_HIGH_SCORE, _NEAR_CANDIDATE_LOW_SCORE]
                for field in score_fields
            ):
                raise SemanticTrialError(
                    f"semantic case {index} text candidates reveal the answer by score"
                )
            if category == "text-correction" and (
                expected_candidate.get("candidateId")
                == n_best[0].get("candidateId")
                or expected.get("textChangeRequired") is not True
            ):
                raise SemanticTrialError(
                    f"semantic case {index} text correction answer has incumbent score bias"
                )
            if category == "lexical-protected-preservation" and (
                expected_candidate.get("candidateId")
                != n_best[0].get("candidateId")
                or expected.get("textChangeRequired") is not False
            ):
                raise SemanticTrialError(
                    f"semantic case {index} lexical preservation state is invalid"
                )
    if set(recording_splits.values()) != set(SPLITS):
        raise SemanticTrialError("semantic trial manifest lacks both splits")
    actual_counts = {
        "cases": len(cases),
        "categories": dict(
            sorted(Counter(str(item["category"]) for item in cases).items())
        ),
        "splits": dict(
            sorted(Counter(str(item["split"]) for item in cases).items())
        ),
        "corpora": dict(
            sorted(Counter(str(item["corpus"]) for item in cases).items())
        ),
        "recordings": len(partition_by_recording),
        "usedRecordings": len(recording_splits),
    }
    if document.get("counts") != actual_counts:
        raise SemanticTrialError("semantic trial manifest counts mismatch")
    return {**document, "canonicalSha256": expected_digest}


def load_manifest(path: Path) -> dict[str, Any]:
    return validate_manifest(read_json_strict(path.resolve(strict=True)))


def _positive_case_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("case count must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument(
        "--dependency-path",
        type=Path,
        default=DEFAULT_DEPENDENCY_PATH,
    )
    parser.add_argument(
        "--cases-per-category",
        type=_positive_case_count,
        default=24,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_manifest(
        eval_root=args.eval_root,
        dependency_path=args.dependency_path,
        cases_per_category=args.cases_per_category,
    )
    atomic_write_json_no_replace(args.output.resolve(), manifest)
    print(
        f"wrote {manifest['counts']['cases']} semantic cases to "
        f"{args.output.resolve()} ({manifest['canonicalSha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
