from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.errors import WorkerError
from backend.voice_activity import (
    build_voice_activity,
    validate_voice_activity,
    with_voice_activity_classification,
)


def _speech_activity() -> dict[str, object]:
    return build_voice_activity(
        job_id="voice-test",
        source_sha256="a" * 64,
        media_duration_ms=10_000,
        normalization_profile="mono-16khz-f32-v1",
        provider={"id": "FunASR", "version": "1.2.0"},
        windows=(
            {"id": "vad-1", "startMs": 1_000, "endMs": 4_000},
            {"id": "vad-2", "startMs": 5_000, "endMs": 9_000},
        ),
        minimum_window_ms=120,
        classification="transcribable-speech-detected",
        has_transcribable_speech=True,
    )


def test_voice_activity_computes_exact_coverage() -> None:
    value = _speech_activity()

    assert value["speechWindowCount"] == 2
    assert value["speechDurationMs"] == 7_000
    assert value["speechRatio"] == 0.7
    assert value["hasSpeechCandidates"] is True
    assert value["hasTranscribableSpeech"] is True


def test_voice_activity_reclassifies_non_lexical_without_losing_vad() -> None:
    value = with_voice_activity_classification(
        _speech_activity(),
        classification="no-lexical-speech-detected",
        has_transcribable_speech=False,
    )

    assert value["hasSpeechCandidates"] is True
    assert value["hasTranscribableSpeech"] is False
    assert len(value["windows"]) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(speechDurationMs=1),
        lambda value: value.update(speechRatio=0.1),
        lambda value: value.update(hasSpeechCandidates=False),
        lambda value: value.update(classification="unsupported"),
        lambda value: value["windows"].append(
            {
                "id": "overlap",
                "startMs": 8_000,
                "endMs": 9_500,
                "durationMs": 1_500,
            }
        ),
    ],
)
def test_voice_activity_rejects_inconsistent_evidence(mutation) -> None:
    value = _speech_activity()
    mutation(value)

    with pytest.raises(WorkerError) as captured:
        validate_voice_activity(value)

    assert captured.value.code == "VOICE_ACTIVITY_INVALID"


def test_voice_activity_contract_declares_closed_versioned_shape() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "contracts"
            / "voice-activity.schema.json"
        ).read_text(encoding="utf-8")
    )

    assert schema["additionalProperties"] is False
    assert schema["properties"]["schemaVersion"]["const"] == "1.0.0"
    assert schema["$id"].endswith("/1.0.0")
    assert set(schema["required"]) == set(schema["properties"])
