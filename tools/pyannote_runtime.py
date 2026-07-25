"""Isolated, offline Pyannote waveform inference entrypoint."""

from __future__ import annotations

import json
import math
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_REQUEST_BYTES = 1024 * 1024


def _request() -> dict[str, Any]:
    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("request size is invalid")
    value = json.loads(payload.decode("utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schemaVersion") not in {"1.0.0", "1.1.0"}
    ):
        raise ValueError("request contract is invalid")
    return value


def _bounded_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} is invalid")
    return value


def _local_path(value: Any, *, field: str, directory: bool) -> Path:
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise ValueError(f"{field} is invalid")
    path = Path(value).expanduser().resolve(strict=True)
    if directory and not path.is_dir():
        raise ValueError(f"{field} is not a directory")
    if not directory and not path.is_file():
        raise ValueError(f"{field} is not a file")
    return path


def _speaker_count_kwargs(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("speakerCountConstraints is invalid")
    if set(value) == {"numSpeakers"}:
        count = _bounded_integer(
            value.get("numSpeakers"),
            field="speakerCountConstraints.numSpeakers",
            minimum=1,
        )
        return {"num_speakers": count}
    if set(value) == {"minSpeakers", "maxSpeakers"}:
        minimum = _bounded_integer(
            value.get("minSpeakers"),
            field="speakerCountConstraints.minSpeakers",
            minimum=1,
        )
        maximum = _bounded_integer(
            value.get("maxSpeakers"),
            field="speakerCountConstraints.maxSpeakers",
            minimum=1,
        )
        if maximum < minimum:
            raise ValueError("speakerCountConstraints bounds are invalid")
        return {
            "min_speakers": minimum,
            "max_speakers": maximum,
        }
    raise ValueError("speakerCountConstraints uses unsupported fields")


def _annotation(result: Any, *, field: str) -> Any:
    candidate = (
        result.get(field)
        if isinstance(result, Mapping)
        else getattr(result, field, None)
    )
    if (
        field == "speaker_diarization"
        and candidate is None
        and callable(getattr(result, "itertracks", None))
    ):
        candidate = result
    if candidate is None or not callable(getattr(candidate, "itertracks", None)):
        raise ValueError(f"result has no {field} annotation")
    return candidate


def _turns(
    result: Any,
    *,
    field: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(
        _annotation(result, field=field).itertracks(yield_label=True)
    ):
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes, bytearray))
            or len(raw) not in {2, 3}
        ):
            raise ValueError(f"track {index} is malformed")
        interval = raw[0]
        label = str(raw[-1] or "").strip()
        raw_start = float(getattr(interval, "start"))
        raw_end = float(getattr(interval, "end"))
        if (
            not label
            or not math.isfinite(raw_start)
            or not math.isfinite(raw_end)
        ):
            raise ValueError(f"track {index} is invalid")
        turn_start = max(
            start_ms,
            min(end_ms, start_ms + round(raw_start * 1000.0)),
        )
        turn_end = max(
            turn_start,
            min(end_ms, start_ms + round(raw_end * 1000.0)),
        )
        if turn_end > turn_start:
            output.append(
                {
                    "startMs": turn_start,
                    "endMs": turn_end,
                    "localSpeaker": label,
                }
            )
    return sorted(
        output,
        key=lambda item: (
            item["startMs"],
            item["endMs"],
            item["localSpeaker"],
        ),
    )


def run() -> dict[str, Any]:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYANNOTE_METRICS_ENABLED": "0",
            "DO_NOT_TRACK": "1",
        }
    )
    request = _request()
    model_path = _local_path(
        request.get("modelPath"),
        field="modelPath",
        directory=True,
    )
    audio_path = _local_path(
        request.get("audioPath"),
        field="audioPath",
        directory=False,
    )
    device = request.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device is invalid")
    start_ms = _bounded_integer(request.get("startMs"), field="startMs")
    end_ms = _bounded_integer(request.get("endMs"), field="endMs", minimum=1)
    if end_ms <= start_ms:
        raise ValueError("inference interval is invalid")
    speaker_count_kwargs = _speaker_count_kwargs(
        request.get("speakerCountConstraints")
    )

    with redirect_stdout(sys.stderr):
        import soundfile
        import torch
        from pyannote.audio import Pipeline

        samples, sample_rate = soundfile.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
        if sample_rate != 16_000 or samples.shape[1] != 1:
            raise ValueError("audio must be mono 16 kHz")
        first_sample = round(start_ms * sample_rate / 1000)
        last_sample = round(end_ms * sample_rate / 1000)
        if first_sample < 0 or last_sample > samples.shape[0]:
            raise ValueError("inference interval exceeds audio")
        clip = samples[first_sample:last_sample, 0]
        if clip.size == 0:
            raise ValueError("inference interval is empty")
        pipeline = Pipeline.from_pretrained(str(model_path))
        pipeline.to(torch.device(device.strip()))
        result = pipeline(
            {
                "waveform": torch.from_numpy(clip).unsqueeze(0),
                "sample_rate": sample_rate,
            },
            **speaker_count_kwargs,
        )
    return {
        "schemaVersion": "1.1.0",
        "status": "ok",
        "speakerTurns": _turns(
            result,
            field="speaker_diarization",
            start_ms=start_ms,
            end_ms=end_ms,
        ),
        "exclusiveSpeakerTurns": _turns(
            result,
            field="exclusive_speaker_diarization",
            start_ms=start_ms,
            end_ms=end_ms,
        ),
    }


def main() -> int:
    try:
        response = run()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schemaVersion": "1.0.0",
                    "status": "error",
                    "errorType": type(exc).__name__,
                },
                separators=(",", ":"),
            )
        )
        return 1
    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
