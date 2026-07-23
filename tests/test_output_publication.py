from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.errors import JobCancelled, WorkerError
from backend.output_orchestration import MediaProbeArtifact, OutputExecutionPlan
from backend.output_publication import (
    OutputPublicationCancelled,
    OutputPublicationError,
    publish_output_plans,
)
from backend.output_recipe import parse_output_recipe
from backend.subtitles import (
    CuePolicy,
    SubtitleFormat,
    SubtitleOutputMode,
    SubtitleStyle,
)


def _recipe(
    *,
    formats: list[str],
    modes: list[str],
) -> Any:
    return parse_output_recipe(
        {
            "schemaVersion": "1.0.0",
            "report": {
                "template": "soft-glass",
                "font": "system-sans",
                "customFontFamily": "",
                "pageSize": "a4",
                "density": "balanced",
                "accentColor": "#6959d2",
            },
            "subtitles": {
                "enabled": True,
                "theme": "youtube-clean",
                "size": "medium",
                "safeArea": "broadcast",
                "position": "smart",
                "speakerPalette": "adaptive-spectrum",
                "backgroundOpacity": 72,
                "maximumLines": 2,
                "avoidVisualCollisions": True,
                "wordProgressHighlight": False,
            },
            "delivery": {
                "formats": formats,
                "subtitleModes": modes,
                "includeMediaMetadata": True,
                "preserveSourceMedia": True,
                "fileNamePattern": "{sourceStem}-{artifact}",
            },
            "finishing": {
                "includeCover": True,
                "includeChapters": True,
                "includeTimestamps": True,
                "includeHeader": True,
                "includeFooter": True,
                "includeSpeakerIndex": True,
                "includeConfidenceNotes": False,
                "chapterStyle": "semantic",
                "timestampStyle": "segment",
                "customTitle": "",
            },
        }
    )


def _document() -> dict[str, Any]:
    return {
        "speakers": [
            {"id": "speaker-1", "displayName": "Alice"},
            {"id": "speaker-2", "displayName": "Bob"},
        ],
        "segments": [
            {
                "startMs": 0,
                "endMs": 2400,
                "speakerId": "speaker-1",
                "rawText": "Hello.",
                "normalizedText": "Hello.",
                "displayText": "Hello.",
            },
            {
                "startMs": 2500,
                "endMs": 5000,
                "speakerId": "speaker-2",
                "rawText": "Welcome.",
                "normalizedText": "Welcome.",
                "displayText": "Welcome.",
            },
        ],
    }


def _plan(
    source: Path,
    output: Path,
    mode: SubtitleOutputMode,
    *,
    sidecar_paths: dict[SubtitleFormat, Path],
    style: SubtitleStyle | None = None,
    media_output: Path | None = None,
    customization_sha256: str | None = None,
) -> OutputExecutionPlan:
    ordered_formats = tuple(
        sorted(sidecar_paths, key=lambda value: value.value)
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    return OutputExecutionPlan(
        customization_sha256=customization_sha256
        or hashlib.sha256(mode.value.encode()).hexdigest(),
        source_path=source.resolve(),
        output_directory=output.resolve(),
        media_probe_artifact=MediaProbeArtifact(
            path=(output / "media-probe.v1.json").resolve(),
            size_bytes=0,
            sha256=hashlib.sha256(b"").hexdigest(),
            source_path=source.resolve(),
            source_size_bytes=source.stat().st_size,
            source_sha256=source_sha256,
            probe_fingerprint_sha256=hashlib.sha256(
                b"output-publication-test-probe"
            ).hexdigest(),
        ),
        report_enabled=False,
        report_config={},
        exports_config={},
        safety_config={
            "preserveSourceMedia": True,
            "overwriteSourceMedia": False,
        },
        reversibility_config={
            "sourceMediaImmutable": True,
            "derivedArtifactOnly": True,
        },
        subtitle_enabled=True,
        subtitle_config={
            "speakerColors": {
                "mode": "automatic",
                "seed": "test",
                "algorithm": "oklch-hash-v1",
                "overrides": [],
            },
        },
        subtitle_formats=ordered_formats,
        subtitle_style=style or SubtitleStyle(),
        cue_policy=CuePolicy(
            max_characters_per_line=42,
            max_lines=2,
            max_reading_speed=30.0,
            min_cue_ms=500,
            max_cue_ms=7000,
            gap_ms=80,
        ),
        subtitle_theme="youtube-clean",
        speaker_color_mode="automatic",
        speaker_color_seed="test",
        speaker_color_overrides={},
        sidecar_paths=tuple(
            (subtitle_format, sidecar_paths[subtitle_format])
            for subtitle_format in ordered_formats
        ),
        delivery_mode=mode,
        delivery_output_path=media_output,
        subtitle_codec=None,
        burn_in_strategy=(
            "h264-high-quality"
            if mode is SubtitleOutputMode.BURN_IN
            else None
        ),
        visual_qa_required=mode is not SubtitleOutputMode.SIDECAR,
        presentation_limitations=(),
    )


def _fixture(
    tmp_path: Path,
    *,
    modes: tuple[SubtitleOutputMode, ...] = (
        SubtitleOutputMode.SIDECAR,
        SubtitleOutputMode.SOFT_MUX,
        SubtitleOutputMode.BURN_IN,
    ),
    formats: tuple[SubtitleFormat, ...] = (
        SubtitleFormat.ASS,
        SubtitleFormat.SRT,
        SubtitleFormat.WEBVTT,
    ),
) -> tuple[Path, Path, list[OutputExecutionPlan]]:
    source = tmp_path / "source.extension-does-not-matter"
    source.write_bytes(b"immutable-source-media")
    output = tmp_path / "output"
    output.mkdir()
    sidecars = {
        subtitle_format: output
        / {
            SubtitleFormat.ASS: "meeting.ass",
            SubtitleFormat.SRT: "meeting.srt",
            SubtitleFormat.WEBVTT: "meeting.vtt",
        }[subtitle_format]
        for subtitle_format in formats
    }
    plans = [
        _plan(
            source,
            output,
            mode,
            sidecar_paths=sidecars,
            media_output=(
                None
                if mode is SubtitleOutputMode.SIDECAR
                else output
                / (
                    "meeting-soft.mkv"
                    if mode is SubtitleOutputMode.SOFT_MUX
                    else "meeting-burn.mp4"
                )
            ),
        )
        for mode in modes
    ]
    return source, output, plans


@dataclass(frozen=True)
class _Evidence:
    size_bytes: int
    sha256: str


@dataclass
class _Staged:
    customer_output_path: Path
    quarantine_path: Path
    receipt: Any
    quarantine_evidence: _Evidence


class FakeExecutor:
    def __init__(self) -> None:
        self.deliver_calls: list[tuple[Any, bytes]] = []
        self.stage_calls: list[Any] = []
        self.publish_calls: list[Any] = []
        self.rollback_calls: list[tuple[Any, str, Any]] = []
        self.fail_public_format: SubtitleFormat | None = None
        self.fail_stage_mode: SubtitleOutputMode | None = None
        self.fail_publish_mode: SubtitleOutputMode | None = None
        self._counter = 0

    def deliver(
        self,
        plan: Any,
        *,
        sidecar_payload: str | bytes | None = None,
        **_: Any,
    ) -> Any:
        payload = (
            sidecar_payload.encode("utf-8")
            if isinstance(sidecar_payload, str)
            else bytes(sidecar_payload or b"")
        )
        self.deliver_calls.append((plan, payload))
        private = ".mts-private-subtitle-" in str(plan.output_path)
        if (
            not private
            and self.fail_public_format is plan.subtitle_format
        ):
            raise RuntimeError("injected public sidecar failure")
        output = Path(plan.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        evidence = _evidence(output)
        return SimpleNamespace(
            output_path=str(output),
            output_evidence=evidence,
        )

    def stage_media_delivery(self, plan: Any, **_: Any) -> _Staged:
        self.stage_calls.append(plan)
        if self.fail_stage_mode is plan.mode:
            raise RuntimeError("injected stage failure")
        subtitle = Path(plan.subtitle_path)
        assert subtitle.is_file()
        self._counter += 1
        customer = Path(plan.output_path)
        quarantine = customer.with_name(
            f".{customer.name}.quarantine-{self._counter}"
        )
        quarantine.write_bytes(
            b"media:"
            + plan.mode.value.encode("ascii")
            + b":"
            + subtitle.read_bytes()
        )
        evidence = _evidence(quarantine)
        return _Staged(
            customer_output_path=customer,
            quarantine_path=quarantine,
            receipt=SimpleNamespace(
                mode=plan.mode,
                source_path=plan.source_path,
            ),
            quarantine_evidence=evidence,
        )

    def publish_staged_media(
        self,
        staged: _Staged,
        *,
        visual_qa_evidence: dict[str, Any],
    ) -> Any:
        assert visual_qa_evidence["passed"] is True
        self.publish_calls.append(staged)
        mode = staged.receipt.mode
        if self.fail_publish_mode is mode:
            raise RuntimeError("injected media publication failure")
        os.link(staged.quarantine_path, staged.customer_output_path)
        staged.quarantine_path.unlink()
        evidence = _evidence(staged.customer_output_path)
        return SimpleNamespace(
            receipt=SimpleNamespace(
                mode=mode,
                output_path=str(staged.customer_output_path),
                output_evidence=evidence,
            ),
            published_evidence=evidence,
        )

    def rollback_staged_media(
        self,
        staged: _Staged,
        *,
        reason: str,
        published_evidence: Any = None,
    ) -> dict[str, Any]:
        self.rollback_calls.append((staged, reason, published_evidence))
        customer_state = "absent"
        if staged.customer_output_path.exists():
            if published_evidence is not None:
                staged.customer_output_path.unlink()
                customer_state = "rolled-back"
            else:
                customer_state = "existing-preserved"
        quarantine_state = "absent"
        if staged.quarantine_path.exists():
            staged.quarantine_path.unlink()
            quarantine_state = "removed"
        return {
            "status": "rolled-back",
            "customerOutputPath": str(staged.customer_output_path),
            "customerOutputState": customer_state,
            "quarantine": {
                "path": str(staged.quarantine_path),
                "state": quarantine_state,
            },
            "sourceMediaImmutable": True,
            "cleanupError": None,
        }


def _evidence(path: Path) -> _Evidence:
    payload = path.read_bytes()
    return _Evidence(
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _qa(**kwargs: Any) -> dict[str, Any]:
    assert Path(kwargs["rendered_path"]).is_file()
    return {
        "passed": True,
        "representativeFrames": [0, 2500, 5000],
        "safeArea": True,
        "glyphs": True,
    }


def _customer_paths(result: Any) -> list[Path]:
    return [
        Path(item["path"])
        for item in result.to_dict()["customerArtifacts"]
    ]


def test_multimode_publication_deduplicates_sidecars_and_hides_private_ass(
    tmp_path: Path,
) -> None:
    source, output, plans = _fixture(tmp_path)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    executor = FakeExecutor()

    result = publish_output_plans(
        _recipe(
            formats=["srt", "webvtt"],
            modes=["sidecar", "soft-mux", "burn-in"],
        ),
        plans,
        _document(),
        executor=executor,
        visual_qa_hook=_qa,
        subtitle_language="en",
    )

    manifest = result.to_dict()
    artifacts = manifest["customerArtifacts"]
    assert [item["subtitleFormat"] for item in artifacts[:2]] == [
        "srt",
        "webvtt",
    ]
    assert [item["deliveryMode"] for item in artifacts[2:]] == [
        "soft-mux",
        "burn-in",
    ]
    assert len(executor.deliver_calls) == 3
    assert sum(
        call[0].subtitle_format is SubtitleFormat.SRT
        for call in executor.deliver_calls
    ) == 1
    assert sum(
        call[0].subtitle_format is SubtitleFormat.WEBVTT
        for call in executor.deliver_calls
    ) == 1
    private_calls = [
        call
        for call in executor.deliver_calls
        if ".mts-private-subtitle-" in str(call[0].output_path)
    ]
    assert len(private_calls) == 1
    private_path = str(private_calls[0][0].output_path)
    assert private_path not in json.dumps(manifest)
    assert not Path(private_path).exists()
    assert not list(output.glob(".mts-private-subtitle-*"))
    assert not list(output.glob("*.ass"))
    assert manifest["internalEvidence"]["privateAssCarrier"] == {
        "created": True,
        "customerArtifact": False,
        "pathDisclosed": False,
        "unpredictableName": True,
        "payloadSha256": hashlib.sha256(private_calls[0][1]).hexdigest(),
        "sizeBytes": len(private_calls[0][1]),
        "cleanup": "removed",
        "reason": "internal-media-carrier",
    }
    assert before == hashlib.sha256(source.read_bytes()).hexdigest()


def test_explicit_public_ass_is_published_once_and_reused_as_carrier(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(
            SubtitleOutputMode.SOFT_MUX,
            SubtitleOutputMode.BURN_IN,
        ),
    )
    executor = FakeExecutor()

    result = publish_output_plans(
        _recipe(
            formats=["ass"],
            modes=["soft-mux", "burn-in"],
        ),
        plans,
        _document(),
        executor=executor,
        visual_qa_hook=_qa,
    )

    ass_calls = [
        call
        for call in executor.deliver_calls
        if call[0].subtitle_format is SubtitleFormat.ASS
    ]
    assert len(ass_calls) == 1
    assert Path(ass_calls[0][0].output_path) == output / "meeting.ass"
    assert all(
        Path(plan.subtitle_path) == output / "meeting.ass"
        for plan in executor.stage_calls
    )
    internal = result.to_dict()["internalEvidence"]["privateAssCarrier"]
    assert internal["created"] is False
    assert internal["cleanup"] == "not-created"
    assert internal["reason"] == "public-ass-reused"
    assert (output / "meeting.ass").is_file()


def test_identical_format_across_three_plans_is_published_once(
    tmp_path: Path,
) -> None:
    _, _, plans = _fixture(tmp_path)
    executor = FakeExecutor()

    publish_output_plans(
        _recipe(
            formats=["srt"],
            modes=["sidecar", "soft-mux", "burn-in"],
        ),
        plans,
        _document(),
        executor=executor,
        visual_qa_hook=_qa,
    )

    public_srt = [
        call
        for call in executor.deliver_calls
        if call[0].subtitle_format is SubtitleFormat.SRT
    ]
    assert len(public_srt) == 1


def test_conflicting_ass_payloads_fail_before_any_write(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(
            SubtitleOutputMode.SOFT_MUX,
            SubtitleOutputMode.BURN_IN,
        ),
    )
    plans[1] = _plan(
        plans[1].source_path,
        output,
        SubtitleOutputMode.BURN_IN,
        sidecar_paths=dict(plans[1].sidecar_paths),
        style=SubtitleStyle(primary_color="#FF0000"),
        media_output=output / "meeting-burn.mp4",
    )
    executor = FakeExecutor()

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(
                formats=["srt"],
                modes=["soft-mux", "burn-in"],
            ),
            plans,
            _document(),
            executor=executor,
            visual_qa_hook=_qa,
        )

    assert raised.value.code == "OUTPUT_PUBLICATION_PAYLOAD_CONFLICT"
    assert executor.deliver_calls == []
    assert executor.stage_calls == []
    assert list(output.iterdir()) == []


def test_same_public_format_with_different_targets_fails_closed(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(
            SubtitleOutputMode.SIDECAR,
            SubtitleOutputMode.SOFT_MUX,
        ),
    )
    changed = dict(plans[1].sidecar_paths)
    changed[SubtitleFormat.SRT] = output / "different.srt"
    plans[1] = _plan(
        plans[1].source_path,
        output,
        SubtitleOutputMode.SOFT_MUX,
        sidecar_paths=changed,
        media_output=output / "meeting-soft.mkv",
    )

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(
                formats=["srt"],
                modes=["sidecar", "soft-mux"],
            ),
            plans,
            _document(),
            executor=FakeExecutor(),
            visual_qa_hook=_qa,
        )

    assert raised.value.code == "OUTPUT_PUBLICATION_TARGET_CONFLICT"


def test_duplicate_output_path_across_formats_fails_closed(
    tmp_path: Path,
) -> None:
    source, output, _ = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SIDECAR,),
    )
    duplicate = output / "same.srt"
    plan = _plan(
        source,
        output,
        SubtitleOutputMode.SIDECAR,
        sidecar_paths={
            SubtitleFormat.SRT: duplicate,
            SubtitleFormat.WEBVTT: duplicate,
        },
    )

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(formats=["srt", "webvtt"], modes=["sidecar"]),
            [plan],
            _document(),
            executor=FakeExecutor(),
            visual_qa_hook=None,
        )

    assert raised.value.code == "OUTPUT_PUBLICATION_DUPLICATE_OUTPUT_PATH"


def test_media_output_path_conflict_fails_before_staging(
    tmp_path: Path,
) -> None:
    source, output, plans = _fixture(
        tmp_path,
        modes=(
            SubtitleOutputMode.SOFT_MUX,
            SubtitleOutputMode.BURN_IN,
        ),
    )
    plans[1] = _plan(
        source,
        output,
        SubtitleOutputMode.BURN_IN,
        sidecar_paths=dict(plans[1].sidecar_paths),
        media_output=output / "meeting-soft.mkv",
    )
    executor = FakeExecutor()

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(
                formats=["srt"],
                modes=["soft-mux", "burn-in"],
            ),
            plans,
            _document(),
            executor=executor,
            visual_qa_hook=_qa,
        )

    assert raised.value.code == "OUTPUT_PUBLICATION_DUPLICATE_OUTPUT_PATH"
    assert executor.stage_calls == []


def test_visual_qa_false_rolls_back_all_quarantine_and_private_carrier(
    tmp_path: Path,
) -> None:
    source, output, plans = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SOFT_MUX,),
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    executor = FakeExecutor()

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(formats=["srt"], modes=["soft-mux"]),
            plans,
            _document(),
            executor=executor,
            visual_qa_hook=lambda **_: {"passed": False},
        )

    assert raised.value.code == "OUTPUT_PUBLICATION_FAILED"
    rollback = raised.value.details["rollback"]
    assert rollback["status"] == "rolled-back"
    assert rollback["media"][0]["quarantineState"] == "removed"
    assert rollback["privateAssCarrier"]["status"] == "removed"
    assert not list(output.iterdir())
    assert source_hash == hashlib.sha256(source.read_bytes()).hexdigest()


def test_all_media_are_staged_and_qa_passed_before_first_publication(
    tmp_path: Path,
) -> None:
    _, _, plans = _fixture(
        tmp_path,
        modes=(
            SubtitleOutputMode.SOFT_MUX,
            SubtitleOutputMode.BURN_IN,
        ),
    )
    executor = FakeExecutor()
    qa_modes: list[SubtitleOutputMode] = []

    def qa(**kwargs: Any) -> dict[str, Any]:
        assert len(executor.stage_calls) == 2
        assert executor.publish_calls == []
        qa_modes.append(kwargs["delivery_receipt"].mode)
        return {"passed": True, "frames": [0, 1000]}

    publish_output_plans(
        _recipe(
            formats=["srt"],
            modes=["soft-mux", "burn-in"],
        ),
        plans,
        _document(),
        executor=executor,
        visual_qa_hook=qa,
    )

    assert qa_modes == [
        SubtitleOutputMode.SOFT_MUX,
        SubtitleOutputMode.BURN_IN,
    ]
    assert len(executor.publish_calls) == 2


def test_missing_visual_qa_hook_fails_before_private_output(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SOFT_MUX,),
    )
    executor = FakeExecutor()

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(formats=["srt"], modes=["soft-mux"]),
            plans,
            _document(),
            executor=executor,
            visual_qa_hook=None,
        )

    assert raised.value.code == "OUTPUT_PUBLICATION_VISUAL_QA_REQUIRED"
    assert executor.deliver_calls == []
    assert list(output.iterdir()) == []


def test_second_qa_failure_rolls_back_both_staged_media(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(
            SubtitleOutputMode.SOFT_MUX,
            SubtitleOutputMode.BURN_IN,
        ),
    )
    executor = FakeExecutor()
    calls = 0

    def qa(**_: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"passed": calls == 1}

    with pytest.raises(OutputPublicationError):
        publish_output_plans(
            _recipe(
                formats=["srt"],
                modes=["soft-mux", "burn-in"],
            ),
            plans,
            _document(),
            executor=executor,
            visual_qa_hook=qa,
        )

    assert len(executor.rollback_calls) == 2
    assert executor.publish_calls == []
    assert not list(output.iterdir())


def test_second_media_publication_failure_rolls_back_first_publication(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(
            SubtitleOutputMode.SOFT_MUX,
            SubtitleOutputMode.BURN_IN,
        ),
    )
    executor = FakeExecutor()
    executor.fail_publish_mode = SubtitleOutputMode.BURN_IN

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(
                formats=["srt"],
                modes=["soft-mux", "burn-in"],
            ),
            plans,
            _document(),
            executor=executor,
            visual_qa_hook=_qa,
        )

    assert raised.value.details["rollback"]["status"] == "rolled-back"
    assert not (output / "meeting-soft.mkv").exists()
    assert not (output / "meeting-burn.mp4").exists()
    assert not list(output.iterdir())


def test_sidecar_failure_after_media_publication_rolls_back_media(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SOFT_MUX,),
    )
    executor = FakeExecutor()
    executor.fail_public_format = SubtitleFormat.SRT

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(formats=["srt"], modes=["soft-mux"]),
            plans,
            _document(),
            executor=executor,
            visual_qa_hook=_qa,
        )

    assert raised.value.details["rollback"]["status"] == "rolled-back"
    assert not (output / "meeting-soft.mkv").exists()
    assert not list(output.iterdir())


def test_cancellation_after_staging_returns_rollback_evidence(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SOFT_MUX,),
    )
    executor = FakeExecutor()

    def cancellation() -> None:
        if executor.stage_calls:
            raise JobCancelled()

    with pytest.raises(OutputPublicationCancelled) as raised:
        publish_output_plans(
            _recipe(formats=["srt"], modes=["soft-mux"]),
            plans,
            _document(),
            executor=executor,
            visual_qa_hook=_qa,
            cancellation_check=cancellation,
        )

    rollback = raised.value.details["outputPublication"]
    assert rollback["status"] == "rolled-back"
    assert rollback["media"][0]["quarantineState"] == "removed"
    assert rollback["sourceMediaImmutable"] is True
    assert not list(output.iterdir())


def test_existing_customer_output_is_never_replaced(
    tmp_path: Path,
) -> None:
    source, output, _ = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SIDECAR,),
    )
    existing = output / "meeting.srt"
    existing.write_text("customer-owned", encoding="utf-8")
    plan = _plan(
        source,
        output,
        SubtitleOutputMode.SIDECAR,
        sidecar_paths={SubtitleFormat.SRT: existing},
    )

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(formats=["srt"], modes=["sidecar"]),
            [plan],
            _document(),
            executor=FakeExecutor(),
            visual_qa_hook=None,
        )

    assert raised.value.code == "OUTPUT_PUBLICATION_OUTPUT_EXISTS"
    assert existing.read_text(encoding="utf-8") == "customer-owned"


def test_recipe_and_plan_modes_must_match_exactly(tmp_path: Path) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SOFT_MUX,),
    )

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(
                formats=["srt"],
                modes=["soft-mux", "burn-in"],
            ),
            plans,
            _document(),
            executor=FakeExecutor(),
            visual_qa_hook=_qa,
        )

    assert raised.value.code == "OUTPUT_PUBLICATION_MODE_CONFLICT"
    assert list(output.iterdir()) == []


def test_manifest_is_stable_and_excludes_nondeterministic_private_paths(
    tmp_path: Path,
) -> None:
    _, _, plans = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SOFT_MUX,),
    )
    executor = FakeExecutor()
    result = publish_output_plans(
        _recipe(formats=["srt"], modes=["soft-mux"]),
        plans,
        _document(),
        executor=executor,
        visual_qa_hook=_qa,
    )

    first = result.to_dict()
    second = result.to_dict()
    assert first == second
    assert result.manifest_sha256 == first["manifestSha256"]
    serialized = json.dumps(first, sort_keys=True)
    private_path = next(
        str(call[0].output_path)
        for call in executor.deliver_calls
        if ".mts-private-subtitle-" in str(call[0].output_path)
    )
    assert private_path not in serialized
    assert "quarantine-" not in serialized


def test_source_hash_is_unchanged_on_success_and_receipts_are_exact(
    tmp_path: Path,
) -> None:
    source, _, plans = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SIDECAR,),
    )
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = publish_output_plans(
        _recipe(formats=["srt", "webvtt"], modes=["sidecar"]),
        plans,
        _document(),
        executor=FakeExecutor(),
        visual_qa_hook=None,
    )

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    for path, receipt in zip(
        _customer_paths(result),
        result.to_dict()["customerArtifacts"],
        strict=True,
    ):
        payload = path.read_bytes()
        assert receipt["sizeBytes"] == len(payload)
        assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
        assert receipt["sourceIntegrity"] == {
            "unchanged": True,
            "sourceSha256": before,
        }


def test_sidecar_mode_without_explicit_format_is_rejected(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SIDECAR,),
    )

    with pytest.raises(OutputPublicationError) as raised:
        publish_output_plans(
            _recipe(formats=["pdf"], modes=["sidecar"]),
            plans,
            _document(),
            executor=FakeExecutor(),
            visual_qa_hook=None,
        )

    assert raised.value.code == "OUTPUT_PUBLICATION_PUBLIC_FORMAT_REQUIRED"
    assert list(output.iterdir()) == []


def test_qa_exception_rolls_back_without_leaking_private_paths(
    tmp_path: Path,
) -> None:
    _, output, plans = _fixture(
        tmp_path,
        modes=(SubtitleOutputMode.SOFT_MUX,),
    )

    def qa(**_: Any) -> dict[str, Any]:
        raise RuntimeError("visual inspection service failed")

    with pytest.raises(WorkerError) as raised:
        publish_output_plans(
            _recipe(formats=["srt"], modes=["soft-mux"]),
            plans,
            _document(),
            executor=FakeExecutor(),
            visual_qa_hook=qa,
        )

    serialized = json.dumps(raised.value.as_payload())
    assert ".mts-private-subtitle-" not in serialized
    assert str(output.resolve()) not in serialized
    assert ".meeting-soft.mkv.quarantine-" not in serialized
    assert not list(output.iterdir())
