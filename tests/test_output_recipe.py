from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from backend.output_recipe import (
    OutputRecipeError,
    compile_output_customizations,
    parse_output_recipe,
    render_recipe_file_name,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE_SCHEMA = ROOT / "contracts" / "output-recipe.schema.json"


def recipe_payload(
    *,
    formats: list[str] | None = None,
    modes: list[str] | None = None,
    subtitles_enabled: bool = True,
) -> dict[str, object]:
    return {
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
            "enabled": subtitles_enabled,
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
            "formats": formats or ["pdf", "srt", "webvtt"],
            "subtitleModes": (
                modes
                if modes is not None
                else (["sidecar"] if subtitles_enabled else [])
            ),
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


def test_recipe_matches_its_portable_schema() -> None:
    schema = json.loads(RECIPE_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(recipe_payload())


def test_recipe_is_strict_canonical_and_hash_stable() -> None:
    payload = recipe_payload()
    first = parse_output_recipe(payload)
    second = parse_output_recipe(dict(reversed(list(payload.items()))))

    assert first.canonical_json() == second.canonical_json()
    assert first.deterministic_hash() == second.deterministic_hash()
    assert first.formats == ("pdf", "srt", "webvtt")
    assert first.subtitle_modes == ("sidecar",)
    assert first.render_pdf


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update({"unknown": True}),
            "invalid fields",
        ),
        (
            lambda payload: payload["delivery"].update(
                {"formats": ["pdf", "docx"]}
            ),
            "must be one of",
        ),
        (
            lambda payload: payload["subtitles"].update(
                {"wordProgressHighlight": True}
            ),
            "verified word-level timings",
        ),
        (
            lambda payload: payload["delivery"].update(
                {"fileNamePattern": "..\\escape-{artifact}"}
            ),
            "path separators",
        ),
    ],
)
def test_recipe_rejects_unknown_dead_or_unsafe_controls(
    mutation,
    message: str,
) -> None:
    payload = recipe_payload()
    mutation(payload)
    with pytest.raises(OutputRecipeError, match=message):
        parse_output_recipe(payload)


def test_recipe_compiles_every_selected_subtitle_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "meeting.mov"
    source.write_bytes(b"immutable-source")
    output = tmp_path / "output"
    output.mkdir()
    recipe = parse_output_recipe(
        recipe_payload(
            formats=["pdf", "html", "txt", "srt", "webvtt", "ass"],
            modes=["sidecar", "soft-mux", "burn-in"],
        )
    )
    media_probe = SimpleNamespace(
        video_stream_indexes=(0,),
        has_hdr_video=False,
    )
    artifact = SimpleNamespace(sha256="a" * 64)

    compiled = compile_output_customizations(
        recipe,
        source_path=source,
        output_directory=output,
        media_probe=media_probe,
        media_probe_artifact=artifact,
    )

    assert [item.delivery_mode for item in compiled] == [
        "sidecar",
        "soft-mux",
        "burn-in",
    ]
    payloads = [item.customization.canonical_dict() for item in compiled]
    assert all(item["report"]["enabled"] for item in payloads)
    assert payloads[0]["exports"]["transcriptFormats"] == ["txt", "html"]
    assert payloads[0]["delivery"]["outputTarget"]["binding"] == "deferred"
    assert payloads[1]["delivery"]["outputTarget"]["outputPath"].endswith(
        ".mkv"
    )
    assert payloads[2]["delivery"]["outputTarget"]["outputPath"].endswith(
        ".mp4"
    )
    assert payloads[2]["delivery"]["burnIn"]["dynamicRangeEvidence"] == {
        "verified": True,
        "method": "container-and-frame-probe",
        "probeSha256": "a" * 64,
    }
    assert all(
        item["safety"]["preserveSourceMedia"] is True for item in payloads
    )


def test_audio_only_burn_in_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"immutable-source")
    output = tmp_path / "output"
    output.mkdir()
    recipe = parse_output_recipe(
        recipe_payload(formats=["srt"], modes=["burn-in"])
    )

    with pytest.raises(OutputRecipeError, match="audio-only"):
        compile_output_customizations(
            recipe,
            source_path=source,
            output_directory=output,
            media_probe=SimpleNamespace(
                video_stream_indexes=(),
                has_hdr_video=False,
            ),
            media_probe_artifact=SimpleNamespace(sha256="a" * 64),
        )


def test_file_name_template_is_bounded_and_path_safe() -> None:
    recipe = parse_output_recipe(recipe_payload())
    value = render_recipe_file_name(
        recipe,
        source_stem='meeting:"one"',
        artifact="transcript",
        language="zh-Hans",
        generated_date="2026-07-23",
        speaker_count=5,
    )

    assert value == "meeting--one--transcript"
    assert "/" not in value
    assert "\\" not in value
