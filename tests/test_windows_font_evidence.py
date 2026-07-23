from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from backend.subtitle_visual_evidence import (
    FontEvidenceObservation,
    SubtitleVisualEvidenceCollector,
    VerifiedFontClaim,
)
from backend.windows_font_evidence import (
    WindowsFontEvidenceError,
    WindowsFontEvidenceProvider,
    WindowsFontRegistryEntry,
)


FONT_KEY = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"


def _build_font(
    path: Path,
    *,
    family: str,
    style: str = "Regular",
    code_points: set[int],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    builder = FontBuilder(unitsPerEm=1_000, isTTF=True)
    glyph_names = [".notdef"] + [
        f"uni{code_point:06X}" for code_point in sorted(code_points)
    ]
    builder.setupGlyphOrder(glyph_names)

    glyphs: dict[str, Any] = {}
    metrics: dict[str, tuple[int, int]] = {}
    for glyph_name in glyph_names:
        pen = TTGlyphPen(None)
        if glyph_name != ".notdef":
            pen.moveTo((70, 0))
            pen.lineTo((530, 0))
            pen.lineTo((530, 700))
            pen.lineTo((70, 700))
            pen.closePath()
        glyphs[glyph_name] = pen.glyph()
        metrics[glyph_name] = (600, 0)

    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(metrics)
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupCharacterMap(
        {
            code_point: f"uni{code_point:06X}"
            for code_point in sorted(code_points)
        }
    )
    full_name = family if style == "Regular" else f"{family} {style}"
    builder.setupNameTable(
        {
            "familyName": family,
            "styleName": style,
            "uniqueFontIdentifier": f"{family}-{style}-fixture",
            "fullName": full_name,
            "psName": f"{family}-{style}".replace(" ", ""),
            "version": "Version 1.000",
        }
    )
    builder.setupOS2(
        sTypoAscender=800,
        sTypoDescender=-200,
        usWinAscent=800,
        usWinDescent=200,
    )
    builder.setupPost()
    builder.setupMaxp()
    builder.save(path)
    return path.resolve(strict=True)


def _entry(
    *,
    label: str,
    path: Path | str,
    hive: str = "HKLM",
) -> WindowsFontRegistryEntry:
    return WindowsFontRegistryEntry(
        hive=hive,
        key_path=FONT_KEY,
        view="registry64",
        value_name=f"{label} (TrueType)",
        value_data=str(path),
    )


def _cue(
    *,
    text: str,
    families: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "cueId": "cue-001",
        "styleId": "youtube-clean",
        "text": text,
        "requestedFontFamilies": families or ["Fixture Sans"],
    }


def _frames(tmp_path: Path) -> tuple[Path, Path]:
    rendered = tmp_path / "rendered.png"
    source = tmp_path / "source.png"
    rendered.write_bytes(b"rendered-frame-fixture")
    source.write_bytes(b"source-frame-fixture")
    return rendered.resolve(), source.resolve()


def _provider(
    tmp_path: Path,
    *,
    roots: list[Path],
    entries: list[WindowsFontRegistryEntry],
) -> WindowsFontEvidenceProvider:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return WindowsFontEvidenceProvider(
        registry_reader=lambda: tuple(entries),
        allowed_font_roots=roots,
        evidence_root=(tmp_path / "evidence").resolve(),
        platform_name="Windows",
    )


def _collect(
    provider: WindowsFontEvidenceProvider,
    *,
    cue: dict[str, Any],
    frames: tuple[Path, Path],
) -> FontEvidenceObservation:
    observation = provider.collect(
        cue=cue,
        frame_id="frame-001",
        timestamp_ms=1_250,
        rendered_frame_path=frames[0],
        source_frame_path=frames[1],
    )
    assert observation is not None
    return observation


def _artifact(observation: FontEvidenceObservation) -> dict[str, Any]:
    assert observation.evidence_artifact_path is not None
    return json.loads(
        Path(observation.evidence_artifact_path).read_text(encoding="utf-8")
    )


def test_collect_verifies_registered_font_metadata_hash_and_mixed_glyphs(
    tmp_path: Path,
) -> None:
    font_root = tmp_path / "fonts"
    text = "Hello 世界 ★ ©"
    code_points = {ord(character) for character in text if not character.isspace()}
    font = _build_font(
        font_root / "fixture-sans.ttf",
        family="Fixture Sans",
        code_points=code_points,
    )
    provider = _provider(
        tmp_path,
        roots=[font_root],
        entries=[_entry(label="Fixture Sans", path=font)],
    )

    observation = _collect(
        provider,
        cue=_cue(text=text),
        frames=_frames(tmp_path),
    )

    assert observation.resolved_family == "Fixture Sans"
    assert observation.resolution_verified is True
    assert observation.glyph_coverage_verified is True
    assert observation.verification_method == "font-cmap-and-shaping"
    assert observation.covered_renderable_code_points == len(
        [character for character in text if not character.isspace()]
    )
    assert observation.missing_code_points == ()
    assert observation.tofu_glyph_count == 0
    assert isinstance(observation.installation, VerifiedFontClaim)
    assert observation.installation.status == "verified-installed"
    assert Path(observation.installation.font_artifact_path) == font

    evidence = _artifact(observation)
    assert evidence["localDiscovery"]["networkUsed"] is False
    assert evidence["resolution"]["requestedFamily"] == "Fixture Sans"
    assert evidence["resolution"]["resolvedFamily"] == "Fixture Sans"
    assert evidence["resolution"]["style"] == "Regular"
    assert evidence["installation"]["registryLabelMatchesMetadata"] is True
    assert evidence["installation"]["fontArtifactSha256"] == hashlib.sha256(
        font.read_bytes()
    ).hexdigest()
    assert evidence["glyphCoverage"]["verified"] is True
    assert {
        "U+0048",
        "U+4E16",
        "U+754C",
        "U+2605",
        "U+00A9",
    } <= set(evidence["glyphCoverage"]["requestedUniqueCodePoints"])


def test_observation_normalizes_through_existing_visual_evidence_protocol(
    tmp_path: Path,
) -> None:
    font_root = tmp_path / "fonts"
    text = "A界★"
    font = _build_font(
        font_root / "protocol.ttf",
        family="Protocol Sans",
        code_points={ord(character) for character in text},
    )
    provider = _provider(
        tmp_path,
        roots=[font_root],
        entries=[_entry(label="Protocol Sans", path=font)],
    )
    rendered, source = _frames(tmp_path)
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.write_bytes(b"local-tool-fixture")
    ffprobe.write_bytes(b"local-tool-fixture")
    collector = SubtitleVisualEvidenceCollector(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        font_evidence_provider=provider,
    )

    normalized = collector._collect_font_evidence(
        cue=_cue(text=text, families=["Protocol Sans"]),
        frame_id="frame-001",
        timestamp_ms=1_250,
        rendered_frame=rendered,
        source_frame=source,
    )

    assert normalized is not None
    assert normalized["resolvedFamily"] == "Protocol Sans"
    assert normalized["resolutionVerified"] is True
    assert normalized["glyphCoverageVerified"] is True
    assert normalized["expectedRenderableCodePoints"] == 3
    assert normalized["coveredRenderableCodePoints"] == 3
    assert normalized["missingCodePoints"] == []
    assert normalized["installation"]["status"] == "verified-installed"
    assert normalized["installation"]["fontArtifactSha256"] == (
        hashlib.sha256(font.read_bytes()).hexdigest()
    )


def test_collect_reports_missing_cjk_and_symbol_without_false_success(
    tmp_path: Path,
) -> None:
    font_root = tmp_path / "fonts"
    font = _build_font(
        font_root / "latin-only.ttf",
        family="Fixture Sans",
        code_points={ord("A")},
    )
    provider = _provider(
        tmp_path,
        roots=[font_root],
        entries=[_entry(label="Fixture Sans", path=font)],
    )

    observation = _collect(
        provider,
        cue=_cue(text="A界★界"),
        frames=_frames(tmp_path),
    )

    assert observation.resolution_verified is True
    assert observation.glyph_coverage_verified is False
    assert observation.covered_renderable_code_points == 1
    assert observation.missing_code_points == ("U+2605", "U+754C")
    assert observation.tofu_glyph_count == 3
    evidence = _artifact(observation)
    assert evidence["glyphCoverage"] == {
        "coveredRenderableCodePoints": 1,
        "expectedRenderableCodePoints": 4,
        "method": "unicode-cmap-non-notdef-per-codepoint-v1",
        "missingCodePoints": ["U+2605", "U+754C"],
        "missingOccurrences": 3,
        "requestedUniqueCodePoints": ["U+0041", "U+2605", "U+754C"],
        "verified": False,
    }


def test_font_stack_uses_metadata_verified_complete_local_fallback(
    tmp_path: Path,
) -> None:
    font_root = tmp_path / "fonts"
    latin = _build_font(
        font_root / "latin.ttf",
        family="Latin Sans",
        code_points={ord("A")},
    )
    universal = _build_font(
        font_root / "universal.ttf",
        family="Universal Sans",
        code_points={ord("A"), ord("界"), ord("★")},
    )
    provider = _provider(
        tmp_path,
        roots=[font_root],
        entries=[
            _entry(label="Latin Sans", path=latin),
            _entry(label="Universal Sans", path=universal),
        ],
    )

    observation = _collect(
        provider,
        cue=_cue(
            text="A界★",
            families=["Latin Sans", "Universal Sans"],
        ),
        frames=_frames(tmp_path),
    )

    assert observation.resolved_family == "Universal Sans"
    assert observation.glyph_coverage_verified is True
    assert observation.covered_renderable_code_points == 3


def test_registry_font_label_cannot_spoof_parsed_family_or_style(
    tmp_path: Path,
) -> None:
    font_root = tmp_path / "fonts"
    actual = _build_font(
        font_root / "actual.ttf",
        family="Actual Sans",
        style="Regular",
        code_points={ord("A")},
    )
    frames = _frames(tmp_path)

    family_spoof = _provider(
        tmp_path / "family-spoof",
        roots=[font_root],
        entries=[_entry(label="Trusted Sans", path=actual)],
    )
    family_result = _collect(
        family_spoof,
        cue=_cue(text="A", families=["Trusted Sans"]),
        frames=frames,
    )
    assert family_result.resolution_verified is False
    assert family_result.resolved_family is None
    assert family_result.installation is None

    style_spoof = _provider(
        tmp_path / "style-spoof",
        roots=[font_root],
        entries=[_entry(label="Actual Sans Bold", path=actual)],
    )
    style_result = _collect(
        style_spoof,
        cue=_cue(text="A", families=["Actual Sans"]),
        frames=frames,
    )
    assert style_result.resolution_verified is False
    assert style_result.installation is None


def test_registered_path_must_remain_inside_trusted_root(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "trusted-fonts"
    trusted_root.mkdir()
    outside = _build_font(
        tmp_path / "outside" / "outside.ttf",
        family="Outside Sans",
        code_points={ord("A")},
    )
    provider = _provider(
        tmp_path,
        roots=[trusted_root],
        entries=[_entry(label="Outside Sans", path=outside)],
    )

    observation = _collect(
        provider,
        cue=_cue(text="A", families=["Outside Sans"]),
        frames=_frames(tmp_path),
    )

    assert observation.resolution_verified is False
    assert observation.installation is None


def test_relative_traversal_and_ambiguous_relative_font_are_rejected(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "fonts-a"
    second_root = tmp_path / "fonts-b"
    _build_font(
        first_root / "shared.ttf",
        family="Shared Sans",
        code_points={ord("A")},
    )
    _build_font(
        second_root / "shared.ttf",
        family="Shared Sans",
        code_points={ord("A")},
    )
    frames = _frames(tmp_path)

    ambiguous = _provider(
        tmp_path / "ambiguous",
        roots=[first_root, second_root],
        entries=[_entry(label="Shared Sans", path="shared.ttf")],
    )
    assert (
        _collect(
            ambiguous,
            cue=_cue(text="A", families=["Shared Sans"]),
            frames=frames,
        ).resolution_verified
        is False
    )

    traversal = _provider(
        tmp_path / "traversal",
        roots=[first_root],
        entries=[_entry(label="Shared Sans", path=r"..\shared.ttf")],
    )
    assert (
        _collect(
            traversal,
            cue=_cue(text="A", families=["Shared Sans"]),
            frames=frames,
        ).resolution_verified
        is False
    )


def test_symlinked_registered_font_is_rejected_when_supported(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    outside = _build_font(
        tmp_path / "outside" / "font.ttf",
        family="Linked Sans",
        code_points={ord("A")},
    )
    linked = trusted_root / "linked.ttf"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("font symlinks are unavailable on this host")
    provider = _provider(
        tmp_path,
        roots=[trusted_root],
        entries=[_entry(label="Linked Sans", path=linked)],
    )

    observation = _collect(
        provider,
        cue=_cue(text="A", families=["Linked Sans"]),
        frames=_frames(tmp_path),
    )

    assert observation.resolution_verified is False
    assert observation.installation is None


def test_font_content_change_invalidates_cached_hash_and_coverage(
    tmp_path: Path,
) -> None:
    font_root = tmp_path / "fonts"
    font_path = font_root / "mutable.ttf"
    font = _build_font(
        font_path,
        family="Mutable Sans",
        code_points={ord("A")},
    )
    provider = _provider(
        tmp_path,
        roots=[font_root],
        entries=[_entry(label="Mutable Sans", path=font)],
    )
    frames = _frames(tmp_path)
    cue = _cue(text="AB", families=["Mutable Sans"])

    first = _collect(provider, cue=cue, frames=frames)
    first_evidence = _artifact(first)
    assert first.glyph_coverage_verified is False
    assert first.missing_code_points == ("U+0042",)

    _build_font(
        font_path,
        family="Mutable Sans",
        code_points={ord("A"), ord("B")},
    )
    os.utime(font_path, None)

    second = _collect(provider, cue=cue, frames=frames)
    second_evidence = _artifact(second)
    assert second.glyph_coverage_verified is True
    assert second.missing_code_points == ()
    assert (
        first_evidence["installation"]["fontArtifactSha256"]
        != second_evidence["installation"]["fontArtifactSha256"]
    )
    assert (
        Path(first.evidence_artifact_path)
        != Path(second.evidence_artifact_path)
    )


def test_evidence_is_deterministic_for_identical_local_inputs(
    tmp_path: Path,
) -> None:
    font_root = tmp_path / "fonts"
    font = _build_font(
        font_root / "deterministic.ttf",
        family="Deterministic Sans",
        code_points={ord("A"), ord("界"), ord("★")},
    )
    provider = _provider(
        tmp_path,
        roots=[font_root],
        entries=[_entry(label="Deterministic Sans", path=font)],
    )
    frames = _frames(tmp_path)
    cue = _cue(text="A界★", families=["Deterministic Sans"])

    first = _collect(provider, cue=cue, frames=frames)
    first_path = Path(first.evidence_artifact_path)
    first_bytes = first_path.read_bytes()
    second = _collect(provider, cue=cue, frames=frames)

    assert first == second
    assert Path(second.evidence_artifact_path) == first_path
    assert Path(second.evidence_artifact_path).read_bytes() == first_bytes
    assert provider.descriptor.configuration_sha256 == (
        provider.descriptor.configuration_sha256
    )


def test_non_windows_host_override_fails_closed_without_registry_or_io(
    tmp_path: Path,
) -> None:
    registry_called = False

    def registry_reader() -> tuple[WindowsFontRegistryEntry, ...]:
        nonlocal registry_called
        registry_called = True
        raise AssertionError("non-Windows provider must not read the registry")

    provider = WindowsFontEvidenceProvider(
        registry_reader=registry_reader,
        allowed_font_roots=[tmp_path / "missing-fonts"],
        platform_name="Linux",
    )

    result = provider.collect(
        cue=_cue(text="A"),
        frame_id="frame-001",
        timestamp_ms=0,
        rendered_frame_path=(tmp_path / "missing-rendered.png").resolve(),
        source_frame_path=(tmp_path / "missing-source.png").resolve(),
    )

    assert result is None
    assert registry_called is False


@pytest.mark.parametrize(
    ("cue", "frame_id", "timestamp_ms", "same_frames", "message"),
    [
        (
            {"cueId": "cue", "styleId": "style", "text": "A"},
            "frame",
            0,
            False,
            "requestedFontFamilies",
        ),
        (
            _cue(text="A", families=["Fixture Sans", " fixture  sans "]),
            "frame",
            0,
            False,
            "trimmed non-empty",
        ),
        (
            _cue(text="A", families=["https://fonts.invalid/a.ttf"]),
            "frame",
            0,
            False,
            "names, not paths or URLs",
        ),
        (
            _cue(text="A"),
            " frame ",
            0,
            False,
            "trimmed non-empty",
        ),
        (
            _cue(text="A"),
            "frame",
            True,
            False,
            "timestamp_ms",
        ),
        (
            _cue(text="\ud800"),
            "frame",
            0,
            False,
            "non-scalar Unicode",
        ),
        (
            _cue(text="A"),
            "frame",
            0,
            True,
            "must be distinct",
        ),
    ],
)
def test_invalid_collect_input_fails_closed(
    tmp_path: Path,
    cue: dict[str, Any],
    frame_id: str,
    timestamp_ms: int,
    same_frames: bool,
    message: str,
) -> None:
    font_root = tmp_path / "fonts"
    font_root.mkdir()
    provider = _provider(tmp_path, roots=[font_root], entries=[])
    rendered, source = _frames(tmp_path)
    if same_frames:
        source = rendered

    with pytest.raises(WindowsFontEvidenceError, match=message):
        provider.collect(
            cue=cue,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            rendered_frame_path=rendered,
            source_frame_path=source,
        )


def test_invalid_registry_shape_and_malformed_font_fail_closed(
    tmp_path: Path,
) -> None:
    font_root = tmp_path / "fonts"
    font_root.mkdir()
    malformed = font_root / "malformed.ttf"
    malformed.write_bytes(b"this is not an OpenType font")
    frames = _frames(tmp_path)

    malformed_provider = _provider(
        tmp_path / "malformed",
        roots=[font_root],
        entries=[_entry(label="Malformed Sans", path=malformed)],
    )
    malformed_result = _collect(
        malformed_provider,
        cue=_cue(text="A", families=["Malformed Sans"]),
        frames=frames,
    )
    assert malformed_result.resolution_verified is False
    assert malformed_result.installation is None

    invalid_registry_provider = WindowsFontEvidenceProvider(
        registry_reader=lambda: (
            WindowsFontRegistryEntry(
                hive="HKCR",
                key_path=FONT_KEY,
                view="registry64",
                value_name="Bad Font",
                value_data=str(malformed),
            ),
        ),
        allowed_font_roots=[font_root],
        evidence_root=(tmp_path / "bad-registry-evidence").resolve(),
        platform_name="Windows",
    )
    with pytest.raises(WindowsFontEvidenceError, match="HKLM or HKCU"):
        invalid_registry_provider.collect(
            cue=_cue(text="A", families=["Malformed Sans"]),
            frame_id="frame-001",
            timestamp_ms=1,
            rendered_frame_path=frames[0],
            source_frame_path=frames[1],
        )
