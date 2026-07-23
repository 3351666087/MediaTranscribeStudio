from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from backend.output_customization import (
    OUTPUT_CUSTOMIZATION_KIND,
    OUTPUT_CUSTOMIZATION_SCHEMA_VERSION,
    OutputCustomizationChange,
    OutputCustomizationError,
    apply_output_customization_patch,
    canonical_output_customization_dict,
    canonical_output_customization_json,
    default_output_customization,
    deterministic_output_customization_hash,
    resolve_output_customization,
    validate_output_customization,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "output-customization.schema.json"
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _payload() -> dict[str, Any]:
    return default_output_customization().canonical_dict()


def _assert_domain_error(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(OutputCustomizationError, match=match):
        validate_output_customization(payload)


def _font_evidence(method: str = "package-manifest") -> dict[str, Any]:
    return {
        "verified": True,
        "method": method,
        "manifestSha256": SHA_A,
        "fontFileSha256": [SHA_C, SHA_B],
    }


def _karaoke_evidence() -> dict[str, Any]:
    return {
        "source": "forced-aligner",
        "verified": True,
        "subtitleSha256": SHA_A,
        "transcriptSha256": SHA_B,
        "wordCount": 42,
    }


def _range_evidence() -> dict[str, Any]:
    return {
        "verified": True,
        "method": "ffprobe-color-metadata",
        "probeSha256": SHA_C,
    }


def test_schema_is_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(_schema())


def test_safe_defaults_are_complete_schema_valid_and_not_execution_ready() -> None:
    snapshot = default_output_customization()
    payload = snapshot.canonical_dict()

    _validator().validate(payload)
    assert payload["schemaVersion"] == OUTPUT_CUSTOMIZATION_SCHEMA_VERSION
    assert payload["kind"] == OUTPUT_CUSTOMIZATION_KIND
    assert payload["profile"] == "studio-balanced"
    assert payload["report"]["layout"] == "modern-editorial"
    assert payload["report"]["density"] == "comfortable"
    assert payload["report"]["paper"]["size"] == "A4"
    assert payload["subtitle"]["theme"] == "youtube-clean"
    assert payload["subtitle"]["format"] == "ass"
    assert payload["delivery"]["mode"] == "sidecar"
    assert payload["delivery"]["executionReady"] is False
    assert payload["delivery"]["outputTarget"] == {
        "binding": "deferred",
        "sourcePath": None,
        "outputPath": None,
    }
    assert payload["safety"]["preserveSourceMedia"] is True
    assert payload["safety"]["overwriteSourceMedia"] is False
    assert payload["safety"]["rejectHdrBurnIn"] is True
    assert payload["safety"]["rejectFakeKaraoke"] is True
    assert payload["reversibility"]["workflowReversible"] is True
    assert (
        payload["reversibility"]["derivedMediaBitstreamReversible"] is True
    )


def test_canonical_json_and_hash_are_deterministic() -> None:
    first = default_output_customization()
    second = resolve_output_customization({})

    assert first.canonical_json() == second.canonical_json()
    assert first.deterministic_hash() == second.deterministic_hash()
    assert ": " not in first.canonical_json()
    assert ", " not in first.canonical_json()
    assert "\n" not in first.canonical_json()
    assert first.deterministic_hash() == hashlib.sha256(
        first.canonical_json().encode("utf-8")
    ).hexdigest()


def test_canonical_helper_functions_match_snapshot_methods() -> None:
    payload = _payload()
    snapshot = validate_output_customization(payload)

    assert canonical_output_customization_dict(payload) == payload
    assert canonical_output_customization_json(payload) == snapshot.canonical_json()
    assert (
        deterministic_output_customization_hash(payload)
        == snapshot.deterministic_hash()
    )


def test_canonical_dict_is_detached_from_immutable_snapshot() -> None:
    snapshot = default_output_customization()
    first = snapshot.canonical_dict()
    first["report"]["layout"] = "compact-review"

    assert snapshot.canonical_dict()["report"]["layout"] == "modern-editorial"


def test_resolver_does_not_mutate_caller_overrides() -> None:
    overrides = {
        "report": {"accentColor": "#aabbcc"},
        "exports": {"transcriptFormats": ["pdf", "json"]},
    }
    before = copy.deepcopy(overrides)

    resolved = resolve_output_customization(overrides).canonical_dict()

    assert overrides == before
    assert resolved["report"]["accentColor"] == "#AABBCC"
    assert resolved["exports"]["transcriptFormats"] == ["json", "pdf"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"unknown": True}, "unknown property"),
        ({"report": {"unknown": True}}, "unknown property"),
        ({"subtitle": {"safeArea": {"leftPercent": 5}}}, "unknown property"),
        ({"delivery": {"burnIn": {"allowHdr": True}}}, "unknown property"),
        ({"safety": {"unsafeEscapeHatch": True}}, "unknown property"),
    ],
)
def test_partial_resolver_rejects_unknown_properties(
    overrides: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(OutputCustomizationError, match=match):
        resolve_output_customization(overrides)


def test_complete_validator_rejects_missing_and_unknown_properties() -> None:
    missing = _payload()
    del missing["report"]["density"]
    _assert_domain_error(missing, "report is missing: density")

    unknown = _payload()
    unknown["subtitle"]["speakerColors"]["speakerCountLimit"] = 5
    _assert_domain_error(unknown, "unknown properties")


def test_normalization_uses_nfc_uppercase_colors_and_semantic_list_order() -> None:
    decomposed = "Cafe\u0301"
    payload = resolve_output_customization(
        {
            "report": {"header": {"text": decomposed}},
            "subtitle": {
                "speakerColors": {
                    "overrides": [
                        {"speakerId": "speaker-z", "color": "#abcdef"},
                        {"speakerId": "speaker-a", "color": "#123abc"},
                    ]
                }
            },
            "exports": {
                "reportFormats": ["docx", "pdf", "html"],
                "transcriptFormats": ["pdf", "json", "markdown"],
                "dataFormats": ["tsv", "json"],
            },
        }
    ).canonical_dict()

    assert payload["report"]["header"]["text"] == "Café"
    assert payload["subtitle"]["speakerColors"]["overrides"] == [
        {"speakerId": "speaker-a", "color": "#123ABC"},
        {"speakerId": "speaker-z", "color": "#ABCDEF"},
    ]
    assert payload["exports"]["reportFormats"] == ["pdf", "html", "docx"]
    assert payload["exports"]["transcriptFormats"] == [
        "json",
        "markdown",
        "pdf",
    ]
    assert payload["exports"]["dataFormats"] == ["json", "tsv"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numbers_fail_closed(bad: float) -> None:
    payload = _payload()
    payload["report"]["paper"]["marginsMm"]["top"] = bad
    _assert_domain_error(payload, "must be finite")


@pytest.mark.parametrize(
    "layout",
    [
        "modern-editorial",
        "conversation-focus",
        "executive-brief",
        "compact-review",
        "accessible-large-print",
        "archive-monochrome",
    ],
)
def test_all_existing_report_layouts_are_supported(layout: str) -> None:
    payload = resolve_output_customization(
        {"report": {"layout": layout}}
    ).canonical_dict()
    _validator().validate(payload)
    assert payload["report"]["layout"] == layout


def test_report_diy_controls_are_materialized() -> None:
    payload = resolve_output_customization(
        {
            "profile": "custom",
            "report": {
                "layout": "conversation-focus",
                "density": "relaxed",
                "paper": {
                    "size": "Letter",
                    "orientation": "landscape",
                    "marginsMm": {
                        "top": 12,
                        "right": 13,
                        "bottom": 14,
                        "left": 15,
                    },
                },
                "header": {
                    "text": "Interview archive",
                    "showChapter": True,
                },
                "footer": {"showDocumentTitle": True},
                "speakerLegend": {
                    "position": "appendix",
                    "showSpeakingTime": True,
                },
                "accentColor": "#146EF5",
                "highContrast": True,
            },
        }
    ).canonical_dict()

    _validator().validate(payload)
    assert payload["profile"] == "custom"
    assert payload["report"]["paper"]["orientation"] == "landscape"
    assert payload["report"]["header"]["showChapter"] is True
    assert payload["report"]["speakerLegend"]["position"] == "appendix"
    assert payload["report"]["accentColor"] == "#146EF5"


def test_cover_disable_is_explicit_and_consistent() -> None:
    payload = resolve_output_customization(
        {"report": {"cover": {"enabled": False, "style": "none"}}}
    ).canonical_dict()
    _validator().validate(payload)
    assert payload["report"]["cover"]["enabled"] is False

    inconsistent = _payload()
    inconsistent["report"]["cover"]["enabled"] = False
    _assert_domain_error(inconsistent, "disabled report.cover")


def test_cover_artwork_requires_hash_bound_asset_identity() -> None:
    artwork = {
        "assetId": "day-background-v2",
        "sha256": SHA_A,
        "fit": "cover",
        "opacity": 0.85,
    }
    payload = resolve_output_customization(
        {"report": {"cover": {"artwork": artwork}}}
    ).canonical_dict()
    _validator().validate(payload)
    assert payload["report"]["cover"]["artwork"] == artwork

    bad = _payload()
    bad["report"]["cover"]["artwork"] = {
        **artwork,
        "sha256": "not-evidence",
    }
    _assert_domain_error(bad, "lowercase SHA-256")


def test_enabled_watermark_requires_visible_text() -> None:
    with pytest.raises(OutputCustomizationError, match="requires visible text"):
        resolve_output_customization(
            {"report": {"watermark": {"enabled": True}}}
        )

    payload = resolve_output_customization(
        {
            "report": {
                "watermark": {
                    "enabled": True,
                    "text": "CONFIDENTIAL",
                    "repeat": "grid",
                    "opacity": 0.12,
                }
            }
        }
    ).canonical_dict()
    _validator().validate(payload)


def test_fixed_interval_chapters_require_bounded_interval() -> None:
    with pytest.raises(OutputCustomizationError, match="must be a number"):
        resolve_output_customization(
            {"report": {"chapters": {"source": "fixed-interval"}}}
        )

    payload = resolve_output_customization(
        {
            "report": {
                "chapters": {
                    "source": "fixed-interval",
                    "intervalMinutes": 15,
                    "pageBreakBefore": True,
                }
            }
        }
    ).canonical_dict()
    assert payload["report"]["chapters"]["intervalMinutes"] == 15.0
    _validator().validate(payload)


@pytest.mark.parametrize(
    ("timestamp_format", "frame_rate"),
    [
        ("smpte-non-drop", 24),
        ("smpte-non-drop", 25),
        ("smpte-drop-frame", 29.97),
        ("smpte-drop-frame", 59.94),
    ],
)
def test_supported_smpte_timestamp_formats(
    timestamp_format: str,
    frame_rate: float,
) -> None:
    payload = resolve_output_customization(
        {
            "report": {
                "timestamp": {
                    "format": timestamp_format,
                    "frameRate": frame_rate,
                }
            }
        }
    ).canonical_dict()
    _validator().validate(payload)
    assert payload["report"]["timestamp"]["frameRate"] == float(frame_rate)


def test_timestamp_frame_rate_cross_field_rules_fail_closed() -> None:
    with pytest.raises(OutputCustomizationError, match="only valid for SMPTE"):
        resolve_output_customization(
            {"report": {"timestamp": {"frameRate": 24}}}
        )
    with pytest.raises(OutputCustomizationError, match="29.97 or 59.94"):
        resolve_output_customization(
            {
                "report": {
                    "timestamp": {
                        "format": "smpte-drop-frame",
                        "frameRate": 24,
                    }
                }
            }
        )


@pytest.mark.parametrize("font_path", ["report", "subtitle"])
def test_verified_font_claim_requires_sha256_bound_evidence(
    font_path: str,
) -> None:
    with pytest.raises(OutputCustomizationError, match="require.*evidence"):
        resolve_output_customization(
            {font_path: {"fontPack": {"availabilityClaim": "verified-packaged"}}}
        )

    payload = resolve_output_customization(
        {
            font_path: {
                "fontPack": {
                    "availabilityClaim": "verified-packaged",
                    "embeddingClaim": "verified-embedded",
                    "evidence": _font_evidence("pdfbox-font-inspection"),
                }
            }
        }
    ).canonical_dict()
    _validator().validate(payload)
    assert payload[font_path]["fontPack"]["evidence"]["fontFileSha256"] == [
        SHA_B,
        SHA_C,
    ]


def test_font_evidence_cannot_exist_without_a_verified_claim() -> None:
    with pytest.raises(OutputCustomizationError, match="requires a verified"):
        resolve_output_customization(
            {"report": {"fontPack": {"evidence": _font_evidence()}}}
        )


def test_false_font_verification_and_duplicate_fallbacks_are_rejected() -> None:
    evidence = _font_evidence()
    evidence["verified"] = False
    with pytest.raises(OutputCustomizationError, match="verified must be true"):
        resolve_output_customization(
            {
                "report": {
                    "fontPack": {
                        "availabilityClaim": "verified-packaged",
                        "evidence": evidence,
                    }
                }
            }
        )

    with pytest.raises(OutputCustomizationError, match="duplicates"):
        resolve_output_customization(
            {
                "subtitle": {
                    "fontPack": {
                        "fallbacks": {"latin": ["Inter", "Inter"]}
                    }
                }
            }
        )


def test_any_number_of_speaker_color_overrides_is_supported() -> None:
    overrides = [
        {
            "speakerId": f"speaker-{index:04d}",
            "color": f"#{index % 0xFFFFFF:06x}",
        }
        for index in range(1024)
    ]
    payload = resolve_output_customization(
        {"subtitle": {"speakerColors": {"overrides": list(reversed(overrides))}}}
    ).canonical_dict()

    _validator().validate(payload)
    assert len(payload["subtitle"]["speakerColors"]["overrides"]) == 1024
    assert payload["subtitle"]["speakerColors"]["overrides"][0]["speakerId"] == (
        "speaker-0000"
    )
    assert payload["subtitle"]["speakerColors"]["overrides"][-1][
        "speakerId"
    ] == "speaker-1023"
    assert "speakerCount" not in payload["subtitle"]["speakerColors"]
    assert "maximumSpeakers" not in payload["subtitle"]["speakerColors"]


def test_duplicate_speaker_identifiers_fail_even_with_different_colors() -> None:
    with pytest.raises(OutputCustomizationError, match="repeat speakerId"):
        resolve_output_customization(
            {
                "subtitle": {
                    "speakerColors": {
                        "overrides": [
                            {"speakerId": "speaker-1", "color": "#FF0000"},
                            {"speakerId": "speaker-1", "color": "#00FF00"},
                        ]
                    }
                }
            }
        )


@pytest.mark.parametrize(
    ("mode", "algorithm", "minimum_delta"),
    [
        ("automatic", "oklch-hash-v1", 18),
        ("accessible", "accessible-oklch-hash-v1", 24),
        ("monochrome", "monochrome-v1", 0),
    ],
)
def test_speaker_color_modes_require_compatible_algorithms(
    mode: str,
    algorithm: str,
    minimum_delta: float,
) -> None:
    payload = resolve_output_customization(
        {
            "subtitle": {
                "speakerColors": {
                    "mode": mode,
                    "algorithm": algorithm,
                    "minimumDeltaE": minimum_delta,
                }
            }
        }
    ).canonical_dict()
    _validator().validate(payload)

    bad = copy.deepcopy(payload)
    bad["subtitle"]["speakerColors"]["algorithm"] = "oklch-hash-v1"
    if mode != "automatic":
        _assert_domain_error(bad, "incompatible with mode")


def test_karaoke_is_off_by_default_and_rejects_unbound_evidence() -> None:
    payload = _payload()
    assert payload["subtitle"]["karaoke"] == {"mode": "off", "evidence": None}

    payload["subtitle"]["karaoke"]["evidence"] = _karaoke_evidence()
    _assert_domain_error(payload, "must be null")


def test_word_progress_requires_exact_hash_bound_timing_and_ass() -> None:
    with pytest.raises(OutputCustomizationError, match="must be an object"):
        resolve_output_customization(
            {"subtitle": {"karaoke": {"mode": "word-progress"}}}
        )

    payload = resolve_output_customization(
        {
            "subtitle": {
                "theme": "karaoke-highlight",
                "karaoke": {
                    "mode": "word-progress",
                    "evidence": _karaoke_evidence(),
                },
            }
        }
    ).canonical_dict()
    _validator().validate(payload)
    assert payload["subtitle"]["format"] == "ass"

    with pytest.raises(OutputCustomizationError, match="requires.*ass"):
        resolve_output_customization(
            {
                "subtitle": {
                    "format": "webvtt",
                    "karaoke": {
                        "mode": "word-progress",
                        "evidence": _karaoke_evidence(),
                    },
                },
                "exports": {"subtitleAlternates": ["srt", "ass"]},
            }
        )


@pytest.mark.parametrize(
    "source",
    ["segment-interpolation", "synthetic-even-split", "language-model"],
)
def test_fake_karaoke_timing_sources_are_rejected(source: str) -> None:
    evidence = _karaoke_evidence()
    evidence["source"] = source
    with pytest.raises(OutputCustomizationError, match="must be one of"):
        resolve_output_customization(
            {
                "subtitle": {
                    "karaoke": {
                        "mode": "word-progress",
                        "evidence": evidence,
                    }
                }
            }
        )


def test_karaoke_theme_name_cannot_fake_word_progress() -> None:
    with pytest.raises(
        OutputCustomizationError,
        match="requires verified word-progress",
    ):
        resolve_output_customization(
            {"subtitle": {"theme": "karaoke-highlight"}}
        )


def test_bound_output_paths_become_execution_ready() -> None:
    payload = resolve_output_customization(
        {
            "delivery": {
                "mode": "soft-mux",
                "outputTarget": {
                    "binding": "bound",
                    "sourcePath": r"D:\media\source.mov",
                    "outputPath": r"D:\exports\source-subtitled.mkv",
                },
            }
        }
    ).canonical_dict()

    assert payload["delivery"]["executionReady"] is True
    assert (
        payload["reversibility"]["derivedMediaBitstreamReversible"] is True
    )
    _validator().validate(payload)


@pytest.mark.parametrize(
    ("source", "output"),
    [
        (r"D:\media\source.mov", r"d:\MEDIA\.\source.mov"),
        (r"\\server\share\a.mov", r"\\SERVER\SHARE\folder\..\a.mov"),
        ("/tmp/media/a.mov", "/tmp/media/./a.mov"),
    ],
)
def test_source_output_aliases_are_rejected(
    source: str,
    output: str,
) -> None:
    with pytest.raises(OutputCustomizationError, match="cannot alias"):
        resolve_output_customization(
            {
                "delivery": {
                    "outputTarget": {
                        "binding": "bound",
                        "sourcePath": source,
                        "outputPath": output,
                    }
                }
            }
        )


def test_deferred_paths_must_both_be_null() -> None:
    payload = _payload()
    payload["delivery"]["outputTarget"]["sourcePath"] = "source.mov"
    _assert_domain_error(payload, "deferred delivery paths")


def test_execution_ready_is_derived_not_user_asserted() -> None:
    payload = _payload()
    payload["delivery"]["executionReady"] = True
    _assert_domain_error(payload, "must exactly reflect")

    with pytest.raises(OutputCustomizationError, match="must exactly reflect"):
        resolve_output_customization({"delivery": {"executionReady": True}})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("overwriteExisting", True, "must remain false"),
        ("atomicPublication", False, "must remain true"),
    ],
)
def test_delivery_publication_safety_cannot_be_disabled(
    field: str,
    value: bool,
    match: str,
) -> None:
    payload = _payload()
    payload["delivery"][field] = value
    _assert_domain_error(payload, match)


@pytest.mark.parametrize(
    "strategy",
    [
        "h264-high-quality",
        "h265-high-quality",
        "vp9-high-quality",
        "av1-high-quality",
        "prores-422-hq",
        "ffv1-lossless",
    ],
)
def test_burn_in_requires_explicit_strategy_verified_sdr_and_visual_qa(
    strategy: str,
) -> None:
    payload = resolve_output_customization(
        {
            "delivery": {
                "mode": "burn-in",
                "burnIn": {
                    "strategy": strategy,
                    "sourceDynamicRange": "sdr",
                    "dynamicRangeEvidence": _range_evidence(),
                },
            }
        }
    ).canonical_dict()

    assert (
        payload["reversibility"]["derivedMediaBitstreamReversible"] is False
    )
    assert payload["delivery"]["burnIn"]["requireVisualQa"] is True
    _validator().validate(payload)


def test_burn_in_fails_closed_for_unknown_or_hdr_sources() -> None:
    with pytest.raises(OutputCustomizationError, match="verified SDR"):
        resolve_output_customization(
            {
                "delivery": {
                    "mode": "burn-in",
                    "burnIn": {"strategy": "h264-high-quality"},
                }
            }
        )

    with pytest.raises(OutputCustomizationError, match="verified SDR"):
        resolve_output_customization(
            {
                "delivery": {
                    "mode": "burn-in",
                    "burnIn": {
                        "strategy": "h265-high-quality",
                        "sourceDynamicRange": "hdr",
                        "dynamicRangeEvidence": _range_evidence(),
                    },
                }
            }
        )


def test_burn_in_strategy_cannot_hide_in_non_burn_delivery() -> None:
    with pytest.raises(OutputCustomizationError, match="must be null"):
        resolve_output_customization(
            {
                "delivery": {
                    "burnIn": {
                        "strategy": "h264-high-quality",
                        "sourceDynamicRange": "sdr",
                        "dynamicRangeEvidence": _range_evidence(),
                    }
                }
            }
        )


def test_dynamic_range_claims_are_evidence_bound() -> None:
    with pytest.raises(OutputCustomizationError, match="must be an object"):
        resolve_output_customization(
            {"delivery": {"burnIn": {"sourceDynamicRange": "sdr"}}}
        )

    with pytest.raises(OutputCustomizationError, match="verified must be true"):
        evidence = _range_evidence()
        evidence["verified"] = False
        resolve_output_customization(
            {
                "delivery": {
                    "burnIn": {
                        "sourceDynamicRange": "hdr",
                        "dynamicRangeEvidence": evidence,
                    }
                }
            }
        )


def test_report_and_subtitle_export_consistency_is_strict() -> None:
    with pytest.raises(OutputCustomizationError, match="enabled reports"):
        resolve_output_customization({"exports": {"reportFormats": []}})

    payload = resolve_output_customization(
        {
            "report": {"enabled": False},
            "exports": {"reportFormats": []},
        }
    ).canonical_dict()
    _validator().validate(payload)

    with pytest.raises(OutputCustomizationError, match="disabled subtitles"):
        resolve_output_customization(
            {
                "subtitle": {"enabled": False},
            }
        )

    payload = resolve_output_customization(
        {
            "subtitle": {"enabled": False},
            "exports": {"subtitleAlternates": []},
        }
    ).canonical_dict()
    _validator().validate(payload)


def test_primary_subtitle_format_cannot_be_duplicated_as_alternate() -> None:
    with pytest.raises(OutputCustomizationError, match="cannot repeat"):
        resolve_output_customization(
            {"exports": {"subtitleAlternates": ["srt", "ass"]}}
        )


def test_export_lists_reject_duplicates_and_unknown_formats() -> None:
    with pytest.raises(OutputCustomizationError, match="duplicates"):
        resolve_output_customization(
            {"exports": {"transcriptFormats": ["json", "json"]}}
        )
    with pytest.raises(OutputCustomizationError, match="must be one of"):
        resolve_output_customization(
            {"exports": {"dataFormats": ["json", "pickle"]}}
        )


@pytest.mark.parametrize(
    "template",
    [
        "{sourceStem}-{artifact}",
        "{date}-{sourceStem}-{language}-{speakerCount}",
        "{artifact}",
    ],
)
def test_safe_filename_templates_are_supported(template: str) -> None:
    payload = resolve_output_customization(
        {"exports": {"fileNameTemplate": template}}
    ).canonical_dict()
    _validator().validate(payload)
    assert payload["exports"]["fileNameTemplate"] == template


@pytest.mark.parametrize(
    ("template", "match"),
    [
        ("fixed-name", "requires at least"),
        ("../{artifact}", "path separators"),
        (r"..\{artifact}", "path separators"),
        ("{unknown}", "unknown tokens"),
        ("{{artifact}", "malformed"),
    ],
)
def test_unsafe_or_ambiguous_filename_templates_fail(
    template: str,
    match: str,
) -> None:
    with pytest.raises(OutputCustomizationError, match=match):
        resolve_output_customization(
            {"exports": {"fileNameTemplate": template}}
        )


@pytest.mark.parametrize(
    ("key", "unsafe_value"),
    [
        ("preserveSourceMedia", False),
        ("overwriteSourceMedia", True),
        ("overwriteExistingOutputs", True),
        ("transcriptAuthoritative", False),
        ("presentationOnly", False),
        ("fontClaimsRequireEvidence", False),
        ("rejectFakeKaraoke", False),
        ("rejectHdrBurnIn", False),
        ("visualQaRequiredForBurnIn", False),
    ],
)
def test_safety_invariants_cannot_be_overridden(
    key: str,
    unsafe_value: bool,
) -> None:
    with pytest.raises(OutputCustomizationError, match=rf"safety\.{key}"):
        resolve_output_customization({"safety": {key: unsafe_value}})


def test_reversibility_claims_must_match_delivery_mode() -> None:
    payload = _payload()
    payload["reversibility"]["derivedMediaBitstreamReversible"] = False
    _assert_domain_error(payload, "must match")

    with pytest.raises(OutputCustomizationError, match="must match"):
        resolve_output_customization(
            {
                "reversibility": {
                    "derivedMediaBitstreamReversible": False,
                }
            }
        )


def test_reversible_change_restores_exact_hash_bound_before_snapshot() -> None:
    before = default_output_customization()
    change = apply_output_customization_patch(
        before,
        {
            "report": {
                "density": "compact",
                "accentColor": "#00aaff",
            },
            "subtitle": {"fontSizePx": 64},
        },
    )

    change.verify()
    after = change.after()
    assert after.canonical_dict()["report"]["density"] == "compact"
    assert after.canonical_dict()["report"]["accentColor"] == "#00AAFF"
    assert change.before_sha256 == before.deterministic_hash()
    assert change.after_sha256 == after.deterministic_hash()
    restored = change.revert(after)
    assert restored.canonical_json() == before.canonical_json()
    assert restored.deterministic_hash() == before.deterministic_hash()


def test_revert_rejects_wrong_current_snapshot() -> None:
    before = default_output_customization()
    first = apply_output_customization_patch(
        before,
        {"report": {"density": "compact"}},
    )
    unrelated = resolve_output_customization(
        {"report": {"density": "relaxed"}}
    )

    with pytest.raises(OutputCustomizationError, match="does not match"):
        first.revert(unrelated)


def test_change_receipt_detects_hash_and_json_tampering() -> None:
    change = apply_output_customization_patch(
        default_output_customization(),
        {"subtitle": {"fontSizePx": 60}},
    )
    bad_hash = replace(change, after_sha256="0" * 64)
    with pytest.raises(OutputCustomizationError, match="hash mismatch"):
        bad_hash.verify()

    noncanonical_after = json.dumps(
        json.loads(change.after_json),
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )
    noncanonical = replace(
        change,
        after_json=noncanonical_after,
        after_sha256=hashlib.sha256(noncanonical_after.encode("utf-8")).hexdigest(),
    )
    with pytest.raises(OutputCustomizationError, match="not canonical"):
        noncanonical.verify()


def test_change_receipt_rejects_non_customization_current() -> None:
    change = apply_output_customization_patch(
        default_output_customization(),
        {"report": {"density": "compact"}},
    )
    with pytest.raises(OutputCustomizationError, match="current must be"):
        change.revert(object())  # type: ignore[arg-type]


def test_schema_rejects_unknown_fields_and_unsafe_constants() -> None:
    validator = _validator()

    unknown = _payload()
    unknown["delivery"]["shell"] = True
    assert list(validator.iter_errors(unknown))

    overwrite = _payload()
    overwrite["safety"]["overwriteSourceMedia"] = True
    assert list(validator.iter_errors(overwrite))

    fake_karaoke = _payload()
    fake_karaoke["subtitle"]["karaoke"] = {
        "mode": "word-progress",
        "evidence": {
            **_karaoke_evidence(),
            "source": "synthetic-even-split",
        },
    }
    assert list(validator.iter_errors(fake_karaoke))

    hdr_burn = _payload()
    hdr_burn["delivery"]["mode"] = "burn-in"
    hdr_burn["delivery"]["burnIn"] = {
        "strategy": "h265-high-quality",
        "sourceDynamicRange": "hdr",
        "dynamicRangeEvidence": _range_evidence(),
        "requireVisualQa": True,
    }
    hdr_burn["reversibility"]["derivedMediaBitstreamReversible"] = False
    assert list(validator.iter_errors(hdr_burn))


def test_schema_rejects_unsupported_verified_font_claim_without_evidence() -> None:
    payload = _payload()
    payload["report"]["fontPack"]["availabilityClaim"] = "verified-installed"
    assert list(_validator().iter_errors(payload))


def test_bound_paths_runtime_gate_is_stronger_than_portable_json_schema() -> None:
    payload = _payload()
    payload["delivery"]["outputTarget"] = {
        "binding": "bound",
        "sourcePath": r"D:\Media\source.mov",
        "outputPath": r"d:\media\.\source.mov",
    }
    payload["delivery"]["executionReady"] = True

    _validator().validate(payload)
    _assert_domain_error(payload, "cannot alias")


def test_control_characters_are_rejected_from_paths_and_text() -> None:
    with pytest.raises(OutputCustomizationError, match="control characters"):
        resolve_output_customization(
            {"report": {"header": {"text": "unsafe\u0000header"}}}
        )

    with pytest.raises(OutputCustomizationError, match="control characters"):
        resolve_output_customization(
            {
                "delivery": {
                    "outputTarget": {
                        "binding": "bound",
                        "sourcePath": "source.mov",
                        "outputPath": "output\u0000.mov",
                    }
                }
            }
        )


def test_boolean_is_not_accepted_as_integer_or_number() -> None:
    payload = _payload()
    payload["subtitle"]["fontSizePx"] = True
    _assert_domain_error(payload, "must be an integer")

    payload = _payload()
    payload["subtitle"]["lineHeight"] = False
    _assert_domain_error(payload, "must be a number")


def test_full_custom_configuration_remains_schema_valid() -> None:
    payload = resolve_output_customization(
        {
            "profile": "accessibility",
            "report": {
                "layout": "accessible-large-print",
                "density": "relaxed",
                "paper": {
                    "size": "Legal",
                    "orientation": "landscape",
                    "marginsMm": {
                        "top": 22,
                        "right": 22,
                        "bottom": 22,
                        "left": 22,
                    },
                },
                "cover": {
                    "style": "immersive",
                    "artwork": {
                        "assetId": "night-scene",
                        "sha256": SHA_A,
                        "fit": "cover",
                        "opacity": 0.7,
                    },
                },
                "watermark": {
                    "enabled": True,
                    "text": "REVIEW COPY",
                    "repeat": "diagonal",
                },
                "timestamp": {
                    "format": "smpte-drop-frame",
                    "frameRate": 29.97,
                    "showEnd": True,
                },
                "highContrast": True,
            },
            "subtitle": {
                "theme": "speaker-color",
                "fontSizePx": 72,
                "fontWeight": 800,
                "safeArea": {
                    "horizontalPercent": 8,
                    "topPercent": 7,
                    "bottomPercent": 10,
                },
                "speakerColors": {
                    "mode": "accessible",
                    "algorithm": "accessible-oklch-hash-v1",
                    "minimumDeltaE": 24,
                    "overrides": [
                        {"speakerId": "host", "color": "#55CCFF"},
                        {"speakerId": "guest", "color": "#FFCC55"},
                    ],
                },
            },
            "delivery": {
                "mode": "soft-mux",
                "outputTarget": {
                    "binding": "bound",
                    "sourcePath": r"D:\Media\meeting.mov",
                    "outputPath": r"D:\Exports\meeting-captioned.mkv",
                },
                "softMux": {
                    "container": "matroska",
                    "subtitleCodec": "ass",
                },
            },
            "exports": {
                "reportFormats": ["docx", "pdf", "html"],
                "transcriptFormats": [
                    "pdf",
                    "docx",
                    "html",
                    "markdown",
                    "txt",
                    "json",
                ],
                "dataFormats": ["tsv", "csv", "json"],
                "packageFormat": "zip",
                "fileNameTemplate": "{date}-{sourceStem}-{artifact}-{language}",
            },
        }
    ).canonical_dict()

    _validator().validate(payload)
    round_trip = validate_output_customization(payload)
    assert round_trip.canonical_dict() == payload
    assert round_trip.deterministic_hash() == hashlib.sha256(
        round_trip.canonical_json().encode("utf-8")
    ).hexdigest()


def test_output_customization_change_type_is_frozen_and_explicit() -> None:
    change = apply_output_customization_patch(
        default_output_customization(),
        {"report": {"density": "compact"}},
    )
    assert isinstance(change, OutputCustomizationChange)
    with pytest.raises(Exception):
        change.after_sha256 = SHA_A  # type: ignore[misc]
