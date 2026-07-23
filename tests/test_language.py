from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.documents import assemble_transcript_document
from backend.errors import WorkerError
from backend.language import (
    MULTIPLE_LANGUAGES,
    UNDETERMINED_LANGUAGE,
    normalize_language_tag,
    normalize_qwen_language_candidates,
    qwen_language_for_request,
    qwen_supported_primary_language_tags,
    reconcile_detected_languages,
)
from backend.models import (
    SpeakerCountEstimate,
    SpeakerCountMode,
    SpeakerCountPolicy,
    SpeakerScore,
    TranscriptSegment,
    TranscriptionResult,
)


class LanguageTagTests(unittest.TestCase):
    def test_canonicalizes_practical_bcp47_forms(self) -> None:
        cases = {
            "en-US": "en-US",
            "es-419": "es-419",
            "fr": "fr",
            "ja-JP": "ja-JP",
            "ar": "ar",
            "zh-Hans": "zh-Hans",
            "sr_latn_rs": "sr-Latn-RS",
            "de-DE-u-co-phonebk": "de-DE-u-co-phonebk",
            "zh-cmn-Hans-CN": "zh-cmn-Hans-CN",
            "x-PrIvAtE": "x-private",
            "UND": "und",
            "MUL": "mul",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_language_tag(value), expected)

    def test_canonicalizes_variants_extensions_and_private_use(self) -> None:
        self.assertEqual(
            normalize_language_tag(
                "SL_ROZAJ_BISKE_1994_A_EXT1_0_FOO_X_PrIvAtE"
            ),
            "sl-rozaj-biske-1994-a-ext1-0-foo-x-private",
        )

    def test_auto_is_request_only(self) -> None:
        self.assertEqual(
            normalize_language_tag("AUTO", allow_auto=True),
            "auto",
        )
        with self.assertRaisesRegex(ValueError, "request input"):
            normalize_language_tag("auto")

    def test_rejects_malformed_language_tags(self) -> None:
        malformed = (
            "",
            " en-US",
            "en-US ",
            "123",
            "e",
            "en--US",
            "zh-汉",
            "en-a",
            "x",
            "en-x",
            "sl-rozaj-ROZAJ",
            "en-a-foo-a-bar",
            "en-" + "-".join(["abcdefgh"] * 32),
        )
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_language_tag(value)


class QwenLanguageContractTests(unittest.TestCase):
    def test_auto_and_explicit_language_mapping(self) -> None:
        self.assertIsNone(qwen_language_for_request("auto"))
        self.assertEqual(qwen_language_for_request("en-US"), "English")
        self.assertEqual(qwen_language_for_request("zh-Hans"), "Chinese")
        self.assertEqual(qwen_language_for_request("yue-HK"), "Cantonese")
        self.assertEqual(qwen_language_for_request("fil-PH"), "Filipino")

    def test_unsupported_explicit_language_fails_closed(self) -> None:
        for language in ("he", "und", "mul", "x-private"):
            with self.subTest(language=language):
                with self.assertRaisesRegex(ValueError, "does not"):
                    qwen_language_for_request(language)

    def test_qwen_capability_inventory_is_stable_and_machine_readable(self) -> None:
        supported = qwen_supported_primary_language_tags()
        self.assertEqual(supported, tuple(sorted(set(supported))))
        self.assertIn("en", supported)
        self.assertIn("zh", supported)
        self.assertIn("yue", supported)
        self.assertNotIn("he", supported)

    def test_normalizes_merged_and_list_qwen_outputs(self) -> None:
        self.assertEqual(
            normalize_qwen_language_candidates("Chinese,English"),
            ("zh", "en"),
        )
        self.assertEqual(
            normalize_qwen_language_candidates(
                ["Portuguese", "pt-BR", "Portuguese", "unknown"]
            ),
            ("pt", "pt-BR"),
        )
        self.assertEqual(normalize_qwen_language_candidates(None), ())

    def test_duration_weighted_reconciliation(self) -> None:
        self.assertEqual(
            reconcile_detected_languages(
                [
                    {"language": "English", "speechDurationMs": 9000},
                    {"language": "Chinese", "speechDurationMs": 1000},
                ]
            ),
            "en",
        )
        self.assertEqual(
            reconcile_detected_languages(
                [
                    {"language": "English", "speechDurationMs": 5000},
                    {"language": "Chinese", "speechDurationMs": 5000},
                ]
            ),
            MULTIPLE_LANGUAGES,
        )
        self.assertEqual(
            reconcile_detected_languages(
                [{"language": "Chinese,English", "speechDurationMs": 5000}]
            ),
            MULTIPLE_LANGUAGES,
        )
        self.assertEqual(
            reconcile_detected_languages([], requested_language="auto"),
            UNDETERMINED_LANGUAGE,
        )

    def test_explicit_request_is_preserved_exactly(self) -> None:
        self.assertEqual(
            reconcile_detected_languages(
                [{"language": "Chinese", "speechDurationMs": 5000}],
                requested_language="EN_us",
            ),
            "en-US",
        )


class TranscriptSegmentLanguageTests(unittest.TestCase):
    @staticmethod
    def segment(**overrides: object) -> TranscriptSegment:
        values: dict[str, object] = {
            "segment_id": "segment-001",
            "start_ms": 0,
            "end_ms": 1000,
            "speaker_id": "speaker-1",
            "raw_text": "Original text.",
            "normalized_text": "Original text.",
            "display_text": "Original text.",
            "confidence": 0.92,
            "speaker_scores": (SpeakerScore("speaker-1", 0.95),),
            "speaker_margin": 0.95,
        }
        values.update(overrides)
        return TranscriptSegment(**values)

    def test_explicit_language_is_canonicalized_and_serialized(self) -> None:
        segment = self.segment(language="SR_latn_rs")
        self.assertEqual(segment.language, "sr-Latn-RS")
        self.assertEqual(segment.as_dict()["language"], "sr-Latn-RS")

    def test_language_is_derived_from_asr_evidence(self) -> None:
        segment = self.segment(
            evidence={"asr": {"language": "PT_br"}},
        )
        self.assertEqual(segment.language, "pt-BR")
        self.assertEqual(segment.as_dict()["language"], "pt-BR")

    def test_explicit_malformed_language_is_rejected(self) -> None:
        with self.assertRaises(WorkerError) as raised:
            self.segment(language="en-a")
        self.assertEqual(raised.exception.code, "ADAPTER_RESULT_INVALID")

    def test_request_only_auto_language_cannot_be_persisted_on_segment(self) -> None:
        with self.assertRaises(WorkerError) as raised:
            self.segment(language="auto")
        self.assertEqual(raised.exception.code, "ADAPTER_RESULT_INVALID")

    def test_request_only_auto_language_cannot_be_persisted_on_result(self) -> None:
        with self.assertRaises(WorkerError) as raised:
            TranscriptionResult(
                segments=(self.segment(language="en"),),
                duration_ms=1000,
                language="auto",
            )
        self.assertEqual(raised.exception.code, "ADAPTER_RESULT_INVALID")

    def test_transcript_assembly_rejects_request_only_auto_language(self) -> None:
        result = TranscriptionResult(
            segments=(self.segment(language="en"),),
            duration_ms=1000,
            language="en",
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            source = Path(temp_directory) / "source.wav"
            source.write_bytes(b"test-audio")
            with self.assertRaises(WorkerError) as raised:
                assemble_transcript_document(
                    job_id="language-test",
                    source_path=source,
                    policy=SpeakerCountPolicy(
                        mode=SpeakerCountMode.MANUAL,
                        manual_count=1,
                    ),
                    estimate=SpeakerCountEstimate.manual(1),
                    speaker_count=1,
                    result=result,
                    title=None,
                    language="auto",
                    adapter_id="test-adapter",
                    adapter_version="1",
                )
        self.assertEqual(raised.exception.code, "ADAPTER_RESULT_INVALID")


if __name__ == "__main__":
    unittest.main()
