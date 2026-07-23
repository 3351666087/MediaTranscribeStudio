package com.mediatranscribestudio.pdf;

import com.fasterxml.jackson.databind.JsonNode;
import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.inspect.PdfMetadataWriter;
import com.mediatranscribestudio.pdf.render.CanonicalXhtmlRenderer;
import com.mediatranscribestudio.pdf.render.OpenHtmlPdfRenderer;
import com.mediatranscribestudio.pdf.render.RenderProfile;
import com.mediatranscribestudio.pdf.support.JsonSupport;
import com.mediatranscribestudio.pdf.validation.ReportDocumentValidator;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class MultilingualRenderingTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void canonicalizesConcreteBcp47LanguageTags() {
        Map<String, String> cases = new LinkedHashMap<>();
        cases.put("zh-CN", "zh-CN");
        cases.put("zh-Hans", "zh-Hans");
        cases.put("EN-us", "en-US");
        cases.put("EN_us", "en-US");
        cases.put("UND", "und");
        cases.put("es-419", "es-419");
        cases.put("sr-Latn-RS", "sr-Latn-RS");
        cases.put("de-DE-u-co-phonebk", "de-DE-u-co-phonebk");
        cases.put("iw-IL", "he-IL");
        cases.put("i-klingon", "tlh");
        cases.put("en-GB-oed", "en-GB-x-oed");
        cases.put("sgn-BE-FR", "sfb");
        cases.put("x-private", "x-private");

        cases.forEach((input, expected) ->
                assertEquals(expected,
                        ReportDocumentValidator.canonicalizeLanguageTag(input),
                        input));
    }

    @Test
    void rejectsAutoAndMalformedLanguageTags() {
        String tooLong = "en-x-" + "abcdefgh-".repeat(32) + "abcdef";
        String[] invalid = {
                null,
                "",
                " ",
                "auto",
                "AUTO",
                "en__US",
                "en--US",
                "-en",
                "en-",
                "en-x",
                "en US",
                "i-madeup",
                "sl-rozaj-ROZAJ",
                "en-u-ca-gregory-u-nu-latn",
                "en-a-foo-a-bar",
                tooLong
        };

        for (String language : invalid) {
            assertThrows(IllegalArgumentException.class,
                    () -> ReportDocumentValidator.canonicalizeLanguageTag(language),
                    String.valueOf(language));
        }
    }

    @Test
    void validatorRejectsAutoAndDefaultsMissingSegmentLanguage() {
        ReportDocument documentAuto = TestFixtures.document(2);
        documentAuto.language = "auto";
        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(documentAuto));

        ReportDocument segmentAuto = TestFixtures.document(2);
        segmentAuto.segments.get(0).language = "AUTO";
        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(segmentAuto));

        ReportDocument defaulted = TestFixtures.document(2);
        defaulted.language = "fa-IR";
        defaulted.segments.forEach(segment -> segment.language = null);
        assertDoesNotThrow(() -> ReportDocumentValidator.validate(defaulted));
        defaulted.segments.forEach(segment -> assertEquals("und", segment.language));
    }

    @Test
    void separatesEnglishInterfaceFallbackFromRtlTranscriptLanguage() {
        ReportDocument document = TestFixtures.document(2);
        document.language = "FA-ir";
        document.title = null;
        document.segments.get(0).language = "FA-ir";
        document.segments.get(1).language = "EN-us";

        String xhtml = render(document);

        assertTrue(xhtml.contains(
                "<html xmlns=\"http://www.w3.org/1999/xhtml\""
                        + " lang=\"en-US\" xml:lang=\"en-US\" dir=\"ltr\">"));
        assertTrue(xhtml.contains(
                "<p lang=\"fa-IR\" xml:lang=\"fa-IR\" dir=\"rtl\">"));
        assertTrue(xhtml.contains(
                "<p lang=\"en-US\" xml:lang=\"en-US\" dir=\"ltr\">"));
        assertTrue(xhtml.contains("Original-Language Transcript"));
        assertTrue(xhtml.contains("Speaker legend"));
        assertFalse(xhtml.contains("中文原文逐字稿"));
        assertFalse(xhtml.contains("说话人图例"));
        assertFalse(xhtml.toLowerCase().contains("lang=\"auto\""));
    }

    @Test
    void determinesRtlFromLanguageAndExplicitScript() {
        for (String language : new String[]{"ar-SA", "fa-IR", "he-IL", "ur-PK", "az-Arab"}) {
            assertTrue(ReportDocumentValidator.isRightToLeftLanguage(language), language);
        }
        for (String language : new String[]{"ar-Latn", "zh-CN", "en-US", "x-private"}) {
            assertFalse(ReportDocumentValidator.isRightToLeftLanguage(language), language);
        }
    }

    @Test
    void preservesEstablishedChineseCopyAndLtrLayout() {
        ReportDocument document = TestFixtures.document(5);
        String xhtml = render(document);

        assertTrue(xhtml.contains("lang=\"zh-CN\" xml:lang=\"zh-CN\" dir=\"ltr\""));
        assertTrue(xhtml.contains("5 位已解析角色"));
        assertTrue(xhtml.contains("中文原文逐字稿"));
        assertTrue(xhtml.contains("本报告不翻译、不总结"));
    }

    @Test
    void explicitChineseReportLocaleDoesNotChangeArabicTranscriptEvidence() {
        ReportDocument document = TestFixtures.document(2);
        document.language = "ar-SA";
        document.reportLocale = "zh-CN";
        document.title = null;
        String original = "\u0647\u0630\u0627 \u0646\u0635 \u0627\u0644\u0645\u0635\u062f\u0631.";
        document.segments.forEach(segment -> {
            segment.language = "ar-SA";
            segment.rawText = original;
            segment.normalizedText = original;
            segment.displayText = original;
        });

        String xhtml = render(document);

        assertTrue(xhtml.contains("lang=\"zh-CN\" xml:lang=\"zh-CN\" dir=\"ltr\""));
        assertTrue(xhtml.contains("中文原文逐字稿"));
        assertTrue(xhtml.contains(
                "<p lang=\"ar-SA\" xml:lang=\"ar-SA\" dir=\"rtl\">" + original + "</p>"));
        assertTrue(xhtml.contains(original));
    }

    @Test
    void canonicalizesAndRejectsPersistedReportLocalesFailClosed() {
        ReportDocument canonical = TestFixtures.document(2);
        canonical.reportLocale = "EN_us";
        ReportDocumentValidator.validate(canonical);
        assertEquals("en-US", canonical.reportLocale);

        for (String invalid : new String[]{"auto", "AUTO", "en-a", "en__US"}) {
            ReportDocument document = TestFixtures.document(2);
            document.reportLocale = invalid;
            assertThrows(IllegalArgumentException.class,
                    () -> ReportDocumentValidator.validate(document),
                    invalid);
        }
    }

    @Test
    void reportSchemaSharesTheBcp47DefinitionForDocumentsAndSegments() throws Exception {
        try (InputStream stream = MultilingualRenderingTest.class.getResourceAsStream(
                "/schemas/report-document.schema.json")) {
            assertNotNull(stream);
            JsonNode schema = JsonSupport.mapper().readTree(stream);
            JsonNode languageTag = schema.path("$defs").path("languageTag");
            String languagePattern = languageTag.path("pattern").asText();
            Pattern pattern = Pattern.compile(languagePattern);

            assertEquals("#/$defs/languageTag",
                    schema.path("properties").path("language").path("$ref").asText());
            assertEquals("#/$defs/languageTag",
                    schema.path("properties").path("reportLocale").path("$ref").asText());
            assertEquals("#/$defs/languageTag",
                    schema.path("$defs").path("segment").path("properties")
                            .path("language").path("$ref").asText());
            assertEquals(255, languageTag.path("maxLength").asInt());
            assertFalse(languageTag.has("enum"));

            for (String valid : new String[]{
                    "zh-CN",
                    "EN-us",
                    "und",
                    "und-Arab",
                    "es-419",
                    "sr-Latn-RS",
                    "de-DE-u-co-phonebk",
                    "ar-SA",
                    "fa-IR",
                    "he-IL",
                    "ur-PK",
                    "az-Arab",
                    "ar-Latn",
                    "x-private"
            }) {
                assertTrue(pattern.matcher(valid).matches(), valid);
            }
            for (String invalid : new String[]{
                    "auto",
                    "AUTO",
                    "en_US",
                    "en--US",
                    "en-x",
                    "en US",
                    "i-madeup",
                    "i-klingon",
                    "en-GB-oed",
                    "sgn-BE-FR"
            }) {
                assertFalse(pattern.matcher(invalid).matches(), invalid);
            }
        }
    }

    @Test
    void cssScopesRtlMirroringAfterTheExistingChineseRules() throws Exception {
        String css;
        try (InputStream stream = MultilingualRenderingTest.class.getResourceAsStream(
                "/templates/report.css")) {
            assertNotNull(stream);
            css = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }

        int baseTextRule = css.indexOf(".text-cell p {");
        int rtlRootRule = css.indexOf("html[dir=\"rtl\"]");
        assertTrue(baseTextRule >= 0);
        assertTrue(rtlRootRule > baseTextRule);
        assertTrue(css.contains(".text-cell p[dir=\"rtl\"]"));
        assertTrue(css.contains(".text-cell p[dir=\"ltr\"]"));
        assertTrue(css.contains("direction: rtl;"));
        assertTrue(css.contains("direction: ltr;"));
        assertTrue(css.contains("unicode-bidi: embed;"));
    }

    @Test
    void rendersRtlXhtmlToAReadablePdf() throws Exception {
        ReportDocument report = TestFixtures.document(2);
        report.language = "ar-SA";
        report.title = "محضر الاجتماع";
        report.segments.forEach(segment -> {
            segment.language = "ar-SA";
            segment.rawText = "مرحبًا بكم في تقرير الاجتماع المحلي.";
            segment.normalizedText = segment.rawText;
            segment.displayText = segment.rawText;
        });

        Path output = temporaryDirectory.resolve("rtl/report.pdf");
        new OpenHtmlPdfRenderer().render(render(report), output);

        assertTrue(Files.isRegularFile(output));
        assertTrue(Files.size(output) > 1_000);
        try (PDDocument pdf = PDDocument.load(output.toFile())) {
            assertTrue(pdf.getNumberOfPages() > 0);
        }
    }

    @Test
    void writesTruthfulEnglishFallbackAndTranscriptLanguageMetadata() throws Exception {
        ReportDocument report = TestFixtures.document(2);
        report.language = "fa-IR";
        report.reportLocale = "ar-SA";
        report.title = null;
        report.segments.forEach(segment -> segment.language = "fa-IR");

        Path output = temporaryDirectory.resolve("metadata/report.pdf");
        new OpenHtmlPdfRenderer().render(render(report), output);
        new PdfMetadataWriter().apply(output, report, "request-global", "a".repeat(64));

        try (PDDocument pdf = PDDocument.load(output.toFile())) {
            assertEquals("Original-Language Transcript",
                    pdf.getDocumentInformation().getTitle());
            assertEquals(
                    "Original-language transcript with dynamic arbitrary-N speakers and "
                            + "millisecond timestamps",
                    pdf.getDocumentInformation().getSubject());
            assertEquals("fa-IR",
                    pdf.getDocumentInformation()
                            .getCustomMetadataValue("MTS-Transcript-Language"));
            assertEquals("en-US",
                    pdf.getDocumentInformation()
                            .getCustomMetadataValue("MTS-Report-Locale"));
        }
    }

    private static String render(ReportDocument document) {
        return new CanonicalXhtmlRenderer().render(
                document, RenderProfile.forRound(1, 14));
    }
}
