package com.mediatranscribestudio.pdf;

import com.fasterxml.jackson.databind.JsonNode;
import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.contract.QualityReport;
import com.mediatranscribestudio.pdf.qa.AestheticFacetId;
import com.mediatranscribestudio.pdf.qa.HardGateId;
import com.mediatranscribestudio.pdf.qa.QualityEvaluator;
import com.mediatranscribestudio.pdf.render.CanonicalXhtmlRenderer;
import com.mediatranscribestudio.pdf.render.RenderProfile;
import com.mediatranscribestudio.pdf.render.SpeakerPalette;
import com.mediatranscribestudio.pdf.support.JsonSupport;
import com.mediatranscribestudio.pdf.validation.ReportDocumentValidator;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class ContractAndPaletteTest {
    @Test
    void acceptsPositiveDynamicCountsAndRejectsIncompleteScoreVectors() {
        for (int count : new int[]{1, 2, 5, 8, 13, 64}) {
            assertDoesNotThrow(() -> ReportDocumentValidator.validate(TestFixtures.document(count)));
        }
        ReportDocument invalid = TestFixtures.document(8);
        invalid.segments.get(0).evidence.speaker.scores.remove(0);
        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(invalid));
    }

    @Test
    void validatorFailsClosedOnDynamicCardinalityAndEvidenceDrift() {
        assertInvalid(document -> document.speakerPolicy.resolvedCount = 0);
        assertInvalid(document -> document.speakerPolicy.speakerIds.remove(0));
        assertInvalid(document -> document.speakers.remove(0));
        assertInvalid(document -> document.speakers.get(0).id = "speaker-2");
        assertInvalid(document -> document.speakers.get(0).order = 2);
        assertInvalid(document -> document.speakers.get(0).colorToken = "speaker.2");
        assertInvalid(document -> document.segments.get(0).speakerId = "speaker-999");
        assertInvalid(document -> document.segments.removeIf(
                segment -> "speaker-8".equals(segment.speakerId)));
        assertInvalid(document -> document.speakerPolicy.detection.estimatedCount = 7);
        assertInvalid(document -> document.speakerPolicy.detection.confidence = Double.NaN);
        assertInvalid(document -> {
            document.speakerPolicy.mode = "manual";
            document.speakerPolicy.requestedCount = 7;
        });
        assertInvalid(document -> document.segments.get(0)
                .evidence.speaker.scores.get(1).speakerId = "speaker-1");
        assertInvalid(document -> document.segments.get(0)
                .evidence.speaker.scores.get(0).score = Double.POSITIVE_INFINITY);
        assertInvalid(document -> document.segments.get(0).confidence = Double.NaN);
        assertInvalid(document -> document.segments.get(0).evidence.speaker.locked = null);
    }

    @Test
    void paletteIsDeterministicUnboundedAndAccessible() {
        SpeakerPalette palette = new SpeakerPalette();
        Set<String> accents = new HashSet<>();
        for (int order = 1; order <= 128; order++) {
            SpeakerPalette.SpeakerStyle style = palette.style(order, "speaker." + order);
            assertTrue(SpeakerPalette.contrast(style.foreground(), style.accent()) >= 4.5);
            assertTrue(SpeakerPalette.contrast(style.accent(), style.background()) >= 4.5);
            accents.add(style.accent());
        }
        assertTrue(accents.size() > 100);
        assertThrows(IllegalArgumentException.class, () -> palette.style(4, "speaker.5"));
    }

    @Test
    void xhtmlIsOfflineParseableAndUsesDynamicLegend() {
        ReportDocument document = TestFixtures.document(64);
        document.language = "sr-Latn-RS";
        document.segments.get(0).rawText = "特殊字符 A&B <离线> \"引号\" '单引号' 必须安全保留。";
        document.segments.get(0).normalizedText = document.segments.get(0).rawText;
        document.segments.get(0).displayText = document.segments.get(0).rawText;
        document.title = "标题 \"安全\" </style><script>拒绝注入</script>";
        String xhtml = new CanonicalXhtmlRenderer().render(
                document, RenderProfile.forRound(1, 14));
        String lower = xhtml.toLowerCase(Locale.ROOT);
        assertTrue(xhtml.contains("64 resolved roles"));
        assertTrue(xhtml.contains("lang=\"en-US\""));
        assertTrue(xhtml.contains("xml:lang=\"en-US\""));
        assertTrue(xhtml.contains("<strong>sr-Latn-RS</strong>"));
        for (int index = 1; index <= 64; index++) {
            assertTrue(xhtml.contains("speaker-" + index));
        }
        assertTrue(xhtml.contains("A&amp;B"));
        assertTrue(xhtml.contains("&lt;离线&gt;"));
        assertFalse(lower.contains("</style><script>"));
        assertFalse(lower.matches("(?s).*\\b(?:src|href)\\s*=\\s*[\"']https?://.*"));
    }

    @Test
    void xhtmlRendererValidatesItsInputsBeforeRendering() {
        ReportDocument invalid = TestFixtures.document(8);
        invalid.speakerPolicy.speakerIds.remove(0);
        CanonicalXhtmlRenderer renderer = new CanonicalXhtmlRenderer();
        assertThrows(IllegalArgumentException.class,
                () -> renderer.render(invalid, RenderProfile.forRound(1, 14)));
        assertThrows(IllegalArgumentException.class,
                () -> renderer.render(TestFixtures.document(8), null));
    }

    @Test
    void aestheticWeightsSumExactlyToOne() {
        BigDecimal sum = Arrays.stream(AestheticFacetId.values())
                .map(item -> BigDecimal.valueOf(item.weight()))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        assertEquals(14, AestheticFacetId.values().length);
        assertEquals(0, BigDecimal.ONE.compareTo(sum));
    }

    @Test
    void hardGatesAndFacetsMustBeCanonicalCompleteUniqueAndOrdered() {
        QualityReport valid = canonicalQualityReport();
        assertDoesNotThrow(() -> QualityEvaluator.requireCanonicalChecks(valid));

        QualityReport missingGate = canonicalQualityReport();
        missingGate.hardGates.remove(0);
        assertThrows(IllegalStateException.class,
                () -> QualityEvaluator.requireCanonicalChecks(missingGate));

        QualityReport duplicateGate = canonicalQualityReport();
        duplicateGate.hardGates.get(1).id = duplicateGate.hardGates.get(0).id;
        assertThrows(IllegalStateException.class,
                () -> QualityEvaluator.requireCanonicalChecks(duplicateGate));

        QualityReport missingFacet = canonicalQualityReport();
        missingFacet.facets.remove(0);
        assertThrows(IllegalStateException.class,
                () -> QualityEvaluator.requireCanonicalChecks(missingFacet));

        QualityReport duplicateFacet = canonicalQualityReport();
        duplicateFacet.facets.get(1).id = duplicateFacet.facets.get(0).id;
        assertThrows(IllegalStateException.class,
                () -> QualityEvaluator.requireCanonicalChecks(duplicateFacet));

        QualityReport weightDrift = canonicalQualityReport();
        weightDrift.facets.get(0).weight = 0.09;
        assertThrows(IllegalStateException.class,
                () -> QualityEvaluator.requireCanonicalChecks(weightDrift));
    }

    @Test
    void qualityReportSchemaLocksCanonicalChecksWithoutGapsOrDuplicates() throws Exception {
        try (InputStream stream = ContractAndPaletteTest.class.getResourceAsStream(
                "/schemas/pdf-quality-report.schema.json")) {
            assertNotNull(stream);
            JsonNode schema = JsonSupport.mapper().readTree(stream);
            assertCanonicalSchemaArray(
                    schema.path("properties").path("hardGates"),
                    HardGateId.ids()
            );
            assertCanonicalSchemaArray(
                    schema.path("properties").path("facets"),
                    AestheticFacetId.ids()
            );
        }
    }

    private static void assertInvalid(Consumer<ReportDocument> mutation) {
        ReportDocument document = TestFixtures.document(8);
        mutation.accept(document);
        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    private static QualityReport canonicalQualityReport() {
        QualityReport report = new QualityReport();
        for (HardGateId id : HardGateId.values()) {
            QualityReport.HardGate gate = new QualityReport.HardGate();
            gate.id = id.id();
            report.hardGates.add(gate);
        }
        for (AestheticFacetId id : AestheticFacetId.values()) {
            QualityReport.Facet facet = new QualityReport.Facet();
            facet.id = id.id();
            facet.weight = id.weight();
            report.facets.add(facet);
        }
        return report;
    }

    private static void assertCanonicalSchemaArray(JsonNode arraySchema, List<String> expectedIds) {
        assertEquals(expectedIds.size(), arraySchema.path("minItems").asInt());
        assertEquals(expectedIds.size(), arraySchema.path("maxItems").asInt());
        assertTrue(arraySchema.path("uniqueItems").asBoolean());
        assertFalse(arraySchema.path("items").asBoolean(true));

        List<String> actualIds = new ArrayList<>();
        for (JsonNode item : arraySchema.path("prefixItems")) {
            actualIds.add(item.path("allOf").path(1)
                    .path("properties").path("id").path("const").asText());
        }
        assertEquals(expectedIds, actualIds);
        assertEquals(actualIds.size(), new HashSet<>(actualIds).size());
    }
}
