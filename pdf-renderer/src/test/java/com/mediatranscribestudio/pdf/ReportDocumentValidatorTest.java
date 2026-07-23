package com.mediatranscribestudio.pdf;

import com.fasterxml.jackson.databind.node.TextNode;
import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.validation.ReportDocumentValidator;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class ReportDocumentValidatorTest {
    @Test
    void unchangedTextWithoutRevisionsIsAccepted() {
        assertDoesNotThrow(() -> ReportDocumentValidator.validate(TestFixtures.document(2)));
    }

    @Test
    void changedTextWithoutRevisionIsRejected() {
        ReportDocument document = TestFixtures.document(2);
        document.segments.get(0).displayText = "未经审计的改写";
        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    @Test
    void automaticallyAppliedSemanticEditIsRejected() {
        ReportDocument document = TestFixtures.document(2);
        ReportDocument.SemanticEvidence semantic = new ReportDocument.SemanticEvidence();
        semantic.provider = "local-llm";
        semantic.decision = "normalize";
        semantic.autoApplied = true;
        document.segments.get(0).evidence.semantic = semantic;
        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    @Test
    void auditedManualTextRevisionIsAccepted() {
        ReportDocument document = TestFixtures.document(2);
        ReportDocument.Segment segment = document.segments.get(0);
        String edited = segment.rawText + "人工确认。";
        segment.normalizedText = edited;
        segment.displayText = edited;
        segment.revisions = List.of(revision(
                "revision-1", "manual", segment.rawText, edited, "human-reviewer"));
        assertDoesNotThrow(() -> ReportDocumentValidator.validate(document));
    }

    @Test
    void modelOnlyTextRevisionIsRejected() {
        ReportDocument document = TestFixtures.document(2);
        ReportDocument.Segment segment = document.segments.get(0);
        String edited = segment.rawText + "模型改写。";
        segment.normalizedText = edited;
        segment.displayText = edited;
        segment.revisions = List.of(revision(
                "revision-1", "llm", segment.rawText, edited, "local-llm"));
        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    @Test
    void multilingualDocumentAndSegmentLanguagesAreAccepted() {
        ReportDocument document = TestFixtures.document(2);
        document.language = "sr-Latn-RS";
        document.segments.get(0).language = "pt-BR";
        document.segments.get(1).language = "zh-Hans-CN";

        assertDoesNotThrow(() -> ReportDocumentValidator.validate(document));
    }

    @Test
    void missingRequiredDocumentLanguageIsRejected() {
        ReportDocument document = TestFixtures.document(2);
        document.language = null;

        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    @Test
    void missingOptionalSegmentLanguageDefaultsToUndetermined() {
        ReportDocument document = TestFixtures.document(2);
        document.segments.get(0).language = null;

        ReportDocumentValidator.validate(document);

        assertEquals("und", document.segments.get(0).language);
    }

    @Test
    void explicitUndeterminedLanguageIsAcceptedAndCanonicalized() {
        ReportDocument document = TestFixtures.document(2);
        document.language = "UND";
        document.segments.get(0).language = "und";

        ReportDocumentValidator.validate(document);

        assertEquals("und", document.language);
        assertEquals("und", document.segments.get(0).language);
    }

    @Test
    void concreteLanguageTagsAreCanonicalizedAndStored() {
        ReportDocument document = TestFixtures.document(2);
        document.language = "EN_us";
        document.segments.get(0).language = "SR_latn_rs";

        ReportDocumentValidator.validate(document);

        assertEquals("en-US", document.language);
        assertEquals("sr-Latn-RS", document.segments.get(0).language);
    }

    @Test
    void requestOnlyAutoLanguageIsRejected() {
        ReportDocument document = TestFixtures.document(2);
        document.language = "auto";

        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    @Test
    void persistedSegmentAutoLanguageIsRejected() {
        ReportDocument document = TestFixtures.document(2);
        document.segments.get(0).language = "AUTO";

        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    @Test
    void malformedDocumentLanguageIsRejected() {
        ReportDocument document = TestFixtures.document(2);
        document.language = "sl-rozaj-ROZAJ";

        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    @Test
    void malformedSegmentLanguageIsRejected() {
        ReportDocument document = TestFixtures.document(2);
        document.segments.get(0).language = "en-a";

        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    @Test
    void malformedUnderscoreLanguageIsRejected() {
        ReportDocument document = TestFixtures.document(2);
        document.language = "en__US";

        assertThrows(IllegalArgumentException.class,
                () -> ReportDocumentValidator.validate(document));
    }

    private static ReportDocument.Revision revision(
            String id,
            String source,
            String before,
            String after,
            String actor
    ) {
        ReportDocument.Revision revision = new ReportDocument.Revision();
        revision.revisionId = id;
        revision.type = "text";
        revision.source = source;
        revision.reasonCode = "SEMANTIC_CORRECTION";
        revision.before = TextNode.valueOf(before);
        revision.after = TextNode.valueOf(after);
        revision.actor = actor;
        revision.occurredAt = "2026-07-21T00:00:00Z";
        return revision;
    }
}
