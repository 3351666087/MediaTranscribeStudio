package com.mediatranscribestudio.pdf;

import com.fasterxml.jackson.databind.JsonNode;
import com.mediatranscribestudio.pdf.contract.BlindReviewPdfScanRequest;
import com.mediatranscribestudio.pdf.contract.BlindReviewPdfScanResult;
import com.mediatranscribestudio.pdf.inspect.BlindReviewPdfInspector;
import com.mediatranscribestudio.pdf.support.Hashing;
import com.mediatranscribestudio.pdf.support.JsonSupport;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;
import org.apache.pdfbox.pdmodel.PDDocumentNameDictionary;
import org.apache.pdfbox.pdmodel.PDEmbeddedFilesNameTreeNode;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDMetadata;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.common.filespecification.PDComplexFileSpecification;
import org.apache.pdfbox.pdmodel.common.filespecification.PDEmbeddedFile;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.interactive.annotation.PDAnnotationText;
import org.apache.pdfbox.pdmodel.interactive.form.PDAcroForm;
import org.apache.pdfbox.pdmodel.interactive.form.PDTextField;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class BlindReviewPdfInspectorTest {
    private static final String INFO_SECRET = "qwen-secret-info-model";
    private static final String XMP_SECRET = "challenger-xmp-model-identity";
    private static final String ATTACHMENT_SECRET =
            "D:/secret/source/case-one.wav";
    private static final String ANNOTATION_SECRET =
            "case-one-annotation-identity";
    private static final String FORM_SECRET = "production-form-role";
    private static final String PAGE_SECRET =
            "production-page-model-identity";
    private static final List<String> SENSITIVE_VALUES = List.of(
            INFO_SECRET,
            XMP_SECRET,
            ATTACHMENT_SECRET,
            ANNOTATION_SECRET,
            FORM_SECRET,
            PAGE_SECRET
    );

    @TempDir
    Path temporaryDirectory;

    @Test
    void scansCompressedAndStructuredPdfSurfacesFailClosed() throws Exception {
        Path pdf = leakingPdf();
        String raw = Files.readString(pdf, StandardCharsets.ISO_8859_1);
        assertFalse(raw.contains(XMP_SECRET));
        assertFalse(raw.contains(ATTACHMENT_SECRET));
        assertFalse(raw.contains(PAGE_SECRET));

        BlindReviewPdfScanResult result = new BlindReviewPdfInspector().inspect(
                request(pdf, SENSITIVE_VALUES));

        assertEquals("blocked", result.status);
        assertEquals(Hashing.sha256(pdf), result.pdfSha256);
        assertTrue(result.allRequiredSurfacesScanned);
        assertEquals(1, result.pageCount);
        assertEquals(1, result.pageTextCount);
        assertTrue(result.documentInfoEntryCount >= 1);
        assertEquals(1, result.xmpPacketCount);
        assertEquals(1, result.attachmentCount);
        assertEquals(1, result.annotationCount);
        assertEquals(1, result.formFieldCount);
        assertTrue(result.scannedCosObjectCount > 0);
        assertTrue(result.decodedStreamBytes > 0);
        Set<String> locations = result.findings.stream()
                .map(finding -> finding.location)
                .collect(Collectors.toSet());
        assertTrue(locations.containsAll(Set.of(
                "document-info",
                "xmp",
                "attachment",
                "annotation",
                "form",
                "page-text"
        )));
        assertTrue(result.findings.stream().allMatch(
                finding -> finding.sensitiveValueSha256.matches("[a-f0-9]{64}")));
    }

    @Test
    void cleanStructuredScanPassesAndStdinProtocolReturnsOneJson()
            throws Exception {
        Path pdf = leakingPdf();
        BlindReviewPdfScanRequest request = request(
                pdf,
                List.of("identity-that-is-definitely-absent")
        );

        BlindReviewPdfScanResult direct =
                new BlindReviewPdfInspector().inspect(request);
        assertEquals("passed", direct.status);
        assertTrue(direct.findings.isEmpty());

        Protocol protocol = invoke(request);
        assertEquals(0, protocol.exitCode);
        assertEquals(1, protocol.stdout.lines().count());
        JsonNode payload = JsonSupport.mapper().readTree(protocol.stdout.trim());
        assertEquals("passed", payload.path("status").asText());
        assertEquals("PDFBox", payload.path("validator").asText());
        assertTrue(payload.path("allRequiredSurfacesScanned").asBoolean());
    }

    @Test
    void stdinProtocolBlocksLeakAndRejectsUnknownRequestFields()
            throws Exception {
        Path pdf = leakingPdf();
        Protocol blocked = invoke(request(pdf, SENSITIVE_VALUES));
        assertEquals(4, blocked.exitCode);
        assertEquals(
                "blocked",
                JsonSupport.mapper().readTree(blocked.stdout.trim())
                        .path("status").asText()
        );
        assertTrue(blocked.stderr.contains("blocked identity leakage"));

        String valid = JsonSupport.compact(
                request(pdf, List.of("absent-sensitive-value")));
        String unknown = valid.substring(0, valid.length() - 1)
                + ",\"unknown\":true}";
        Protocol invalid = invoke(unknown);
        assertNotEquals(0, invalid.exitCode);
        assertEquals(
                "failed",
                JsonSupport.mapper().readTree(invalid.stdout.trim())
                        .path("status").asText()
        );
        assertFalse(
                JsonSupport.mapper().readTree(invalid.stdout.trim())
                        .path("allRequiredSurfacesScanned").asBoolean()
        );
    }

    private Path leakingPdf() throws Exception {
        Path pdf = temporaryDirectory.resolve("structured-leaks.pdf");
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage(PDRectangle.A4);
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(
                    document,
                    page,
                    PDPageContentStream.AppendMode.OVERWRITE,
                    true,
                    true
            )) {
                content.beginText();
                content.setFont(PDType1Font.HELVETICA, 12);
                content.newLineAtOffset(72, 720);
                content.showText(PAGE_SECRET);
                content.endText();
            }

            PDDocumentInformation information =
                    document.getDocumentInformation();
            information.setCustomMetadataValue("HiddenModel", INFO_SECRET);

            PDMetadata metadata = new PDMetadata(document);
            String xmp = "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">"
                    + "<identity>" + XMP_SECRET + "</identity>"
                    + "</x:xmpmeta>";
            try (OutputStream output = metadata.createOutputStream(
                    COSName.FLATE_DECODE)) {
                output.write(xmp.getBytes(StandardCharsets.UTF_8));
            }
            document.getDocumentCatalog().setMetadata(metadata);

            PDComplexFileSpecification specification =
                    new PDComplexFileSpecification();
            specification.setFile("review-note.txt");
            specification.setFileUnicode("review-note.txt");
            specification.setFileDescription("blind review attachment");
            byte[] attachmentBytes = ATTACHMENT_SECRET.getBytes(
                    StandardCharsets.UTF_8);
            PDEmbeddedFile embedded = new PDEmbeddedFile(
                    document,
                    new ByteArrayInputStream(attachmentBytes),
                    COSName.FLATE_DECODE
            );
            embedded.setSize(attachmentBytes.length);
            specification.setEmbeddedFile(embedded);
            specification.setEmbeddedFileUnicode(embedded);
            PDEmbeddedFilesNameTreeNode embeddedFiles =
                    new PDEmbeddedFilesNameTreeNode();
            embeddedFiles.setNames(Map.of("review-note.txt", specification));
            PDDocumentNameDictionary names = new PDDocumentNameDictionary(
                    document.getDocumentCatalog());
            names.setEmbeddedFiles(embeddedFiles);
            document.getDocumentCatalog().setNames(names);

            PDAnnotationText annotation = new PDAnnotationText();
            annotation.setContents(ANNOTATION_SECRET);
            annotation.setRectangle(new PDRectangle(72, 680, 24, 24));
            page.getAnnotations().add(annotation);

            PDAcroForm form = new PDAcroForm(document);
            document.getDocumentCatalog().setAcroForm(form);
            PDTextField field = new PDTextField(form);
            field.setPartialName("review-field");
            field.getCOSObject().setString(COSName.V, FORM_SECRET);
            form.setFields(List.of(field));

            document.save(pdf.toFile());
        }
        return pdf;
    }

    private static BlindReviewPdfScanRequest request(
            Path pdf,
            List<String> sensitiveValues
    ) throws Exception {
        BlindReviewPdfScanRequest request = new BlindReviewPdfScanRequest();
        request.schemaVersion = "1.0.0";
        request.pdfPath = pdf.toAbsolutePath().normalize().toString();
        request.expectedPdfSha256 = Hashing.sha256(pdf);
        request.sensitiveValues = sensitiveValues;
        return request;
    }

    private static Protocol invoke(BlindReviewPdfScanRequest request)
            throws Exception {
        return invoke(JsonSupport.compact(request));
    }

    private static Protocol invoke(String requestJson) {
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();
        int exit = Main.run(
                new String[]{"--blind-review-scan"},
                new PrintStream(stdout, true, StandardCharsets.UTF_8),
                new PrintStream(stderr, true, StandardCharsets.UTF_8),
                new ByteArrayInputStream(
                        requestJson.getBytes(StandardCharsets.UTF_8))
        );
        return new Protocol(
                exit,
                stdout.toString(StandardCharsets.UTF_8),
                stderr.toString(StandardCharsets.UTF_8)
        );
    }

    private record Protocol(int exitCode, String stdout, String stderr) {
    }
}
