package com.mediatranscribestudio.pdf;

import com.mediatranscribestudio.pdf.app.RenderApplication;
import com.mediatranscribestudio.pdf.contract.QualityReport;
import com.mediatranscribestudio.pdf.contract.RenderResult;
import com.mediatranscribestudio.pdf.qa.AestheticFacetId;
import com.mediatranscribestudio.pdf.qa.HardGateId;
import com.mediatranscribestudio.pdf.support.Hashing;
import com.mediatranscribestudio.pdf.support.JsonSupport;
import com.mediatranscribestudio.pdf.validation.ReportDocumentValidator;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.io.TempDir;

import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class DynamicSpeakerRenderingTest {
    @TempDir
    Path temporaryDirectory;

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 5, 8, 13, 64, 129})
    void rendersArbitrarySpeakerCountsWithAllQualityEvidence(int speakerCount) throws Exception {
        Path output = temporaryDirectory.resolve("n-" + speakerCount);
        RenderResult result = new RenderApplication().render(
                TestFixtures.request(output, speakerCount));

        assertEquals("passed", result.status,
                Files.readString(output.resolve(result.artifacts.qualityReportPath))
                        + "\nEXTRACTED:\n"
                        + Files.readString(output.resolve("artifacts/pdf-extracted-text.txt")));
        assertEquals("passed", result.quality.status);
        assertTrue(result.quality.hardGatesPassed);
        assertTrue(result.quality.score >= 85);
        assertTrue(Files.isRegularFile(output.resolve(result.artifacts.pdfPath)));
        assertTrue(Files.isRegularFile(output.resolve(result.artifacts.contactSheetPath)));
        assertTrue(Files.isDirectory(output.resolve(result.artifacts.screenshotsDirectory)));

        QualityReport report = JsonSupport.read(
                output.resolve(result.artifacts.qualityReportPath), QualityReport.class);
        assertEquals(HardGateId.ids(),
                report.hardGates.stream().map(gate -> gate.id).toList());
        assertEquals(HardGateId.values().length, report.hardGates.size());
        assertEquals(HardGateId.values().length,
                report.hardGates.stream().map(gate -> gate.id).distinct().count());
        assertTrue(report.hardGates.stream().allMatch(gate -> "passed".equals(gate.status)));
        assertEquals(AestheticFacetId.ids(),
                report.facets.stream().map(facet -> facet.id).toList());
        assertEquals(AestheticFacetId.values().length,
                report.facets.stream().map(facet -> facet.id).distinct().count());
        assertTrue(report.facets.stream().allMatch(facet -> "passed".equals(facet.status)));
        BigDecimal weight = report.facets.stream()
                .map(facet -> BigDecimal.valueOf(facet.weight))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        assertEquals(0, BigDecimal.ONE.compareTo(weight));

        Path pdf = output.resolve(result.artifacts.pdfPath);
        List<String> expectedSpeakerIds =
                ReportDocumentValidator.canonicalSpeakerIds(speakerCount);
        String expectedSpeakerIdValue = String.join(",", expectedSpeakerIds);
        try (PDDocument document = PDDocument.load(pdf.toFile())) {
            PDDocumentInformation metadata = document.getDocumentInformation();
            assertEquals(Integer.toString(speakerCount),
                    metadata.getCustomMetadataValue("MTS-Speaker-Count"));
            assertEquals(expectedSpeakerIdValue,
                    metadata.getCustomMetadataValue("MTS-Speaker-Ids"));
            assertEquals(Hashing.sha256(expectedSpeakerIdValue),
                    metadata.getCustomMetadataValue("MTS-Speaker-Set-SHA256"));
            assertEquals("OpenHTMLtoPDF 1.0.10 + Apache PDFBox 2.0.30",
                    metadata.getProducer());
        }
    }
}
