package com.mediatranscribestudio.pdf.app;

import com.mediatranscribestudio.pdf.contract.ArtifactManifest;
import com.mediatranscribestudio.pdf.contract.PdfInspection;
import com.mediatranscribestudio.pdf.contract.QualityReport;
import com.mediatranscribestudio.pdf.contract.RenderRequest;
import com.mediatranscribestudio.pdf.contract.RenderResult;
import com.mediatranscribestudio.pdf.contract.RepairQueue;
import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.inspect.PdfBoxInspector;
import com.mediatranscribestudio.pdf.inspect.PdfMetadataWriter;
import com.mediatranscribestudio.pdf.qa.QualityEvaluator;
import com.mediatranscribestudio.pdf.render.CanonicalXhtmlRenderer;
import com.mediatranscribestudio.pdf.render.ContactSheetRenderer;
import com.mediatranscribestudio.pdf.render.OpenHtmlPdfRenderer;
import com.mediatranscribestudio.pdf.render.PageImageRenderer;
import com.mediatranscribestudio.pdf.render.RenderProfile;
import com.mediatranscribestudio.pdf.support.ArtifactManifestBuilder;
import com.mediatranscribestudio.pdf.support.AtomicFiles;
import com.mediatranscribestudio.pdf.support.DesignPackOfflineValidator;
import com.mediatranscribestudio.pdf.support.Hashing;
import com.mediatranscribestudio.pdf.support.JsonSupport;
import com.mediatranscribestudio.pdf.support.OfflineValidationResult;
import com.mediatranscribestudio.pdf.support.TranscriptIntegrity;
import com.mediatranscribestudio.pdf.validation.RenderRequestValidator;
import com.mediatranscribestudio.pdf.validation.ReportDocumentValidator;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;

public final class RenderApplication {
    public static final String RENDERER_VERSION = "3.0.0";

    private final CanonicalXhtmlRenderer xhtmlRenderer = new CanonicalXhtmlRenderer();
    private final OpenHtmlPdfRenderer pdfRenderer = new OpenHtmlPdfRenderer();
    private final PdfMetadataWriter metadataWriter = new PdfMetadataWriter();
    private final PdfBoxInspector inspector = new PdfBoxInspector();
    private final PageImageRenderer pageRenderer = new PageImageRenderer();
    private final ContactSheetRenderer contactSheetRenderer = new ContactSheetRenderer();
    private final QualityEvaluator qualityEvaluator = new QualityEvaluator();
    private final ArtifactManifestBuilder manifestBuilder = new ArtifactManifestBuilder();
    private final DesignPackOfflineValidator designValidator = new DesignPackOfflineValidator();

    public RenderResult render(RenderRequest request) throws IOException {
        RenderRequestValidator.Paths paths = RenderRequestValidator.validate(request);
        byte[] reportSnapshot = Files.readAllBytes(paths.reportDocument());
        String reportSnapshotSha256 = Hashing.sha256(reportSnapshot);
        ReportDocument document = JsonSupport.mapper().readValue(reportSnapshot, ReportDocument.class);
        ReportDocumentValidator.validate(document);
        Path output = paths.outputDirectory();
        Files.createDirectories(output);
        resetGeneratedDirectories(output);

        Path xhtmlPath = output.resolve("render/report.xhtml");
        Path pdfPath = output.resolve("render/report.pdf");
        Path workingXhtmlPath = output.resolve("qa/current/report.xhtml");
        Path workingPdfPath = output.resolve("qa/current/report.pdf");
        Path screenshots = output.resolve("artifacts/screens");
        Path contactSheet = output.resolve("artifacts/contact-sheet.png");
        Path inspectionPath = output.resolve("artifacts/pdf-inspection.json");
        Path textPath = output.resolve("artifacts/pdf-extracted-text.txt");
        Path qualityPath = output.resolve("artifacts/quality-report.json");
        Path repairPath = output.resolve("artifacts/repair-queue.json");
        Path manifestPath = output.resolve("artifacts/manifest.json");
        Files.createDirectories(workingXhtmlPath.getParent());
        Files.createDirectories(screenshots);

        String baselineHash = TranscriptIntegrity.sha256(document);
        QualityReport quality = null;
        PageImageRenderer.Result pageEvidence = null;
        int rounds = 0;
        for (int round = 1; round <= request.qualityPolicy.maxRounds; round++) {
            rounds = round;
            requireSnapshotUnchanged(paths.reportDocument(), reportSnapshotSha256);
            RenderProfile profile = RenderProfile.forRound(round, request.renderer.page.marginMm);
            String xhtml = xhtmlRenderer.render(document, profile);
            OfflineValidationResult offlineValidation = designValidator.verify(xhtml);
            AtomicFiles.write(workingXhtmlPath, xhtml.getBytes(StandardCharsets.UTF_8));
            pdfRenderer.render(xhtml, workingPdfPath);
            metadataWriter.apply(workingPdfPath, document, request.requestId, baselineHash);

            PdfBoxInspector.InspectionResult inspectionResult =
                    inspector.inspect(workingPdfPath, document);
            PdfInspection inspection = inspectionResult.inspection();
            JsonSupport.write(inspectionPath, inspection);
            AtomicFiles.write(textPath, inspectionResult.extractedText()
                    .getBytes(StandardCharsets.UTF_8));
            pageEvidence = pageRenderer.render(workingPdfPath, screenshots,
                    request.qualityPolicy.captureDpi);
            contactSheetRenderer.render(pageEvidence.pages(), contactSheet);

            quality = qualityEvaluator.evaluate(
                    output,
                    round,
                    request.qualityPolicy.maxRounds,
                    request.qualityPolicy.minimumScore,
                    profile,
                    document,
                    xhtml,
                    workingXhtmlPath,
                    workingPdfPath,
                    inspectionPath,
                    textPath,
                    contactSheet,
                    inspection,
                    pageEvidence,
                    baselineHash,
                    baselineHash,
                    offlineValidation
            );
            RepairQueue repairQueue = qualityEvaluator.standaloneQueue(quality);
            JsonSupport.write(qualityPath, quality);
            JsonSupport.write(repairPath, repairQueue);
            archiveRound(output, round, workingXhtmlPath, workingPdfPath, screenshots, contactSheet,
                    inspectionPath, textPath, qualityPath, repairPath);
            if (!"repair-required".equals(quality.status)) {
                break;
            }
        }
        if (quality == null || pageEvidence == null) {
            throw new IOException("no render round executed");
        }

        requireSnapshotUnchanged(paths.reportDocument(), reportSnapshotSha256);
        boolean passed = "passed".equals(quality.status);
        if (passed) {
            AtomicFiles.copy(workingXhtmlPath, xhtmlPath);
            AtomicFiles.copy(workingPdfPath, pdfPath);
        } else {
            deleteRecursively(output.resolve("render"));
        }

        ArtifactManifest manifest = manifestBuilder.build(
                output,
                request.jobId,
                document.documentId,
                RENDERER_VERSION,
                document.generatedAt,
                passed,
                paths.reportDocument(),
                passed ? xhtmlPath : null,
                passed ? pdfPath : null,
                pageEvidence.pages(),
                contactSheet,
                qualityPath,
                repairPath
        );
        JsonSupport.write(manifestPath, manifest);

        RenderResult result = new RenderResult();
        result.requestId = request.requestId;
        result.jobId = request.jobId;
        result.status = passed ? "passed" : "blocked";
        result.rendererVersion = RENDERER_VERSION;
        result.roundsCompleted = rounds;
        result.artifacts.reportDocumentPath = relative(output, paths.reportDocument());
        result.artifacts.htmlPath = passed ? "render/report.xhtml" : "unavailable";
        result.artifacts.pdfPath = passed ? "render/report.pdf" : "unavailable";
        result.artifacts.manifestPath = "artifacts/manifest.json";
        result.artifacts.qualityReportPath = "artifacts/quality-report.json";
        result.artifacts.repairQueuePath = "artifacts/repair-queue.json";
        result.artifacts.screenshotsDirectory = "artifacts/screens";
        result.artifacts.contactSheetPath = "artifacts/contact-sheet.png";
        result.quality.status = result.status;
        result.quality.hardGatesPassed = quality.hardGatesPassed;
        result.quality.score = quality.score;
        if (!"passed".equals(result.status)) {
            result.error = RenderResult.Error.of(
                    "PDF_QUALITY_BLOCKED",
                    "PDF quality gates did not reach the required terminal pass state",
                    false
            );
        }
        JsonSupport.write(output.resolve("artifacts/render-result.json"), result);
        return result;
    }

    private static void requireSnapshotUnchanged(Path report, String expectedSha256)
            throws IOException {
        String currentSha256 = Hashing.sha256(Files.readAllBytes(report));
        if (!expectedSha256.equals(currentSha256)) {
            throw new IOException("report document changed after validation; render aborted");
        }
    }

    private static void archiveRound(
            Path output,
            int round,
            Path xhtml,
            Path pdf,
            Path screenshots,
            Path contactSheet,
            Path inspection,
            Path text,
            Path quality,
            Path repair
    ) throws IOException {
        Path root = output.resolve(String.format("qa/round-%02d", round));
        deleteRecursively(root);
        AtomicFiles.copy(xhtml, root.resolve("report.xhtml"));
        AtomicFiles.copy(pdf, root.resolve("report.pdf"));
        AtomicFiles.copy(contactSheet, root.resolve("contact-sheet.png"));
        AtomicFiles.copy(inspection, root.resolve("pdf-inspection.json"));
        AtomicFiles.copy(text, root.resolve("pdf-extracted-text.txt"));
        AtomicFiles.copy(quality, root.resolve("quality-report.json"));
        AtomicFiles.copy(repair, root.resolve("repair-queue.json"));
        try (var pages = Files.list(screenshots)) {
            for (Path page : pages.sorted().toList()) {
                AtomicFiles.copy(page, root.resolve("screens").resolve(page.getFileName()));
            }
        }
    }

    private static void resetGeneratedDirectories(Path output) throws IOException {
        for (String child : new String[]{"render", "artifacts", "qa"}) {
            deleteRecursively(output.resolve(child));
        }
    }

    private static void deleteRecursively(Path path) throws IOException {
        if (!Files.exists(path)) {
            return;
        }
        try (var paths = Files.walk(path)) {
            for (Path item : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(item);
            }
        }
    }

    private static String relative(Path root, Path path) {
        return root.toAbsolutePath().normalize().relativize(path.toAbsolutePath().normalize())
                .toString().replace('\\', '/');
    }
}
