package com.mediatranscribestudio.pdf;

import com.mediatranscribestudio.pdf.app.RenderApplication;
import com.mediatranscribestudio.pdf.contract.BlindReviewPdfScanRequest;
import com.mediatranscribestudio.pdf.contract.BlindReviewPdfScanResult;
import com.mediatranscribestudio.pdf.contract.RenderRequest;
import com.mediatranscribestudio.pdf.contract.RenderResult;
import com.mediatranscribestudio.pdf.inspect.BlindReviewPdfInspector;
import com.mediatranscribestudio.pdf.support.JsonSupport;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

public final class Main {
    private static final int MAX_STDIN_REQUEST_BYTES = 8 * 1024 * 1024;

    private Main() {
    }

    public static void main(String[] args) {
        PrintStream protocolOut = System.out;
        PrintStream stderr = System.err;
        System.setOut(new PrintStream(OutputStream.nullOutputStream(), true, StandardCharsets.UTF_8));
        int exit = run(args, protocolOut, stderr, System.in);
        protocolOut.flush();
        stderr.flush();
        if (exit != 0) {
            System.exit(exit);
        }
    }

    public static int run(String[] args, PrintStream stdout, PrintStream stderr) {
        return run(args, stdout, stderr, System.in);
    }

    public static int run(
            String[] args,
            PrintStream stdout,
            PrintStream stderr,
            InputStream stdin
    ) {
        if (args.length == 1 && "--blind-review-scan".equals(args[0])) {
            return runBlindReviewScan(stdin, stdout, stderr);
        }
        RenderRequest request = null;
        try {
            if (args.length != 2 || !"--request".equals(args[0])) {
                throw new IllegalArgumentException(
                        "usage: java -jar pdf-renderer.jar --request <request.json>");
            }
            Path requestPath = Path.of(args[1]).toAbsolutePath().normalize();
            request = JsonSupport.read(requestPath, RenderRequest.class);
            RenderResult result = new RenderApplication().render(request);
            stdout.println(JsonSupport.compact(result));
            if ("passed".equals(result.status)) {
                return 0;
            }
            stderr.println("PDF renderer blocked by quality gates");
            return 3;
        } catch (Exception exception) {
            RenderResult failure = new RenderResult();
            failure.requestId = request == null || request.requestId == null ? "unknown" : request.requestId;
            failure.jobId = request == null || request.jobId == null ? "unknown" : request.jobId;
            failure.status = "failed";
            failure.rendererVersion = RenderApplication.RENDERER_VERSION;
            failure.roundsCompleted = 0;
            failure.quality.status = "failed";
            failure.quality.hardGatesPassed = false;
            failure.quality.score = 0.0;
            failure.artifacts.reportDocumentPath = "unavailable";
            failure.artifacts.htmlPath = "unavailable";
            failure.artifacts.pdfPath = "unavailable";
            failure.artifacts.manifestPath = "unavailable";
            failure.artifacts.qualityReportPath = "unavailable";
            failure.artifacts.repairQueuePath = "unavailable";
            failure.artifacts.screenshotsDirectory = "unavailable";
            failure.artifacts.contactSheetPath = "unavailable";
            failure.error = RenderResult.Error.of(
                    "PDF_RENDER_FAILED",
                    safeMessage(exception),
                    false
            );
            try {
                stdout.println(JsonSupport.compact(failure));
            } catch (Exception serializationFailure) {
                stdout.println("{\"schemaVersion\":\"1.0.0\",\"requestId\":\"unknown\","
                        + "\"jobId\":\"unknown\",\"status\":\"failed\","
                        + "\"rendererVersion\":\"3.0.0\",\"roundsCompleted\":0,"
                        + "\"artifacts\":{},\"quality\":{\"status\":\"failed\","
                        + "\"hardGatesPassed\":false,\"score\":0},"
                        + "\"error\":{\"code\":\"PDF_RENDER_FAILED\","
                        + "\"message\":\"unserializable failure\",\"retryable\":false}}");
            }
            stderr.println("PDF renderer failed: " + safeMessage(exception));
            return exception instanceof IllegalArgumentException ? 2 : 1;
        }
    }

    private static int runBlindReviewScan(
            InputStream stdin,
            PrintStream stdout,
            PrintStream stderr
    ) {
        BlindReviewPdfScanResult result;
        try {
            byte[] requestBytes = readBoundedStdin(stdin);
            BlindReviewPdfScanRequest request = JsonSupport.mapper().readValue(
                    new ByteArrayInputStream(requestBytes),
                    BlindReviewPdfScanRequest.class
            );
            result = new BlindReviewPdfInspector().inspect(request);
            stdout.println(JsonSupport.compact(result));
            if ("passed".equals(result.status)) {
                return 0;
            }
            stderr.println("PDFBox blind review scan blocked identity leakage");
            return 4;
        } catch (Exception exception) {
            result = new BlindReviewPdfScanResult();
            result.status = "failed";
            result.allRequiredSurfacesScanned = false;
            result.error = BlindReviewPdfScanResult.Error.of(
                    "PDF_BLIND_REVIEW_SCAN_FAILED",
                    safeMessage(exception)
            );
            try {
                stdout.println(JsonSupport.compact(result));
            } catch (Exception serializationFailure) {
                stdout.println("{\"schemaVersion\":\"1.0.0\","
                        + "\"artifactType\":\"blind-review-pdf-identity-scan\","
                        + "\"validator\":\"PDFBox\","
                        + "\"validatorVersion\":\"2.0.30\","
                        + "\"status\":\"failed\","
                        + "\"allRequiredSurfacesScanned\":false,"
                        + "\"findings\":[],"
                        + "\"error\":{\"code\":\"PDF_BLIND_REVIEW_SCAN_FAILED\","
                        + "\"message\":\"unserializable failure\","
                        + "\"retryable\":false}}");
            }
            stderr.println(
                    "PDFBox blind review scan failed: " + safeMessage(exception));
            return exception instanceof IllegalArgumentException ? 2 : 1;
        }
    }

    private static byte[] readBoundedStdin(InputStream stdin) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[16 * 1024];
        int read;
        while ((read = stdin.read(buffer)) >= 0) {
            if (output.size() + read > MAX_STDIN_REQUEST_BYTES) {
                throw new IllegalArgumentException(
                        "blind review scan request exceeds the stdin limit");
            }
            output.write(buffer, 0, read);
        }
        if (output.size() == 0) {
            throw new IllegalArgumentException(
                    "blind review scan request is empty");
        }
        return output.toByteArray();
    }

    private static String safeMessage(Exception exception) {
        String message = exception.getMessage();
        if (message == null || message.isBlank()) {
            message = exception.getClass().getSimpleName();
        }
        return message.length() <= 2000 ? message : message.substring(0, 2000);
    }
}
