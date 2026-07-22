package com.mediatranscribestudio.pdf;

import com.mediatranscribestudio.pdf.app.RenderApplication;
import com.mediatranscribestudio.pdf.contract.RenderRequest;
import com.mediatranscribestudio.pdf.contract.RenderResult;
import com.mediatranscribestudio.pdf.support.JsonSupport;

import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

public final class Main {
    private Main() {
    }

    public static void main(String[] args) {
        PrintStream protocolOut = System.out;
        PrintStream stderr = System.err;
        System.setOut(new PrintStream(OutputStream.nullOutputStream(), true, StandardCharsets.UTF_8));
        int exit = run(args, protocolOut, stderr);
        protocolOut.flush();
        stderr.flush();
        if (exit != 0) {
            System.exit(exit);
        }
    }

    public static int run(String[] args, PrintStream stdout, PrintStream stderr) {
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

    private static String safeMessage(Exception exception) {
        String message = exception.getMessage();
        if (message == null || message.isBlank()) {
            message = exception.getClass().getSimpleName();
        }
        return message.length() <= 2000 ? message : message.substring(0, 2000);
    }
}
