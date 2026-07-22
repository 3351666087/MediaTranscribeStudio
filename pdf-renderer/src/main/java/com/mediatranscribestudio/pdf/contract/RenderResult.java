package com.mediatranscribestudio.pdf.contract;

public final class RenderResult {
    public String schemaVersion = "1.0.0";
    public String requestId;
    public String jobId;
    public String status;
    public String rendererVersion;
    public Integer roundsCompleted;
    public Artifacts artifacts = new Artifacts();
    public Quality quality = new Quality();
    public Error error;

    public static final class Artifacts {
        public String reportDocumentPath;
        public String htmlPath;
        public String pdfPath;
        public String manifestPath;
        public String qualityReportPath;
        public String repairQueuePath;
        public String screenshotsDirectory;
        public String contactSheetPath;
    }

    public static final class Quality {
        public String status;
        public Boolean hardGatesPassed;
        public Double score;
    }

    public static final class Error {
        public String code;
        public String message;
        public Boolean retryable;

        public static Error of(String code, String message, boolean retryable) {
            Error error = new Error();
            error.code = code;
            error.message = message;
            error.retryable = retryable;
            return error;
        }
    }
}
