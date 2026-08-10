package com.mediatranscribestudio.pdf.contract;

import java.util.ArrayList;
import java.util.List;

public final class BlindReviewPdfScanResult {
    public String schemaVersion = "1.0.0";
    public String artifactType = "blind-review-pdf-identity-scan";
    public String validator = "PDFBox";
    public String validatorVersion = "2.0.30";
    public String status;
    public String pdfSha256;
    public int sensitiveValueCount;
    public int pageCount;
    public int pageTextCount;
    public int documentInfoEntryCount;
    public int xmpPacketCount;
    public int attachmentCount;
    public int annotationCount;
    public int formFieldCount;
    public int scannedCosObjectCount;
    public long decodedStreamBytes;
    public boolean allRequiredSurfacesScanned;
    public List<Finding> findings = new ArrayList<>();
    public Error error;

    public static final class Finding {
        public String location;
        public String sensitiveValueSha256;

        public static Finding of(String location, String digest) {
            Finding finding = new Finding();
            finding.location = location;
            finding.sensitiveValueSha256 = digest;
            return finding;
        }
    }

    public static final class Error {
        public String code;
        public String message;
        public Boolean retryable;

        public static Error of(String code, String message) {
            Error error = new Error();
            error.code = code;
            error.message = message;
            error.retryable = false;
            return error;
        }
    }
}
