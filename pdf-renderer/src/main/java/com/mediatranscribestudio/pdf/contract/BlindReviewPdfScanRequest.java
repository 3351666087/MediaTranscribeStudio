package com.mediatranscribestudio.pdf.contract;

import java.util.ArrayList;
import java.util.List;

public final class BlindReviewPdfScanRequest {
    public String schemaVersion;
    public String pdfPath;
    public String expectedPdfSha256;
    public List<String> sensitiveValues = new ArrayList<>();
}
