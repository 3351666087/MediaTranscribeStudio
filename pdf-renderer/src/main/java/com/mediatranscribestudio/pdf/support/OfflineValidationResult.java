package com.mediatranscribestudio.pdf.support;

import java.util.List;

public record OfflineValidationResult(
        boolean verified,
        String policyVersion,
        String contractVersion,
        String xhtmlSha256,
        List<String> checks,
        String failureReason
) {
    public OfflineValidationResult {
        checks = checks == null ? List.of() : List.copyOf(checks);
    }
}
