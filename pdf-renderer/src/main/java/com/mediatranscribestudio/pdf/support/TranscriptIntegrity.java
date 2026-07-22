package com.mediatranscribestudio.pdf.support;

import com.mediatranscribestudio.pdf.contract.ReportDocument;

public final class TranscriptIntegrity {
    private TranscriptIntegrity() {
    }

    public static String sha256(ReportDocument document) {
        StringBuilder canonical = new StringBuilder(document.documentId).append('\n');
        for (ReportDocument.Segment segment : document.segments) {
            canonical.append(segment.id).append('\u001f')
                    .append(segment.startMs).append('\u001f')
                    .append(segment.endMs).append('\u001f')
                    .append(segment.speakerId).append('\u001f')
                    .append(segment.rawText).append('\u001f')
                    .append(segment.normalizedText).append('\u001f')
                    .append(segment.displayText).append('\n');
        }
        return Hashing.sha256(canonical.toString());
    }
}
