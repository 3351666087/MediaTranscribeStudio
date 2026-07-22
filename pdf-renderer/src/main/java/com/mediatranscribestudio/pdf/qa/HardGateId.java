package com.mediatranscribestudio.pdf.qa;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public enum HardGateId {
    PDF_OPENABLE("PDF-OPENABLE"),
    PDF_PAGE_COUNT("PDF-PAGE-COUNT"),
    PDF_PAGE_SIZE("PDF-PAGE-SIZE"),
    PDF_TRANSCRIPT_TEXT_INTEGRITY("PDF-TRANSCRIPT-TEXT-INTEGRITY"),
    PDF_SEGMENT_COUNT("PDF-SEGMENT-COUNT"),
    PDF_TIMESTAMP_INTEGRITY("PDF-TIMESTAMP-INTEGRITY"),
    PDF_SPEAKER_SET_INTEGRITY("PDF-SPEAKER-SET-INTEGRITY"),
    PDF_FONT_EMBEDDED("PDF-FONT-EMBEDDED"),
    PDF_NO_BLANK_PAGES("PDF-NO-BLANK-PAGES"),
    PDF_NO_CONTENT_OVERFLOW("PDF-NO-CONTENT-OVERFLOW"),
    PDF_OFFLINE_ASSETS("PDF-OFFLINE-ASSETS"),
    PDF_PAGE_EVIDENCE("PDF-PAGE-EVIDENCE"),
    PDF_IMMUTABLE_CONTENT_HASH("PDF-IMMUTABLE-CONTENT-HASH");

    static {
        Set<String> ids = new HashSet<>();
        for (HardGateId item : values()) {
            if (!ids.add(item.id)) {
                throw new ExceptionInInitializerError("duplicate hard gate ID: " + item.id);
            }
        }
    }

    private final String id;

    HardGateId(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }

    public static List<String> ids() {
        return Arrays.stream(values()).map(HardGateId::id).toList();
    }
}
