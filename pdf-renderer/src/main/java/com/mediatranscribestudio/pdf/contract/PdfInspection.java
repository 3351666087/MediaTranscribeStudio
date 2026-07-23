package com.mediatranscribestudio.pdf.contract;

import java.util.ArrayList;
import java.util.List;

public final class PdfInspection {
    public String schemaVersion = "1.0.0";
    public String validator = "PDFBox";
    public String validatorVersion = "2.0.30";
    public String documentId;
    public String pdfSha256;
    public boolean openable;
    public int pageCount;
    public boolean allPagesA4;
    public boolean transcriptTextIntegrity;
    public boolean segmentCountIntegrity;
    public boolean timestampIntegrity;
    public boolean speakerSetIntegrity;
    public boolean allFontsEmbedded;
    public boolean searchableText;
    public boolean noReplacementCharacters;
    public String extractedTextSha256 = "";
    public String failure;
    public List<Page> pages = new ArrayList<>();
    public List<Font> fonts = new ArrayList<>();
    public List<String> missingSegmentIds = new ArrayList<>();
    public List<String> duplicateSegmentIds = new ArrayList<>();
    public List<String> missingTimestamps = new ArrayList<>();
    public List<String> missingSpeakerIds = new ArrayList<>();

    public static final class Page {
        public int pageNumber;
        public float widthPt;
        public float heightPt;
        public boolean a4;
        public int extractedCharacterCount;
    }

    public static final class Font {
        public String name;
        public String subtype;
        public boolean embedded;
    }
}
