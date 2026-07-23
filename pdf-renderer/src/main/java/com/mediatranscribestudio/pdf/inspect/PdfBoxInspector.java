package com.mediatranscribestudio.pdf.inspect;

import com.mediatranscribestudio.pdf.contract.PdfInspection;
import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.support.Hashing;
import com.mediatranscribestudio.pdf.support.TextNormalization;
import com.mediatranscribestudio.pdf.support.Timecodes;
import org.apache.pdfbox.cos.COSBase;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDResources;
import org.apache.pdfbox.pdmodel.font.PDFont;
import org.apache.pdfbox.pdmodel.graphics.PDXObject;
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public final class PdfBoxInspector {
    private static final float A4_WIDTH = 595.276f;
    private static final float A4_HEIGHT = 841.890f;
    private static final float TOLERANCE = 2.0f;

    public InspectionResult inspect(Path pdf, ReportDocument source) {
        PdfInspection result = new PdfInspection();
        result.documentId = source.documentId;
        try {
            result.pdfSha256 = Hashing.sha256(pdf);
        } catch (IOException exception) {
            result.openable = false;
            result.failure = exception.getClass().getSimpleName() + ": " + exception.getMessage();
            return new InspectionResult(result, "");
        }
        String extracted = "";
        try (PDDocument document = PDDocument.load(pdf.toFile())) {
            result.openable = true;
            result.pageCount = document.getNumberOfPages();
            PDFTextStripper stripper = new PDFTextStripper();
            stripper.setSortByPosition(false);
            extracted = stripper.getText(document);
            result.extractedTextSha256 = Hashing.sha256(extracted);
            result.noReplacementCharacters = extracted.indexOf('\ufffd') < 0;
            String normalized = TextNormalization.forIntegrity(extracted);
            result.searchableText = normalized.codePointCount(0, normalized.length()) >=
                    source.segments.stream()
                            .mapToInt(segment -> TextNormalization.forIntegrity(segment.displayText)
                                    .codePointCount(0,
                                            TextNormalization.forIntegrity(segment.displayText).length()))
                            .sum();

            result.allPagesA4 = result.pageCount > 0;
            for (int index = 0; index < document.getNumberOfPages(); index++) {
                PDPage page = document.getPage(index);
                PdfInspection.Page pageResult = new PdfInspection.Page();
                pageResult.pageNumber = index + 1;
                pageResult.widthPt = page.getMediaBox().getWidth();
                pageResult.heightPt = page.getMediaBox().getHeight();
                pageResult.a4 = isA4(pageResult.widthPt, pageResult.heightPt);
                result.allPagesA4 &= pageResult.a4;
                PDFTextStripper pageStripper = new PDFTextStripper();
                pageStripper.setStartPage(index + 1);
                pageStripper.setEndPage(index + 1);
                String pageText = TextNormalization.forIntegrity(pageStripper.getText(document));
                pageResult.extractedCharacterCount = pageText.codePointCount(0, pageText.length());
                result.pages.add(pageResult);
            }

            result.transcriptTextIntegrity = true;
            for (ReportDocument.Segment segment : source.segments) {
                if (!normalized.contains(TextNormalization.forIntegrity(segment.displayText))) {
                    result.transcriptTextIntegrity = false;
                    result.missingSegmentIds.add(segment.id);
                }
            }

            result.segmentCountIntegrity = true;
            for (ReportDocument.Segment segment : source.segments) {
                int occurrences = occurrences(normalized, TextNormalization.forIntegrity(segment.id));
                if (occurrences == 0) {
                    result.segmentCountIntegrity = false;
                    if (!result.missingSegmentIds.contains(segment.id)) {
                        result.missingSegmentIds.add(segment.id);
                    }
                } else if (occurrences != 1) {
                    result.segmentCountIntegrity = false;
                    result.duplicateSegmentIds.add(segment.id);
                }
            }

            Map<String, Integer> expectedTimestamps = new LinkedHashMap<>();
            for (ReportDocument.Segment segment : source.segments) {
                expectedTimestamps.merge(Timecodes.milliseconds(segment.startMs), 1, Integer::sum);
                expectedTimestamps.merge(Timecodes.milliseconds(segment.endMs), 1, Integer::sum);
            }
            result.timestampIntegrity = true;
            for (Map.Entry<String, Integer> entry : expectedTimestamps.entrySet()) {
                int actual = occurrences(extracted, entry.getKey());
                if (actual != entry.getValue()) {
                    result.timestampIntegrity = false;
                    result.missingTimestamps.add(entry.getKey() + " expected="
                            + entry.getValue() + " actual=" + actual);
                }
            }

            result.speakerSetIntegrity = true;
            for (ReportDocument.Speaker speaker : source.speakers) {
                if (!normalized.contains(TextNormalization.forIntegrity(speaker.id))
                        || !normalized.contains(TextNormalization.forIntegrity(speaker.displayName))
                        || !normalized.contains(TextNormalization.forIntegrity(speaker.shortLabel))) {
                    result.speakerSetIntegrity = false;
                    result.missingSpeakerIds.add(speaker.id);
                }
            }

            Set<COSBase> resources = Collections.newSetFromMap(new IdentityHashMap<>());
            Set<COSBase> fonts = Collections.newSetFromMap(new IdentityHashMap<>());
            for (PDPage page : document.getPages()) {
                inspectResources(page.getResources(), result, resources, fonts);
            }
            result.allFontsEmbedded = !result.fonts.isEmpty()
                    && result.fonts.stream().allMatch(font -> font.embedded);
        } catch (Exception exception) {
            result.openable = false;
            result.failure = exception.getClass().getSimpleName() + ": " + exception.getMessage();
        }
        return new InspectionResult(result, extracted);
    }

    private static void inspectResources(
            PDResources resources,
            PdfInspection result,
            Set<COSBase> visitedResources,
            Set<COSBase> visitedFonts
    ) throws IOException {
        if (resources == null || !visitedResources.add(resources.getCOSObject())) {
            return;
        }
        for (COSName name : resources.getFontNames()) {
            PDFont font = resources.getFont(name);
            if (font == null || !visitedFonts.add(font.getCOSObject())) {
                continue;
            }
            PdfInspection.Font fontResult = new PdfInspection.Font();
            fontResult.name = font.getName();
            fontResult.subtype = font.getCOSObject().getNameAsString(COSName.SUBTYPE, "UNKNOWN");
            fontResult.embedded = font.isEmbedded();
            result.fonts.add(fontResult);
        }
        for (COSName name : resources.getXObjectNames()) {
            PDXObject object = resources.getXObject(name);
            if (object instanceof PDFormXObject form) {
                inspectResources(form.getResources(), result, visitedResources, visitedFonts);
            }
        }
    }

    private static int occurrences(String haystack, String needle) {
        if (needle.isEmpty()) {
            return 0;
        }
        int count = 0;
        int cursor = 0;
        while ((cursor = haystack.indexOf(needle, cursor)) >= 0) {
            count++;
            cursor += needle.length();
        }
        return count;
    }

    private static boolean isA4(float width, float height) {
        return (near(width, A4_WIDTH) && near(height, A4_HEIGHT))
                || (near(width, A4_HEIGHT) && near(height, A4_WIDTH));
    }

    private static boolean near(float actual, float expected) {
        return Math.abs(actual - expected) <= TOLERANCE;
    }

    public record InspectionResult(PdfInspection inspection, String extractedText) {
    }
}
