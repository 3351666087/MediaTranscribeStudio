package com.mediatranscribestudio.pdf.inspect;

import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.support.Hashing;
import com.mediatranscribestudio.pdf.validation.ReportDocumentValidator;
import org.apache.pdfbox.cos.COSArray;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.cos.COSString;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.OffsetDateTime;
import java.util.GregorianCalendar;
import java.util.HexFormat;
import java.util.List;

public final class PdfMetadataWriter {
    public void apply(
            Path pdf,
            ReportDocument source,
            String requestId,
            String integritySha256
    ) throws IOException {
        ReportDocumentValidator.validate(source);
        require(Files.isRegularFile(pdf), "PDF must be an existing regular file");
        requireText(requestId, "requestId");
        require(integritySha256 != null && integritySha256.matches("[a-f0-9]{64}"),
                "integritySha256 must be a lowercase SHA-256 value");

        List<String> speakerIds =
                ReportDocumentValidator.canonicalSpeakerIds(source.speakerPolicy.resolvedCount);
        String speakerIdValue = String.join(",", speakerIds);
        Path temporary = pdf.resolveSibling("." + pdf.getFileName() + ".metadata");
        Files.deleteIfExists(temporary);
        try {
            try (PDDocument document = PDDocument.load(pdf.toFile())) {
                PDDocumentInformation information = document.getDocumentInformation();
                information.setTitle(title(source));
                information.setSubject("中文逐字稿，包含动态任意 N 说话人和毫秒级时间戳");
                information.setCreator("MediaTranscribeStudio PDF Renderer 3.0.0");
                information.setProducer("OpenHTMLtoPDF 1.0.10 + Apache PDFBox 2.0.30");
                GregorianCalendar generatedAt = GregorianCalendar.from(
                        OffsetDateTime.parse(source.generatedAt).toZonedDateTime());
                information.setCreationDate((GregorianCalendar) generatedAt.clone());
                information.setModificationDate((GregorianCalendar) generatedAt.clone());
                information.setCustomMetadataValue("MTS-Request-Id", requestId);
                information.setCustomMetadataValue("MTS-Transcript-SHA256", integritySha256);
                information.setCustomMetadataValue(
                        "MTS-Speaker-Count",
                        Integer.toString(source.speakerPolicy.resolvedCount)
                );
                information.setCustomMetadataValue("MTS-Speaker-Ids", speakerIdValue);
                information.setCustomMetadataValue(
                        "MTS-Speaker-Set-SHA256",
                        Hashing.sha256(speakerIdValue)
                );
                COSString stableId = new COSString(HexFormat.of().parseHex(
                        integritySha256.substring(0, 32)));
                COSArray identifiers = new COSArray();
                identifiers.add(stableId);
                identifiers.add(stableId);
                document.getDocument().getTrailer().setItem(COSName.ID, identifiers);
                document.save(temporary.toFile());
            }
            Files.move(temporary, pdf, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private static String title(ReportDocument source) {
        return source.title == null || source.title.isBlank() ? "中文逐字稿" : source.title;
    }

    private static void requireText(String value, String field) {
        require(value != null && !value.isBlank(), field + " is required");
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}
