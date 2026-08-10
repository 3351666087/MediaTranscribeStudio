package com.mediatranscribestudio.pdf.inspect;

import com.mediatranscribestudio.pdf.contract.BlindReviewPdfScanRequest;
import com.mediatranscribestudio.pdf.contract.BlindReviewPdfScanResult;
import com.mediatranscribestudio.pdf.support.Hashing;
import org.apache.pdfbox.cos.COSArray;
import org.apache.pdfbox.cos.COSBase;
import org.apache.pdfbox.cos.COSDictionary;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.cos.COSObject;
import org.apache.pdfbox.cos.COSStream;
import org.apache.pdfbox.cos.COSString;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentCatalog;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;
import org.apache.pdfbox.pdmodel.PDDocumentNameDictionary;
import org.apache.pdfbox.pdmodel.PDEmbeddedFilesNameTreeNode;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.common.PDMetadata;
import org.apache.pdfbox.pdmodel.common.PDNameTreeNode;
import org.apache.pdfbox.pdmodel.common.filespecification.PDComplexFileSpecification;
import org.apache.pdfbox.pdmodel.common.filespecification.PDEmbeddedFile;
import org.apache.pdfbox.pdmodel.common.filespecification.PDFileSpecification;
import org.apache.pdfbox.pdmodel.interactive.annotation.PDAnnotation;
import org.apache.pdfbox.pdmodel.interactive.annotation.PDAnnotationFileAttachment;
import org.apache.pdfbox.pdmodel.interactive.form.PDAcroForm;
import org.apache.pdfbox.pdmodel.interactive.form.PDField;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

public final class BlindReviewPdfInspector {
    private static final Pattern SHA256 = Pattern.compile("^[a-f0-9]{64}$");
    private static final long MAX_PDF_BYTES = 512L * 1024L * 1024L;
    private static final int MAX_SENSITIVE_VALUES = 1024;
    private static final int MAX_SENSITIVE_VALUE_LENGTH = 8192;
    private static final int MIN_SENSITIVE_VALUE_LENGTH = 4;
    private static final int MAX_COS_OBJECTS = 200_000;
    private static final int MAX_COS_DEPTH = 256;
    private static final int MAX_STREAM_BYTES = 64 * 1024 * 1024;
    private static final long MAX_TOTAL_STREAM_BYTES = 512L * 1024L * 1024L;
    private static final int MAX_FINDINGS = 256;

    public BlindReviewPdfScanResult inspect(BlindReviewPdfScanRequest request)
            throws IOException {
        ValidatedRequest validated = validate(request);
        BlindReviewPdfScanResult result = new BlindReviewPdfScanResult();
        result.pdfSha256 = validated.expectedPdfSha256();
        result.sensitiveValueCount = validated.sensitiveValues().size();
        Matcher matcher = new Matcher(validated.sensitiveValues(), result);
        ScanContext context = new ScanContext(result, matcher);

        String beforeSha256 = Hashing.sha256(validated.pdf());
        if (!beforeSha256.equals(validated.expectedPdfSha256())) {
            throw new IllegalArgumentException(
                    "PDF SHA-256 does not match the frozen request");
        }

        try (PDDocument document = PDDocument.load(validated.pdf().toFile())) {
            if (document.isEncrypted()) {
                throw new IOException(
                        "encrypted PDFs cannot enter a blind reviewer packet");
            }
            result.pageCount = document.getNumberOfPages();
            scanPageText(document, context);
            scanDocumentInfo(document.getDocumentInformation(), context);
            PDDocumentCatalog catalog = document.getDocumentCatalog();
            scanXmp(catalog.getMetadata(), context);
            scanNamedAttachments(catalog.getNames(), context);
            scanAnnotations(document, context);
            scanForm(catalog.getAcroForm(), context);
            scanCos(
                    document.getDocument().getTrailer(),
                    "cos-object",
                    context,
                    Collections.newSetFromMap(new IdentityHashMap<>()),
                    0
            );
            result.allRequiredSurfacesScanned = true;
        }

        String afterSha256 = Hashing.sha256(validated.pdf());
        if (!beforeSha256.equals(afterSha256)) {
            throw new IOException("PDF changed during PDFBox blind scan");
        }
        result.pdfSha256 = afterSha256;
        result.status = result.findings.isEmpty() ? "passed" : "blocked";
        return result;
    }

    private static ValidatedRequest validate(BlindReviewPdfScanRequest request)
            throws IOException {
        if (request == null || !"1.0.0".equals(request.schemaVersion)) {
            throw new IllegalArgumentException(
                    "blind PDF scan schemaVersion must be 1.0.0");
        }
        if (request.pdfPath == null || request.pdfPath.isBlank()) {
            throw new IllegalArgumentException(
                    "blind PDF scan pdfPath must be non-empty");
        }
        if (request.expectedPdfSha256 == null
                || !SHA256.matcher(request.expectedPdfSha256).matches()) {
            throw new IllegalArgumentException(
                    "blind PDF scan expectedPdfSha256 must be lowercase SHA-256");
        }
        Path pdf = Path.of(request.pdfPath).toAbsolutePath().normalize();
        if (!Files.isRegularFile(pdf)) {
            throw new IllegalArgumentException(
                    "blind PDF scan target is not a regular file");
        }
        long size = Files.size(pdf);
        if (size <= 0 || size > MAX_PDF_BYTES) {
            throw new IllegalArgumentException(
                    "blind PDF scan target has an unsupported size");
        }
        if (request.sensitiveValues == null
                || request.sensitiveValues.isEmpty()
                || request.sensitiveValues.size() > MAX_SENSITIVE_VALUES) {
            throw new IllegalArgumentException(
                    "blind PDF scan sensitiveValues are missing or excessive");
        }
        Map<String, SensitiveValue> values = new LinkedHashMap<>();
        for (String raw : request.sensitiveValues) {
            if (raw == null
                    || raw.length() > MAX_SENSITIVE_VALUE_LENGTH) {
                throw new IllegalArgumentException(
                        "blind PDF scan contains an invalid sensitive value");
            }
            String stripped = raw.strip();
            String normalized = normalize(stripped);
            if (normalized.codePointCount(0, normalized.length())
                    < MIN_SENSITIVE_VALUE_LENGTH) {
                continue;
            }
            values.putIfAbsent(
                    normalized,
                    new SensitiveValue(
                            normalized,
                            Hashing.sha256(stripped)
                    )
            );
        }
        if (values.isEmpty()) {
            throw new IllegalArgumentException(
                    "blind PDF scan has no usable sensitive values");
        }
        return new ValidatedRequest(
                pdf,
                request.expectedPdfSha256,
                List.copyOf(values.values())
        );
    }

    private static void scanPageText(
            PDDocument document,
            ScanContext context
    ) throws IOException {
        for (int pageIndex = 0; pageIndex < document.getNumberOfPages(); pageIndex++) {
            PDFTextStripper stripper = new PDFTextStripper();
            stripper.setSortByPosition(false);
            stripper.setStartPage(pageIndex + 1);
            stripper.setEndPage(pageIndex + 1);
            context.matcher.scan("page-text", stripper.getText(document));
            context.result.pageTextCount++;
        }
    }

    private static void scanDocumentInfo(
            PDDocumentInformation information,
            ScanContext context
    ) throws IOException {
        if (information == null) {
            return;
        }
        COSDictionary dictionary = information.getCOSObject();
        context.result.documentInfoEntryCount = dictionary.size();
        scanCos(
                dictionary,
                "document-info",
                context,
                Collections.newSetFromMap(new IdentityHashMap<>()),
                0
        );
    }

    private static void scanXmp(
            PDMetadata metadata,
            ScanContext context
    ) throws IOException {
        if (metadata == null) {
            return;
        }
        context.result.xmpPacketCount = 1;
        COSStream stream = metadata.getCOSObject();
        try (InputStream input = metadata.exportXMPMetadata()) {
            context.scanDecodedStream(stream, input, "xmp");
        }
        scanCos(
                stream,
                "xmp",
                context,
                Collections.newSetFromMap(new IdentityHashMap<>()),
                0
        );
    }

    private static void scanNamedAttachments(
            PDDocumentNameDictionary names,
            ScanContext context
    ) throws IOException {
        if (names == null || names.getEmbeddedFiles() == null) {
            return;
        }
        scanAttachmentTree(names.getEmbeddedFiles(), context);
    }

    private static void scanAttachmentTree(
            PDNameTreeNode<PDComplexFileSpecification> node,
            ScanContext context
    ) throws IOException {
        Map<String, PDComplexFileSpecification> names = node.getNames();
        if (names != null) {
            for (Map.Entry<String, PDComplexFileSpecification> entry
                    : names.entrySet()) {
                context.matcher.scan("attachment", entry.getKey());
                scanFileSpecification(entry.getValue(), context);
            }
        }
        List<PDNameTreeNode<PDComplexFileSpecification>> kids = node.getKids();
        if (kids != null) {
            for (PDNameTreeNode<PDComplexFileSpecification> child : kids) {
                scanAttachmentTree(child, context);
            }
        }
    }

    private static void scanFileSpecification(
            PDFileSpecification specification,
            ScanContext context
    ) throws IOException {
        if (!(specification instanceof PDComplexFileSpecification complex)) {
            if (specification != null) {
                scanCos(
                        specification.getCOSObject(),
                        "attachment",
                        context,
                        Collections.newSetFromMap(new IdentityHashMap<>()),
                        0
                );
            }
            return;
        }
        context.matcher.scan("attachment", complex.getFilename());
        context.matcher.scan("attachment", complex.getFile());
        context.matcher.scan("attachment", complex.getFileUnicode());
        context.matcher.scan("attachment", complex.getFileDos());
        context.matcher.scan("attachment", complex.getFileMac());
        context.matcher.scan("attachment", complex.getFileUnix());
        context.matcher.scan("attachment", complex.getFileDescription());
        for (PDEmbeddedFile embedded : new PDEmbeddedFile[]{
                complex.getEmbeddedFile(),
                complex.getEmbeddedFileUnicode(),
                complex.getEmbeddedFileDos(),
                complex.getEmbeddedFileMac(),
                complex.getEmbeddedFileUnix()
        }) {
            if (embedded == null
                    || !context.attachmentStreams.add(embedded.getCOSObject())) {
                continue;
            }
            context.result.attachmentCount++;
            try (InputStream input = embedded.createInputStream()) {
                context.scanDecodedStream(
                        embedded.getCOSObject(),
                        input,
                        "attachment"
                );
            }
        }
        scanCos(
                complex.getCOSObject(),
                "attachment",
                context,
                Collections.newSetFromMap(new IdentityHashMap<>()),
                0
        );
    }

    private static void scanAnnotations(
            PDDocument document,
            ScanContext context
    ) throws IOException {
        for (PDPage page : document.getPages()) {
            for (PDAnnotation annotation : page.getAnnotations()) {
                context.result.annotationCount++;
                context.matcher.scan("annotation", annotation.getContents());
                context.matcher.scan(
                        "annotation",
                        annotation.getAnnotationName()
                );
                context.matcher.scan("annotation", annotation.getModifiedDate());
                context.matcher.scan("annotation", annotation.getSubtype());
                if (annotation instanceof PDAnnotationFileAttachment fileAttachment) {
                    scanFileSpecification(fileAttachment.getFile(), context);
                }
                scanCos(
                        annotation.getCOSObject(),
                        "annotation",
                        context,
                        Collections.newSetFromMap(new IdentityHashMap<>()),
                        0
                );
            }
        }
    }

    private static void scanForm(
            PDAcroForm form,
            ScanContext context
    ) throws IOException {
        if (form == null) {
            return;
        }
        for (PDField field : form.getFieldTree()) {
            context.result.formFieldCount++;
            context.matcher.scan("form", field.getFullyQualifiedName());
            context.matcher.scan("form", field.getPartialName());
            context.matcher.scan("form", field.getAlternateFieldName());
            context.matcher.scan("form", field.getMappingName());
            context.matcher.scan("form", field.getValueAsString());
            scanCos(
                    field.getCOSObject(),
                    "form",
                    context,
                    Collections.newSetFromMap(new IdentityHashMap<>()),
                    0
            );
        }
        scanCos(
                form.getCOSObject(),
                "form",
                context,
                Collections.newSetFromMap(new IdentityHashMap<>()),
                0
        );
    }

    private static void scanCos(
            COSBase base,
            String location,
            ScanContext context,
            Set<COSBase> visited,
            int depth
    ) throws IOException {
        if (base == null || depth > MAX_COS_DEPTH || !visited.add(base)) {
            if (depth > MAX_COS_DEPTH) {
                throw new IOException("PDF COS graph exceeds the scan depth limit");
            }
            return;
        }
        if (context.uniqueCosObjects.add(base)) {
            context.result.scannedCosObjectCount++;
            if (context.result.scannedCosObjectCount > MAX_COS_OBJECTS) {
                throw new IOException(
                        "PDF COS graph exceeds the scan object limit");
            }
        }
        if (base instanceof COSObject object) {
            scanCos(object.getObject(), location, context, visited, depth + 1);
        } else if (base instanceof COSString string) {
            context.matcher.scan(location, string.getString());
            context.matcher.scanBytes(location, string.getBytes());
        } else if (base instanceof COSName name) {
            context.matcher.scan(location, name.getName());
        } else if (base instanceof COSArray array) {
            for (COSBase item : array) {
                scanCos(item, location, context, visited, depth + 1);
            }
        } else if (base instanceof COSDictionary dictionary) {
            for (COSName key : dictionary.keySet()) {
                context.matcher.scan(location, key.getName());
            }
            if (dictionary instanceof COSStream stream
                    && !context.decodedStreams.contains(stream)) {
                try (InputStream input = stream.createInputStream()) {
                    context.scanDecodedStream(stream, input, location);
                }
            }
            for (COSName key : dictionary.keySet()) {
                scanCos(
                        dictionary.getDictionaryObject(key),
                        location,
                        context,
                        visited,
                        depth + 1
                );
            }
        }
    }

    private static byte[] readBounded(InputStream input, int maximum)
            throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[64 * 1024];
        int read;
        while ((read = input.read(buffer)) >= 0) {
            if (output.size() + read > maximum) {
                throw new IOException(
                        "decoded PDF stream exceeds the per-stream limit");
            }
            output.write(buffer, 0, read);
        }
        return output.toByteArray();
    }

    private static String normalize(String value) {
        return Normalizer.normalize(value, Normalizer.Form.NFKC)
                .toLowerCase(Locale.ROOT);
    }

    private record SensitiveValue(String normalized, String sha256) {
    }

    private record ValidatedRequest(
            Path pdf,
            String expectedPdfSha256,
            List<SensitiveValue> sensitiveValues
    ) {
    }

    private static final class Matcher {
        private final List<SensitiveValue> values;
        private final BlindReviewPdfScanResult result;
        private final Set<String> findingKeys = Collections.newSetFromMap(
                new LinkedHashMap<>());

        private Matcher(
                List<SensitiveValue> values,
                BlindReviewPdfScanResult result
        ) {
            this.values = new ArrayList<>(values);
            this.result = result;
        }

        private void scan(String location, String value) {
            if (value == null || value.isEmpty()) {
                return;
            }
            String normalized = normalize(value);
            for (SensitiveValue sensitive : values) {
                if (normalized.contains(sensitive.normalized())) {
                    addFinding(location, sensitive.sha256());
                }
            }
        }

        private void scanBytes(String location, byte[] value) {
            if (value == null || value.length == 0) {
                return;
            }
            scan(location, new String(value, StandardCharsets.UTF_8));
            scan(location, new String(value, StandardCharsets.UTF_16LE));
            scan(location, new String(value, StandardCharsets.UTF_16BE));
            scan(location, new String(value, StandardCharsets.ISO_8859_1));
        }

        private void addFinding(String location, String digest) {
            String key = location + "\0" + digest;
            if (!findingKeys.add(key)
                    || result.findings.size() >= MAX_FINDINGS) {
                return;
            }
            result.findings.add(
                    BlindReviewPdfScanResult.Finding.of(location, digest));
        }
    }

    private static final class ScanContext {
        private final BlindReviewPdfScanResult result;
        private final Matcher matcher;
        private final Set<COSBase> uniqueCosObjects =
                Collections.newSetFromMap(new IdentityHashMap<>());
        private final Set<COSBase> decodedStreams =
                Collections.newSetFromMap(new IdentityHashMap<>());
        private final Set<COSBase> attachmentStreams =
                Collections.newSetFromMap(new IdentityHashMap<>());

        private ScanContext(
                BlindReviewPdfScanResult result,
                Matcher matcher
        ) {
            this.result = result;
            this.matcher = matcher;
        }

        private void scanDecodedStream(
                COSStream stream,
                InputStream input,
                String location
        ) throws IOException {
            if (!decodedStreams.add(stream)) {
                return;
            }
            byte[] decoded = readBounded(input, MAX_STREAM_BYTES);
            if (result.decodedStreamBytes + decoded.length
                    > MAX_TOTAL_STREAM_BYTES) {
                throw new IOException(
                        "decoded PDF streams exceed the aggregate limit");
            }
            result.decodedStreamBytes += decoded.length;
            matcher.scanBytes(location, decoded);
        }
    }
}
