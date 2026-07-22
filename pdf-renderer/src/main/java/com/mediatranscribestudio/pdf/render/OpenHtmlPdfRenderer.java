package com.mediatranscribestudio.pdf.render;

import com.mediatranscribestudio.pdf.support.AtomicFiles;
import com.mediatranscribestudio.pdf.support.Hashing;
import com.mediatranscribestudio.pdf.support.OfflinePolicy;
import com.openhtmltopdf.outputdevice.helper.BaseRendererBuilder;
import com.openhtmltopdf.pdfboxout.PdfRendererBuilder;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

public final class OpenHtmlPdfRenderer {
    private static final String FONT_RESOURCE = "/fonts/LXGWWenKai-Regular.ttf";
    private static final String FONT_SHA256 =
            "39ad71264b588165b469e35e6afb162a378dacd1f95348160240ba9038ac3009";

    public void render(String xhtml, Path output) throws IOException {
        OfflinePolicy.requireOfflineXhtml(xhtml);
        if (output == null) {
            throw new IllegalArgumentException("output path is required");
        }
        Files.createDirectories(output.toAbsolutePath().normalize().getParent());
        Path temporaryPdf = output.resolveSibling("." + output.getFileName() + ".rendering");
        Path temporaryFont = output.getParent().resolve(
                ".resources/LXGWWenKai-Regular.ttf");
        try {
            try (InputStream input = OpenHtmlPdfRenderer.class.getResourceAsStream(FONT_RESOURCE)) {
                if (input == null) {
                    throw new IOException("bundled CJK font is missing");
                }
                AtomicFiles.write(temporaryFont, input.readAllBytes());
            }
            if (!FONT_SHA256.equals(Hashing.sha256(temporaryFont))) {
                throw new IOException("bundled CJK font checksum mismatch");
            }
            try (OutputStream stream = Files.newOutputStream(temporaryPdf)) {
                PdfRendererBuilder builder = new PdfRendererBuilder();
                builder.useFastMode();
                builder.withHtmlContent(xhtml, null);
                builder.useFont(temporaryFont.toFile(), "MTS CJK", 400,
                        BaseRendererBuilder.FontStyle.NORMAL, true);
                builder.useFont(temporaryFont.toFile(), "MTS CJK", 700,
                        BaseRendererBuilder.FontStyle.NORMAL, true);
                builder.toStream(stream);
                builder.run();
            } catch (Exception exception) {
                if (exception instanceof IOException ioException) {
                    throw ioException;
                }
                throw new IOException("OpenHTMLtoPDF rendering failed", exception);
            }
            AtomicFiles.copy(temporaryPdf, output);
        } finally {
            Files.deleteIfExists(temporaryPdf);
            Files.deleteIfExists(temporaryFont);
            Files.deleteIfExists(temporaryFont.getParent());
        }
    }
}
