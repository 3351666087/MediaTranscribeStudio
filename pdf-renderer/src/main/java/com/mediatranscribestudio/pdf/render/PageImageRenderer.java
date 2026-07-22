package com.mediatranscribestudio.pdf.render;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class PageImageRenderer {
    public Result render(Path pdf, Path pagesDirectory, int dpi) throws IOException {
        deleteDirectory(pagesDirectory);
        Files.createDirectories(pagesDirectory);
        List<PageEvidence> evidence = new ArrayList<>();
        try (PDDocument document = PDDocument.load(pdf.toFile())) {
            PDFRenderer renderer = new PDFRenderer(document);
            for (int index = 0; index < document.getNumberOfPages(); index++) {
                BufferedImage image = renderer.renderImageWithDPI(index, dpi, ImageType.RGB);
                Path output = pagesDirectory.resolve(String.format("page-%03d.png", index + 1));
                if (!ImageIO.write(image, "png", output.toFile())) {
                    throw new IOException("PNG ImageIO writer is unavailable");
                }
                evidence.add(analyze(output, index + 1, image));
                image.flush();
            }
        }
        return new Result(dpi, List.copyOf(evidence));
    }

    private static PageEvidence analyze(Path path, int pageNumber, BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        long ink = 0;
        int minX = width;
        int minY = height;
        int maxX = -1;
        int maxY = -1;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >>> 16) & 0xff;
                int green = (rgb >>> 8) & 0xff;
                int blue = rgb & 0xff;
                if (red < 247 || green < 247 || blue < 247) {
                    ink++;
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        double ratio = ink / (double) (width * (long) height);
        boolean blank = ratio < 0.0006;
        int safety = Math.max(3, Math.round(Math.min(width, height) * 0.003f));
        boolean edgeTouch = !blank && (minX <= safety || minY <= safety
                || maxX >= width - safety - 1 || maxY >= height - safety - 1);
        return new PageEvidence(path, pageNumber, width, height, ratio, blank, edgeTouch);
    }

    private static void deleteDirectory(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return;
        }
        try (var paths = Files.walk(directory)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(path);
            }
        }
    }

    public record Result(int dpi, List<PageEvidence> pages) {
        public int blankPageCount() {
            return (int) pages.stream().filter(PageEvidence::blank).count();
        }

        public int edgeTouchPageCount() {
            return (int) pages.stream().filter(PageEvidence::edgeTouch).count();
        }

        public double minimumInkRatio() {
            return pages.stream().map(PageEvidence::inkRatio)
                    .min(Comparator.naturalOrder()).orElse(0.0);
        }

        public double maximumInkRatio() {
            return pages.stream().map(PageEvidence::inkRatio)
                    .max(Comparator.naturalOrder()).orElse(0.0);
        }
    }

    public record PageEvidence(
            Path path,
            int pageNumber,
            int width,
            int height,
            double inkRatio,
            boolean blank,
            boolean edgeTouch
    ) {
    }
}
