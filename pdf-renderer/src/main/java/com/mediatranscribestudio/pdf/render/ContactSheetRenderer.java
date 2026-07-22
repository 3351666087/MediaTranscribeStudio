package com.mediatranscribestudio.pdf.render;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontFormatException;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class ContactSheetRenderer {
    private static final int THUMBNAIL_WIDTH = 320;
    private static final int LABEL_HEIGHT = 28;
    private static final int GAP = 16;

    public void render(List<PageImageRenderer.PageEvidence> pages, Path output) throws IOException {
        if (pages.isEmpty()) {
            throw new IllegalArgumentException("at least one page image is required");
        }
        int columns = pages.size() == 1 ? 1 : pages.size() <= 4 ? 2 : 3;
        List<BufferedImage> images = new ArrayList<>();
        int thumbnailHeight = 0;
        try {
            for (PageImageRenderer.PageEvidence page : pages) {
                BufferedImage source = ImageIO.read(page.path().toFile());
                if (source == null) {
                    throw new IOException("unable to read page image " + page.path());
                }
                images.add(source);
                thumbnailHeight = Math.max(thumbnailHeight,
                        (int) Math.round(source.getHeight()
                                * (THUMBNAIL_WIDTH / (double) source.getWidth())));
            }
            int rows = (images.size() + columns - 1) / columns;
            int width = GAP + columns * (THUMBNAIL_WIDTH + GAP);
            int height = GAP + rows * (thumbnailHeight + LABEL_HEIGHT + GAP);
            BufferedImage sheet = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            Graphics2D graphics = sheet.createGraphics();
            try {
                graphics.setColor(new Color(244, 242, 238));
                graphics.fillRect(0, 0, width, height);
                graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BICUBIC);
                graphics.setFont(loadBundledFont());
                for (int index = 0; index < images.size(); index++) {
                    int row = index / columns;
                    int column = index % columns;
                    int x = GAP + column * (THUMBNAIL_WIDTH + GAP);
                    int y = GAP + row * (thumbnailHeight + LABEL_HEIGHT + GAP);
                    BufferedImage source = images.get(index);
                    int scaledHeight = (int) Math.round(source.getHeight()
                            * (THUMBNAIL_WIDTH / (double) source.getWidth()));
                    graphics.setColor(Color.WHITE);
                    graphics.fillRect(x, y, THUMBNAIL_WIDTH, thumbnailHeight);
                    graphics.drawImage(source, x, y, THUMBNAIL_WIDTH, scaledHeight, null);
                    graphics.setColor(new Color(23, 24, 22));
                    graphics.drawString(String.format("PAGE %03d", index + 1),
                            x, y + thumbnailHeight + 19);
                }
            } finally {
                graphics.dispose();
            }
            Files.createDirectories(output.toAbsolutePath().normalize().getParent());
            if (!ImageIO.write(sheet, "png", output.toFile())) {
                throw new IOException("PNG ImageIO writer is unavailable");
            }
            sheet.flush();
        } finally {
            images.forEach(BufferedImage::flush);
        }
    }

    private static Font loadBundledFont() throws IOException {
        try (InputStream input = ContactSheetRenderer.class.getResourceAsStream(
                "/fonts/LXGWWenKai-Regular.ttf")) {
            if (input == null) {
                throw new IOException("bundled contact-sheet font is missing");
            }
            return Font.createFont(Font.TRUETYPE_FONT, input).deriveFont(Font.BOLD, 14f);
        } catch (FontFormatException exception) {
            throw new IOException("bundled contact-sheet font is invalid", exception);
        }
    }
}
