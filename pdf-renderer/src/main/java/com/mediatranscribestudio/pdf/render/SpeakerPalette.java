package com.mediatranscribestudio.pdf.render;

import java.awt.Color;

public final class SpeakerPalette {
    private static final double GOLDEN_ANGLE = 137.508;

    public SpeakerStyle style(int order, String colorToken) {
        if (order < 1 || !("speaker." + order).equals(colorToken)) {
            throw new IllegalArgumentException("speaker colorToken/order mismatch");
        }
        double hue = (18.0 + (order - 1) * GOLDEN_ANGLE) % 360.0;
        Color accent = hsl(hue, 0.64, 0.27);
        Color background = hsl(hue, 0.48, 0.94);
        Color foreground = contrast(Color.WHITE, accent) >= 4.5
                ? Color.WHITE : new Color(23, 24, 22);
        return new SpeakerStyle(hex(accent), hex(background), hex(foreground));
    }

    public static double contrast(String first, String second) {
        return contrast(Color.decode(first), Color.decode(second));
    }

    private static Color hsl(double hue, double saturation, double lightness) {
        double chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
        double h = hue / 60.0;
        double x = chroma * (1 - Math.abs(h % 2 - 1));
        double r = 0;
        double g = 0;
        double b = 0;
        if (h < 1) {
            r = chroma;
            g = x;
        } else if (h < 2) {
            r = x;
            g = chroma;
        } else if (h < 3) {
            g = chroma;
            b = x;
        } else if (h < 4) {
            g = x;
            b = chroma;
        } else if (h < 5) {
            r = x;
            b = chroma;
        } else {
            r = chroma;
            b = x;
        }
        double offset = lightness - chroma / 2;
        return new Color((float) (r + offset), (float) (g + offset), (float) (b + offset));
    }

    private static double contrast(Color first, Color second) {
        double a = luminance(first);
        double b = luminance(second);
        return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    }

    private static double luminance(Color color) {
        return 0.2126 * channel(color.getRed())
                + 0.7152 * channel(color.getGreen())
                + 0.0722 * channel(color.getBlue());
    }

    private static double channel(int component) {
        double value = component / 255.0;
        return value <= 0.03928 ? value / 12.92
                : Math.pow((value + 0.055) / 1.055, 2.4);
    }

    private static String hex(Color color) {
        return String.format("#%02x%02x%02x", color.getRed(), color.getGreen(), color.getBlue());
    }

    public record SpeakerStyle(String accent, String background, String foreground) {
    }
}
