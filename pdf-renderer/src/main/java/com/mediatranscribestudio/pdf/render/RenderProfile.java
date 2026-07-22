package com.mediatranscribestudio.pdf.render;

import java.util.List;

public record RenderProfile(
        String id,
        double pageMarginMm,
        double bodyFontPt,
        double bodyLineHeight,
        double turnPaddingMm,
        double legendFontPt
) {
    private static final List<RenderProfile> PROFILES = List.of(
            new RenderProfile("balanced", 0, 9.4, 1.58, 2.4, 8.2),
            new RenderProfile("compact", 0, 9.1, 1.50, 2.1, 7.9),
            new RenderProfile("roomier-margin", 2, 9.0, 1.50, 2.0, 7.8),
            new RenderProfile("conservative-pagination", 3, 8.8, 1.46, 1.8, 7.6),
            new RenderProfile("final-safe", 4, 8.6, 1.42, 1.6, 7.4)
    );

    public static RenderProfile forRound(int round, double requestedMarginMm) {
        if (round < 1 || round > PROFILES.size()) {
            throw new IllegalArgumentException("round must be between 1 and 5");
        }
        RenderProfile base = PROFILES.get(round - 1);
        return new RenderProfile(
                base.id,
                Math.min(30, requestedMarginMm + base.pageMarginMm),
                base.bodyFontPt,
                base.bodyLineHeight,
                base.turnPaddingMm,
                base.legendFontPt
        );
    }
}
