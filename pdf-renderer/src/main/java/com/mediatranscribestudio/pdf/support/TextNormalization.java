package com.mediatranscribestudio.pdf.support;

import java.text.Normalizer;

public final class TextNormalization {
    private TextNormalization() {
    }

    public static String forIntegrity(String value) {
        if (value == null) {
            return "";
        }
        return Normalizer.normalize(value, Normalizer.Form.NFKC)
                .replaceAll("[\\p{Z}\\s\\p{Cc}]+", "");
    }
}
