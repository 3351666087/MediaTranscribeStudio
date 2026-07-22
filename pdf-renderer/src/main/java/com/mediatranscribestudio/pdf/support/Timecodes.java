package com.mediatranscribestudio.pdf.support;

import java.util.Locale;

public final class Timecodes {
    private Timecodes() {
    }

    public static String milliseconds(long value) {
        long hours = value / 3_600_000;
        long minutes = (value % 3_600_000) / 60_000;
        long seconds = (value % 60_000) / 1_000;
        long millis = value % 1_000;
        return String.format(Locale.ROOT, "%02d:%02d:%02d.%03d", hours, minutes, seconds, millis);
    }

    public static String duration(long value) {
        long hours = value / 3_600_000;
        long minutes = (value % 3_600_000) / 60_000;
        long seconds = (value % 60_000) / 1_000;
        return String.format(Locale.ROOT, "%02d:%02d:%02d", hours, minutes, seconds);
    }
}
