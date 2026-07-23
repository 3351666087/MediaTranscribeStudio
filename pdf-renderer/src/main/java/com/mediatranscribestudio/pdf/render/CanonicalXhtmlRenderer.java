package com.mediatranscribestudio.pdf.render;

import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.i18n.ReportCopy;
import com.mediatranscribestudio.pdf.support.Timecodes;
import com.mediatranscribestudio.pdf.validation.ReportDocumentValidator;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

public final class CanonicalXhtmlRenderer {
    private static final int LEGEND_ROWS_PER_PAGE = 12;

    private final String cssTemplate = loadCss();

    public String render(ReportDocument document, RenderProfile profile) {
        ReportDocumentValidator.validate(document);
        if (profile == null) {
            throw new IllegalArgumentException("render profile is required");
        }
        ReportCopy copy = ReportCopy.forDocument(document);
        String css = cssTemplate
                .replace("{{PAGE_MARGIN_MM}}", number(profile.pageMarginMm()))
                .replace("{{BODY_FONT_PT}}", number(profile.bodyFontPt()))
                .replace("{{BODY_LINE_HEIGHT}}", number(profile.bodyLineHeight()))
                .replace("{{TURN_PADDING_MM}}", number(profile.turnPaddingMm()))
                .replace("{{LEGEND_FONT_PT}}", number(profile.legendFontPt()))
                .replace("{{FOOTER_TITLE}}", cssString(title(document, copy)));
        String documentLanguage = ReportDocumentValidator.canonicalizeLanguageTag(document.language);

        Map<String, ReportDocument.Speaker> speakers = new LinkedHashMap<>();
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (ReportDocument.Speaker speaker : document.speakers) {
            speakers.put(speaker.id, speaker);
            counts.put(speaker.id, 0);
        }
        for (ReportDocument.Segment segment : document.segments) {
            counts.computeIfPresent(segment.speakerId, (ignored, value) -> value + 1);
        }

        int legendColumns = document.speakers.size() <= 2 ? 2 : 3;
        long estimatedCapacity = Math.max(
                64L * 1024L,
                Math.addExact(
                        Math.multiplyExact((long) document.segments.size(), 512L),
                        Math.multiplyExact((long) document.speakers.size(), 384L)
                )
        );
        StringBuilder html = new StringBuilder((int) Math.min(estimatedCapacity, 16L * 1024L * 1024L));
        html.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
                .append("<!DOCTYPE html>\n")
                .append("<html xmlns=\"http://www.w3.org/1999/xhtml\" lang=\"")
                .append(copy.locale()).append("\" xml:lang=\"")
                .append(copy.locale()).append("\" dir=\"")
                .append(copy.direction()).append("\">\n")
                .append("<head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" />")
                .append("<title>").append(escape(title(document, copy))).append("</title>")
                .append("<style type=\"text/css\">").append(css).append("</style></head><body>\n")
                .append("<div class=\"running-header\">").append(escape(title(document, copy)))
                .append(" · ").append(escape(copy.runningDescriptor())).append("</div>\n")
                .append("<section class=\"cover\">\n")
                .append("<p class=\"eyebrow\">MEDIATRANSCRIBESTUDIO · VERIFIED TRANSCRIPT</p>")
                .append("<h1>").append(escape(title(document, copy))).append("</h1>")
                .append("<p class=\"subtitle\">").append(escape(copy.subtitle())).append("</p>")
                .append("<table class=\"meta-grid\"><tbody><tr>")
                .append(meta(copy.sourceFile(), document.source.fileName))
                .append(meta(copy.language(), documentLanguage))
                .append("</tr><tr>")
                .append(meta(copy.duration(), Timecodes.duration(document.source.durationMs)))
                .append(meta(copy.transcriptSegments(),
                        copy.segmentCount(document.segments.size())))
                .append("</tr><tr>")
                .append(meta(copy.speakers(), copy.speakerCount(document.speakers.size())))
                .append(meta(copy.speakerStrategy(),
                        copy.policyLabel(document.speakerPolicy.mode)))
                .append("</tr></tbody></table>")
                .append("<div class=\"legend-heading\"><h2>")
                .append(escape(copy.speakerLegend())).append("</h2><span>")
                .append(escape(copy.resolvedRoles(document.speakers.size())))
                .append("</span></div>");

        int legendPageCapacity = Math.multiplyExact(legendColumns, LEGEND_ROWS_PER_PAGE);
        for (int chunkStart = 0; chunkStart < document.speakers.size();
             chunkStart += legendPageCapacity) {
            int chunkEnd = Math.min(document.speakers.size(), chunkStart + legendPageCapacity);
            if (chunkStart > 0) {
                html.append("<div class=\"legend-page-break\"></div>")
                        .append("<div class=\"legend-heading legend-heading-continuation\">")
                        .append("<h2>").append(escape(copy.speakerLegendContinuation()))
                        .append("</h2><span>")
                        .append(chunkStart + 1).append("–").append(chunkEnd)
                        .append(" / ").append(document.speakers.size()).append("</span></div>");
            }
            html.append("<table class=\"legend-table\"><tbody>");
            for (int index = chunkStart; index < chunkEnd; index++) {
                int chunkIndex = index - chunkStart;
                if (chunkIndex % legendColumns == 0) {
                    html.append("<tr>");
                }
                ReportDocument.Speaker speaker = document.speakers.get(index);
                SpeakerPalette.SpeakerStyle style =
                        new SpeakerPalette().style(speaker.order, speaker.colorToken);
                html.append("<td><div class=\"legend-card\" style=\"border-left-color:")
                        .append(style.accent()).append(";border-right-color:")
                        .append(style.accent()).append(";background:").append(style.background())
                        .append(";color:").append(style.accent()).append("\">")
                        .append("<span class=\"legend-index\" style=\"background:")
                        .append(style.accent()).append(";color:").append(style.foreground()).append("\">")
                        .append(String.format("%02d", speaker.order)).append("</span>")
                        .append("<span class=\"legend-copy\"><strong>").append(escape(speaker.displayName))
                        .append("</strong><small>").append(escape(speaker.id)).append(" · ")
                        .append(escape(copy.legendSegmentCount(counts.get(speaker.id))))
                        .append("</small></span></div></td>");
                if (chunkIndex % legendColumns == legendColumns - 1) {
                    html.append("</tr>");
                }
            }
            int remainder = (chunkEnd - chunkStart) % legendColumns;
            if (remainder != 0) {
                for (int index = remainder; index < legendColumns; index++) {
                    html.append("<td></td>");
                }
                html.append("</tr>");
            }
            html.append("</tbody></table>");
        }

        html.append("<p class=\"policy-note\">").append(escape(copy.policyNote())).append("</p>")
                .append("</section><main>")
                .append("<section class=\"transcript-intro\"><span>TRANSCRIPT</span>")
                .append("<h2>").append(escape(copy.transcriptHeading())).append("</h2><p>")
                .append(escape(copy.transcriptDescription())).append("</p></section>");

        long bucket = Long.MIN_VALUE;
        for (int index = 0; index < document.segments.size(); index++) {
            ReportDocument.Segment segment = document.segments.get(index);
            long nextBucket = segment.startMs / 300_000;
            if (nextBucket != bucket) {
                bucket = nextBucket;
                long from = bucket * 300_000;
                long to = Math.min(document.source.durationMs, from + 300_000);
                html.append("<section class=\"time-section\"><span>")
                        .append(escape(copy.timeSection())).append("</span><h2>")
                        .append(Timecodes.duration(from)).append(" - ").append(Timecodes.duration(to))
                        .append("</h2></section>");
            }
            ReportDocument.Speaker speaker = speakers.get(segment.speakerId);
            String segmentLanguage = segment.language == null
                    ? documentLanguage
                    : ReportDocumentValidator.canonicalizeLanguageTag(segment.language);
            String segmentDirection = direction(segmentLanguage);
            SpeakerPalette.SpeakerStyle style =
                    new SpeakerPalette().style(speaker.order, speaker.colorToken);
            boolean longTurn = segment.displayText.codePointCount(0, segment.displayText.length()) > 260;
            html.append("<article class=\"turn").append(longTurn ? " long-turn" : "")
                    .append("\"><table class=\"turn-table\"><tbody><tr>")
                    .append("<td class=\"speaker-cell\"><span class=\"speaker-badge\" style=\"background:")
                    .append(style.accent()).append(";color:").append(style.foreground()).append("\">")
                    .append(String.format("%02d", speaker.order)).append("</span>")
                    .append("<strong>").append(escape(speaker.shortLabel)).append("</strong>")
                    .append("<small>").append(escape(speaker.id)).append("</small></td>")
                    .append("<td class=\"time-cell\"><span>")
                    .append(Timecodes.milliseconds(segment.startMs)).append("</span><span>")
                    .append(Timecodes.milliseconds(segment.endMs)).append("</span><small>")
                    .append(String.format("#%03d · %s", index + 1, segment.id)).append("</small></td>")
                    .append("<td class=\"text-cell\"><p lang=\"").append(segmentLanguage)
                    .append("\" xml:lang=\"").append(segmentLanguage)
                    .append("\" dir=\"").append(segmentDirection).append("\">")
                    .append(escape(segment.displayText)).append("</p>")
                    .append("<div class=\"turn-meta\">ASR ")
                    .append(String.format("%.1f%%", segment.confidence * 100.0));
            if (Boolean.TRUE.equals(segment.evidence.boundary.overlapDetected)
                    || segment.overlapGroupId != null) {
                html.append(" · ").append(escape(copy.overlapSpeech()));
            }
            if (Boolean.TRUE.equals(segment.evidence.speaker.locked)) {
                html.append(" · ").append(escape(copy.speakerLocked()));
            }
            html.append("</div></td></tr></tbody></table></article>");
        }
        html.append("<p class=\"end-mark\">").append(escape(copy.endMark())).append("</p>")
                .append("</main></body></html>");
        return html.toString();
    }

    private static String title(ReportDocument document, ReportCopy copy) {
        return document.title == null || document.title.isBlank()
                ? copy.defaultTitle()
                : document.title;
    }

    private static String direction(String language) {
        return ReportDocumentValidator.isRightToLeftLanguage(language) ? "rtl" : "ltr";
    }

    private static String meta(String label, Object value) {
        return "<td><small>" + escape(label) + "</small><strong>"
                + escape(String.valueOf(value)) + "</strong></td>";
    }

    private static String escape(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&#39;");
    }

    private static String cssString(String value) {
        StringBuilder escaped = new StringBuilder(value.length() + 16);
        value.codePoints().forEach(codePoint -> {
            switch (codePoint) {
                case '\\' -> escaped.append("\\5c ");
                case '"' -> escaped.append("\\22 ");
                case '\n' -> escaped.append("\\a ");
                case '\r' -> escaped.append("\\d ");
                case '\f' -> escaped.append("\\c ");
                case '<' -> escaped.append("\\3c ");
                case '>' -> escaped.append("\\3e ");
                case '&' -> escaped.append("\\26 ");
                default -> escaped.appendCodePoint(codePoint);
            }
        });
        return escaped.toString();
    }

    private static String number(double value) {
        return Double.toString(value);
    }

    private static String loadCss() {
        try (InputStream input = CanonicalXhtmlRenderer.class.getResourceAsStream("/templates/report.css")) {
            if (input == null) {
                throw new IllegalStateException("missing /templates/report.css");
            }
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException exception) {
            throw new IllegalStateException("unable to load report.css", exception);
        }
    }
}
