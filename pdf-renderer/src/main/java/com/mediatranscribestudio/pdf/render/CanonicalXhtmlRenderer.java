package com.mediatranscribestudio.pdf.render;

import com.mediatranscribestudio.pdf.contract.ReportDocument;
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
        String css = cssTemplate
                .replace("{{PAGE_MARGIN_MM}}", number(profile.pageMarginMm()))
                .replace("{{BODY_FONT_PT}}", number(profile.bodyFontPt()))
                .replace("{{BODY_LINE_HEIGHT}}", number(profile.bodyLineHeight()))
                .replace("{{TURN_PADDING_MM}}", number(profile.turnPaddingMm()))
                .replace("{{LEGEND_FONT_PT}}", number(profile.legendFontPt()))
                .replace("{{FOOTER_TITLE}}", cssString(title(document)));

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
                .append("<html xmlns=\"http://www.w3.org/1999/xhtml\" lang=\"zh-CN\" xml:lang=\"zh-CN\">\n")
                .append("<head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" />")
                .append("<title>").append(escape(title(document))).append("</title>")
                .append("<style type=\"text/css\">").append(css).append("</style></head><body>\n")
                .append("<div class=\"running-header\">").append(escape(title(document)))
                .append(" · 中文逐字稿 · 说话人标注</div>\n")
                .append("<section class=\"cover\">\n")
                .append("<p class=\"eyebrow\">MEDIATRANSCRIBESTUDIO · VERIFIED TRANSCRIPT</p>")
                .append("<h1>").append(escape(title(document))).append("</h1>")
                .append("<p class=\"subtitle\">保留原始文本、时间边界、说话人证据与修订审计；本报告不翻译、不总结。</p>")
                .append("<table class=\"meta-grid\"><tbody><tr>")
                .append(meta("源文件", document.source.fileName))
                .append(meta("语言", document.language))
                .append("</tr><tr>")
                .append(meta("总时长", Timecodes.duration(document.source.durationMs)))
                .append(meta("逐字稿段落", document.segments.size() + " 段"))
                .append("</tr><tr>")
                .append(meta("说话人数", document.speakers.size() + " 人"))
                .append(meta("人数策略", policyLabel(document.speakerPolicy.mode)))
                .append("</tr></tbody></table>")
                .append("<div class=\"legend-heading\"><h2>说话人图例</h2><span>")
                .append(document.speakers.size()).append(" 位已解析角色</span></div>");

        int legendPageCapacity = Math.multiplyExact(legendColumns, LEGEND_ROWS_PER_PAGE);
        for (int chunkStart = 0; chunkStart < document.speakers.size();
             chunkStart += legendPageCapacity) {
            int chunkEnd = Math.min(document.speakers.size(), chunkStart + legendPageCapacity);
            if (chunkStart > 0) {
                html.append("<div class=\"legend-page-break\"></div>")
                        .append("<div class=\"legend-heading legend-heading-continuation\">")
                        .append("<h2>说话人图例（续）</h2><span>")
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
                        .append(style.accent()).append(";background:").append(style.background())
                        .append(";color:").append(style.accent()).append("\">")
                        .append("<span class=\"legend-index\" style=\"background:")
                        .append(style.accent()).append(";color:").append(style.foreground()).append("\">")
                        .append(String.format("%02d", speaker.order)).append("</span>")
                        .append("<span class=\"legend-copy\"><strong>").append(escape(speaker.displayName))
                        .append("</strong><small>").append(escape(speaker.id)).append(" · ")
                        .append(counts.get(speaker.id)).append(" 段</small></span></div></td>");
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

        html.append("<p class=\"policy-note\">质量检查和自动修复只能调整模板、CSS、字体、页边距与分页参数；")
                .append("不得改写正文、角色、时间戳、重叠关系或审计记录。</p>")
                .append("</section><main>")
                .append("<section class=\"transcript-intro\"><span>TRANSCRIPT</span>")
                .append("<h2>中文原文逐字稿</h2><p>时间戳精确到毫秒；每段显示稳定段号、说话人和声学置信度。</p></section>");

        long bucket = Long.MIN_VALUE;
        for (int index = 0; index < document.segments.size(); index++) {
            ReportDocument.Segment segment = document.segments.get(index);
            long nextBucket = segment.startMs / 300_000;
            if (nextBucket != bucket) {
                bucket = nextBucket;
                long from = bucket * 300_000;
                long to = Math.min(document.source.durationMs, from + 300_000);
                html.append("<section class=\"time-section\"><span>时间分区</span><h2>")
                        .append(Timecodes.duration(from)).append(" - ").append(Timecodes.duration(to))
                        .append("</h2></section>");
            }
            ReportDocument.Speaker speaker = speakers.get(segment.speakerId);
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
                    .append("<td class=\"text-cell\"><p>").append(escape(segment.displayText)).append("</p>")
                    .append("<div class=\"turn-meta\">ASR ")
                    .append(String.format("%.1f%%", segment.confidence * 100.0));
            if (Boolean.TRUE.equals(segment.evidence.boundary.overlapDetected)
                    || segment.overlapGroupId != null) {
                html.append(" · 重叠语音");
            }
            if (Boolean.TRUE.equals(segment.evidence.speaker.locked)) {
                html.append(" · 角色已锁定");
            }
            html.append("</div></td></tr></tbody></table></article>");
        }
        html.append("<p class=\"end-mark\">逐字稿结束 · 内容哈希由 PDF 质量系统验证</p>")
                .append("</main></body></html>");
        return html.toString();
    }

    private static String title(ReportDocument document) {
        return document.title == null || document.title.isBlank() ? "中文逐字稿" : document.title;
    }

    private static String policyLabel(String mode) {
        return switch (mode) {
            case "manual" -> "手动指定";
            case "hybrid" -> "自动检测 + 手动约束";
            default -> "自动检测";
        };
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
