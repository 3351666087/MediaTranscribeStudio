package com.mediatranscribestudio.pdf.i18n;

import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.validation.ReportDocumentValidator;

import java.util.Locale;

/**
 * Locale-aware interface copy for the deterministic PDF report.
 *
 * <p>The transcript language and the report interface locale are intentionally
 * independent. Chinese keeps the established visual copy; every other locale
 * receives a professional English fallback until a native copy pack is added.
 * Transcript text is never translated by this class.</p>
 */
public final class ReportCopy {
    private static final String CHINESE_LOCALE = "zh-CN";
    private static final String ENGLISH_LOCALE = "en-US";

    private final boolean chinese;

    private ReportCopy(boolean chinese) {
        this.chinese = chinese;
    }

    public static ReportCopy forDocument(ReportDocument document) {
        if (document == null) {
            throw new IllegalArgumentException("report document is required");
        }
        String requestedLocale = document.reportLocale == null
                ? document.language
                : document.reportLocale;
        String canonical = ReportDocumentValidator.canonicalizeLanguageTag(requestedLocale);
        String primary = canonical.split("-", 2)[0].toLowerCase(Locale.ROOT);
        return new ReportCopy("zh".equals(primary));
    }

    public String locale() {
        return chinese ? CHINESE_LOCALE : ENGLISH_LOCALE;
    }

    public String direction() {
        return "ltr";
    }

    public String defaultTitle() {
        return chinese ? "中文原文逐字稿" : "Original-Language Transcript";
    }

    public String runningDescriptor() {
        return chinese
                ? "中文逐字稿 · 说话人标注"
                : "Original-Language Transcript · Speaker Attribution";
    }

    public String subtitle() {
        return chinese
                ? "保留原始文本、时间边界、说话人证据与修订审计；本报告不翻译、不总结。"
                : "Preserves original text, time boundaries, speaker evidence, and revision audit. "
                + "This report does not translate or summarize.";
    }

    public String sourceFile() {
        return chinese ? "源文件" : "Source file";
    }

    public String language() {
        return chinese ? "语言" : "Language";
    }

    public String duration() {
        return chinese ? "总时长" : "Duration";
    }

    public String transcriptSegments() {
        return chinese ? "逐字稿段落" : "Transcript segments";
    }

    public String speakers() {
        return chinese ? "说话人数" : "Speakers";
    }

    public String speakerStrategy() {
        return chinese ? "人数策略" : "Speaker strategy";
    }

    public String segmentCount(int count) {
        return chinese ? count + " 段" : count + (count == 1 ? " segment" : " segments");
    }

    public String speakerCount(int count) {
        return chinese ? count + " 人" : count + (count == 1 ? " speaker" : " speakers");
    }

    public String speakerLegend() {
        return chinese ? "说话人图例" : "Speaker legend";
    }

    public String speakerLegendContinuation() {
        return chinese ? "说话人图例（续）" : "Speaker legend (continued)";
    }

    public String resolvedRoles(int count) {
        return chinese
                ? count + " 位已解析角色"
                : count + (count == 1 ? " resolved role" : " resolved roles");
    }

    public String legendSegmentCount(int count) {
        return chinese ? count + " 段" : count + (count == 1 ? " segment" : " segments");
    }

    public String policyNote() {
        return chinese
                ? "质量检查和自动修复只能调整模板、CSS、字体、页边距与分页参数；"
                + "不得改写正文、角色、时间戳、重叠关系或审计记录。"
                : "Quality checks and automatic repairs may adjust only templates, CSS, fonts, "
                + "margins, and pagination. They must never rewrite transcript text, speakers, "
                + "timestamps, overlap relationships, or audit records.";
    }

    public String transcriptHeading() {
        return chinese ? "中文原文逐字稿" : "Original-Language Transcript";
    }

    public String transcriptDescription() {
        return chinese
                ? "时间戳精确到毫秒；每段显示稳定段号、说话人和声学置信度。"
                : "Timestamps are precise to milliseconds; every turn includes a stable segment "
                + "identifier, speaker attribution, and acoustic confidence.";
    }

    public String timeSection() {
        return chinese ? "时间分区" : "Time section";
    }

    public String overlapSpeech() {
        return chinese ? "重叠语音" : "Overlapping speech";
    }

    public String speakerLocked() {
        return chinese ? "角色已锁定" : "Speaker locked";
    }

    public String endMark() {
        return chinese
                ? "逐字稿结束 · 内容哈希由 PDF 质量系统验证"
                : "End of transcript · Content hash verified by the PDF quality system";
    }

    public String policyLabel(String mode) {
        return switch (mode) {
            case "manual" -> chinese ? "手动指定" : "Manual";
            case "hybrid" -> chinese ? "自动检测 + 手动约束" : "Automatic + manual bounds";
            default -> chinese ? "自动检测" : "Automatic";
        };
    }

    public String metadataSubject() {
        return chinese
                ? "原文逐字稿，包含动态任意 N 说话人和毫秒级时间戳"
                : "Original-language transcript with dynamic arbitrary-N speakers and "
                + "millisecond timestamps";
    }
}
