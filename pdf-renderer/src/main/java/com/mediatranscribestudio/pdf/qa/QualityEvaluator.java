package com.mediatranscribestudio.pdf.qa;

import com.mediatranscribestudio.pdf.contract.PdfInspection;
import com.mediatranscribestudio.pdf.contract.QualityReport;
import com.mediatranscribestudio.pdf.contract.RepairQueue;
import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.render.PageImageRenderer;
import com.mediatranscribestudio.pdf.render.RenderProfile;
import com.mediatranscribestudio.pdf.render.SpeakerPalette;
import com.mediatranscribestudio.pdf.support.Hashing;
import com.mediatranscribestudio.pdf.support.OfflineValidationResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

public final class QualityEvaluator {
    public QualityReport evaluate(
            Path outputRoot,
            int round,
            int maxRounds,
            double minimumScore,
            RenderProfile profile,
            ReportDocument source,
            String xhtml,
            Path xhtmlPath,
            Path pdfPath,
            Path inspectionPath,
            Path extractedTextPath,
            Path contactSheet,
            PdfInspection inspection,
            PageImageRenderer.Result pageEvidence,
            String baselineHash,
            String currentHash,
            OfflineValidationResult offlineValidation
    ) throws IOException {
        Objects.requireNonNull(offlineValidation, "offlineValidation");
        boolean offlineAssets = offlineValidation.verified();
        QualityReport report = new QualityReport();
        report.documentId = source.documentId;
        report.round = round;
        report.minimumScore = Math.max(85.0, minimumScore);

        addEvidence(report, outputRoot, "evidence-pdf", "pdf", pdfPath, null);
        addEvidence(report, outputRoot, "evidence-xhtml", "test-output", xhtmlPath, null);
        addEvidence(report, outputRoot, "evidence-inspection", "font-inspection",
                inspectionPath, null);
        addEvidence(report, outputRoot, "evidence-text", "text-extraction",
                extractedTextPath, null);
        addEvidence(report, outputRoot, "evidence-contact-sheet", "contact-sheet",
                contactSheet, null);
        for (PageImageRenderer.PageEvidence page : pageEvidence.pages()) {
            addEvidence(report, outputRoot, String.format("evidence-page-%03d", page.pageNumber()),
                    "page-image", page.path(), page.pageNumber());
        }

        boolean pageEvidenceComplete = inspection.pageCount > 0
                && pageEvidence.pages().size() == inspection.pageCount
                && pageEvidence.pages().stream().allMatch(page -> Files.isRegularFile(page.path()))
                && Files.isRegularFile(contactSheet);
        gate(report, HardGateId.PDF_OPENABLE, inspection.openable,
                inspection.openable ? "PDFBox 成功打开 PDF" : String.valueOf(inspection.failure),
                "evidence-pdf");
        gate(report, HardGateId.PDF_PAGE_COUNT, inspection.pageCount > 0,
                "页数=" + inspection.pageCount, "evidence-pdf");
        gate(report, HardGateId.PDF_PAGE_SIZE, inspection.allPagesA4,
                "全部页面为 A4=" + inspection.allPagesA4, "evidence-inspection");
        gate(report, HardGateId.PDF_TRANSCRIPT_TEXT_INTEGRITY,
                inspection.transcriptTextIntegrity && inspection.searchableText
                        && inspection.noReplacementCharacters,
                "缺失段=" + inspection.missingSegmentIds
                        + "，可搜索=" + inspection.searchableText
                        + "，替换字符=" + !inspection.noReplacementCharacters,
                "evidence-text");
        gate(report, HardGateId.PDF_SEGMENT_COUNT, inspection.segmentCountIntegrity,
                "缺失段号=" + inspection.missingSegmentIds
                        + "，重复段号=" + inspection.duplicateSegmentIds,
                "evidence-text");
        gate(report, HardGateId.PDF_TIMESTAMP_INTEGRITY, inspection.timestampIntegrity,
                "时间戳差异=" + inspection.missingTimestamps, "evidence-text");
        gate(report, HardGateId.PDF_SPEAKER_SET_INTEGRITY, inspection.speakerSetIntegrity,
                "缺失说话人=" + inspection.missingSpeakerIds
                        + "，解析人数=" + source.speakers.size(),
                "evidence-text");
        gate(report, HardGateId.PDF_FONT_EMBEDDED, inspection.allFontsEmbedded,
                "字体数量=" + inspection.fonts.size() + "，全部嵌入=" + inspection.allFontsEmbedded,
                "evidence-inspection");
        gate(report, HardGateId.PDF_NO_BLANK_PAGES, pageEvidence.blankPageCount() == 0,
                "空白页=" + pageEvidence.blankPageCount(), "evidence-contact-sheet");
        gate(report, HardGateId.PDF_NO_CONTENT_OVERFLOW,
                pageEvidence.edgeTouchPageCount() == 0,
                "触碰页面边缘的页面=" + pageEvidence.edgeTouchPageCount(),
                "evidence-contact-sheet");
        gate(report, HardGateId.PDF_OFFLINE_ASSETS, offlineAssets,
                "内置 Design Pack 离线契约=" + offlineValidation.policyVersion()
                        + "，XHTML-SHA256=" + offlineValidation.xhtmlSha256()
                        + "，验证=" + offlineAssets,
                "evidence-xhtml");
        gate(report, HardGateId.PDF_PAGE_EVIDENCE, pageEvidenceComplete,
                "PNG=" + pageEvidence.pages().size() + "，PDF页=" + inspection.pageCount
                        + "，联系表=" + Files.isRegularFile(contactSheet),
                "evidence-contact-sheet");
        gate(report, HardGateId.PDF_IMMUTABLE_CONTENT_HASH, baselineHash.equals(currentHash),
                "基线=" + baselineHash + "，当前=" + currentHash, "evidence-inspection");

        double minimumContrast = source.speakers.stream().mapToDouble(speaker -> {
            SpeakerPalette.SpeakerStyle style =
                    new SpeakerPalette().style(speaker.order, speaker.colorToken);
            return Math.min(
                    SpeakerPalette.contrast(style.accent(), style.background()),
                    SpeakerPalette.contrast(style.foreground(), style.accent())
            );
        }).min().orElse(0);
        boolean hierarchy = List.of(
                "class=\"cover\"", "class=\"meta-grid\"", "class=\"legend-table\"",
                "class=\"transcript-intro\"", "class=\"time-section\"",
                "class=\"turn\"", "class=\"speaker-badge\""
        ).stream().allMatch(xhtml::contains);
        boolean healthyDensity = pageEvidence.minimumInkRatio() >= 0.001
                && pageEvidence.maximumInkRatio() <= 0.55;

        facet(report, AestheticFacetId.COHERENCE,
                score(inspection.allPagesA4 && offlineAssets, 97, 45),
                "A4、离线资源与统一设计令牌保持一致", "evidence-contact-sheet");
        facet(report, AestheticFacetId.DISTINCTION,
                score(inspection.speakerSetIntegrity && minimumContrast >= 4.5, 98, 55),
                String.format(Locale.ROOT, "动态 N 角色标记最小对比度 %.2f", minimumContrast),
                "evidence-contact-sheet");
        facet(report, AestheticFacetId.REFINEMENT,
                score(pageEvidence.blankPageCount() == 0 && pageEvidence.edgeTouchPageCount() == 0,
                        96, 50),
                "无空白页、无边缘触碰", "evidence-contact-sheet");
        facet(report, AestheticFacetId.PROPORTION,
                score(inspection.allPagesA4 && pageEvidence.edgeTouchPageCount() == 0, 95, 50),
                "A4 页面、边距与段落比例稳定", "evidence-contact-sheet");
        facet(report, AestheticFacetId.HIERARCHY,
                score(hierarchy, 97, 45),
                "封面、元数据、图例、时间分区与正文层级齐全", "evidence-xhtml");
        facet(report, AestheticFacetId.TYPOGRAPHY,
                score(inspection.allFontsEmbedded && inspection.searchableText
                        && inspection.noReplacementCharacters, 100, 0),
                "离线 CJK 字体嵌入且文本可搜索", "evidence-inspection");
        facet(report, AestheticFacetId.COLOR_RELATIONSHIPS,
                minimumContrast >= 4.5 ? 97 : minimumContrast >= 3 ? 84 : 40,
                String.format(Locale.ROOT, "最小文本对比度 %.2f", minimumContrast),
                "evidence-contact-sheet");
        facet(report, AestheticFacetId.RHYTHM,
                score(pageEvidence.blankPageCount() == 0, 96, 50),
                "页面与段落节奏连续", "evidence-contact-sheet");
        facet(report, AestheticFacetId.DENSITY,
                score(healthyDensity, 95, 62),
                String.format(Locale.ROOT, "墨迹比例 %.4f..%.4f",
                        pageEvidence.minimumInkRatio(), pageEvidence.maximumInkRatio()),
                "evidence-contact-sheet");
        facet(report, AestheticFacetId.RESTRAINT,
                score(offlineAssets && !xhtml.toLowerCase(Locale.ROOT).contains("<script"), 100, 0),
                "无脚本、无远程资产、无非必要装饰", "evidence-xhtml");
        facet(report, AestheticFacetId.REAL_CONTENT_STRESS,
                score(inspection.transcriptTextIntegrity && inspection.segmentCountIntegrity
                        && inspection.timestampIntegrity && inspection.speakerSetIntegrity, 100, 0),
                "正文、段号、时间戳和任意 N 说话人全部存活", "evidence-text");
        facet(report, AestheticFacetId.FONT_FAILURE,
                score(inspection.allFontsEmbedded && inspection.noReplacementCharacters, 100, 0),
                "字体故障哨兵通过", "evidence-inspection");
        facet(report, AestheticFacetId.IMAGE_FAILURE,
                score(pageEvidenceComplete && pageEvidence.blankPageCount() == 0, 100, 0),
                "逐页 PNG 与联系表故障哨兵通过", "evidence-contact-sheet");
        facet(report, AestheticFacetId.SCRIPT_FAILURE,
                score(offlineAssets, 100, 0),
                "脚本与远程资源故障哨兵通过", "evidence-xhtml");

        requireCanonicalChecks(report);
        report.score = round(report.facets.stream()
                .mapToDouble(facet -> facet.score * facet.weight)
                .sum(), 2);
        report.hardGatesPassed = report.hardGates.stream()
                .allMatch(gate -> "passed".equals(gate.status));
        buildRepairs(report);
        boolean facetsPassed = report.facets.stream().allMatch(facet -> "passed".equals(facet.status));
        if (report.hardGatesPassed && facetsPassed && report.score >= report.minimumScore) {
            report.status = "passed";
        } else if (round < maxRounds && report.repairQueue.stream()
                .anyMatch(repair -> !"blocked".equals(repair.status))) {
            report.status = "repair-required";
        } else {
            report.status = "blocked";
        }
        return report;
    }

    public static void requireCanonicalChecks(QualityReport report) {
        if (report == null) {
            throw new IllegalArgumentException("quality report is required");
        }
        List<String> actualGates = report.hardGates == null
                ? List.of()
                : report.hardGates.stream()
                .map(gate -> gate == null ? null : gate.id)
                .toList();
        if (!HardGateId.ids().equals(actualGates)) {
            throw new IllegalStateException(
                    "hard gates must exactly match the canonical unique ordered set; expected="
                            + HardGateId.ids() + ", actual=" + actualGates);
        }

        List<String> actualFacets = report.facets == null
                ? List.of()
                : report.facets.stream()
                .map(facet -> facet == null ? null : facet.id)
                .toList();
        if (!AestheticFacetId.ids().equals(actualFacets)) {
            throw new IllegalStateException(
                    "aesthetic facets must exactly match the canonical unique ordered set; expected="
                            + AestheticFacetId.ids() + ", actual=" + actualFacets);
        }
        List<Double> expectedWeights = new ArrayList<>();
        for (AestheticFacetId id : AestheticFacetId.values()) {
            expectedWeights.add(id.weight());
        }
        List<Double> actualWeights = report.facets.stream()
                .map(facet -> facet.weight)
                .toList();
        if (!Objects.equals(expectedWeights, actualWeights)) {
            throw new IllegalStateException(
                    "aesthetic facet weights must exactly match the canonical order; expected="
                            + expectedWeights + ", actual=" + actualWeights);
        }
    }

    public RepairQueue standaloneQueue(QualityReport report) {
        RepairQueue queue = new RepairQueue();
        queue.documentId = report.documentId;
        queue.round = report.round;
        queue.repairs.addAll(report.repairQueue);
        queue.status = report.repairQueue.isEmpty() ? "empty"
                : report.repairQueue.stream().anyMatch(item -> !"blocked".equals(item.status))
                ? "actionable" : "blocked";
        return queue;
    }

    private static void buildRepairs(QualityReport report) {
        int priority = 0;
        for (QualityReport.HardGate gate : report.hardGates) {
            if ("passed".equals(gate.status)) {
                continue;
            }
            boolean safe = switch (gate.id) {
                case "PDF-PAGE-SIZE", "PDF-NO-BLANK-PAGES",
                        "PDF-NO-CONTENT-OVERFLOW", "PDF-PAGE-EVIDENCE" -> true;
                default -> false;
            };
            report.repairQueue.add(repair("repair-" + (++priority), priority,
                    safe ? "high" : "critical", gate.id,
                    safe ? scope(gate.id) : "template",
                    safe ? "仅调整确定性版式 profile 后完整重渲染；正文哈希必须保持不变。"
                            : "停止自动修复；该问题涉及内容、字体、离线策略或 PDF 完整性，必须人工处理。",
                    safe ? "pending" : "blocked"));
        }
        for (QualityReport.Facet facet : report.facets) {
            if ("passed".equals(facet.status)) {
                continue;
            }
            report.repairQueue.add(repair("repair-" + (++priority), priority,
                    "medium", facet.id, "css",
                    "仅调整 CSS、页边距、字体尺寸、行距和分页参数后完整重渲染。",
                    "pending"));
        }
    }

    private static String scope(String id) {
        return switch (id) {
            case "PDF-PAGE-SIZE", "PDF-NO-CONTENT-OVERFLOW" -> "page-margin";
            case "PDF-NO-BLANK-PAGES" -> "pagination";
            default -> "image-size";
        };
    }

    private static QualityReport.Repair repair(
            String id, int priority, String severity, String checkId,
            String scope, String action, String status
    ) {
        QualityReport.Repair repair = new QualityReport.Repair();
        repair.id = id;
        repair.priority = priority;
        repair.severity = severity;
        repair.checkId = checkId;
        repair.safeScope = scope;
        repair.action = action;
        repair.status = status;
        return repair;
    }

    private static void gate(
            QualityReport report,
            HardGateId id,
            boolean passed,
            String message,
            String evidenceId
    ) {
        QualityReport.HardGate gate = new QualityReport.HardGate();
        gate.id = id.id();
        gate.status = passed ? "passed" : "failed";
        gate.message = message;
        gate.evidenceIds.add(evidenceId);
        report.hardGates.add(gate);
    }

    private static void facet(
            QualityReport report,
            AestheticFacetId id,
            double score,
            String message,
            String evidenceId
    ) {
        QualityReport.Facet facet = new QualityReport.Facet();
        facet.id = id.id();
        facet.status = score >= 85 ? "passed" : "failed";
        facet.score = score;
        facet.weight = id.weight();
        facet.message = message;
        facet.evidenceIds.add(evidenceId);
        report.facets.add(facet);
    }

    private static void addEvidence(
            QualityReport report,
            Path outputRoot,
            String id,
            String type,
            Path path,
            Integer pageNumber
    ) throws IOException {
        QualityReport.Evidence evidence = new QualityReport.Evidence();
        evidence.id = id;
        evidence.type = type;
        evidence.relativePath = relative(outputRoot, path);
        evidence.sha256 = Hashing.sha256(path);
        evidence.verified = true;
        evidence.pageNumber = pageNumber;
        report.evidence.add(evidence);
    }

    private static String relative(Path root, Path path) {
        return root.toAbsolutePath().normalize().relativize(path.toAbsolutePath().normalize())
                .toString().replace('\\', '/');
    }

    private static double score(boolean condition, double pass, double fail) {
        return condition ? pass : fail;
    }

    private static double round(double value, int digits) {
        double factor = Math.pow(10, digits);
        return Math.round(value * factor) / factor;
    }
}
