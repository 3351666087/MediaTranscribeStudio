package com.mediatranscribestudio.pdf;

import com.mediatranscribestudio.pdf.contract.RenderRequest;
import com.mediatranscribestudio.pdf.contract.ReportDocument;
import com.mediatranscribestudio.pdf.qa.AestheticFacetId;
import com.mediatranscribestudio.pdf.support.JsonSupport;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class TestFixtures {
    private TestFixtures() {
    }

    public static ReportDocument document(int speakerCount) {
        if (speakerCount < 1) {
            throw new IllegalArgumentException("speakerCount must be positive");
        }
        ReportDocument document = new ReportDocument();
        document.schemaVersion = "1.0.0";
        document.documentId = "synthetic-pdf-" + speakerCount + "-speakers";
        document.generatedAt = "2026-07-21T00:00:00Z";
        document.title = "星云计划中文逐字稿 - " + speakerCount + " 位说话人";
        document.language = "zh-CN";
        document.source = new ReportDocument.Source();
        document.source.fileName = "synthetic-" + speakerCount + ".wav";
        document.source.mediaType = "audio/wav";

        document.speakerPolicy = new ReportDocument.SpeakerPolicy();
        document.speakerPolicy.mode = "auto";
        document.speakerPolicy.resolvedCount = speakerCount;
        document.speakerPolicy.requireExactSet = true;
        document.speakerPolicy.unknownSpeakerAllowed = false;
        document.speakerPolicy.speakerChangeRequiresEvidence = true;
        document.speakerPolicy.minimumCount = 1;
        document.speakerPolicy.maximumCount = Math.addExact(speakerCount, 3);
        document.speakerPolicy.speakerIds = new ArrayList<>();
        document.speakerPolicy.detection = new ReportDocument.SpeakerCountDetection();
        document.speakerPolicy.detection.provider = "synthetic-dynamic-n";
        document.speakerPolicy.detection.estimatedCount = speakerCount;
        document.speakerPolicy.detection.confidence = 0.96;
        document.speakerPolicy.detection.candidates = new ArrayList<>();

        document.speakers = new ArrayList<>();
        for (int index = 1; index <= speakerCount; index++) {
            String id = "speaker-" + index;
            document.speakerPolicy.speakerIds.add(id);
            ReportDocument.Speaker speaker = new ReportDocument.Speaker();
            speaker.id = id;
            speaker.order = index;
            speaker.displayName = "测试说话人" + index;
            speaker.shortLabel = "角色" + index;
            speaker.colorToken = "speaker." + index;
            speaker.role = index == 1 ? "主持人" : "参与者";
            speaker.aliases = List.of("S" + index);
            document.speakers.add(speaker);
        }

        int segmentCount = Math.max(6, Math.multiplyExact(speakerCount, 2));
        document.segments = new ArrayList<>();
        for (int index = 0; index < segmentCount; index++) {
            int speakerIndex = index % speakerCount;
            ReportDocument.Segment segment = new ReportDocument.Segment();
            segment.id = String.format("segment-%03d", index + 1);
            segment.startMs = index * 15_000L;
            segment.endMs = segment.startMs + 12_600L;
            segment.speakerId = "speaker-" + (speakerIndex + 1);
            segment.rawText = "这是第" + (index + 1) + "段合成中文原文，用于验证离线 PDF 渲染。";
            segment.normalizedText = segment.rawText;
            segment.displayText = segment.rawText;
            segment.language = "zh-CN";
            segment.confidence = 0.94;
            segment.reviewStatus = "locked";
            segment.sourceSegmentIds = List.of("source-" + (index + 1));
            segment.evidence = evidence(segment.speakerId, speakerCount);
            segment.revisions = new ArrayList<>();
            document.segments.add(segment);
        }
        document.source.durationMs = document.segments.get(document.segments.size() - 1).endMs + 5_000;
        document.provenance = new ReportDocument.Provenance();
        document.provenance.pipelineVersion = "synthetic-2026-07-21";
        document.provenance.offline = true;
        document.provenance.models = new ArrayList<>();
        ReportDocument.Model asr = new ReportDocument.Model();
        asr.role = "asr";
        asr.name = "Qwen3-ASR-1.7B";
        document.provenance.models.add(asr);
        ReportDocument.Model speaker = new ReportDocument.Model();
        speaker.role = "speaker";
        speaker.name = "CAM++";
        document.provenance.models.add(speaker);
        document.privacy = new ReportDocument.Privacy();
        document.privacy.containsRealMeetingText = false;
        document.privacy.exportApproved = true;
        return document;
    }

    public static RenderRequest request(Path output, int speakerCount) throws IOException {
        Files.createDirectories(output.resolve("input"));
        Path reportPath = output.resolve("input/report-document.json");
        JsonSupport.write(reportPath, document(speakerCount));
        RenderRequest request = new RenderRequest();
        request.schemaVersion = "1.0.0";
        request.requestId = "request-" + speakerCount;
        request.jobId = "job-" + speakerCount;
        request.reportDocumentPath = reportPath.toAbsolutePath().toString();
        request.outputDirectory = output.toAbsolutePath().toString();
        request.renderer = new RenderRequest.Renderer();
        request.renderer.provider = "java-openhtmltopdf";
        request.renderer.offline = true;
        request.renderer.templateId = "mts-cute-transcript-v1";
        request.renderer.page = new RenderRequest.Page();
        request.renderer.page.size = "A4";
        request.renderer.page.marginMm = 14.0;
        request.renderer.fontPolicy = new RenderRequest.FontPolicy();
        request.renderer.fontPolicy.requireEmbeddedCjk = true;
        request.renderer.fontPolicy.allowSystemFallback = false;
        request.renderer.fontPolicy.preferredFont = "LXGW WenKai";
        request.qualityPolicy = new RenderRequest.QualityPolicy();
        request.qualityPolicy.requireHardGates = true;
        request.qualityPolicy.minimumScore = 85.0;
        request.qualityPolicy.maxRounds = 5;
        request.qualityPolicy.captureDpi = 96;
        request.qualityPolicy.facetIds = AestheticFacetId.ids();
        return request;
    }

    private static ReportDocument.Evidence evidence(String assignment, int speakerCount) {
        ReportDocument.Evidence evidence = new ReportDocument.Evidence();
        evidence.asr = new ReportDocument.AsrEvidence();
        evidence.asr.provider = "local";
        evidence.asr.model = "Qwen3-ASR-1.7B";
        evidence.asr.confidence = 0.94;
        evidence.boundary = new ReportDocument.BoundaryEvidence();
        evidence.boundary.provider = "FunASR";
        evidence.boundary.model = "timestamp-aligner";
        evidence.boundary.confidence = 0.95;
        evidence.boundary.overlapDetected = false;
        evidence.speaker = new ReportDocument.SpeakerEvidence();
        evidence.speaker.provider = "CAM++";
        evidence.speaker.model = "CAM++";
        evidence.speaker.assignment = assignment;
        evidence.speaker.locked = true;
        evidence.speaker.margin = 0.42;
        evidence.speaker.scores = new ArrayList<>();
        for (int index = 1; index <= speakerCount; index++) {
            ReportDocument.SpeakerScore score = new ReportDocument.SpeakerScore();
            score.speakerId = "speaker-" + index;
            score.score = score.speakerId.equals(assignment) ? 0.95 : 0.10;
            evidence.speaker.scores.add(score);
        }
        return evidence;
    }
}
