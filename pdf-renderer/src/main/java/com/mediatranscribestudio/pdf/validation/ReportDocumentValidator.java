package com.mediatranscribestudio.pdf.validation;

import com.fasterxml.jackson.databind.JsonNode;
import com.mediatranscribestudio.pdf.contract.ReportDocument;

import java.time.OffsetDateTime;
import java.time.format.DateTimeParseException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

public final class ReportDocumentValidator {
    private static final Set<String> LANGUAGES = Set.of("zh", "zh-CN", "zh-Hans");
    private static final Set<String> MODES = Set.of("auto", "manual", "hybrid");
    private static final Set<String> REVISION_TYPES =
            Set.of("text", "speaker", "boundary", "split", "merge");
    private static final Set<String> REVISION_SOURCES =
            Set.of("acoustic", "deterministic", "llm", "manual");
    private static final Set<String> FORBIDDEN_REVISION_TYPES =
            Set.of("translation", "summary", "analysis");

    private ReportDocumentValidator() {
    }

    public static void validate(ReportDocument document) {
        require(document != null, "report document is required");
        require("1.0.0".equals(document.schemaVersion), "schemaVersion must be 1.0.0");
        requireText(document.documentId, "documentId");
        try {
            OffsetDateTime.parse(document.generatedAt);
        } catch (DateTimeParseException | NullPointerException exception) {
            throw new IllegalArgumentException("generatedAt must be RFC 3339", exception);
        }
        require(LANGUAGES.contains(document.language), "language must be Chinese");
        require(document.source != null, "source is required");
        requireText(document.source.fileName, "source.fileName");
        requireText(document.source.mediaType, "source.mediaType");
        require(document.source.durationMs != null && document.source.durationMs >= 1,
                "source.durationMs must be >= 1");
        validatePolicy(document.speakerPolicy);

        int count = document.speakerPolicy.resolvedCount;
        List<String> canonical = canonicalSpeakerIds(count);
        require(document.speakers != null && document.speakers.size() == count,
                "speakers.size must equal resolvedCount");
        require(canonical.equals(document.speakerPolicy.speakerIds),
                "speakerPolicy.speakerIds must be speaker-1..speaker-N");
        Set<String> speakerIds = new HashSet<>();
        for (int index = 0; index < count; index++) {
            ReportDocument.Speaker speaker = document.speakers.get(index);
            require(speaker != null, "speaker is required");
            require(canonical.get(index).equals(speaker.id), "speaker IDs must be canonical and ordered");
            require(Integer.valueOf(index + 1).equals(speaker.order), "speaker.order must be contiguous 1..N");
            require(("speaker." + (index + 1)).equals(speaker.colorToken),
                    "speaker.colorToken must match speaker.order");
            requireText(speaker.displayName, "speaker.displayName");
            requireText(speaker.shortLabel, "speaker.shortLabel");
            require(speakerIds.add(speaker.id), "duplicate speaker ID");
        }

        require(document.segments != null && !document.segments.isEmpty(), "segments must not be empty");
        Set<String> segmentIds = new HashSet<>();
        Set<String> revisionIds = new HashSet<>();
        Set<String> usedSpeakers = new HashSet<>();
        long previousStart = -1;
        for (ReportDocument.Segment segment : document.segments) {
            require(segment != null, "segment is required");
            requireText(segment.id, "segment.id");
            require(segmentIds.add(segment.id), "segment IDs must be unique");
            require(segment.startMs != null && segment.startMs >= 0, "segment.startMs must be >= 0");
            require(segment.endMs != null && segment.endMs > segment.startMs,
                    "segment.endMs must be greater than startMs");
            require(segment.endMs <= document.source.durationMs,
                    "segment.endMs exceeds source duration");
            require(segment.startMs >= previousStart, "segments must be ordered by startMs");
            previousStart = segment.startMs;
            require(speakerIds.contains(segment.speakerId), "segment has unknown speakerId");
            usedSpeakers.add(segment.speakerId);
            requireText(segment.rawText, "segment.rawText");
            requireText(segment.normalizedText, "segment.normalizedText");
            requireText(segment.displayText, "segment.displayText");
            require(segment.confidence != null && unit(segment.confidence),
                    "segment.confidence must be in [0,1]");
            validateEvidence(segment, canonical);
            validateTextAudit(segment, revisionIds);
        }
        require(usedSpeakers.equals(speakerIds), "every declared speaker must occur in the transcript");
        require(document.provenance != null, "provenance is required");
        requireText(document.provenance.pipelineVersion, "provenance.pipelineVersion");
        require(Boolean.TRUE.equals(document.provenance.offline), "provenance.offline must be true");
        require(document.provenance.models != null && !document.provenance.models.isEmpty(),
                "provenance.models must not be empty");
    }

    public static List<String> canonicalSpeakerIds(int count) {
        require(count >= 1, "speaker count must be positive");
        return IntStream.rangeClosed(1, count).mapToObj(i -> "speaker-" + i).toList();
    }

    private static void validatePolicy(ReportDocument.SpeakerPolicy policy) {
        require(policy != null, "speakerPolicy is required");
        require(MODES.contains(policy.mode), "speakerPolicy.mode must be auto/manual/hybrid");
        require(policy.resolvedCount != null && policy.resolvedCount >= 1,
                "speakerPolicy.resolvedCount must be positive");
        require(Boolean.TRUE.equals(policy.requireExactSet), "speakerPolicy.requireExactSet must be true");
        require(Boolean.FALSE.equals(policy.unknownSpeakerAllowed),
                "speakerPolicy.unknownSpeakerAllowed must be false");
        require(Boolean.TRUE.equals(policy.speakerChangeRequiresEvidence),
                "speakerPolicy.speakerChangeRequiresEvidence must be true");
        if ("manual".equals(policy.mode) || "hybrid".equals(policy.mode)) {
            require(policy.resolvedCount.equals(policy.requestedCount),
                    "manual/hybrid requestedCount must equal resolvedCount");
        }
        if ("auto".equals(policy.mode) || "hybrid".equals(policy.mode)) {
            require(policy.detection != null
                            && policy.resolvedCount.equals(policy.detection.estimatedCount),
                    "auto/hybrid detection must resolve the declared count");
            requireText(policy.detection.provider, "speakerPolicy.detection.provider");
            if (policy.detection.confidence != null) {
                require(unit(policy.detection.confidence),
                        "speakerPolicy.detection.confidence must be in [0,1]");
            }
            if (policy.detection.candidates != null) {
                Set<Integer> candidateCounts = new HashSet<>();
                for (ReportDocument.CountCandidate candidate : policy.detection.candidates) {
                    require(candidate != null, "speaker count candidate is required");
                    require(candidate.count != null && candidate.count >= 1,
                            "speaker count candidate.count must be positive");
                    require(candidateCounts.add(candidate.count),
                            "speaker count candidate counts must be unique");
                    require(candidate.confidence != null && unit(candidate.confidence),
                            "speaker count candidate.confidence must be in [0,1]");
                }
            }
        }
        if (policy.minimumCount != null) {
            require(policy.minimumCount >= 1 && policy.minimumCount <= policy.resolvedCount,
                    "minimumCount exceeds resolvedCount");
        }
        if (policy.maximumCount != null) {
            require(policy.maximumCount >= policy.resolvedCount,
                    "maximumCount is below resolvedCount");
        }
    }

    private static void validateEvidence(ReportDocument.Segment segment, List<String> canonical) {
        require(segment.evidence != null, "segment.evidence is required");
        require(segment.evidence.asr != null, "evidence.asr is required");
        requireText(segment.evidence.asr.provider, "evidence.asr.provider");
        requireText(segment.evidence.asr.model, "evidence.asr.model");
        require(segment.evidence.asr.confidence != null && unit(segment.evidence.asr.confidence),
                "evidence.asr.confidence must be in [0,1]");
        require(segment.evidence.boundary != null, "evidence.boundary is required");
        requireText(segment.evidence.boundary.provider, "evidence.boundary.provider");
        require(segment.evidence.boundary.confidence != null
                        && unit(segment.evidence.boundary.confidence),
                "evidence.boundary.confidence must be in [0,1]");
        require(segment.evidence.speaker != null, "evidence.speaker is required");
        requireText(segment.evidence.speaker.provider, "evidence.speaker.provider");
        require(segment.speakerId.equals(segment.evidence.speaker.assignment),
                "speaker assignment must equal segment.speakerId");
        require(segment.evidence.speaker.locked != null, "evidence.speaker.locked is required");
        if (segment.evidence.speaker.margin != null) {
            require(segment.evidence.speaker.margin >= -2
                            && segment.evidence.speaker.margin <= 2
                            && Double.isFinite(segment.evidence.speaker.margin),
                    "speaker margin must be finite and in [-2,2]");
        }
        require(segment.evidence.speaker.scores != null
                        && segment.evidence.speaker.scores.size() == canonical.size(),
                "speaker score vector must cover all speakers");
        for (int index = 0; index < canonical.size(); index++) {
            ReportDocument.SpeakerScore score = segment.evidence.speaker.scores.get(index);
            require(score != null && canonical.get(index).equals(score.speakerId),
                    "speaker score vector must preserve canonical order");
            require(score.score != null && score.score >= -2 && score.score <= 2
                            && Double.isFinite(score.score),
                    "speaker score must be finite and in [-2,2]");
        }
        if (segment.evidence.semantic != null) {
            require(!Boolean.TRUE.equals(segment.evidence.semantic.autoApplied),
                    "semantic.autoApplied must not be true");
        }
    }

    private static void validateTextAudit(ReportDocument.Segment segment, Set<String> revisionIds) {
        require(segment.revisions != null, "segment.revisions is required");
        if (segment.revisions.isEmpty()) {
            require(segment.rawText.equals(segment.normalizedText)
                            && segment.rawText.equals(segment.displayText),
                    "segments without revisions must preserve rawText as normalizedText and displayText");
            return;
        }

        String expectedBefore = segment.rawText;
        boolean hasManualRevision = false;
        for (ReportDocument.Revision revision : segment.revisions) {
            require(revision != null, "revision is required");
            requireText(revision.revisionId, "revision.revisionId");
            require(revisionIds.add(revision.revisionId), "revision IDs must be globally unique");
            requireText(revision.type, "revision.type");
            require(!FORBIDDEN_REVISION_TYPES.contains(revision.type),
                    "translation, summary, and analysis revisions are forbidden");
            require(REVISION_TYPES.contains(revision.type), "revision.type is unsupported");
            requireText(revision.source, "revision.source");
            require(REVISION_SOURCES.contains(revision.source), "revision.source is unsupported");
            requireText(revision.reasonCode, "revision.reasonCode");
            requireText(revision.actor, "revision.actor");
            try {
                OffsetDateTime.parse(revision.occurredAt);
            } catch (DateTimeParseException | NullPointerException exception) {
                throw new IllegalArgumentException("revision.occurredAt must be RFC 3339", exception);
            }
            String before = textNode(revision.before, "revision.before");
            String after = textNode(revision.after, "revision.after");
            require(expectedBefore.equals(before), "revision chain must start at rawText and be continuous");
            expectedBefore = after;
            hasManualRevision |= "manual".equals(revision.source);
        }
        require(hasManualRevision, "edited text requires at least one manual revision");
        require(expectedBefore.equals(segment.normalizedText)
                        && expectedBefore.equals(segment.displayText),
                "final revision must equal normalizedText and displayText");
    }

    private static String textNode(JsonNode node, String field) {
        require(node != null && node.isTextual() && !node.textValue().isBlank(),
                field + " must be a non-empty text node");
        return node.textValue();
    }

    private static boolean unit(double value) {
        return value >= 0 && value <= 1 && Double.isFinite(value);
    }

    private static void requireText(String value, String field) {
        require(value != null && !value.isBlank(), field + " is required");
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}
