package com.mediatranscribestudio.pdf.contract;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.List;

public final class ReportDocument {
    public String schemaVersion;
    public String documentId;
    public String generatedAt;
    public String title;
    public String language;
    public Source source;
    public SpeakerPolicy speakerPolicy;
    public List<Speaker> speakers;
    public List<Segment> segments;
    public Analysis analysis;
    public Provenance provenance;
    public Privacy privacy;

    public static final class Source {
        public String fileName;
        public String mediaType;
        public Long durationMs;
        public String sha256;
    }

    public static final class SpeakerPolicy {
        public String mode;
        public Integer resolvedCount;
        public Integer requestedCount;
        public Integer minimumCount;
        public Integer maximumCount;
        public Boolean requireExactSet;
        public List<String> speakerIds;
        public SpeakerCountDetection detection;
        public Boolean unknownSpeakerAllowed;
        public Boolean speakerChangeRequiresEvidence;
    }

    public static final class SpeakerCountDetection {
        public String provider;
        public Integer estimatedCount;
        public Double confidence;
        public List<CountCandidate> candidates;
    }

    public static final class CountCandidate {
        public Integer count;
        public Double confidence;
    }

    public static final class Speaker {
        public String id;
        public Integer order;
        public String displayName;
        public String shortLabel;
        public String colorToken;
        public String role;
        public List<String> aliases;
    }

    public static final class Segment {
        public String id;
        public Long startMs;
        public Long endMs;
        public String speakerId;
        public String rawText;
        public String normalizedText;
        public String displayText;
        public String language;
        public Double confidence;
        public String reviewStatus;
        public String overlapGroupId;
        public List<String> sourceSegmentIds;
        public Evidence evidence;
        public List<Revision> revisions;
    }

    public static final class Evidence {
        public AsrEvidence asr;
        public BoundaryEvidence boundary;
        public SpeakerEvidence speaker;
        public SemanticEvidence semantic;
        public AudioReview audioReview;
    }

    public static final class AsrEvidence {
        public String provider;
        public String model;
        public String modelRevision;
        public Double confidence;
        public Integer wordCount;
    }

    public static final class BoundaryEvidence {
        public String provider;
        public String model;
        public Double confidence;
        public Boolean overlapDetected;
    }

    public static final class SpeakerEvidence {
        public String provider;
        public String model;
        public String assignment;
        public Boolean locked;
        public Double margin;
        public List<SpeakerScore> scores;
    }

    public static final class SpeakerScore {
        public String speakerId;
        public Double score;
    }

    public static final class SemanticEvidence {
        public String provider;
        public String model;
        public String decision;
        public Boolean autoApplied;
        public List<String> reasonCodes;
    }

    public static final class AudioReview {
        public String status;
        public String reviewer;
        public String notes;
    }

    public static final class Revision {
        public String revisionId;
        public String type;
        public String source;
        public String reasonCode;
        public JsonNode before;
        public JsonNode after;
        public String model;
        public String actor;
        public String occurredAt;
    }

    public static final class Analysis {
        public String summary;
        public List<String> keyPoints;
        public List<String> actionItems;
    }

    public static final class Provenance {
        public String pipelineVersion;
        public List<Model> models;
        public Boolean offline;
        public String configSha256;
    }

    public static final class Model {
        public String role;
        public String name;
        public String revision;
        public String quantization;
    }

    public static final class Privacy {
        public Boolean containsRealMeetingText;
        public Boolean exportApproved;
        public String redactionNotes;
    }
}
