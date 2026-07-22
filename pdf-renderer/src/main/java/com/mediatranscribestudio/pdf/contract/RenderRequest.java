package com.mediatranscribestudio.pdf.contract;

import java.util.List;

public final class RenderRequest {
    public String schemaVersion;
    public String requestId;
    public String jobId;
    public String reportDocumentPath;
    public String outputDirectory;
    public Renderer renderer;
    public QualityPolicy qualityPolicy;

    public static final class Renderer {
        public String provider;
        public Boolean offline;
        public Page page;
        public FontPolicy fontPolicy;
        public String templateId;
    }

    public static final class Page {
        public String size;
        public Double marginMm;
    }

    public static final class FontPolicy {
        public Boolean requireEmbeddedCjk;
        public Boolean allowSystemFallback;
        public String preferredFont;
    }

    public static final class QualityPolicy {
        public Boolean requireHardGates;
        public Double minimumScore;
        public Integer maxRounds;
        public Integer captureDpi;
        public List<String> facetIds;
    }
}
