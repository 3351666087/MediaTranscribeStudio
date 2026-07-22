package com.mediatranscribestudio.pdf.contract;

import java.util.ArrayList;
import java.util.List;

public final class ArtifactManifest {
    public String schemaVersion = "1.0.0";
    public String jobId;
    public String documentId;
    public String rendererVersion;
    public String createdAt;
    public List<Artifact> artifacts = new ArrayList<>();

    public static final class Artifact {
        public String artifactId;
        public String type;
        public String relativePath;
        public String mimeType;
        public String sha256;
        public Long bytes;
        public Boolean verified;
        public Integer pageNumber;
        public String producedBy;
        public List<String> sourceArtifactIds;
    }
}
