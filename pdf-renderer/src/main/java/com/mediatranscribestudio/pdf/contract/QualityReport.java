package com.mediatranscribestudio.pdf.contract;

import java.util.ArrayList;
import java.util.List;

public final class QualityReport {
    public String schemaVersion = "1.0.0";
    public String documentId;
    public Integer round;
    public String status;
    public Double minimumScore;
    public Double score;
    public Boolean hardGatesPassed;
    public List<HardGate> hardGates = new ArrayList<>();
    public List<Facet> facets = new ArrayList<>();
    public List<Evidence> evidence = new ArrayList<>();
    public List<Repair> repairQueue = new ArrayList<>();
    public List<Regression> regressions = new ArrayList<>();

    public static final class HardGate {
        public String id;
        public String status;
        public String message;
        public List<String> evidenceIds = new ArrayList<>();
    }

    public static final class Facet {
        public String id;
        public String status;
        public Double score;
        public Double weight;
        public String message;
        public List<String> evidenceIds = new ArrayList<>();
    }

    public static final class Evidence {
        public String id;
        public String type;
        public String relativePath;
        public String sha256;
        public Boolean verified;
        public Integer pageNumber;
    }

    public static final class Repair {
        public String id;
        public Integer priority;
        public String severity;
        public String checkId;
        public String safeScope;
        public String action;
        public String status;
    }

    public static final class Regression {
        public String checkId;
        public String previousStatus;
        public String currentStatus;
    }
}
