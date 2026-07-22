package com.mediatranscribestudio.pdf.qa;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public enum AestheticFacetId {
    COHERENCE("AESTHETIC-COHERENCE", 0.08),
    DISTINCTION("AESTHETIC-DISTINCTION", 0.07),
    REFINEMENT("AESTHETIC-REFINEMENT", 0.08),
    PROPORTION("AESTHETIC-PROPORTION", 0.07),
    HIERARCHY("AESTHETIC-HIERARCHY", 0.08),
    TYPOGRAPHY("AESTHETIC-TYPOGRAPHY", 0.10),
    COLOR_RELATIONSHIPS("AESTHETIC-COLOR-RELATIONSHIPS", 0.07),
    RHYTHM("AESTHETIC-RHYTHM", 0.07),
    DENSITY("AESTHETIC-DENSITY", 0.07),
    RESTRAINT("AESTHETIC-RESTRAINT", 0.06),
    REAL_CONTENT_STRESS("AESTHETIC-REAL-CONTENT-STRESS", 0.10),
    FONT_FAILURE("AESTHETIC-FONT-FAILURE", 0.05),
    IMAGE_FAILURE("AESTHETIC-IMAGE-FAILURE", 0.05),
    SCRIPT_FAILURE("AESTHETIC-SCRIPT-FAILURE", 0.05);

    static {
        BigDecimal sum = Arrays.stream(values())
                .map(item -> BigDecimal.valueOf(item.weight))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        if (sum.compareTo(BigDecimal.ONE) != 0) {
            throw new ExceptionInInitializerError("aesthetic facet weights must sum to exactly 1");
        }
        Set<String> ids = new HashSet<>();
        for (AestheticFacetId item : values()) {
            if (!ids.add(item.id)) {
                throw new ExceptionInInitializerError("duplicate aesthetic facet ID: " + item.id);
            }
        }
    }

    private final String id;
    private final double weight;

    AestheticFacetId(String id, double weight) {
        this.id = id;
        this.weight = weight;
    }

    public String id() {
        return id;
    }

    public double weight() {
        return weight;
    }

    public static List<String> ids() {
        return Arrays.stream(values()).map(AestheticFacetId::id).toList();
    }
}
