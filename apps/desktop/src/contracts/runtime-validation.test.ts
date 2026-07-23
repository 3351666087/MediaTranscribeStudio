import type {
  CreateJobRequest,
  ReviewDecision,
  SpeakerCountDetection,
  SpeakerCountPolicy,
  SpeakerProfile,
  StudioSnapshot,
} from "./studio";
import { PDF_HARD_GATE_IDS } from "./studio";
import {
  assertCreateJobRequest,
  assertReviewDecision,
  assertSpeakerCountPolicy,
  ContractValidationError,
  parseStudioSnapshot,
} from "./runtime-validation";
import {
  createStudioFixture,
  studioFixture,
} from "../mocks/studio-fixture";

const dynamicCounts = [1, 2, 5, 8, 13, 21, 64] as const;

function createSpeakers(count: number): SpeakerProfile[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `speaker-${index + 1}`,
    label: `角色 ${index + 1}`,
    shortLabel: `S${index + 1}`,
    color: `hsl(${(index * 137 + 252) % 360} 52% 49%)`,
    roleHint: "测试角色",
    sampleStatus: "missing",
    locked: false,
    reviewStatus: "pending",
  }));
}

function detectionFor(count: number): SpeakerCountDetection {
  const lower = Math.max(1, count - 1);
  const candidates = [
    { count, confidence: 0.92 },
    ...(lower === count ? [] : [{ count: lower, confidence: 0.74 }]),
    { count: count + 1, confidence: 0.63 },
  ];
  return {
    estimatedCount: count,
    confidence: 0.92,
    candidates,
    provider: "CAM++ test provider",
  };
}

function snapshotFor(
  count: number,
  policy: SpeakerCountPolicy = { mode: "manual", count },
): StudioSnapshot {
  const snapshot = structuredClone(studioFixture);
  snapshot.job.speakerPolicy = policy;
  snapshot.job.speakerCount = count;
  snapshot.job.speakerDetection =
    policy.mode === "manual" ? null : detectionFor(count);
  snapshot.job.reviewOpenCount = 0;
  snapshot.speakers = createSpeakers(count);
  snapshot.reviews = [];
  return snapshot;
}

function validCreateRequest(
  policy: SpeakerCountPolicy,
  speakerLabels: string[],
): CreateJobRequest {
  return {
    title: "动态角色会议",
    mediaPath: "D:\\media\\meeting.mov",
    outputDirectory: "D:\\output",
    strategyId: "balanced",
    language: "auto",
    localLlmMode: "disabled",
    localLlmModel: "qwen3.5:4b",
    localLlmEndpoint: "http://127.0.0.1:11434",
    localLlmEndpointPolicy: "loopback-only",
    localLlmAutoApply: false,
    translationTargets: [],
    polish: false,
    summary: false,
    outputLocale: "en-US",
    businessPromptVersion: "business-v1",
    speakerPolicy: policy,
    speakerLabels,
  };
}

function validReviewDecision(
  overrides: Partial<ReviewDecision> = {},
): ReviewDecision {
  return {
    reviewId: "review-181",
    speakerId: "speaker-2",
    normalizedText: "人工校对后的中文原文。",
    reason: "人工判断该片段应归属第二位说话人。",
    evidence: "本地复听并对比前后片段声纹与问答关系。",
    confidence: 0.95,
    ...overrides,
  };
}

function auditedSnapshot(): StudioSnapshot {
  const snapshot = structuredClone(studioFixture);
  const review = snapshot.reviews[0];
  const previousSpeakerId = review.currentSpeakerId;
  const previousNormalizedText = review.normalizedText;
  review.currentSpeakerId = "speaker-2";
  review.normalizedText = "人工校对后的中文原文。";
  review.reviewed = true;
  review.locked = true;
  review.auditTrail = [
    {
      id: "review-181:human:1:1784678400000",
      sequence: 1,
      recordedAtUnixMs: 1784678400000,
      actor: "human",
      reason: "人工判断该片段应归属第二位说话人。",
      evidence: "本地复听并对比前后片段声纹与问答关系。",
      confidence: 0.95,
      previousSpeakerId,
      speakerId: "speaker-2",
      previousNormalizedText,
      normalizedText: review.normalizedText,
    },
  ];
  snapshot.job.reviewOpenCount -= 1;
  return snapshot;
}

function mutableSnapshot(): {
  strategies: Array<Record<string, unknown>>;
  reviews: Array<Record<string, unknown>>;
} {
  return structuredClone(studioFixture) as unknown as {
    strategies: Array<Record<string, unknown>>;
    reviews: Array<Record<string, unknown>>;
  };
}

describe("dynamic speaker runtime validation", () => {
  it.each(dynamicCounts)(
    "accepts a %i-speaker snapshot with contiguous speaker-1..speaker-N ids",
    (count) => {
      const parsed = parseStudioSnapshot(snapshotFor(count));

      expect(parsed.speakers).toHaveLength(count);
      expect(parsed.speakers.map((speaker) => speaker.id)).toEqual(
        Array.from({ length: count }, (_, index) => `speaker-${index + 1}`),
      );
      expect(parsed.speakers.at(-1)?.id).toBe(`speaker-${count}`);
    },
  );

  it.each([
    { mode: "auto" } as const,
    {
      mode: "hybrid",
      minSpeakers: 2,
      priorCount: 5,
      maxSpeakers: 9,
    } as const,
  ])("accepts an unresolved $mode snapshot before verified media evidence", (policy) => {
    const snapshot = structuredClone(studioFixture);
    snapshot.job.speakerPolicy = policy;
    snapshot.job.speakerCount = null;
    snapshot.job.speakerDetection = null;
    snapshot.job.reviewOpenCount = 0;
    snapshot.speakers = [];
    snapshot.reviews = [];

    const parsed = parseStudioSnapshot(snapshot);

    expect(parsed.job.speakerCount).toBeNull();
    expect(parsed.job.speakerDetection).toBeNull();
    expect(parsed.speakers).toEqual([]);
  });

  it("rejects partial or fabricated unresolved speaker state", () => {
    const countWithoutDetection = structuredClone(studioFixture);
    countWithoutDetection.job.speakerPolicy = { mode: "auto" };
    countWithoutDetection.job.speakerCount = 5;
    countWithoutDetection.job.speakerDetection = null;
    countWithoutDetection.speakers = createSpeakers(5);
    countWithoutDetection.reviews = [];
    expect(() => parseStudioSnapshot(countWithoutDetection)).toThrow(
      /jointly unresolved/u,
    );

    const rosterWithoutCount = structuredClone(studioFixture);
    rosterWithoutCount.job.speakerPolicy = { mode: "auto" };
    rosterWithoutCount.job.speakerCount = null;
    rosterWithoutCount.job.speakerDetection = null;
    rosterWithoutCount.speakers = createSpeakers(1);
    rosterWithoutCount.reviews = [];
    expect(() => parseStudioSnapshot(rosterWithoutCount)).toThrow(
      /must remain empty/u,
    );
  });

  it("accepts auto, arbitrary positive-safe manual counts, and bounded hybrid policies", () => {
    expect(() => assertSpeakerCountPolicy({ mode: "auto" })).not.toThrow();
    expect(() =>
      assertSpeakerCountPolicy({
        mode: "manual",
        count: Number.MAX_SAFE_INTEGER,
      }),
    ).not.toThrow();
    expect(() =>
      assertSpeakerCountPolicy({
        mode: "hybrid",
        minSpeakers: 1,
        priorCount: 13,
        maxSpeakers: 21,
      }),
    ).not.toThrow();
  });

  it.each([
    { mode: "manual", count: 0 },
    { mode: "manual", count: -1 },
    { mode: "manual", count: 1.5 },
    { mode: "manual", count: Number.MAX_SAFE_INTEGER + 1 },
    { mode: "hybrid", minSpeakers: 5, priorCount: 4, maxSpeakers: 8 },
    { mode: "hybrid", minSpeakers: 2, priorCount: 9, maxSpeakers: 8 },
    { mode: "hybrid", minSpeakers: 8, priorCount: 8, maxSpeakers: 2 },
  ])("rejects invalid speaker policy %#", (policy) => {
    expect(() => assertSpeakerCountPolicy(policy)).toThrow(
      ContractValidationError,
    );
  });

  it.each([
    {
      label: "duplicate candidate counts",
      mutate: (detection: SpeakerCountDetection) => {
        detection.candidates = [
          { count: 5, confidence: 0.92 },
          { count: 5, confidence: 0.74 },
        ];
      },
    },
    {
      label: "out-of-range confidence",
      mutate: (detection: SpeakerCountDetection) => {
        detection.confidence = 1.1;
        detection.candidates[0].confidence = 1.1;
      },
    },
    {
      label: "ascending confidence",
      mutate: (detection: SpeakerCountDetection) => {
        detection.candidates = [
          { count: 5, confidence: 0.92 },
          { count: 4, confidence: 0.93 },
        ];
      },
    },
    {
      label: "missing estimatedCount candidate",
      mutate: (detection: SpeakerCountDetection) => {
        detection.candidates = [
          { count: 4, confidence: 0.92 },
          { count: 6, confidence: 0.74 },
        ];
      },
    },
    {
      label: "top candidate mismatch",
      mutate: (detection: SpeakerCountDetection) => {
        detection.candidates = [
          { count: 4, confidence: 0.92 },
          { count: 5, confidence: 0.74 },
        ];
      },
    },
  ])("rejects malformed speaker detection: $label", ({ mutate }) => {
    const snapshot = snapshotFor(5, { mode: "auto" });
    mutate(snapshot.job.speakerDetection as SpeakerCountDetection);

    expect(() => parseStudioSnapshot(snapshot)).toThrow(
      ContractValidationError,
    );
  });

  it("rejects non-contiguous speaker ids and unknown fields", () => {
    const nonContiguous = snapshotFor(5);
    nonContiguous.speakers[4].id = "speaker-6";
    expect(() => parseStudioSnapshot(nonContiguous)).toThrow(
      /expected speaker-5/u,
    );

    const unknownField = snapshotFor(2) as StudioSnapshot & {
      fixedSpeakerLimit?: number;
    };
    unknownField.fixedSpeakerLimit = 5;
    expect(() => parseStudioSnapshot(unknownField)).toThrow(/unknown fields/u);
  });

  it("requires reference-based quality metrics to remain unavailable without labels", () => {
    const parsed = parseStudioSnapshot(createStudioFixture(8));

    expect(parsed.diarizationQuality.der.status).toBe("unavailable");
    expect(parsed.diarizationQuality.jer.status).toBe("unavailable");
    expect(parsed.diarizationQuality.confusion.status).toBe("unavailable");
    expect(parsed.diarizationQuality.overlapF1.status).toBe("unavailable");
    expect(parsed.diarizationQuality.reviewRate).toMatchObject({
      status: "available",
      value: 1.48,
      unit: "percent",
    });
  });

  it("requires every strategy to publish the rejected local semantic-model verdict", () => {
    const parsed = parseStudioSnapshot(structuredClone(studioFixture));

    parsed.strategies.forEach((strategy) => {
      expect(strategy.semanticModel).toBe("qwen3.5:4b");
      expect(strategy.semanticModelStatus).toBe("reject_for_production");
      expect(strategy.semanticModelEvaluation).toContain(
        "the local semantic model is disabled",
      );
      expect(strategy.semanticGuardrail).toContain(
        "cannot auto-edit transcript text or speakers",
      );
    });
  });

  it.each([
    {
      label: "missing status",
      mutate: (strategy: Record<string, unknown>) => {
        delete strategy.semanticModelStatus;
      },
      expected: /semanticModelStatus/u,
    },
    {
      label: "non-rejected status",
      mutate: (strategy: Record<string, unknown>) => {
        strategy.semanticModelStatus = "production_ready";
      },
      expected: /reject_for_production/u,
    },
    {
      label: "different model",
      mutate: (strategy: Record<string, unknown>) => {
        strategy.semanticModel = "qwen3.5:9b";
      },
      expected: /qwen3\.5:4b/u,
    },
    {
      label: "empty evaluation",
      mutate: (strategy: Record<string, unknown>) => {
        strategy.semanticModelEvaluation = "";
      },
      expected: /semanticModelEvaluation/u,
    },
  ])("rejects invalid semantic-model production metadata: $label", ({ mutate, expected }) => {
    const snapshot = mutableSnapshot();
    mutate(snapshot.strategies[0]);

    expect(() => parseStudioSnapshot(snapshot)).toThrow(expected);
  });

  it.each(["semantic_conflict", "term_uncertain"])(
    "rejects the retired review reason %s",
    (reason) => {
      const snapshot = mutableSnapshot();
      snapshot.reviews[0].reasons = [reason];

      expect(() => parseStudioSnapshot(snapshot)).toThrow(/reasons\[0\]/u);
    },
  );

  it.each(["speaker_count_uncertain", "local_audio_review"] as const)(
    "accepts the acoustic-first review reason %s",
    (reason) => {
      const snapshot = mutableSnapshot();
      snapshot.reviews[0].reasons = [reason];

      expect(() => parseStudioSnapshot(snapshot)).not.toThrow();
    },
  );

  it("accepts measured performance and rejects invalid latency, rates, and duplicate stages", () => {
    expect(() => parseStudioSnapshot(structuredClone(studioFixture))).not.toThrow();

    const invertedLatency = structuredClone(studioFixture);
    if (invertedLatency.performance.status !== "measured") {
      throw new Error("测试 fixture 必须包含已测量性能数据。");
    }
    invertedLatency.performance.stageLatency[0].p50Ms =
      invertedLatency.performance.stageLatency[0].p95Ms + 1;
    expect(() => parseStudioSnapshot(invertedLatency)).toThrow(
      /p50Ms must not be greater than p95Ms/u,
    );

    const invalidRate = structuredClone(studioFixture);
    if (invalidRate.performance.status !== "measured") {
      throw new Error("测试 fixture 必须包含已测量性能数据。");
    }
    invalidRate.performance.cacheHitRate = 100.1;
    expect(() => parseStudioSnapshot(invalidRate)).toThrow(/0–100 range/u);

    const duplicateStage = structuredClone(studioFixture);
    if (duplicateStage.performance.status !== "measured") {
      throw new Error("测试 fixture 必须包含已测量性能数据。");
    }
    duplicateStage.performance.stageLatency[1].stageId =
      duplicateStage.performance.stageLatency[0].stageId;
    expect(() => parseStudioSnapshot(duplicateStage)).toThrow(
      /must not repeat a pipeline stage/u,
    );
  });

  it("rejects out-of-range quality percentages and legacy worker naming", () => {
    const invalidQuality = createStudioFixture(5, {
      hasReferenceLabels: true,
    });
    if (invalidQuality.diarizationQuality.der.status !== "available") {
      throw new Error("测试 fixture 必须包含可用 DER。");
    }
    invalidQuality.diarizationQuality.der.value = 101;
    expect(() => parseStudioSnapshot(invalidQuality)).toThrow(/0–100 range/u);

    const legacyWorker = structuredClone(studioFixture) as unknown as {
      system: Record<string, unknown>;
    };
    delete legacyWorker.system.inferenceWorker;
    legacyWorker.system.pythonWorker = "ready";
    expect(() => parseStudioSnapshot(legacyWorker)).toThrow(/unknown fields/u);
  });

  it("rejects reviewOpenCount values that disagree with the open queue", () => {
    const snapshot = structuredClone(studioFixture);
    snapshot.job.reviewOpenCount += 1;

    expect(() => parseStudioSnapshot(snapshot)).toThrow(
      /must match the number of unreviewed segments/u,
    );
  });

  function evaluatedPdfSnapshot(): StudioSnapshot {
    const snapshot = structuredClone(studioFixture);
    snapshot.pdfQuality.status = "passed";
    snapshot.pdfQuality.passNumber = 2;
    snapshot.pdfQuality.score = 93;
    snapshot.pdfQuality.pageCount = 7;
    snapshot.pdfQuality.renderedAt = "2026-07-22T10:00:00Z";
    snapshot.pdfQuality.hardGates.forEach((gate) => {
      gate.status = "passed";
      gate.detail = `Verified evidence for ${gate.id}.`;
    });
    snapshot.pdfQuality.facets.forEach((facet) => {
      facet.status = "passed";
      facet.score = 93;
      facet.evidence = `Verified visual evidence for ${facet.id}.`;
    });
    snapshot.pdfQuality.repairQueue = [];
    snapshot.pdfQuality.evidenceDigest = `sha256:${"a".repeat(64)}`;
    return snapshot;
  }

  function unresolvedPdfSnapshot({
    status,
    repairStatus,
    passNumber = 2,
  }: {
    status: "repair-required" | "blocked";
    repairStatus: "open" | "blocked";
    passNumber?: 1 | 2 | 3 | 4 | 5;
  }): StudioSnapshot {
    const snapshot = evaluatedPdfSnapshot();
    const facet = snapshot.pdfQuality.facets[0];
    facet.status = "repair";
    facet.score = 72;
    snapshot.pdfQuality.status = status;
    snapshot.pdfQuality.passNumber = passNumber;
    snapshot.pdfQuality.score = 91.32;
    snapshot.pdfQuality.repairQueue = [
      {
        id: "repair-aesthetic-coherence",
        priority: 1,
        sourceId: facet.id,
        title: "Repair visual coherence",
        detail: "Adjust the safe CSS scope and render the complete document again.",
        safeScope: "css",
        status: repairStatus,
      },
    ];
    return snapshot;
  }

  it("requires the Java/PDF hard-gate IDs in the exact 13-item order", () => {
    const parsed = parseStudioSnapshot(structuredClone(studioFixture));

    expect(parsed.pdfQuality.hardGates.map(({ id }) => id)).toEqual([
      "PDF-OPENABLE",
      "PDF-PAGE-COUNT",
      "PDF-PAGE-SIZE",
      "PDF-TRANSCRIPT-TEXT-INTEGRITY",
      "PDF-SEGMENT-COUNT",
      "PDF-TIMESTAMP-INTEGRITY",
      "PDF-SPEAKER-SET-INTEGRITY",
      "PDF-FONT-EMBEDDED",
      "PDF-NO-BLANK-PAGES",
      "PDF-NO-CONTENT-OVERFLOW",
      "PDF-OFFLINE-ASSETS",
      "PDF-PAGE-EVIDENCE",
      "PDF-IMMUTABLE-CONTENT-HASH",
    ]);
    expect(parsed.pdfQuality.hardGates.map(({ id }) => id)).toEqual(
      PDF_HARD_GATE_IDS,
    );
  });

  it.each([
    {
      label: "missing item",
      mutate: (gates: Array<Record<string, unknown>>) => {
        gates.pop();
      },
    },
    {
      label: "duplicate item",
      mutate: (gates: Array<Record<string, unknown>>) => {
        gates[1].id = gates[0].id;
      },
    },
    {
      label: "legacy or unknown item",
      mutate: (gates: Array<Record<string, unknown>>) => {
        gates[0].id = "PDF-TEXT-INTEGRITY";
      },
    },
    {
      label: "wrong order",
      mutate: (gates: Array<Record<string, unknown>>) => {
        [gates[0], gates[1]] = [gates[1], gates[0]];
      },
    },
  ])("rejects a PDF hard-gate set with $label", ({ mutate }) => {
    const snapshot = structuredClone(studioFixture) as unknown as {
      pdfQuality: { hardGates: Array<Record<string, unknown>> };
    };
    mutate(snapshot.pdfQuality.hardGates);

    expect(() => parseStudioSnapshot(snapshot)).toThrow(
      ContractValidationError,
    );
  });

  it("requires the Design Pack facet IDs in the exact 14-item order", () => {
    const snapshot = evaluatedPdfSnapshot();
    [snapshot.pdfQuality.facets[0], snapshot.pdfQuality.facets[1]] = [
      snapshot.pdfQuality.facets[1],
      snapshot.pdfQuality.facets[0],
    ];

    expect(() => parseStudioSnapshot(snapshot)).toThrow(
      /Design Pack specification order/u,
    );
  });

  it("accepts only evidence-derived pending, passed, repair-required, and blocked states", () => {
    expect(() =>
      parseStudioSnapshot(structuredClone(studioFixture)),
    ).not.toThrow();
    expect(() => parseStudioSnapshot(evaluatedPdfSnapshot())).not.toThrow();
    expect(() =>
      parseStudioSnapshot(
        unresolvedPdfSnapshot({
          status: "repair-required",
          repairStatus: "open",
        }),
      ),
    ).not.toThrow();
    expect(() =>
      parseStudioSnapshot(
        unresolvedPdfSnapshot({
          status: "blocked",
          repairStatus: "blocked",
        }),
      ),
    ).not.toThrow();
    expect(() =>
      parseStudioSnapshot(
        unresolvedPdfSnapshot({
          status: "blocked",
          repairStatus: "open",
          passNumber: 5,
        }),
      ),
    ).not.toThrow();
  });

  it.each([
    {
      label: "failed hard gate",
      mutate: (snapshot: StudioSnapshot) => {
        const gate = snapshot.pdfQuality.hardGates[0];
        gate.status = "failed";
        snapshot.pdfQuality.repairQueue = [
          {
            id: "repair-openable",
            priority: 1,
            sourceId: gate.id,
            title: "Repair PDF openability",
            detail: "Stop release and rebuild the PDF from verified inputs.",
            safeScope: "template",
            status: "blocked",
          },
        ];
      },
    },
    {
      label: "pending hard gate",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.hardGates[0].status = "pending";
      },
    },
    {
      label: "score below the threshold",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.score = 84;
      },
    },
    {
      label: "zero pages",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.pageCount = 0;
      },
    },
    {
      label: "placeholder evidence digest",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.evidenceDigest = "Not generated";
      },
    },
    {
      label: "all-zero evidence digest",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.evidenceDigest = "0".repeat(64);
      },
    },
    {
      label: "facet requiring repair",
      mutate: (snapshot: StudioSnapshot) => {
        const facet = snapshot.pdfQuality.facets[0];
        facet.status = "repair";
        facet.score = 70;
        snapshot.pdfQuality.repairQueue = [
          {
            id: "repair-coherence",
            priority: 1,
            sourceId: facet.id,
            title: "Repair coherence",
            detail: "Apply a safe CSS correction and render again.",
            safeScope: "css",
            status: "open",
          },
        ];
      },
    },
    {
      label: "open repair",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.repairQueue = [
          {
            id: "stale-open-repair",
            priority: 1,
            sourceId: snapshot.pdfQuality.facets[0].id,
            title: "Stale repair",
            detail: "This repair contradicts the passing source check.",
            safeScope: "css",
            status: "open",
          },
        ];
      },
    },
    {
      label: "blocked repair",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.repairQueue = [
          {
            id: "stale-blocked-repair",
            priority: 1,
            sourceId: snapshot.pdfQuality.hardGates[0].id,
            title: "Stale blocked repair",
            detail: "This repair contradicts the passing source check.",
            safeScope: "template",
            status: "blocked",
          },
        ];
      },
    },
  ])("rejects status=passed with $label", ({ mutate }) => {
    const snapshot = evaluatedPdfSnapshot();
    mutate(snapshot);

    expect(() => parseStudioSnapshot(snapshot)).toThrow(
      ContractValidationError,
    );
  });

  it.each([
    {
      label: "a non-zero report score",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.score = 1;
      },
    },
    {
      label: "a rendered page",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.pageCount = 1;
      },
    },
    {
      label: "a verified digest",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.evidenceDigest = "b".repeat(64);
      },
    },
    {
      label: "a passed hard gate",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.hardGates[0].status = "passed";
      },
    },
    {
      label: "a scored facet",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.facets[0].score = 1;
      },
    },
    {
      label: "a repair queue item",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.pdfQuality.repairQueue = [
          {
            id: "premature-repair",
            priority: 1,
            sourceId: snapshot.pdfQuality.facets[0].id,
            title: "Premature repair",
            detail: "No evaluated PDF exists yet.",
            safeScope: "css",
            status: "open",
          },
        ];
      },
    },
  ])("rejects status=pending with $label", ({ mutate }) => {
    const snapshot = structuredClone(studioFixture);
    mutate(snapshot);

    expect(() => parseStudioSnapshot(snapshot)).toThrow(
      ContractValidationError,
    );
  });

  it.each([
    {
      label: "all checks passing",
      snapshot: () => {
        const snapshot = evaluatedPdfSnapshot();
        snapshot.pdfQuality.status = "repair-required";
        return snapshot;
      },
    },
    {
      label: "only blocked repairs",
      snapshot: () =>
        unresolvedPdfSnapshot({
          status: "repair-required",
          repairStatus: "blocked",
        }),
    },
    {
      label: "no repair for the unresolved check",
      snapshot: () => {
        const snapshot = unresolvedPdfSnapshot({
          status: "repair-required",
          repairStatus: "open",
        });
        snapshot.pdfQuality.repairQueue = [];
        return snapshot;
      },
    },
  ])("rejects status=repair-required with $label", ({ snapshot }) => {
    expect(() => parseStudioSnapshot(snapshot())).toThrow(
      ContractValidationError,
    );
  });

  it.each([
    {
      label: "all checks passing",
      snapshot: () => {
        const snapshot = evaluatedPdfSnapshot();
        snapshot.pdfQuality.status = "blocked";
        return snapshot;
      },
    },
    {
      label: "an actionable repair before the final pass",
      snapshot: () =>
        unresolvedPdfSnapshot({
          status: "blocked",
          repairStatus: "open",
          passNumber: 4,
        }),
    },
  ])("rejects status=blocked with $label", ({ snapshot }) => {
    expect(() => parseStudioSnapshot(snapshot())).toThrow(
      ContractValidationError,
    );
  });

  it.each([
    {
      label: "pending with a non-zero score",
      status: "pending" as const,
      score: 1,
    },
    {
      label: "passed below the threshold",
      status: "passed" as const,
      score: 84,
    },
    {
      label: "repair at or above the threshold",
      status: "repair" as const,
      score: 85,
    },
  ])("rejects a facet status/score contradiction: $label", ({ status, score }) => {
    const snapshot =
      status === "pending"
        ? structuredClone(studioFixture)
        : evaluatedPdfSnapshot();
    snapshot.pdfQuality.facets[0].status = status;
    snapshot.pdfQuality.facets[0].score = score;

    expect(() => parseStudioSnapshot(snapshot)).toThrow(
      ContractValidationError,
    );
  });

  it("rejects duplicate repair IDs and repairs disconnected from unresolved checks", () => {
    const duplicate = unresolvedPdfSnapshot({
      status: "repair-required",
      repairStatus: "open",
    });
    duplicate.pdfQuality.repairQueue.push({
      ...duplicate.pdfQuality.repairQueue[0],
    });

    expect(() => parseStudioSnapshot(duplicate)).toThrow(/must be unique/u);

    const disconnected = unresolvedPdfSnapshot({
      status: "repair-required",
      repairStatus: "open",
    });
    disconnected.pdfQuality.repairQueue[0].sourceId =
      disconnected.pdfQuality.hardGates[0].id;

    expect(() => parseStudioSnapshot(disconnected)).toThrow(
      /must reference a failed hard gate or a facet marked for repair/u,
    );
  });

  it("accepts a complete structured review decision", () => {
    expect(() => assertReviewDecision(validReviewDecision())).not.toThrow();
  });

  it.each([
    {
      label: "empty reason",
      value: validReviewDecision({ reason: "" }),
    },
    {
      label: "blank reason",
      value: validReviewDecision({ reason: "   " }),
    },
    {
      label: "empty evidence",
      value: validReviewDecision({ evidence: "" }),
    },
    {
      label: "blank evidence",
      value: validReviewDecision({ evidence: "\t" }),
    },
    {
      label: "confidence below zero",
      value: validReviewDecision({ confidence: -0.01 }),
    },
    {
      label: "confidence above one",
      value: validReviewDecision({ confidence: 1.01 }),
    },
    {
      label: "NaN confidence",
      value: validReviewDecision({ confidence: Number.NaN }),
    },
    {
      label: "infinite confidence",
      value: validReviewDecision({ confidence: Number.POSITIVE_INFINITY }),
    },
  ])("rejects a review decision with $label", ({ value }) => {
    expect(() => assertReviewDecision(value)).toThrow(
      ContractValidationError,
    );
  });

  it("rejects the retired note field and any unknown review-decision field", () => {
    const legacy = {
      reviewId: "review-181",
      speakerId: "speaker-2",
      normalizedText: "人工校对后的中文原文。",
      note: "旧审计说明。",
    };
    const extended = {
      ...validReviewDecision(),
      modelConfidence: 0.95,
    };

    expect(() => assertReviewDecision(legacy)).toThrow(/unknown fields/u);
    expect(() => assertReviewDecision(extended)).toThrow(/unknown fields/u);
  });

  it("accepts an append-only human audit trail with continuous sequence", () => {
    const parsed = parseStudioSnapshot(auditedSnapshot());

    expect(parsed.reviews[0].auditTrail).toHaveLength(1);
    expect(parsed.reviews[0].auditTrail[0]).toMatchObject({
      sequence: 1,
      actor: "human",
      reason: "人工判断该片段应归属第二位说话人。",
      confidence: 0.95,
    });
  });

  it.each([
    {
      label: "unknown audit field",
      mutate: (snapshot: StudioSnapshot) => {
        (
          snapshot.reviews[0].auditTrail[0] as unknown as Record<string, unknown>
        ).modelSuggestion = "禁止";
      },
    },
    {
      label: "non-contiguous sequence",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.reviews[0].auditTrail[0].sequence = 2;
      },
    },
    {
      label: "empty audit evidence",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.reviews[0].auditTrail[0].evidence = " ";
      },
    },
    {
      label: "out-of-range audit confidence",
      mutate: (snapshot: StudioSnapshot) => {
        snapshot.reviews[0].auditTrail[0].confidence = 1.01;
      },
    },
  ])("rejects an audit trail with $label", ({ mutate }) => {
    const snapshot = auditedSnapshot();
    mutate(snapshot);

    expect(() => parseStudioSnapshot(snapshot)).toThrow(
      ContractValidationError,
    );
  });

  it("requires reviewed state and auditTrail presence to agree", () => {
    const reviewedWithoutAudit = auditedSnapshot();
    reviewedWithoutAudit.reviews[0].auditTrail = [];
    expect(() => parseStudioSnapshot(reviewedWithoutAudit)).toThrow(
      /a reviewed segment must contain at least one human audit event/u,
    );

    const auditWithoutReviewed = auditedSnapshot();
    auditWithoutReviewed.reviews[0].reviewed = false;
    auditWithoutReviewed.job.reviewOpenCount += 1;
    expect(() => parseStudioSnapshot(auditWithoutReviewed)).toThrow(
      /an unreviewed segment cannot contain human audit events/u,
    );
  });

  it.each(dynamicCounts)(
    "accepts empty or complete labels and rejects partial labels for %i manual speakers",
    (count) => {
      const labels = Array.from({ length: count }, (_, index) => `角色 ${index + 1}`);
      expect(() =>
        assertCreateJobRequest(
          validCreateRequest({ mode: "manual", count }, []),
        ),
      ).not.toThrow();
      expect(() =>
        assertCreateJobRequest(
          validCreateRequest({ mode: "manual", count }, labels),
        ),
      ).not.toThrow();
      if (count > 1) {
        expect(() =>
          assertCreateJobRequest(
            validCreateRequest({ mode: "manual", count }, labels.slice(0, -1)),
          ),
        ).toThrow(/accepts either an empty label list or exactly/u);
      }
    },
  );

  it("enforces empty labels for auto and empty-or-complete labels for hybrid", () => {
    expect(() =>
      assertCreateJobRequest(validCreateRequest({ mode: "auto" }, [])),
    ).not.toThrow();
    expect(() =>
      assertCreateJobRequest(
        validCreateRequest({ mode: "auto" }, ["不应存在"]),
      ),
    ).toThrow(/Automatic mode accepts only an empty label list/u);

    const hybrid: SpeakerCountPolicy = {
      mode: "hybrid",
      minSpeakers: 2,
      priorCount: 8,
      maxSpeakers: 13,
    };
    const labels = Array.from({ length: 8 }, (_, index) => `角色 ${index + 1}`);
    expect(() =>
      assertCreateJobRequest(validCreateRequest(hybrid, [])),
    ).not.toThrow();
    expect(() =>
      assertCreateJobRequest(validCreateRequest(hybrid, labels)),
    ).not.toThrow();
    expect(() =>
      assertCreateJobRequest(validCreateRequest(hybrid, labels.slice(0, -1))),
    ).toThrow(
      /accepts either an empty label list or exactly 8 speaker labels/u,
    );
  });
});
