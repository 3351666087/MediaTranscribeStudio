import {
  AESTHETIC_FACET_IDS,
  PDF_HARD_GATE_IDS,
  STUDIO_CONTRACT_VERSION,
  type AestheticFacet,
  type ModelStrategy,
  type PdfHardGate,
  type PdfHardGateId,
  type PercentMetric,
  type ReviewReason,
  type ReviewSegment,
  type SpeakerCountCandidate,
  type SpeakerCountPolicy,
  type SpeakerProfile,
  type StudioSnapshot,
} from "../contracts/studio";

const roleSeeds = [
  ["Facilitator", "Agenda, decisions, and follow-through"],
  ["Product", "Requirements and user experience"],
  ["ML", "ASR, voiceprints, and model strategy"],
  ["Engineering", "Performance, deployment, and reliability"],
  ["Quality", "Review, evidence, and delivery"],
  ["Design", "Information hierarchy and accessibility"],
  ["Operations", "Workflow, delivery, and collaboration"],
  ["Observer", "Risk logging and supporting evidence"],
] as const;

const reviewSeeds: Array<{
  id: string;
  startMs: number;
  endMs: number;
  timestampLabel: string;
  rawText: string;
  normalizedText: string;
  reasons: ReviewReason[];
  confidence: number;
  locked: boolean;
}> = [
  {
    id: "review-181",
    startMs: 1_122_120,
    endMs: 1_128_840,
    timestampLabel: "00:18:42.120–00:18:48.840",
    rawText: "那这个边界我觉得可以先锁然后我补充一下不是这个意思",
    normalizedText: "那这个边界我觉得可以先锁。然后我补充一下，不是这个意思。",
    reasons: ["overlap_detected", "speaker_close_score", "local_audio_review"],
    confidence: 0.52,
    locked: false,
  },
  {
    id: "review-206",
    startMs: 1_455_060,
    endMs: 1_462_740,
    timestampLabel: "00:24:15.060–00:24:22.740",
    rawText: "这块用千问三应该就可以显存刚好卡在七点多",
    normalizedText: "这块用 Qwen3 应该就可以，显存刚好在 7GB 多。",
    reasons: ["speaker_close_score", "speaker_count_uncertain", "local_audio_review"],
    confidence: 0.61,
    locked: false,
  },
  {
    id: "review-244",
    startMs: 2_080_320,
    endMs: 2_086_710,
    timestampLabel: "00:34:40.320–00:34:46.710",
    rawText: "这个分数低于八十五就不能说通过这个要写死",
    normalizedText: "这个分数低于 85 就不能说通过，这个要写死。",
    reasons: ["timestamp_boundary", "local_audio_review"],
    confidence: 0.67,
    locked: true,
  },
  {
    id: "review-259",
    startMs: 2_308_900,
    endMs: 2_315_420,
    timestampLabel: "00:38:28.900–00:38:35.420",
    rawText: "这句跟前后声纹都不太像先升级别自动并到最近的人",
    normalizedText: "这句跟前后声纹都不太像，先升级，别自动并到最近的人。",
    reasons: ["speaker_outlier", "speaker_count_uncertain", "local_audio_review"],
    confidence: 0.39,
    locked: false,
  },
];

function speakerColor(index: number): string {
  const hue = Math.round((index * 137.508 + 252) % 360);
  return `hsl(${hue} 52% 49%)`;
}

function assertSpeakerCount(speakerCount: number): void {
  if (
    !Number.isSafeInteger(speakerCount) ||
    speakerCount < 1
  ) {
    throw new Error("The fixture speaker count must be a positive safe integer.");
  }
}

export function createSpeakerProfiles(speakerCount: number): SpeakerProfile[] {
  assertSpeakerCount(speakerCount);
  return Array.from({ length: speakerCount }, (_, index) => {
    const [role, roleHint] = roleSeeds[index] ?? [
      `Role ${index + 1}`,
      "Awaiting local voiceprint, boundary, and audio review",
    ];
    return {
      id: `speaker-${index + 1}` as const,
      label: `Speaker ${index + 1} · ${role}`,
      shortLabel: `S${index + 1}`,
      color: speakerColor(index),
      roleHint,
      sampleStatus: index % 6 === 3 ? "needs_review" : "ready",
      locked: index === 0 || index % 5 === 2,
      reviewStatus: index % 6 === 3 ? "needs_review" : "confirmed",
    };
  });
}

export const modelStrategies: ModelStrategy[] = [
  {
    id: "balanced",
    label: "Balanced",
    description: "CAM++ dynamic clustering, Qwen3-ASR, and FunASR boundaries work together; difficult segments enter local human review.",
    asrModel: "Qwen3-ASR-1.7B",
    diarizationModel: "CAM++ · dynamic clustering",
    semanticModel: "qwen3.5:4b",
    semanticModelStatus: "reject_for_production",
    semanticModelEvaluation: "Production evaluation: the local semantic model is disabled and cannot auto-edit transcript text or speakers.",
    estimatedVramGb: 7.2,
    semanticGuardrail: "The local semantic model is disabled in production and cannot auto-edit transcript text or speakers.",
    recommended: true,
  },
  {
    id: "quality",
    label: "Quality first",
    description: "Difficult segments receive local reruns, a second overlap pass, and human listening while preserving the full acoustic evidence chain.",
    asrModel: "Qwen3-ASR-1.7B · dual-window review",
    diarizationModel: "CAM++ · second overlap pass",
    semanticModel: "qwen3.5:4b",
    semanticModelStatus: "reject_for_production",
    semanticModelEvaluation: "Production evaluation: the local semantic model is disabled and cannot auto-edit transcript text or speakers.",
    estimatedVramGb: 8,
    semanticGuardrail: "The local semantic model is disabled in production and cannot auto-edit transcript text or speakers.",
  },
  {
    id: "memory-saver",
    label: "Memory saver",
    description: "Reduces VRAM use while retaining acoustic anomaly escalation, local audio review, and human confirmation.",
    asrModel: "SenseVoiceSmall",
    diarizationModel: "CAM++ · CPU clustering",
    semanticModel: "qwen3.5:4b",
    semanticModelStatus: "reject_for_production",
    semanticModelEvaluation: "Production evaluation: the local semantic model is disabled and cannot auto-edit transcript text or speakers.",
    estimatedVramGb: 4.4,
    semanticGuardrail: "The local semantic model is disabled in production and cannot auto-edit transcript text or speakers.",
  },
];

const hardGateSeed: Record<PdfHardGateId, Omit<PdfHardGate, "id">> = {
  "PDF-OPENABLE": {
    label: "PDF opens successfully",
    status: "pending",
    detail: "Awaiting PDFBox validation after this job generates a PDF.",
  },
  "PDF-PAGE-COUNT": {
    label: "Page count matches",
    status: "pending",
    detail: "Awaiting comparison of rendered pages, PDFBox pages, and the evidence manifest.",
  },
  "PDF-PAGE-SIZE": {
    label: "Page sizes are compliant",
    status: "pending",
    detail: "Awaiting A4 size validation for every page in this job.",
  },
  "PDF-TRANSCRIPT-TEXT-INTEGRITY": {
    label: "Transcript text is intact",
    status: "pending",
    detail: "Awaiting comparison of PDFBox-extracted text with the immutable transcript manifest.",
  },
  "PDF-SEGMENT-COUNT": {
    label: "Segment count matches",
    status: "pending",
    detail: "Awaiting comparison of transcript and PDF speech-segment counts.",
  },
  "PDF-TIMESTAMP-INTEGRITY": {
    label: "Timestamps are intact",
    status: "pending",
    detail: "Awaiting validation that timestamps are extractable, monotonic, and aligned with boundaries.",
  },
  "PDF-SPEAKER-SET-INTEGRITY": {
    label: "Speaker set is complete",
    status: "pending",
    detail: "Awaiting validation of the dynamic speaker legend, labels, and non-color markers.",
  },
  "PDF-FONT-EMBEDDED": {
    label: "Required fonts are embedded",
    status: "pending",
    detail: "Awaiting checks for multilingual font embedding and deterministic fallback.",
  },
  "PDF-NO-BLANK-PAGES": {
    label: "No blank pages",
    status: "pending",
    detail: "Awaiting page-by-page confirmation of verifiable content.",
  },
  "PDF-NO-CONTENT-OVERFLOW": {
    label: "No content overflow",
    status: "pending",
    detail: "Awaiting page evidence checks for safe areas and content overflow.",
  },
  "PDF-OFFLINE-ASSETS": {
    label: "All assets are offline",
    status: "pending",
    detail: "Awaiting a scan for remote URLs, scripts, fonts, images, and telemetry dependencies.",
  },
  "PDF-PAGE-EVIDENCE": {
    label: "Page evidence is complete",
    status: "pending",
    detail: "Awaiting generation and linking of per-page PNGs and the contact sheet.",
  },
  "PDF-IMMUTABLE-CONTENT-HASH": {
    label: "Immutable content hashes match",
    status: "pending",
    detail: "Awaiting SHA-256 manifests for the transcript, PDF, and page evidence.",
  },
};

const hardGates: PdfHardGate[] = PDF_HARD_GATE_IDS.map((id) => ({
  id,
  ...hardGateSeed[id],
}));

const facetLabels: Record<AestheticFacet["id"], string> = {
  "AESTHETIC-COHERENCE": "Coherence",
  "AESTHETIC-DISTINCTION": "Distinction",
  "AESTHETIC-REFINEMENT": "Refinement",
  "AESTHETIC-PROPORTION": "Proportion",
  "AESTHETIC-HIERARCHY": "Hierarchy",
  "AESTHETIC-TYPOGRAPHY": "Typography",
  "AESTHETIC-COLOR-RELATIONSHIPS": "Color relationships",
  "AESTHETIC-RHYTHM": "Rhythm",
  "AESTHETIC-DENSITY": "Information density",
  "AESTHETIC-RESTRAINT": "Restraint",
  "AESTHETIC-REAL-CONTENT-STRESS": "Real-content stress",
  "AESTHETIC-FONT-FAILURE": "Font failure",
  "AESTHETIC-IMAGE-FAILURE": "Image failure",
  "AESTHETIC-SCRIPT-FAILURE": "Script failure",
};

const facets: AestheticFacet[] = AESTHETIC_FACET_IDS.map((id) => ({
  id,
  label: facetLabels[id],
  score: 0,
  status: "pending",
  evidence: "Awaiting per-page PNGs, a contact sheet, and failure-scenario evidence for this job.",
}));

const unavailableReferenceMetric = (): PercentMetric => ({
  status: "unavailable",
  reason: "Reference labels are missing, so this metric cannot be calculated or inferred.",
});

function qualityMetric(value: number, source: string): PercentMetric {
  return {
    status: "available",
    value,
    unit: "percent",
    source,
  };
}

function createDetection(
  speakerCount: number,
  policy: SpeakerCountPolicy,
): StudioSnapshot["job"]["speakerDetection"] {
  if (policy.mode === "manual") {
    return null;
  }
  const candidates: SpeakerCountCandidate[] = [
    { count: speakerCount, confidence: 0.92 },
  ];
  for (const [offset, confidence] of [
    [-1, 0.74],
    [1, 0.63],
  ] as const) {
    const count = speakerCount + offset;
    if (count < 1) {
      continue;
    }
    if (
      policy.mode === "hybrid" &&
      (count < policy.minSpeakers || count > policy.maxSpeakers)
    ) {
      continue;
    }
    candidates.push({ count, confidence });
  }
  return {
    estimatedCount: speakerCount,
    confidence: candidates[0].confidence,
    candidates,
    provider: "CAM++ · spectral clustering",
  };
}

function createReviews(speakers: SpeakerProfile[]): ReviewSegment[] {
  const waveform = [12, 24, 46, 62, 35, 75, 88, 41, 30, 66, 51, 28, 18, 44, 72, 59];
  return reviewSeeds.map((seed, index) => {
    const currentIndex = index % speakers.length;
    const alternateIndex = (currentIndex + 1) % speakers.length;
    const currentSpeaker = speakers[currentIndex];
    const alternateSpeaker = speakers[alternateIndex];
    const candidates = [
      {
        speakerId: currentSpeaker.id,
        score: seed.confidence,
        evidence: "CAM++ top-1; strong acoustic continuity with adjacent segments",
      },
    ];
    if (alternateSpeaker.id !== currentSpeaker.id) {
      candidates.push({
        speakerId: alternateSpeaker.id,
        score: Math.max(0.1, seed.confidence - 0.04),
        evidence: "CAM++ top-2; similar local voiceprint score, for human listening comparison only",
      });
    }
    return {
      ...seed,
      currentSpeakerId: currentSpeaker.id,
      candidates,
      confidenceBand: seed.confidence < 0.55 ? "low" : "medium",
      waveform: waveform.map((height, waveformIndex) =>
        Math.max(10, (height + index * 9 + waveformIndex * 3) % 92),
      ),
      reviewed: false,
      auditTrail: [],
    };
  });
}

export interface StudioFixtureOptions {
  speakerPolicy?: SpeakerCountPolicy;
  hasReferenceLabels?: boolean;
}

export function createStudioFixture(
  speakerCount: number,
  options: StudioFixtureOptions = {},
): StudioSnapshot {
  assertSpeakerCount(speakerCount);
  const speakers = createSpeakerProfiles(speakerCount);
  const speakerPolicy =
    options.speakerPolicy ??
    ({
      mode: "hybrid",
      minSpeakers: Math.max(1, speakerCount - 3),
      priorCount: speakerCount,
      maxSpeakers: speakerCount + 3,
    } satisfies SpeakerCountPolicy);
  const reviews = createReviews(speakers);
  const hasReferenceLabels = options.hasReferenceLabels ?? false;

  return {
    contractVersion: STUDIO_CONTRACT_VERSION,
    job: {
      id: "job-demo-001",
      title: `Product review · ${speakerCount} dynamic speakers`,
      sourcePath: "C:\\Media\\sample-meeting.mov",
      durationLabel: "45:42",
      status: "review_required",
      progress: 86,
      startedAt: "2026-01-15 09:30",
      speakerPolicy,
      speakerCount,
      speakerDetection: createDetection(speakerCount, speakerPolicy),
      reviewOpenCount: reviews.length,
      activeStrategyId: "balanced",
    },
    speakers,
    strategies: modelStrategies,
    stages: [
      {
        id: "media",
        label: "Media preprocessing",
        shortLabel: "Preprocess",
        status: "completed",
        progress: 100,
        detail: "48 kHz → 16 kHz mono; loudness normalization complete",
        durationLabel: "00:31",
      },
      {
        id: "vad",
        label: "Speech boundaries",
        shortLabel: "Boundaries",
        status: "completed",
        progress: 100,
        detail: "FSMN-VAD · 294 candidate segments",
        durationLabel: "00:18",
      },
      {
        id: "speaker",
        label: "Speaker voiceprints",
        shortLabel: "Voiceprints",
        status: "completed",
        progress: 100,
        detail: `CAM++ · ${speakerPolicy.mode} policy · ${speakerCount} speakers detected dynamically · overlap candidates escalated`,
        durationLabel: "02:43",
      },
      {
        id: "asr",
        label: "Source-language recognition",
        shortLabel: "Recognition",
        status: "completed",
        progress: 100,
        detail: "Qwen3-ASR-1.7B · 271 valid speech segments",
        durationLabel: "08:22",
      },
      {
        id: "alignment",
        label: "Time alignment",
        shortLabel: "Alignment",
        status: "completed",
        progress: 100,
        detail: "FunASR word boundaries written to the audit trail",
        durationLabel: "01:11",
      },
      {
        id: "review",
        label: "Acoustic evidence review",
        shortLabel: "Review",
        status: "warning",
        progress: 91,
        detail: `${reviews.length} low-margin, boundary, overlap, or outlier segments await human confirmation`,
        durationLabel: "03:48",
      },
      {
        id: "document",
        label: "Report assembly",
        shortLabel: "Report",
        status: "pending",
        progress: 0,
        detail: "Awaiting an empty review queue",
      },
      {
        id: "pdf",
        label: "PDF rendering and QA",
        shortLabel: "PDF",
        status: "pending",
        progress: 0,
        detail: "OpenHTMLtoPDF + PDFBox",
      },
    ],
    events: [
      {
        id: "evt-052",
        sequence: 52,
        type: "warning",
        stageId: "review",
        severity: "warning",
        timestamp: "13:23:46",
        title: "Overlap and outlier candidates detected",
        detail: "Low-margin, uncertain-count, boundary-conflict, overlap, and outlier segments entered the human review queue.",
      },
      {
        id: "evt-051",
        sequence: 51,
        type: "stage.progress",
        stageId: "review",
        severity: "info",
        timestamp: "13:23:29",
        title: "Acoustic evidence review 91%",
        detail: "High-confidence CAM++ and human decisions remain locked; close candidates enter local audio review only.",
      },
      {
        id: "evt-050",
        sequence: 50,
        type: "artifact.created",
        stageId: "alignment",
        severity: "success",
        timestamp: "13:19:41",
        title: "Time-boundary evidence saved",
        detail: "alignment-evidence.json · SHA-256 awaiting final manifest aggregation.",
      },
      {
        id: "evt-001",
        sequence: 1,
        type: "job.started",
        severity: "success",
        timestamp: "13:06:12",
        title: "Job started",
        detail: "Input media registered; output directory is writable.",
      },
    ],
    reviews,
    artifacts: [
      {
        id: "artifact-report",
        name: "report-document.json",
        kind: "transcript-json",
        relativePath: "output/report-document.json",
        sizeLabel: "1.8 MB",
        createdAt: "Awaiting review",
        integrity: "pending",
      },
      {
        id: "artifact-transcript",
        name: "source-transcript.zh-Hans.txt",
        kind: "transcript-text",
        relativePath: "output/source-transcript.zh-Hans.txt",
        sizeLabel: "62 KB",
        createdAt: "13:23",
        integrity: "verified",
        sha256: "a0c6…f93",
      },
      {
        id: "artifact-pdf",
        name: "source-transcript.zh-Hans.pdf",
        kind: "pdf",
        relativePath: "output/pdf/source-transcript.zh-Hans.pdf",
        sizeLabel: "3.4 MB",
        createdAt: "Awaiting render",
        integrity: "pending",
      },
      {
        id: "artifact-contact",
        name: "contact-sheet.png",
        kind: "contact-sheet",
        relativePath: "artifacts/screens/contact-sheet.png",
        sizeLabel: "1.2 MB",
        createdAt: "Awaiting render",
        integrity: "pending",
      },
      {
        id: "artifact-quality",
        name: "pdf-quality-report.json",
        kind: "quality-report",
        relativePath: "artifacts/qa/pdf-quality-report.json",
        sizeLabel: "24 KB",
        createdAt: "Reference preview",
        integrity: "verified",
        sha256: "9bd4…2ce",
      },
      {
        id: "artifact-repair",
        name: "repair-queue.json",
        kind: "repair-queue",
        relativePath: "artifacts/qa/repair-queue.json",
        sizeLabel: "4 KB",
        createdAt: "Reference preview",
        integrity: "verified",
        sha256: "27f1…8aa",
      },
    ],
    diarizationQuality: {
      der: hasReferenceLabels
        ? qualityMetric(7.4, "Local human reference labels")
        : unavailableReferenceMetric(),
      jer: hasReferenceLabels
        ? qualityMetric(11.8, "Local human reference labels")
        : unavailableReferenceMetric(),
      confusion: hasReferenceLabels
        ? qualityMetric(3.1, "Local human reference labels")
        : unavailableReferenceMetric(),
      overlapF1: hasReferenceLabels
        ? qualityMetric(82.6, "Local human reference labels")
        : unavailableReferenceMetric(),
      reviewRate: qualityMetric(
        Number(((reviews.length / 271) * 100).toFixed(2)),
        "Pending review segments / valid speech segments",
      ),
    },
    performance: {
      status: "measured",
      sourceLabel: "Local demonstration sample (not a production benchmark)",
      rtf: 0.37,
      stageLatency: [
        { stageId: "media", p50Ms: 84, p95Ms: 121 },
        { stageId: "vad", p50Ms: 19, p95Ms: 43 },
        { stageId: "speaker", p50Ms: 73, p95Ms: 168 },
        { stageId: "asr", p50Ms: 412, p95Ms: 796 },
        { stageId: "alignment", p50Ms: 38, p95Ms: 92 },
        { stageId: "review", p50Ms: 126, p95Ms: 544 },
        { stageId: "document", p50Ms: 41, p95Ms: 87 },
        { stageId: "pdf", p50Ms: 1180, p95Ms: 1840 },
      ],
      cacheHitRate: 78.4,
      selectiveEscalationRate: 6.8,
      recomputeRate: 2.2,
      peakResources: {
        cpuPercent: 84,
        ramGb: 10.6,
        vramGb: 7.4,
      },
    },
    pdfQuality: {
      status: "pending",
      passNumber: 1,
      score: 0,
      minimumScore: 85,
      pageCount: 0,
      renderedAt: "Awaiting render for this job",
      hardGates: hardGates.map((gate) =>
        gate.id === "PDF-SPEAKER-SET-INTEGRITY"
          ? {
              ...gate,
              detail: `Awaiting validation of the legend, labels, and non-color markers for ${speakerCount} dynamic speakers.`,
            }
          : gate,
      ),
      facets,
      repairQueue: [],
      evidenceDigest: "Awaiting per-page evidence and immutable hashes for this job.",
    },
    system: {
      offline: true,
      backendMode: "mock",
      gpuLabel: "NVIDIA RTX 4070 Laptop",
      vramLabel: "8 GB",
      inferenceWorker: "ready",
      javaRenderer: "ready",
    },
  };
}

export const studioFixture = createStudioFixture(8);
