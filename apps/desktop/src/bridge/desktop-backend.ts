import type {
  ArtifactOpenResult,
  CreateJobRequest,
  CreateJobResult,
  DesktopBackend,
  ReviewDecision,
  ReviewSegment,
  SpeakerProfile,
  StudioSnapshot,
  UpdateSpeakerRequest,
} from "../contracts/studio";
import {
  assertCreateJobRequest,
  assertReviewDecision,
  assertUpdateSpeakerRequest,
  parseStudioSnapshot,
} from "../contracts/runtime-validation";
import {
  createStudioFixture,
  studioFixture,
} from "../mocks/studio-fixture";
import { TauriDesktopBackend } from "./tauri-backend";

const pause = async (duration = 180): Promise<void> => {
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, duration);
  });
};

const cloneSnapshot = (source: StudioSnapshot = studioFixture): StudioSnapshot =>
  parseStudioSnapshot(structuredClone(source));

const EVIDENCE_SPEAKER_COUNTS = new Set([13, 64, 129]);
const EVIDENCE_LOAD_FAILURE =
  "Visual evidence mode: the local backend returned no workspace snapshot. The app stopped safely without modifying any data.";

interface MockEvidenceState {
  snapshot: StudioSnapshot;
  initialLoadFailure: string | null;
}

function resolveMockEvidence(search: string): MockEvidenceState {
  const params = new URLSearchParams(search);
  const evidenceMode = params.get("evidence");

  if (evidenceMode === "failure") {
    return {
      snapshot: cloneSnapshot(),
      initialLoadFailure: EVIDENCE_LOAD_FAILURE,
    };
  }

  if (evidenceMode === "speakers") {
    const speakerCount = Number(params.get("speakers"));
    if (EVIDENCE_SPEAKER_COUNTS.has(speakerCount)) {
      return {
        snapshot: cloneSnapshot(createStudioFixture(speakerCount)),
        initialLoadFailure: null,
      };
    }
  }

  return {
    snapshot: cloneSnapshot(),
    initialLoadFailure: null,
  };
}

function speakerColor(index: number): string {
  const hue = Math.round((index * 137.508 + 252) % 360);
  return `hsl(${hue} 52% 49%)`;
}

function createSpeakerProfiles(
  count: number,
  labels: readonly string[],
): SpeakerProfile[] {
  try {
    return Array.from({ length: count }, (_, index) => ({
      id: `speaker-${index + 1}` as const,
      label: labels[index]?.trim() || `Speaker ${index + 1}`,
      shortLabel: `S${index + 1}`,
      color: speakerColor(index),
      roleHint:
        "Awaiting local voiceprint, boundary, and focused-audio review",
      sampleStatus: "missing" as const,
      locked: false,
      reviewStatus: "pending" as const,
    }));
  } catch (error) {
    throw new Error(
      `This device could not allocate the complete ${count}-speaker profile list. The task was not created, and no speaker was truncated, merged, or silently downgraded.${
        error instanceof Error ? ` ${error.message}` : ""
      }`,
    );
  }
}

export class MockDesktopBackend implements DesktopBackend {
  private snapshot: StudioSnapshot;
  private readonly initialLoadFailure: string | null;

  constructor(
    search = typeof window === "undefined" ? "" : window.location.search,
  ) {
    const evidence = resolveMockEvidence(search);
    this.snapshot = evidence.snapshot;
    this.initialLoadFailure = evidence.initialLoadFailure;
  }

  async getSnapshot(): Promise<StudioSnapshot> {
    await pause(80);
    if (this.initialLoadFailure) {
      throw new Error(this.initialLoadFailure);
    }
    return structuredClone(this.snapshot);
  }

  async createJob(request: CreateJobRequest): Promise<CreateJobResult> {
    assertCreateJobRequest(request);
    await pause(320);
    const speakerCount =
      request.speakerPolicy.mode === "manual"
        ? request.speakerPolicy.count
        : null;
    const speakerDetection = null;
    const speakers =
      speakerCount === null
        ? []
        : createSpeakerProfiles(speakerCount, request.speakerLabels);
    this.snapshot.job = {
      ...this.snapshot.job,
      id: `mock-${request.title.trim().replace(/\s+/g, "-") || "meeting"}`,
      title: request.title.trim(),
      sourcePath: request.mediaPath.trim(),
      status: "queued",
      progress: 0,
      speakerPolicy: structuredClone(request.speakerPolicy),
      speakerCount,
      speakerDetection,
      reviewOpenCount: 0,
      activeStrategyId: request.strategyId,
    };
    this.snapshot.speakers = speakers;
    this.snapshot.reviews = [];
    this.snapshot.artifacts = [];
    this.snapshot.diarizationQuality = {
      der: {
        status: "unavailable",
        reason:
          "The new task has no reference labels, so DER cannot be measured or inferred.",
      },
      jer: {
        status: "unavailable",
        reason:
          "The new task has no reference labels, so JER cannot be measured or inferred.",
      },
      confusion: {
        status: "unavailable",
        reason:
          "The new task has no reference labels, so speaker confusion cannot be measured or inferred.",
      },
      overlapF1: {
        status: "unavailable",
        reason:
          "The new task has no reference labels, so overlap F1 cannot be measured or inferred.",
      },
      reviewRate: {
        status: "available",
        value: 0,
        unit: "percent",
        source: "Pending review segments / generated valid speech segments",
      },
    };
    this.snapshot.performance = {
      status: "unavailable",
      reason:
        "The task has not run, so model-stage and resource-sampling data are unavailable.",
    };
    this.snapshot.events = [
      {
        id: `event-${this.snapshot.job.id}`,
        sequence: 1,
        type: "job.started",
        stageId: "media",
        severity: "success",
        timestamp: "Registered",
        title: "Preview task created safely",
        detail:
          speakerCount === null
            ? "Awaiting media analysis. No speaker count, acoustic estimate, or roster has been invented by preview mode."
            : `Created the complete ${speakerCount}-speaker roster requested by the manual policy.`,
      },
    ];
    this.snapshot = parseStudioSnapshot(this.snapshot);
    return {
      accepted: true,
      jobId: this.snapshot.job.id,
      message:
        "The preview task was created. The local model runtime will execute it when real IPC is available.",
    };
  }

  async cancelJob(jobId: string): Promise<void> {
    await pause();
    if (this.snapshot.job.id === jobId) {
      this.snapshot.job.status = "cancelled";
    }
  }

  async updateSpeaker(request: UpdateSpeakerRequest): Promise<SpeakerProfile> {
    assertUpdateSpeakerRequest(request);
    await pause(160);
    const speaker = this.snapshot.speakers.find(
      (item) => item.id === request.speakerId,
    );
    if (!speaker) {
      throw new Error("That speaker does not exist in the current task.");
    }
    if (
      speaker.locked &&
      request.locked &&
      speaker.label !== request.label.trim()
    ) {
      throw new Error("Unlock the speaker before changing the name.");
    }
    speaker.label = request.label.trim();
    speaker.locked = request.locked;
    speaker.reviewStatus = request.reviewStatus;
    return structuredClone(speaker);
  }

  async applyReviewDecision(decision: ReviewDecision): Promise<ReviewSegment> {
    assertReviewDecision(decision);
    await pause(240);
    if (!this.snapshot.speakers.some((speaker) => speaker.id === decision.speakerId)) {
      throw new Error(
        "The review decision references a speaker outside the current task.",
      );
    }
    const review = this.snapshot.reviews.find((item) => item.id === decision.reviewId);
    if (!review) {
      throw new Error("The review segment was not found.");
    }
    if (review.locked && decision.speakerId !== review.currentSpeakerId) {
      throw new Error(
        "A human-locked segment cannot be reassigned to another speaker.",
      );
    }

    const normalizedText = decision.normalizedText.trim();
    if (normalizedText.length === 0) {
      throw new Error(
        "The human-reviewed transcript cannot be blank after trimming.",
      );
    }
    const previousSpeakerId = review.currentSpeakerId;
    const previousNormalizedText = review.normalizedText;
    const sequence = review.auditTrail.length + 1;
    const recordedAtUnixMs = Date.now();
    const auditEvent = {
      id: `${review.id}:human:${sequence}:${recordedAtUnixMs}`,
      sequence,
      recordedAtUnixMs,
      actor: "human" as const,
      reason: decision.reason.trim(),
      evidence: decision.evidence.trim(),
      confidence: decision.confidence,
      previousSpeakerId,
      speakerId: decision.speakerId,
      previousNormalizedText,
      normalizedText,
    };

    review.currentSpeakerId = decision.speakerId;
    review.normalizedText = normalizedText;
    review.auditTrail.push(auditEvent);
    review.reviewed = true;
    review.locked = true;
    this.snapshot.job.reviewOpenCount = this.snapshot.reviews.filter(
      (item) => !item.reviewed,
    ).length;
    return structuredClone(review);
  }

  async openArtifact(artifactId: string): Promise<ArtifactOpenResult> {
    await pause(100);
    const artifact = this.snapshot.artifacts.find((item) => item.id === artifactId);
    if (!artifact) {
      throw new Error("The artifact does not exist or has not been generated.");
    }
    return {
      artifactId,
      canonicalPath: artifact.relativePath,
      opened: false,
      message:
        "The secure browser preview does not access the local file system.",
    };
  }
}

export function isTauriRuntime(): boolean {
  return (
    typeof window !== "undefined" &&
    (window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== undefined
  );
}

/**
 * Browser previews and unit tests explicitly use a mock without file-system
 * access. The Tauri WebView uses only the strict IPC adapter with fixed command
 * names. IPC failures never fall back silently to the mock.
 */
export function createDesktopBackend(tauriRuntime = isTauriRuntime()): DesktopBackend {
  return tauriRuntime ? new TauriDesktopBackend() : new MockDesktopBackend();
}

export const desktopBackend = createDesktopBackend();
