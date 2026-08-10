import type { OutputCustomization } from "./output-customization";

export const STUDIO_CONTRACT_VERSION = "1.6.0" as const;

export const PDF_HARD_GATE_IDS = [
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
] as const;

export const AESTHETIC_FACET_IDS = [
  "AESTHETIC-COHERENCE",
  "AESTHETIC-DISTINCTION",
  "AESTHETIC-REFINEMENT",
  "AESTHETIC-PROPORTION",
  "AESTHETIC-HIERARCHY",
  "AESTHETIC-TYPOGRAPHY",
  "AESTHETIC-COLOR-RELATIONSHIPS",
  "AESTHETIC-RHYTHM",
  "AESTHETIC-DENSITY",
  "AESTHETIC-RESTRAINT",
  "AESTHETIC-REAL-CONTENT-STRESS",
  "AESTHETIC-FONT-FAILURE",
  "AESTHETIC-IMAGE-FAILURE",
  "AESTHETIC-SCRIPT-FAILURE",
] as const;

export const REVIEW_COMMAND_DEFINITIONS = {
  "speaker.update": {
    ipcCommand: "update_speaker",
    label: "Speaker profile update",
  },
  "review.apply": {
    ipcCommand: "apply_review_decision",
    label: "Review decision submission",
  },
} as const;

export type StudioContractVersion = typeof STUDIO_CONTRACT_VERSION;
export type PdfHardGateId = (typeof PDF_HARD_GATE_IDS)[number];
export type AestheticFacetId = (typeof AESTHETIC_FACET_IDS)[number];
export type ReviewCommandKind = keyof typeof REVIEW_COMMAND_DEFINITIONS;
export type ReviewCommandPhase =
  | "idle"
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "unavailable";
export type AppSection = "overview" | "review" | "artifacts" | "pdf-qa";
export type SpeakerId = `speaker-${number}`;
export type SpeakerReviewStatus = "pending" | "needs_review" | "confirmed";
export type SpeakerCountPolicy =
  | {
      mode: "auto";
    }
  | {
      mode: "manual";
      count: number;
    }
  | {
      mode: "hybrid";
      minSpeakers: number;
      maxSpeakers: number;
      priorCount: number;
    };
export type JobStatus =
  | "draft"
  | "queued"
  | "running"
  | "review_required"
  | "completed"
  | "failed"
  | "cancelled";
export type StageStatus = "pending" | "running" | "completed" | "warning" | "blocked";
export type Severity = "info" | "success" | "warning" | "error";
export type ConfidenceBand = "high" | "medium" | "low";
export type ArtifactKind =
  | "transcript-json"
  | "transcript-text"
  | "pdf"
  | "page-image"
  | "contact-sheet"
  | "quality-report"
  | "repair-queue"
  | "log";
export type ReviewReason =
  | "speaker_close_score"
  | "speaker_count_uncertain"
  | "overlap_detected"
  | "timestamp_boundary"
  | "speaker_outlier"
  | "local_audio_review";
export type ModelStrategyId = "balanced" | "quality" | "memory-saver";
export type LocalLlmMode = "disabled" | "business";

export interface SpeakerProfile {
  id: SpeakerId;
  label: string;
  shortLabel: string;
  color: string;
  roleHint: string;
  sampleStatus: "ready" | "missing" | "needs_review";
  locked: boolean;
  reviewStatus: SpeakerReviewStatus;
}

export interface SpeakerCountCandidate {
  count: number;
  confidence: number;
}

export interface SpeakerCountDetection {
  estimatedCount: number;
  confidence: number;
  candidates: SpeakerCountCandidate[];
  provider?: string;
}

export interface ModelStrategy {
  id: ModelStrategyId;
  label: string;
  description: string;
  asrModel: string;
  diarizationModel: string;
  semanticModel: string;
  semanticModelStatus: "suggestion_only";
  semanticModelEvaluation: string;
  estimatedVramGb: number;
  semanticGuardrail: string;
  recommended?: boolean;
}

export interface PipelineStage {
  id:
    | "media"
    | "vad"
    | "speaker"
    | "asr"
    | "alignment"
    | "review"
    | "document"
    | "pdf";
  label: string;
  shortLabel: string;
  status: StageStatus;
  progress: number;
  detail: string;
  durationLabel?: string;
}

export interface StudioEvent {
  id: string;
  sequence: number;
  type:
    | "job.started"
    | "stage.started"
    | "stage.progress"
    | "artifact.created"
    | "review.required"
    | "review.decision.persisted"
    | "warning"
    | "job.failed"
    | "job.completed"
    | "job.cancelled";
  stageId?: PipelineStage["id"];
  severity: Severity;
  timestamp: string;
  title: string;
  detail: string;
}

export interface SpeakerCandidate {
  speakerId: SpeakerProfile["id"];
  score: number;
  evidence: string;
}

export interface ReviewAuditEvent {
  id: string;
  sequence: number;
  recordedAtUnixMs: number;
  actor: "human";
  reason: string;
  evidence: string;
  confidence: number;
  previousSpeakerId: SpeakerProfile["id"];
  speakerId: SpeakerProfile["id"];
  previousNormalizedText: string;
  normalizedText: string;
}

export interface ReviewSegment {
  id: string;
  startMs: number;
  endMs: number;
  timestampLabel: string;
  rawText: string;
  normalizedText: string;
  currentSpeakerId: SpeakerProfile["id"];
  candidates: SpeakerCandidate[];
  reasons: ReviewReason[];
  confidence: number;
  confidenceBand: ConfidenceBand;
  waveform: readonly number[];
  locked: boolean;
  reviewed: boolean;
  auditTrail: ReviewAuditEvent[];
}

export interface ArtifactItem {
  id: string;
  name: string;
  kind: ArtifactKind;
  relativePath: string;
  sizeLabel: string;
  createdAt: string;
  integrity: "verified" | "pending" | "failed";
  sha256?: string;
}

export interface PdfHardGate {
  id: PdfHardGateId;
  label: string;
  status: "pending" | "passed" | "failed";
  detail: string;
}

export interface AestheticFacet {
  id: AestheticFacetId;
  label: string;
  score: number;
  status: "pending" | "passed" | "repair";
  evidence: string;
}

export interface RepairQueueItem {
  id: string;
  priority: 1 | 2 | 3 | 4;
  sourceId: PdfHardGate["id"] | AestheticFacetId;
  title: string;
  detail: string;
  safeScope: "template" | "css" | "font" | "image" | "pagination";
  status: "open" | "applied" | "blocked";
}

export interface PdfQualityReport {
  status: "pending" | "passed" | "repair-required" | "blocked";
  passNumber: 1 | 2 | 3 | 4 | 5;
  score: number;
  minimumScore: 85;
  pageCount: number;
  renderedAt: string;
  hardGates: PdfHardGate[];
  facets: AestheticFacet[];
  repairQueue: RepairQueueItem[];
  evidenceDigest: string;
}

export interface ReviewCommandStatus {
  command: ReviewCommandKind;
  ipcCommand: (typeof REVIEW_COMMAND_DEFINITIONS)[ReviewCommandKind]["ipcCommand"];
  phase: ReviewCommandPhase;
  label: string;
  message: string;
  backendMode: SystemStatus["backendMode"] | "unknown";
  startedAtUnixMs: number | null;
  completedAtUnixMs: number | null;
}

export interface JobSummary {
  id: string;
  title: string;
  sourcePath: string;
  durationLabel: string;
  status: JobStatus;
  progress: number;
  startedAt: string;
  speakerPolicy: SpeakerCountPolicy;
  speakerCount: number | null;
  speakerDetection: SpeakerCountDetection | null;
  reviewOpenCount: number;
  activeStrategyId: ModelStrategyId;
}

export type PercentMetric =
  | {
      status: "available";
      value: number;
      unit: "percent";
      source: string;
    }
  | {
      status: "unavailable";
      reason: string;
    };

export interface DiarizationQualityMetrics {
  der: PercentMetric;
  jer: PercentMetric;
  confusion: PercentMetric;
  overlapF1: PercentMetric;
  reviewRate: PercentMetric;
}

export interface StagePerformance {
  stageId: PipelineStage["id"];
  p50Ms: number;
  p95Ms: number;
}

export type PerformanceMetrics =
  | {
      status: "measured";
      sourceLabel: string;
      rtf: number;
      stageLatency: StagePerformance[];
      cacheHitRate: number;
      selectiveEscalationRate: number;
      recomputeRate: number;
      peakResources: {
        cpuPercent: number;
        ramGb: number;
        vramGb: number;
      };
    }
  | {
      status: "unavailable";
      reason: string;
    };

export interface SystemStatus {
  offline: true;
  backendMode: "mock" | "tauri-ipc";
  gpuLabel: string;
  vramLabel: string;
  inferenceWorker: "ready" | "missing" | "busy";
  javaRenderer: "ready" | "missing" | "busy";
}

export interface StudioSnapshot {
  contractVersion: StudioContractVersion;
  job: JobSummary;
  speakers: SpeakerProfile[];
  strategies: ModelStrategy[];
  stages: PipelineStage[];
  events: StudioEvent[];
  reviews: ReviewSegment[];
  artifacts: ArtifactItem[];
  diarizationQuality: DiarizationQualityMetrics;
  performance: PerformanceMetrics;
  pdfQuality: PdfQualityReport;
  system: SystemStatus;
}

export interface CreateJobRequest {
  title: string;
  mediaPath: string;
  outputDirectory: string;
  strategyId: ModelStrategyId;
  speakerPolicy: SpeakerCountPolicy;
  speakerLabels: string[];
  language: string;
  localLlmMode: LocalLlmMode;
  localLlmModel: string;
  localLlmEndpoint: string;
  /** Stable provider id, e.g. ollama-loopback, openai, anthropic, or custom. */
  llmProvider?: string;
  /** Name of the environment variable that contains the API key. */
  llmApiKeyEnv?: string | null;
  /** Optional HTTP(S) proxy; the secret itself is never part of this request. */
  llmProxyUrl?: string | null;
  localLlmEndpointPolicy: "loopback-only" | "remote-explicit";
  localLlmAutoApply: false;
  translationTargets: string[];
  summary: boolean;
  outputLocale: string;
  businessPromptVersion: "business-v1";
  /**
   * Canonical native output recipe forwarded unchanged through Tauri to the
   * worker trust boundary after strict TypeScript and Rust validation.
   */
  outputCustomization?: OutputCustomization;
}

export interface CreateJobResult {
  accepted: boolean;
  jobId: string;
  message: string;
}

export interface JobRuntimeStatus {
  jobId: string;
  status: JobStatus;
  revision: number;
  acceptedByWorker: boolean;
  projected: boolean;
  inFlight: boolean;
  workerEventRouteRegistered: boolean;
  cancellable: boolean;
  volatileOnly: true;
}

export interface StudioJobItem extends JobRuntimeStatus {
  title: string | null;
  sourcePath: string | null;
  progress: number | null;
}

export interface ArtifactOpenResult {
  artifactId: string;
  canonicalPath: string;
  opened: boolean;
  message: string;
}

export type IpcErrorCode =
  | "invalid_request"
  | "invalid_path"
  | "not_found"
  | "conflict"
  | "path_boundary_violation"
  | "state_unavailable";

export interface IpcErrorPayload {
  code: IpcErrorCode;
  message: string;
}

export interface ReviewDecision {
  reviewId: string;
  speakerId: SpeakerId;
  normalizedText: string;
  reason: string;
  evidence: string;
  confidence: number;
}

export interface UpdateSpeakerRequest {
  speakerId: SpeakerId;
  label: string;
  locked: boolean;
  reviewStatus: SpeakerReviewStatus;
}

export interface DesktopBackend {
  getSnapshot(): Promise<StudioSnapshot>;
  listJobs(): Promise<JobRuntimeStatus[]>;
  selectJob(jobId: string): Promise<StudioSnapshot>;
  createJob(request: CreateJobRequest): Promise<CreateJobResult>;
  cancelJob(jobId: string): Promise<void>;
  updateSpeaker(request: UpdateSpeakerRequest): Promise<SpeakerProfile>;
  applyReviewDecision(decision: ReviewDecision): Promise<ReviewSegment>;
  openArtifact(artifactId: string): Promise<ArtifactOpenResult>;
}
