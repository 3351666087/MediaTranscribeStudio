import {
  AESTHETIC_FACET_IDS,
  PDF_HARD_GATE_IDS,
  REVIEW_COMMAND_DEFINITIONS,
  STUDIO_CONTRACT_VERSION,
  type ArtifactOpenResult,
  type CreateJobRequest,
  type CreateJobResult,
  type IpcErrorCode,
  type ReviewDecision,
  type ReviewCommandStatus,
  type ReviewSegment,
  type SpeakerCountDetection,
  type SpeakerCountPolicy,
  type SpeakerId,
  type SpeakerProfile,
  type StudioSnapshot,
  type UpdateSpeakerRequest,
} from "./studio";

export class ContractValidationError extends Error {
  constructor(message: string) {
    super(`IPC contract validation failed: ${message}`);
    this.name = "ContractValidationError";
  }
}

type UnknownRecord = Record<string, unknown>;

const JOB_STATUSES = [
  "draft",
  "queued",
  "running",
  "review_required",
  "completed",
  "failed",
  "cancelled",
] as const;
const STAGE_STATUSES = ["pending", "running", "completed", "warning", "blocked"] as const;
const SEVERITIES = ["info", "success", "warning", "error"] as const;
const CONFIDENCE_BANDS = ["high", "medium", "low"] as const;
const ARTIFACT_KINDS = [
  "transcript-json",
  "transcript-text",
  "pdf",
  "page-image",
  "contact-sheet",
  "quality-report",
  "repair-queue",
  "log",
] as const;
const REVIEW_REASONS = [
  "speaker_close_score",
  "speaker_count_uncertain",
  "overlap_detected",
  "timestamp_boundary",
  "speaker_outlier",
  "local_audio_review",
] as const;
const STRATEGY_IDS = ["balanced", "quality", "memory-saver"] as const;
const STAGE_IDS = [
  "media",
  "vad",
  "speaker",
  "asr",
  "alignment",
  "review",
  "document",
  "pdf",
] as const;
const EVENT_TYPES = [
  "job.started",
  "stage.started",
  "stage.progress",
  "artifact.created",
  "review.required",
  "review.decision.persisted",
  "warning",
  "job.failed",
  "job.completed",
  "job.cancelled",
] as const;
const SAMPLE_STATUSES = ["ready", "missing", "needs_review"] as const;
const SPEAKER_REVIEW_STATUSES = ["pending", "needs_review", "confirmed"] as const;
const SPEAKER_POLICY_MODES = ["auto", "manual", "hybrid"] as const;
const INTEGRITY_STATUSES = ["verified", "pending", "failed"] as const;
const PDF_STATUSES = ["pending", "passed", "repair-required", "blocked"] as const;
const GATE_STATUSES = ["pending", "passed", "failed"] as const;
const REVIEW_AUDIT_ACTORS = ["human"] as const;
const FACET_STATUSES = ["pending", "passed", "repair"] as const;
const REPAIR_STATUSES = ["open", "applied", "blocked"] as const;
const REPAIR_SCOPES = ["template", "css", "font", "image", "pagination"] as const;
const BACKEND_MODES = ["mock", "tauri-ipc"] as const;
const WORKER_STATUSES = ["ready", "missing", "busy"] as const;
const REVIEW_COMMAND_KINDS = ["speaker.update", "review.apply"] as const;
const REVIEW_COMMAND_PHASES = [
  "idle",
  "pending",
  "running",
  "succeeded",
  "failed",
  "unavailable",
] as const;
const REVIEW_COMMAND_BACKEND_MODES = ["unknown", ...BACKEND_MODES] as const;
const PDF_REPAIR_SOURCE_IDS = [
  ...PDF_HARD_GATE_IDS,
  ...AESTHETIC_FACET_IDS,
] as const;
const IPC_ERROR_CODES = [
  "invalid_request",
  "invalid_path",
  "not_found",
  "conflict",
  "path_boundary_violation",
  "state_unavailable",
] as const satisfies readonly IpcErrorCode[];
const LOCAL_LLM_MODES = ["disabled", "business"] as const;
const ASCII_ALPHA = /^[A-Za-z]+$/u;
const ASCII_ALPHANUMERIC = /^[A-Za-z0-9]+$/u;

function isLanguageVariant(value: string): boolean {
  return (
    (value.length >= 5 &&
      value.length <= 8 &&
      ASCII_ALPHANUMERIC.test(value)) ||
    (value.length === 4 &&
      /^[0-9]/u.test(value) &&
      ASCII_ALPHANUMERIC.test(value))
  );
}

/**
 * Validates the practical BCP-47 subset accepted by the Rust/Tauri boundary.
 * The original spelling and casing are deliberately preserved by callers.
 */
export function isPracticalLanguageTag(
  value: string,
  { allowAuto = false }: { allowAuto?: boolean } = {},
): boolean {
  if (allowAuto && value === "auto") {
    return true;
  }
  if (
    value === "auto" ||
    value.length === 0 ||
    value.length > 128 ||
    !/^[\x00-\x7f]+$/u.test(value) ||
    value.startsWith("-") ||
    value.endsWith("-") ||
    value.includes("--") ||
    /[\u0000-\u001f\u007f]/u.test(value)
  ) {
    return false;
  }

  const subtags = value.split("-");
  if (subtags[0]?.toLocaleLowerCase("en-US") === "x") {
    return (
      subtags.length >= 2 &&
      subtags
        .slice(1)
        .every(
          (subtag) =>
            subtag.length >= 1 &&
            subtag.length <= 8 &&
            ASCII_ALPHANUMERIC.test(subtag),
        )
    );
  }

  const primary = subtags[0] ?? "";
  if (
    primary.length < 2 ||
    primary.length > 8 ||
    !ASCII_ALPHA.test(primary)
  ) {
    return false;
  }

  let index = 1;
  let extlangCount = 0;
  while (
    extlangCount < 3 &&
    subtags[index]?.length === 3 &&
    ASCII_ALPHA.test(subtags[index] ?? "")
  ) {
    index += 1;
    extlangCount += 1;
  }

  if (
    subtags[index]?.length === 4 &&
    ASCII_ALPHA.test(subtags[index] ?? "")
  ) {
    index += 1;
  }

  const region = subtags[index] ?? "";
  if (
    (region.length === 2 && ASCII_ALPHA.test(region)) ||
    (region.length === 3 && /^[0-9]{3}$/u.test(region))
  ) {
    index += 1;
  }

  const variants = new Set<string>();
  while (isLanguageVariant(subtags[index] ?? "")) {
    const normalized = (subtags[index] ?? "").toLocaleLowerCase("en-US");
    if (variants.has(normalized)) {
      return false;
    }
    variants.add(normalized);
    index += 1;
  }

  const extensionSingletons = new Set<string>();
  while (index < subtags.length) {
    const singleton = subtags[index] ?? "";
    if (singleton.toLocaleLowerCase("en-US") === "x") {
      break;
    }
    if (
      singleton.length !== 1 ||
      !ASCII_ALPHANUMERIC.test(singleton)
    ) {
      break;
    }
    const normalized = singleton.toLocaleLowerCase("en-US");
    if (extensionSingletons.has(normalized)) {
      return false;
    }
    extensionSingletons.add(normalized);
    index += 1;

    const extensionStart = index;
    while (
      (subtags[index]?.length ?? 0) >= 2 &&
      (subtags[index]?.length ?? 0) <= 8 &&
      ASCII_ALPHANUMERIC.test(subtags[index] ?? "")
    ) {
      index += 1;
    }
    if (index === extensionStart) {
      return false;
    }
  }

  if ((subtags[index] ?? "").toLocaleLowerCase("en-US") === "x") {
    index += 1;
    const privateStart = index;
    while (
      (subtags[index]?.length ?? 0) >= 1 &&
      (subtags[index]?.length ?? 0) <= 8 &&
      ASCII_ALPHANUMERIC.test(subtags[index] ?? "")
    ) {
      index += 1;
    }
    if (index === privateStart) {
      return false;
    }
  }

  return index === subtags.length;
}

function fail(path: string, expectation: string): never {
  throw new ContractValidationError(`${path} ${expectation}`);
}

function record(value: unknown, path: string): UnknownRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(path, "must be an object.");
  }
  return value as UnknownRecord;
}

function exactKeys(value: UnknownRecord, path: string, allowedKeys: readonly string[]): void {
  const unknownKeys = Object.keys(value).filter((key) => !allowedKeys.includes(key));
  if (unknownKeys.length > 0) {
    fail(path, `contains unknown fields: ${unknownKeys.join(", ")}.`);
  }
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    fail(path, "must be an array.");
  }
  return value;
}

function string(
  value: unknown,
  path: string,
  { min = 0, max = 4096 }: { min?: number; max?: number } = {},
): string {
  if (typeof value !== "string") {
    fail(path, "must be a string.");
  }
  if (value.length < min || value.length > max || value.includes("\0")) {
    fail(path, `must be ${min}–${max} characters and must not contain NUL.`);
  }
  return value;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") {
    fail(path, "must be a boolean.");
  }
  return value;
}

function finiteNumber(
  value: unknown,
  path: string,
  { min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY, integer = false } = {},
): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < min ||
    value > max ||
    (integer && !Number.isInteger(value))
  ) {
    fail(
      path,
      `must be ${integer ? "an integer" : "a finite number"} in the ${min}–${max} range.`,
    );
  }
  return value;
}

function positiveSafeInteger(value: unknown, path: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value <= 0
  ) {
    fail(path, "must be a positive safe integer.");
  }
  return value;
}

function enumeration<const T extends readonly string[]>(
  value: unknown,
  path: string,
  values: T,
): T[number] {
  if (typeof value !== "string" || !values.includes(value)) {
    fail(path, `must be one of: ${values.join(", ")}.`);
  }
  return value;
}

function optionalString(value: unknown, path: string, max = 4096): void {
  if (value !== undefined) {
    string(value, path, { max });
  }
}

function ensureNoControlCharacters(value: string, path: string): void {
  if (/[\u0000-\u001f\u007f]/u.test(value)) {
    fail(path, "must not contain control characters.");
  }
}

function languageTag(
  value: unknown,
  path: string,
  { allowAuto = false }: { allowAuto?: boolean } = {},
): string {
  const tag = string(value, path, { min: 1, max: 128 });
  ensureNoControlCharacters(tag, path);
  if (allowAuto && tag === "auto") {
    return tag;
  }
  if (!isPracticalLanguageTag(tag, { allowAuto })) {
    fail(
      path,
      allowAuto
        ? 'must be "auto" or a practical BCP-47 language tag such as en-US or zh-Hans.'
        : "must be a concrete practical BCP-47 language tag; auto is not allowed.",
    );
  }
  return tag;
}

function loopbackEndpoint(value: unknown, path: string): string {
  const endpoint = string(value, path, { min: 1, max: 2048 });
  ensureNoControlCharacters(endpoint, path);
  let parsed: URL;
  try {
    parsed = new URL(endpoint);
  } catch {
    fail(path, "must be a valid absolute loopback URL.");
  }
  const isLoopbackHost =
    parsed.hostname === "localhost" ||
    parsed.hostname === "127.0.0.1" ||
    parsed.hostname === "[::1]";
  if (
    !isLoopbackHost ||
    !["http:", "https:"].includes(parsed.protocol) ||
    parsed.username.length > 0 ||
    parsed.password.length > 0
  ) {
    fail(path, "must use HTTP(S) with localhost, 127.0.0.1, or [::1].");
  }
  return endpoint;
}

function nonBlankAuditText(value: unknown, path: string): string {
  const text = string(value, path, { min: 1, max: 4096 });
  if (text.trim().length === 0) {
    fail(path, "must not be blank after trimming.");
  }
  ensureNoControlCharacters(text, path);
  return text;
}

export function assertSafeArtifactRelativePath(value: unknown, path = "relativePath"): string {
  const relativePath = string(value, path, { min: 1, max: 1024 });
  if (
    relativePath.startsWith("/") ||
    relativePath.startsWith("\\") ||
    /^[a-zA-Z]:/u.test(relativePath)
  ) {
    fail(
      path,
      "must be a relative path inside the output root, not an absolute path.",
    );
  }
  const components = relativePath.split(/[\\/]/u);
  if (components.some((component) => component === "" || component === "." || component === "..")) {
    fail(path, "must not contain empty components, `.` or `..`.");
  }
  return relativePath;
}

function validateSpeakerId(value: unknown, path: string): SpeakerId {
  const speakerId = string(value, path, { min: 9, max: 32 });
  if (!/^speaker-[1-9]\d*$/u.test(speakerId)) {
    fail(path, "must match speaker-1..speaker-N with a positive integer N.");
  }
  const sequence = Number(speakerId.slice("speaker-".length));
  if (!Number.isSafeInteger(sequence)) {
    fail(path, "must contain a positive safe-integer sequence number.");
  }
  return speakerId as SpeakerId;
}

function validateSpeaker(value: unknown, path: string): void {
  const item = record(value, path);
  exactKeys(item, path, [
    "id",
    "label",
    "shortLabel",
    "color",
    "roleHint",
    "sampleStatus",
    "locked",
    "reviewStatus",
  ]);
  validateSpeakerId(item.id, `${path}.id`);
  string(item.label, `${path}.label`, { min: 1, max: 64 });
  string(item.shortLabel, `${path}.shortLabel`, { min: 1, max: 32 });
  string(item.color, `${path}.color`, { min: 1, max: 32 });
  string(item.roleHint, `${path}.roleHint`, { max: 256 });
  enumeration(item.sampleStatus, `${path}.sampleStatus`, SAMPLE_STATUSES);
  boolean(item.locked, `${path}.locked`);
  enumeration(
    item.reviewStatus,
    `${path}.reviewStatus`,
    SPEAKER_REVIEW_STATUSES,
  );
}

function validateReviewSegment(
  value: unknown,
  path: string,
  speakerIds?: ReadonlySet<SpeakerId>,
): void {
  const item = record(value, path);
  exactKeys(item, path, [
    "id",
    "startMs",
    "endMs",
    "timestampLabel",
    "rawText",
    "normalizedText",
    "currentSpeakerId",
    "candidates",
    "reasons",
    "confidence",
    "confidenceBand",
    "waveform",
    "locked",
    "reviewed",
    "auditTrail",
  ]);
  string(item.id, `${path}.id`, { min: 1, max: 128 });
  const startMs = finiteNumber(item.startMs, `${path}.startMs`, { min: 0, integer: true });
  const endMs = finiteNumber(item.endMs, `${path}.endMs`, { min: 0, integer: true });
  if (startMs > endMs) {
    fail(path, "startMs must not be greater than endMs.");
  }
  string(item.timestampLabel, `${path}.timestampLabel`, { min: 1, max: 128 });
  string(item.rawText, `${path}.rawText`, { max: 100_000 });
  string(item.normalizedText, `${path}.normalizedText`, { max: 100_000 });
  const currentSpeakerId = validateSpeakerId(
    item.currentSpeakerId,
    `${path}.currentSpeakerId`,
  );
  if (speakerIds && !speakerIds.has(currentSpeakerId)) {
    fail(
      `${path}.currentSpeakerId`,
      "must reference a speaker in the current task.",
    );
  }
  array(item.candidates, `${path}.candidates`).forEach((candidate, index) => {
    const entry = record(candidate, `${path}.candidates[${index}]`);
    exactKeys(entry, `${path}.candidates[${index}]`, [
      "speakerId",
      "score",
      "evidence",
    ]);
    const candidateSpeakerId = validateSpeakerId(
      entry.speakerId,
      `${path}.candidates[${index}].speakerId`,
    );
    if (speakerIds && !speakerIds.has(candidateSpeakerId)) {
      fail(
        `${path}.candidates[${index}].speakerId`,
        "must reference a speaker in the current task.",
      );
    }
    finiteNumber(entry.score, `${path}.candidates[${index}].score`, { min: 0, max: 1 });
    string(entry.evidence, `${path}.candidates[${index}].evidence`, { max: 4096 });
  });
  array(item.reasons, `${path}.reasons`).forEach((reason, index) =>
    enumeration(reason, `${path}.reasons[${index}]`, REVIEW_REASONS),
  );
  finiteNumber(item.confidence, `${path}.confidence`, { min: 0, max: 1 });
  enumeration(item.confidenceBand, `${path}.confidenceBand`, CONFIDENCE_BANDS);
  array(item.waveform, `${path}.waveform`).forEach((point, index) =>
    finiteNumber(point, `${path}.waveform[${index}]`, { min: 0, max: 100 }),
  );
  boolean(item.locked, `${path}.locked`);
  boolean(item.reviewed, `${path}.reviewed`);
  const auditTrail = array(item.auditTrail, `${path}.auditTrail`);
  auditTrail.forEach((event, index) => {
    const eventPath = `${path}.auditTrail[${index}]`;
    const entry = record(event, eventPath);
    exactKeys(entry, eventPath, [
      "id",
      "sequence",
      "recordedAtUnixMs",
      "actor",
      "reason",
      "evidence",
      "confidence",
      "previousSpeakerId",
      "speakerId",
      "previousNormalizedText",
      "normalizedText",
    ]);
    string(entry.id, `${eventPath}.id`, { min: 1, max: 256 });
    const sequence = finiteNumber(entry.sequence, `${eventPath}.sequence`, {
      min: 1,
      integer: true,
    });
    if (sequence !== index + 1) {
      fail(
        `${eventPath}.sequence`,
        `must be contiguous and equal ${index + 1}.`,
      );
    }
    finiteNumber(entry.recordedAtUnixMs, `${eventPath}.recordedAtUnixMs`, {
      min: 0,
      integer: true,
    });
    enumeration(entry.actor, `${eventPath}.actor`, REVIEW_AUDIT_ACTORS);
    nonBlankAuditText(entry.reason, `${eventPath}.reason`);
    nonBlankAuditText(entry.evidence, `${eventPath}.evidence`);
    finiteNumber(entry.confidence, `${eventPath}.confidence`, { min: 0, max: 1 });
    const previousSpeakerId = validateSpeakerId(
      entry.previousSpeakerId,
      `${eventPath}.previousSpeakerId`,
    );
    const eventSpeakerId = validateSpeakerId(entry.speakerId, `${eventPath}.speakerId`);
    if (speakerIds && !speakerIds.has(previousSpeakerId)) {
      fail(
        `${eventPath}.previousSpeakerId`,
        "must reference a speaker in the current task.",
      );
    }
    if (speakerIds && !speakerIds.has(eventSpeakerId)) {
      fail(
        `${eventPath}.speakerId`,
        "must reference a speaker in the current task.",
      );
    }
    string(entry.previousNormalizedText, `${eventPath}.previousNormalizedText`, {
      max: 100_000,
    });
    string(entry.normalizedText, `${eventPath}.normalizedText`, {
      min: 1,
      max: 100_000,
    });
  });
  if (item.reviewed === false && auditTrail.length > 0) {
    fail(
      `${path}.auditTrail`,
      "an unreviewed segment cannot contain human audit events.",
    );
  }
  if (item.reviewed === true && auditTrail.length === 0) {
    fail(
      `${path}.auditTrail`,
      "a reviewed segment must contain at least one human audit event.",
    );
  }
}

export function assertSpeakerCountPolicy(
  value: unknown,
  path = "speakerPolicy",
): asserts value is SpeakerCountPolicy {
  const policy = record(value, path);
  const mode = enumeration(policy.mode, `${path}.mode`, SPEAKER_POLICY_MODES);
  if (mode === "auto") {
    exactKeys(policy, path, ["mode"]);
    return;
  }
  if (mode === "manual") {
    exactKeys(policy, path, ["mode", "count"]);
    positiveSafeInteger(policy.count, `${path}.count`);
    return;
  }
  exactKeys(policy, path, [
    "mode",
    "minSpeakers",
    "maxSpeakers",
    "priorCount",
  ]);
  const minSpeakers = positiveSafeInteger(
    policy.minSpeakers,
    `${path}.minSpeakers`,
  );
  const maxSpeakers = positiveSafeInteger(
    policy.maxSpeakers,
    `${path}.maxSpeakers`,
  );
  const priorCount = positiveSafeInteger(
    policy.priorCount,
    `${path}.priorCount`,
  );
  if (minSpeakers > maxSpeakers) {
    fail(path, "minSpeakers must not be greater than maxSpeakers.");
  }
  if (priorCount < minSpeakers || priorCount > maxSpeakers) {
    fail(path, "priorCount must be between minSpeakers and maxSpeakers.");
  }
}

function validateSpeakerCountDetection(
  value: unknown,
  path: string,
): SpeakerCountDetection {
  const detection = record(value, path);
  exactKeys(detection, path, [
    "estimatedCount",
    "confidence",
    "candidates",
    "provider",
  ]);
  const estimatedCount = positiveSafeInteger(
    detection.estimatedCount,
    `${path}.estimatedCount`,
  );
  const confidence = finiteNumber(detection.confidence, `${path}.confidence`, {
    min: 0,
    max: 1,
  });
  if (detection.provider !== undefined) {
    const provider = string(detection.provider, `${path}.provider`, {
      min: 1,
      max: 256,
    });
    ensureNoControlCharacters(provider, `${path}.provider`);
  }

  const candidates = array(detection.candidates, `${path}.candidates`);
  if (candidates.length === 0) {
    fail(
      `${path}.candidates`,
      "must contain at least one speaker-count candidate.",
    );
  }
  const seenCounts = new Set<number>();
  let previousConfidence = Number.POSITIVE_INFINITY;
  candidates.forEach((candidate, index) => {
    const candidatePath = `${path}.candidates[${index}]`;
    const item = record(candidate, candidatePath);
    exactKeys(item, candidatePath, ["count", "confidence"]);
    const count = positiveSafeInteger(item.count, `${candidatePath}.count`);
    const candidateConfidence = finiteNumber(
      item.confidence,
      `${candidatePath}.confidence`,
      { min: 0, max: 1 },
    );
    if (seenCounts.has(count)) {
      fail(
        `${path}.candidates`,
        `must not contain duplicate speaker-count candidate ${count}.`,
      );
    }
    if (candidateConfidence > previousConfidence) {
      fail(
        `${path}.candidates`,
        "must be sorted by non-increasing confidence.",
      );
    }
    seenCounts.add(count);
    previousConfidence = candidateConfidence;
  });
  if (
    !candidates.some(
      (candidate) => record(candidate, path).count === estimatedCount,
    )
  ) {
    fail(
      `${path}.candidates`,
      "must contain a candidate matching estimatedCount.",
    );
  }
  const first = record(candidates[0], `${path}.candidates[0]`);
  if (first.count !== estimatedCount || first.confidence !== confidence) {
    fail(
      path,
      "the first candidate must exactly match top-level estimatedCount and confidence.",
    );
  }
  return value as SpeakerCountDetection;
}

type PdfGateStatus = (typeof GATE_STATUSES)[number];
type PdfFacetStatus = (typeof FACET_STATUSES)[number];
type PdfRepairStatus = (typeof REPAIR_STATUSES)[number];
type PdfRepairSourceId = (typeof PDF_REPAIR_SOURCE_IDS)[number];
type PdfReportStatus = (typeof PDF_STATUSES)[number];

interface ValidatedPdfGate {
  id: PdfRepairSourceId;
  status: PdfGateStatus;
}

interface ValidatedPdfFacet {
  id: PdfRepairSourceId;
  score: number;
  status: PdfFacetStatus;
}

interface ValidatedPdfRepair {
  id: string;
  sourceId: PdfRepairSourceId;
  status: PdfRepairStatus;
}

const VERIFIED_PDF_EVIDENCE_DIGEST = /^[a-f0-9]{64}$/iu;
const PENDING_PDF_EVIDENCE_DIGEST =
  /^(?:not generated|pending|awaiting(?:\s|$))/iu;
const PDF_AESTHETIC_FACET_WEIGHTS = [
  0.08,
  0.07,
  0.08,
  0.07,
  0.08,
  0.1,
  0.07,
  0.07,
  0.07,
  0.06,
  0.1,
  0.05,
  0.05,
  0.05,
] as const;

function hasVerifiedPdfEvidenceDigest(value: string): boolean {
  const normalized = value.trim().replace(/^sha256:/iu, "");
  return (
    VERIFIED_PDF_EVIDENCE_DIGEST.test(normalized) &&
    !/^0{64}$/u.test(normalized)
  );
}

function hasPendingPdfEvidenceDigest(value: string): boolean {
  return PENDING_PDF_EVIDENCE_DIGEST.test(value.trim());
}

function validatePdfQuality(value: unknown, path: string): void {
  const report = record(value, path);
  exactKeys(report, path, [
    "status",
    "passNumber",
    "score",
    "minimumScore",
    "pageCount",
    "renderedAt",
    "hardGates",
    "facets",
    "repairQueue",
    "evidenceDigest",
  ]);
  const reportStatus = enumeration(report.status, `${path}.status`, PDF_STATUSES);
  const passNumber = finiteNumber(report.passNumber, `${path}.passNumber`, {
    min: 1,
    max: 5,
    integer: true,
  });
  const score = finiteNumber(report.score, `${path}.score`, {
    min: 0,
    max: 100,
  });
  if (report.minimumScore !== 85) {
    fail(`${path}.minimumScore`, "must be fixed at 85.");
  }
  const minimumScore = report.minimumScore;
  const pageCount = finiteNumber(report.pageCount, `${path}.pageCount`, {
    min: 0,
    integer: true,
  });
  string(report.renderedAt, `${path}.renderedAt`, { min: 1, max: 128 });
  const hardGates = array(report.hardGates, `${path}.hardGates`);
  if (hardGates.length !== PDF_HARD_GATE_IDS.length) {
    fail(
      `${path}.hardGates`,
      `must contain exactly ${PDF_HARD_GATE_IDS.length} specification-defined items; received ${hardGates.length}.`,
    );
  }
  const validatedHardGates: ValidatedPdfGate[] = [];
  hardGates.forEach((gate, index) => {
    const item = record(gate, `${path}.hardGates[${index}]`);
    exactKeys(item, `${path}.hardGates[${index}]`, [
      "id",
      "label",
      "status",
      "detail",
    ]);
    const gateId = enumeration(
      item.id,
      `${path}.hardGates[${index}].id`,
      PDF_HARD_GATE_IDS,
    );
    if (gateId !== PDF_HARD_GATE_IDS[index]) {
      fail(
        `${path}.hardGates[${index}].id`,
        `must be ${PDF_HARD_GATE_IDS[index]} at this position in the Java/PDF specification order.`,
      );
    }
    string(item.label, `${path}.hardGates[${index}].label`, { min: 1, max: 128 });
    const gateStatus = enumeration(
      item.status,
      `${path}.hardGates[${index}].status`,
      GATE_STATUSES,
    );
    string(item.detail, `${path}.hardGates[${index}].detail`, {
      min: 1,
      max: 4096,
    });
    validatedHardGates.push({ id: gateId, status: gateStatus });
  });
  const facets = array(report.facets, `${path}.facets`);
  if (facets.length !== AESTHETIC_FACET_IDS.length) {
    fail(
      `${path}.facets`,
      `must contain exactly ${AESTHETIC_FACET_IDS.length} specification-defined items; received ${facets.length}.`,
    );
  }
  const validatedFacets: ValidatedPdfFacet[] = [];
  facets.forEach((facet, index) => {
    const item = record(facet, `${path}.facets[${index}]`);
    exactKeys(item, `${path}.facets[${index}]`, [
      "id",
      "label",
      "score",
      "status",
      "evidence",
    ]);
    const facetId = enumeration(
      item.id,
      `${path}.facets[${index}].id`,
      AESTHETIC_FACET_IDS,
    );
    if (facetId !== AESTHETIC_FACET_IDS[index]) {
      fail(
        `${path}.facets[${index}].id`,
        `must be ${AESTHETIC_FACET_IDS[index]} at this position in the Design Pack specification order.`,
      );
    }
    string(item.label, `${path}.facets[${index}].label`, { min: 1, max: 128 });
    const facetScore = finiteNumber(
      item.score,
      `${path}.facets[${index}].score`,
      { min: 0, max: 100 },
    );
    const facetStatus = enumeration(
      item.status,
      `${path}.facets[${index}].status`,
      FACET_STATUSES,
    );
    string(item.evidence, `${path}.facets[${index}].evidence`, {
      min: 1,
      max: 4096,
    });
    if (facetStatus === "pending" && facetScore !== 0) {
      fail(
        `${path}.facets[${index}]`,
        "a pending aesthetic facet must have score 0.",
      );
    }
    if (facetStatus === "passed" && facetScore < minimumScore) {
      fail(
        `${path}.facets[${index}]`,
        "a passed aesthetic facet must meet the minimum score.",
      );
    }
    if (facetStatus === "repair" && facetScore >= minimumScore) {
      fail(
        `${path}.facets[${index}]`,
        "an aesthetic facet marked for repair must be below the minimum score.",
      );
    }
    validatedFacets.push({
      id: facetId,
      score: facetScore,
      status: facetStatus,
    });
  });
  const evidenceDerivedScore =
    Math.round(
      validatedFacets.reduce(
        (total, facet, index) =>
          total + facet.score * PDF_AESTHETIC_FACET_WEIGHTS[index],
        0,
      ) * 100,
    ) / 100;
  if (Math.abs(score - evidenceDerivedScore) > 0.001) {
    fail(
      `${path}.score`,
      `must equal the weighted score derived from all 14 aesthetic facets (${evidenceDerivedScore}).`,
    );
  }
  const repairQueue = array(report.repairQueue, `${path}.repairQueue`);
  const validatedRepairs: ValidatedPdfRepair[] = [];
  const repairIds = new Set<string>();
  repairQueue.forEach((repair, index) => {
    const item = record(repair, `${path}.repairQueue[${index}]`);
    exactKeys(item, `${path}.repairQueue[${index}]`, [
      "id",
      "priority",
      "sourceId",
      "title",
      "detail",
      "safeScope",
      "status",
    ]);
    const repairId = string(item.id, `${path}.repairQueue[${index}].id`, {
      min: 1,
      max: 128,
    });
    if (repairIds.has(repairId)) {
      fail(`${path}.repairQueue[${index}].id`, "must be unique.");
    }
    repairIds.add(repairId);
    finiteNumber(item.priority, `${path}.repairQueue[${index}].priority`, {
      min: 1,
      max: 4,
      integer: true,
    });
    const sourceId = enumeration(
      item.sourceId,
      `${path}.repairQueue[${index}].sourceId`,
      PDF_REPAIR_SOURCE_IDS,
    );
    string(item.title, `${path}.repairQueue[${index}].title`, { min: 1, max: 256 });
    string(item.detail, `${path}.repairQueue[${index}].detail`, {
      min: 1,
      max: 4096,
    });
    enumeration(item.safeScope, `${path}.repairQueue[${index}].safeScope`, REPAIR_SCOPES);
    const repairStatus = enumeration(
      item.status,
      `${path}.repairQueue[${index}].status`,
      REPAIR_STATUSES,
    );
    validatedRepairs.push({
      id: repairId,
      sourceId,
      status: repairStatus,
    });
  });
  const evidenceDigest = string(
    report.evidenceDigest,
    `${path}.evidenceDigest`,
    { min: 1, max: 512 },
  );

  const allHardGatesPending = validatedHardGates.every(
    ({ status }) => status === "pending",
  );
  const allFacetsPending = validatedFacets.every(
    ({ status }) => status === "pending",
  );
  const hasPendingChecks =
    validatedHardGates.some(({ status }) => status === "pending") ||
    validatedFacets.some(({ status }) => status === "pending");
  const pendingEvidenceIsConsistent =
    score === 0 &&
    pageCount === 0 &&
    allHardGatesPending &&
    allFacetsPending &&
    repairQueue.length === 0 &&
    hasPendingPdfEvidenceDigest(evidenceDigest);

  let derivedStatus: PdfReportStatus;
  if (pendingEvidenceIsConsistent) {
    derivedStatus = "pending";
  } else {
    if (hasPendingChecks) {
      fail(
        path,
        "cannot mix pending PDF checks with rendered evidence or terminal checks.",
      );
    }
    if (pageCount === 0) {
      fail(
        `${path}.pageCount`,
        "must be greater than 0 after PDF evaluation begins.",
      );
    }
    if (!hasVerifiedPdfEvidenceDigest(evidenceDigest)) {
      fail(
        `${path}.evidenceDigest`,
        "must be a non-zero SHA-256 digest for every evaluated PDF report.",
      );
    }

    const unresolvedSources = new Set<PdfRepairSourceId>([
      ...validatedHardGates
        .filter(({ status }) => status === "failed")
        .map(({ id }) => id),
      ...validatedFacets
        .filter(({ status }) => status === "repair")
        .map(({ id }) => id),
    ]);
    const unresolvedRepairs = validatedRepairs.filter(
      ({ status }) => status === "open" || status === "blocked",
    );

    unresolvedRepairs.forEach(({ sourceId }, index) => {
      if (!unresolvedSources.has(sourceId)) {
        fail(
          `${path}.repairQueue[${index}].sourceId`,
          "an open or blocked repair must reference a failed hard gate or a facet marked for repair.",
        );
      }
    });
    unresolvedSources.forEach((sourceId) => {
      if (!unresolvedRepairs.some((repair) => repair.sourceId === sourceId)) {
        fail(
          `${path}.repairQueue`,
          `must contain an open or blocked repair for unresolved check ${sourceId}.`,
        );
      }
    });

    const allHardGatesPassed = validatedHardGates.every(
      ({ status }) => status === "passed",
    );
    const allFacetsPassed = validatedFacets.every(
      ({ status }) => status === "passed",
    );
    const hasOpenRepair = validatedRepairs.some(
      ({ status }) => status === "open",
    );
    const hasBlockedRepair = validatedRepairs.some(
      ({ status }) => status === "blocked",
    );
    const hasUnresolvedRepair = hasOpenRepair || hasBlockedRepair;

    if (allHardGatesPassed && allFacetsPassed && score >= minimumScore) {
      if (hasUnresolvedRepair) {
        fail(
          `${path}.repairQueue`,
          "a passing report cannot contain open or blocked repairs.",
        );
      }
      derivedStatus = "passed";
    } else {
      if (allHardGatesPassed && allFacetsPassed) {
        fail(
          `${path}.score`,
          "cannot be below the minimum when every hard gate and aesthetic facet passed.",
        );
      }
      if (unresolvedSources.size === 0 || !hasUnresolvedRepair) {
        fail(
          path,
          "a non-passing report must identify unresolved checks and their open or blocked repairs.",
        );
      }
      derivedStatus =
        passNumber < 5 && hasOpenRepair ? "repair-required" : "blocked";
    }
  }

  if (reportStatus !== derivedStatus) {
    fail(
      `${path}.status`,
      `must equal the evidence-derived fail-closed status ${derivedStatus}; received ${reportStatus}.`,
    );
  }

  if (derivedStatus === "pending") {
    if (!hasPendingPdfEvidenceDigest(evidenceDigest)) {
      fail(
        `${path}.evidenceDigest`,
        "must explicitly state that evidence is awaiting generation.",
      );
    }
    if (validatedFacets.some(({ score: facetScore }) => facetScore !== 0)) {
      fail(
        `${path}.facets`,
        "all 14 aesthetic facet scores must be 0 in a pending report.",
      );
    }
  }
}

function validatePercentMetric(value: unknown, path: string): void {
  const metric = record(value, path);
  if (metric.status === "available") {
    exactKeys(metric, path, ["status", "value", "unit", "source"]);
    finiteNumber(metric.value, `${path}.value`, { min: 0, max: 100 });
    if (metric.unit !== "percent") {
      fail(`${path}.unit`, "must be exactly percent.");
    }
    string(metric.source, `${path}.source`, { min: 1, max: 256 });
    return;
  }
  if (metric.status === "unavailable") {
    exactKeys(metric, path, ["status", "reason"]);
    string(metric.reason, `${path}.reason`, { min: 1, max: 512 });
    return;
  }
  fail(`${path}.status`, "must be available or unavailable.");
}

function validateDiarizationQuality(value: unknown, path: string): void {
  const quality = record(value, path);
  exactKeys(quality, path, [
    "der",
    "jer",
    "confusion",
    "overlapF1",
    "reviewRate",
  ]);
  validatePercentMetric(quality.der, `${path}.der`);
  validatePercentMetric(quality.jer, `${path}.jer`);
  validatePercentMetric(quality.confusion, `${path}.confusion`);
  validatePercentMetric(quality.overlapF1, `${path}.overlapF1`);
  validatePercentMetric(quality.reviewRate, `${path}.reviewRate`);
}

function validatePerformance(value: unknown, path: string): void {
  const performance = record(value, path);
  if (performance.status === "unavailable") {
    exactKeys(performance, path, ["status", "reason"]);
    string(performance.reason, `${path}.reason`, { min: 1, max: 512 });
    return;
  }
  if (performance.status !== "measured") {
    fail(`${path}.status`, "must be measured or unavailable.");
  }
  exactKeys(performance, path, [
    "status",
    "sourceLabel",
    "rtf",
    "stageLatency",
    "cacheHitRate",
    "selectiveEscalationRate",
    "recomputeRate",
    "peakResources",
  ]);
  string(performance.sourceLabel, `${path}.sourceLabel`, {
    min: 1,
    max: 256,
  });
  finiteNumber(performance.rtf, `${path}.rtf`, { min: 0 });
  finiteNumber(performance.cacheHitRate, `${path}.cacheHitRate`, {
    min: 0,
    max: 100,
  });
  finiteNumber(
    performance.selectiveEscalationRate,
    `${path}.selectiveEscalationRate`,
    { min: 0, max: 100 },
  );
  finiteNumber(performance.recomputeRate, `${path}.recomputeRate`, {
    min: 0,
    max: 100,
  });

  const seenStages = new Set<string>();
  array(performance.stageLatency, `${path}.stageLatency`).forEach(
    (stage, index) => {
      const stagePath = `${path}.stageLatency[${index}]`;
      const item = record(stage, stagePath);
      exactKeys(item, stagePath, ["stageId", "p50Ms", "p95Ms"]);
      const stageId = enumeration(item.stageId, `${stagePath}.stageId`, STAGE_IDS);
      if (seenStages.has(stageId)) {
        fail(`${stagePath}.stageId`, "must not repeat a pipeline stage.");
      }
      seenStages.add(stageId);
      const p50Ms = finiteNumber(item.p50Ms, `${stagePath}.p50Ms`, {
        min: 0,
      });
      const p95Ms = finiteNumber(item.p95Ms, `${stagePath}.p95Ms`, {
        min: 0,
      });
      if (p50Ms > p95Ms) {
        fail(stagePath, "p50Ms must not be greater than p95Ms.");
      }
    },
  );

  const resources = record(performance.peakResources, `${path}.peakResources`);
  exactKeys(resources, `${path}.peakResources`, [
    "cpuPercent",
    "ramGb",
    "vramGb",
  ]);
  finiteNumber(resources.cpuPercent, `${path}.peakResources.cpuPercent`, {
    min: 0,
    max: 100,
  });
  finiteNumber(resources.ramGb, `${path}.peakResources.ramGb`, { min: 0 });
  finiteNumber(resources.vramGb, `${path}.peakResources.vramGb`, { min: 0 });
}

export function parseStudioSnapshot(value: unknown): StudioSnapshot {
  const snapshot = record(value, "snapshot");
  exactKeys(snapshot, "snapshot", [
    "contractVersion",
    "job",
    "speakers",
    "strategies",
    "stages",
    "events",
    "reviews",
    "artifacts",
    "diarizationQuality",
    "performance",
    "pdfQuality",
    "system",
  ]);
  if (snapshot.contractVersion !== STUDIO_CONTRACT_VERSION) {
    fail(
      "snapshot.contractVersion",
      `must equal ${STUDIO_CONTRACT_VERSION}; received ${String(snapshot.contractVersion)}.`,
    );
  }

  const job = record(snapshot.job, "snapshot.job");
  exactKeys(job, "snapshot.job", [
    "id",
    "title",
    "sourcePath",
    "durationLabel",
    "status",
    "progress",
    "startedAt",
    "speakerPolicy",
    "speakerCount",
    "speakerDetection",
    "reviewOpenCount",
    "activeStrategyId",
  ]);
  string(job.id, "snapshot.job.id", { min: 1, max: 128 });
  string(job.title, "snapshot.job.title", { min: 1, max: 256 });
  string(job.sourcePath, "snapshot.job.sourcePath", { max: 4096 });
  string(job.durationLabel, "snapshot.job.durationLabel", { max: 128 });
  enumeration(job.status, "snapshot.job.status", JOB_STATUSES);
  finiteNumber(job.progress, "snapshot.job.progress", { min: 0, max: 100 });
  string(job.startedAt, "snapshot.job.startedAt", { max: 128 });
  assertSpeakerCountPolicy(job.speakerPolicy, "snapshot.job.speakerPolicy");
  const speakerCount =
    job.speakerCount === null
      ? null
      : positiveSafeInteger(job.speakerCount, "snapshot.job.speakerCount");
  const policy = job.speakerPolicy;
  let detection: SpeakerCountDetection | null = null;
  if (job.speakerDetection !== null) {
    detection = validateSpeakerCountDetection(
      job.speakerDetection,
      "snapshot.job.speakerDetection",
    );
  }
  if (policy.mode === "manual") {
    if (speakerCount === null) {
      fail(
        "snapshot.job.speakerCount",
        "manual mode must provide the explicitly requested speaker count.",
      );
    }
    if (detection !== null) {
      fail("snapshot.job.speakerDetection", "must be null in manual mode.");
    }
    if (speakerCount !== policy.count) {
      fail(
        "snapshot.job.speakerCount",
        "must match the manually specified count.",
      );
    }
  } else if (speakerCount === null || detection === null) {
    if (speakerCount !== null || detection !== null) {
      fail(
        "snapshot.job",
        `${policy.mode} mode must keep speakerCount and speakerDetection jointly unresolved until verified media evidence is available.`,
      );
    }
  } else {
    if (speakerCount !== detection.estimatedCount) {
      fail(
        "snapshot.job.speakerCount",
        "must match the detection result's estimatedCount.",
      );
    }
    if (
      policy.mode === "hybrid" &&
      (speakerCount < policy.minSpeakers || speakerCount > policy.maxSpeakers)
    ) {
      fail(
        "snapshot.job.speakerCount",
        "must be within the hybrid policy's minimum and maximum speaker counts.",
      );
    }
  }
  finiteNumber(job.reviewOpenCount, "snapshot.job.reviewOpenCount", {
    min: 0,
    integer: true,
  });
  enumeration(job.activeStrategyId, "snapshot.job.activeStrategyId", STRATEGY_IDS);

  const speakers = array(snapshot.speakers, "snapshot.speakers");
  if (speakerCount === null && speakers.length !== 0) {
    fail(
      "snapshot.speakers",
      "must remain empty while the automatic or hybrid speaker count is unresolved.",
    );
  }
  if (speakerCount !== null && speakers.length !== speakerCount) {
    fail(
      "snapshot.speakers",
      `must match speakerCount; expected ${speakerCount} speakers and received ${speakers.length}.`,
    );
  }
  speakers.forEach((speaker, index) => validateSpeaker(speaker, `snapshot.speakers[${index}]`));
  const speakerIds = new Set<SpeakerId>();
  speakers.forEach((speaker, index) => {
    const speakerId = validateSpeakerId(
      record(speaker, `snapshot.speakers[${index}]`).id,
      `snapshot.speakers[${index}].id`,
    );
    const expectedId: SpeakerId = `speaker-${index + 1}`;
    if (speakerId !== expectedId) {
      fail(
        `snapshot.speakers[${index}].id`,
        `must use contiguous sequential IDs; expected ${expectedId}.`,
      );
    }
    if (speakerIds.has(speakerId)) {
      fail("snapshot.speakers", `must not reuse ${speakerId}.`);
    }
    speakerIds.add(speakerId);
  });

  const strategies = array(snapshot.strategies, "snapshot.strategies");
  if (strategies.length === 0) {
    fail(
      "snapshot.strategies",
      "must contain at least one local processing strategy.",
    );
  }
  strategies.forEach((strategy, index) => {
    const item = record(strategy, `snapshot.strategies[${index}]`);
    exactKeys(item, `snapshot.strategies[${index}]`, [
      "id",
      "label",
      "description",
      "asrModel",
      "diarizationModel",
      "semanticModel",
      "semanticModelStatus",
      "semanticModelEvaluation",
      "estimatedVramGb",
      "semanticGuardrail",
      "recommended",
    ]);
    enumeration(item.id, `snapshot.strategies[${index}].id`, STRATEGY_IDS);
    string(item.label, `snapshot.strategies[${index}].label`, { min: 1, max: 128 });
    string(item.description, `snapshot.strategies[${index}].description`, { max: 4096 });
    string(item.asrModel, `snapshot.strategies[${index}].asrModel`, { min: 1, max: 256 });
    string(item.diarizationModel, `snapshot.strategies[${index}].diarizationModel`, {
      min: 1,
      max: 256,
    });
    string(item.semanticModel, `snapshot.strategies[${index}].semanticModel`, {
      min: 1,
      max: 256,
    });
    if (item.semanticModel !== "qwen3.5:4b") {
      fail(
        `snapshot.strategies[${index}].semanticModel`,
        "the production semantic model must remain fixed at qwen3.5:4b.",
      );
    }
    if (item.semanticModelStatus !== "reject_for_production") {
      fail(
        `snapshot.strategies[${index}].semanticModelStatus`,
        "semanticModelStatus must be reject_for_production.",
      );
    }
    string(
      item.semanticModelEvaluation,
      `snapshot.strategies[${index}].semanticModelEvaluation`,
      { min: 1, max: 4096 },
    );
    finiteNumber(item.estimatedVramGb, `snapshot.strategies[${index}].estimatedVramGb`, {
      min: 0,
      max: 1024,
    });
    string(item.semanticGuardrail, `snapshot.strategies[${index}].semanticGuardrail`, {
      min: 1,
      max: 4096,
    });
    if (item.recommended !== undefined) {
      boolean(item.recommended, `snapshot.strategies[${index}].recommended`);
    }
  });

  array(snapshot.stages, "snapshot.stages").forEach((stage, index) => {
    const item = record(stage, `snapshot.stages[${index}]`);
    exactKeys(item, `snapshot.stages[${index}]`, [
      "id",
      "label",
      "shortLabel",
      "status",
      "progress",
      "detail",
      "durationLabel",
    ]);
    enumeration(item.id, `snapshot.stages[${index}].id`, STAGE_IDS);
    string(item.label, `snapshot.stages[${index}].label`, { min: 1, max: 128 });
    string(item.shortLabel, `snapshot.stages[${index}].shortLabel`, { min: 1, max: 64 });
    enumeration(item.status, `snapshot.stages[${index}].status`, STAGE_STATUSES);
    finiteNumber(item.progress, `snapshot.stages[${index}].progress`, { min: 0, max: 100 });
    string(item.detail, `snapshot.stages[${index}].detail`, { max: 4096 });
    optionalString(item.durationLabel, `snapshot.stages[${index}].durationLabel`, 128);
  });

  array(snapshot.events, "snapshot.events").forEach((event, index) => {
    const item = record(event, `snapshot.events[${index}]`);
    exactKeys(item, `snapshot.events[${index}]`, [
      "id",
      "sequence",
      "type",
      "stageId",
      "severity",
      "timestamp",
      "title",
      "detail",
    ]);
    string(item.id, `snapshot.events[${index}].id`, { min: 1, max: 128 });
    finiteNumber(item.sequence, `snapshot.events[${index}].sequence`, { min: 0, integer: true });
    enumeration(item.type, `snapshot.events[${index}].type`, EVENT_TYPES);
    if (item.stageId !== undefined) {
      enumeration(item.stageId, `snapshot.events[${index}].stageId`, STAGE_IDS);
    }
    enumeration(item.severity, `snapshot.events[${index}].severity`, SEVERITIES);
    string(item.timestamp, `snapshot.events[${index}].timestamp`, { max: 128 });
    string(item.title, `snapshot.events[${index}].title`, { min: 1, max: 256 });
    string(item.detail, `snapshot.events[${index}].detail`, { max: 4096 });
  });

  const reviews = array(snapshot.reviews, "snapshot.reviews");
  reviews.forEach((review, index) =>
    validateReviewSegment(review, `snapshot.reviews[${index}]`, speakerIds),
  );
  const openReviewCount = reviews.filter(
    (review, index) =>
      record(review, `snapshot.reviews[${index}]`).reviewed === false,
  ).length;
  if (job.reviewOpenCount !== openReviewCount) {
    fail(
      "snapshot.job.reviewOpenCount",
      `must match the number of unreviewed segments; expected ${openReviewCount}.`,
    );
  }

  array(snapshot.artifacts, "snapshot.artifacts").forEach((artifact, index) => {
    const item = record(artifact, `snapshot.artifacts[${index}]`);
    exactKeys(item, `snapshot.artifacts[${index}]`, [
      "id",
      "name",
      "kind",
      "relativePath",
      "sizeLabel",
      "createdAt",
      "integrity",
      "sha256",
    ]);
    string(item.id, `snapshot.artifacts[${index}].id`, { min: 1, max: 128 });
    string(item.name, `snapshot.artifacts[${index}].name`, { min: 1, max: 512 });
    enumeration(item.kind, `snapshot.artifacts[${index}].kind`, ARTIFACT_KINDS);
    assertSafeArtifactRelativePath(
      item.relativePath,
      `snapshot.artifacts[${index}].relativePath`,
    );
    string(item.sizeLabel, `snapshot.artifacts[${index}].sizeLabel`, { max: 128 });
    string(item.createdAt, `snapshot.artifacts[${index}].createdAt`, { max: 128 });
    enumeration(item.integrity, `snapshot.artifacts[${index}].integrity`, INTEGRITY_STATUSES);
    optionalString(item.sha256, `snapshot.artifacts[${index}].sha256`, 256);
  });

  validateDiarizationQuality(
    snapshot.diarizationQuality,
    "snapshot.diarizationQuality",
  );
  validatePerformance(snapshot.performance, "snapshot.performance");
  validatePdfQuality(snapshot.pdfQuality, "snapshot.pdfQuality");
  const system = record(snapshot.system, "snapshot.system");
  exactKeys(system, "snapshot.system", [
    "offline",
    "backendMode",
    "gpuLabel",
    "vramLabel",
    "inferenceWorker",
    "javaRenderer",
  ]);
  if (system.offline !== true) {
    fail("snapshot.system.offline", "must be exactly true.");
  }
  enumeration(system.backendMode, "snapshot.system.backendMode", BACKEND_MODES);
  string(system.gpuLabel, "snapshot.system.gpuLabel", { max: 256 });
  string(system.vramLabel, "snapshot.system.vramLabel", { max: 128 });
  enumeration(
    system.inferenceWorker,
    "snapshot.system.inferenceWorker",
    WORKER_STATUSES,
  );
  enumeration(system.javaRenderer, "snapshot.system.javaRenderer", WORKER_STATUSES);

  return value as StudioSnapshot;
}

export function assertCreateJobRequest(value: unknown): asserts value is CreateJobRequest {
  const request = record(value, "createJobRequest");
  exactKeys(request, "createJobRequest", [
    "title",
    "mediaPath",
    "outputDirectory",
    "strategyId",
    "speakerPolicy",
    "speakerLabels",
    "language",
    "localLlmMode",
    "localLlmModel",
    "localLlmEndpoint",
    "localLlmEndpointPolicy",
    "localLlmAutoApply",
    "translationTargets",
    "polish",
    "summary",
    "outputLocale",
    "businessPromptVersion",
  ]);
  const title = string(request.title, "createJobRequest.title", { min: 1, max: 80 });
  ensureNoControlCharacters(title, "createJobRequest.title");
  const mediaPath = string(request.mediaPath, "createJobRequest.mediaPath", {
    min: 1,
    max: 4096,
  });
  ensureNoControlCharacters(mediaPath, "createJobRequest.mediaPath");
  const outputDirectory = string(
    request.outputDirectory,
    "createJobRequest.outputDirectory",
    { min: 1, max: 4096 },
  );
  ensureNoControlCharacters(outputDirectory, "createJobRequest.outputDirectory");
  enumeration(request.strategyId, "createJobRequest.strategyId", STRATEGY_IDS);
  assertSpeakerCountPolicy(request.speakerPolicy, "createJobRequest.speakerPolicy");
  const labels = array(request.speakerLabels, "createJobRequest.speakerLabels");
  const policy = request.speakerPolicy;
  const expectedLabelCount =
    policy.mode === "auto"
      ? 0
      : policy.mode === "manual"
        ? policy.count
        : policy.priorCount;
  const labelsAreComplete =
    labels.length === expectedLabelCount ||
    (policy.mode !== "auto" && labels.length === 0);
  if (!labelsAreComplete) {
    fail(
      "createJobRequest.speakerLabels",
      policy.mode === "auto"
        ? `Automatic mode accepts only an empty label list; received ${labels.length}.`
        : `This policy accepts either an empty label list or exactly ${expectedLabelCount} speaker labels; received ${labels.length}.`,
    );
  }
  labels.forEach((label, index) => {
    const text = string(label, `createJobRequest.speakerLabels[${index}]`, {
      min: 1,
      max: 32,
    });
    ensureNoControlCharacters(text, `createJobRequest.speakerLabels[${index}]`);
  });
  languageTag(request.language, "createJobRequest.language", { allowAuto: true });
  const localLlmMode = enumeration(
    request.localLlmMode,
    "createJobRequest.localLlmMode",
    LOCAL_LLM_MODES,
  );
  const localLlmModel = string(
    request.localLlmModel,
    "createJobRequest.localLlmModel",
    { min: 1, max: 256 },
  );
  ensureNoControlCharacters(localLlmModel, "createJobRequest.localLlmModel");
  loopbackEndpoint(
    request.localLlmEndpoint,
    "createJobRequest.localLlmEndpoint",
  );
  if (request.localLlmEndpointPolicy !== "loopback-only") {
    fail(
      "createJobRequest.localLlmEndpointPolicy",
      'must be exactly "loopback-only".',
    );
  }
  if (boolean(request.localLlmAutoApply, "createJobRequest.localLlmAutoApply")) {
    fail(
      "createJobRequest.localLlmAutoApply",
      "must remain false so business outputs never overwrite the immutable transcript.",
    );
  }
  const translationTargets = array(
    request.translationTargets,
    "createJobRequest.translationTargets",
  );
  const seenTargets = new Set<string>();
  translationTargets.forEach((target, index) => {
    const tag = languageTag(
      target,
      `createJobRequest.translationTargets[${index}]`,
    );
    const comparisonKey = tag.toLocaleLowerCase("en-US");
    if (seenTargets.has(comparisonKey)) {
      fail(
        `createJobRequest.translationTargets[${index}]`,
        "duplicates another target language (comparison is case-insensitive).",
      );
    }
    seenTargets.add(comparisonKey);
  });
  const polish = boolean(request.polish, "createJobRequest.polish");
  const summary = boolean(request.summary, "createJobRequest.summary");
  languageTag(request.outputLocale, "createJobRequest.outputLocale");
  if (request.businessPromptVersion !== "business-v1") {
    fail(
      "createJobRequest.businessPromptVersion",
      'must be exactly "business-v1".',
    );
  }
  const hasBusinessTask =
    translationTargets.length > 0 || polish || summary;
  if (
    (localLlmMode === "business" && !hasBusinessTask) ||
    (localLlmMode === "disabled" && hasBusinessTask)
  ) {
    fail(
      "createJobRequest.localLlmMode",
      hasBusinessTask
        ? 'must be "business" when translation, polishing, or summary output is selected.'
        : 'must be "disabled" when no business-processing output is selected.',
    );
  }
}

export function assertReviewDecision(value: unknown): asserts value is ReviewDecision {
  const decision = record(value, "reviewDecision");
  exactKeys(decision, "reviewDecision", [
    "reviewId",
    "speakerId",
    "normalizedText",
    "reason",
    "evidence",
    "confidence",
  ]);
  string(decision.reviewId, "reviewDecision.reviewId", { min: 1, max: 128 });
  validateSpeakerId(decision.speakerId, "reviewDecision.speakerId");
  string(decision.normalizedText, "reviewDecision.normalizedText", {
    min: 1,
    max: 100_000,
  });
  nonBlankAuditText(decision.reason, "reviewDecision.reason");
  nonBlankAuditText(decision.evidence, "reviewDecision.evidence");
  finiteNumber(decision.confidence, "reviewDecision.confidence", { min: 0, max: 1 });
}

export function assertUpdateSpeakerRequest(
  value: unknown,
): asserts value is UpdateSpeakerRequest {
  const request = record(value, "updateSpeakerRequest");
  exactKeys(request, "updateSpeakerRequest", [
    "speakerId",
    "label",
    "locked",
    "reviewStatus",
  ]);
  validateSpeakerId(request.speakerId, "updateSpeakerRequest.speakerId");
  const label = string(request.label, "updateSpeakerRequest.label", {
    min: 1,
    max: 64,
  });
  ensureNoControlCharacters(label, "updateSpeakerRequest.label");
  boolean(request.locked, "updateSpeakerRequest.locked");
  enumeration(
    request.reviewStatus,
    "updateSpeakerRequest.reviewStatus",
    SPEAKER_REVIEW_STATUSES,
  );
}

export function parseCreateJobResult(value: unknown): CreateJobResult {
  const result = record(value, "createJobResult");
  exactKeys(result, "createJobResult", ["accepted", "jobId", "message"]);
  boolean(result.accepted, "createJobResult.accepted");
  string(result.jobId, "createJobResult.jobId", { min: 1, max: 128 });
  string(result.message, "createJobResult.message", { min: 1, max: 4096 });
  return value as CreateJobResult;
}

export function parseReviewSegment(value: unknown): ReviewSegment {
  validateReviewSegment(value, "reviewSegment");
  return value as ReviewSegment;
}

export function parseSpeakerProfile(value: unknown): SpeakerProfile {
  validateSpeaker(value, "speakerProfile");
  return value as SpeakerProfile;
}

export function parseArtifactOpenResult(value: unknown): ArtifactOpenResult {
  const result = record(value, "artifactOpenResult");
  exactKeys(result, "artifactOpenResult", [
    "artifactId",
    "canonicalPath",
    "opened",
    "message",
  ]);
  string(result.artifactId, "artifactOpenResult.artifactId", { min: 1, max: 128 });
  const canonicalPath = string(result.canonicalPath, "artifactOpenResult.canonicalPath", {
    min: 1,
    max: 4096,
  });
  ensureNoControlCharacters(canonicalPath, "artifactOpenResult.canonicalPath");
  boolean(result.opened, "artifactOpenResult.opened");
  string(result.message, "artifactOpenResult.message", { min: 1, max: 4096 });
  return value as ArtifactOpenResult;
}

export function parseReviewCommandStatus(value: unknown): ReviewCommandStatus {
  const status = record(value, "reviewCommandStatus");
  exactKeys(status, "reviewCommandStatus", [
    "command",
    "ipcCommand",
    "phase",
    "label",
    "message",
    "backendMode",
    "startedAtUnixMs",
    "completedAtUnixMs",
  ]);
  const command = enumeration(
    status.command,
    "reviewCommandStatus.command",
    REVIEW_COMMAND_KINDS,
  );
  const definition = REVIEW_COMMAND_DEFINITIONS[command];
  if (status.ipcCommand !== definition.ipcCommand) {
    fail(
      "reviewCommandStatus.ipcCommand",
      `command ${command} must map to ${definition.ipcCommand}.`,
    );
  }
  if (status.label !== definition.label) {
    fail(
      "reviewCommandStatus.label",
      `command ${command} must use the label "${definition.label}".`,
    );
  }
  const phase = enumeration(
    status.phase,
    "reviewCommandStatus.phase",
    REVIEW_COMMAND_PHASES,
  );
  string(status.message, "reviewCommandStatus.message", { min: 1, max: 4096 });
  enumeration(
    status.backendMode,
    "reviewCommandStatus.backendMode",
    REVIEW_COMMAND_BACKEND_MODES,
  );
  const startedAtUnixMs =
    status.startedAtUnixMs === null
      ? null
      : finiteNumber(
          status.startedAtUnixMs,
          "reviewCommandStatus.startedAtUnixMs",
          { min: 0, integer: true },
        );
  const completedAtUnixMs =
    status.completedAtUnixMs === null
      ? null
      : finiteNumber(
          status.completedAtUnixMs,
          "reviewCommandStatus.completedAtUnixMs",
          { min: 0, integer: true },
        );
  if (phase === "idle") {
    if (startedAtUnixMs !== null || completedAtUnixMs !== null) {
      fail("reviewCommandStatus", "must not contain execution times while idle.");
    }
  } else {
    if (startedAtUnixMs === null) {
      fail(
        "reviewCommandStatus.startedAtUnixMs",
        `must record a start time while ${phase}.`,
      );
    }
    if (phase === "pending" || phase === "running") {
      if (completedAtUnixMs !== null) {
        fail(
          "reviewCommandStatus.completedAtUnixMs",
          `must not record a completion time while ${phase}.`,
        );
      }
    } else if (completedAtUnixMs === null) {
      fail(
        "reviewCommandStatus.completedAtUnixMs",
        `must record a completion time while ${phase}.`,
      );
    } else if (completedAtUnixMs < startedAtUnixMs) {
      fail(
        "reviewCommandStatus.completedAtUnixMs",
        "must not be earlier than the start time.",
      );
    }
  }
  return value as ReviewCommandStatus;
}

export function assertVoidResult(value: unknown, command: string): void {
  if (value !== null && value !== undefined) {
    fail(command, "must return no value.");
  }
}

export function parseIpcError(value: unknown): Error {
  const generic = () =>
    new Error(
      "The Tauri IPC call failed without a recognizable structured backend error.",
    );
  const safeMessage = (message: unknown): message is string =>
    typeof message === "string" &&
    message.length >= 1 &&
    message.length <= 4096 &&
    !/[\u0000-\u001f\u007f]/u.test(message);

  if (value instanceof Error) {
    const code = (value as Error & { code?: unknown }).code;
    if (
      typeof code === "string" &&
      IPC_ERROR_CODES.includes(code as IpcErrorCode) &&
      safeMessage(value.message)
    ) {
      return new Error(`[${code}] ${value.message}`);
    }
    return safeMessage(value.message) ? new Error(value.message) : generic();
  }
  if (typeof value === "string") {
    return safeMessage(value) ? new Error(value) : generic();
  }
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    const candidate = value as UnknownRecord;
    const keys = Object.keys(candidate);
    if (
      keys.length === 2 &&
      Object.prototype.hasOwnProperty.call(candidate, "code") &&
      Object.prototype.hasOwnProperty.call(candidate, "message") &&
      typeof candidate.code === "string" &&
      IPC_ERROR_CODES.includes(candidate.code as IpcErrorCode) &&
      safeMessage(candidate.message)
    ) {
      return new Error(`[${candidate.code}] ${candidate.message}`);
    }
  }
  return generic();
}
