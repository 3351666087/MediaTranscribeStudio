mod job_registry;
mod worker_supervisor;

use job_registry::{
    DispatchPermit, IdempotencyKey, JobId, JobRegistry, RegisterOutcome, RegistryError,
    RegistryJobStatus, MAX_DISPATCH_LIMIT,
};
use serde::{
    de::{Error as DeError, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    fs::{self, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tauri::{Emitter, Manager, State};
use worker_supervisor::{
    ResponseKind, WorkerError, WorkerEvent, WorkerNotification, WorkerResponse, WorkerSupervisor,
};

const CONTRACT_VERSION: &str = "1.6.0";
const JS_MAX_SAFE_INTEGER: usize = 9_007_199_254_740_991usize;
const LOCAL_LLM_LOOPBACK_POLICY: &str = "loopback-only";
const LOCAL_LLM_REMOTE_POLICY: &str = "remote-explicit";
const DEFAULT_LLM_PROVIDER: &str = "ollama-loopback";
const PRODUCTION_LOCAL_LLM_MODEL: &str = "qwen3.5:27b-q4_K_M";
const BUSINESS_PROMPT_VERSION: &str = "business-v1";
const MAX_OUTPUT_CUSTOMIZATION_BYTES: usize = 256 * 1024;
static NEXT_JOB_SEQUENCE: AtomicU64 = AtomicU64::new(1);
static NEXT_MUTATION_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum IpcErrorCode {
    InvalidRequest,
    InvalidPath,
    NotFound,
    Conflict,
    PathBoundaryViolation,
    StateUnavailable,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct IpcError {
    code: IpcErrorCode,
    message: String,
}

impl IpcError {
    fn new(code: IpcErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

type IpcResult<T> = Result<T, IpcError>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum ModelStrategyId {
    Balanced,
    Quality,
    MemorySaver,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum LocalLlmMode {
    Disabled,
    Business,
}

impl LocalLlmMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Business => "business",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum JobStatus {
    Draft,
    Registered,
    Queued,
    Running,
    ReviewRequired,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StageStatus {
    Pending,
    Running,
    Completed,
    Warning,
    Blocked,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Severity {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ConfidenceBand {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReviewReason {
    SpeakerCloseScore,
    SpeakerCountUncertain,
    OverlapDetected,
    TimestampBoundary,
    SpeakerOutlier,
    LocalAudioReview,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SampleStatus {
    Ready,
    Missing,
    NeedsReview,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SpeakerReviewStatus {
    Pending,
    NeedsReview,
    Confirmed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum ArtifactKind {
    TranscriptJson,
    TranscriptText,
    Pdf,
    PageImage,
    ContactSheet,
    QualityReport,
    RepairQueue,
    Log,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum IntegrityStatus {
    Verified,
    Pending,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum PdfStatus {
    Pending,
    Passed,
    RepairRequired,
    Blocked,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum GateStatus {
    Pending,
    Passed,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum FacetStatus {
    Pending,
    Passed,
    Repair,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RepairStatus {
    Open,
    Applied,
    Blocked,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RepairScope {
    Template,
    Css,
    Font,
    Image,
    Pagination,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum BackendMode {
    Mock,
    TauriIpc,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WorkerStatus {
    Ready,
    Missing,
    Busy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum SpeakerCountPolicy {
    Auto {},
    Manual {
        count: usize,
    },
    Hybrid {
        #[serde(rename = "minSpeakers")]
        min_speakers: usize,
        #[serde(rename = "maxSpeakers")]
        max_speakers: usize,
        #[serde(rename = "priorCount")]
        prior_count: usize,
    },
}

impl SpeakerCountPolicy {
    fn validate(&self) -> IpcResult<()> {
        match self {
            Self::Auto {} => Ok(()),
            Self::Manual { count } => validate_speaker_count(*count, "speakerPolicy.count"),
            Self::Hybrid {
                min_speakers,
                max_speakers,
                prior_count,
            } => {
                validate_speaker_count(*min_speakers, "speakerPolicy.minSpeakers")?;
                validate_speaker_count(*max_speakers, "speakerPolicy.maxSpeakers")?;
                validate_speaker_count(*prior_count, "speakerPolicy.priorCount")?;
                if min_speakers > max_speakers {
                    return Err(IpcError::new(
                        IpcErrorCode::InvalidRequest,
                        "speakerPolicy.minSpeakers must not be greater than maxSpeakers.",
                    ));
                }
                if prior_count < min_speakers || prior_count > max_speakers {
                    return Err(IpcError::new(
                        IpcErrorCode::InvalidRequest,
                        "speakerPolicy.priorCount must be between minSpeakers and maxSpeakers.",
                    ));
                }
                Ok(())
            }
        }
    }

    fn expected_label_count(&self) -> usize {
        match self {
            Self::Auto {} => 0,
            Self::Manual { count } => *count,
            Self::Hybrid { prior_count, .. } => *prior_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SpeakerCountCandidate {
    count: usize,
    confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SpeakerCountDetection {
    estimated_count: usize,
    confidence: f32,
    candidates: Vec<SpeakerCountCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SpeakerProfile {
    id: String,
    label: String,
    short_label: String,
    color: String,
    role_hint: String,
    sample_status: SampleStatus,
    locked: bool,
    review_status: SpeakerReviewStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelStrategy {
    id: ModelStrategyId,
    label: String,
    description: String,
    asr_model: String,
    diarization_model: String,
    semantic_model: String,
    semantic_model_status: String,
    semantic_model_evaluation: String,
    estimated_vram_gb: f32,
    semantic_guardrail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    recommended: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PipelineStage {
    id: String,
    label: String,
    short_label: String,
    status: StageStatus,
    progress: u8,
    detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StudioEvent {
    id: String,
    sequence: u64,
    #[serde(rename = "type")]
    event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage_id: Option<String>,
    severity: Severity,
    timestamp: String,
    title: String,
    detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SpeakerCandidate {
    speaker_id: String,
    score: f32,
    evidence: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReviewActor {
    Human,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReviewAuditEvent {
    id: String,
    sequence: usize,
    recorded_at_unix_ms: u64,
    actor: ReviewActor,
    reason: String,
    evidence: String,
    confidence: f32,
    previous_speaker_id: String,
    speaker_id: String,
    previous_normalized_text: String,
    normalized_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReviewSegment {
    id: String,
    start_ms: u64,
    end_ms: u64,
    timestamp_label: String,
    raw_text: String,
    normalized_text: String,
    current_speaker_id: String,
    candidates: Vec<SpeakerCandidate>,
    reasons: Vec<ReviewReason>,
    confidence: f32,
    confidence_band: ConfidenceBand,
    waveform: Vec<f32>,
    locked: bool,
    reviewed: bool,
    audit_trail: Vec<ReviewAuditEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactItem {
    id: String,
    name: String,
    kind: ArtifactKind,
    relative_path: String,
    size_label: String,
    created_at: String,
    integrity: IntegrityStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PdfHardGate {
    id: String,
    label: String,
    status: GateStatus,
    detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AestheticFacet {
    id: String,
    label: String,
    score: f32,
    status: FacetStatus,
    evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RepairQueueItem {
    id: String,
    priority: u8,
    source_id: String,
    title: String,
    detail: String,
    safe_scope: RepairScope,
    status: RepairStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PdfQualityReport {
    status: PdfStatus,
    pass_number: u8,
    score: f32,
    minimum_score: u8,
    page_count: u32,
    rendered_at: String,
    hard_gates: Vec<PdfHardGate>,
    facets: Vec<AestheticFacet>,
    repair_queue: Vec<RepairQueueItem>,
    evidence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JobSummary {
    id: String,
    title: String,
    source_path: String,
    duration_label: String,
    status: JobStatus,
    progress: u8,
    started_at: String,
    speaker_policy: SpeakerCountPolicy,
    speaker_count: Option<usize>,
    speaker_detection: Option<SpeakerCountDetection>,
    review_open_count: usize,
    active_strategy_id: ModelStrategyId,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PercentUnit {
    Percent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "status",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum PercentMetric {
    Available {
        value: f32,
        unit: PercentUnit,
        source: String,
    },
    Unavailable {
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DiarizationQualityMetrics {
    der: PercentMetric,
    jer: PercentMetric,
    confusion: PercentMetric,
    overlap_f1: PercentMetric,
    review_rate: PercentMetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StagePerformance {
    stage_id: String,
    p50_ms: f32,
    p95_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PeakResources {
    cpu_percent: f32,
    ram_gb: f32,
    vram_gb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "status",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum PerformanceMetrics {
    Measured {
        source_label: String,
        rtf: f32,
        stage_latency: Vec<StagePerformance>,
        cache_hit_rate: f32,
        selective_escalation_rate: f32,
        recompute_rate: f32,
        peak_resources: PeakResources,
    },
    Unavailable {
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SystemStatus {
    offline: bool,
    backend_mode: BackendMode,
    gpu_label: String,
    vram_label: String,
    inference_worker: WorkerStatus,
    java_renderer: WorkerStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StudioSnapshot {
    contract_version: String,
    job: JobSummary,
    speakers: Vec<SpeakerProfile>,
    strategies: Vec<ModelStrategy>,
    stages: Vec<PipelineStage>,
    events: Vec<StudioEvent>,
    reviews: Vec<ReviewSegment>,
    artifacts: Vec<ArtifactItem>,
    diarization_quality: DiarizationQualityMetrics,
    performance: PerformanceMetrics,
    pdf_quality: PdfQualityReport,
    system: SystemStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CreateJobRequest {
    #[serde(default)]
    idempotency_key: Option<String>,
    title: String,
    media_path: PathBuf,
    output_directory: PathBuf,
    strategy_id: ModelStrategyId,
    speaker_policy: SpeakerCountPolicy,
    speaker_labels: Vec<String>,
    language: String,
    local_llm_mode: LocalLlmMode,
    local_llm_model: String,
    local_llm_endpoint: String,
    local_llm_endpoint_policy: String,
    #[serde(default)]
    llm_provider: Option<String>,
    #[serde(default)]
    llm_api_key_env: Option<String>,
    #[serde(default)]
    llm_proxy_url: Option<String>,
    local_llm_auto_apply: bool,
    translation_targets: Vec<String>,
    summary: bool,
    output_locale: String,
    business_prompt_version: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_output_customization"
    )]
    output_customization: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CreateJobResult {
    accepted: bool,
    job_id: String,
    replayed: bool,
    status: JobStatus,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobRuntimeStatus {
    job_id: String,
    status: JobStatus,
    revision: u64,
    accepted_by_worker: bool,
    projected: bool,
    in_flight: bool,
    worker_event_route_registered: bool,
    cancellable: bool,
    volatile_only: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobSnapshotUpdate {
    job_id: String,
    status: JobStatus,
    revision: u64,
    projected: bool,
    snapshot: StudioSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReviewDecision {
    review_id: String,
    speaker_id: String,
    normalized_text: String,
    reason: String,
    evidence: String,
    confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct UpdateSpeakerRequest {
    speaker_id: String,
    label: String,
    locked: bool,
    review_status: SpeakerReviewStatus,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactOpenResult {
    artifact_id: String,
    canonical_path: String,
    opened: bool,
    message: String,
}

#[derive(Debug, Clone)]
struct AppState {
    snapshot: StudioSnapshot,
    output_root: Option<PathBuf>,
    worker_evidence: WorkerEvidenceLedger,
    request_fingerprint: String,
    worker_accepted: bool,
}

#[derive(Debug, Clone, Default)]
struct WorkerEvidenceLedger {
    verified_artifacts: BTreeMap<PathBuf, VerifiedArtifactEvidence>,
    worker_review_open_count: Option<usize>,
    open_review_segment_ids: BTreeSet<String>,
    open_review_segment_item_count: usize,
    pipeline_segment_count: Option<usize>,
    pipeline_duration_ms: Option<u64>,
    transcript_duration_ms: Option<u64>,
    transcript_segment_count: Option<usize>,
    transcript_speaker_count_mode: Option<String>,
    transcript_speaker_count_estimate: Option<Value>,
    transcript_segments: BTreeMap<String, TranscriptSegmentEvidence>,
    reference_quality: Option<ReferenceQualityEvidence>,
}

#[derive(Debug, Clone)]
struct VerifiedArtifactEvidence {
    artifact_type: String,
    canonical_path: PathBuf,
    sha256: String,
}

#[derive(Debug, Clone)]
struct TranscriptSegmentEvidence {
    id: String,
    start_ms: u64,
    end_ms: u64,
    speaker_id: String,
    raw_text: String,
    normalized_text: String,
    display_text: String,
    confidence: f32,
    speaker_scores: Vec<TranscriptSpeakerScoreEvidence>,
    human_locked: bool,
}

#[derive(Debug, Clone)]
struct TranscriptSpeakerScoreEvidence {
    speaker_id: String,
    score: f32,
}

#[derive(Debug, Clone, Copy)]
struct ReferenceQualityEvidence {
    der: f32,
    jer: f32,
    speaker_confusion: f32,
    overlap_f1: f32,
}

struct StudioStore {
    registry: JobRegistry<AppState>,
    draft_snapshot: StudioSnapshot,
    dispatch_permits: Mutex<HashMap<JobId, DispatchPermit>>,
}

#[derive(Default)]
struct JobCommandGate(Mutex<HashMap<JobId, Arc<tokio::sync::Mutex<()>>>>);

#[derive(Debug, Clone)]
struct PreparedJob {
    idempotency_key: Option<String>,
    title: String,
    media_path: PathBuf,
    output_directory: PathBuf,
    strategy_id: ModelStrategyId,
    speaker_policy: SpeakerCountPolicy,
    speaker_labels: Vec<String>,
    language: String,
    local_llm_mode: LocalLlmMode,
    local_llm_model: String,
    local_llm_endpoint: String,
    local_llm_endpoint_policy: String,
    llm_provider: String,
    llm_api_key_env: Option<String>,
    llm_proxy_url: Option<String>,
    local_llm_auto_apply: bool,
    translation_targets: Vec<String>,
    summary: bool,
    output_locale: String,
    business_prompt_version: String,
    output_customization: Option<Map<String, Value>>,
}

fn deserialize_optional_output_customization<'de, D>(
    deserializer: D,
) -> Result<Option<Map<String, Value>>, D::Error>
where
    D: Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::Object(customization) => Ok(Some(customization)),
        _ => Err(D::Error::custom(
            "outputCustomization must be a JSON object when provided",
        )),
    }
}

fn validate_speaker_count(value: usize, field: &str) -> IpcResult<()> {
    if value == 0 || value > JS_MAX_SAFE_INTEGER {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must be a positive safe integer."),
        ));
    }
    Ok(())
}

fn validate_text(value: &str, field: &str, max_chars: usize, allow_empty: bool) -> IpcResult<()> {
    let trimmed = value.trim();
    if (!allow_empty && trimmed.is_empty()) || trimmed.chars().count() > max_chars {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} has an invalid length."),
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must not contain control characters."),
        ));
    }
    Ok(())
}

fn is_ascii_alpha(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_alphabetic())
}

fn is_ascii_alphanumeric(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn is_variant_subtag(value: &str) -> bool {
    (value.len() >= 5 && value.len() <= 8 && is_ascii_alphanumeric(value))
        || (value.len() == 4
            && value.as_bytes().first().is_some_and(u8::is_ascii_digit)
            && is_ascii_alphanumeric(value))
}

fn validate_language_tag(value: &str, field: &str, allow_auto: bool) -> IpcResult<()> {
    validate_text(value, field, 128, false)?;
    if allow_auto && value == "auto" {
        return Ok(());
    }
    if value == "auto" {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must be a concrete BCP-47 language tag; auto is not allowed."),
        ));
    }
    if !value.is_ascii() || value.starts_with('-') || value.ends_with('-') || value.contains("--") {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must be a practical BCP-47 language tag."),
        ));
    }

    let subtags: Vec<&str> = value.split('-').collect();
    if subtags.is_empty() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must be a practical BCP-47 language tag."),
        ));
    }

    if subtags[0].eq_ignore_ascii_case("x") {
        if subtags.len() < 2
            || subtags[1..].iter().any(|subtag| {
                subtag.is_empty() || subtag.len() > 8 || !is_ascii_alphanumeric(subtag)
            })
        {
            return Err(IpcError::new(
                IpcErrorCode::InvalidRequest,
                format!("{field} must be a practical BCP-47 language tag."),
            ));
        }
        return Ok(());
    }

    let primary = subtags[0];
    if primary.len() < 2 || primary.len() > 8 || !is_ascii_alpha(primary) {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must be a practical BCP-47 language tag."),
        ));
    }

    let mut index = 1;
    let mut extlang_count = 0;
    while extlang_count < 3
        && subtags
            .get(index)
            .is_some_and(|subtag| subtag.len() == 3 && is_ascii_alpha(subtag))
    {
        index += 1;
        extlang_count += 1;
    }

    if subtags
        .get(index)
        .is_some_and(|subtag| subtag.len() == 4 && is_ascii_alpha(subtag))
    {
        index += 1;
    }

    if subtags.get(index).is_some_and(|subtag| {
        (subtag.len() == 2 && is_ascii_alpha(subtag))
            || (subtag.len() == 3 && subtag.bytes().all(|byte| byte.is_ascii_digit()))
    }) {
        index += 1;
    }

    let mut variants = std::collections::HashSet::new();
    while let Some(subtag) = subtags.get(index) {
        if !is_variant_subtag(subtag) {
            break;
        }
        let normalized = subtag.to_ascii_lowercase();
        if !variants.insert(normalized) {
            return Err(IpcError::new(
                IpcErrorCode::InvalidRequest,
                format!("{field} contains a duplicate variant subtag."),
            ));
        }
        index += 1;
    }

    let mut extension_singletons = std::collections::HashSet::new();
    while let Some(singleton) = subtags.get(index) {
        if singleton.eq_ignore_ascii_case("x") {
            break;
        }
        if singleton.len() != 1
            || !is_ascii_alphanumeric(singleton)
            || singleton.eq_ignore_ascii_case("x")
        {
            break;
        }
        let normalized = singleton.to_ascii_lowercase();
        if !extension_singletons.insert(normalized) {
            return Err(IpcError::new(
                IpcErrorCode::InvalidRequest,
                format!("{field} contains a duplicate extension singleton."),
            ));
        }
        index += 1;
        let extension_start = index;
        while subtags.get(index).is_some_and(|subtag| {
            subtag.len() >= 2 && subtag.len() <= 8 && is_ascii_alphanumeric(subtag)
        }) {
            index += 1;
        }
        if index == extension_start {
            return Err(IpcError::new(
                IpcErrorCode::InvalidRequest,
                format!("{field} contains an extension without any value subtags."),
            ));
        }
    }

    if subtags
        .get(index)
        .is_some_and(|subtag| subtag.eq_ignore_ascii_case("x"))
    {
        index += 1;
        let private_start = index;
        while subtags.get(index).is_some_and(|subtag| {
            !subtag.is_empty() && subtag.len() <= 8 && is_ascii_alphanumeric(subtag)
        }) {
            index += 1;
        }
        if index == private_start {
            return Err(IpcError::new(
                IpcErrorCode::InvalidRequest,
                format!("{field} contains an empty private-use section."),
            ));
        }
    }

    if index != subtags.len() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must be a practical BCP-47 language tag."),
        ));
    }
    Ok(())
}

fn validate_llm_endpoint(value: &str, field: &str, policy: &str) -> IpcResult<()> {
    validate_text(value, field, 2_048, false)?;
    if policy != LOCAL_LLM_LOOPBACK_POLICY && policy != LOCAL_LLM_REMOTE_POLICY {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!(
                "{field} policy must be {LOCAL_LLM_LOOPBACK_POLICY} or {LOCAL_LLM_REMOTE_POLICY}."
            ),
        ));
    }
    let (scheme, remainder) = value.split_once("://").ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must be an absolute HTTP(S) URL."),
        )
    })?;
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must use HTTP or HTTPS."),
        ));
    }
    if remainder.contains('?') || remainder.contains('#') {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must not contain a query or fragment."),
        ));
    }
    let authority = remainder
        .split(['/', '?', '#'])
        .next()
        .filter(|authority| !authority.is_empty())
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::InvalidRequest,
                format!("{field} must include a loopback host."),
            )
        })?;
    if authority.contains('@') {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must not include credentials."),
        ));
    }

    let (host, port) = if let Some(ipv6) = authority.strip_prefix('[') {
        let closing = ipv6.find(']').ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::InvalidRequest,
                format!("{field} contains an invalid IPv6 host."),
            )
        })?;
        let host = &ipv6[..closing];
        let suffix = &ipv6[closing + 1..];
        let port = if suffix.is_empty() {
            None
        } else {
            Some(suffix.strip_prefix(':').ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::InvalidRequest,
                    format!("{field} contains an invalid authority."),
                )
            })?)
        };
        (host, port)
    } else if let Some((host, port)) = authority.rsplit_once(':') {
        (host, Some(port))
    } else {
        (authority, None)
    };

    let loopback = host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1";
    if policy == LOCAL_LLM_LOOPBACK_POLICY && !loopback {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must use localhost, 127.0.0.1, or [::1]."),
        ));
    }
    if policy == LOCAL_LLM_REMOTE_POLICY && !scheme.eq_ignore_ascii_case("https") && !loopback {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} remote endpoints must use HTTPS."),
        ));
    }
    if let Some(port) = port {
        let valid_port = !port.is_empty()
            && port.bytes().all(|byte| byte.is_ascii_digit())
            && port.parse::<u16>().is_ok_and(|value| value > 0);
        if !valid_port {
            return Err(IpcError::new(
                IpcErrorCode::InvalidRequest,
                format!("{field} contains an invalid port."),
            ));
        }
    }
    Ok(())
}

fn validate_provider_id(value: &str, field: &str) -> IpcResult<()> {
    validate_text(value, field, 96, false)?;
    let mut bytes = value.trim().bytes();
    let valid_first = bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphanumeric());
    let valid_rest =
        bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if !valid_first || !valid_rest {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} contains an invalid provider identifier."),
        ));
    }
    Ok(())
}

fn validate_environment_name(value: &str, field: &str) -> IpcResult<()> {
    validate_text(value, field, 128, false)?;
    let mut bytes = value.trim().bytes();
    let valid_first = bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_uppercase() || byte == b'_');
    let valid_rest =
        bytes.all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_');
    if !valid_first || !valid_rest {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} must be an environment-variable name, never a raw API key."),
        ));
    }
    Ok(())
}

fn validate_business_processing(request: &CreateJobRequest) -> IpcResult<()> {
    validate_language_tag(&request.language, "language", true)?;
    validate_text(&request.local_llm_model, "localLlmModel", 256, false)?;
    let endpoint_policy = request.local_llm_endpoint_policy.trim();
    validate_llm_endpoint(
        &request.local_llm_endpoint,
        "localLlmEndpoint",
        endpoint_policy,
    )?;
    let provider = request
        .llm_provider
        .as_deref()
        .unwrap_or(DEFAULT_LLM_PROVIDER);
    validate_provider_id(provider, "llmProvider")?;
    if provider == DEFAULT_LLM_PROVIDER && endpoint_policy != LOCAL_LLM_LOOPBACK_POLICY {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            "Ollama must use the loopback-only endpoint policy.",
        ));
    }
    if let Some(api_key_env) = request.llm_api_key_env.as_deref() {
        validate_environment_name(api_key_env, "llmApiKeyEnv")?;
    }
    if let Some(proxy_url) = request.llm_proxy_url.as_deref() {
        validate_llm_endpoint(proxy_url, "llmProxyUrl", LOCAL_LLM_REMOTE_POLICY)?;
    }
    if request.local_llm_auto_apply {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            "localLlmAutoApply must remain false so derived outputs never overwrite the transcript.",
        ));
    }
    if request.business_prompt_version != BUSINESS_PROMPT_VERSION {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("businessPromptVersion must be {BUSINESS_PROMPT_VERSION}."),
        ));
    }

    validate_language_tag(&request.output_locale, "outputLocale", false)?;
    let mut targets = std::collections::HashSet::new();
    for (index, target) in request.translation_targets.iter().enumerate() {
        validate_language_tag(target, &format!("translationTargets[{index}]"), false)?;
        if !targets.insert(target.to_ascii_lowercase()) {
            return Err(IpcError::new(
                IpcErrorCode::InvalidRequest,
                "translationTargets must not contain duplicate language tags.",
            ));
        }
    }

    let has_business_work = !request.translation_targets.is_empty() || request.summary;
    let mode_matches_work = matches!(
        (request.local_llm_mode, has_business_work),
        (LocalLlmMode::Business, true) | (LocalLlmMode::Disabled, false)
    );
    if !mode_matches_work {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            "localLlmMode must be business exactly when translation or summary work is requested.",
        ));
    }
    Ok(())
}

fn validate_output_customization(
    output_customization: &Option<Map<String, Value>>,
) -> IpcResult<()> {
    let Some(customization) = output_customization else {
        return Ok(());
    };
    let encoded = serde_json::to_vec(customization).map_err(|error| {
        IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("outputCustomization is not valid finite JSON: {error}"),
        )
    })?;
    if encoded.len() > MAX_OUTPUT_CUSTOMIZATION_BYTES {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!(
                "outputCustomization exceeds the {MAX_OUTPUT_CUSTOMIZATION_BYTES}-byte IPC limit."
            ),
        ));
    }
    Ok(())
}

fn validate_speaker_id(value: &str) -> IpcResult<()> {
    let sequence = value.strip_prefix("speaker-").ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::InvalidRequest,
            "speakerId must match speaker-1..speaker-N.",
        )
    })?;
    if sequence.is_empty()
        || sequence.starts_with('0')
        || !sequence.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            "speakerId must match speaker-1..speaker-N, where N is a positive integer without leading zeroes.",
        ));
    }
    let value = sequence.parse::<usize>().map_err(|_| {
        IpcError::new(
            IpcErrorCode::InvalidRequest,
            "speakerId sequence exceeds the positive safe-integer range.",
        )
    })?;
    validate_speaker_count(value, "speakerId sequence")
}

fn canonical_media_file(path: &Path) -> IpcResult<PathBuf> {
    if !path.is_absolute() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidPath,
            "The media file must use an absolute path.",
        ));
    }
    let canonical = fs::canonicalize(path).map_err(|_| {
        IpcError::new(
            IpcErrorCode::NotFound,
            "The media file does not exist or could not be resolved.",
        )
    })?;
    if !canonical.is_file() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidPath,
            "The media path must point to a regular file.",
        ));
    }
    Ok(canonical)
}

fn writable_output_directory(path: &Path) -> IpcResult<PathBuf> {
    if !path.is_absolute() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidPath,
            "The output directory must use an absolute path.",
        ));
    }
    let canonical = fs::canonicalize(path).map_err(|_| {
        IpcError::new(
            IpcErrorCode::NotFound,
            "The output directory does not exist or could not be resolved.",
        )
    })?;
    if !canonical.is_dir() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidPath,
            "The output path must point to a directory.",
        ));
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let probe = canonical.join(format!(
        ".media-transcribe-studio-write-probe-{}-{nonce}.tmp",
        std::process::id()
    ));
    let mut handle = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|_| {
            IpcError::new(
                IpcErrorCode::InvalidPath,
                "The output directory is not writable, or the write probe collided.",
            )
        })?;
    handle.write_all(b"write-probe").map_err(|_| {
        IpcError::new(
            IpcErrorCode::InvalidPath,
            "The output-directory write probe failed.",
        )
    })?;
    drop(handle);
    fs::remove_file(&probe).map_err(|_| {
        IpcError::new(
            IpcErrorCode::InvalidPath,
            "The output-directory write probe could not be removed safely.",
        )
    })?;
    Ok(canonical)
}

fn prepare_job(request: CreateJobRequest) -> IpcResult<PreparedJob> {
    validate_text(&request.title, "title", 80, false)?;
    if let Some(key) = request.idempotency_key.as_deref() {
        validate_text(key, "idempotencyKey", 1_024, false)?;
        IdempotencyKey::new(key.trim().to_owned()).map_err(registry_ipc_error)?;
    }
    validate_business_processing(&request)?;
    validate_output_customization(&request.output_customization)?;
    request.speaker_policy.validate()?;
    let expected_label_count = request.speaker_policy.expected_label_count();
    let labels_are_complete = request.speaker_labels.len() == expected_label_count
        || (!matches!(&request.speaker_policy, SpeakerCountPolicy::Auto {})
            && request.speaker_labels.is_empty());
    if !labels_are_complete {
        let message = if matches!(&request.speaker_policy, SpeakerCountPolicy::Auto {}) {
            format!(
                "Automatic mode accepts only an empty label list; received {}.",
                request.speaker_labels.len()
            )
        } else {
            format!(
                "This policy accepts either an empty label list or exactly {expected_label_count} speaker labels; received {}.",
                request.speaker_labels.len()
            )
        };
        return Err(IpcError::new(IpcErrorCode::InvalidRequest, message));
    }
    for (index, label) in request.speaker_labels.iter().enumerate() {
        validate_text(label, &format!("speakerLabels[{index}]"), 32, false)?;
    }
    let media_path = canonical_media_file(&request.media_path)?;
    let output_directory = writable_output_directory(&request.output_directory)?;
    Ok(PreparedJob {
        idempotency_key: request.idempotency_key.map(|key| key.trim().to_owned()),
        title: request.title.trim().to_owned(),
        media_path,
        output_directory,
        strategy_id: request.strategy_id,
        speaker_policy: request.speaker_policy,
        speaker_labels: request
            .speaker_labels
            .into_iter()
            .map(|label| label.trim().to_owned())
            .collect(),
        language: request.language,
        local_llm_mode: request.local_llm_mode,
        local_llm_model: request.local_llm_model.trim().to_owned(),
        local_llm_endpoint: request.local_llm_endpoint.trim().to_owned(),
        local_llm_endpoint_policy: request.local_llm_endpoint_policy.trim().to_owned(),
        llm_provider: request
            .llm_provider
            .map(|value| value.trim().to_owned())
            .unwrap_or_else(|| DEFAULT_LLM_PROVIDER.to_owned()),
        llm_api_key_env: request.llm_api_key_env.map(|value| value.trim().to_owned()),
        llm_proxy_url: request.llm_proxy_url.map(|value| value.trim().to_owned()),
        local_llm_auto_apply: request.local_llm_auto_apply,
        translation_targets: request.translation_targets,
        summary: request.summary,
        output_locale: request.output_locale,
        business_prompt_version: request.business_prompt_version,
        output_customization: request.output_customization,
    })
}

fn safe_relative_path(relative_path: &str) -> IpcResult<PathBuf> {
    validate_text(relative_path, "relativePath", 1024, false)?;
    let relative = Path::new(relative_path);
    // Worker artifact paths are serialized with forward slashes.  Reject
    // Windows separators and drive prefixes even on Unix, where Rust would
    // otherwise treat `C:\\...` as an ordinary relative filename.
    let has_windows_drive_prefix = relative_path.len() >= 2
        && relative_path.as_bytes()[0].is_ascii_alphabetic()
        && relative_path.as_bytes()[1] == b':';
    let has_unsafe_text_component = relative_path
        .split(['/', '\\'])
        .any(|component| component.is_empty() || component == "." || component == "..");
    if relative.is_absolute()
        || has_windows_drive_prefix
        || relative_path.contains('\\')
        || has_unsafe_text_component
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(IpcError::new(
            IpcErrorCode::PathBoundaryViolation,
            "An artifact path must be relative and contain no `.`, `..`, root, or drive-prefix components.",
        ));
    }
    Ok(relative.to_path_buf())
}

fn resolve_artifact_path(output_root: &Path, relative_path: &str) -> IpcResult<PathBuf> {
    let canonical_root = fs::canonicalize(output_root).map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The output root is no longer valid. Create the task again.",
        )
    })?;
    let relative = safe_relative_path(relative_path)?;
    let candidate = fs::canonicalize(canonical_root.join(relative)).map_err(|_| {
        IpcError::new(
            IpcErrorCode::NotFound,
            "The artifact does not exist or has not been generated.",
        )
    })?;
    if !candidate.starts_with(&canonical_root) {
        return Err(IpcError::new(
            IpcErrorCode::PathBoundaryViolation,
            "The resolved artifact path escapes the output root.",
        ));
    }
    if !candidate.is_file() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidPath,
            "The artifact must be a regular file.",
        ));
    }
    Ok(candidate)
}

fn next_job_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    let sequence = NEXT_JOB_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("job-{timestamp}-{}-{sequence}", std::process::id())
}

fn derive_worker_output_directory(selected_root: &Path, job_id: &str) -> IpcResult<PathBuf> {
    validate_text(job_id, "jobId", 160, false)?;
    if !job_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            "jobId may contain only ASCII letters, digits, hyphens, and underscores.",
        ));
    }
    if !selected_root.is_absolute() || !selected_root.is_dir() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidPath,
            "The worker output root must be an existing absolute directory.",
        ));
    }

    let candidate = selected_root.join(job_id);
    if !candidate.starts_with(selected_root) {
        return Err(IpcError::new(
            IpcErrorCode::PathBoundaryViolation,
            "The worker output directory escapes the selected output root.",
        ));
    }
    if candidate.exists() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidPath,
            "The worker task output directory already exists; the task will not start because overwriting is prohibited.",
        ));
    }
    Ok(candidate)
}

fn worker_path(path: &Path, field: &str) -> IpcResult<String> {
    path.to_str().map(str::to_owned).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::InvalidPath,
            format!("{field} is not valid UTF-8 and cannot be written safely to the worker JSON protocol."),
        )
    })
}

fn json_count(value: usize, field: &str) -> IpcResult<Value> {
    let value = u64::try_from(value).map_err(|_| {
        IpcError::new(
            IpcErrorCode::InvalidRequest,
            format!("{field} exceeds the worker JSON integer range."),
        )
    })?;
    Ok(Value::from(value))
}

fn build_job_start_payload(
    prepared: &PreparedJob,
    job_id: &str,
    worker_output_directory: &Path,
) -> IpcResult<Map<String, Value>> {
    let mut payload = Map::new();
    payload.insert("jobId".to_owned(), Value::String(job_id.to_owned()));
    payload.insert(
        "sourcePath".to_owned(),
        Value::String(worker_path(&prepared.media_path, "sourcePath")?),
    );
    payload.insert(
        "outputDirectory".to_owned(),
        Value::String(worker_path(worker_output_directory, "outputDirectory")?),
    );
    payload.insert("renderPdf".to_owned(), Value::Bool(true));
    payload.insert("title".to_owned(), Value::String(prepared.title.clone()));
    payload.insert(
        "language".to_owned(),
        Value::String(prepared.language.clone()),
    );
    payload.insert(
        "localLlmMode".to_owned(),
        Value::String(prepared.local_llm_mode.as_str().to_owned()),
    );
    payload.insert(
        "localLlmModel".to_owned(),
        Value::String(prepared.local_llm_model.clone()),
    );
    payload.insert(
        "localLlmEndpoint".to_owned(),
        Value::String(prepared.local_llm_endpoint.clone()),
    );
    payload.insert(
        "localLlmEndpointPolicy".to_owned(),
        Value::String(prepared.local_llm_endpoint_policy.clone()),
    );
    payload.insert(
        "llmProvider".to_owned(),
        Value::String(prepared.llm_provider.clone()),
    );
    if let Some(api_key_env) = &prepared.llm_api_key_env {
        payload.insert(
            "llmApiKeyEnv".to_owned(),
            Value::String(api_key_env.clone()),
        );
    }
    if let Some(proxy_url) = &prepared.llm_proxy_url {
        payload.insert("llmProxyUrl".to_owned(), Value::String(proxy_url.clone()));
    }
    payload.insert(
        "localLlmAutoApply".to_owned(),
        Value::Bool(prepared.local_llm_auto_apply),
    );
    payload.insert(
        "translationTargets".to_owned(),
        Value::Array(
            prepared
                .translation_targets
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        ),
    );
    payload.insert("summary".to_owned(), Value::Bool(prepared.summary));
    payload.insert(
        "outputLocale".to_owned(),
        Value::String(prepared.output_locale.clone()),
    );
    payload.insert(
        "businessPromptVersion".to_owned(),
        Value::String(prepared.business_prompt_version.clone()),
    );
    if let Some(output_customization) = &prepared.output_customization {
        payload.insert(
            "outputCustomization".to_owned(),
            Value::Object(output_customization.clone()),
        );
    }

    match &prepared.speaker_policy {
        SpeakerCountPolicy::Auto {} => {
            payload.insert(
                "speakerCountMode".to_owned(),
                Value::String("auto".to_owned()),
            );
        }
        SpeakerCountPolicy::Manual { count } => {
            payload.insert(
                "speakerCountMode".to_owned(),
                Value::String("manual".to_owned()),
            );
            payload.insert(
                "speakerCount".to_owned(),
                json_count(*count, "speakerCount")?,
            );
            if !prepared.speaker_labels.is_empty() {
                payload.insert(
                    "speakerRoles".to_owned(),
                    Value::Array(
                        prepared
                            .speaker_labels
                            .iter()
                            .cloned()
                            .map(Value::String)
                            .collect(),
                    ),
                );
            }
        }
        SpeakerCountPolicy::Hybrid {
            min_speakers,
            max_speakers,
            prior_count,
        } => {
            payload.insert(
                "speakerCountMode".to_owned(),
                Value::String("hybrid".to_owned()),
            );
            payload.insert(
                "speakerCountBounds".to_owned(),
                Value::Object(Map::from_iter([
                    (
                        "min".to_owned(),
                        json_count(*min_speakers, "speakerCountBounds.min")?,
                    ),
                    (
                        "max".to_owned(),
                        json_count(*max_speakers, "speakerCountBounds.max")?,
                    ),
                ])),
            );
            payload.insert(
                "speakerCountPrior".to_owned(),
                json_count(*prior_count, "speakerCountPrior")?,
            );
        }
    }

    Ok(payload)
}

fn prepared_job_fingerprint(prepared: &PreparedJob) -> IpcResult<String> {
    let fingerprint_output = prepared
        .output_directory
        .join(".media-transcribe-studio-idempotency");
    let payload = build_job_start_payload(prepared, "idempotency-request", &fingerprint_output)?;
    let bytes = serde_json::to_vec(&payload).map_err(|error| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("Unable to fingerprint the normalized task request: {error}"),
        )
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn worker_ipc_error(context: &str, error: WorkerError) -> IpcError {
    IpcError::new(
        IpcErrorCode::StateUnavailable,
        format!("{context}: worker {:?}: {}", error.kind, error.message),
    )
}

fn response_payload_string<'a>(response: &'a WorkerResponse, key: &str) -> IpcResult<&'a str> {
    response
        .payload
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("The worker response is missing string field payload.{key}."),
            )
        })
}

fn validate_response_identity(
    response: &WorkerResponse,
    expected_kind: ResponseKind,
    expected_job_id: &str,
) -> IpcResult<()> {
    if response.kind != expected_kind {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!(
                "Worker response type mismatch: expected {:?}, received {:?}.",
                expected_kind, response.kind
            ),
        ));
    }
    let actual_job_id = response_payload_string(response, "jobId")?;
    if actual_job_id != expected_job_id {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!(
                "Worker response jobId mismatch: expected {expected_job_id}, received {actual_job_id}."
            ),
        ));
    }
    Ok(())
}

fn validate_start_accepted(response: &WorkerResponse, expected_job_id: &str) -> IpcResult<()> {
    validate_response_identity(response, ResponseKind::Accepted, expected_job_id)?;
    if response.payload.len() != 2 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The accepted job.start payload must contain exactly jobId and status.",
        ));
    }
    let status = response_payload_string(response, "status")?;
    if status != "queued" {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("The accepted job.start status must be queued; received {status}."),
        ));
    }
    Ok(())
}

fn validate_cancel_accepted(response: &WorkerResponse, expected_job_id: &str) -> IpcResult<String> {
    validate_response_identity(response, ResponseKind::Accepted, expected_job_id)?;
    if response.payload.len() != 2 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The accepted job.cancel payload must contain exactly jobId and status.",
        ));
    }
    let status = response_payload_string(response, "status")?;
    if !matches!(
        status,
        "queued" | "running" | "review_required" | "completed" | "failed" | "cancelled"
    ) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("The accepted job.cancel response returned unknown status {status}."),
        ));
    }
    Ok(status.to_owned())
}

#[derive(Debug, Clone)]
struct HumanMutationEnvelope {
    decision_id: String,
    reason: String,
    evidence: Vec<String>,
    confidence: f64,
    audit: Map<String, Value>,
    submitted_at_unix_ms: u64,
}

#[derive(Debug, Clone)]
struct PreparedReviewMutation {
    job_id: String,
    speaker_count: usize,
    review_id: String,
    speaker_id: String,
    raw_text: String,
    normalized_text: String,
    previous_speaker_id: String,
    previous_normalized_text: String,
    envelope: HumanMutationEnvelope,
    payload: Map<String, Value>,
}

#[derive(Debug, Clone)]
struct PreparedSpeakerRename {
    job_id: String,
    speaker_count: usize,
    speaker_id: String,
    previous_label: String,
    next_label: String,
    envelope: HumanMutationEnvelope,
    payload: Map<String, Value>,
}

#[derive(Debug, Clone)]
struct MutationReceipt {
    decision: Map<String, Value>,
    open_count: usize,
    document_hash: String,
    speaker_count: usize,
}

fn unix_time_ms() -> IpcResult<u64> {
    let milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The system clock is unavailable.",
            )
        })?
        .as_millis();
    u64::try_from(milliseconds).map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The system clock value is too large for the desktop audit contract.",
        )
    })
}

fn next_mutation_decision_id(kind: &str, submitted_at_unix_ms: u64) -> String {
    let sequence = NEXT_MUTATION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("desktop-{kind}-{submitted_at_unix_ms}-{sequence}")
}

fn exact_object_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    context: &str,
) -> IpcResult<()> {
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        let mut actual = object.keys().cloned().collect::<Vec<_>>();
        actual.sort();
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!(
                "{context} fields must be exactly {:?}; received {:?}.",
                expected, actual
            ),
        ));
    }
    Ok(())
}

fn response_payload_object<'a>(
    response: &'a WorkerResponse,
    key: &str,
) -> IpcResult<&'a Map<String, Value>> {
    response
        .payload
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("The worker response payload.{key} must be an object."),
            )
        })
}

fn response_payload_usize(response: &WorkerResponse, key: &str) -> IpcResult<usize> {
    let value = response
        .payload
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("The worker response payload.{key} must be a non-negative integer."),
            )
        })?;
    usize::try_from(value).map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("The worker response payload.{key} is too large for this runtime."),
        )
    })
}

fn validate_human_mutation_decision(
    decision: &Map<String, Value>,
    expected_command: &str,
    expected: &HumanMutationEnvelope,
) -> IpcResult<()> {
    exact_object_keys(
        decision,
        &[
            "decisionId",
            "command",
            "reason",
            "evidence",
            "confidence",
            "audit",
            "recordedAt",
        ],
        "The worker mutation decision",
    )?;
    if decision.get("decisionId").and_then(Value::as_str) != Some(expected.decision_id.as_str()) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The worker mutation decisionId does not match the submitted human decision.",
        ));
    }
    if decision.get("command").and_then(Value::as_str) != Some(expected_command) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The worker mutation decision command does not match the requested command.",
        ));
    }
    if decision.get("reason").and_then(Value::as_str) != Some(expected.reason.as_str()) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The worker mutation decision reason does not match the submitted human evidence.",
        ));
    }
    let expected_evidence = Value::Array(
        expected
            .evidence
            .iter()
            .cloned()
            .map(Value::String)
            .collect(),
    );
    if decision.get("evidence") != Some(&expected_evidence) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The worker mutation decision evidence does not match the submitted audit evidence.",
        ));
    }
    let confidence = decision
        .get("confidence")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The worker mutation decision confidence must be numeric.",
            )
        })?;
    if !confidence.is_finite() || (confidence - expected.confidence).abs() > 1e-9 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The worker mutation decision confidence does not match the submitted confidence.",
        ));
    }
    if decision.get("audit") != Some(&Value::Object(expected.audit.clone())) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The worker mutation decision audit object does not match the submitted human audit.",
        ));
    }
    let recorded_at = decision
        .get("recordedAt")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The worker mutation decision is missing recordedAt.",
            )
        })?;
    validate_text(
        recorded_at,
        "worker mutation decision.recordedAt",
        128,
        false,
    )?;
    Ok(())
}

fn validate_mutation_completed(
    response: &WorkerResponse,
    expected_job_id: &str,
    expected_command: &str,
    expected: &HumanMutationEnvelope,
    expected_speaker_count: usize,
) -> IpcResult<MutationReceipt> {
    validate_response_identity(response, ResponseKind::Completed, expected_job_id)?;
    exact_object_keys(
        &response.payload,
        &[
            "jobId",
            "status",
            "command",
            "decision",
            "openCount",
            "documentHash",
            "speakerCount",
        ],
        "The completed worker mutation payload",
    )?;
    if response_payload_string(response, "status")? != "review_required" {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A completed human mutation must preserve worker status review_required.",
        ));
    }
    if response_payload_string(response, "command")? != expected_command {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The completed worker mutation command does not match the requested command.",
        ));
    }
    let decision = response_payload_object(response, "decision")?;
    validate_human_mutation_decision(decision, expected_command, expected)?;
    let document_hash = response_payload_string(response, "documentHash")?.to_owned();
    validate_sha256(&document_hash, "worker mutation payload.documentHash")?;
    let speaker_count = response_payload_usize(response, "speakerCount")?;
    validate_speaker_count(speaker_count, "worker mutation payload.speakerCount")?;
    if speaker_count != expected_speaker_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The completed worker mutation speakerCount conflicts with the committed desktop task.",
        ));
    }
    Ok(MutationReceipt {
        decision: decision.clone(),
        open_count: response_payload_usize(response, "openCount")?,
        document_hash,
        speaker_count,
    })
}

fn validate_queue_decision_base(
    decision: &Map<String, Value>,
    receipt: &MutationReceipt,
) -> IpcResult<()> {
    for key in [
        "decisionId",
        "command",
        "reason",
        "evidence",
        "confidence",
        "audit",
        "recordedAt",
    ] {
        if decision.get(key) != receipt.decision.get(key) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!(
                    "The review.queue audit decision field {key} does not match the completed mutation receipt."
                ),
            ));
        }
    }
    Ok(())
}

fn validate_review_queue_snapshot(
    response: &WorkerResponse,
    expected_job_id: &str,
    receipt: &MutationReceipt,
) -> IpcResult<Map<String, Value>> {
    validate_response_identity(response, ResponseKind::Completed, expected_job_id)?;
    exact_object_keys(
        &response.payload,
        &[
            "jobId",
            "status",
            "openCount",
            "documentHash",
            "speakerCount",
            "queue",
        ],
        "The review.queue response payload",
    )?;
    if response_payload_string(response, "status")? != "review_required" {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review.queue verification response must preserve status review_required.",
        ));
    }
    if response_payload_usize(response, "openCount")? != receipt.open_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review.queue openCount does not match the completed mutation receipt.",
        ));
    }
    if response_payload_string(response, "documentHash")? != receipt.document_hash {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review.queue documentHash does not match the completed mutation receipt.",
        ));
    }
    if response_payload_usize(response, "speakerCount")? != receipt.speaker_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review.queue speakerCount does not match the completed mutation receipt.",
        ));
    }
    let queue = response_payload_object(response, "queue")?;
    if queue.get("schemaVersion").and_then(Value::as_str) != Some("2.0.0") {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review queue schemaVersion must be 2.0.0.",
        ));
    }
    if queue.get("jobId").and_then(Value::as_str) != Some(expected_job_id) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review queue jobId does not match the committed task.",
        ));
    }
    let queue_open_count = queue
        .get("openCount")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The persisted review queue openCount must be a non-negative integer.",
            )
        })?;
    if queue_open_count != receipt.open_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review queue openCount does not match the mutation receipt.",
        ));
    }
    let items = queue
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The persisted review queue items field must be an array.",
            )
        })?;
    let mut item_ids = std::collections::HashSet::new();
    let mut counted_open = 0usize;
    for item in items {
        let item = item.as_object().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Every persisted review queue item must be an object.",
            )
        })?;
        let item_id = item.get("id").and_then(Value::as_str).ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Every persisted review queue item must have a string id.",
            )
        })?;
        validate_text(item_id, "persisted review queue item.id", 256, false)?;
        if !item_ids.insert(item_id) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Persisted review queue item IDs must be unique.",
            ));
        }
        match item.get("status").and_then(Value::as_str) {
            Some("open") => counted_open += 1,
            Some("accepted" | "rejected") => {}
            _ => {
                return Err(IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "Every persisted review queue item must have a supported status.",
                ));
            }
        }
    }
    if counted_open != receipt.open_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review queue open item count does not match openCount.",
        ));
    }
    let decisions = queue
        .get("decisions")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The persisted review queue decisions field must be an array.",
            )
        })?;
    let mut decision_ids = std::collections::HashSet::new();
    for decision in decisions {
        let decision = decision.as_object().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Every persisted review queue decision must be an object.",
            )
        })?;
        let decision_id = decision
            .get("decisionId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "Every persisted review queue decision must have a string decisionId.",
                )
            })?;
        validate_text(
            decision_id,
            "persisted review queue decision.decisionId",
            160,
            false,
        )?;
        if !decision_ids.insert(decision_id) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Persisted review queue decision IDs must be unique.",
            ));
        }
    }
    Ok(queue.clone())
}

fn validate_review_submission_proof(
    queue: &Map<String, Value>,
    prepared: &PreparedReviewMutation,
    receipt: &MutationReceipt,
) -> IpcResult<()> {
    let items = queue
        .get("items")
        .and_then(Value::as_array)
        .expect("review queue shape was validated");
    let matches = items
        .iter()
        .filter_map(Value::as_object)
        .filter(|item| item.get("id").and_then(Value::as_str) == Some(prepared.review_id.as_str()))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review queue does not contain exactly one submitted review item.",
        ));
    }
    let item = matches[0];
    if item.get("scope").and_then(Value::as_str) != Some("segment")
        || item.get("status").and_then(Value::as_str) != Some("accepted")
        || item.get("speakerId").and_then(Value::as_str) != Some(prepared.speaker_id.as_str())
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review item does not prove the accepted target speaker assignment.",
        ));
    }
    let text = item.get("text").and_then(Value::as_object).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review item is missing its text snapshot.",
        )
    })?;
    exact_object_keys(
        text,
        &["rawText", "normalizedText", "displayText"],
        "The persisted review item text snapshot",
    )?;
    if text.get("rawText").and_then(Value::as_str) != Some(prepared.raw_text.as_str())
        || text.get("normalizedText").and_then(Value::as_str)
            != Some(prepared.normalized_text.as_str())
        || text.get("displayText").and_then(Value::as_str)
            != Some(prepared.normalized_text.as_str())
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review item text snapshot does not prove the requested source-language revision.",
        ));
    }
    if item.get("decision") != Some(&Value::Object(receipt.decision.clone())) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review item decision does not match the completed mutation receipt.",
        ));
    }
    let decisions = queue
        .get("decisions")
        .and_then(Value::as_array)
        .expect("review queue shape was validated");
    let matching = decisions
        .iter()
        .filter_map(Value::as_object)
        .filter(|decision| {
            decision.get("decisionId").and_then(Value::as_str)
                == Some(prepared.envelope.decision_id.as_str())
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review queue does not contain exactly one matching audit decision.",
        ));
    }
    let decision = matching[0];
    exact_object_keys(
        decision,
        &[
            "decisionId",
            "command",
            "reason",
            "evidence",
            "confidence",
            "audit",
            "recordedAt",
            "itemId",
            "status",
        ],
        "The persisted review submission audit decision",
    )?;
    validate_queue_decision_base(decision, receipt)?;
    if decision.get("itemId").and_then(Value::as_str) != Some(prepared.review_id.as_str())
        || decision.get("status").and_then(Value::as_str) != Some("accepted")
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review audit decision is not associated with the submitted item.",
        ));
    }
    Ok(())
}

fn validate_speaker_rename_proof(
    queue: &Map<String, Value>,
    prepared: &PreparedSpeakerRename,
    receipt: &MutationReceipt,
) -> IpcResult<()> {
    let decisions = queue
        .get("decisions")
        .and_then(Value::as_array)
        .expect("review queue shape was validated");
    let matching = decisions
        .iter()
        .filter_map(Value::as_object)
        .filter(|decision| {
            decision.get("decisionId").and_then(Value::as_str)
                == Some(prepared.envelope.decision_id.as_str())
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted review queue does not contain exactly one speaker rename decision.",
        ));
    }
    let decision = matching[0];
    exact_object_keys(
        decision,
        &[
            "decisionId",
            "command",
            "reason",
            "evidence",
            "confidence",
            "audit",
            "recordedAt",
            "speakerId",
            "after",
        ],
        "The persisted speaker rename audit decision",
    )?;
    validate_queue_decision_base(decision, receipt)?;
    if decision.get("speakerId").and_then(Value::as_str) != Some(prepared.speaker_id.as_str()) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted speaker rename audit references a different speaker.",
        ));
    }
    let after = decision
        .get("after")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The persisted speaker rename audit is missing its after object.",
            )
        })?;
    exact_object_keys(
        after,
        &["name"],
        "The persisted speaker rename after object",
    )?;
    if after.get("name").and_then(Value::as_str) != Some(prepared.next_label.as_str()) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted speaker rename audit does not prove the requested name.",
        ));
    }
    Ok(())
}

fn speaker_color(index: usize) -> String {
    let hue = (((index % 360) * 137) + 252) % 360;
    format!("hsl({hue} 52% 49%)")
}

fn speaker_profile(index: usize, label: Option<&str>) -> SpeakerProfile {
    let sequence = index + 1;
    SpeakerProfile {
        id: format!("speaker-{sequence}"),
        label: label
            .map(str::to_owned)
            .unwrap_or_else(|| format!("Speaker {sequence}")),
        short_label: format!("S{sequence}"),
        color: speaker_color(index),
        role_hint: "Awaiting local voiceprint, boundary, and audio review".to_owned(),
        sample_status: SampleStatus::Missing,
        locked: false,
        review_status: SpeakerReviewStatus::Pending,
    }
}

fn create_speaker_profiles(count: usize, labels: &[String]) -> IpcResult<Vec<SpeakerProfile>> {
    validate_speaker_count(count, "speakerCount")?;
    let mut speakers = Vec::new();
    speakers.try_reserve_exact(count).map_err(|_| {
        IpcError::new(
            IpcErrorCode::InvalidRequest,
            "The requested speaker roster could not be allocated in full on this device. The task was not created, and no speaker was truncated, merged, or silently downgraded.",
        )
    })?;
    for index in 0..count {
        speakers.push(speaker_profile(
            index,
            labels.get(index).map(String::as_str),
        ));
    }
    Ok(speakers)
}

fn build_registered_job_state(
    prepared: &PreparedJob,
    job_id: &str,
    worker_output_directory: PathBuf,
    request_fingerprint: String,
) -> IpcResult<AppState> {
    let (speaker_count, speakers) = match &prepared.speaker_policy {
        SpeakerCountPolicy::Manual { count } => (
            Some(*count),
            create_speaker_profiles(*count, &prepared.speaker_labels)?,
        ),
        SpeakerCountPolicy::Auto {} | SpeakerCountPolicy::Hybrid { .. } => (None, Vec::new()),
    };
    let mut snapshot = default_snapshot();
    snapshot.job = JobSummary {
        id: job_id.to_owned(),
        title: prepared.title.clone(),
        source_path: worker_path(&prepared.media_path, "sourcePath")?,
        duration_label: "Awaiting processing".to_owned(),
        status: JobStatus::Registered,
        progress: 0,
        started_at: "Registered in volatile desktop runtime".to_owned(),
        speaker_policy: prepared.speaker_policy.clone(),
        speaker_count,
        speaker_detection: None,
        review_open_count: 0,
        active_strategy_id: prepared.strategy_id,
    };
    snapshot.speakers = speakers;
    snapshot.reviews.clear();
    snapshot.artifacts.clear();
    snapshot.diarization_quality = default_diarization_quality();
    snapshot.performance = PerformanceMetrics::Unavailable {
        reason: "The task has not run, so model-stage and resource-sampling data are unavailable."
            .to_owned(),
    };
    snapshot.pdf_quality = default_pdf_quality();
    snapshot.events = vec![StudioEvent {
        id: format!("event-{job_id}"),
        sequence: 0,
        event_type: "job.registered".to_owned(),
        stage_id: Some("media".to_owned()),
        severity: Severity::Info,
        timestamp: "Desktop runtime".to_owned(),
        title: "Task identity registered".to_owned(),
        detail: "The request is registered in volatile desktop memory. It is not yet accepted by the worker and is not durably persisted by Rust.".to_owned(),
    }];
    Ok(AppState {
        snapshot,
        output_root: Some(worker_output_directory),
        worker_evidence: WorkerEvidenceLedger::default(),
        request_fingerprint,
        worker_accepted: false,
    })
}

fn commit_accepted_job_in_state(state: &mut AppState, job_id: &str) -> IpcResult<CreateJobResult> {
    validate_current_job(state, job_id)?;
    if state.snapshot.job.status != JobStatus::Registered {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "Only a newly registered task can commit worker acceptance.",
        ));
    }
    state.snapshot.job.status = JobStatus::Queued;
    state.snapshot.job.started_at = "Accepted by worker".to_owned();
    state.worker_accepted = true;
    state.snapshot.events.push(StudioEvent {
        id: format!("event-{job_id}-accepted"),
        sequence: 1,
        event_type: "job.accepted".to_owned(),
        stage_id: Some("media".to_owned()),
        severity: Severity::Success,
        timestamp: "Queued".to_owned(),
        title: "Task explicitly accepted by worker".to_owned(),
        detail: "The media file and output directory passed boundary validation, and the production worker returned accepted/queued.".to_owned(),
    });
    Ok(CreateJobResult {
        accepted: true,
        job_id: job_id.to_owned(),
        replayed: false,
        status: JobStatus::Queued,
        message: "The production worker explicitly accepted and queued the task.".to_owned(),
    })
}

fn validate_current_job(state: &AppState, job_id: &str) -> IpcResult<()> {
    validate_text(job_id, "jobId", 128, false)?;
    if state.snapshot.job.id != job_id {
        return Err(IpcError::new(
            IpcErrorCode::NotFound,
            "The requested task was not found.",
        ));
    }
    Ok(())
}

fn commit_cancelled_job_in_state(state: &mut AppState, job_id: &str) -> IpcResult<()> {
    validate_current_job(state, job_id)?;
    state.snapshot.job.status = JobStatus::Cancelled;
    Ok(())
}

fn commit_failed_job_in_state(state: &mut AppState, job_id: &str, message: &str) -> IpcResult<()> {
    validate_current_job(state, job_id)?;
    state.snapshot.job.status = JobStatus::Failed;
    state.snapshot.system.inference_worker = WorkerStatus::Missing;
    let sequence = state
        .snapshot
        .events
        .last()
        .map_or(0, |event| event.sequence.saturating_add(1));
    state.snapshot.events.push(StudioEvent {
        id: format!("job-start-failure-{job_id}-{sequence}"),
        sequence,
        event_type: "job.failed".to_owned(),
        stage_id: None,
        severity: Severity::Error,
        timestamp: "Worker supervisor".to_owned(),
        title: "Task failed closed before or during worker acceptance".to_owned(),
        detail: message.to_owned(),
    });
    Ok(())
}

fn resolved_speaker_count(state: &AppState) -> IpcResult<usize> {
    let count = state.snapshot.job.speaker_count.ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "Human mutations require a worker-verified resolved speaker count.",
        )
    })?;
    validate_speaker_count(count, "job.speakerCount")?;
    if state.snapshot.speakers.len() != count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The desktop speaker roster does not match the worker-verified speaker count.",
        ));
    }
    Ok(count)
}

fn ensure_review_active(state: &AppState) -> IpcResult<()> {
    if !matches!(state.snapshot.job.status, JobStatus::ReviewRequired) {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "Worker-backed human mutations require the active task to be in review_required status.",
        ));
    }
    Ok(())
}

fn prepare_review_mutation(
    state: &AppState,
    decision: ReviewDecision,
) -> IpcResult<PreparedReviewMutation> {
    ensure_review_active(state)?;
    let job_id = state.snapshot.job.id.clone();
    validate_current_job(state, &job_id)?;
    let speaker_count = resolved_speaker_count(state)?;
    validate_text(&decision.review_id, "reviewId", 128, false)?;
    validate_speaker_id(&decision.speaker_id)?;
    validate_text(&decision.normalized_text, "normalizedText", 100_000, false)?;
    validate_text(&decision.reason, "reason", 4096, false)?;
    validate_text(&decision.evidence, "evidence", 4096, false)?;
    if !decision.confidence.is_finite() || !(0.0..=1.0).contains(&decision.confidence) {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            "confidence must be a finite number between 0 and 1.",
        ));
    }

    let review = state
        .snapshot
        .reviews
        .iter()
        .find(|review| review.id == decision.review_id)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::NotFound,
                "The requested review segment was not found.",
            )
        })?;
    let target_speaker = state
        .snapshot
        .speakers
        .iter()
        .find(|speaker| speaker.id == decision.speaker_id)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::NotFound,
                "The review decision references a speaker outside the current task.",
            )
        })?;
    if review.reviewed || review.locked {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "A human-reviewed or locked segment cannot overwrite an existing human decision.",
        ));
    }
    if target_speaker.locked && review.current_speaker_id != decision.speaker_id {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "A human-locked speaker cannot be reassigned through segment review.",
        ));
    }

    let reason = decision.reason.trim().to_owned();
    let evidence = vec![decision.evidence.trim().to_owned()];
    let normalized_text = decision.normalized_text.trim().to_owned();
    let submitted_at_unix_ms = unix_time_ms()?;
    let decision_id = next_mutation_decision_id("review", submitted_at_unix_ms);
    let audit = Map::from_iter([
        ("actor".to_owned(), Value::String("desktop-user".to_owned())),
        ("source".to_owned(), Value::String("human".to_owned())),
        (
            "client".to_owned(),
            Value::String("MediaTranscribeStudio desktop".to_owned()),
        ),
        ("jobId".to_owned(), Value::String(job_id.clone())),
        (
            "reviewItemId".to_owned(),
            Value::String(decision.review_id.clone()),
        ),
        (
            "targetSpeakerId".to_owned(),
            Value::String(decision.speaker_id.clone()),
        ),
        (
            "submittedAtUnixMs".to_owned(),
            Value::Number(submitted_at_unix_ms.into()),
        ),
    ]);
    let envelope = HumanMutationEnvelope {
        decision_id,
        reason,
        evidence,
        confidence: f64::from(decision.confidence),
        audit,
        submitted_at_unix_ms,
    };
    let payload = Map::from_iter([
        ("jobId".to_owned(), Value::String(job_id.clone())),
        (
            "itemId".to_owned(),
            Value::String(decision.review_id.clone()),
        ),
        ("action".to_owned(), Value::String("accept".to_owned())),
        (
            "decisionId".to_owned(),
            Value::String(envelope.decision_id.clone()),
        ),
        ("reason".to_owned(), Value::String(envelope.reason.clone())),
        (
            "evidence".to_owned(),
            Value::Array(
                envelope
                    .evidence
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        ),
        ("confidence".to_owned(), Value::from(envelope.confidence)),
        ("audit".to_owned(), Value::Object(envelope.audit.clone())),
        (
            "targetSpeakerId".to_owned(),
            Value::String(decision.speaker_id.clone()),
        ),
        (
            "normalizedText".to_owned(),
            Value::String(normalized_text.clone()),
        ),
        (
            "displayText".to_owned(),
            Value::String(normalized_text.clone()),
        ),
        ("rawText".to_owned(), Value::String(review.raw_text.clone())),
    ]);
    Ok(PreparedReviewMutation {
        job_id,
        speaker_count,
        review_id: decision.review_id,
        speaker_id: decision.speaker_id,
        raw_text: review.raw_text.clone(),
        normalized_text,
        previous_speaker_id: review.current_speaker_id.clone(),
        previous_normalized_text: review.normalized_text.clone(),
        envelope,
        payload,
    })
}

fn commit_review_mutation_in_state(
    state: &mut AppState,
    prepared: &PreparedReviewMutation,
    receipt: &MutationReceipt,
    reconciled: ReconciledMutationEvidence,
) -> IpcResult<ReviewSegment> {
    validate_current_job(state, &prepared.job_id)?;
    ensure_review_active(state)?;
    if resolved_speaker_count(state)? != receipt.speaker_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The active desktop speaker count changed after the worker persisted the review.",
        ));
    }
    let target_speaker = state
        .snapshot
        .speakers
        .iter()
        .find(|speaker| speaker.id == prepared.speaker_id)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The verified review target speaker disappeared before desktop reconciliation.",
            )
        })?;
    let index = state
        .snapshot
        .reviews
        .iter()
        .position(|review| review.id == prepared.review_id)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The worker persisted the review, but the desktop review item disappeared before reconciliation.",
            )
        })?;
    let current = &state.snapshot.reviews[index];
    if current.reviewed
        || current.locked
        || current.raw_text != prepared.raw_text
        || current.current_speaker_id != prepared.previous_speaker_id
        || current.normalized_text != prepared.previous_normalized_text
    {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "The worker persisted the review, but the desktop review state changed before reconciliation.",
        ));
    }
    if target_speaker.locked && current.current_speaker_id != prepared.speaker_id {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "The worker persisted the review, but a conflicting local human speaker lock appeared before reconciliation.",
        ));
    }
    let output_root = state.output_root.clone().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The worker mutation cannot reconcile without a committed output root.",
        )
    })?;
    let mut next_snapshot = state.snapshot.clone();
    let mut next_evidence = state.worker_evidence.clone();
    apply_reconciled_mutation_evidence(
        &mut next_snapshot,
        &mut next_evidence,
        &output_root,
        reconciled,
    )?;
    let review = next_snapshot
        .reviews
        .iter_mut()
        .find(|review| review.id == prepared.review_id)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The durable review queue lost the submitted review item during reconciliation.",
            )
        })?;
    if review.raw_text != prepared.raw_text
        || review.normalized_text != prepared.normalized_text
        || review.current_speaker_id != prepared.speaker_id
        || !review.reviewed
        || !review.locked
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The reconciled transcript and review queue do not prove the requested review mutation.",
        ));
    }
    let event = ReviewAuditEvent {
        id: prepared.envelope.decision_id.clone(),
        sequence: review.audit_trail.len() + 1,
        recorded_at_unix_ms: prepared.envelope.submitted_at_unix_ms,
        actor: ReviewActor::Human,
        reason: prepared.envelope.reason.clone(),
        evidence: prepared.envelope.evidence.join("\n"),
        confidence: prepared.envelope.confidence as f32,
        previous_speaker_id: prepared.previous_speaker_id.clone(),
        speaker_id: prepared.speaker_id.clone(),
        previous_normalized_text: prepared.previous_normalized_text.clone(),
        normalized_text: prepared.normalized_text.clone(),
    };
    if review
        .audit_trail
        .iter()
        .any(|existing| existing.id == event.id)
    {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "The desktop audit trail already contains this persisted review decision.",
        ));
    }
    review.audit_trail.push(event);
    let committed = review.clone();
    state.snapshot = next_snapshot;
    state.worker_evidence = next_evidence;
    Ok(committed)
}

fn prepare_speaker_rename(
    state: &AppState,
    request: UpdateSpeakerRequest,
) -> IpcResult<PreparedSpeakerRename> {
    ensure_review_active(state)?;
    let job_id = state.snapshot.job.id.clone();
    validate_current_job(state, &job_id)?;
    let speaker_count = resolved_speaker_count(state)?;
    validate_speaker_id(&request.speaker_id)?;
    validate_text(&request.label, "label", 64, false)?;
    let speaker = state
        .snapshot
        .speakers
        .iter()
        .find(|speaker| speaker.id == request.speaker_id)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::NotFound,
                "The speaker does not exist in the current task.",
            )
        })?;
    if request.locked != speaker.locked || request.review_status != speaker.review_status {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            "The production worker currently exposes speaker.rename only; speaker lock and review-status changes are rejected rather than stored in desktop memory.",
        ));
    }
    if speaker.locked {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "A human-locked speaker cannot be renamed.",
        ));
    }
    let next_label = request.label.trim().to_owned();
    if next_label == speaker.label {
        return Err(IpcError::new(
            IpcErrorCode::InvalidRequest,
            "The requested speaker name is unchanged, so no worker-backed mutation can be recorded.",
        ));
    }
    if state
        .snapshot
        .speakers
        .iter()
        .any(|candidate| candidate.id != speaker.id && candidate.label == next_label)
    {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "Speaker names must remain unique.",
        ));
    }
    let submitted_at_unix_ms = unix_time_ms()?;
    let decision_id = next_mutation_decision_id("speaker-rename", submitted_at_unix_ms);
    let reason = "Human renamed the speaker profile in the desktop application.".to_owned();
    let evidence = vec![
        "desktop-ui:explicit-human-speaker-rename".to_owned(),
        format!("desktop-state:{job_id}/{}", request.speaker_id),
    ];
    let audit = Map::from_iter([
        ("actor".to_owned(), Value::String("desktop-user".to_owned())),
        ("source".to_owned(), Value::String("human".to_owned())),
        (
            "client".to_owned(),
            Value::String("MediaTranscribeStudio desktop".to_owned()),
        ),
        ("jobId".to_owned(), Value::String(job_id.clone())),
        (
            "speakerId".to_owned(),
            Value::String(request.speaker_id.clone()),
        ),
        (
            "previousName".to_owned(),
            Value::String(speaker.label.clone()),
        ),
        ("newName".to_owned(), Value::String(next_label.clone())),
        (
            "submittedAtUnixMs".to_owned(),
            Value::Number(submitted_at_unix_ms.into()),
        ),
    ]);
    let envelope = HumanMutationEnvelope {
        decision_id,
        reason,
        evidence,
        confidence: 1.0,
        audit,
        submitted_at_unix_ms,
    };
    let payload = Map::from_iter([
        ("jobId".to_owned(), Value::String(job_id.clone())),
        (
            "speakerId".to_owned(),
            Value::String(request.speaker_id.clone()),
        ),
        ("name".to_owned(), Value::String(next_label.clone())),
        (
            "decisionId".to_owned(),
            Value::String(envelope.decision_id.clone()),
        ),
        ("reason".to_owned(), Value::String(envelope.reason.clone())),
        (
            "evidence".to_owned(),
            Value::Array(
                envelope
                    .evidence
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        ),
        ("confidence".to_owned(), Value::from(envelope.confidence)),
        ("audit".to_owned(), Value::Object(envelope.audit.clone())),
    ]);
    Ok(PreparedSpeakerRename {
        job_id,
        speaker_count,
        speaker_id: request.speaker_id,
        previous_label: speaker.label.clone(),
        next_label,
        envelope,
        payload,
    })
}

fn commit_speaker_rename_in_state(
    state: &mut AppState,
    prepared: &PreparedSpeakerRename,
    receipt: &MutationReceipt,
    reconciled: ReconciledMutationEvidence,
) -> IpcResult<SpeakerProfile> {
    validate_current_job(state, &prepared.job_id)?;
    ensure_review_active(state)?;
    if resolved_speaker_count(state)? != receipt.speaker_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The active desktop speaker count changed after the worker persisted the rename.",
        ));
    }
    let speaker = state
        .snapshot
        .speakers
        .iter_mut()
        .find(|speaker| speaker.id == prepared.speaker_id)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The worker persisted the rename, but the desktop speaker disappeared before reconciliation.",
            )
        })?;
    if speaker.label != prepared.previous_label || speaker.locked {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "The worker persisted the rename, but the desktop speaker state changed before reconciliation.",
        ));
    }
    let output_root = state.output_root.clone().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The worker mutation cannot reconcile without a committed output root.",
        )
    })?;
    let mut next_snapshot = state.snapshot.clone();
    let mut next_evidence = state.worker_evidence.clone();
    apply_reconciled_mutation_evidence(
        &mut next_snapshot,
        &mut next_evidence,
        &output_root,
        reconciled,
    )?;
    let speaker = next_snapshot
        .speakers
        .iter()
        .find(|speaker| speaker.id == prepared.speaker_id)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The durable transcript lost the renamed speaker during reconciliation.",
            )
        })?;
    if speaker.label != prepared.next_label {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The reconciled transcript does not prove the requested speaker name.",
        ));
    }
    let committed = speaker.clone();
    state.snapshot = next_snapshot;
    state.worker_evidence = next_evidence;
    Ok(committed)
}

fn open_artifact_in_state(state: &AppState, artifact_id: &str) -> IpcResult<ArtifactOpenResult> {
    validate_text(artifact_id, "artifactId", 128, false)?;
    let artifact = state
        .snapshot
        .artifacts
        .iter()
        .find(|artifact| artifact.id == artifact_id)
        .ok_or_else(|| IpcError::new(IpcErrorCode::NotFound, "The artifact ID does not exist."))?;
    let root = state.output_root.as_deref().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A controlled output root has not been established.",
        )
    })?;
    let path = resolve_artifact_path(root, &artifact.relative_path)?;
    Ok(ArtifactOpenResult {
        artifact_id: artifact_id.to_owned(),
        canonical_path: path.to_string_lossy().into_owned(),
        opened: false,
        message: "The path passed output-root boundary validation. No shell was invoked, and the file opener is not connected yet.".to_owned(),
    })
}

impl StudioStore {
    fn new() -> Self {
        Self {
            registry: JobRegistry::with_dispatch_limit(MAX_DISPATCH_LIMIT)
                .expect("the built-in desktop dispatch limit must be valid"),
            draft_snapshot: default_snapshot(),
            dispatch_permits: Mutex::new(HashMap::new()),
        }
    }

    fn projected_state(&self) -> IpcResult<Option<AppState>> {
        self.registry
            .projected_snapshot()
            .map(|snapshot| snapshot.map(|snapshot| snapshot.runtime_state))
            .map_err(registry_ipc_error)
    }

    fn projected_job_id(&self) -> IpcResult<Option<JobId>> {
        self.registry.projected_job_id().map_err(registry_ipc_error)
    }

    fn select_job(&self, job_id: &JobId) -> IpcResult<()> {
        self.registry
            .select_projection(job_id)
            .map(|_| ())
            .map_err(registry_ipc_error)
    }

    fn hold_dispatch_permit(&self, job_id: JobId, permit: DispatchPermit) -> IpcResult<()> {
        use std::collections::hash_map::Entry;

        let mut permits = self.dispatch_permits.lock().map_err(|_| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The volatile dispatch-permit ledger is unavailable.",
            )
        })?;
        match permits.entry(job_id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(permit);
                Ok(())
            }
            Entry::Occupied(_) => Err(IpcError::new(
                IpcErrorCode::Conflict,
                format!("Task {job_id} already owns a volatile dispatch permit."),
            )),
        }
    }

    fn release_dispatch_permit(&self, job_id: &JobId) -> IpcResult<()> {
        let permit = self
            .dispatch_permits
            .lock()
            .map_err(|_| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "The volatile dispatch-permit ledger is unavailable.",
                )
            })?
            .remove(job_id);
        if let Some(permit) = permit {
            permit.release().map_err(registry_ipc_error)?;
        }
        Ok(())
    }

    fn discard_gate_if_unpublished(
        &self,
        gates: &JobCommandGate,
        job_id: &JobId,
        expected: &Arc<tokio::sync::Mutex<()>>,
    ) -> IpcResult<()> {
        match self.registry.status(job_id) {
            Err(RegistryError::UnknownJob { .. }) => gates.remove_unpublished(job_id, expected),
            Ok(_) => Ok(()),
            Err(error) => Err(registry_ipc_error(error)),
        }
    }
}

impl JobCommandGate {
    fn for_job(&self, job_id: &JobId) -> IpcResult<Arc<tokio::sync::Mutex<()>>> {
        let mut gates = self.0.lock().map_err(|_| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The per-task command-gate registry is unavailable.",
            )
        })?;
        Ok(Arc::clone(
            gates
                .entry(job_id.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        ))
    }

    fn remove_unpublished(
        &self,
        job_id: &JobId,
        expected: &Arc<tokio::sync::Mutex<()>>,
    ) -> IpcResult<()> {
        let mut gates = self.0.lock().map_err(|_| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The per-task command-gate registry is unavailable.",
            )
        })?;
        let Some(current) = gates.get(job_id) else {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("The unpublished command gate for candidate task {job_id} disappeared."),
            ));
        };
        if !Arc::ptr_eq(current, expected) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!(
                    "The unpublished command gate for candidate task {job_id} was unexpectedly replaced."
                ),
            ));
        }
        if Arc::strong_count(current) != 2 {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!(
                    "The unpublished command gate for candidate task {job_id} has unexpected external owners."
                ),
            ));
        }
        gates.remove(job_id);
        Ok(())
    }
}

fn registry_ipc_error(error: RegistryError) -> IpcError {
    let code = match error {
        RegistryError::InvalidJobId | RegistryError::InvalidIdempotencyKey => {
            IpcErrorCode::InvalidRequest
        }
        RegistryError::UnknownJob { .. } => IpcErrorCode::NotFound,
        RegistryError::DuplicateJob { .. }
        | RegistryError::StaleJobToken { .. }
        | RegistryError::InvalidStatusTransition { .. }
        | RegistryError::DispatcherSaturated { .. }
        | RegistryError::JobAlreadyDispatched { .. }
        | RegistryError::StaleDispatchPermit { .. } => IpcErrorCode::Conflict,
        #[cfg(test)]
        RegistryError::EventTargetMismatch { .. } | RegistryError::StaleProjectionToken { .. } => {
            IpcErrorCode::Conflict
        }
        RegistryError::InvalidDispatcherLimit { .. }
        | RegistryError::ForeignToken { .. }
        | RegistryError::SequenceExhausted { .. }
        | RegistryError::StatePoisoned { .. }
        | RegistryError::InvariantViolation { .. } => IpcErrorCode::StateUnavailable,
    };
    IpcError::new(code, error.to_string())
}

fn parse_job_id(value: &str) -> IpcResult<JobId> {
    validate_text(value, "jobId", 256, false)?;
    JobId::new(value.to_owned()).map_err(registry_ipc_error)
}

fn registry_status_for_job(status: JobStatus) -> RegistryJobStatus {
    match status {
        JobStatus::Draft | JobStatus::Registered => RegistryJobStatus::Registered,
        JobStatus::Queued => RegistryJobStatus::Queued,
        JobStatus::Running => RegistryJobStatus::Running,
        JobStatus::ReviewRequired => RegistryJobStatus::ReviewRequired,
        JobStatus::Completed => RegistryJobStatus::Completed,
        JobStatus::Failed => RegistryJobStatus::Failed,
        JobStatus::Cancelled => RegistryJobStatus::Cancelled,
    }
}

fn projected_state(store: &StudioStore) -> IpcResult<AppState> {
    store.projected_state()?.ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "No task is selected in the volatile desktop runtime.",
        )
    })
}

fn job_state(store: &StudioStore, job_id: &JobId) -> IpcResult<AppState> {
    store
        .registry
        .snapshot(job_id)
        .map(|snapshot| snapshot.runtime_state)
        .map_err(registry_ipc_error)
}

fn mutate_job_state<R>(
    store: &StudioStore,
    job_id: &JobId,
    mutate: impl FnOnce(&mut AppState) -> IpcResult<R>,
) -> IpcResult<R> {
    let (_, token) = store
        .registry
        .snapshot_with_token(job_id)
        .map_err(registry_ipc_error)?;
    store
        .registry
        .try_mutate_runtime(&token, mutate)
        .map_err(registry_ipc_error)?
        .map(|(result, _)| result)
}

fn transition_job_state<R>(
    store: &StudioStore,
    job_id: &JobId,
    next_status: RegistryJobStatus,
    mutate: impl FnOnce(&mut AppState) -> IpcResult<R>,
) -> IpcResult<R> {
    let (_, token) = store
        .registry
        .snapshot_with_token(job_id)
        .map_err(registry_ipc_error)?;
    store
        .registry
        .try_transition_status_with(&token, next_status, mutate)
        .map_err(registry_ipc_error)
        .and_then(|result| result.map(|(value, _)| value))
}

fn fail_job_and_release(store: &StudioStore, job_id: &JobId, message: &str) -> IpcResult<()> {
    let transition = transition_job_state(store, job_id, RegistryJobStatus::Failed, |state| {
        commit_failed_job_in_state(state, job_id.as_str(), message)
    });
    let release = store.release_dispatch_permit(job_id);
    match (transition, release) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(transition_error), Ok(())) => Err(transition_error),
        (Ok(()), Err(release_error)) => Err(release_error),
        (Err(transition_error), Err(release_error)) => Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!(
                "Task failure projection and dispatch-permit release both failed closed: {}; {}",
                transition_error.message, release_error.message
            ),
        )),
    }
}

fn primary_error_with_cleanup(primary: IpcError, cleanup: IpcResult<()>) -> IpcError {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup_error) => IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!(
                "{} Cleanup also failed closed: {}",
                primary.message, cleanup_error.message
            ),
        ),
    }
}

fn job_snapshot_update(store: &StudioStore, job_id: &JobId) -> IpcResult<JobSnapshotUpdate> {
    let registered = store
        .registry
        .snapshot(job_id)
        .map_err(registry_ipc_error)?;
    let projected = store.projected_job_id()?.as_ref() == Some(job_id);
    Ok(JobSnapshotUpdate {
        job_id: job_id.as_str().to_owned(),
        status: registered.runtime_state.snapshot.job.status,
        revision: registered.revision,
        projected,
        snapshot: registered.runtime_state.snapshot,
    })
}

#[derive(Debug)]
struct TranscriptProjection {
    speaker_count: usize,
    speaker_detection: Option<SpeakerCountDetection>,
    speakers: Vec<SpeakerProfile>,
    duration_label: String,
    duration_ms: u64,
    speaker_count_mode: String,
    speaker_count_estimate: Value,
    segments: BTreeMap<String, TranscriptSegmentEvidence>,
}

fn event_field<'a>(payload: &'a Map<String, Value>, field: &str) -> IpcResult<&'a Value> {
    payload.get(field).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("Worker event payload is missing {field}."),
        )
    })
}

fn event_string<'a>(
    payload: &'a Map<String, Value>,
    field: &str,
    max_chars: usize,
) -> IpcResult<&'a str> {
    let value = event_field(payload, field)?.as_str().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("Worker event payload.{field} must be a string."),
        )
    })?;
    validate_text(
        value,
        &format!("worker event payload.{field}"),
        max_chars,
        false,
    )?;
    Ok(value)
}

fn event_nonnegative_integer(payload: &Map<String, Value>, field: &str) -> IpcResult<u64> {
    event_field(payload, field)?.as_u64().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("Worker event payload.{field} must be a non-negative integer."),
        )
    })
}

#[derive(Debug, Clone)]
enum StrictJsonValue {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    String(String),
    Array(Vec<StrictJsonValue>),
    Object(BTreeMap<String, StrictJsonValue>),
}

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StrictJsonVisitor;

        impl<'de> Visitor<'de> for StrictJsonVisitor {
            type Value = StrictJsonValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("strict finite JSON")
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(StrictJsonValue::Null)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(StrictJsonValue::Null)
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(StrictJsonValue::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(StrictJsonValue::I64(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(StrictJsonValue::U64(value))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: DeError,
            {
                if !value.is_finite() {
                    return Err(E::custom("JSON numbers must be finite"));
                }
                Ok(StrictJsonValue::F64(value))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(StrictJsonValue::String(value.to_owned()))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(StrictJsonValue::String(value))
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<StrictJsonValue>()? {
                    values.push(value);
                }
                Ok(StrictJsonValue::Array(values))
            }

            fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key, value)) = object.next_entry::<String, StrictJsonValue>()? {
                    if values.insert(key.clone(), value).is_some() {
                        return Err(A::Error::custom(format!(
                            "duplicate JSON object key {key:?}"
                        )));
                    }
                }
                Ok(StrictJsonValue::Object(values))
            }
        }

        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

impl StrictJsonValue {
    fn into_value(self) -> IpcResult<Value> {
        match self {
            Self::Null => Ok(Value::Null),
            Self::Bool(value) => Ok(Value::Bool(value)),
            Self::I64(value) => Ok(Value::from(value)),
            Self::U64(value) => Ok(Value::from(value)),
            Self::F64(value) => serde_json::Number::from_f64(value)
                .map(Value::Number)
                .ok_or_else(|| {
                    IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        "Strict JSON contained a non-finite number.",
                    )
                }),
            Self::String(value) => Ok(Value::String(value)),
            Self::Array(values) => values
                .into_iter()
                .map(Self::into_value)
                .collect::<IpcResult<Vec<_>>>()
                .map(Value::Array),
            Self::Object(values) => values
                .into_iter()
                .map(|(key, value)| Ok((key, value.into_value()?)))
                .collect::<IpcResult<Map<_, _>>>()
                .map(Value::Object),
        }
    }

    fn write_canonical(&self, output: &mut String) -> IpcResult<()> {
        match self {
            Self::Null => output.push_str("null"),
            Self::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Self::I64(value) => output.push_str(&value.to_string()),
            Self::U64(value) => output.push_str(&value.to_string()),
            Self::F64(value) => output.push_str(&python_json_float(*value)?),
            Self::String(value) => output.push_str(&json_string(value)?),
            Self::Array(values) => {
                output.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(',');
                    }
                    value.write_canonical(output)?;
                }
                output.push(']');
            }
            Self::Object(values) => {
                output.push('{');
                for (index, (key, value)) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(',');
                    }
                    output.push_str(&json_string(key)?);
                    output.push(':');
                    value.write_canonical(output)?;
                }
                output.push('}');
            }
        }
        Ok(())
    }
}

fn json_string(value: &str) -> IpcResult<String> {
    serde_json::to_string(value).map_err(|error| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("Unable to encode strict JSON text: {error}"),
        )
    })
}

fn normalize_json_exponent(raw: &str) -> IpcResult<String> {
    let (mantissa, exponent) = raw.split_once(['e', 'E']).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A floating-point exponent could not be normalized.",
        )
    })?;
    let exponent = exponent.parse::<i32>().map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A floating-point exponent is outside the canonical JSON range.",
        )
    })?;
    let sign = if exponent < 0 { '-' } else { '+' };
    let magnitude = exponent.unsigned_abs();
    Ok(format!("{mantissa}e{sign}{magnitude:02}"))
}

fn decimal_to_json_exponent(raw: &str) -> IpcResult<String> {
    let (negative, unsigned) = raw
        .strip_prefix('-')
        .map_or((false, raw), |value| (true, value));
    let (integer, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let digits = format!("{integer}{fraction}");
    let first_nonzero = digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Zero must not be converted to exponential JSON notation.",
            )
        })?;
    let decimal_position = i32::try_from(integer.len()).map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A floating-point decimal position is too large.",
        )
    })?;
    let first_nonzero = i32::try_from(first_nonzero).map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A floating-point significant position is too large.",
        )
    })?;
    let exponent = decimal_position - first_nonzero - 1;
    let significant = digits
        .trim_start_matches('0')
        .trim_end_matches('0')
        .as_bytes();
    let first = char::from(significant[0]);
    let remainder = std::str::from_utf8(&significant[1..]).map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A floating-point mantissa was not valid UTF-8.",
        )
    })?;
    let mut mantissa = String::new();
    if negative {
        mantissa.push('-');
    }
    mantissa.push(first);
    if !remainder.is_empty() {
        mantissa.push('.');
        mantissa.push_str(remainder);
    }
    let sign = if exponent < 0 { '-' } else { '+' };
    Ok(format!("{mantissa}e{sign}{:02}", exponent.unsigned_abs()))
}

fn exponent_to_json_decimal(raw: &str) -> IpcResult<String> {
    let (mantissa, exponent) = raw.split_once(['e', 'E']).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A floating-point exponent could not be expanded.",
        )
    })?;
    let exponent = exponent.parse::<i32>().map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A floating-point exponent is outside the canonical JSON range.",
        )
    })?;
    let (negative, unsigned) = mantissa
        .strip_prefix('-')
        .map_or((false, mantissa), |value| (true, value));
    let (integer, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let digits = format!("{integer}{fraction}");
    let decimal_position = i32::try_from(integer.len()).map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A floating-point decimal position is too large.",
        )
    })? + exponent;
    let mut output = String::new();
    if negative {
        output.push('-');
    }
    if decimal_position <= 0 {
        output.push_str("0.");
        for _ in 0..decimal_position.unsigned_abs() {
            output.push('0');
        }
        output.push_str(&digits);
    } else {
        let position = usize::try_from(decimal_position).map_err(|_| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "A floating-point decimal position is too large.",
            )
        })?;
        if position >= digits.len() {
            output.push_str(&digits);
            for _ in digits.len()..position {
                output.push('0');
            }
            output.push_str(".0");
        } else {
            output.push_str(&digits[..position]);
            output.push('.');
            output.push_str(&digits[position..]);
        }
    }
    Ok(output)
}

fn python_json_float(value: f64) -> IpcResult<String> {
    if !value.is_finite() {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "Canonical JSON does not permit non-finite numbers.",
        ));
    }
    let raw = serde_json::to_string(&value).map_err(|error| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("Unable to encode a canonical JSON number: {error}"),
        )
    })?;
    if value == 0.0 {
        return Ok(raw);
    }
    let absolute = value.abs();
    let use_exponent = !(1e-4..1e16).contains(&absolute);
    match (use_exponent, raw.contains(['e', 'E'])) {
        (true, true) => normalize_json_exponent(&raw),
        (true, false) => decimal_to_json_exponent(&raw),
        (false, true) => exponent_to_json_decimal(&raw),
        (false, false) => Ok(raw),
    }
}

fn write_value_canonical(value: &Value, output: &mut String) -> IpcResult<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                output.push_str(&value.to_string());
            } else if let Some(value) = value.as_u64() {
                output.push_str(&value.to_string());
            } else if let Some(value) = value.as_f64() {
                output.push_str(&python_json_float(value)?);
            } else {
                return Err(IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "Canonical JSON encountered an unsupported number.",
                ));
            }
        }
        Value::String(value) => output.push_str(&json_string(value)?),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_value_canonical(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let sorted = values.iter().collect::<BTreeMap<_, _>>();
            for (index, (key, value)) in sorted.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(&json_string(key)?);
                output.push(':');
                write_value_canonical(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn strict_json_document(bytes: &[u8], context: &str) -> IpcResult<(Value, Vec<u8>)> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let strict = StrictJsonValue::deserialize(&mut deserializer).map_err(|error| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{context} is not strict JSON: {error}"),
        )
    })?;
    deserializer.end().map_err(|error| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{context} has trailing JSON data: {error}"),
        )
    })?;
    let mut canonical = String::new();
    strict.write_canonical(&mut canonical)?;
    canonical.push('\n');
    Ok((strict.into_value()?, canonical.into_bytes()))
}

fn canonical_json_bytes(value: &Value) -> IpcResult<Vec<u8>> {
    let mut canonical = String::new();
    write_value_canonical(value, &mut canonical)?;
    canonical.push('\n');
    Ok(canonical.into_bytes())
}

fn canonical_json_sha256(value: &Value) -> IpcResult<String> {
    let digest = Sha256::digest(canonical_json_bytes(value)?);
    Ok(format!("{digest:x}"))
}

fn strict_json_file(path: &Path, context: &str) -> IpcResult<(Value, String)> {
    let bytes = fs::read(path).map_err(|error| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("Unable to read {context}: {error}"),
        )
    })?;
    let (value, canonical) = strict_json_document(&bytes, context)?;
    let digest = Sha256::digest(canonical);
    Ok((value, format!("{digest:x}")))
}

fn validate_sha256(value: &str, field: &str) -> IpcResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{field} must be a lowercase SHA-256 digest."),
        ));
    }
    Ok(())
}

fn resolve_worker_artifact_path(output_root: &Path, raw_path: &str) -> IpcResult<PathBuf> {
    validate_text(raw_path, "worker artifact path", 4096, false)?;
    let path = Path::new(raw_path);
    if !path.is_absolute() {
        return Err(IpcError::new(
            IpcErrorCode::PathBoundaryViolation,
            "Worker artifact paths must be absolute before desktop projection.",
        ));
    }
    let canonical_root = fs::canonicalize(output_root).map_err(|error| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("The task output root cannot be resolved: {error}"),
        )
    })?;
    let canonical_path = fs::canonicalize(path).map_err(|error| {
        IpcError::new(
            IpcErrorCode::NotFound,
            format!("The worker announced an artifact that cannot be resolved: {error}"),
        )
    })?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(IpcError::new(
            IpcErrorCode::PathBoundaryViolation,
            "The worker announced an artifact outside the current task output root.",
        ));
    }
    if !canonical_path.is_file() {
        return Err(IpcError::new(
            IpcErrorCode::InvalidPath,
            "The worker-announced artifact is not a regular file.",
        ));
    }
    Ok(canonical_path)
}

fn relative_worker_artifact_path(output_root: &Path, artifact: &Path) -> IpcResult<String> {
    let canonical_root = fs::canonicalize(output_root).map_err(|error| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("The task output root cannot be resolved: {error}"),
        )
    })?;
    let relative = artifact.strip_prefix(&canonical_root).map_err(|_| {
        IpcError::new(
            IpcErrorCode::PathBoundaryViolation,
            "The verified artifact cannot be made relative to the task output root.",
        )
    })?;
    let value = relative.to_string_lossy().replace('\\', "/");
    let _ = safe_relative_path(&value)?;
    Ok(value)
}

fn format_duration_ms(duration_ms: u64) -> String {
    let total_seconds = duration_ms / 1_000;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

fn json_positive_usize(value: Option<&Value>, field: &str) -> IpcResult<usize> {
    let number = value.and_then(Value::as_u64).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{field} must be a positive integer."),
        )
    })?;
    let number = usize::try_from(number).map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{field} is too large for this runtime."),
        )
    })?;
    validate_speaker_count(number, field)?;
    Ok(number)
}

fn json_probability(value: Option<&Value>, field: &str) -> IpcResult<f32> {
    let number = value.and_then(Value::as_f64).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{field} must be a finite probability."),
        )
    })?;
    if !number.is_finite() || !(0.0..=1.0).contains(&number) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{field} must be between 0 and 1."),
        ));
    }
    Ok(number as f32)
}

fn segment_text(segment: &Map<String, Value>, field: &str, segment_id: &str) -> IpcResult<String> {
    let value = segment.get(field).and_then(Value::as_str).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("Transcript segment {segment_id}.{field} must be source-language text."),
        )
    })?;
    validate_text(
        value,
        &format!("transcript segment {segment_id}.{field}"),
        100_000,
        false,
    )?;
    Ok(value.to_owned())
}

fn project_transcript_document(
    document: &Value,
    expected_job_id: &str,
    policy: &SpeakerCountPolicy,
) -> IpcResult<TranscriptProjection> {
    let root = document.as_object().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript document root must be an object.",
        )
    })?;
    if root.get("schemaVersion").and_then(Value::as_str) != Some("2.0.0") {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript document schemaVersion must be 2.0.0.",
        ));
    }
    if root.get("jobId").and_then(Value::as_str) != Some(expected_job_id) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript document jobId does not match the active desktop task.",
        ));
    }
    let speaker_policy = root
        .get("speakerPolicy")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript document is missing speakerPolicy.",
            )
        })?;
    let resolved_count = json_positive_usize(
        speaker_policy.get("resolvedCount"),
        "transcript speakerPolicy.resolvedCount",
    )?;
    match policy {
        SpeakerCountPolicy::Manual { count } if *count != resolved_count => {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The verified transcript resolved count conflicts with the manually fixed count.",
            ));
        }
        SpeakerCountPolicy::Hybrid {
            min_speakers,
            max_speakers,
            ..
        } if resolved_count < *min_speakers || resolved_count > *max_speakers => {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The verified transcript resolved count violates the hybrid bounds.",
            ));
        }
        _ => {}
    }
    let speaker_ids = speaker_policy
        .get("speakerIds")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript speakerPolicy.speakerIds must be an array.",
            )
        })?;
    if speaker_ids.len() != resolved_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript speaker ID count does not match resolvedCount.",
        ));
    }
    for (index, value) in speaker_ids.iter().enumerate() {
        let expected = format!("speaker-{}", index + 1);
        if value.as_str() != Some(expected.as_str()) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript speaker IDs must be the contiguous canonical sequence.",
            ));
        }
    }
    let estimate = speaker_policy
        .get("estimate")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript speaker policy is missing count-estimate evidence.",
            )
        })?;
    let estimated_count = json_positive_usize(
        estimate.get("estimatedCount"),
        "transcript speakerPolicy.estimate.estimatedCount",
    )?;
    let confidence = json_probability(
        estimate.get("confidence"),
        "transcript speakerPolicy.estimate.confidence",
    )?;
    if estimated_count != resolved_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript count estimate does not match resolvedCount.",
        ));
    }
    let method = estimate
        .get("method")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript count estimate is missing its method.",
            )
        })?;
    validate_text(
        method,
        "transcript speakerPolicy.estimate.method",
        256,
        false,
    )?;
    let speaker_count_mode = speaker_policy
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript speakerPolicy.mode must be a string.",
            )
        })?;
    let expected_mode = match policy {
        SpeakerCountPolicy::Auto {} => "auto",
        SpeakerCountPolicy::Manual { .. } => "manual",
        SpeakerCountPolicy::Hybrid { .. } => "hybrid",
    };
    if speaker_count_mode != expected_mode {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript speakerPolicy.mode conflicts with the committed speaker-count policy.",
        ));
    }
    let candidate_range = estimate
        .get("candidateRange")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript count estimate is missing candidateRange.",
            )
        })?;
    let candidate_min = json_positive_usize(
        candidate_range.get("min"),
        "transcript speakerPolicy.estimate.candidateRange.min",
    )?;
    let candidate_max = json_positive_usize(
        candidate_range.get("max"),
        "transcript speakerPolicy.estimate.candidateRange.max",
    )?;
    if candidate_min > candidate_max
        || resolved_count < candidate_min
        || resolved_count > candidate_max
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript candidate range is inconsistent with resolvedCount.",
        ));
    }
    if let SpeakerCountPolicy::Hybrid {
        min_speakers,
        max_speakers,
        ..
    } = policy
    {
        if candidate_min < *min_speakers || candidate_max > *max_speakers {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript candidate range violates the hybrid bounds.",
            ));
        }
    }
    let speaker_values = root
        .get("speakers")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript speakers field must be an array.",
            )
        })?;
    if speaker_values.len() != resolved_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript speaker roster does not match resolvedCount.",
        ));
    }
    let mut speakers = Vec::with_capacity(resolved_count);
    for (index, value) in speaker_values.iter().enumerate() {
        let speaker = value.as_object().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Each transcript speaker must be an object.",
            )
        })?;
        let expected_id = format!("speaker-{}", index + 1);
        if speaker.get("id").and_then(Value::as_str) != Some(expected_id.as_str()) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript speaker roster must use contiguous canonical IDs.",
            ));
        }
        let label = speaker
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or(expected_id.as_str())
            .trim()
            .to_owned();
        validate_text(&label, "transcript speaker role", 64, false)?;
        let mut profile = speaker_profile(index, Some(&label));
        profile.role_hint = format!("Verified local diarization evidence · count method: {method}");
        profile.sample_status = SampleStatus::Ready;
        profile.review_status = SpeakerReviewStatus::NeedsReview;
        speakers.push(profile);
    }
    let duration_ms = root
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("durationMs"))
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript source.durationMs must be a non-negative integer.",
            )
        })?;
    if duration_ms == 0 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript source.durationMs must be positive.",
        ));
    }
    let segment_values = root
        .get("segments")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript segments field must be an array.",
            )
        })?;
    if segment_values.is_empty() {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The transcript must contain at least one verified speech segment.",
        ));
    }
    let mut segments = BTreeMap::new();
    for (index, value) in segment_values.iter().enumerate() {
        let segment = value.as_object().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("The transcript segment at index {index} must be an object."),
            )
        })?;
        let id = segment
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("The transcript segment at index {index} is missing id."),
                )
            })?
            .to_owned();
        validate_text(&id, "transcript segment.id", 160, false)?;
        let start_ms = segment
            .get("startMs")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Transcript segment {id} has invalid startMs."),
                )
            })?;
        let end_ms = segment
            .get("endMs")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Transcript segment {id} has invalid endMs."),
                )
            })?;
        if end_ms <= start_ms || end_ms > duration_ms {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Transcript segment {id} has out-of-range time boundaries."),
            ));
        }
        let speaker_id = segment
            .get("speakerId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Transcript segment {id} is missing speakerId."),
                )
            })?
            .to_owned();
        if !speaker_ids
            .iter()
            .any(|candidate| candidate.as_str() == Some(speaker_id.as_str()))
        {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Transcript segment {id} references an unknown speaker."),
            ));
        }
        let raw_text = segment_text(segment, "rawText", &id)?;
        let normalized_text = segment_text(segment, "normalizedText", &id)?;
        let display_text = segment_text(segment, "displayText", &id)?;
        let confidence =
            json_probability(segment.get("confidence"), "transcript segment.confidence")?;
        let human_locked = segment
            .get("humanLocked")
            .and_then(Value::as_bool)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Transcript segment {id} has invalid humanLocked evidence."),
                )
            })?;
        let speaker_score_values = segment
            .get("speakerScores")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Transcript segment {id} is missing speakerScores."),
                )
            })?;
        if speaker_score_values.len() != resolved_count {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!(
                    "Transcript segment {id} speakerScores do not cover the canonical speaker set."
                ),
            ));
        }
        let mut speaker_scores = Vec::with_capacity(resolved_count);
        for (score_index, score) in speaker_score_values.iter().enumerate() {
            let score = score.as_object().ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Transcript segment {id} has an invalid speaker score."),
                )
            })?;
            let expected_id = format!("speaker-{}", score_index + 1);
            if score.get("speakerId").and_then(Value::as_str) != Some(expected_id.as_str()) {
                return Err(IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!(
                        "Transcript segment {id} speakerScores must follow the canonical speaker order."
                    ),
                ));
            }
            let value = score.get("score").and_then(Value::as_f64).ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Transcript segment {id} has a non-numeric speaker score."),
                )
            })?;
            if !value.is_finite() || !(-2.0..=2.0).contains(&value) {
                return Err(IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Transcript segment {id} has an out-of-range speaker score."),
                ));
            }
            speaker_scores.push(TranscriptSpeakerScoreEvidence {
                speaker_id: expected_id,
                score: value as f32,
            });
        }
        let evidence = TranscriptSegmentEvidence {
            id: id.clone(),
            start_ms,
            end_ms,
            speaker_id,
            raw_text,
            normalized_text,
            display_text,
            confidence,
            speaker_scores,
            human_locked,
        };
        if segments.insert(id.clone(), evidence).is_some() {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Transcript segment IDs must be unique; duplicate {id}."),
            ));
        }
    }
    let speaker_detection = if matches!(policy, SpeakerCountPolicy::Manual { .. }) {
        None
    } else {
        Some(SpeakerCountDetection {
            estimated_count,
            confidence,
            candidates: vec![SpeakerCountCandidate {
                count: estimated_count,
                confidence,
            }],
            provider: Some(method.to_owned()),
        })
    };
    Ok(TranscriptProjection {
        speaker_count: resolved_count,
        speaker_detection,
        speakers,
        duration_label: format_duration_ms(duration_ms),
        duration_ms,
        speaker_count_mode: speaker_count_mode.to_owned(),
        speaker_count_estimate: Value::Object(estimate.clone()),
        segments,
    })
}

fn artifact_extension(path: &Path) -> IpcResult<String> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Worker artifacts must use an allowlisted file extension.",
            )
        })
}

fn artifact_kind_for_worker_type(artifact_type: &str, path: &Path) -> IpcResult<ArtifactKind> {
    let extension = artifact_extension(path)?;
    let require = |allowed: &[&str]| -> IpcResult<()> {
        if allowed.iter().any(|candidate| *candidate == extension) {
            Ok(())
        } else {
            Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Artifact type {artifact_type} is incompatible with .{extension} content."),
            ))
        }
    };
    let kind = match artifact_type {
        "transcript-document-v2"
        | "pipeline-metrics-v1"
        | "business-manifest-v1"
        | "business-variant-v1"
        | "pdf-render-manifest-v1"
        | "pdf-report-document-v1" => {
            require(&["json"])?;
            ArtifactKind::TranscriptJson
        }
        "review-queue-v2" | "pdf-repair-queue-v1" => {
            require(&["json"])?;
            ArtifactKind::RepairQueue
        }
        "pdf-quality-report-v1" => {
            require(&["json"])?;
            ArtifactKind::QualityReport
        }
        "pdf-contact-sheet" => {
            require(&["png", "jpg", "jpeg", "webp"])?;
            ArtifactKind::ContactSheet
        }
        "pdf-page-evidence" => {
            require(&["png", "jpg", "jpeg", "webp"])?;
            ArtifactKind::PageImage
        }
        "pdf" => {
            require(&["pdf"])?;
            ArtifactKind::Pdf
        }
        "pdf-canonical-xhtml" => {
            require(&["html", "xhtml"])?;
            ArtifactKind::TranscriptText
        }
        "pdf-render-artifact" => match extension.as_str() {
            "json" => ArtifactKind::TranscriptJson,
            "pdf" => ArtifactKind::Pdf,
            "png" | "jpg" | "jpeg" | "webp" => ArtifactKind::PageImage,
            "html" | "xhtml" | "txt" | "srt" | "vtt" => ArtifactKind::TranscriptText,
            _ => {
                return Err(IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Artifact type pdf-render-artifact does not allow .{extension} files."),
                ));
            }
        },
        _ => {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Unsupported worker artifact type {artifact_type}."),
            ));
        }
    };
    Ok(kind)
}

fn artifact_type_is_singleton(artifact_type: &str) -> bool {
    !matches!(
        artifact_type,
        "business-variant-v1" | "pdf-page-evidence" | "pdf-render-artifact"
    )
}

fn artifact_content(
    artifact_type: &str,
    path: &Path,
    announced_sha256: &str,
) -> IpcResult<(ArtifactKind, Option<Value>, u64)> {
    validate_sha256(announced_sha256, "artifact.created payload.sha256")?;
    let kind = artifact_kind_for_worker_type(artifact_type, path)?;
    let bytes = fs::read(path).map_err(|error| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("Unable to read the announced artifact bytes: {error}"),
        )
    })?;
    let size = u64::try_from(bytes.len()).map_err(|_| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The announced artifact is too large for this runtime.",
        )
    })?;
    let (document, actual_sha256) = if artifact_extension(path)? == "json" {
        let (document, canonical) = strict_json_document(&bytes, "the announced JSON artifact")?;
        let digest = Sha256::digest(canonical);
        (Some(document), format!("{digest:x}"))
    } else {
        let digest = Sha256::digest(&bytes);
        (None, format!("{digest:x}"))
    };
    if actual_sha256 != announced_sha256 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The announced artifact SHA-256 does not match its verified content.",
        ));
    }
    Ok((kind, document, size))
}

#[derive(Debug, Clone)]
struct ReviewQueueProjection {
    reviews: Vec<ReviewSegment>,
    worker_open_count: usize,
    open_segment_item_count: usize,
    open_segment_ids: BTreeSet<String>,
}

fn format_timestamp_ms(value: u64) -> String {
    let hours = value / 3_600_000;
    let minutes = (value / 60_000) % 60;
    let seconds = (value / 1_000) % 60;
    let milliseconds = value % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}")
}

fn review_reason(reason_code: &str) -> ReviewReason {
    match reason_code {
        "SPEAKER_MARGIN_LOW" => ReviewReason::SpeakerCloseScore,
        "SPEAKER_COUNT_LOW_CONFIDENCE" | "SPEAKER_COUNT_RANGE_WIDE" => {
            ReviewReason::SpeakerCountUncertain
        }
        "OVERLAP_REVIEW_REQUIRED" | "OVERLAP_DETECTOR_UNAVAILABLE" => ReviewReason::OverlapDetected,
        "BOUNDARY_CONFLICT" => ReviewReason::TimestampBoundary,
        "PROTECTED_SPEAKER_REVIEW_REQUIRED"
        | "SECONDARY_LOW_MARGIN_UNRESOLVED"
        | "SECONDARY_NO_REFERENCE"
        | "SECONDARY_REFERENCE_INSUFFICIENT"
        | "SECONDARY_VERIFIER_UNAVAILABLE"
        | "SECONDARY_VERIFIER_UNRESOLVED"
        | "SECONDARY_BUDGET_EXHAUSTED" => ReviewReason::SpeakerOutlier,
        _ => ReviewReason::LocalAudioReview,
    }
}

fn confidence_band(confidence: f32) -> ConfidenceBand {
    if confidence >= 0.85 {
        ConfidenceBand::High
    } else if confidence >= 0.65 {
        ConfidenceBand::Medium
    } else {
        ConfidenceBand::Low
    }
}

fn project_review_queue_document(
    document: &Value,
    expected_job_id: &str,
    evidence: &WorkerEvidenceLedger,
) -> IpcResult<ReviewQueueProjection> {
    let root = document.as_object().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review queue root must be an object.",
        )
    })?;
    exact_object_keys(
        root,
        &[
            "schemaVersion",
            "jobId",
            "speakerCountMode",
            "speakerCountEstimate",
            "createdAt",
            "updatedAt",
            "items",
            "decisions",
            "openCount",
        ],
        "The review queue document",
    )?;
    if root.get("schemaVersion").and_then(Value::as_str) != Some("2.0.0") {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review queue schemaVersion must be 2.0.0.",
        ));
    }
    if root.get("jobId").and_then(Value::as_str) != Some(expected_job_id) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review queue jobId does not match the committed task.",
        ));
    }
    let expected_mode = evidence
        .transcript_speaker_count_mode
        .as_deref()
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Review queue projection requires verified transcript speaker-count evidence.",
            )
        })?;
    if root.get("speakerCountMode").and_then(Value::as_str) != Some(expected_mode) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review queue speakerCountMode conflicts with the verified transcript.",
        ));
    }
    let expected_estimate = evidence
        .transcript_speaker_count_estimate
        .as_ref()
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Review queue projection requires verified transcript count-estimate evidence.",
            )
        })?;
    if root.get("speakerCountEstimate") != Some(expected_estimate) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review queue count estimate conflicts with the verified transcript.",
        ));
    }
    for field in ["createdAt", "updatedAt"] {
        let timestamp = root.get(field).and_then(Value::as_str).ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("The review queue {field} must be a timestamp string."),
            )
        })?;
        validate_text(timestamp, &format!("review queue {field}"), 128, false)?;
    }
    let announced_open_count = root
        .get("openCount")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The review queue openCount must be a non-negative integer.",
            )
        })?;
    let decisions = root
        .get("decisions")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The review queue decisions field must be an array.",
            )
        })?;
    let mut decision_ids = BTreeSet::new();
    for decision in decisions {
        let decision = decision.as_object().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Every review queue decision must be an object.",
            )
        })?;
        let decision_id = decision
            .get("decisionId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "Every review queue decision must include decisionId.",
                )
            })?;
        validate_text(decision_id, "review queue decision.decisionId", 160, false)?;
        if !decision_ids.insert(decision_id.to_owned()) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Review queue decision IDs must be unique.",
            ));
        }
    }
    let items = root.get("items").and_then(Value::as_array).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review queue items field must be an array.",
        )
    })?;
    let mut item_ids = BTreeSet::new();
    let mut reviews = Vec::new();
    let mut counted_open = 0usize;
    let mut open_segment_item_count = 0usize;
    let mut open_segment_ids = BTreeSet::new();
    for item in items {
        let item = item.as_object().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Every review queue item must be an object.",
            )
        })?;
        let item_id = item.get("id").and_then(Value::as_str).ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Every review queue item must include id.",
            )
        })?;
        validate_text(item_id, "review queue item.id", 256, false)?;
        if !item_ids.insert(item_id.to_owned()) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Review queue item IDs must be unique.",
            ));
        }
        let status = item.get("status").and_then(Value::as_str).ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Review queue item {item_id} is missing status."),
            )
        })?;
        let is_open = match status {
            "open" => true,
            "accepted" | "rejected" => {
                if !item.get("decision").is_some_and(Value::is_object) {
                    return Err(IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        format!(
                            "Resolved review queue item {item_id} must contain a human decision."
                        ),
                    ));
                }
                false
            }
            _ => {
                return Err(IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Review queue item {item_id} has unsupported status {status}."),
                ));
            }
        };
        if is_open {
            counted_open += 1;
        }
        let reason_code = item
            .get("reasonCode")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Review queue item {item_id} is missing reasonCode."),
                )
            })?;
        validate_text(reason_code, "review queue item.reasonCode", 160, false)?;
        match item.get("scope").and_then(Value::as_str) {
            Some("job") => {
                if item.get("segmentId").is_some() {
                    return Err(IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        format!("Job-scope review item {item_id} must not fabricate a segmentId."),
                    ));
                }
                if item.get("speakerCountEstimate") != Some(expected_estimate) {
                    return Err(IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        format!(
                            "Job-scope review item {item_id} count evidence conflicts with the transcript."
                        ),
                    ));
                }
            }
            Some("segment") => {
                let segment_id =
                    item.get("segmentId")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            IpcError::new(
                                IpcErrorCode::StateUnavailable,
                                format!(
                                    "Segment-scope review item {item_id} is missing segmentId."
                                ),
                            )
                        })?;
                let segment = evidence
                    .transcript_segments
                    .get(segment_id)
                    .ok_or_else(|| {
                        IpcError::new(
                            IpcErrorCode::StateUnavailable,
                            format!(
                            "Review queue item {item_id} references unknown segment {segment_id}."
                        ),
                        )
                    })?;
                if segment.id != segment_id {
                    return Err(IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        format!(
                            "Review queue item {item_id} segment identity conflicts with transcript evidence."
                        ),
                    ));
                }
                let time_range = item
                    .get("timeRange")
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        IpcError::new(
                            IpcErrorCode::StateUnavailable,
                            format!("Review queue item {item_id} is missing timeRange."),
                        )
                    })?;
                if time_range.get("startMs").and_then(Value::as_u64) != Some(segment.start_ms)
                    || time_range.get("endMs").and_then(Value::as_u64) != Some(segment.end_ms)
                {
                    return Err(IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        format!(
                            "Review queue item {item_id} time boundaries conflict with the transcript."
                        ),
                    ));
                }
                if item.get("speakerId").and_then(Value::as_str)
                    != Some(segment.speaker_id.as_str())
                {
                    return Err(IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        format!(
                            "Review queue item {item_id} speaker assignment conflicts with the transcript."
                        ),
                    ));
                }
                let text = item.get("text").and_then(Value::as_object).ok_or_else(|| {
                    IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        format!("Review queue item {item_id} is missing text evidence."),
                    )
                })?;
                if text.get("rawText").and_then(Value::as_str) != Some(segment.raw_text.as_str())
                    || text.get("normalizedText").and_then(Value::as_str)
                        != Some(segment.normalized_text.as_str())
                    || text.get("displayText").and_then(Value::as_str)
                        != Some(segment.display_text.as_str())
                {
                    return Err(IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        format!(
                            "Review queue item {item_id} text evidence conflicts with the transcript."
                        ),
                    ));
                }
                let candidate_values = item
                    .get("speakerCandidates")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        IpcError::new(
                            IpcErrorCode::StateUnavailable,
                            format!("Review queue item {item_id} is missing speakerCandidates."),
                        )
                    })?;
                if candidate_values.len() != segment.speaker_scores.len() {
                    return Err(IpcError::new(
                        IpcErrorCode::StateUnavailable,
                        format!("Review queue item {item_id} speakerCandidates are incomplete."),
                    ));
                }
                let mut candidates = Vec::with_capacity(candidate_values.len());
                for (candidate, expected) in candidate_values.iter().zip(&segment.speaker_scores) {
                    let candidate = candidate.as_object().ok_or_else(|| {
                        IpcError::new(
                            IpcErrorCode::StateUnavailable,
                            format!(
                                "Review queue item {item_id} has an invalid speaker candidate."
                            ),
                        )
                    })?;
                    let score =
                        candidate
                            .get("score")
                            .and_then(Value::as_f64)
                            .ok_or_else(|| {
                                IpcError::new(
                                    IpcErrorCode::StateUnavailable,
                                    format!(
                                    "Review queue item {item_id} has a non-numeric speaker score."
                                ),
                                )
                            })?;
                    if candidate.get("speakerId").and_then(Value::as_str)
                        != Some(expected.speaker_id.as_str())
                        || !score.is_finite()
                        || (score - f64::from(expected.score)).abs() > 1e-6
                    {
                        return Err(IpcError::new(
                            IpcErrorCode::StateUnavailable,
                            format!(
                                "Review queue item {item_id} speaker evidence conflicts with the transcript."
                            ),
                        ));
                    }
                    candidates.push(SpeakerCandidate {
                        speaker_id: expected.speaker_id.clone(),
                        score: expected.score,
                        evidence: "Verified transcript speakerScores evidence".to_owned(),
                    });
                }
                let evidence_refs = item
                    .get("evidenceRefs")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        IpcError::new(
                            IpcErrorCode::StateUnavailable,
                            format!("Review queue item {item_id} is missing evidenceRefs."),
                        )
                    })?;
                for reference in evidence_refs {
                    let reference = reference.as_str().ok_or_else(|| {
                        IpcError::new(
                            IpcErrorCode::StateUnavailable,
                            format!(
                                "Review queue item {item_id} contains a non-string evidence reference."
                            ),
                        )
                    })?;
                    validate_text(reference, "review queue evidenceRef", 4096, false)?;
                }
                if is_open {
                    open_segment_item_count += 1;
                    open_segment_ids.insert(segment_id.to_owned());
                }
                reviews.push(ReviewSegment {
                    id: item_id.to_owned(),
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                    timestamp_label: format!(
                        "{}–{}",
                        format_timestamp_ms(segment.start_ms),
                        format_timestamp_ms(segment.end_ms)
                    ),
                    raw_text: segment.raw_text.clone(),
                    normalized_text: segment.normalized_text.clone(),
                    current_speaker_id: segment.speaker_id.clone(),
                    candidates,
                    reasons: vec![review_reason(reason_code)],
                    confidence: segment.confidence,
                    confidence_band: confidence_band(segment.confidence),
                    waveform: Vec::new(),
                    locked: segment.human_locked || !is_open,
                    reviewed: !is_open,
                    audit_trail: Vec::new(),
                });
            }
            _ => {
                return Err(IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    format!("Review queue item {item_id} has unsupported scope."),
                ));
            }
        }
    }
    if counted_open != announced_open_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review queue openCount does not match its open item count.",
        ));
    }
    Ok(ReviewQueueProjection {
        reviews,
        worker_open_count: counted_open,
        open_segment_item_count,
        open_segment_ids,
    })
}

#[derive(Debug, Clone)]
struct PipelineMetricsProjection {
    duration_ms: u64,
    segment_count: usize,
    reference_quality: Option<ReferenceQualityEvidence>,
}

fn json_object_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    context: &str,
) -> IpcResult<&'a Map<String, Value>> {
    object.get(field).and_then(Value::as_object).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{context}.{field} must be an object."),
        )
    })
}

fn json_array_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    context: &str,
) -> IpcResult<&'a Vec<Value>> {
    object.get(field).and_then(Value::as_array).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{context}.{field} must be an array."),
        )
    })
}

fn json_nonnegative_f64(object: &Map<String, Value>, field: &str, context: &str) -> IpcResult<f64> {
    let value = object.get(field).and_then(Value::as_f64).ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{context}.{field} must be numeric."),
        )
    })?;
    if !value.is_finite() || value < 0.0 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!("{context}.{field} must be finite and non-negative."),
        ));
    }
    Ok(value)
}

fn json_nonnegative_usize(
    object: &Map<String, Value>,
    field: &str,
    context: &str,
) -> IpcResult<usize> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("{context}.{field} must be a non-negative integer."),
            )
        })
}

fn rate_matches(actual: f64, numerator: usize, denominator: usize) -> bool {
    let expected = if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    };
    (actual - expected).abs() <= 1e-8
}

fn project_pipeline_metrics_document(
    document: &Value,
    expected_job_id: &str,
    evidence: &WorkerEvidenceLedger,
) -> IpcResult<PipelineMetricsProjection> {
    let root = document.as_object().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics root must be an object.",
        )
    })?;
    let reference_available = root
        .get("referenceEvaluation")
        .and_then(Value::as_object)
        .and_then(|reference| reference.get("available"))
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "pipeline metrics referenceEvaluation.available must be boolean.",
            )
        })?;
    let expected_top_level = if reference_available {
        &[
            "schemaVersion",
            "jobId",
            "offline",
            "durationMs",
            "runtime",
            "cache",
            "routing",
            "escalations",
            "cascade",
            "resources",
            "policy",
            "referenceEvaluation",
            "quality",
        ][..]
    } else {
        &[
            "schemaVersion",
            "jobId",
            "offline",
            "durationMs",
            "runtime",
            "cache",
            "routing",
            "escalations",
            "cascade",
            "resources",
            "policy",
            "referenceEvaluation",
        ][..]
    };
    exact_object_keys(root, expected_top_level, "The pipeline metrics document")?;
    if root.get("schemaVersion").and_then(Value::as_str) != Some("1.0.0") {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics schemaVersion must be 1.0.0.",
        ));
    }
    if root.get("jobId").and_then(Value::as_str) != Some(expected_job_id) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics jobId does not match the committed task.",
        ));
    }
    if root.get("offline").and_then(Value::as_bool) != Some(true) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The production pipeline metrics must prove offline execution.",
        ));
    }
    let duration_ms = root
        .get("durationMs")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "pipeline metrics durationMs must be a positive integer.",
            )
        })?;
    if duration_ms == 0 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "pipeline metrics durationMs must be positive.",
        ));
    }
    if evidence
        .transcript_duration_ms
        .is_some_and(|transcript_duration| transcript_duration != duration_ms)
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics duration conflicts with verified transcript evidence.",
        ));
    }

    let runtime = json_object_field(root, "runtime", "pipeline metrics")?;
    exact_object_keys(
        runtime,
        &["elapsedMs", "rtf", "stages"],
        "The pipeline metrics runtime",
    )?;
    let elapsed_ms = json_nonnegative_f64(runtime, "elapsedMs", "pipeline metrics runtime")?;
    let rtf = json_nonnegative_f64(runtime, "rtf", "pipeline metrics runtime")?;
    if (rtf - elapsed_ms / duration_ms as f64).abs() > 1e-8 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics RTF is inconsistent with elapsedMs and durationMs.",
        ));
    }
    let stages = json_object_field(runtime, "stages", "pipeline metrics runtime")?;
    for (stage_name, stage) in stages {
        validate_text(stage_name, "pipeline metrics stage name", 160, false)?;
        let stage = stage.as_object().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Pipeline metrics stage {stage_name} must be an object."),
            )
        })?;
        exact_object_keys(
            stage,
            &["count", "totalMs", "p50Ms", "p95Ms"],
            "A pipeline metrics stage",
        )?;
        let count = json_nonnegative_usize(stage, "count", "pipeline metrics stage")?;
        if count == 0 {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Pipeline metrics stage {stage_name} has zero samples."),
            ));
        }
        let total = json_nonnegative_f64(stage, "totalMs", "pipeline metrics stage")?;
        let p50 = json_nonnegative_f64(stage, "p50Ms", "pipeline metrics stage")?;
        let p95 = json_nonnegative_f64(stage, "p95Ms", "pipeline metrics stage")?;
        if p50 > p95 + 1e-6 || p95 > total + 1e-6 {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Pipeline metrics stage {stage_name} percentiles are inconsistent."),
            ));
        }
    }

    let cache = json_object_field(root, "cache", "pipeline metrics")?;
    exact_object_keys(
        cache,
        &[
            "requests",
            "hits",
            "misses",
            "recomputations",
            "hitRate",
            "recomputationRate",
            "byStage",
        ],
        "The pipeline metrics cache",
    )?;
    let requests = json_nonnegative_usize(cache, "requests", "pipeline metrics cache")?;
    let hits = json_nonnegative_usize(cache, "hits", "pipeline metrics cache")?;
    let misses = json_nonnegative_usize(cache, "misses", "pipeline metrics cache")?;
    let recomputations = json_nonnegative_usize(cache, "recomputations", "pipeline metrics cache")?;
    if hits.checked_add(misses) != Some(requests) || recomputations > requests {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics aggregate cache counters are inconsistent.",
        ));
    }
    let hit_rate = json_nonnegative_f64(cache, "hitRate", "pipeline metrics cache")?;
    let recomputation_rate =
        json_nonnegative_f64(cache, "recomputationRate", "pipeline metrics cache")?;
    if !rate_matches(hit_rate, hits, requests)
        || !rate_matches(recomputation_rate, recomputations, requests)
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics aggregate cache rates are inconsistent.",
        ));
    }
    let by_stage = json_object_field(cache, "byStage", "pipeline metrics cache")?;
    let mut stage_requests = 0usize;
    let mut stage_hits = 0usize;
    let mut stage_misses = 0usize;
    let mut stage_recomputations = 0usize;
    for (stage_name, stage) in by_stage {
        validate_text(stage_name, "pipeline metrics cache stage", 160, false)?;
        let stage = stage.as_object().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Pipeline cache stage {stage_name} must be an object."),
            )
        })?;
        exact_object_keys(
            stage,
            &[
                "requests",
                "hits",
                "misses",
                "recomputations",
                "hitRate",
                "recomputationRate",
            ],
            "A pipeline metrics cache stage",
        )?;
        let current_requests =
            json_nonnegative_usize(stage, "requests", "pipeline metrics cache stage")?;
        let current_hits = json_nonnegative_usize(stage, "hits", "pipeline metrics cache stage")?;
        let current_misses =
            json_nonnegative_usize(stage, "misses", "pipeline metrics cache stage")?;
        let current_recomputations =
            json_nonnegative_usize(stage, "recomputations", "pipeline metrics cache stage")?;
        if current_hits.checked_add(current_misses) != Some(current_requests)
            || current_recomputations > current_requests
        {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Pipeline cache stage {stage_name} counters are inconsistent."),
            ));
        }
        if !rate_matches(
            json_nonnegative_f64(stage, "hitRate", "pipeline metrics cache stage")?,
            current_hits,
            current_requests,
        ) || !rate_matches(
            json_nonnegative_f64(stage, "recomputationRate", "pipeline metrics cache stage")?,
            current_recomputations,
            current_requests,
        ) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Pipeline cache stage {stage_name} rates are inconsistent."),
            ));
        }
        stage_requests = stage_requests
            .checked_add(current_requests)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "Pipeline cache request totals overflowed this runtime.",
                )
            })?;
        stage_hits = stage_hits.checked_add(current_hits).ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Pipeline cache hit totals overflowed this runtime.",
            )
        })?;
        stage_misses = stage_misses.checked_add(current_misses).ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Pipeline cache miss totals overflowed this runtime.",
            )
        })?;
        stage_recomputations = stage_recomputations
            .checked_add(current_recomputations)
            .ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "Pipeline cache recomputation totals overflowed this runtime.",
                )
            })?;
    }
    if (
        stage_requests,
        stage_hits,
        stage_misses,
        stage_recomputations,
    ) != (requests, hits, misses, recomputations)
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics by-stage cache totals do not match the aggregate.",
        ));
    }

    let routing = json_object_field(root, "routing", "pipeline metrics")?;
    exact_object_keys(
        routing,
        &[
            "segments",
            "escalated",
            "escalationRate",
            "protected",
            "reasonCounts",
        ],
        "The pipeline metrics routing",
    )?;
    let segment_count = json_nonnegative_usize(routing, "segments", "pipeline metrics routing")?;
    let escalated = json_nonnegative_usize(routing, "escalated", "pipeline metrics routing")?;
    let protected = json_nonnegative_usize(routing, "protected", "pipeline metrics routing")?;
    if escalated > segment_count || protected > segment_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics routing counters exceed the segment count.",
        ));
    }
    if !rate_matches(
        json_nonnegative_f64(routing, "escalationRate", "pipeline metrics routing")?,
        escalated,
        segment_count,
    ) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics escalation rate is inconsistent.",
        ));
    }
    let reason_counts = json_object_field(routing, "reasonCounts", "pipeline metrics routing")?;
    for (reason, count) in reason_counts {
        validate_text(reason, "pipeline metrics routing reason", 160, false)?;
        if count.as_u64().is_none() {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Pipeline routing reason {reason} must have an integer count."),
            ));
        }
    }
    if evidence
        .transcript_segment_count
        .is_some_and(|count| count != segment_count)
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The pipeline metrics segment count conflicts with verified transcript evidence.",
        ));
    }

    json_array_field(root, "escalations", "pipeline metrics")?;
    json_array_field(root, "cascade", "pipeline metrics")?;
    json_object_field(root, "policy", "pipeline metrics")?;
    let resources = json_object_field(root, "resources", "pipeline metrics")?;
    exact_object_keys(
        resources,
        &["peakRamMb", "peakVramMb"],
        "The pipeline metrics resources",
    )?;
    json_nonnegative_f64(resources, "peakRamMb", "pipeline metrics resources")?;
    json_nonnegative_f64(resources, "peakVramMb", "pipeline metrics resources")?;
    let reference = json_object_field(root, "referenceEvaluation", "pipeline metrics")?;
    exact_object_keys(
        reference,
        &["available"],
        "The pipeline metrics referenceEvaluation",
    )?;
    let reference_quality = if reference_available {
        let quality = json_object_field(root, "quality", "pipeline metrics")?;
        exact_object_keys(
            quality,
            &["der", "jer", "speakerConfusion", "overlapF1"],
            "The pipeline metrics quality",
        )?;
        let der = json_nonnegative_f64(quality, "der", "pipeline metrics quality")?;
        let jer = json_nonnegative_f64(quality, "jer", "pipeline metrics quality")?;
        let speaker_confusion =
            json_nonnegative_f64(quality, "speakerConfusion", "pipeline metrics quality")?;
        let overlap_f1 = json_nonnegative_f64(quality, "overlapF1", "pipeline metrics quality")?;
        if overlap_f1 > 1.0 {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The pipeline metrics overlapF1 must be between 0 and 1.",
            ));
        }
        Some(ReferenceQualityEvidence {
            der: (der * 100.0) as f32,
            jer: (jer * 100.0) as f32,
            speaker_confusion: (speaker_confusion * 100.0) as f32,
            overlap_f1: (overlap_f1 * 100.0) as f32,
        })
    } else {
        if root.contains_key("quality") {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Pipeline quality metrics cannot exist without reference labels.",
            ));
        }
        None
    };
    Ok(PipelineMetricsProjection {
        duration_ms,
        segment_count,
        reference_quality,
    })
}

fn available_percent(value: f32, source: &str) -> PercentMetric {
    PercentMetric::Available {
        value,
        unit: PercentUnit::Percent,
        source: source.to_owned(),
    }
}

fn apply_reference_quality(
    snapshot: &mut StudioSnapshot,
    quality: Option<ReferenceQualityEvidence>,
) {
    if let Some(quality) = quality {
        let source = "Verified pipeline-metrics-v1 reference-label evaluation";
        snapshot.diarization_quality.der = available_percent(quality.der, source);
        snapshot.diarization_quality.jer = available_percent(quality.jer, source);
        snapshot.diarization_quality.confusion =
            available_percent(quality.speaker_confusion, source);
        snapshot.diarization_quality.overlap_f1 = available_percent(quality.overlap_f1, source);
    } else {
        snapshot.diarization_quality.der = unavailable_reference_metric("DER");
        snapshot.diarization_quality.jer = unavailable_reference_metric("JER");
        snapshot.diarization_quality.confusion = unavailable_reference_metric("confusion");
        snapshot.diarization_quality.overlap_f1 = unavailable_reference_metric("overlap F1");
    }
}

fn refresh_review_rate(snapshot: &mut StudioSnapshot, evidence: &WorkerEvidenceLedger) {
    snapshot.diarization_quality.review_rate = match (
        evidence.pipeline_segment_count,
        evidence.worker_review_open_count,
    ) {
        (Some(segment_count), Some(_)) if segment_count > 0 => available_percent(
            evidence.open_review_segment_ids.len() as f32 * 100.0 / segment_count as f32,
            "Unique open review segment IDs / verified pipeline routing segments",
        ),
        (Some(0), Some(0)) => available_percent(
            0.0,
            "No routed speech segments and no open segment-scope reviews",
        ),
        (Some(0), Some(_)) => PercentMetric::Unavailable {
            reason: "Open segment review evidence cannot be reconciled with zero routed segments."
                .to_owned(),
        },
        _ => PercentMetric::Unavailable {
            reason:
                "Review rate requires both verified pipeline metrics and the durable review queue."
                    .to_owned(),
        },
    };
}

#[derive(Debug)]
struct PreparedArtifactProjection {
    item: ArtifactItem,
    evidence: VerifiedArtifactEvidence,
    transcript: Option<TranscriptProjection>,
    review_queue: Option<ReviewQueueProjection>,
    pipeline_metrics: Option<PipelineMetricsProjection>,
}

fn prepare_artifact_projection(
    event: &WorkerEvent,
    output_root: &Path,
    policy: &SpeakerCountPolicy,
    evidence: &WorkerEvidenceLedger,
) -> IpcResult<PreparedArtifactProjection> {
    exact_object_keys(
        &event.payload,
        &["artifactType", "path", "sha256"],
        "The artifact.created payload",
    )?;
    let artifact_type = event_string(&event.payload, "artifactType", 160)?;
    let raw_path = event_string(&event.payload, "path", 4096)?;
    let announced_sha256 = event_string(&event.payload, "sha256", 64)?;
    let canonical_path = resolve_worker_artifact_path(output_root, raw_path)?;
    let relative_path = relative_worker_artifact_path(output_root, &canonical_path)?;
    let (kind, document, size) =
        artifact_content(artifact_type, &canonical_path, announced_sha256)?;
    let transcript = match (artifact_type, document.as_ref()) {
        ("transcript-document-v2", Some(document)) => Some(project_transcript_document(
            document,
            &event.job_id,
            policy,
        )?),
        _ => None,
    };
    if let Some(transcript) = transcript.as_ref() {
        if evidence
            .pipeline_duration_ms
            .is_some_and(|duration| duration != transcript.duration_ms)
            || evidence
                .pipeline_segment_count
                .is_some_and(|count| count != transcript.segments.len())
        {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The verified transcript conflicts with previously accepted pipeline metrics.",
            ));
        }
    }
    let review_queue = match (artifact_type, document.as_ref()) {
        ("review-queue-v2", Some(document)) => Some(project_review_queue_document(
            document,
            &event.job_id,
            evidence,
        )?),
        _ => None,
    };
    let pipeline_metrics = match (artifact_type, document.as_ref()) {
        ("pipeline-metrics-v1", Some(document)) => Some(project_pipeline_metrics_document(
            document,
            &event.job_id,
            evidence,
        )?),
        _ => None,
    };
    let name = canonical_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(artifact_type)
        .to_owned();
    Ok(PreparedArtifactProjection {
        item: ArtifactItem {
            id: event.event_id.clone(),
            name,
            kind,
            relative_path,
            size_label: format!("{size} bytes"),
            created_at: event.timestamp.clone(),
            integrity: IntegrityStatus::Verified,
            sha256: Some(announced_sha256.to_owned()),
        },
        evidence: VerifiedArtifactEvidence {
            artifact_type: artifact_type.to_owned(),
            canonical_path,
            sha256: announced_sha256.to_owned(),
        },
        transcript,
        review_queue,
        pipeline_metrics,
    })
}

fn validate_artifact_ledger_update(
    ledger: &WorkerEvidenceLedger,
    candidate: &VerifiedArtifactEvidence,
) -> IpcResult<()> {
    if let Some(existing) = ledger.verified_artifacts.get(&candidate.canonical_path) {
        if existing.artifact_type != candidate.artifact_type || existing.sha256 != candidate.sha256
        {
            return Err(IpcError::new(
                IpcErrorCode::Conflict,
                "A worker artifact path was already bound to different type or content evidence.",
            ));
        }
    }
    if artifact_type_is_singleton(&candidate.artifact_type)
        && ledger.verified_artifacts.values().any(|existing| {
            existing.artifact_type == candidate.artifact_type
                && existing.canonical_path != candidate.canonical_path
        })
    {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            format!(
                "Singleton artifact type {} was announced at more than one path.",
                candidate.artifact_type
            ),
        ));
    }
    Ok(())
}

fn preserve_review_audit_trails(projected: &mut [ReviewSegment], existing: &[ReviewSegment]) {
    let audit_by_id = existing
        .iter()
        .map(|review| (review.id.as_str(), review.audit_trail.clone()))
        .collect::<BTreeMap<_, _>>();
    for review in projected {
        if let Some(audit) = audit_by_id.get(review.id.as_str()) {
            review.audit_trail.clone_from(audit);
        }
    }
}

fn commit_artifact_projection(
    state: &mut AppState,
    prepared: PreparedArtifactProjection,
) -> IpcResult<()> {
    validate_artifact_ledger_update(&state.worker_evidence, &prepared.evidence)?;
    let mut next_snapshot = state.snapshot.clone();
    let mut next_evidence = state.worker_evidence.clone();

    if let Some(transcript) = prepared.transcript {
        if next_evidence
            .pipeline_duration_ms
            .is_some_and(|duration| duration != transcript.duration_ms)
            || next_evidence
                .pipeline_segment_count
                .is_some_and(|count| count != transcript.segments.len())
        {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The transcript artifact no longer reconciles with committed pipeline evidence.",
            ));
        }
        next_snapshot.job.speaker_count = Some(transcript.speaker_count);
        next_snapshot.job.speaker_detection = transcript.speaker_detection;
        next_snapshot.job.duration_label = transcript.duration_label;
        next_snapshot.speakers = transcript.speakers;
        next_evidence.transcript_duration_ms = Some(transcript.duration_ms);
        next_evidence.transcript_segment_count = Some(transcript.segments.len());
        next_evidence.transcript_speaker_count_mode = Some(transcript.speaker_count_mode);
        next_evidence.transcript_speaker_count_estimate = Some(transcript.speaker_count_estimate);
        next_evidence.transcript_segments = transcript.segments;
    }

    if let Some(mut review_queue) = prepared.review_queue {
        preserve_review_audit_trails(&mut review_queue.reviews, &next_snapshot.reviews);
        next_snapshot.reviews = review_queue.reviews;
        next_snapshot.job.review_open_count = review_queue.open_segment_item_count;
        next_evidence.worker_review_open_count = Some(review_queue.worker_open_count);
        next_evidence.open_review_segment_item_count = review_queue.open_segment_item_count;
        next_evidence.open_review_segment_ids = review_queue.open_segment_ids;
    }

    if let Some(metrics) = prepared.pipeline_metrics {
        if next_evidence
            .transcript_duration_ms
            .is_some_and(|duration| duration != metrics.duration_ms)
            || next_evidence
                .transcript_segment_count
                .is_some_and(|count| count != metrics.segment_count)
        {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The pipeline metrics artifact no longer reconciles with committed transcript evidence.",
            ));
        }
        next_evidence.pipeline_duration_ms = Some(metrics.duration_ms);
        next_evidence.pipeline_segment_count = Some(metrics.segment_count);
        next_evidence.reference_quality = metrics.reference_quality;
        apply_reference_quality(&mut next_snapshot, metrics.reference_quality);
        next_snapshot.performance = PerformanceMetrics::Unavailable {
            reason: "Verified pipeline-metrics-v1 does not report measured peak CPU, so the complete desktop performance contract cannot be truthfully populated."
                .to_owned(),
        };
    }

    next_evidence
        .verified_artifacts
        .insert(prepared.evidence.canonical_path.clone(), prepared.evidence);
    if let Some(existing) = next_snapshot
        .artifacts
        .iter_mut()
        .find(|item| item.relative_path == prepared.item.relative_path)
    {
        *existing = prepared.item;
    } else {
        next_snapshot.artifacts.push(prepared.item);
    }
    refresh_review_rate(&mut next_snapshot, &next_evidence);
    state.snapshot = next_snapshot;
    state.worker_evidence = next_evidence;
    Ok(())
}

fn verified_artifact_for_type<'a>(
    evidence: &'a WorkerEvidenceLedger,
    artifact_type: &str,
) -> IpcResult<&'a VerifiedArtifactEvidence> {
    let matches = evidence
        .verified_artifacts
        .values()
        .filter(|artifact| artifact.artifact_type == artifact_type)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!(
                "Exactly one verified {artifact_type} artifact is required; found {}.",
                matches.len()
            ),
        ));
    }
    Ok(matches[0])
}

fn validate_review_required_event(
    event: &WorkerEvent,
    output_root: &Path,
    evidence: &WorkerEvidenceLedger,
) -> IpcResult<usize> {
    exact_object_keys(
        &event.payload,
        &["reviewQueuePath", "openCount"],
        "The review.required payload",
    )?;
    let announced_count =
        event_nonnegative_integer(&event.payload, "openCount").and_then(|value| {
            usize::try_from(value).map_err(|_| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "Worker review openCount is too large for this runtime.",
                )
            })
        })?;
    if announced_count == 0 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "review.required cannot announce zero durable open review items.",
        ));
    }
    if evidence.worker_review_open_count != Some(announced_count) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "review.required openCount does not match the verified durable review queue.",
        ));
    }
    let announced_path = resolve_worker_artifact_path(
        output_root,
        event_string(&event.payload, "reviewQueuePath", 4096)?,
    )?;
    let queue_artifact = verified_artifact_for_type(evidence, "review-queue-v2")?;
    if announced_path != queue_artifact.canonical_path {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "review.required points to a path other than the verified review queue.",
        ));
    }
    artifact_content(
        &queue_artifact.artifact_type,
        &queue_artifact.canonical_path,
        &queue_artifact.sha256,
    )?;
    Ok(announced_count)
}

fn validate_persisted_human_decision_event(event: &WorkerEvent, state: &AppState) -> IpcResult<()> {
    exact_object_keys(
        &event.payload,
        &[
            "jobId",
            "status",
            "command",
            "decision",
            "openCount",
            "documentHash",
            "speakerCount",
        ],
        "The review.decision.persisted payload",
    )?;
    if event.payload.get("jobId").and_then(Value::as_str) != Some(event.job_id.as_str()) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted human-decision event jobId does not match its event envelope.",
        ));
    }
    if event.payload.get("status").and_then(Value::as_str) != Some("review_required") {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "A persisted human-decision event must preserve review_required status.",
        ));
    }
    let command = event
        .payload
        .get("command")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The persisted human-decision event command must be a string.",
            )
        })?;
    if !matches!(command, "review.submit" | "speaker.rename") {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted human-decision event command is unsupported.",
        ));
    }
    let decision = event
        .payload
        .get("decision")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The persisted human-decision event is missing its audit decision.",
            )
        })?;
    exact_object_keys(
        decision,
        &[
            "decisionId",
            "command",
            "reason",
            "evidence",
            "confidence",
            "audit",
            "recordedAt",
        ],
        "The persisted human-decision audit",
    )?;
    if decision.get("command").and_then(Value::as_str) != Some(command) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted human-decision audit command conflicts with its event payload.",
        ));
    }
    for field in ["decisionId", "reason", "recordedAt"] {
        let value = decision.get(field).and_then(Value::as_str).ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("The persisted human-decision audit {field} must be a string."),
            )
        })?;
        validate_text(
            value,
            &format!("persisted human-decision audit.{field}"),
            if field == "reason" { 4096 } else { 160 },
            false,
        )?;
    }
    let evidence = decision
        .get("evidence")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The persisted human-decision audit evidence must be an array.",
            )
        })?;
    if evidence.is_empty() {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted human-decision audit must contain evidence.",
        ));
    }
    for item in evidence {
        let item = item.as_str().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "Persisted human-decision evidence entries must be strings.",
            )
        })?;
        validate_text(item, "persisted human-decision evidence", 4096, false)?;
    }
    let confidence = decision
        .get("confidence")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The persisted human-decision confidence must be numeric.",
            )
        })?;
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted human-decision confidence must be between zero and one.",
        ));
    }
    let audit = decision
        .get("audit")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The persisted human-decision audit metadata must be an object.",
            )
        })?;
    if audit.get("jobId").and_then(Value::as_str) != Some(event.job_id.as_str())
        || audit.get("actor").and_then(Value::as_str) != Some("desktop-user")
        || audit.get("source").and_then(Value::as_str) != Some("human")
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted human-decision audit metadata lacks committed human/job evidence.",
        ));
    }
    event_nonnegative_integer(&event.payload, "openCount")?;
    let document_hash = event_string(&event.payload, "documentHash", 64)?;
    validate_sha256(
        document_hash,
        "review.decision.persisted payload.documentHash",
    )?;
    let speaker_count =
        event_nonnegative_integer(&event.payload, "speakerCount").and_then(|value| {
            usize::try_from(value).map_err(|_| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "Persisted human-decision speakerCount is too large.",
                )
            })
        })?;
    if Some(speaker_count) != state.snapshot.job.speaker_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted human-decision speakerCount conflicts with the committed task.",
        ));
    }
    Ok(())
}

const REQUIRED_COMPLETION_ARTIFACT_TYPES: &[&str] = &[
    "transcript-document-v2",
    "review-queue-v2",
    "pdf",
    "pdf-quality-report-v1",
    "pdf-render-manifest-v1",
    "pdf-report-document-v1",
    "pdf-canonical-xhtml",
    "pdf-repair-queue-v1",
    "pdf-contact-sheet",
];

#[derive(Debug)]
struct ReconciledMutationEvidence {
    transcript_path: PathBuf,
    transcript_sha256: String,
    transcript: TranscriptProjection,
    review_queue_path: PathBuf,
    review_queue_sha256: String,
    review_queue: ReviewQueueProjection,
}

fn reconcile_mutation_files(
    state: &AppState,
    expected_job_id: &str,
    receipt: &MutationReceipt,
    queue_readback: &Map<String, Value>,
) -> IpcResult<ReconciledMutationEvidence> {
    validate_current_job(state, expected_job_id)?;
    let output_root = state.output_root.as_deref().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "Worker mutation reconciliation requires a committed output root.",
        )
    })?;
    let transcript_artifact =
        verified_artifact_for_type(&state.worker_evidence, "transcript-document-v2")?;
    let queue_artifact = verified_artifact_for_type(&state.worker_evidence, "review-queue-v2")?;
    let transcript_path = resolve_worker_artifact_path(
        output_root,
        &transcript_artifact.canonical_path.to_string_lossy(),
    )?;
    let review_queue_path = resolve_worker_artifact_path(
        output_root,
        &queue_artifact.canonical_path.to_string_lossy(),
    )?;

    let (transcript_document, transcript_sha256) = strict_json_file(
        &transcript_path,
        "the persisted transcript after human mutation",
    )?;
    if transcript_sha256 != receipt.document_hash {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted transcript hash does not match the completed mutation receipt.",
        ));
    }
    let transcript = project_transcript_document(
        &transcript_document,
        expected_job_id,
        &state.snapshot.job.speaker_policy,
    )?;
    if transcript.speaker_count != receipt.speaker_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The persisted transcript speaker count conflicts with the mutation receipt.",
        ));
    }
    if state
        .worker_evidence
        .pipeline_duration_ms
        .is_some_and(|duration| duration != transcript.duration_ms)
        || state
            .worker_evidence
            .pipeline_segment_count
            .is_some_and(|count| count != transcript.segments.len())
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The mutated transcript conflicts with verified pipeline metrics.",
        ));
    }

    let (review_queue_document, review_queue_sha256) = strict_json_file(
        &review_queue_path,
        "the persisted review queue after human mutation",
    )?;
    if review_queue_document != Value::Object(queue_readback.clone()) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review.queue response does not equal the durable review queue file.",
        ));
    }
    if canonical_json_sha256(&Value::Object(queue_readback.clone()))? != review_queue_sha256 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The review.queue response canonical hash does not match the durable review queue.",
        ));
    }
    let mut projection_evidence = state.worker_evidence.clone();
    projection_evidence.transcript_duration_ms = Some(transcript.duration_ms);
    projection_evidence.transcript_segment_count = Some(transcript.segments.len());
    projection_evidence.transcript_speaker_count_mode = Some(transcript.speaker_count_mode.clone());
    projection_evidence.transcript_speaker_count_estimate =
        Some(transcript.speaker_count_estimate.clone());
    projection_evidence.transcript_segments = transcript.segments.clone();
    let review_queue = project_review_queue_document(
        &review_queue_document,
        expected_job_id,
        &projection_evidence,
    )?;
    if review_queue.worker_open_count != receipt.open_count {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The durable review queue open count conflicts with the mutation receipt.",
        ));
    }

    Ok(ReconciledMutationEvidence {
        transcript_path,
        transcript_sha256,
        transcript,
        review_queue_path,
        review_queue_sha256,
        review_queue,
    })
}

fn update_snapshot_artifact_hash(
    snapshot: &mut StudioSnapshot,
    output_root: &Path,
    path: &Path,
    sha256: &str,
) -> IpcResult<()> {
    let relative_path = relative_worker_artifact_path(output_root, path)?;
    let matches = snapshot
        .artifacts
        .iter_mut()
        .filter(|artifact| artifact.relative_path == relative_path)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "Mutation reconciliation requires exactly one matching snapshot artifact.",
        ));
    }
    let artifact = matches
        .into_iter()
        .next()
        .expect("one artifact was verified");
    artifact.integrity = IntegrityStatus::Verified;
    artifact.sha256 = Some(sha256.to_owned());
    Ok(())
}

fn apply_reconciled_mutation_evidence(
    snapshot: &mut StudioSnapshot,
    evidence: &mut WorkerEvidenceLedger,
    output_root: &Path,
    reconciled: ReconciledMutationEvidence,
) -> IpcResult<()> {
    let transcript_entry = evidence
        .verified_artifacts
        .get(&reconciled.transcript_path)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The reconciled transcript is absent from the verified artifact ledger.",
            )
        })?;
    if transcript_entry.artifact_type != "transcript-document-v2" {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The reconciled transcript path is bound to the wrong artifact type.",
        ));
    }
    let queue_entry = evidence
        .verified_artifacts
        .get(&reconciled.review_queue_path)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "The reconciled review queue is absent from the verified artifact ledger.",
            )
        })?;
    if queue_entry.artifact_type != "review-queue-v2" {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The reconciled review queue path is bound to the wrong artifact type.",
        ));
    }
    artifact_content(
        "transcript-document-v2",
        &reconciled.transcript_path,
        &reconciled.transcript_sha256,
    )?;
    artifact_content(
        "review-queue-v2",
        &reconciled.review_queue_path,
        &reconciled.review_queue_sha256,
    )?;

    let ReconciledMutationEvidence {
        transcript_path,
        transcript_sha256,
        transcript,
        review_queue_path,
        review_queue_sha256,
        mut review_queue,
    } = reconciled;
    if evidence
        .pipeline_duration_ms
        .is_some_and(|duration| duration != transcript.duration_ms)
        || evidence
            .pipeline_segment_count
            .is_some_and(|count| count != transcript.segments.len())
    {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "Mutation reconciliation conflicts with committed pipeline evidence.",
        ));
    }
    preserve_review_audit_trails(&mut review_queue.reviews, &snapshot.reviews);
    let speaker_review_state = snapshot
        .speakers
        .iter()
        .map(|speaker| (speaker.id.clone(), (speaker.locked, speaker.review_status)))
        .collect::<BTreeMap<_, _>>();
    let mut reconciled_speakers = transcript.speakers;
    for speaker in &mut reconciled_speakers {
        if let Some((locked, review_status)) = speaker_review_state.get(&speaker.id) {
            speaker.locked = *locked;
            speaker.review_status = *review_status;
        }
    }
    snapshot.job.speaker_count = Some(transcript.speaker_count);
    snapshot.job.speaker_detection = transcript.speaker_detection;
    snapshot.job.duration_label = transcript.duration_label;
    snapshot.speakers = reconciled_speakers;
    snapshot.reviews = review_queue.reviews;
    snapshot.job.review_open_count = review_queue.open_segment_item_count;
    evidence.transcript_duration_ms = Some(transcript.duration_ms);
    evidence.transcript_segment_count = Some(transcript.segments.len());
    evidence.transcript_speaker_count_mode = Some(transcript.speaker_count_mode);
    evidence.transcript_speaker_count_estimate = Some(transcript.speaker_count_estimate);
    evidence.transcript_segments = transcript.segments;
    evidence.worker_review_open_count = Some(review_queue.worker_open_count);
    evidence.open_review_segment_item_count = review_queue.open_segment_item_count;
    evidence.open_review_segment_ids = review_queue.open_segment_ids;

    evidence
        .verified_artifacts
        .get_mut(&transcript_path)
        .expect("the transcript ledger entry was validated")
        .sha256
        .clone_from(&transcript_sha256);
    evidence
        .verified_artifacts
        .get_mut(&review_queue_path)
        .expect("the review queue ledger entry was validated")
        .sha256
        .clone_from(&review_queue_sha256);
    update_snapshot_artifact_hash(snapshot, output_root, &transcript_path, &transcript_sha256)?;
    update_snapshot_artifact_hash(
        snapshot,
        output_root,
        &review_queue_path,
        &review_queue_sha256,
    )?;

    if let Some(stage) = snapshot
        .stages
        .iter_mut()
        .find(|stage| stage.id == "review")
    {
        if review_queue.worker_open_count == 0 {
            stage.status = StageStatus::Completed;
            stage.progress = 100;
            stage.detail =
                "The durable review queue has no remaining open human decisions.".to_owned();
        } else {
            stage.status = StageStatus::Warning;
            stage.detail = format!(
                "{} durable review items remain open; {} are segment-scope items.",
                review_queue.worker_open_count, review_queue.open_segment_item_count
            );
        }
    }
    refresh_review_rate(snapshot, evidence);
    Ok(())
}

fn validate_job_completed_event(event: &WorkerEvent, state: &AppState) -> IpcResult<()> {
    let expected_fields = if event.payload.contains_key("operation") {
        &["status", "operation", "artifactPaths", "business"][..]
    } else {
        &["status", "artifactPaths", "business"][..]
    };
    exact_object_keys(&event.payload, expected_fields, "The job.completed payload")?;
    if event.payload.get("status").and_then(Value::as_str) != Some("completed") {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "job.completed payload.status must be completed.",
        ));
    }
    if let Some(operation) = event.payload.get("operation") {
        if !matches!(operation.as_str(), Some("resume" | "rerender")) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "job.completed operation must be resume or rerender when present.",
            ));
        }
    }
    if !event.payload.get("business").is_some_and(Value::is_object) {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "job.completed business evidence must be an object.",
        ));
    }
    if !matches!(state.snapshot.job.status, JobStatus::Running) {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "job.completed is only valid for the committed task while it is running.",
        ));
    }
    if state.worker_evidence.worker_review_open_count != Some(0)
        || state.worker_evidence.open_review_segment_item_count != 0
        || !state.worker_evidence.open_review_segment_ids.is_empty()
    {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "job.completed cannot be accepted while durable review evidence remains open or unknown.",
        ));
    }
    if state
        .snapshot
        .stages
        .iter()
        .any(|stage| matches!(stage.status, StageStatus::Warning | StageStatus::Blocked))
    {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "job.completed cannot overwrite a warning or blocked pipeline stage.",
        ));
    }
    let output_root = state.output_root.as_deref().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "job.completed requires a committed output root.",
        )
    })?;
    let artifact_paths = event
        .payload
        .get("artifactPaths")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "job.completed artifactPaths must be an array.",
            )
        })?;
    let mut announced_paths = BTreeSet::new();
    for path in artifact_paths {
        let path = path.as_str().ok_or_else(|| {
            IpcError::new(
                IpcErrorCode::StateUnavailable,
                "job.completed artifactPaths entries must be strings.",
            )
        })?;
        let canonical = resolve_worker_artifact_path(output_root, path)?;
        if !announced_paths.insert(canonical) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                "job.completed artifactPaths must not contain duplicates.",
            ));
        }
    }
    let verified_paths = state
        .worker_evidence
        .verified_artifacts
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    if announced_paths != verified_paths {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "job.completed artifactPaths must exactly match the verified artifact ledger.",
        ));
    }
    let mut artifact_types = BTreeSet::new();
    for artifact in state.worker_evidence.verified_artifacts.values() {
        artifact_content(
            &artifact.artifact_type,
            &artifact.canonical_path,
            &artifact.sha256,
        )?;
        artifact_types.insert(artifact.artifact_type.as_str());
    }
    for required in REQUIRED_COMPLETION_ARTIFACT_TYPES {
        if !artifact_types.contains(required) {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("job.completed is missing required artifact type {required}."),
            ));
        }
    }
    Ok(())
}

fn stage_id_for_worker_stage(stage: &str) -> Option<&'static str> {
    match stage {
        "queued" => Some("media"),
        "transcription" => Some("asr"),
        "validation" => Some("alignment"),
        "review_required" => Some("review"),
        "business_processing" => Some("document"),
        "rendering" => Some("pdf"),
        "completed" => Some("pdf"),
        _ => None,
    }
}

fn worker_event_presentation(event: &WorkerEvent) -> (Severity, String, String) {
    let detail_value = event
        .payload
        .get("message")
        .or_else(|| event.payload.get("status"))
        .or_else(|| event.payload.get("stage"))
        .map(|value| match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| "Verified local worker event".to_owned());
    match event.event_type.as_str() {
        "job.failed" => (
            Severity::Error,
            "Task failed closed".to_owned(),
            detail_value,
        ),
        "warning" => (Severity::Warning, "Worker warning".to_owned(), detail_value),
        "job.completed" => (Severity::Success, "Task completed".to_owned(), detail_value),
        "job.cancelled" => (Severity::Warning, "Task cancelled".to_owned(), detail_value),
        "review.required" => (
            Severity::Warning,
            "Human review required".to_owned(),
            detail_value,
        ),
        "review.decision.persisted" => (
            Severity::Success,
            "Human review decision persisted".to_owned(),
            detail_value,
        ),
        "artifact.created" => (
            Severity::Success,
            "Artifact verified".to_owned(),
            detail_value,
        ),
        "stage.started" | "stage.progress" => (
            Severity::Info,
            "Pipeline stage updated".to_owned(),
            detail_value,
        ),
        _ => (Severity::Info, "Task started".to_owned(), detail_value),
    }
}

fn project_worker_event_in_state(state: &mut AppState, event: &WorkerEvent) -> IpcResult<()> {
    if state.snapshot.job.id != event.job_id {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "The worker event belongs to a different registered task.",
        ));
    }
    let output_root = state.output_root.clone().ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The registered task has no controlled output root for worker evidence.",
        )
    })?;
    let policy = state.snapshot.job.speaker_policy.clone();
    let evidence = state.worker_evidence.clone();

    let prepared_artifact = (event.event_type == "artifact.created")
        .then(|| prepare_artifact_projection(event, &output_root, &policy, &evidence))
        .transpose()?;
    let review_required_count = (event.event_type == "review.required")
        .then(|| validate_review_required_event(event, &output_root, &evidence))
        .transpose()?;

    match event.event_type.as_str() {
        "job.started" => {
            state.snapshot.job.status = JobStatus::Running;
            state.snapshot.job.progress = state.snapshot.job.progress.max(1);
            state.snapshot.system.inference_worker = WorkerStatus::Busy;
        }
        "stage.started" => {
            let stage = event_string(&event.payload, "stage", 160)?;
            if let Some(stage_id) = stage_id_for_worker_stage(stage) {
                if let Some(item) = state
                    .snapshot
                    .stages
                    .iter_mut()
                    .find(|item| item.id == stage_id)
                {
                    item.status = StageStatus::Running;
                    item.progress = item.progress.max(1);
                    item.detail = format!("Verified worker stage: {stage}");
                }
            }
            state.snapshot.job.status = JobStatus::Running;
        }
        "stage.progress" => {
            let progress = event_nonnegative_integer(&event.payload, "progress")?;
            if progress > 100 {
                return Err(IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "Worker stage progress must be between 0 and 100.",
                ));
            }
            state.snapshot.job.progress = progress as u8;
            if let Some(stage) = event.payload.get("stage").and_then(Value::as_str) {
                if let Some(stage_id) = stage_id_for_worker_stage(stage) {
                    if let Some(item) = state
                        .snapshot
                        .stages
                        .iter_mut()
                        .find(|item| item.id == stage_id)
                    {
                        item.status = StageStatus::Running;
                        item.progress = progress as u8;
                    }
                }
            }
        }
        "artifact.created" => {
            let artifact = prepared_artifact.ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "The artifact event was not verified before projection.",
                )
            })?;
            commit_artifact_projection(state, artifact)?;
        }
        "review.required" => {
            let worker_open_count = review_required_count.ok_or_else(|| {
                IpcError::new(
                    IpcErrorCode::StateUnavailable,
                    "The review.required event was not verified before projection.",
                )
            })?;
            if state.worker_evidence.worker_review_open_count != Some(worker_open_count) {
                return Err(IpcError::new(
                    IpcErrorCode::Conflict,
                    "The durable review queue changed before review.required projection.",
                ));
            }
            let segment_open_count = state.worker_evidence.open_review_segment_item_count;
            state.snapshot.job.review_open_count = segment_open_count;
            state.snapshot.job.status = JobStatus::ReviewRequired;
            if let Some(stage) = state
                .snapshot
                .stages
                .iter_mut()
                .find(|stage| stage.id == "review")
            {
                stage.status = StageStatus::Warning;
                stage.detail = format!(
                    "{worker_open_count} durable review items require a human decision; {segment_open_count} are segment-scope items projected into the transcript review list."
                );
            }
        }
        "review.decision.persisted" => {
            validate_persisted_human_decision_event(event, state)?;
        }
        "job.completed" => {
            validate_job_completed_event(event, state)?;
            state.snapshot.job.status = JobStatus::Completed;
            state.snapshot.job.progress = 100;
            state.snapshot.system.inference_worker = WorkerStatus::Ready;
            for stage in &mut state.snapshot.stages {
                stage.status = StageStatus::Completed;
                stage.progress = 100;
            }
        }
        "job.failed" => {
            state.snapshot.job.status = JobStatus::Failed;
            state.snapshot.system.inference_worker = WorkerStatus::Missing;
        }
        "job.cancelled" => {
            state.snapshot.job.status = JobStatus::Cancelled;
            state.snapshot.system.inference_worker = WorkerStatus::Ready;
        }
        "warning" => {}
        other => {
            return Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!("Unsupported worker event reached projection: {other}."),
            ));
        }
    }
    let stage_id = event
        .payload
        .get("stage")
        .and_then(Value::as_str)
        .and_then(stage_id_for_worker_stage)
        .map(str::to_owned);
    let (severity, title, detail) = worker_event_presentation(event);
    state.snapshot.events.push(StudioEvent {
        id: event.event_id.clone(),
        sequence: event.sequence,
        event_type: event.event_type.clone(),
        stage_id,
        severity,
        timestamp: event.timestamp.clone(),
        title,
        detail,
    });
    if state.snapshot.events.len() > 512 {
        let excess = state.snapshot.events.len() - 512;
        state.snapshot.events.drain(..excess);
    }
    Ok(())
}

fn project_worker_event(store: &StudioStore, event: &WorkerEvent) -> IpcResult<JobSnapshotUpdate> {
    let job_id = parse_job_id(&event.job_id)?;
    let (registered, token) = store
        .registry
        .snapshot_with_token(&job_id)
        .map_err(registry_ipc_error)?;
    if registered.status == RegistryJobStatus::Registered {
        return Err(IpcError::new(
            IpcErrorCode::Conflict,
            "Worker events cannot mutate a task before job.start acceptance is committed.",
        ));
    }

    let mut next_state = registered.runtime_state;
    project_worker_event_in_state(&mut next_state, event)?;
    let next_status = registry_status_for_job(next_state.snapshot.job.status);
    store
        .registry
        .transition_status_with(&token, next_status, |state| *state = next_state)
        .map_err(registry_ipc_error)?;
    if matches!(
        next_status,
        RegistryJobStatus::Completed | RegistryJobStatus::Failed | RegistryJobStatus::Cancelled
    ) {
        store.release_dispatch_permit(&job_id)?;
    }
    job_snapshot_update(store, &job_id)
}

fn worker_event_target_ready(store: &StudioStore, job_id: &str) -> IpcResult<bool> {
    let job_id = parse_job_id(job_id)?;
    match store.registry.snapshot(&job_id) {
        Ok(snapshot) => Ok(snapshot.status != RegistryJobStatus::Registered
            && snapshot.runtime_state.output_root.is_some()),
        Err(RegistryError::UnknownJob { .. }) => Ok(false),
        Err(error) => Err(registry_ipc_error(error)),
    }
}

fn project_worker_failure(
    store: &StudioStore,
    job_ids: &[String],
    message: &str,
) -> IpcResult<Vec<JobSnapshotUpdate>> {
    let mut updates = Vec::new();
    for value in job_ids {
        let job_id = parse_job_id(value)?;
        let status = match store.registry.status(&job_id) {
            Ok(status) => status,
            Err(RegistryError::UnknownJob { .. }) => continue,
            Err(error) => return Err(registry_ipc_error(error)),
        };
        if matches!(
            status,
            RegistryJobStatus::Completed | RegistryJobStatus::Failed | RegistryJobStatus::Cancelled
        ) {
            continue;
        }
        transition_job_state(store, &job_id, RegistryJobStatus::Failed, |state| {
            commit_failed_job_in_state(state, value, message)
        })?;
        store.release_dispatch_permit(&job_id)?;
        updates.push(job_snapshot_update(store, &job_id)?);
    }
    Ok(updates)
}

#[tauri::command]
fn get_snapshot(state: State<'_, StudioStore>) -> IpcResult<StudioSnapshot> {
    Ok(state
        .projected_state()?
        .map(|state| state.snapshot)
        .unwrap_or_else(|| state.draft_snapshot.clone()))
}

#[tauri::command]
fn get_job_snapshot(state: State<'_, StudioStore>, job_id: String) -> IpcResult<StudioSnapshot> {
    let job_id = parse_job_id(&job_id)?;
    Ok(job_state(&state, &job_id)?.snapshot)
}

async fn runtime_status(
    state: &StudioStore,
    worker: &WorkerSupervisor,
    job_id: &JobId,
) -> IpcResult<JobRuntimeStatus> {
    let snapshot = state
        .registry
        .snapshot(job_id)
        .map_err(registry_ipc_error)?;
    let projected = state.projected_job_id()?.as_ref() == Some(job_id);
    let dispatcher = state
        .registry
        .dispatcher_snapshot()
        .map_err(registry_ipc_error)?;
    let worker_event_route_registered = worker.routes_job_events(job_id.as_str()).await;
    let cancellable = matches!(
        snapshot.status,
        RegistryJobStatus::Queued | RegistryJobStatus::Running | RegistryJobStatus::ReviewRequired
    ) && worker_event_route_registered;
    Ok(JobRuntimeStatus {
        job_id: job_id.as_str().to_owned(),
        status: snapshot.runtime_state.snapshot.job.status,
        revision: snapshot.revision,
        accepted_by_worker: snapshot.runtime_state.worker_accepted,
        projected,
        in_flight: dispatcher
            .active_job_ids
            .iter()
            .any(|active| active == job_id),
        worker_event_route_registered,
        cancellable,
        volatile_only: true,
    })
}

#[tauri::command]
async fn get_job_status(
    state: State<'_, StudioStore>,
    worker: State<'_, WorkerSupervisor>,
    job_id: String,
) -> IpcResult<JobRuntimeStatus> {
    let job_id = parse_job_id(&job_id)?;
    runtime_status(&state, &worker, &job_id).await
}

#[tauri::command]
async fn list_jobs(
    state: State<'_, StudioStore>,
    worker: State<'_, WorkerSupervisor>,
) -> IpcResult<Vec<JobRuntimeStatus>> {
    let job_ids = state.registry.job_ids().map_err(registry_ipc_error)?;
    let mut statuses = Vec::with_capacity(job_ids.len());
    for job_id in job_ids {
        statuses.push(runtime_status(&state, &worker, &job_id).await?);
    }
    Ok(statuses)
}

#[tauri::command]
fn select_job(state: State<'_, StudioStore>, job_id: String) -> IpcResult<StudioSnapshot> {
    let job_id = parse_job_id(&job_id)?;
    state.select_job(&job_id)?;
    Ok(job_state(&state, &job_id)?.snapshot)
}

#[tauri::command]
async fn create_job(
    state: State<'_, StudioStore>,
    worker: State<'_, WorkerSupervisor>,
    gate: State<'_, JobCommandGate>,
    request: CreateJobRequest,
) -> IpcResult<CreateJobResult> {
    let prepared = prepare_job(request)?;
    let candidate_job_id = parse_job_id(&next_job_id())?;
    let idempotency_key = IdempotencyKey::new(
        prepared
            .idempotency_key
            .clone()
            .unwrap_or_else(|| format!("auto:{}", candidate_job_id.as_str())),
    )
    .map_err(registry_ipc_error)?;
    let request_fingerprint = prepared_job_fingerprint(&prepared)?;
    let worker_output_directory =
        derive_worker_output_directory(&prepared.output_directory, candidate_job_id.as_str())?;
    let runtime_state = build_registered_job_state(
        &prepared,
        candidate_job_id.as_str(),
        worker_output_directory.clone(),
        request_fingerprint.clone(),
    )?;
    let payload = build_job_start_payload(
        &prepared,
        candidate_job_id.as_str(),
        &worker_output_directory,
    )?;
    let candidate_gate = gate.for_job(&candidate_job_id)?;
    let _candidate_guard = candidate_gate.lock().await;
    let registration = match state.registry.register_or_get(
        candidate_job_id.clone(),
        idempotency_key,
        runtime_state,
    ) {
        Ok(registration) => registration,
        Err(error) => {
            let primary = registry_ipc_error(error);
            let cleanup =
                state.discard_gate_if_unpublished(&gate, &candidate_job_id, &candidate_gate);
            return Err(primary_error_with_cleanup(primary, cleanup));
        }
    };
    if let RegisterOutcome::Existing { job_id, .. } = registration {
        state.discard_gate_if_unpublished(&gate, &candidate_job_id, &candidate_gate)?;
        let existing = state
            .registry
            .snapshot(&job_id)
            .map_err(registry_ipc_error)?;
        if existing.runtime_state.request_fingerprint != request_fingerprint {
            return Err(IpcError::new(
                IpcErrorCode::Conflict,
                "The idempotency key is already bound to a different normalized task request.",
            ));
        }
        state.select_job(&job_id)?;
        let status = existing.runtime_state.snapshot.job.status;
        return Ok(CreateJobResult {
            accepted: existing.runtime_state.worker_accepted,
            job_id: job_id.as_str().to_owned(),
            replayed: true,
            status,
            message: "The idempotency key already exists in this volatile desktop session; job.start was not replayed.".to_owned(),
        });
    }
    let job_id = registration.job_id().clone();
    debug_assert_eq!(job_id, candidate_job_id);
    if let Err(error) = state.select_job(&job_id) {
        let cleanup = fail_job_and_release(&state, &job_id, &error.message);
        return Err(primary_error_with_cleanup(error, cleanup));
    }
    let permit = match state.registry.try_acquire_dispatch(&job_id) {
        Ok(permit) => permit,
        Err(error) => {
            let message = format!(
                "The task was registered but not dispatched because the volatile desktop in-flight limit was reached: {error}"
            );
            let primary = registry_ipc_error(error);
            let cleanup = fail_job_and_release(&state, &job_id, &message);
            return Err(primary_error_with_cleanup(primary, cleanup));
        }
    };
    if let Err(error) = state.hold_dispatch_permit(job_id.clone(), permit) {
        let cleanup = fail_job_and_release(&state, &job_id, &error.message);
        return Err(primary_error_with_cleanup(error, cleanup));
    }

    let response = match worker
        .request("job.start", payload, ResponseKind::Accepted)
        .await
    {
        Ok(response) => response,
        Err(error) => {
            let ipc_error = worker_ipc_error("Unable to start the task", error);
            let cleanup = fail_job_and_release(&state, &job_id, &ipc_error.message);
            return Err(primary_error_with_cleanup(ipc_error, cleanup));
        }
    };
    if let Err(error) = validate_start_accepted(&response, job_id.as_str()) {
        worker
            .fail_closed(format!(
                "The accepted job.start response failed desktop semantic validation: {}",
                error.message
            ))
            .await;
        let cleanup = fail_job_and_release(&state, &job_id, &error.message);
        return Err(primary_error_with_cleanup(error, cleanup));
    }

    let result = match transition_job_state(&state, &job_id, RegistryJobStatus::Queued, |runtime| {
        commit_accepted_job_in_state(runtime, job_id.as_str())
    }) {
        Ok(result) => result,
        Err(error) => {
            let primary = IpcError::new(
                error.code,
                format!(
                    "The worker explicitly accepted the task, but the desktop could not commit local state. To prevent duplicate work, it will never automatically replay job.start: {}",
                    error.message
                ),
            );
            worker
                .fail_closed(format!(
                    "job.start was accepted for {}, but desktop state acceptance could not be committed: {}",
                    job_id.as_str(),
                    primary.message
                ))
                .await;
            let cleanup = fail_job_and_release(&state, &job_id, &primary.message);
            return Err(primary_error_with_cleanup(primary, cleanup));
        }
    };
    Ok(result)
}

#[tauri::command]
async fn cancel_job(
    state: State<'_, StudioStore>,
    worker: State<'_, WorkerSupervisor>,
    gate: State<'_, JobCommandGate>,
    job_id: String,
) -> IpcResult<()> {
    let job_id = parse_job_id(&job_id)?;
    let command_gate = gate.for_job(&job_id)?;
    let _command_guard = command_gate.lock().await;
    let registered = state
        .registry
        .snapshot(&job_id)
        .map_err(registry_ipc_error)?;
    match registered.status {
        RegistryJobStatus::Cancelled => return Ok(()),
        RegistryJobStatus::Completed | RegistryJobStatus::Failed => {
            return Err(IpcError::new(
                IpcErrorCode::Conflict,
                "A completed or failed task cannot be cancelled.",
            ));
        }
        RegistryJobStatus::Registered => {
            return Err(IpcError::new(
                IpcErrorCode::Conflict,
                "The task is registered but job.start acceptance is not committed yet.",
            ));
        }
        RegistryJobStatus::Queued
        | RegistryJobStatus::Running
        | RegistryJobStatus::ReviewRequired => {}
    }
    if !worker.routes_job_events(job_id.as_str()).await {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            "The current worker generation has no verified event route for this task; cancellation cannot be claimed.",
        ));
    }

    let mut payload = Map::new();
    payload.insert(
        "jobId".to_owned(),
        Value::String(job_id.as_str().to_owned()),
    );
    let response = worker
        .request("job.cancel", payload, ResponseKind::Accepted)
        .await
        .map_err(|error| worker_ipc_error("Unable to cancel the task", error))?;
    let status = match validate_cancel_accepted(&response, job_id.as_str()) {
        Ok(status) => status,
        Err(error) => {
            worker
                .fail_closed(format!(
                    "The accepted job.cancel response failed desktop semantic validation: {}",
                    error.message
                ))
                .await;
            return Err(error);
        }
    };
    if status != "cancelled" {
        return Err(IpcError::new(
            IpcErrorCode::StateUnavailable,
            format!(
                "The worker accepted the cancellation request, but the current task status is still {status}; the desktop will not fabricate a cancelled state."
            ),
        ));
    }

    if let Err(error) =
        transition_job_state(&state, &job_id, RegistryJobStatus::Cancelled, |runtime| {
            commit_cancelled_job_in_state(runtime, job_id.as_str())
        })
    {
        worker
            .fail_closed(format!(
                "job.cancel was accepted for {}, but desktop cancellation state could not be committed: {}",
                job_id.as_str(),
                error.message
            ))
            .await;
        return Err(error);
    }
    if let Err(error) = state.release_dispatch_permit(&job_id) {
        worker
            .fail_closed(format!(
                "job.cancel was committed for {}, but the dispatch permit could not be released: {}",
                job_id.as_str(),
                error.message
            ))
            .await;
        return Err(error);
    }
    Ok(())
}

#[tauri::command]
async fn update_speaker(
    state: State<'_, StudioStore>,
    worker: State<'_, WorkerSupervisor>,
    gate: State<'_, JobCommandGate>,
    request: UpdateSpeakerRequest,
) -> IpcResult<SpeakerProfile> {
    let job_id = state.projected_job_id()?.ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "No task is selected for the speaker mutation.",
        )
    })?;
    let command_gate = gate.for_job(&job_id)?;
    let _command_guard = command_gate.lock().await;
    let prepared = prepare_speaker_rename(&job_state(&state, &job_id)?, request)?;
    let response = worker
        .request(
            "speaker.rename",
            prepared.payload.clone(),
            ResponseKind::Completed,
        )
        .await
        .map_err(|error| worker_ipc_error("Unable to persist the speaker rename", error))?;
    let receipt = match validate_mutation_completed(
        &response,
        &prepared.job_id,
        "speaker.rename",
        &prepared.envelope,
        prepared.speaker_count,
    ) {
        Ok(receipt) => receipt,
        Err(error) => {
            worker
                .fail_closed(format!(
                    "The completed speaker.rename response failed desktop semantic validation: {}",
                    error.message
                ))
                .await;
            return Err(error);
        }
    };
    let queue_response = match worker
        .request(
            "review.queue",
            Map::from_iter([("jobId".to_owned(), Value::String(prepared.job_id.clone()))]),
            ResponseKind::Completed,
        )
        .await
    {
        Ok(response) => response,
        Err(error) => {
            worker
                .fail_closed(format!(
                    "speaker.rename completed, but review.queue could not prove durable persistence: {}",
                    error.message
                ))
                .await;
            return Err(worker_ipc_error(
                "The worker may have persisted the speaker rename, but durable readback failed; do not retry automatically",
                error,
            ));
        }
    };
    let queue = match validate_review_queue_snapshot(&queue_response, &prepared.job_id, &receipt)
        .and_then(|queue| {
            validate_speaker_rename_proof(&queue, &prepared, &receipt)?;
            Ok(queue)
        }) {
        Ok(queue) => queue,
        Err(error) => {
            worker
                .fail_closed(format!(
                    "speaker.rename completed, but its review.queue persistence proof was invalid: {}",
                    error.message
                ))
                .await;
            return Err(error);
        }
    };
    let reconciliation = reconcile_mutation_files(
        &job_state(&state, &job_id)?,
        &prepared.job_id,
        &receipt,
        &queue,
    );
    let reconciled = match reconciliation {
        Ok(reconciled) => reconciled,
        Err(error) => {
            worker
                .fail_closed(format!(
                    "speaker.rename completed, but durable transcript/review files failed reconciliation: {}",
                    error.message
                ))
                .await;
            return Err(error);
        }
    };

    let committed = mutate_job_state(&state, &job_id, |runtime| {
        commit_speaker_rename_in_state(runtime, &prepared, &receipt, reconciled)
    });
    match committed {
        Ok(speaker) => Ok(speaker),
        Err(error) => {
            worker
                .fail_closed(format!(
                    "speaker.rename was durably verified, but desktop reconciliation failed: {}",
                    error.message
                ))
                .await;
            Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!(
                    "The worker durably persisted the speaker rename, but desktop reconciliation failed; do not retry automatically: {}",
                    error.message
                ),
            ))
        }
    }
}

#[tauri::command]
async fn apply_review_decision(
    state: State<'_, StudioStore>,
    worker: State<'_, WorkerSupervisor>,
    gate: State<'_, JobCommandGate>,
    decision: ReviewDecision,
) -> IpcResult<ReviewSegment> {
    let job_id = state.projected_job_id()?.ok_or_else(|| {
        IpcError::new(
            IpcErrorCode::StateUnavailable,
            "No task is selected for the review mutation.",
        )
    })?;
    let command_gate = gate.for_job(&job_id)?;
    let _command_guard = command_gate.lock().await;
    let prepared = prepare_review_mutation(&job_state(&state, &job_id)?, decision)?;
    let response = worker
        .request(
            "review.submit",
            prepared.payload.clone(),
            ResponseKind::Completed,
        )
        .await
        .map_err(|error| worker_ipc_error("Unable to persist the review decision", error))?;
    let receipt = match validate_mutation_completed(
        &response,
        &prepared.job_id,
        "review.submit",
        &prepared.envelope,
        prepared.speaker_count,
    ) {
        Ok(receipt) => receipt,
        Err(error) => {
            worker
                .fail_closed(format!(
                    "The completed review.submit response failed desktop semantic validation: {}",
                    error.message
                ))
                .await;
            return Err(error);
        }
    };
    let queue_response = match worker
        .request(
            "review.queue",
            Map::from_iter([("jobId".to_owned(), Value::String(prepared.job_id.clone()))]),
            ResponseKind::Completed,
        )
        .await
    {
        Ok(response) => response,
        Err(error) => {
            worker
                .fail_closed(format!(
                    "review.submit completed, but review.queue could not prove durable persistence: {}",
                    error.message
                ))
                .await;
            return Err(worker_ipc_error(
                "The worker may have persisted the review decision, but durable readback failed; do not retry automatically",
                error,
            ));
        }
    };
    let queue = match validate_review_queue_snapshot(&queue_response, &prepared.job_id, &receipt)
        .and_then(|queue| {
            validate_review_submission_proof(&queue, &prepared, &receipt)?;
            Ok(queue)
        }) {
        Ok(queue) => queue,
        Err(error) => {
            worker
                .fail_closed(format!(
                    "review.submit completed, but its review.queue persistence proof was invalid: {}",
                    error.message
                ))
                .await;
            return Err(error);
        }
    };
    let reconciliation = reconcile_mutation_files(
        &job_state(&state, &job_id)?,
        &prepared.job_id,
        &receipt,
        &queue,
    );
    let reconciled = match reconciliation {
        Ok(reconciled) => reconciled,
        Err(error) => {
            worker
                .fail_closed(format!(
                    "review.submit completed, but durable transcript/review files failed reconciliation: {}",
                    error.message
                ))
                .await;
            return Err(error);
        }
    };

    let committed = mutate_job_state(&state, &job_id, |runtime| {
        commit_review_mutation_in_state(runtime, &prepared, &receipt, reconciled)
    });
    match committed {
        Ok(review) => Ok(review),
        Err(error) => {
            worker
                .fail_closed(format!(
                    "review.submit was durably verified, but desktop reconciliation failed: {}",
                    error.message
                ))
                .await;
            Err(IpcError::new(
                IpcErrorCode::StateUnavailable,
                format!(
                    "The worker durably persisted the review decision, but desktop reconciliation failed; do not retry automatically: {}",
                    error.message
                ),
            ))
        }
    }
}

#[tauri::command]
fn open_artifact(
    state: State<'_, StudioStore>,
    artifact_id: String,
) -> IpcResult<ArtifactOpenResult> {
    open_artifact_in_state(&projected_state(&state)?, &artifact_id)
}

fn default_strategies() -> Vec<ModelStrategy> {
    vec![
        ModelStrategy {
            id: ModelStrategyId::Balanced,
            label: "Balanced".to_owned(),
            description: "CAM++ dynamic clustering, Qwen3-ASR, and FunASR boundaries work together; difficult segments enter local human review.".to_owned(),
            asr_model: "Qwen3-ASR-1.7B".to_owned(),
            diarization_model: "CAM++ · dynamic clustering".to_owned(),
            semantic_model: PRODUCTION_LOCAL_LLM_MODEL.to_owned(),
            semantic_model_status: "suggestion_only".to_owned(),
            semantic_model_evaluation: "Production evaluation: qwen3.5:27b-q4_K_M won the current multilingual semantic challenge; every decision remains evidence-bound and fail-closed.".to_owned(),
            estimated_vram_gb: 7.2,
            semantic_guardrail: "The model may only choose an acoustically backed candidate or abstain; it cannot create speakers, move boundaries, or overwrite raw text.".to_owned(),
            recommended: Some(true),
        },
        ModelStrategy {
            id: ModelStrategyId::Quality,
            label: "Quality first".to_owned(),
            description: "Difficult segments receive local reruns, a second overlap pass, and human listening while preserving the full acoustic evidence chain.".to_owned(),
            asr_model: "Qwen3-ASR-1.7B · dual-window review".to_owned(),
            diarization_model: "CAM++ · second overlap pass".to_owned(),
            semantic_model: PRODUCTION_LOCAL_LLM_MODEL.to_owned(),
            semantic_model_status: "suggestion_only".to_owned(),
            semantic_model_evaluation: "Production evaluation: qwen3.5:27b-q4_K_M won the current multilingual semantic challenge; every decision remains evidence-bound and fail-closed.".to_owned(),
            estimated_vram_gb: 8.0,
            semantic_guardrail: "The model may only choose an acoustically backed candidate or abstain; it cannot create speakers, move boundaries, or overwrite raw text.".to_owned(),
            recommended: None,
        },
        ModelStrategy {
            id: ModelStrategyId::MemorySaver,
            label: "Memory saver".to_owned(),
            description: "Reduces VRAM use while retaining acoustic anomaly escalation, local audio review, and human confirmation.".to_owned(),
            asr_model: "SenseVoiceSmall".to_owned(),
            diarization_model: "CAM++ · CPU clustering".to_owned(),
            semantic_model: PRODUCTION_LOCAL_LLM_MODEL.to_owned(),
            semantic_model_status: "suggestion_only".to_owned(),
            semantic_model_evaluation: "Production evaluation: qwen3.5:27b-q4_K_M won the current multilingual semantic challenge; every decision remains evidence-bound and fail-closed.".to_owned(),
            estimated_vram_gb: 4.4,
            semantic_guardrail: "The model may only choose an acoustically backed candidate or abstain; it cannot create speakers, move boundaries, or overwrite raw text.".to_owned(),
            recommended: None,
        },
    ]
}

fn default_stages() -> Vec<PipelineStage> {
    [
        ("media", "Media preprocessing", "Preprocess"),
        ("vad", "Speech boundaries", "Boundaries"),
        ("speaker", "Dynamic speaker clustering", "Speakers"),
        ("asr", "Multilingual recognition", "Recognition"),
        ("alignment", "Time alignment", "Alignment"),
        ("review", "Acoustic evidence review", "Review"),
        ("document", "Report assembly", "Report"),
        ("pdf", "PDF rendering and QA", "PDF"),
    ]
    .into_iter()
    .map(|(id, label, short_label)| PipelineStage {
        id: id.to_owned(),
        label: label.to_owned(),
        short_label: short_label.to_owned(),
        status: StageStatus::Pending,
        progress: 0,
        detail: "Awaiting the local processing worker".to_owned(),
        duration_label: None,
    })
    .collect()
}

fn default_pdf_quality() -> PdfQualityReport {
    let hard_gates = [
        (
            "PDF-OPENABLE",
            "PDF opens successfully",
            "Awaiting PDFBox validation after this job generates a PDF.",
        ),
        (
            "PDF-PAGE-COUNT",
            "Page count matches",
            "Awaiting comparison of rendered pages, PDFBox pages, and the evidence manifest.",
        ),
        (
            "PDF-PAGE-SIZE",
            "Page sizes are compliant",
            "Awaiting A4 size validation for every page in this job.",
        ),
        (
            "PDF-TRANSCRIPT-TEXT-INTEGRITY",
            "Transcript text is intact",
            "Awaiting comparison of PDFBox-extracted text with the immutable transcript manifest.",
        ),
        (
            "PDF-SEGMENT-COUNT",
            "Segment count matches",
            "Awaiting comparison of transcript and PDF speech-segment counts.",
        ),
        (
            "PDF-TIMESTAMP-INTEGRITY",
            "Timestamps are intact",
            "Awaiting validation that timestamps are extractable, monotonic, and aligned with boundaries.",
        ),
        (
            "PDF-SPEAKER-SET-INTEGRITY",
            "Speaker set is complete",
            "Awaiting validation of the dynamic speaker legend, labels, and non-color markers.",
        ),
        (
            "PDF-FONT-EMBEDDED",
            "Required fonts are embedded",
            "Awaiting checks for multilingual font embedding and deterministic fallback.",
        ),
        (
            "PDF-NO-BLANK-PAGES",
            "No blank pages",
            "Awaiting page-by-page confirmation of verifiable content.",
        ),
        (
            "PDF-NO-CONTENT-OVERFLOW",
            "No content overflow",
            "Awaiting page evidence checks for safe areas and content overflow.",
        ),
        (
            "PDF-OFFLINE-ASSETS",
            "All assets are offline",
            "Awaiting a scan for remote URLs, scripts, fonts, images, and telemetry dependencies.",
        ),
        (
            "PDF-PAGE-EVIDENCE",
            "Page evidence is complete",
            "Awaiting generation and linking of per-page PNGs and the contact sheet.",
        ),
        (
            "PDF-IMMUTABLE-CONTENT-HASH",
            "Immutable content hashes match",
            "Awaiting SHA-256 manifests for the transcript, PDF, and page evidence.",
        ),
    ]
    .into_iter()
    .map(|(id, label, detail)| PdfHardGate {
        id: id.to_owned(),
        label: label.to_owned(),
        status: GateStatus::Pending,
        detail: detail.to_owned(),
    })
    .collect();
    let facets = [
        ("AESTHETIC-COHERENCE", "Coherence"),
        ("AESTHETIC-DISTINCTION", "Distinction"),
        ("AESTHETIC-REFINEMENT", "Refinement"),
        ("AESTHETIC-PROPORTION", "Proportion"),
        ("AESTHETIC-HIERARCHY", "Hierarchy"),
        ("AESTHETIC-TYPOGRAPHY", "Typography"),
        ("AESTHETIC-COLOR-RELATIONSHIPS", "Color relationships"),
        ("AESTHETIC-RHYTHM", "Rhythm"),
        ("AESTHETIC-DENSITY", "Information density"),
        ("AESTHETIC-RESTRAINT", "Restraint"),
        ("AESTHETIC-REAL-CONTENT-STRESS", "Real-content stress"),
        ("AESTHETIC-FONT-FAILURE", "Font failure"),
        ("AESTHETIC-IMAGE-FAILURE", "Image failure"),
        ("AESTHETIC-SCRIPT-FAILURE", "Script failure"),
    ]
    .into_iter()
    .map(|(id, label)| AestheticFacet {
        id: id.to_owned(),
        label: label.to_owned(),
        score: 0.0,
        status: FacetStatus::Pending,
        evidence:
            "Awaiting per-page PNGs, a contact sheet, and failure-scenario evidence for this job."
                .to_owned(),
    })
    .collect();

    PdfQualityReport {
        status: PdfStatus::Pending,
        pass_number: 1,
        score: 0.0,
        minimum_score: 85,
        page_count: 0,
        rendered_at: "Not rendered".to_owned(),
        hard_gates,
        facets,
        repair_queue: Vec::new(),
        evidence_digest: "Not generated".to_owned(),
    }
}

fn unavailable_reference_metric(metric: &str) -> PercentMetric {
    PercentMetric::Unavailable {
        reason: format!(
            "Reference labels are missing, so {metric} cannot be calculated or inferred."
        ),
    }
}

fn default_diarization_quality() -> DiarizationQualityMetrics {
    DiarizationQualityMetrics {
        der: unavailable_reference_metric("DER"),
        jer: unavailable_reference_metric("JER"),
        confusion: unavailable_reference_metric("confusion"),
        overlap_f1: unavailable_reference_metric("overlap F1"),
        review_rate: PercentMetric::Available {
            value: 0.0,
            unit: PercentUnit::Percent,
            source: "Review-pending segments / generated valid speech segments".to_owned(),
        },
    }
}

fn default_snapshot() -> StudioSnapshot {
    StudioSnapshot {
        contract_version: CONTRACT_VERSION.to_owned(),
        job: JobSummary {
            id: "job-draft".to_owned(),
            title: "Awaiting transcription task".to_owned(),
            source_path: String::new(),
            duration_label: "Not processed".to_owned(),
            status: JobStatus::Draft,
            progress: 0,
            started_at: "Not started".to_owned(),
            speaker_policy: SpeakerCountPolicy::Auto {},
            speaker_count: None,
            speaker_detection: None,
            review_open_count: 0,
            active_strategy_id: ModelStrategyId::Balanced,
        },
        speakers: Vec::new(),
        strategies: default_strategies(),
        stages: default_stages(),
        events: Vec::new(),
        reviews: Vec::new(),
        artifacts: Vec::new(),
        diarization_quality: default_diarization_quality(),
        performance: PerformanceMetrics::Unavailable {
            reason:
                "The task has not run, so model-stage and resource-sampling data are unavailable."
                    .to_owned(),
        },
        pdf_quality: default_pdf_quality(),
        system: SystemStatus {
            offline: true,
            backend_mode: BackendMode::TauriIpc,
            gpu_label: "Detected by the local model runtime".to_owned(),
            vram_label: "Not detected".to_owned(),
            inference_worker: WorkerStatus::Missing,
            java_renderer: WorkerStatus::Missing,
        },
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(StudioStore::new())
        .manage(JobCommandGate::default())
        .setup(|app| {
            let runtime_resource_directory = app.path().resource_dir().ok();
            app.manage(WorkerSupervisor::new_with_resource_directory(
                runtime_resource_directory,
            ));
            let supervisor = app.state::<WorkerSupervisor>().inner().clone();
            let projection_supervisor = supervisor.clone();
            let mut worker_events = supervisor.subscribe_events();
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    match worker_events.recv().await {
                        Ok(WorkerNotification::Event(event)) => {
                            let job_id = match parse_job_id(&event.job_id) {
                                Ok(job_id) => job_id,
                                Err(error) => {
                                    projection_supervisor
                                        .fail_closed(format!(
                                            "Worker event contained an invalid desktop job identity: {}",
                                            error.message
                                        ))
                                        .await;
                                    continue;
                                }
                            };
                            let command_gate = {
                                let gates = app_handle.state::<JobCommandGate>();
                                match gates.inner().for_job(&job_id) {
                                    Ok(command_gate) => command_gate,
                                    Err(error) => {
                                        projection_supervisor
                                            .fail_closed(format!(
                                                "Worker event command-gate lookup failed: {}",
                                                error.message
                                            ))
                                            .await;
                                        continue;
                                    }
                                }
                            };
                            let _command_guard = command_gate.lock().await;
                            let target_ready = {
                                let store = app_handle.state::<StudioStore>();
                                worker_event_target_ready(store.inner(), &event.job_id)
                            };
                            match target_ready {
                                Ok(true) => {}
                                Ok(false) => {
                                    projection_supervisor
                                        .fail_closed(format!(
                                            "Worker event {} could not be associated with a committed registered desktop task.",
                                            event.event_id
                                        ))
                                        .await;
                                    continue;
                                }
                                Err(error) => {
                                    projection_supervisor
                                        .fail_closed(format!(
                                            "Worker event target-state validation failed: {}",
                                            error.message
                                        ))
                                        .await;
                                    continue;
                                }
                            }
                            if !projection_supervisor
                                .routes_job_events(job_id.as_str())
                                .await
                                && !matches!(
                                    event.event_type.as_str(),
                                    "job.completed" | "job.failed" | "job.cancelled"
                                )
                            {
                                projection_supervisor
                                    .fail_closed(format!(
                                        "Worker event {} lost its generation-scoped route before desktop projection.",
                                        event.event_id
                                    ))
                                    .await;
                                continue;
                            }
                            let projection = {
                                let store = app_handle.state::<StudioStore>();
                                project_worker_event(store.inner(), &event)
                            };
                            match projection {
                                Ok(update) => {
                                    if let Err(error) = app_handle
                                        .emit("job-snapshot-updated", update.clone())
                                    {
                                        eprintln!(
                                            "Unable to emit targeted worker snapshot: {error}"
                                        );
                                    }
                                    if update.projected {
                                        if let Err(error) =
                                            app_handle.emit("snapshot-updated", update.snapshot)
                                        {
                                            eprintln!(
                                                "Unable to emit projected worker snapshot: {error}"
                                            );
                                        }
                                    }
                                }
                                Err(error) => {
                                    projection_supervisor
                                        .fail_closed(format!(
                                            "Worker event projection failed closed for generation {} event {}: {}",
                                            event.generation, event.event_id, error.message
                                        ))
                                        .await;
                                }
                            }
                        }
                        Ok(WorkerNotification::GenerationFailed {
                            generation,
                            job_ids,
                            message,
                        }) => {
                            let message =
                                format!("Worker generation {generation} failed closed: {message}");
                            for job_id_value in job_ids {
                                let job_id = match parse_job_id(&job_id_value) {
                                    Ok(job_id) => job_id,
                                    Err(error) => {
                                        eprintln!(
                                            "Unable to parse failed worker task identity: {}",
                                            error.message
                                        );
                                        continue;
                                    }
                                };
                                let command_gate = {
                                    let gates = app_handle.state::<JobCommandGate>();
                                    match gates.inner().for_job(&job_id) {
                                        Ok(command_gate) => command_gate,
                                        Err(error) => {
                                            eprintln!(
                                                "Unable to acquire failed task command gate: {}",
                                                error.message
                                            );
                                            continue;
                                        }
                                    }
                                };
                                let _command_guard = command_gate.lock().await;
                                let updates = {
                                    let store = app_handle.state::<StudioStore>();
                                    project_worker_failure(
                                        store.inner(),
                                        std::slice::from_ref(&job_id_value),
                                        &message,
                                    )
                                };
                                let updates = match updates {
                                    Ok(updates) => updates,
                                    Err(error) => {
                                        eprintln!(
                                            "Unable to project failed worker generation for {}: {}",
                                            job_id.as_str(),
                                            error.message
                                        );
                                        continue;
                                    }
                                };
                                for update in updates {
                                    if let Err(error) = app_handle
                                        .emit("job-snapshot-updated", update.clone())
                                    {
                                        eprintln!(
                                            "Unable to emit targeted failed worker snapshot: {error}"
                                        );
                                    }
                                    if update.projected {
                                        if let Err(error) =
                                            app_handle.emit("snapshot-updated", update.snapshot)
                                        {
                                            eprintln!(
                                                "Unable to emit failed projected worker snapshot: {error}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                            projection_supervisor
                                .fail_closed(format!(
                                    "Desktop worker-event projection lagged by {skipped} events; evidence continuity was lost."
                                ))
                                .await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
            });
            tauri::async_runtime::spawn(async move {
                if let Err(error) = supervisor.start().await {
                    eprintln!("Production worker startup failed: {error}");
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_snapshot,
            get_job_snapshot,
            get_job_status,
            list_jobs,
            select_job,
            create_job,
            cancel_job,
            update_speaker,
            apply_review_decision,
            open_artifact
        ])
        .build(tauri::generate_context!())
        .expect("failed to build MediaTranscribe Studio");

    app.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::ExitRequested { .. }) {
            let supervisor = app_handle.state::<WorkerSupervisor>().inner().clone();
            if let Err(error) = tauri::async_runtime::block_on(supervisor.shutdown()) {
                eprintln!("Production worker graceful shutdown failed: {error}");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env,
        fs::{self, File},
    };

    fn temp_workspace(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "media-transcribe-studio-{name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create temp workspace");
        root
    }

    fn valid_request_with_policy(
        root: &Path,
        speaker_policy: SpeakerCountPolicy,
        speaker_labels: Vec<String>,
    ) -> CreateJobRequest {
        let media = root.join("meeting.mov");
        File::create(&media).expect("create media");
        let output = root.join("output");
        fs::create_dir_all(&output).expect("create output");
        CreateJobRequest {
            idempotency_key: None,
            title: "Dynamic-speaker meeting".to_owned(),
            media_path: media,
            output_directory: output,
            strategy_id: ModelStrategyId::Balanced,
            speaker_policy,
            speaker_labels,
            language: "auto".to_owned(),
            local_llm_mode: LocalLlmMode::Disabled,
            local_llm_model: PRODUCTION_LOCAL_LLM_MODEL.to_owned(),
            local_llm_endpoint: "http://127.0.0.1:11434".to_owned(),
            local_llm_endpoint_policy: LOCAL_LLM_LOOPBACK_POLICY.to_owned(),
            llm_provider: Some(DEFAULT_LLM_PROVIDER.to_owned()),
            llm_api_key_env: None,
            llm_proxy_url: None,
            local_llm_auto_apply: false,
            translation_targets: Vec::new(),
            summary: false,
            output_locale: "en".to_owned(),
            business_prompt_version: BUSINESS_PROMPT_VERSION.to_owned(),
            output_customization: None,
        }
    }

    fn valid_manual_request(root: &Path, count: usize) -> CreateJobRequest {
        valid_request_with_policy(root, SpeakerCountPolicy::Manual { count }, labels(count))
    }

    #[test]
    fn accepts_model_override_and_rejects_empty_model() {
        let root = temp_workspace("model-override");
        let mut request = valid_manual_request(&root, 1);
        request.local_llm_model = "vendor/custom-semantic-model".to_owned();

        let prepared = prepare_job(request.clone()).expect("custom model must be accepted");
        assert_eq!(prepared.local_llm_model, "vendor/custom-semantic-model");

        request.local_llm_model.clear();

        let error = match prepare_job(request) {
            Ok(_) => panic!("empty model must fail closed"),
            Err(error) => error,
        };
        assert!(error.message.contains("localLlmModel"));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn validates_remote_provider_transport_without_accepting_raw_keys() {
        let root = temp_workspace("remote-provider");
        let mut request = valid_manual_request(&root, 1);
        request.local_llm_model = "gpt-5".to_owned();
        request.local_llm_endpoint = "https://relay.example.com/v1".to_owned();
        request.local_llm_endpoint_policy = LOCAL_LLM_REMOTE_POLICY.to_owned();
        request.llm_provider = Some("openai-compatible".to_owned());
        request.llm_api_key_env = Some("RELAY_API_KEY".to_owned());
        request.llm_proxy_url = Some("http://127.0.0.1:7890".to_owned());

        let prepared = prepare_job(request.clone()).expect("remote provider must be accepted");
        assert_eq!(prepared.llm_provider, "openai-compatible");
        assert_eq!(prepared.llm_api_key_env.as_deref(), Some("RELAY_API_KEY"));

        request.local_llm_endpoint = "http://relay.example.com/v1".to_owned();
        let error = prepare_job(request.clone()).expect_err("remote HTTP must fail closed");
        assert!(error.message.contains("HTTPS"));

        request.local_llm_endpoint = "https://relay.example.com/v1".to_owned();
        request.llm_api_key_env = Some("sk-raw-secret".to_owned());
        let error = prepare_job(request).expect_err("raw key must fail closed");
        assert!(error.message.contains("environment-variable name"));
        fs::remove_dir_all(root).expect("cleanup");
    }

    fn labels(count: usize) -> Vec<String> {
        (1..=count).map(|index| format!("Role {index}")).collect()
    }

    fn state_with_speakers(count: usize) -> AppState {
        let mut snapshot = default_snapshot();
        snapshot.speakers =
            create_speaker_profiles(count, &[]).expect("create dynamic speaker list");
        snapshot.job.speaker_count = Some(count);
        AppState {
            snapshot,
            output_root: None,
            worker_evidence: WorkerEvidenceLedger::default(),
            request_fingerprint: "test-request".to_owned(),
            worker_accepted: false,
        }
    }

    fn registry_test_state(job_id: &str, status: JobStatus) -> AppState {
        let mut state = state_with_speakers(1);
        state.snapshot.job.id = job_id.to_owned();
        state.snapshot.job.status = status;
        state
    }

    fn register_store_job(store: &StudioStore, job_id: &str) -> JobId {
        let job_id = JobId::new(job_id).expect("valid test job id");
        let outcome = store
            .registry
            .register_or_get(
                job_id.clone(),
                IdempotencyKey::new(format!("request-{job_id}"))
                    .expect("valid test idempotency key"),
                registry_test_state(job_id.as_str(), JobStatus::Registered),
            )
            .expect("register test job");
        assert!(matches!(outcome, RegisterOutcome::Created { .. }));
        job_id
    }

    fn pending_review(id: &str) -> ReviewSegment {
        ReviewSegment {
            id: id.to_owned(),
            start_ms: 0,
            end_ms: 1_000,
            timestamp_label: "00:00:00.000–00:00:01.000".to_owned(),
            raw_text: "原始中文文本".to_owned(),
            normalized_text: "原始中文文本".to_owned(),
            current_speaker_id: "speaker-1".to_owned(),
            candidates: Vec::new(),
            reasons: vec![ReviewReason::LocalAudioReview],
            confidence: 0.5,
            confidence_band: ConfidenceBand::Low,
            waveform: Vec::new(),
            locked: false,
            reviewed: false,
            audit_trail: Vec::new(),
        }
    }

    fn valid_review_decision(review_id: &str, speaker_id: &str) -> ReviewDecision {
        ReviewDecision {
            review_id: review_id.to_owned(),
            speaker_id: speaker_id.to_owned(),
            normalized_text: "人工确认后的中文原文。".to_owned(),
            reason: "依据人工复听确认角色归属。".to_owned(),
            evidence: "对比前后片段声纹及问答关系。".to_owned(),
            confidence: 0.95,
        }
    }

    fn commit_test_job(
        state: &mut AppState,
        request: CreateJobRequest,
        job_id: &str,
    ) -> IpcResult<CreateJobResult> {
        let prepared = prepare_job(request)?;
        let worker_output_directory =
            derive_worker_output_directory(&prepared.output_directory, job_id)?;
        let fingerprint = prepared_job_fingerprint(&prepared)?;
        *state =
            build_registered_job_state(&prepared, job_id, worker_output_directory, fingerprint)?;
        commit_accepted_job_in_state(state, job_id)
    }

    fn worker_response(kind: ResponseKind, job_id: &str, status: &str) -> WorkerResponse {
        WorkerResponse {
            kind,
            payload: Map::from_iter([
                ("jobId".to_owned(), Value::String(job_id.to_owned())),
                ("status".to_owned(), Value::String(status.to_owned())),
            ]),
        }
    }

    fn review_active_state(count: usize, job_id: &str) -> AppState {
        let mut state = state_with_speakers(count);
        state.snapshot.job.id = job_id.to_owned();
        state.snapshot.job.status = JobStatus::ReviewRequired;
        state.snapshot.job.speaker_policy = SpeakerCountPolicy::Manual { count };
        state
    }

    fn speaker_count_estimate(count: usize) -> Value {
        serde_json::json!({
            "estimatedCount": count,
            "confidence": 1.0,
            "candidateRange": {
                "min": count,
                "max": count
            },
            "method": "manual"
        })
    }

    fn transcript_speaker_scores(count: usize, assigned_speaker_id: &str) -> Vec<Value> {
        (1..=count)
            .map(|index| {
                let speaker_id = format!("speaker-{index}");
                serde_json::json!({
                    "speakerId": speaker_id,
                    "score": if speaker_id == assigned_speaker_id {
                        0.95
                    } else {
                        0.05
                    }
                })
            })
            .collect()
    }

    fn mutation_transcript_document(
        job_id: &str,
        speaker_labels: &[String],
        segment_speaker_id: &str,
        raw_text: &str,
        normalized_text: &str,
        human_locked: bool,
    ) -> Value {
        let speaker_count = speaker_labels.len();
        let speaker_ids = (1..=speaker_count)
            .map(|index| Value::String(format!("speaker-{index}")))
            .collect::<Vec<_>>();
        let speakers = speaker_labels
            .iter()
            .enumerate()
            .map(|(index, label)| {
                serde_json::json!({
                    "id": format!("speaker-{}", index + 1),
                    "name": label,
                    "role": label
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "schemaVersion": "2.0.0",
            "jobId": job_id,
            "source": {
                "durationMs": 1_000
            },
            "speakerPolicy": {
                "mode": "manual",
                "resolvedCount": speaker_count,
                "speakerIds": speaker_ids,
                "estimate": speaker_count_estimate(speaker_count)
            },
            "speakers": speakers,
            "segments": [{
                "id": "segment-1",
                "startMs": 0,
                "endMs": 1_000,
                "speakerId": segment_speaker_id,
                "rawText": raw_text,
                "normalizedText": normalized_text,
                "displayText": normalized_text,
                "confidence": 0.95,
                "speakerScores": transcript_speaker_scores(
                    speaker_count,
                    segment_speaker_id
                ),
                "humanLocked": human_locked
            }]
        })
    }

    fn artifact_item(
        id: &str,
        name: &str,
        kind: ArtifactKind,
        relative_path: &str,
        sha256: &str,
    ) -> ArtifactItem {
        ArtifactItem {
            id: id.to_owned(),
            name: name.to_owned(),
            kind,
            relative_path: relative_path.to_owned(),
            size_label: "test fixture".to_owned(),
            created_at: "2026-07-22T12:00:00Z".to_owned(),
            integrity: IntegrityStatus::Verified,
            sha256: Some(sha256.to_owned()),
        }
    }

    fn install_mutation_artifacts(
        state: &mut AppState,
        root: &Path,
        transcript: &Value,
        queue: &Value,
    ) -> (PathBuf, String, PathBuf, String) {
        let transcript_path = root.join("transcript").join("transcript-document.json");
        let queue_path = root.join("review").join("review-queue.json");
        fs::create_dir_all(transcript_path.parent().expect("transcript fixture parent"))
            .expect("create transcript fixture directory");
        fs::create_dir_all(queue_path.parent().expect("queue fixture parent"))
            .expect("create queue fixture directory");
        fs::write(
            &transcript_path,
            canonical_json_bytes(transcript).expect("canonical transcript fixture"),
        )
        .expect("write transcript fixture");
        fs::write(
            &queue_path,
            canonical_json_bytes(queue).expect("canonical queue fixture"),
        )
        .expect("write queue fixture");

        let canonical_root = fs::canonicalize(root).expect("canonical fixture root");
        let transcript_path =
            fs::canonicalize(transcript_path).expect("canonical transcript fixture");
        let queue_path = fs::canonicalize(queue_path).expect("canonical queue fixture");
        let transcript_sha256 =
            canonical_json_sha256(transcript).expect("transcript fixture digest");
        let queue_sha256 = canonical_json_sha256(queue).expect("queue fixture digest");
        state.output_root = Some(canonical_root);
        state.worker_evidence.verified_artifacts.insert(
            transcript_path.clone(),
            VerifiedArtifactEvidence {
                artifact_type: "transcript-document-v2".to_owned(),
                canonical_path: transcript_path.clone(),
                sha256: transcript_sha256.clone(),
            },
        );
        state.worker_evidence.verified_artifacts.insert(
            queue_path.clone(),
            VerifiedArtifactEvidence {
                artifact_type: "review-queue-v2".to_owned(),
                canonical_path: queue_path.clone(),
                sha256: queue_sha256.clone(),
            },
        );
        state.snapshot.artifacts = vec![
            artifact_item(
                "artifact-transcript",
                "Transcript document",
                ArtifactKind::TranscriptJson,
                "transcript/transcript-document.json",
                &transcript_sha256,
            ),
            artifact_item(
                "artifact-review-queue",
                "Review queue",
                ArtifactKind::RepairQueue,
                "review/review-queue.json",
                &queue_sha256,
            ),
        ];
        (transcript_path, transcript_sha256, queue_path, queue_sha256)
    }

    fn transcript_evidence(document: &Value, job_id: &str, count: usize) -> WorkerEvidenceLedger {
        let transcript =
            project_transcript_document(document, job_id, &SpeakerCountPolicy::Manual { count })
                .expect("valid transcript evidence fixture");
        WorkerEvidenceLedger {
            transcript_duration_ms: Some(transcript.duration_ms),
            transcript_segment_count: Some(transcript.segments.len()),
            transcript_speaker_count_mode: Some(transcript.speaker_count_mode),
            transcript_speaker_count_estimate: Some(transcript.speaker_count_estimate),
            transcript_segments: transcript.segments,
            ..WorkerEvidenceLedger::default()
        }
    }

    fn projection_review_queue(job_id: &str, count: usize) -> Value {
        serde_json::json!({
            "schemaVersion": "2.0.0",
            "jobId": job_id,
            "speakerCountMode": "manual",
            "speakerCountEstimate": speaker_count_estimate(count),
            "createdAt": "2026-07-22T12:00:00Z",
            "updatedAt": "2026-07-22T12:01:00Z",
            "items": [
                {
                    "id": "speaker-count-confidence",
                    "scope": "job",
                    "reasonCode": "SPEAKER_COUNT_LOW_CONFIDENCE",
                    "status": "open",
                    "speakerCountEstimate": speaker_count_estimate(count)
                },
                {
                    "id": "review-1",
                    "scope": "segment",
                    "segmentId": "segment-1",
                    "reasonCode": "SPEAKER_MARGIN_LOW",
                    "status": "open",
                    "timeRange": {
                        "startMs": 0,
                        "endMs": 1_000
                    },
                    "speakerId": "speaker-1",
                    "speakerCandidates": transcript_speaker_scores(count, "speaker-1"),
                    "text": {
                        "rawText": "原始中文文本",
                        "normalizedText": "原始中文文本",
                        "displayText": "原始中文文本"
                    },
                    "evidenceRefs": ["local-audio:segment-1"]
                }
            ],
            "decisions": [],
            "openCount": 2
        })
    }

    fn pipeline_metrics_document(job_id: &str, reference_available: bool) -> Value {
        let mut document = serde_json::json!({
            "schemaVersion": "1.0.0",
            "jobId": job_id,
            "offline": true,
            "durationMs": 1_000,
            "runtime": {
                "elapsedMs": 500.0,
                "rtf": 0.5,
                "stages": {
                    "asr": {
                        "count": 1,
                        "totalMs": 500.0,
                        "p50Ms": 400.0,
                        "p95Ms": 500.0
                    }
                }
            },
            "cache": {
                "requests": 2,
                "hits": 1,
                "misses": 1,
                "recomputations": 1,
                "hitRate": 0.5,
                "recomputationRate": 0.5,
                "byStage": {
                    "asr": {
                        "requests": 2,
                        "hits": 1,
                        "misses": 1,
                        "recomputations": 1,
                        "hitRate": 0.5,
                        "recomputationRate": 0.5
                    }
                }
            },
            "routing": {
                "segments": 1,
                "escalated": 1,
                "escalationRate": 1.0,
                "protected": 0,
                "reasonCounts": {
                    "LOW_MARGIN": 1
                }
            },
            "escalations": [],
            "cascade": [],
            "resources": {
                "peakRamMb": 2_048.0,
                "peakVramMb": 4_096.0
            },
            "policy": {},
            "referenceEvaluation": {
                "available": reference_available
            }
        });
        if reference_available {
            document.as_object_mut().expect("metrics object").insert(
                "quality".to_owned(),
                serde_json::json!({
                    "der": 0.125,
                    "jer": 0.25,
                    "speakerConfusion": 0.05,
                    "overlapF1": 0.875
                }),
            );
        }
        document
    }

    fn worker_event(
        job_id: &str,
        event_id: &str,
        event_type: &str,
        payload: Map<String, Value>,
    ) -> WorkerEvent {
        WorkerEvent {
            generation: 1,
            event_id: event_id.to_owned(),
            job_id: job_id.to_owned(),
            sequence: 1,
            timestamp: "2026-07-22T12:00:00Z".to_owned(),
            event_type: event_type.to_owned(),
            payload,
        }
    }

    fn completion_artifact_specs() -> [(&'static str, &'static str); 9] {
        [
            (
                "transcript-document-v2",
                "transcript/transcript-document.json",
            ),
            ("review-queue-v2", "review/review-queue.json"),
            ("pdf", "pdf/report.pdf"),
            ("pdf-quality-report-v1", "pdf/quality-report.json"),
            ("pdf-render-manifest-v1", "pdf/render-manifest.json"),
            ("pdf-report-document-v1", "pdf/report-document.json"),
            ("pdf-canonical-xhtml", "pdf/report.xhtml"),
            ("pdf-repair-queue-v1", "pdf/repair-queue.json"),
            ("pdf-contact-sheet", "pdf/contact-sheet.png"),
        ]
    }

    fn install_completion_artifacts(state: &mut AppState, root: &Path) -> Vec<String> {
        let canonical_root = fs::canonicalize(root).expect("canonical completion root");
        state.output_root = Some(canonical_root);
        let mut paths = Vec::new();
        for (index, (artifact_type, relative_path)) in
            completion_artifact_specs().into_iter().enumerate()
        {
            let path = root.join(relative_path);
            fs::create_dir_all(path.parent().expect("completion artifact parent"))
                .expect("create completion artifact directory");
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .expect("completion artifact extension")
                .to_owned();
            let bytes = if extension == "json" {
                canonical_json_bytes(&serde_json::json!({
                    "artifactType": artifact_type,
                    "fixture": index
                }))
                .expect("canonical completion JSON")
            } else {
                match extension.as_str() {
                    "pdf" => b"%PDF-1.7\ntruthful-worker-evidence\n%%EOF\n".to_vec(),
                    "xhtml" => b"<html><body>verified</body></html>\n".to_vec(),
                    "png" => b"\x89PNG\r\n\x1a\nverified-contact-sheet".to_vec(),
                    _ => panic!("unsupported completion fixture extension"),
                }
            };
            fs::write(&path, bytes).expect("write completion artifact");
            let canonical_path = fs::canonicalize(&path).expect("canonical completion artifact");
            let sha256 = if extension == "json" {
                strict_json_file(&canonical_path, "completion fixture")
                    .expect("strict completion JSON")
                    .1
            } else {
                format!(
                    "{:x}",
                    Sha256::digest(fs::read(&canonical_path).expect("read fixture"))
                )
            };
            state.worker_evidence.verified_artifacts.insert(
                canonical_path.clone(),
                VerifiedArtifactEvidence {
                    artifact_type: artifact_type.to_owned(),
                    canonical_path: canonical_path.clone(),
                    sha256,
                },
            );
            paths.push(canonical_path.to_string_lossy().into_owned());
        }
        paths
    }

    fn completion_event(job_id: &str, artifact_paths: &[String]) -> WorkerEvent {
        worker_event(
            job_id,
            "event-completed",
            "job.completed",
            Map::from_iter([
                ("status".to_owned(), Value::String("completed".to_owned())),
                (
                    "artifactPaths".to_owned(),
                    Value::Array(artifact_paths.iter().cloned().map(Value::String).collect()),
                ),
                ("business".to_owned(), Value::Object(Map::new())),
            ]),
        )
    }

    fn completed_decision(envelope: &HumanMutationEnvelope, command: &str) -> Map<String, Value> {
        Map::from_iter([
            (
                "decisionId".to_owned(),
                Value::String(envelope.decision_id.clone()),
            ),
            ("command".to_owned(), Value::String(command.to_owned())),
            ("reason".to_owned(), Value::String(envelope.reason.clone())),
            (
                "evidence".to_owned(),
                Value::Array(
                    envelope
                        .evidence
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            ),
            ("confidence".to_owned(), Value::from(envelope.confidence)),
            ("audit".to_owned(), Value::Object(envelope.audit.clone())),
            (
                "recordedAt".to_owned(),
                Value::String("2026-07-22T12:00:00Z".to_owned()),
            ),
        ])
    }

    fn completed_mutation_response(
        job_id: &str,
        command: &str,
        envelope: &HumanMutationEnvelope,
        open_count: usize,
        speaker_count: usize,
    ) -> WorkerResponse {
        completed_mutation_response_with_hash(
            job_id,
            command,
            envelope,
            open_count,
            speaker_count,
            &"a".repeat(64),
        )
    }

    fn completed_mutation_response_with_hash(
        job_id: &str,
        command: &str,
        envelope: &HumanMutationEnvelope,
        open_count: usize,
        speaker_count: usize,
        document_hash: &str,
    ) -> WorkerResponse {
        WorkerResponse {
            kind: ResponseKind::Completed,
            payload: Map::from_iter([
                ("jobId".to_owned(), Value::String(job_id.to_owned())),
                (
                    "status".to_owned(),
                    Value::String("review_required".to_owned()),
                ),
                ("command".to_owned(), Value::String(command.to_owned())),
                (
                    "decision".to_owned(),
                    Value::Object(completed_decision(envelope, command)),
                ),
                ("openCount".to_owned(), Value::from(open_count as u64)),
                (
                    "documentHash".to_owned(),
                    Value::String(document_hash.to_owned()),
                ),
                ("speakerCount".to_owned(), Value::from(speaker_count as u64)),
            ]),
        }
    }

    fn review_queue_response(
        job_id: &str,
        receipt: &MutationReceipt,
        items: Vec<Value>,
        decisions: Vec<Value>,
    ) -> WorkerResponse {
        WorkerResponse {
            kind: ResponseKind::Completed,
            payload: Map::from_iter([
                ("jobId".to_owned(), Value::String(job_id.to_owned())),
                (
                    "status".to_owned(),
                    Value::String("review_required".to_owned()),
                ),
                (
                    "openCount".to_owned(),
                    Value::from(receipt.open_count as u64),
                ),
                (
                    "documentHash".to_owned(),
                    Value::String(receipt.document_hash.clone()),
                ),
                (
                    "speakerCount".to_owned(),
                    Value::from(receipt.speaker_count as u64),
                ),
                (
                    "queue".to_owned(),
                    Value::Object(Map::from_iter([
                        (
                            "schemaVersion".to_owned(),
                            Value::String("2.0.0".to_owned()),
                        ),
                        ("jobId".to_owned(), Value::String(job_id.to_owned())),
                        (
                            "speakerCountMode".to_owned(),
                            Value::String("manual".to_owned()),
                        ),
                        (
                            "speakerCountEstimate".to_owned(),
                            speaker_count_estimate(receipt.speaker_count),
                        ),
                        (
                            "createdAt".to_owned(),
                            Value::String("2026-07-22T12:00:00Z".to_owned()),
                        ),
                        (
                            "updatedAt".to_owned(),
                            Value::String("2026-07-22T12:01:00Z".to_owned()),
                        ),
                        (
                            "openCount".to_owned(),
                            Value::from(receipt.open_count as u64),
                        ),
                        ("items".to_owned(), Value::Array(items)),
                        ("decisions".to_owned(), Value::Array(decisions)),
                    ])),
                ),
            ]),
        }
    }

    fn review_submission_queue_response(
        prepared: &PreparedReviewMutation,
        receipt: &MutationReceipt,
    ) -> WorkerResponse {
        let item = Value::Object(Map::from_iter([
            ("id".to_owned(), Value::String(prepared.review_id.clone())),
            ("scope".to_owned(), Value::String("segment".to_owned())),
            (
                "segmentId".to_owned(),
                Value::String("segment-1".to_owned()),
            ),
            (
                "reasonCode".to_owned(),
                Value::String("SPEAKER_MARGIN_LOW".to_owned()),
            ),
            ("status".to_owned(), Value::String("accepted".to_owned())),
            (
                "timeRange".to_owned(),
                serde_json::json!({
                    "startMs": 0,
                    "endMs": 1_000
                }),
            ),
            (
                "speakerId".to_owned(),
                Value::String(prepared.speaker_id.clone()),
            ),
            (
                "speakerCandidates".to_owned(),
                Value::Array(transcript_speaker_scores(
                    prepared.speaker_count,
                    &prepared.speaker_id,
                )),
            ),
            (
                "text".to_owned(),
                Value::Object(Map::from_iter([
                    (
                        "rawText".to_owned(),
                        Value::String(prepared.raw_text.clone()),
                    ),
                    (
                        "normalizedText".to_owned(),
                        Value::String(prepared.normalized_text.clone()),
                    ),
                    (
                        "displayText".to_owned(),
                        Value::String(prepared.normalized_text.clone()),
                    ),
                ])),
            ),
            (
                "evidenceRefs".to_owned(),
                Value::Array(vec![Value::String("local-audio:segment-1".to_owned())]),
            ),
            (
                "decision".to_owned(),
                Value::Object(receipt.decision.clone()),
            ),
        ]));
        let mut decision = receipt.decision.clone();
        decision.insert(
            "itemId".to_owned(),
            Value::String(prepared.review_id.clone()),
        );
        decision.insert("status".to_owned(), Value::String("accepted".to_owned()));
        review_queue_response(
            &prepared.job_id,
            receipt,
            vec![item],
            vec![Value::Object(decision)],
        )
    }

    fn speaker_rename_queue_response(
        prepared: &PreparedSpeakerRename,
        receipt: &MutationReceipt,
    ) -> WorkerResponse {
        let mut decision = receipt.decision.clone();
        decision.insert(
            "speakerId".to_owned(),
            Value::String(prepared.speaker_id.clone()),
        );
        decision.insert(
            "after".to_owned(),
            Value::Object(Map::from_iter([(
                "name".to_owned(),
                Value::String(prepared.next_label.clone()),
            )])),
        );
        review_queue_response(
            &prepared.job_id,
            receipt,
            Vec::new(),
            vec![Value::Object(decision)],
        )
    }

    #[test]
    fn rejects_relative_media_path() {
        let root = temp_workspace("relative-media");
        let mut request = valid_manual_request(&root, 3);
        request.media_path = PathBuf::from("meeting.mov");
        let error = prepare_job(request).expect_err("relative media must fail");
        assert!(matches!(error.code, IpcErrorCode::InvalidPath));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn rejects_missing_media_and_defers_format_support_to_content_probe() {
        let root = temp_workspace("bad-media");
        let mut missing = valid_manual_request(&root, 3);
        missing.media_path = root.join("missing.mov");
        assert!(matches!(
            prepare_job(missing)
                .expect_err("missing media must fail")
                .code,
            IpcErrorCode::NotFound
        ));

        for name in ["meeting.uncommon", "extensionless"] {
            let mut content_probed = valid_manual_request(&root, 3);
            content_probed.media_path = root.join(name);
            File::create(&content_probed.media_path).expect("create media candidate");
            let prepared = prepare_job(content_probed)
                .expect("native intake must defer format support to the worker content probe");
            assert!(prepared.media_path.is_file());
        }
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn rejects_relative_output_directory() {
        let root = temp_workspace("relative-output");
        let mut request = valid_manual_request(&root, 3);
        request.output_directory = PathBuf::from("output");
        let error = prepare_job(request).expect_err("relative output must fail");
        assert!(matches!(error.code, IpcErrorCode::InvalidPath));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn accepts_dynamic_manual_counts_and_contiguous_ids() {
        for count in [1, 2, 5, 8, 13] {
            let speaker_labels = labels(count);
            let profiles = create_speaker_profiles(count, &speaker_labels).expect("dynamic count");
            assert_eq!(profiles.len(), count);
            for (index, speaker) in profiles.iter().enumerate() {
                assert_eq!(speaker.id, format!("speaker-{}", index + 1));
                assert_eq!(speaker.label, format!("Role {}", index + 1));
            }
            assert_eq!(
                profiles.last().expect("last speaker").id,
                format!("speaker-{count}")
            );
        }
    }

    #[test]
    fn validates_auto_manual_and_hybrid_policies() {
        assert!(SpeakerCountPolicy::Auto {}.validate().is_ok());
        assert!(SpeakerCountPolicy::Manual { count: 7 }.validate().is_ok());
        assert!(SpeakerCountPolicy::Hybrid {
            min_speakers: 2,
            max_speakers: 9,
            prior_count: 5,
        }
        .validate()
        .is_ok());

        assert!(matches!(
            SpeakerCountPolicy::Manual { count: 0 }
                .validate()
                .expect_err("zero speakers must fail")
                .code,
            IpcErrorCode::InvalidRequest
        ));
        assert!(matches!(
            SpeakerCountPolicy::Hybrid {
                min_speakers: 5,
                max_speakers: 3,
                prior_count: 4,
            }
            .validate()
            .expect_err("reversed bounds must fail")
            .code,
            IpcErrorCode::InvalidRequest
        ));
        assert!(matches!(
            SpeakerCountPolicy::Hybrid {
                min_speakers: 2,
                max_speakers: 4,
                prior_count: 5,
            }
            .validate()
            .expect_err("prior outside bounds must fail")
            .code,
            IpcErrorCode::InvalidRequest
        ));
    }

    #[test]
    fn serde_rejects_unknown_fields_and_missing_policy() {
        let missing_policy = serde_json::json!({
            "title": "会议",
            "mediaPath": "C:\\meeting.mov",
            "outputDirectory": "C:\\output",
            "strategyId": "balanced",
            "speakerLabels": []
        });
        assert!(serde_json::from_value::<CreateJobRequest>(missing_policy).is_err());

        let unknown_field = serde_json::json!({
            "title": "会议",
            "mediaPath": "C:\\meeting.mov",
            "outputDirectory": "C:\\output",
            "strategyId": "balanced",
            "speakerPolicy": {"mode": "auto", "count": 5},
            "speakerLabels": []
        });
        assert!(serde_json::from_value::<CreateJobRequest>(unknown_field).is_err());
    }

    #[test]
    fn output_customization_serde_accepts_only_json_objects() {
        let root = temp_workspace("output-customization-serde");
        let base =
            serde_json::to_value(valid_manual_request(&root, 3)).expect("serialize valid request");
        assert!(
            base.get("outputCustomization").is_none(),
            "an absent customization must stay absent rather than serialize as null"
        );

        let expected = Map::from_iter([
            (
                "reportStyle".to_owned(),
                Value::String("editorial".to_owned()),
            ),
            (
                "subtitle".to_owned(),
                serde_json::json!({
                    "preset": "youtube",
                    "fontSize": 48,
                    "burnIn": true
                }),
            ),
        ]);
        let mut valid = base.clone();
        valid.as_object_mut().expect("request object").insert(
            "outputCustomization".to_owned(),
            Value::Object(expected.clone()),
        );
        let request =
            serde_json::from_value::<CreateJobRequest>(valid).expect("object must deserialize");
        assert_eq!(request.output_customization.as_ref(), Some(&expected));

        for invalid in [
            Value::Null,
            Value::Array(Vec::new()),
            Value::String("editorial".to_owned()),
            Value::from(1),
            Value::Bool(true),
        ] {
            let mut candidate = base.clone();
            candidate
                .as_object_mut()
                .expect("request object")
                .insert("outputCustomization".to_owned(), invalid);
            assert!(
                serde_json::from_value::<CreateJobRequest>(candidate).is_err(),
                "non-object outputCustomization must fail closed"
            );
        }

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn output_customization_is_forwarded_exactly_and_omitted_when_absent() {
        let root = temp_workspace("output-customization-payload");
        let expected = Map::from_iter([
            (
                "pdf".to_owned(),
                serde_json::json!({
                    "theme": "aurora",
                    "fontFamily": "Noto Sans CJK SC",
                    "qualityGate": {
                        "enabled": true,
                        "minimumScore": 92
                    }
                }),
            ),
            (
                "subtitles".to_owned(),
                serde_json::json!([
                    {"format": "srt"},
                    {"format": "ass", "preset": "youtube"}
                ]),
            ),
        ]);
        let mut request = valid_manual_request(&root, 3);
        request.output_customization = Some(expected.clone());
        let prepared = prepare_job(request).expect("prepare customization request");
        let worker_output =
            derive_worker_output_directory(&prepared.output_directory, "job-custom-output")
                .expect("derive worker output");
        let payload = build_job_start_payload(&prepared, "job-custom-output", &worker_output)
            .expect("build worker payload");
        assert_eq!(
            payload.get("outputCustomization"),
            Some(&Value::Object(expected))
        );

        let without_customization =
            prepare_job(valid_manual_request(&root, 3)).expect("prepare default request");
        let default_output =
            derive_worker_output_directory(&without_customization.output_directory, "job-default")
                .expect("derive default worker output");
        let default_payload =
            build_job_start_payload(&without_customization, "job-default", &default_output)
                .expect("build default worker payload");
        assert!(!default_payload.contains_key("outputCustomization"));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn output_customization_participates_in_idempotency_fingerprint() {
        let root = temp_workspace("output-customization-fingerprint");
        let base = prepare_job(valid_manual_request(&root, 3)).expect("prepare base request");
        let base_fingerprint = prepared_job_fingerprint(&base).expect("base fingerprint");

        let mut customized_request = valid_manual_request(&root, 3);
        customized_request.output_customization = Some(Map::from_iter([(
            "pdf".to_owned(),
            serde_json::json!({"theme": "aurora"}),
        )]));
        let customized = prepare_job(customized_request).expect("prepare customized request");
        let customized_fingerprint =
            prepared_job_fingerprint(&customized).expect("customized fingerprint");

        assert_ne!(base_fingerprint, customized_fingerprint);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn output_customization_rejects_payloads_over_ipc_limit() {
        let root = temp_workspace("output-customization-limit");
        let mut request = valid_manual_request(&root, 3);
        request.output_customization = Some(Map::from_iter([(
            "oversized".to_owned(),
            Value::String("x".repeat(MAX_OUTPUT_CUSTOMIZATION_BYTES)),
        )]));

        let error = prepare_job(request).expect_err("oversized customization must fail");
        assert!(matches!(error.code, IpcErrorCode::InvalidRequest));
        assert!(error.message.contains("IPC limit"));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn allows_empty_or_complete_labels_and_rejects_partial_lists() {
        let root = temp_workspace("label-cardinality");

        let empty_manual =
            valid_request_with_policy(&root, SpeakerCountPolicy::Manual { count: 3 }, Vec::new());
        assert!(prepare_job(empty_manual).is_ok());

        let complete = valid_manual_request(&root, 3);
        assert!(prepare_job(complete).is_ok());

        let partial = valid_request_with_policy(
            &root,
            SpeakerCountPolicy::Manual { count: 3 },
            vec!["主持".to_owned(), "产品".to_owned()],
        );
        assert!(matches!(
            prepare_job(partial)
                .expect_err("partial labels must fail")
                .code,
            IpcErrorCode::InvalidRequest
        ));

        let auto = valid_request_with_policy(&root, SpeakerCountPolicy::Auto {}, Vec::new());
        assert!(prepare_job(auto).is_ok());
        let auto_with_label = valid_request_with_policy(
            &root,
            SpeakerCountPolicy::Auto {},
            vec!["不应出现".to_owned()],
        );
        assert!(matches!(
            prepare_job(auto_with_label)
                .expect_err("auto mode must not accept labels")
                .code,
            IpcErrorCode::InvalidRequest
        ));

        let hybrid = valid_request_with_policy(
            &root,
            SpeakerCountPolicy::Hybrid {
                min_speakers: 2,
                max_speakers: 8,
                prior_count: 4,
            },
            labels(4),
        );
        assert!(prepare_job(hybrid).is_ok());

        let empty_hybrid = valid_request_with_policy(
            &root,
            SpeakerCountPolicy::Hybrid {
                min_speakers: 2,
                max_speakers: 8,
                prior_count: 4,
            },
            Vec::new(),
        );
        assert!(prepare_job(empty_hybrid).is_ok());

        let partial_hybrid = valid_request_with_policy(
            &root,
            SpeakerCountPolicy::Hybrid {
                min_speakers: 2,
                max_speakers: 8,
                prior_count: 4,
            },
            labels(3),
        );
        assert!(matches!(
            prepare_job(partial_hybrid)
                .expect_err("partial hybrid labels must fail")
                .code,
            IpcErrorCode::InvalidRequest
        ));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn creates_jobs_for_all_policy_modes_without_fixed_cardinality() {
        for count in [1, 2, 5, 8, 13] {
            let manual_root = temp_workspace(&format!("manual-job-{count}"));
            let mut manual_state = state_with_speakers(1);
            commit_test_job(
                &mut manual_state,
                valid_manual_request(&manual_root, count),
                &format!("job-manual-{count}"),
            )
            .expect("manual job");
            assert_eq!(manual_state.snapshot.job.speaker_count, Some(count));
            assert_eq!(manual_state.snapshot.speakers.len(), count);
            assert!(manual_state.snapshot.job.speaker_detection.is_none());
            assert_eq!(
                manual_state.snapshot.speakers[count - 1].id,
                format!("speaker-{count}")
            );
            assert_eq!(
                manual_state.snapshot.speakers[count - 1].label,
                format!("Role {count}")
            );
            fs::remove_dir_all(manual_root).expect("cleanup");
        }

        let auto_root = temp_workspace("auto-job");
        let mut auto_state = state_with_speakers(3);
        let auto_request =
            valid_request_with_policy(&auto_root, SpeakerCountPolicy::Auto {}, Vec::new());
        commit_test_job(&mut auto_state, auto_request, "job-auto").expect("auto job");
        assert!(auto_state.snapshot.job.speaker_count.is_none());
        assert!(auto_state.snapshot.job.speaker_detection.is_none());
        assert!(auto_state.snapshot.speakers.is_empty());

        let hybrid_root = temp_workspace("hybrid-job");
        let mut hybrid_state = state_with_speakers(1);
        let hybrid_request = valid_request_with_policy(
            &hybrid_root,
            SpeakerCountPolicy::Hybrid {
                min_speakers: 2,
                max_speakers: 8,
                prior_count: 4,
            },
            vec![
                "主持".to_owned(),
                "产品".to_owned(),
                "工程".to_owned(),
                "设计".to_owned(),
            ],
        );
        commit_test_job(&mut hybrid_state, hybrid_request, "job-hybrid").expect("hybrid job");
        assert!(hybrid_state.snapshot.job.speaker_count.is_none());
        assert!(hybrid_state.snapshot.job.speaker_detection.is_none());
        assert!(hybrid_state.snapshot.speakers.is_empty());

        let default_label_root = temp_workspace("default-label-job");
        let mut default_label_state = state_with_speakers(1);
        let default_label_request = valid_request_with_policy(
            &default_label_root,
            SpeakerCountPolicy::Manual { count: 3 },
            Vec::new(),
        );
        commit_test_job(
            &mut default_label_state,
            default_label_request,
            "job-default-labels",
        )
        .expect("manual job with generated default labels");
        assert_eq!(default_label_state.snapshot.speakers[0].label, "Speaker 1");
        assert_eq!(default_label_state.snapshot.speakers[2].label, "Speaker 3");

        fs::remove_dir_all(auto_root).expect("cleanup");
        fs::remove_dir_all(hybrid_root).expect("cleanup");
        fs::remove_dir_all(default_label_root).expect("cleanup");
    }

    #[test]
    fn builds_strict_worker_payloads_for_all_speaker_modes() {
        let auto_root = temp_workspace("payload-auto");
        let auto = prepare_job(valid_request_with_policy(
            &auto_root,
            SpeakerCountPolicy::Auto {},
            Vec::new(),
        ))
        .expect("prepare auto");
        let auto_output =
            derive_worker_output_directory(&auto.output_directory, "job-payload-auto")
                .expect("auto output");
        let auto_payload =
            build_job_start_payload(&auto, "job-payload-auto", &auto_output).expect("auto payload");
        assert_eq!(auto_payload["speakerCountMode"], "auto");
        for absent in [
            "speakerCount",
            "speakerRoles",
            "speakerCountBounds",
            "speakerCountPrior",
            "strategyId",
        ] {
            assert!(
                !auto_payload.contains_key(absent),
                "{absent} must not be sent in auto mode"
            );
        }

        let manual_root = temp_workspace("payload-manual");
        let manual =
            prepare_job(valid_manual_request(&manual_root, 5)).expect("prepare manual payload");
        let manual_output =
            derive_worker_output_directory(&manual.output_directory, "job-payload-manual")
                .expect("manual output");
        let manual_payload = build_job_start_payload(&manual, "job-payload-manual", &manual_output)
            .expect("manual payload");
        assert_eq!(manual_payload["speakerCountMode"], "manual");
        assert_eq!(manual_payload["speakerCount"], 5);
        assert_eq!(
            manual_payload["speakerRoles"]
                .as_array()
                .expect("speaker roles")
                .len(),
            5
        );
        assert!(!manual_payload.contains_key("strategyId"));

        let empty_manual_root = temp_workspace("payload-empty-manual");
        let empty_manual = prepare_job(valid_request_with_policy(
            &empty_manual_root,
            SpeakerCountPolicy::Manual { count: 3 },
            Vec::new(),
        ))
        .expect("prepare empty manual");
        let empty_manual_output =
            derive_worker_output_directory(&empty_manual.output_directory, "job-empty-manual")
                .expect("empty manual output");
        let empty_manual_payload =
            build_job_start_payload(&empty_manual, "job-empty-manual", &empty_manual_output)
                .expect("empty manual payload");
        assert!(!empty_manual_payload.contains_key("speakerRoles"));

        let hybrid_root = temp_workspace("payload-hybrid");
        let hybrid = prepare_job(valid_request_with_policy(
            &hybrid_root,
            SpeakerCountPolicy::Hybrid {
                min_speakers: 2,
                max_speakers: 8,
                prior_count: 4,
            },
            labels(4),
        ))
        .expect("prepare hybrid");
        let hybrid_output =
            derive_worker_output_directory(&hybrid.output_directory, "job-payload-hybrid")
                .expect("hybrid output");
        let hybrid_payload = build_job_start_payload(&hybrid, "job-payload-hybrid", &hybrid_output)
            .expect("hybrid payload");
        assert_eq!(hybrid_payload["speakerCountMode"], "hybrid");
        assert_eq!(hybrid_payload["speakerCountBounds"]["min"], 2);
        assert_eq!(hybrid_payload["speakerCountBounds"]["max"], 8);
        assert_eq!(hybrid_payload["speakerCountPrior"], 4);
        assert!(!hybrid_payload.contains_key("speakerCount"));
        assert!(!hybrid_payload.contains_key("speakerRoles"));
        assert!(!hybrid_payload.contains_key("strategyId"));

        for payload in [
            &auto_payload,
            &manual_payload,
            &empty_manual_payload,
            &hybrid_payload,
        ] {
            assert_eq!(payload["renderPdf"], true);
            assert_eq!(payload["language"], "auto");
            assert_eq!(payload["localLlmMode"], "disabled");
            assert_eq!(payload["localLlmModel"], PRODUCTION_LOCAL_LLM_MODEL);
            assert_eq!(payload["localLlmAutoApply"], false);
        }

        for root in [auto_root, manual_root, empty_manual_root, hybrid_root] {
            fs::remove_dir_all(root).expect("cleanup");
        }
    }

    #[test]
    fn derives_unique_uncreated_worker_output_directories_and_safe_job_ids() {
        let root = temp_workspace("worker-output");
        let request = valid_manual_request(&root, 2);
        let prepared = prepare_job(request).expect("prepare");
        let first_id = next_job_id();
        let second_id = next_job_id();
        assert_ne!(first_id, second_id);
        for job_id in [&first_id, &second_id] {
            assert!(job_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')));
            let output = derive_worker_output_directory(&prepared.output_directory, job_id)
                .expect("derive output");
            assert!(output.starts_with(&prepared.output_directory));
            assert!(
                !output.exists(),
                "Rust must not pre-create the job directory"
            );
        }
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[tokio::test]
    async fn command_gates_are_stable_per_job_and_independent_across_jobs() {
        let gates = JobCommandGate::default();
        let job_a = JobId::new("job-a").expect("job a");
        let job_b = JobId::new("job-b").expect("job b");
        let first_a = gates.for_job(&job_a).expect("first job a gate");
        let second_a = gates.for_job(&job_a).expect("second job a gate");
        let first_b = gates.for_job(&job_b).expect("job b gate");

        assert!(Arc::ptr_eq(&first_a, &second_a));
        assert!(!Arc::ptr_eq(&first_a, &first_b));
        let guard_a = first_a.lock().await;
        assert!(
            second_a.try_lock().is_err(),
            "same-job commands must serialize"
        );
        let guard_b = first_b
            .try_lock()
            .expect("different-job commands must remain independent");
        drop((guard_b, guard_a));
        let _second_guard = second_a
            .try_lock()
            .expect("same-job gate must become available after release");
    }

    #[test]
    fn replay_candidate_gates_are_discarded_without_removing_published_job_gates() {
        let store = StudioStore::new();
        let gates = JobCommandGate::default();
        let unpublished = JobId::new("job-unpublished").expect("unpublished job");
        let unpublished_gate = gates.for_job(&unpublished).expect("candidate gate");
        store
            .discard_gate_if_unpublished(&gates, &unpublished, &unpublished_gate)
            .expect("discard unpublished gate");
        assert!(!gates
            .0
            .lock()
            .expect("gate registry")
            .contains_key(&unpublished));

        let published = register_store_job(&store, "job-published");
        let published_gate = gates.for_job(&published).expect("published gate");
        store
            .discard_gate_if_unpublished(&gates, &published, &published_gate)
            .expect("published gate must be retained");
        let retained = gates
            .for_job(&published)
            .expect("retained published command gate");
        assert!(Arc::ptr_eq(&published_gate, &retained));
    }

    #[test]
    fn dispatch_ledger_conflict_preserves_existing_permit_and_releases_rejected_one() {
        let store = StudioStore::new();
        let job_a = register_store_job(&store, "job-a");
        let job_b = register_store_job(&store, "job-b");
        let permit_a = store
            .registry
            .try_acquire_dispatch(&job_a)
            .expect("job a permit");
        store
            .hold_dispatch_permit(job_a.clone(), permit_a)
            .expect("hold job a permit");
        let permit_b = store
            .registry
            .try_acquire_dispatch(&job_b)
            .expect("job b permit");

        let error = store
            .hold_dispatch_permit(job_a.clone(), permit_b)
            .expect_err("occupied ledger key must reject a replacement");
        assert!(matches!(error.code, IpcErrorCode::Conflict));
        let dispatcher = store
            .registry
            .dispatcher_snapshot()
            .expect("dispatcher snapshot");
        assert_eq!(dispatcher.in_use, 1);
        assert_eq!(dispatcher.active_job_ids, vec![job_a.clone()]);
        assert!(store
            .dispatch_permits
            .lock()
            .expect("dispatch ledger")
            .contains_key(&job_a));

        store
            .release_dispatch_permit(&job_a)
            .expect("release retained permit");
        assert_eq!(
            store
                .registry
                .dispatcher_snapshot()
                .expect("released dispatcher snapshot")
                .in_use,
            0
        );
    }

    #[test]
    fn worker_acceptance_history_survives_later_failure() {
        let store = StudioStore::new();
        let job_id = register_store_job(&store, "job-accepted-history");
        let registered = store
            .registry
            .snapshot(&job_id)
            .expect("registered snapshot");
        assert!(!registered.runtime_state.worker_accepted);
        assert_eq!(registered.status, RegistryJobStatus::Registered);

        transition_job_state(&store, &job_id, RegistryJobStatus::Queued, |runtime| {
            commit_accepted_job_in_state(runtime, job_id.as_str())
        })
        .expect("commit worker acceptance");
        let accepted = store.registry.snapshot(&job_id).expect("accepted snapshot");
        assert!(accepted.runtime_state.worker_accepted);
        assert_eq!(accepted.status, RegistryJobStatus::Queued);

        transition_job_state(&store, &job_id, RegistryJobStatus::Failed, |runtime| {
            commit_failed_job_in_state(runtime, job_id.as_str(), "controlled later failure")
        })
        .expect("commit later failure");
        let failed = store.registry.snapshot(&job_id).expect("failed snapshot");
        assert!(failed.runtime_state.worker_accepted);
        assert_eq!(failed.status, RegistryJobStatus::Failed);
        assert_eq!(failed.runtime_state.snapshot.job.status, JobStatus::Failed);
    }

    #[test]
    fn validates_job_start_acceptance_strictly() {
        let valid = worker_response(ResponseKind::Accepted, "job-1", "queued");
        assert!(validate_start_accepted(&valid, "job-1").is_ok());

        let wrong_kind = worker_response(ResponseKind::Completed, "job-1", "queued");
        assert!(validate_start_accepted(&wrong_kind, "job-1").is_err());

        let wrong_job = worker_response(ResponseKind::Accepted, "job-2", "queued");
        assert!(validate_start_accepted(&wrong_job, "job-1").is_err());

        let wrong_status = worker_response(ResponseKind::Accepted, "job-1", "running");
        assert!(validate_start_accepted(&wrong_status, "job-1").is_err());

        let mut extra = worker_response(ResponseKind::Accepted, "job-1", "queued");
        extra
            .payload
            .insert("unexpected".to_owned(), Value::Bool(true));
        assert!(validate_start_accepted(&extra, "job-1").is_err());
    }

    #[test]
    fn cancellation_commits_only_after_explicit_cancelled_status() {
        let root = temp_workspace("cancel-state");
        let mut state = state_with_speakers(1);
        commit_test_job(&mut state, valid_manual_request(&root, 2), "job-cancel")
            .expect("commit queued job");
        assert!(matches!(state.snapshot.job.status, JobStatus::Queued));

        let running = worker_response(ResponseKind::Accepted, "job-cancel", "running");
        let status =
            validate_cancel_accepted(&running, "job-cancel").expect("valid running response");
        assert_ne!(status, "cancelled");
        assert!(matches!(state.snapshot.job.status, JobStatus::Queued));

        let cancelled = worker_response(ResponseKind::Accepted, "job-cancel", "cancelled");
        let status =
            validate_cancel_accepted(&cancelled, "job-cancel").expect("valid cancelled response");
        assert_eq!(status, "cancelled");
        commit_cancelled_job_in_state(&mut state, "job-cancel").expect("commit cancellation");
        assert!(matches!(state.snapshot.job.status, JobStatus::Cancelled));

        assert!(commit_cancelled_job_in_state(&mut state, "job-stale").is_err());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn rejects_malformed_cancel_acceptance_without_mutating_state() {
        let mut state = state_with_speakers(1);
        state.snapshot.job.id = "job-current".to_owned();
        state.snapshot.job.status = JobStatus::Queued;

        let wrong_kind = worker_response(ResponseKind::Completed, "job-current", "cancelled");
        assert!(validate_cancel_accepted(&wrong_kind, "job-current").is_err());

        let wrong_job = worker_response(ResponseKind::Accepted, "job-other", "cancelled");
        assert!(validate_cancel_accepted(&wrong_job, "job-current").is_err());

        let unknown_status = worker_response(ResponseKind::Accepted, "job-current", "unknown");
        assert!(validate_cancel_accepted(&unknown_status, "job-current").is_err());

        let mut extra = worker_response(ResponseKind::Accepted, "job-current", "cancelled");
        extra
            .payload
            .insert("unexpected".to_owned(), Value::Bool(true));
        assert!(validate_cancel_accepted(&extra, "job-current").is_err());
        assert!(matches!(state.snapshot.job.status, JobStatus::Queued));
    }

    #[test]
    fn materializes_large_dynamic_speaker_lists_without_a_product_cap() {
        assert!(SpeakerCountPolicy::Manual {
            count: JS_MAX_SAFE_INTEGER,
        }
        .validate()
        .is_ok());

        for count in [21, 64] {
            let speakers =
                create_speaker_profiles(count, &[]).expect("dynamic role list must materialize");
            assert_eq!(speakers.len(), count);
            assert_eq!(
                speakers.last().expect("last speaker").id,
                format!("speaker-{count}")
            );
            assert!(speakers
                .iter()
                .enumerate()
                .all(|(index, speaker)| speaker.id == format!("speaker-{}", index + 1)));
        }
    }

    #[test]
    fn generates_dynamic_non_repeating_palette_for_large_role_lists() {
        let profiles = create_speaker_profiles(13, &labels(13)).expect("dynamic palette");
        let colors = profiles
            .iter()
            .map(|speaker| speaker.color.as_str())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(colors.len(), 13);
        assert!(profiles
            .iter()
            .all(|speaker| speaker.color.starts_with("hsl(")));
    }

    #[test]
    fn validates_speaker_ids_strictly() {
        assert!(validate_speaker_id("speaker-1").is_ok());
        assert!(validate_speaker_id("speaker-17").is_ok());
        for invalid in [
            "speaker-0",
            "speaker-01",
            "speaker-１",
            "speaker-9007199254740992",
            "speaker-1-extra",
        ] {
            assert!(
                validate_speaker_id(invalid).is_err(),
                "{invalid} must be rejected"
            );
        }
    }

    #[test]
    fn prepares_exact_review_submit_payload_with_human_audit_and_job_binding() {
        let mut state = review_active_state(2, "job-review-payload");
        state.snapshot.reviews.push(pending_review("review-1"));
        state.snapshot.job.review_open_count = 1;
        let mut decision = valid_review_decision("review-1", "speaker-2");
        decision.normalized_text = "  人工确认后的中文原文。  ".to_owned();
        decision.reason = "  依据人工复听确认角色归属。  ".to_owned();
        decision.evidence = "  对比前后片段声纹及问答关系。  ".to_owned();

        let prepared =
            prepare_review_mutation(&state, decision.clone()).expect("prepare review.submit");
        let second =
            prepare_review_mutation(&state, decision).expect("prepare unique review.submit");

        exact_object_keys(
            &prepared.payload,
            &[
                "jobId",
                "itemId",
                "action",
                "decisionId",
                "reason",
                "evidence",
                "confidence",
                "audit",
                "targetSpeakerId",
                "normalizedText",
                "displayText",
                "rawText",
            ],
            "review.submit test payload",
        )
        .expect("exact review.submit fields");
        assert_eq!(prepared.job_id, "job-review-payload");
        assert_eq!(prepared.payload["jobId"], "job-review-payload");
        assert_eq!(prepared.payload["itemId"], "review-1");
        assert_eq!(prepared.payload["action"], "accept");
        assert_eq!(prepared.payload["targetSpeakerId"], "speaker-2");
        assert_eq!(prepared.payload["normalizedText"], "人工确认后的中文原文。");
        assert_eq!(prepared.payload["displayText"], "人工确认后的中文原文。");
        assert_eq!(prepared.payload["rawText"], "原始中文文本");
        assert_eq!(prepared.payload["reason"], "依据人工复听确认角色归属。");
        assert_eq!(
            prepared.payload["evidence"],
            serde_json::json!(["对比前后片段声纹及问答关系。"])
        );
        assert_eq!(
            prepared.payload["decisionId"],
            prepared.envelope.decision_id
        );
        assert_ne!(
            prepared.envelope.decision_id, second.envelope.decision_id,
            "human mutation decision IDs must never be reused"
        );
        let audit = prepared.payload["audit"]
            .as_object()
            .expect("human audit object");
        exact_object_keys(
            audit,
            &[
                "actor",
                "source",
                "client",
                "jobId",
                "reviewItemId",
                "targetSpeakerId",
                "submittedAtUnixMs",
            ],
            "review.submit human audit",
        )
        .expect("exact human audit fields");
        assert_eq!(audit["actor"], "desktop-user");
        assert_eq!(audit["source"], "human");
        assert_eq!(audit["jobId"], "job-review-payload");
        assert_eq!(audit["reviewItemId"], "review-1");
        assert_eq!(audit["targetSpeakerId"], "speaker-2");
    }

    #[test]
    fn rejects_malformed_review_completion_without_local_mutation() {
        let mut state = review_active_state(2, "job-review-response");
        state.snapshot.reviews.push(pending_review("review-1"));
        state.snapshot.job.review_open_count = 1;
        let prepared =
            prepare_review_mutation(&state, valid_review_decision("review-1", "speaker-2"))
                .expect("prepare review");
        let valid = completed_mutation_response(
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            0,
            prepared.speaker_count,
        );
        validate_mutation_completed(
            &valid,
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            prepared.speaker_count,
        )
        .expect("valid completed response");
        let before = serde_json::to_value(&state.snapshot).expect("snapshot before validation");

        let mut wrong_kind = valid.clone();
        wrong_kind.kind = ResponseKind::Accepted;
        let mut wrong_job = valid.clone();
        wrong_job
            .payload
            .insert("jobId".to_owned(), Value::String("job-other".to_owned()));
        let mut wrong_command = valid.clone();
        wrong_command.payload.insert(
            "command".to_owned(),
            Value::String("speaker.rename".to_owned()),
        );
        let mut malformed_hash = valid.clone();
        malformed_hash
            .payload
            .insert("documentHash".to_owned(), Value::String("ABC".to_owned()));
        let mut mismatched_decision = valid.clone();
        mismatched_decision
            .payload
            .get_mut("decision")
            .and_then(Value::as_object_mut)
            .expect("decision object")
            .insert(
                "reason".to_owned(),
                Value::String("different reason".to_owned()),
            );
        let mut extra_field = valid;
        extra_field
            .payload
            .insert("unexpected".to_owned(), Value::Bool(true));

        for invalid in [
            wrong_kind,
            wrong_job,
            wrong_command,
            malformed_hash,
            mismatched_decision,
            extra_field,
        ] {
            assert!(
                validate_mutation_completed(
                    &invalid,
                    &prepared.job_id,
                    "review.submit",
                    &prepared.envelope,
                    prepared.speaker_count,
                )
                .is_err(),
                "malformed completion must fail closed"
            );
        }
        assert_eq!(
            serde_json::to_value(&state.snapshot).expect("snapshot after validation"),
            before
        );
    }

    #[test]
    fn review_queue_proof_precedes_commit_and_records_append_only_human_audit() {
        let root = temp_workspace("review-durable-commit");
        let mut state = review_active_state(2, "job-review-commit");
        state.snapshot.reviews.push(pending_review("review-1"));
        state.snapshot.job.review_open_count = 1;
        let prepared =
            prepare_review_mutation(&state, valid_review_decision("review-1", "speaker-2"))
                .expect("prepare review");
        let speaker_labels = state
            .snapshot
            .speakers
            .iter()
            .map(|speaker| speaker.label.clone())
            .collect::<Vec<_>>();
        let transcript = mutation_transcript_document(
            &prepared.job_id,
            &speaker_labels,
            &prepared.speaker_id,
            &prepared.raw_text,
            &prepared.normalized_text,
            true,
        );
        let transcript_hash = canonical_json_sha256(&transcript).expect("review transcript hash");
        let response = completed_mutation_response_with_hash(
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            0,
            prepared.speaker_count,
            &transcript_hash,
        );
        let receipt = validate_mutation_completed(
            &response,
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            prepared.speaker_count,
        )
        .expect("validate completed response");
        let queue_response = review_submission_queue_response(&prepared, &receipt);
        let queue = validate_review_queue_snapshot(&queue_response, &prepared.job_id, &receipt)
            .expect("validate review.queue schema and receipt binding");
        validate_review_submission_proof(&queue, &prepared, &receipt)
            .expect("prove durable review mutation");
        install_mutation_artifacts(
            &mut state,
            &root,
            &transcript,
            &Value::Object(queue.clone()),
        );
        let reconciled = reconcile_mutation_files(&state, &prepared.job_id, &receipt, &queue)
            .expect("reconcile durable transcript and review queue");
        assert!(!state.snapshot.reviews[0].reviewed);
        assert!(!state.snapshot.reviews[0].locked);
        assert!(state.snapshot.reviews[0].audit_trail.is_empty());

        let reviewed = commit_review_mutation_in_state(&mut state, &prepared, &receipt, reconciled)
            .expect("reconcile verified review");
        assert_eq!(reviewed.current_speaker_id, "speaker-2");
        assert_eq!(reviewed.normalized_text, "人工确认后的中文原文。");
        assert_eq!(reviewed.raw_text, "原始中文文本");
        assert!(reviewed.reviewed);
        assert!(reviewed.locked);
        assert_eq!(reviewed.audit_trail.len(), 1);
        let event = &reviewed.audit_trail[0];
        assert_eq!(event.id, prepared.envelope.decision_id);
        assert_eq!(event.sequence, 1);
        assert!(matches!(event.actor, ReviewActor::Human));
        assert_eq!(event.reason, "依据人工复听确认角色归属。");
        assert_eq!(event.evidence, "对比前后片段声纹及问答关系。");
        assert!((event.confidence - 0.95).abs() < f32::EPSILON);
        assert_eq!(event.previous_speaker_id, "speaker-1");
        assert_eq!(event.speaker_id, "speaker-2");
        assert_eq!(event.previous_normalized_text, "原始中文文本");
        assert_eq!(event.normalized_text, "人工确认后的中文原文。");
        assert_eq!(state.snapshot.job.review_open_count, receipt.open_count);

        let before_retry =
            serde_json::to_value(&state.snapshot).expect("snapshot before duplicate review");
        let error = prepare_review_mutation(&state, valid_review_decision("review-1", "speaker-2"))
            .expect_err("duplicate review must fail before worker submission");
        assert!(matches!(error.code, IpcErrorCode::Conflict));
        assert_eq!(
            serde_json::to_value(&state.snapshot).expect("snapshot after duplicate review"),
            before_retry
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn invalid_review_queue_proof_rolls_back_local_state() {
        let mut state = review_active_state(2, "job-review-proof");
        state.snapshot.reviews.push(pending_review("review-1"));
        state.snapshot.job.review_open_count = 1;
        let prepared =
            prepare_review_mutation(&state, valid_review_decision("review-1", "speaker-2"))
                .expect("prepare review");
        let response = completed_mutation_response(
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            0,
            prepared.speaker_count,
        );
        let receipt = validate_mutation_completed(
            &response,
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            prepared.speaker_count,
        )
        .expect("validate completed response");
        let before = serde_json::to_value(&state.snapshot).expect("snapshot before bad proof");

        let mut altered_text = review_submission_queue_response(&prepared, &receipt);
        altered_text
            .payload
            .get_mut("queue")
            .and_then(Value::as_object_mut)
            .and_then(|queue| queue.get_mut("items"))
            .and_then(Value::as_array_mut)
            .and_then(|items| items.first_mut())
            .and_then(Value::as_object_mut)
            .and_then(|item| item.get_mut("text"))
            .and_then(Value::as_object_mut)
            .expect("persisted text object")
            .insert(
                "rawText".to_owned(),
                Value::String("tampered raw text".to_owned()),
            );
        let queue = validate_review_queue_snapshot(&altered_text, &prepared.job_id, &receipt)
            .expect("queue remains structurally valid");
        assert!(
            validate_review_submission_proof(&queue, &prepared, &receipt).is_err(),
            "altered durable proof must fail"
        );

        let mut wrong_kind = review_submission_queue_response(&prepared, &receipt);
        wrong_kind.kind = ResponseKind::Accepted;
        assert!(
            validate_review_queue_snapshot(&wrong_kind, &prepared.job_id, &receipt).is_err(),
            "wrong review.queue response kind must fail"
        );
        assert_eq!(
            serde_json::to_value(&state.snapshot).expect("snapshot after bad proof"),
            before,
            "desktop state must not mutate when durable proof fails"
        );
    }

    #[test]
    fn review_mutation_is_bound_to_committed_job_through_reconciliation() {
        let root = temp_workspace("review-job-binding");
        let mut state = review_active_state(2, "job-review-bound");
        state.snapshot.reviews.push(pending_review("review-1"));
        state.snapshot.job.review_open_count = 1;
        let prepared =
            prepare_review_mutation(&state, valid_review_decision("review-1", "speaker-2"))
                .expect("prepare review");
        assert_eq!(prepared.job_id, state.snapshot.job.id);

        let wrong_job = completed_mutation_response(
            "job-review-stale",
            "review.submit",
            &prepared.envelope,
            0,
            prepared.speaker_count,
        );
        assert!(
            validate_mutation_completed(
                &wrong_job,
                &prepared.job_id,
                "review.submit",
                &prepared.envelope,
                prepared.speaker_count,
            )
            .is_err(),
            "worker completion for another job must fail"
        );

        let speaker_labels = state
            .snapshot
            .speakers
            .iter()
            .map(|speaker| speaker.label.clone())
            .collect::<Vec<_>>();
        let transcript = mutation_transcript_document(
            &prepared.job_id,
            &speaker_labels,
            &prepared.speaker_id,
            &prepared.raw_text,
            &prepared.normalized_text,
            true,
        );
        let transcript_hash = canonical_json_sha256(&transcript).expect("bound transcript hash");
        let response = completed_mutation_response_with_hash(
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            0,
            prepared.speaker_count,
            &transcript_hash,
        );
        let receipt = validate_mutation_completed(
            &response,
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            prepared.speaker_count,
        )
        .expect("valid job-bound receipt");
        let queue_response = review_submission_queue_response(&prepared, &receipt);
        let queue = validate_review_queue_snapshot(&queue_response, &prepared.job_id, &receipt)
            .expect("valid job-bound queue");
        validate_review_submission_proof(&queue, &prepared, &receipt)
            .expect("valid job-bound durable proof");
        install_mutation_artifacts(
            &mut state,
            &root,
            &transcript,
            &Value::Object(queue.clone()),
        );
        let reconciled = reconcile_mutation_files(&state, &prepared.job_id, &receipt, &queue)
            .expect("prepare valid job-bound reconciliation");

        state.snapshot.job.id = "job-replaced-before-commit".to_owned();
        let before_commit =
            serde_json::to_value(&state.snapshot).expect("snapshot before stale commit");
        let error = commit_review_mutation_in_state(&mut state, &prepared, &receipt, reconciled)
            .expect_err("stale job reconciliation must fail");
        assert!(matches!(error.code, IpcErrorCode::NotFound));
        assert_eq!(
            serde_json::to_value(&state.snapshot).expect("snapshot after stale commit"),
            before_commit
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn prepares_and_commits_verified_speaker_rename() {
        let root = temp_workspace("speaker-rename-durable-commit");
        let mut state = review_active_state(3, "job-speaker-rename");
        let prepared = prepare_speaker_rename(
            &state,
            UpdateSpeakerRequest {
                speaker_id: "speaker-2".to_owned(),
                label: "产品负责人".to_owned(),
                locked: false,
                review_status: SpeakerReviewStatus::Pending,
            },
        )
        .expect("prepare speaker.rename");
        exact_object_keys(
            &prepared.payload,
            &[
                "jobId",
                "speakerId",
                "name",
                "decisionId",
                "reason",
                "evidence",
                "confidence",
                "audit",
            ],
            "speaker.rename test payload",
        )
        .expect("exact speaker.rename fields");
        assert_eq!(prepared.payload["jobId"], "job-speaker-rename");
        assert_eq!(prepared.payload["speakerId"], "speaker-2");
        assert_eq!(prepared.payload["name"], "产品负责人");
        assert_eq!(
            prepared.payload["decisionId"],
            prepared.envelope.decision_id
        );
        assert_eq!(prepared.payload["confidence"], 1.0);
        let audit = prepared.payload["audit"]
            .as_object()
            .expect("speaker rename audit");
        assert_eq!(audit["actor"], "desktop-user");
        assert_eq!(audit["source"], "human");
        assert_eq!(audit["jobId"], "job-speaker-rename");
        assert_eq!(audit["speakerId"], "speaker-2");
        assert_eq!(audit["previousName"], "Speaker 2");
        assert_eq!(audit["newName"], "产品负责人");

        let wrong_job = completed_mutation_response(
            "job-speaker-stale",
            "speaker.rename",
            &prepared.envelope,
            0,
            prepared.speaker_count,
        );
        assert!(
            validate_mutation_completed(
                &wrong_job,
                &prepared.job_id,
                "speaker.rename",
                &prepared.envelope,
                prepared.speaker_count,
            )
            .is_err(),
            "speaker.rename completion must remain bound to the committed job"
        );
        let mut speaker_labels = state
            .snapshot
            .speakers
            .iter()
            .map(|speaker| speaker.label.clone())
            .collect::<Vec<_>>();
        speaker_labels[1].clone_from(&prepared.next_label);
        let transcript = mutation_transcript_document(
            &prepared.job_id,
            &speaker_labels,
            "speaker-1",
            "原始中文文本",
            "原始中文文本",
            false,
        );
        let transcript_hash =
            canonical_json_sha256(&transcript).expect("speaker rename transcript hash");
        let response = completed_mutation_response_with_hash(
            &prepared.job_id,
            "speaker.rename",
            &prepared.envelope,
            0,
            prepared.speaker_count,
            &transcript_hash,
        );
        let receipt = validate_mutation_completed(
            &response,
            &prepared.job_id,
            "speaker.rename",
            &prepared.envelope,
            prepared.speaker_count,
        )
        .expect("validate speaker.rename completion");
        let queue_response = speaker_rename_queue_response(&prepared, &receipt);
        let queue = validate_review_queue_snapshot(&queue_response, &prepared.job_id, &receipt)
            .expect("validate speaker rename queue");
        validate_speaker_rename_proof(&queue, &prepared, &receipt)
            .expect("prove persisted speaker rename");
        install_mutation_artifacts(
            &mut state,
            &root,
            &transcript,
            &Value::Object(queue.clone()),
        );
        let reconciled = reconcile_mutation_files(&state, &prepared.job_id, &receipt, &queue)
            .expect("reconcile durable speaker rename");
        assert_eq!(state.snapshot.speakers[1].label, "Speaker 2");

        let renamed = commit_speaker_rename_in_state(&mut state, &prepared, &receipt, reconciled)
            .expect("reconcile verified speaker rename");
        assert_eq!(renamed.id, "speaker-2");
        assert_eq!(renamed.label, "产品负责人");
        assert!(!renamed.locked);
        assert_eq!(renamed.review_status, SpeakerReviewStatus::Pending);
        assert_eq!(state.snapshot.job.review_open_count, receipt.open_count);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn invalid_speaker_rename_proof_rolls_back_local_state() {
        let state = review_active_state(2, "job-speaker-proof");
        let prepared = prepare_speaker_rename(
            &state,
            UpdateSpeakerRequest {
                speaker_id: "speaker-2".to_owned(),
                label: "主持人".to_owned(),
                locked: false,
                review_status: SpeakerReviewStatus::Pending,
            },
        )
        .expect("prepare speaker rename");
        let response = completed_mutation_response(
            &prepared.job_id,
            "speaker.rename",
            &prepared.envelope,
            0,
            prepared.speaker_count,
        );
        let receipt = validate_mutation_completed(
            &response,
            &prepared.job_id,
            "speaker.rename",
            &prepared.envelope,
            prepared.speaker_count,
        )
        .expect("validate speaker rename completion");
        let before = serde_json::to_value(&state.snapshot).expect("snapshot before bad proof");
        let mut queue_response = speaker_rename_queue_response(&prepared, &receipt);
        queue_response
            .payload
            .get_mut("queue")
            .and_then(Value::as_object_mut)
            .and_then(|queue| queue.get_mut("decisions"))
            .and_then(Value::as_array_mut)
            .and_then(|decisions| decisions.first_mut())
            .and_then(Value::as_object_mut)
            .and_then(|decision| decision.get_mut("after"))
            .and_then(Value::as_object_mut)
            .expect("rename after object")
            .insert(
                "name".to_owned(),
                Value::String("different persisted name".to_owned()),
            );
        let queue = validate_review_queue_snapshot(&queue_response, &prepared.job_id, &receipt)
            .expect("queue remains structurally valid");
        assert!(
            validate_speaker_rename_proof(&queue, &prepared, &receipt).is_err(),
            "mismatched persisted name must fail closed"
        );
        assert_eq!(
            serde_json::to_value(&state.snapshot).expect("snapshot after bad proof"),
            before
        );
    }

    #[test]
    fn rejects_unsupported_or_conflicting_speaker_updates_without_mutation() {
        let state = review_active_state(3, "job-speaker-reject");
        let before = serde_json::to_value(&state.snapshot).expect("speaker snapshot before");
        let unsupported_lock = prepare_speaker_rename(
            &state,
            UpdateSpeakerRequest {
                speaker_id: "speaker-2".to_owned(),
                label: "产品负责人".to_owned(),
                locked: true,
                review_status: SpeakerReviewStatus::Pending,
            },
        )
        .expect_err("unsupported lock mutation must fail");
        assert!(matches!(
            unsupported_lock.code,
            IpcErrorCode::InvalidRequest
        ));
        let unsupported_status = prepare_speaker_rename(
            &state,
            UpdateSpeakerRequest {
                speaker_id: "speaker-2".to_owned(),
                label: "产品负责人".to_owned(),
                locked: false,
                review_status: SpeakerReviewStatus::Confirmed,
            },
        )
        .expect_err("unsupported review status mutation must fail");
        assert!(matches!(
            unsupported_status.code,
            IpcErrorCode::InvalidRequest
        ));
        let unchanged = prepare_speaker_rename(
            &state,
            UpdateSpeakerRequest {
                speaker_id: "speaker-2".to_owned(),
                label: "Speaker 2".to_owned(),
                locked: false,
                review_status: SpeakerReviewStatus::Pending,
            },
        )
        .expect_err("unchanged name has no worker-backed mutation");
        assert!(matches!(unchanged.code, IpcErrorCode::InvalidRequest));
        let duplicate = prepare_speaker_rename(
            &state,
            UpdateSpeakerRequest {
                speaker_id: "speaker-2".to_owned(),
                label: "Speaker 1".to_owned(),
                locked: false,
                review_status: SpeakerReviewStatus::Pending,
            },
        )
        .expect_err("duplicate name must fail");
        assert!(matches!(duplicate.code, IpcErrorCode::Conflict));
        let missing = prepare_speaker_rename(
            &state,
            UpdateSpeakerRequest {
                speaker_id: "speaker-4".to_owned(),
                label: "Missing".to_owned(),
                locked: false,
                review_status: SpeakerReviewStatus::Pending,
            },
        )
        .expect_err("unknown speaker must fail");
        assert!(matches!(missing.code, IpcErrorCode::NotFound));
        assert_eq!(
            serde_json::to_value(&state.snapshot).expect("speaker snapshot after"),
            before
        );

        let mut locked_state = review_active_state(2, "job-speaker-locked");
        locked_state.snapshot.speakers[1].locked = true;
        let locked_before =
            serde_json::to_value(&locked_state.snapshot).expect("locked speaker before");
        let locked = prepare_speaker_rename(
            &locked_state,
            UpdateSpeakerRequest {
                speaker_id: "speaker-2".to_owned(),
                label: "静默改名".to_owned(),
                locked: true,
                review_status: SpeakerReviewStatus::Pending,
            },
        )
        .expect_err("human-locked speaker cannot be renamed");
        assert!(matches!(locked.code, IpcErrorCode::Conflict));
        assert_eq!(
            serde_json::to_value(&locked_state.snapshot).expect("locked speaker after"),
            locked_before
        );
    }

    #[test]
    fn rejects_invalid_review_inputs_before_worker_submission() {
        let mut unknown_speaker = review_active_state(2, "job-review-unknown");
        unknown_speaker
            .snapshot
            .reviews
            .push(pending_review("review-1"));
        let before =
            serde_json::to_value(&unknown_speaker.snapshot).expect("unknown speaker before");
        let error = prepare_review_mutation(
            &unknown_speaker,
            valid_review_decision("review-1", "speaker-3"),
        )
        .expect_err("unknown speaker must fail");
        assert!(matches!(error.code, IpcErrorCode::NotFound));
        assert_eq!(
            serde_json::to_value(&unknown_speaker.snapshot).expect("unknown speaker after"),
            before
        );

        for (locked, reviewed) in [(true, false), (false, true)] {
            let mut state = review_active_state(2, "job-review-resolved");
            let mut review = pending_review("review-1");
            review.locked = locked;
            review.reviewed = reviewed;
            state.snapshot.reviews.push(review);
            let before =
                serde_json::to_value(&state.snapshot).expect("resolved review snapshot before");
            let error =
                prepare_review_mutation(&state, valid_review_decision("review-1", "speaker-2"))
                    .expect_err("resolved review must reject overwrite");
            assert!(matches!(error.code, IpcErrorCode::Conflict));
            assert_eq!(
                serde_json::to_value(&state.snapshot).expect("resolved review snapshot after"),
                before
            );
        }

        let mut locked_target = review_active_state(2, "job-review-locked-target");
        locked_target.snapshot.speakers[1].locked = true;
        locked_target
            .snapshot
            .reviews
            .push(pending_review("review-1"));
        let before = serde_json::to_value(&locked_target.snapshot).expect("locked target before");
        let error = prepare_review_mutation(
            &locked_target,
            valid_review_decision("review-1", "speaker-2"),
        )
        .expect_err("locked target reassignment must fail");
        assert!(matches!(error.code, IpcErrorCode::Conflict));
        assert_eq!(
            serde_json::to_value(&locked_target.snapshot).expect("locked target after"),
            before
        );

        for confidence in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
            let mut state = review_active_state(2, "job-review-confidence");
            state.snapshot.reviews.push(pending_review("review-1"));
            let mut decision = valid_review_decision("review-1", "speaker-2");
            decision.confidence = confidence;
            let before = serde_json::to_value(&state.snapshot).expect("confidence snapshot before");
            let error = prepare_review_mutation(&state, decision)
                .expect_err("invalid confidence must fail closed");
            assert!(matches!(error.code, IpcErrorCode::InvalidRequest));
            assert_eq!(
                serde_json::to_value(&state.snapshot).expect("confidence snapshot after"),
                before
            );
        }
    }

    #[test]
    fn review_decision_serde_accepts_ts_fixture_and_denies_unknown_fields() {
        let decision: ReviewDecision = serde_json::from_str(include_str!(
            "../../src/contracts/fixtures/ts-review-decision.json"
        ))
        .expect("TypeScript ReviewDecision fixture must deserialize in Rust");
        assert_eq!(decision.review_id, "review-contract-1");
        assert_eq!(decision.speaker_id, "speaker-2");
        assert_eq!(decision.normalized_text, "人工确认后的中文原文。");
        assert_eq!(decision.reason, "依据人工复听确认角色归属。");
        assert_eq!(decision.evidence, "对比前后片段声纹及问答关系。");
        assert!((decision.confidence - 0.95).abs() < f32::EPSILON);

        for invalid in [
            r#"{"reviewId":"review-1","speakerId":"speaker-1","normalizedText":"文本","reason":"原因","evidence":"证据","confidence":0.9,"note":"legacy"}"#,
            r#"{"reviewId":"review-1","speakerId":"speaker-1","normalizedText":"文本","reason":"原因","evidence":"证据","confidence":0.9,"modelConfidence":0.9}"#,
        ] {
            assert!(
                serde_json::from_str::<ReviewDecision>(invalid).is_err(),
                "deny_unknown_fields must reject retired or unknown fields"
            );
        }
    }

    #[test]
    fn default_snapshot_matches_cross_language_fixture() {
        let actual_json = serde_json::to_string(&default_snapshot())
            .expect("default snapshot must serialize as JSON");
        let actual: serde_json::Value = serde_json::from_str(&actual_json)
            .expect("serialized default snapshot must be valid JSON");
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../../src/contracts/fixtures/rust-default-snapshot.json"
        ))
        .expect("Rust default snapshot fixture must be valid JSON");

        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_artifact_traversal_and_unknown_id() {
        let root = temp_workspace("artifact-boundary");
        for invalid in [
            "../secret.txt",
            "./report.pdf",
            "/absolute/report.pdf",
            "C:\\absolute\\report.pdf",
            "nested//report.pdf",
        ] {
            assert!(
                matches!(
                    safe_relative_path(invalid)
                        .expect_err("unsafe artifact path must fail")
                        .code,
                    IpcErrorCode::PathBoundaryViolation
                ),
                "{invalid} must be rejected"
            );
        }

        let state = AppState {
            snapshot: default_snapshot(),
            output_root: Some(root.clone()),
            worker_evidence: WorkerEvidenceLedger::default(),
            request_fingerprint: "test-request".to_owned(),
            worker_accepted: false,
        };
        assert!(matches!(
            open_artifact_in_state(&state, "missing")
                .expect_err("unknown artifact must fail")
                .code,
            IpcErrorCode::NotFound
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn accepts_canonical_media_output_and_in_root_artifact() {
        let root = temp_workspace("valid-paths");
        let request = valid_manual_request(&root, 3);
        let prepared = prepare_job(request.clone()).expect("valid request");
        assert!(prepared.media_path.is_absolute());
        assert!(prepared.output_directory.is_absolute());

        let artifact_path = prepared.output_directory.join("report.pdf");
        File::create(&artifact_path).expect("create artifact");
        let mut snapshot = default_snapshot();
        snapshot.artifacts.push(ArtifactItem {
            id: "artifact-pdf".to_owned(),
            name: "report.pdf".to_owned(),
            kind: ArtifactKind::Pdf,
            relative_path: "report.pdf".to_owned(),
            size_label: "0 B".to_owned(),
            created_at: "测试".to_owned(),
            integrity: IntegrityStatus::Pending,
            sha256: None,
        });
        let state = AppState {
            snapshot,
            output_root: Some(prepared.output_directory),
            worker_evidence: WorkerEvidenceLedger::default(),
            request_fingerprint: "test-request".to_owned(),
            worker_accepted: false,
        };
        let result = open_artifact_in_state(&state, "artifact-pdf").expect("safe artifact");
        assert_eq!(
            PathBuf::from(result.canonical_path),
            fs::canonicalize(artifact_path).expect("canonical artifact")
        );
        assert!(!result.opened);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn canonical_json_matches_python_contract_and_rejects_duplicate_keys() {
        let input = r#"{"z":10000000000000000,"unicode":"中文·","b":0.0001,"a":1e-7}"#.as_bytes();
        let (value, canonical) =
            strict_json_document(input, "canonical JSON fixture").expect("strict canonical JSON");
        assert_eq!(
            String::from_utf8(canonical).expect("canonical UTF-8"),
            "{\"a\":1e-07,\"b\":0.0001,\"unicode\":\"中文·\",\"z\":10000000000000000}\n"
        );
        assert_eq!(
            canonical_json_sha256(&value).expect("canonical fixture hash"),
            "a6a0801bb65ed76009b1afb4a59af633a6ae192a1f745a3d6d0836cd6cf2daef"
        );
        assert!(
            strict_json_document(br#"{"a":1,"a":2}"#, "duplicate-key fixture").is_err(),
            "duplicate JSON object keys must fail closed"
        );
        assert!(
            strict_json_document(br#"{"value":NaN}"#, "non-finite fixture").is_err(),
            "non-finite JSON numbers must fail closed"
        );
    }

    #[test]
    fn artifact_types_and_sha256_are_verified_against_canonical_or_exact_bytes() {
        assert!(matches!(
            artifact_kind_for_worker_type("pdf-quality-report-v1", Path::new("quality.json"))
                .expect("quality mapping"),
            ArtifactKind::QualityReport
        ));
        for artifact_type in ["pdf-repair-queue-v1", "review-queue-v2"] {
            assert!(matches!(
                artifact_kind_for_worker_type(artifact_type, Path::new("queue.json"))
                    .expect("repair queue mapping"),
                ArtifactKind::RepairQueue
            ));
        }
        assert!(matches!(
            artifact_kind_for_worker_type("pdf-contact-sheet", Path::new("contact.png"))
                .expect("contact sheet mapping"),
            ArtifactKind::ContactSheet
        ));
        assert!(matches!(
            artifact_kind_for_worker_type("pdf-page-evidence", Path::new("page.webp"))
                .expect("page image mapping"),
            ArtifactKind::PageImage
        ));
        assert!(matches!(
            artifact_kind_for_worker_type("pdf", Path::new("report.pdf")).expect("PDF mapping"),
            ArtifactKind::Pdf
        ));
        for artifact_type in [
            "pdf-report-document-v1",
            "business-manifest-v1",
            "business-variant-v1",
        ] {
            assert!(matches!(
                artifact_kind_for_worker_type(artifact_type, Path::new("business.json"))
                    .expect("business/report JSON mapping"),
                ArtifactKind::TranscriptJson
            ));
        }
        assert!(
            artifact_kind_for_worker_type("unknown-artifact", Path::new("unknown.json")).is_err()
        );
        assert!(artifact_kind_for_worker_type("pdf", Path::new("report.json")).is_err());

        let root = temp_workspace("artifact-hashes");
        let json_path = root.join("business.json");
        let raw_json = r#"{"b":2,"a":"中文"}"#.as_bytes();
        fs::write(&json_path, raw_json).expect("write non-canonical JSON");
        let json_value: Value = serde_json::from_slice(raw_json).expect("parse JSON fixture");
        let canonical_hash =
            canonical_json_sha256(&json_value).expect("canonical JSON artifact hash");
        let raw_hash = format!("{:x}", Sha256::digest(raw_json));
        artifact_content("business-manifest-v1", &json_path, &canonical_hash)
            .expect("canonical JSON hash must be accepted");
        assert!(
            artifact_content("business-manifest-v1", &json_path, &raw_hash).is_err(),
            "raw-byte JSON hash must not replace canonical JSON hashing"
        );
        assert!(
            artifact_content(
                "business-manifest-v1",
                &json_path,
                &canonical_hash.to_ascii_uppercase()
            )
            .is_err(),
            "uppercase SHA-256 must fail closed"
        );

        for (artifact_type, file_name, bytes) in [
            ("pdf", "report.pdf", b"%PDF-1.7\nverified\n".as_slice()),
            (
                "pdf-contact-sheet",
                "contact.png",
                b"\x89PNG\r\n\x1a\nverified".as_slice(),
            ),
            (
                "pdf-canonical-xhtml",
                "report.xhtml",
                b"<html>verified</html>\n".as_slice(),
            ),
            (
                "pdf-render-artifact",
                "transcript.txt",
                "精确字节".as_bytes(),
            ),
        ] {
            let path = root.join(file_name);
            fs::write(&path, bytes).expect("write byte-hash fixture");
            let digest = format!("{:x}", Sha256::digest(bytes));
            artifact_content(artifact_type, &path, &digest)
                .expect("exact non-JSON byte hash must be accepted");
            assert!(
                artifact_content(artifact_type, &path, &"0".repeat(64)).is_err(),
                "mismatched exact-byte SHA-256 must fail closed"
            );
        }

        let state = review_active_state(2, "job-artifact-rollback");
        let before = serde_json::to_value(&state.snapshot).expect("artifact snapshot before");
        let bad_event = worker_event(
            "job-artifact-rollback",
            "artifact-bad",
            "artifact.created",
            Map::from_iter([
                (
                    "artifactType".to_owned(),
                    Value::String("business-manifest-v1".to_owned()),
                ),
                (
                    "path".to_owned(),
                    Value::String(json_path.to_string_lossy().into_owned()),
                ),
                ("sha256".to_owned(), Value::String("0".repeat(64))),
            ]),
        );
        assert!(prepare_artifact_projection(
            &bad_event,
            &root,
            &state.snapshot.job.speaker_policy,
            &state.worker_evidence,
        )
        .is_err());
        assert_eq!(
            serde_json::to_value(&state.snapshot).expect("artifact snapshot after"),
            before,
            "failed artifact verification must not mutate desktop state"
        );
        assert!(state.worker_evidence.verified_artifacts.is_empty());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn review_queue_projects_job_scope_without_fabricating_segments() {
        let job_id = "job-review-projection";
        let labels = vec!["主持人".to_owned(), "嘉宾".to_owned()];
        let transcript = mutation_transcript_document(
            job_id,
            &labels,
            "speaker-1",
            "原始中文文本",
            "原始中文文本",
            false,
        );
        let projected_transcript = project_transcript_document(
            &transcript,
            job_id,
            &SpeakerCountPolicy::Manual { count: 2 },
        )
        .expect("project transcript");
        for speaker in &projected_transcript.speakers {
            assert!(speaker.role_hint.contains('·'));
            assert!(!speaker.role_hint.contains("бд"));
            assert!(!speaker.role_hint.contains('路'));
        }
        let evidence = transcript_evidence(&transcript, job_id, 2);
        let queue = projection_review_queue(job_id, 2);
        let projection =
            project_review_queue_document(&queue, job_id, &evidence).expect("project review queue");
        assert_eq!(projection.worker_open_count, 2);
        assert_eq!(projection.open_segment_item_count, 1);
        assert_eq!(projection.open_segment_ids.len(), 1);
        assert_eq!(projection.reviews.len(), 1);
        assert_eq!(projection.reviews[0].id, "review-1");
        assert!(
            projection
                .reviews
                .iter()
                .all(|review| review.id != "speaker-count-confidence"),
            "job-scope review evidence must not fabricate a ReviewSegment"
        );

        let mut extra_field = queue.clone();
        extra_field
            .as_object_mut()
            .expect("queue object")
            .insert("unexpected".to_owned(), Value::Bool(true));
        assert!(project_review_queue_document(&extra_field, job_id, &evidence).is_err());

        let mut duplicate_item = queue.clone();
        let first = duplicate_item["items"][0].clone();
        duplicate_item["items"]
            .as_array_mut()
            .expect("queue items")
            .push(first);
        duplicate_item["openCount"] = Value::from(3_u64);
        assert!(project_review_queue_document(&duplicate_item, job_id, &evidence).is_err());

        let mut wrong_count = queue.clone();
        wrong_count["openCount"] = Value::from(1_u64);
        assert!(project_review_queue_document(&wrong_count, job_id, &evidence).is_err());

        let mut wrong_job = queue;
        wrong_job["jobId"] = Value::String("job-other".to_owned());
        assert!(project_review_queue_document(&wrong_job, job_id, &evidence).is_err());
    }

    #[test]
    fn pipeline_metrics_project_reference_quality_without_fabricating_cpu() {
        let root = temp_workspace("pipeline-metrics");
        let job_id = "job-pipeline-metrics";
        let mut state = review_active_state(2, job_id);
        state.worker_evidence.transcript_duration_ms = Some(1_000);
        state.worker_evidence.transcript_segment_count = Some(1);
        state.output_root = Some(fs::canonicalize(&root).expect("canonical metrics root"));
        let document = pipeline_metrics_document(job_id, true);
        let path = root.join("pipeline-metrics.v1.json");
        fs::write(
            &path,
            canonical_json_bytes(&document).expect("canonical metrics"),
        )
        .expect("write pipeline metrics");
        let digest = canonical_json_sha256(&document).expect("metrics digest");
        let event = worker_event(
            job_id,
            "artifact-pipeline-metrics",
            "artifact.created",
            Map::from_iter([
                (
                    "artifactType".to_owned(),
                    Value::String("pipeline-metrics-v1".to_owned()),
                ),
                (
                    "path".to_owned(),
                    Value::String(path.to_string_lossy().into_owned()),
                ),
                ("sha256".to_owned(), Value::String(digest)),
            ]),
        );
        let prepared = prepare_artifact_projection(
            &event,
            &root,
            &state.snapshot.job.speaker_policy,
            &state.worker_evidence,
        )
        .expect("prepare verified metrics artifact");
        commit_artifact_projection(&mut state, prepared).expect("commit metrics projection");
        match &state.snapshot.performance {
            PerformanceMetrics::Unavailable { reason } => {
                assert!(reason.contains("peak CPU"));
            }
            PerformanceMetrics::Measured { .. } => {
                panic!("pipeline metrics without measured CPU must not fabricate performance")
            }
        }
        match &state.snapshot.diarization_quality.der {
            PercentMetric::Available { value, .. } => {
                assert!((*value - 12.5).abs() < f32::EPSILON);
            }
            PercentMetric::Unavailable { .. } => panic!("reference DER must be available"),
        }
        match &state.snapshot.diarization_quality.overlap_f1 {
            PercentMetric::Available { value, .. } => {
                assert!((*value - 87.5).abs() < f32::EPSILON);
            }
            PercentMetric::Unavailable { .. } => panic!("reference overlap F1 must be available"),
        }

        let unavailable = project_pipeline_metrics_document(
            &pipeline_metrics_document(job_id, false),
            job_id,
            &state.worker_evidence,
        )
        .expect("metrics without reference labels");
        assert!(unavailable.reference_quality.is_none());

        let mut invalid_rate = pipeline_metrics_document(job_id, true);
        invalid_rate["cache"]["hitRate"] = Value::from(0.75);
        assert!(
            project_pipeline_metrics_document(&invalid_rate, job_id, &state.worker_evidence)
                .is_err()
        );
        let wrong_job = pipeline_metrics_document("job-other", true);
        assert!(
            project_pipeline_metrics_document(&wrong_job, job_id, &state.worker_evidence).is_err()
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn durable_reconciliation_failure_rolls_back_review_mutation() {
        let root = temp_workspace("review-reconcile-rollback");
        let mut state = review_active_state(2, "job-review-reconcile-rollback");
        state.snapshot.reviews.push(pending_review("review-1"));
        state.snapshot.job.review_open_count = 1;
        let prepared =
            prepare_review_mutation(&state, valid_review_decision("review-1", "speaker-2"))
                .expect("prepare review");
        let labels = state
            .snapshot
            .speakers
            .iter()
            .map(|speaker| speaker.label.clone())
            .collect::<Vec<_>>();
        let transcript = mutation_transcript_document(
            &prepared.job_id,
            &labels,
            &prepared.speaker_id,
            &prepared.raw_text,
            &prepared.normalized_text,
            true,
        );
        let transcript_hash = canonical_json_sha256(&transcript).expect("rollback transcript hash");
        let response = completed_mutation_response_with_hash(
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            0,
            prepared.speaker_count,
            &transcript_hash,
        );
        let receipt = validate_mutation_completed(
            &response,
            &prepared.job_id,
            "review.submit",
            &prepared.envelope,
            prepared.speaker_count,
        )
        .expect("valid mutation receipt");
        let queue_response = review_submission_queue_response(&prepared, &receipt);
        let queue = validate_review_queue_snapshot(&queue_response, &prepared.job_id, &receipt)
            .expect("valid review queue response");
        let (_, _, queue_path, _) = install_mutation_artifacts(
            &mut state,
            &root,
            &transcript,
            &Value::Object(queue.clone()),
        );
        let before = serde_json::to_value(&state.snapshot).expect("snapshot before reconciliation");
        let mut durable_tamper = queue.clone();
        durable_tamper.insert(
            "updatedAt".to_owned(),
            Value::String("2026-07-22T12:02:00Z".to_owned()),
        );
        fs::write(
            queue_path,
            canonical_json_bytes(&Value::Object(durable_tamper)).expect("canonical tampered queue"),
        )
        .expect("tamper durable queue");
        assert!(
            reconcile_mutation_files(&state, &prepared.job_id, &receipt, &queue).is_err(),
            "durable queue/readback disagreement must fail closed"
        );
        assert_eq!(
            serde_json::to_value(&state.snapshot).expect("snapshot after reconciliation failure"),
            before,
            "failed durable reconciliation must not commit local review state"
        );
        assert!(!state.snapshot.reviews[0].reviewed);
        assert!(state.snapshot.reviews[0].audit_trail.is_empty());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn job_completion_requires_zero_reviews_exact_artifacts_hashes_and_clean_stages() {
        let outer = temp_workspace("completion-gates");
        let root = outer.join("output");
        fs::create_dir_all(&root).expect("create completion output root");
        let job_id = "job-completion-gates";
        let mut state = review_active_state(2, job_id);
        state.snapshot.job.status = JobStatus::Running;
        for stage in &mut state.snapshot.stages {
            stage.status = StageStatus::Completed;
            stage.progress = 100;
        }
        state.worker_evidence.worker_review_open_count = Some(0);
        state.worker_evidence.open_review_segment_item_count = 0;
        state.worker_evidence.open_review_segment_ids.clear();
        let paths = install_completion_artifacts(&mut state, &root);
        let event = completion_event(job_id, &paths);
        validate_job_completed_event(&event, &state).expect("valid completion evidence");

        state.worker_evidence.worker_review_open_count = None;
        assert!(validate_job_completed_event(&event, &state).is_err());
        state.worker_evidence.worker_review_open_count = Some(0);

        state.snapshot.job.status = JobStatus::ReviewRequired;
        assert!(validate_job_completed_event(&event, &state).is_err());
        state.snapshot.job.status = JobStatus::Running;

        state.snapshot.stages[0].status = StageStatus::Warning;
        assert!(validate_job_completed_event(&event, &state).is_err());
        state.snapshot.stages[0].status = StageStatus::Completed;
        state.snapshot.stages[0].status = StageStatus::Blocked;
        assert!(validate_job_completed_event(&event, &state).is_err());
        state.snapshot.stages[0].status = StageStatus::Completed;

        let mut duplicate = event.clone();
        duplicate
            .payload
            .get_mut("artifactPaths")
            .and_then(Value::as_array_mut)
            .expect("completion paths")
            .push(Value::String(paths[0].clone()));
        assert!(validate_job_completed_event(&duplicate, &state).is_err());

        let mut missing_path = event.clone();
        missing_path
            .payload
            .get_mut("artifactPaths")
            .and_then(Value::as_array_mut)
            .expect("completion paths")
            .pop();
        assert!(validate_job_completed_event(&missing_path, &state).is_err());

        let quality_path = state
            .worker_evidence
            .verified_artifacts
            .iter()
            .find(|(_, artifact)| artifact.artifact_type == "pdf-quality-report-v1")
            .map(|(path, _)| path.clone())
            .expect("quality artifact");
        let quality_entry = state
            .worker_evidence
            .verified_artifacts
            .remove(&quality_path)
            .expect("remove quality artifact");
        let without_quality = paths
            .iter()
            .filter(|path| Path::new(path.as_str()) != quality_path)
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            validate_job_completed_event(&completion_event(job_id, &without_quality), &state)
                .is_err(),
            "required artifact types must remain mandatory"
        );
        state
            .worker_evidence
            .verified_artifacts
            .insert(quality_path, quality_entry);

        let outside = outer.join("outside.pdf");
        fs::write(&outside, b"%PDF-1.7\noutside\n").expect("write outside artifact");
        let mut outside_event = event.clone();
        outside_event
            .payload
            .get_mut("artifactPaths")
            .and_then(Value::as_array_mut)
            .expect("completion paths")[0] = Value::String(outside.to_string_lossy().into_owned());
        assert!(validate_job_completed_event(&outside_event, &state).is_err());

        let pdf_path = state
            .worker_evidence
            .verified_artifacts
            .values()
            .find(|artifact| artifact.artifact_type == "pdf")
            .map(|artifact| artifact.canonical_path.clone())
            .expect("PDF artifact");
        let original_pdf = fs::read(&pdf_path).expect("read PDF before drift");
        fs::write(&pdf_path, b"%PDF-1.7\ndrifted\n").expect("drift PDF bytes");
        assert!(
            validate_job_completed_event(&event, &state).is_err(),
            "post-announcement artifact drift must fail completion"
        );
        fs::write(&pdf_path, original_pdf).expect("restore PDF bytes");
        validate_job_completed_event(&event, &state).expect("completion after restored evidence");
        fs::remove_dir_all(outer).expect("cleanup");
    }
}
