use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env, fmt, fs,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex as StdMutex,
    },
    time::Duration,
};
use tokio::{
    io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command},
    sync::{broadcast, mpsc, oneshot, watch, Mutex},
    time::timeout,
};

const SCHEMA_VERSION: &str = "1.0.0";
const RUNTIME_ROOT_ENV: &str = "MTS_RUNTIME_ROOT";
const PYTHON_ENV: &str = "MTS_WORKER_PYTHON";
const PYTHON_HINT_FILE: &str = "worker-python.path";
const PRODUCTION_CONFIG_ENV: &str = "MTS_PRODUCTION_CONFIG";
const REQUEST_TIMEOUT_ENV: &str = "MTS_WORKER_REQUEST_TIMEOUT_MS";
const STARTUP_TIMEOUT_ENV: &str = "MTS_WORKER_STARTUP_TIMEOUT_MS";
const SHUTDOWN_TIMEOUT_ENV: &str = "MTS_WORKER_SHUTDOWN_TIMEOUT_MS";
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_STDOUT_LINE_BYTES: usize = 1024 * 1024;
const MAX_STDIN_LINE_BYTES: usize = 1024 * 1024;
const STDERR_TAIL_BYTES: usize = 64 * 1024;
const WRITE_QUEUE_CAPACITY: usize = 128;
const KILL_QUEUE_CAPACITY: usize = 1;
const EVENT_CHANNEL_CAPACITY: usize = 512;
const MAX_IDENTIFIER_CHARS: usize = 160;
const MAX_TIMESTAMP_CHARS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseKind {
    Accepted,
    Completed,
}

#[derive(Debug, Clone)]
pub struct WorkerResponse {
    pub kind: ResponseKind,
    pub payload: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerErrorKind {
    Configuration,
    Spawn,
    Protocol,
    Timeout,
    Crashed,
    Rejected,
    Overloaded,
    ShuttingDown,
}

#[derive(Debug, Clone)]
pub struct WorkerError {
    pub kind: WorkerErrorKind,
    pub message: String,
}

impl WorkerError {
    fn new(kind: WorkerErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn for_command(mut self, command_type: &str) -> Self {
        if command_type == "job.start" {
            self.message.push_str(
                " The final job.start outcome is unknown; the supervisor failed closed and will never replay this request automatically.",
            );
        } else if matches!(command_type, "review.submit" | "speaker.rename") {
            self.message.push_str(
                " The final human-mutation outcome may be unknown; the supervisor failed closed and will never replay this decision automatically.",
            );
        }
        self
    }
}

impl fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for WorkerError {}

#[derive(Clone)]
pub struct WorkerSupervisor {
    inner: Arc<Inner>,
}

#[derive(Debug, Clone)]
pub struct WorkerEvent {
    pub generation: u64,
    pub event_id: String,
    pub job_id: String,
    pub sequence: u64,
    pub timestamp: String,
    pub event_type: String,
    pub payload: Map<String, Value>,
}

#[derive(Debug, Clone)]
pub enum WorkerNotification {
    Event(WorkerEvent),
    GenerationFailed {
        generation: u64,
        job_ids: Vec<String>,
        message: String,
    },
}

impl Default for WorkerSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerSupervisor {
    pub fn new() -> Self {
        Self::new_with_resource_directory(None)
    }

    pub fn new_with_resource_directory(runtime_resource_directory: Option<PathBuf>) -> Self {
        let (event_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(SupervisorState::default()),
                spawn_lock: Mutex::new(()),
                next_generation: AtomicU64::new(1),
                next_request: AtomicU64::new(1),
                shutting_down: AtomicBool::new(false),
                event_tx,
                runtime_resource_directory,
            }),
        }
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<WorkerNotification> {
        self.inner.event_tx.subscribe()
    }

    /// Returns whether the current worker generation can route verified events
    /// for `job_id`.
    ///
    /// This is intentionally a volatile transport capability check, not a
    /// durable job-status API. The desktop job registry remains authoritative
    /// for in-process identity and state, while a worker restart clears these
    /// generation-scoped routes.
    pub async fn routes_job_events(&self, job_id: &str) -> bool {
        let state = self.inner.state.lock().await;
        let Some(active) = state.active.as_ref().filter(|handle| handle.ready) else {
            return false;
        };
        state
            .jobs
            .get(job_id)
            .is_some_and(|cursor| cursor.generation == active.generation)
    }

    pub async fn request(
        &self,
        command_type: &str,
        payload: Map<String, Value>,
        expected: ResponseKind,
    ) -> Result<WorkerResponse, WorkerError> {
        if self.inner.shutting_down.load(Ordering::Acquire) {
            return Err(WorkerError::new(
                WorkerErrorKind::ShuttingDown,
                "The worker is shutting down and cannot accept new requests.",
            ));
        }

        let generation = self.ensure_ready().await?;
        self.inner
            .request_generation(
                generation,
                command_type,
                payload,
                expected,
                duration_from_env(REQUEST_TIMEOUT_ENV, DEFAULT_REQUEST_TIMEOUT),
            )
            .await
    }

    pub async fn start(&self) -> Result<(), WorkerError> {
        self.ensure_ready().await.map(|_| ())
    }

    pub async fn fail_closed(&self, message: String) {
        let generation = {
            let state = self.inner.state.lock().await;
            state.active.as_ref().map(|handle| handle.generation)
        };
        if let Some(generation) = generation {
            self.inner
                .fail_generation(
                    generation,
                    WorkerError::new(WorkerErrorKind::Protocol, message),
                )
                .await;
        }
    }

    pub async fn shutdown(&self) -> Result<(), WorkerError> {
        if self.inner.shutting_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }

        let handle = {
            let state = self.inner.state.lock().await;
            state.active.clone()
        };
        let Some(handle) = handle else {
            return Ok(());
        };

        handle.expected_exit.store(true, Ordering::Release);
        let shutdown_timeout = duration_from_env(SHUTDOWN_TIMEOUT_ENV, DEFAULT_SHUTDOWN_TIMEOUT);
        let command_result = self
            .inner
            .request_generation(
                handle.generation,
                "worker.shutdown",
                Map::new(),
                ResponseKind::Completed,
                shutdown_timeout,
            )
            .await;

        if command_result.is_ok() && wait_for_exit(handle.exit_rx.clone(), shutdown_timeout).await {
            return Ok(());
        }

        let _ = handle.kill_tx.try_send(());
        let exited_after_kill = wait_for_exit(handle.exit_rx, shutdown_timeout).await;
        command_result?;
        if !exited_after_kill {
            return Err(WorkerError::new(
                WorkerErrorKind::Crashed,
                "The worker did not exit within either the graceful-shutdown or forced-termination window.",
            ));
        }
        Ok(())
    }

    async fn ensure_ready(&self) -> Result<u64, WorkerError> {
        if let Some(generation) = self.ready_generation().await {
            return Ok(generation);
        }

        let _spawn_guard = self.inner.spawn_lock.lock().await;
        if let Some(generation) = self.ready_generation().await {
            return Ok(generation);
        }
        if self.inner.shutting_down.load(Ordering::Acquire) {
            return Err(WorkerError::new(
                WorkerErrorKind::ShuttingDown,
                "The worker is shutting down, so a new process cannot be started.",
            ));
        }

        let generation = self.inner.spawn_process().await?;
        let readiness = self
            .inner
            .request_generation(
                generation,
                "worker.health",
                Map::new(),
                ResponseKind::Completed,
                duration_from_env(STARTUP_TIMEOUT_ENV, DEFAULT_STARTUP_TIMEOUT),
            )
            .await;
        if let Err(error) = readiness {
            self.inner
                .fail_generation(
                    generation,
                    WorkerError::new(
                        error.kind,
                        format!("Worker readiness handshake failed: {}", error.message),
                    ),
                )
                .await;
            return Err(error);
        }

        let mut state = self.inner.state.lock().await;
        match state.active.as_mut() {
            Some(handle) if handle.generation == generation => {
                handle.ready = true;
                Ok(generation)
            }
            _ => Err(WorkerError::new(
                WorkerErrorKind::Crashed,
                "The worker became unavailable after the readiness handshake.",
            )),
        }
    }

    async fn ready_generation(&self) -> Option<u64> {
        let state = self.inner.state.lock().await;
        state
            .active
            .as_ref()
            .filter(|handle| handle.ready)
            .map(|handle| handle.generation)
    }
}

struct Inner {
    state: Mutex<SupervisorState>,
    spawn_lock: Mutex<()>,
    next_generation: AtomicU64,
    next_request: AtomicU64,
    shutting_down: AtomicBool,
    event_tx: broadcast::Sender<WorkerNotification>,
    runtime_resource_directory: Option<PathBuf>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Ok(state) = self.state.try_lock() {
            if let Some(handle) = state.active.as_ref() {
                let _ = handle.kill_tx.try_send(());
            }
        }
    }
}

#[derive(Default)]
struct SupervisorState {
    active: Option<ProcessHandle>,
    pending: HashMap<String, PendingRequest>,
    jobs: HashMap<String, JobEventCursor>,
    seen_event_ids: HashSet<String>,
}

#[derive(Debug)]
struct JobEventCursor {
    generation: u64,
    next_sequence: u64,
}

#[derive(Clone)]
struct ProcessHandle {
    generation: u64,
    write_tx: mpsc::Sender<Vec<u8>>,
    kill_tx: mpsc::Sender<()>,
    exit_rx: watch::Receiver<Option<ExitReport>>,
    expected_exit: Arc<AtomicBool>,
    stderr_tail: Arc<StdMutex<BoundedTail>>,
    ready: bool,
}

struct PendingRequest {
    generation: u64,
    command_type: String,
    job_id: Option<String>,
    expected: ResponseKind,
    sender: oneshot::Sender<Result<WorkerResponse, WorkerError>>,
}

#[derive(Debug, Clone)]
struct ExitReport {
    description: String,
}

#[derive(Debug)]
struct WorkerLaunchConfig {
    python: PathBuf,
    production_config: PathBuf,
    repository_root: PathBuf,
}

impl WorkerLaunchConfig {
    fn resolve(runtime_resource_directory: Option<&Path>) -> Result<Self, WorkerError> {
        let repository_root = repository_root(runtime_resource_directory)?;
        let production_config = resolve_production_config(&repository_root)?;
        let python = resolve_worker_python(&repository_root, production_config.parent())?;
        Ok(Self {
            python,
            production_config,
            repository_root,
        })
    }
}

impl Inner {
    async fn spawn_process(self: &Arc<Self>) -> Result<u64, WorkerError> {
        let launch = WorkerLaunchConfig::resolve(self.runtime_resource_directory.as_deref())?;
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        let mut command = Command::new(&launch.python);
        command
            .arg("-m")
            .arg("backend.worker")
            .arg("--config")
            .arg(&launch.production_config)
            .current_dir(&launch.repository_root)
            .env(PRODUCTION_CONFIG_ENV, &launch.production_config)
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8:strict")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        sanitize_python_environment(&mut command);

        let mut child = command.spawn().map_err(|error| {
            WorkerError::new(
                WorkerErrorKind::Spawn,
                format!(
                    "Unable to start the media-asr worker ({}): {error}",
                    launch.python.display()
                ),
            )
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            WorkerError::new(
                WorkerErrorKind::Spawn,
                "The worker process did not provide piped stdin.",
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            WorkerError::new(
                WorkerErrorKind::Spawn,
                "The worker process did not provide piped stdout.",
            )
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            WorkerError::new(
                WorkerErrorKind::Spawn,
                "The worker process did not provide piped stderr.",
            )
        })?;

        let (write_tx, write_rx) = mpsc::channel(WRITE_QUEUE_CAPACITY);
        let (kill_tx, kill_rx) = mpsc::channel(KILL_QUEUE_CAPACITY);
        let (exit_tx, exit_rx) = watch::channel(None);
        let expected_exit = Arc::new(AtomicBool::new(false));
        let stderr_tail = Arc::new(StdMutex::new(BoundedTail::new(STDERR_TAIL_BYTES)));
        let handle = ProcessHandle {
            generation,
            write_tx,
            kill_tx,
            exit_rx,
            expected_exit: Arc::clone(&expected_exit),
            stderr_tail: Arc::clone(&stderr_tail),
            ready: false,
        };

        {
            let mut state = self.state.lock().await;
            if state.active.is_some() {
                let _ = child.start_kill();
                return Err(WorkerError::new(
                    WorkerErrorKind::Spawn,
                    "Internal error: attempted to start a new process while a worker was already active.",
                ));
            }
            state.jobs.clear();
            state.seen_event_ids.clear();
            state.active = Some(handle);
        }

        let writer_inner = Arc::clone(self);
        tokio::spawn(async move {
            writer_loop(stdin, write_rx, generation, writer_inner).await;
        });
        let stdout_inner = Arc::clone(self);
        tokio::spawn(async move {
            stdout_loop(stdout, generation, stdout_inner).await;
        });
        let stderr_inner = Arc::clone(self);
        tokio::spawn(async move {
            stderr_loop(stderr, generation, stderr_tail, stderr_inner).await;
        });
        let waiter_inner = Arc::clone(self);
        tokio::spawn(async move {
            child_waiter(
                child,
                kill_rx,
                generation,
                expected_exit,
                exit_tx,
                waiter_inner,
            )
            .await;
        });

        Ok(generation)
    }

    async fn request_generation(
        self: &Arc<Self>,
        generation: u64,
        command_type: &str,
        payload: Map<String, Value>,
        expected: ResponseKind,
        request_timeout: Duration,
    ) -> Result<WorkerResponse, WorkerError> {
        validate_command_type(command_type)?;
        let job_id = if command_type == "job.start" {
            let value = payload
                .get("jobId")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    WorkerError::new(
                        WorkerErrorKind::Protocol,
                        "job.start payload must contain a string jobId before it can be registered for worker events.",
                    )
                })?
                .to_owned();
            validate_identifier(&value, "job.start payload.jobId")?;
            Some(value)
        } else {
            None
        };
        let request_id = format!(
            "req-{generation}-{}",
            self.next_request.fetch_add(1, Ordering::Relaxed)
        );
        let envelope = CommandEnvelope {
            schema_version: SCHEMA_VERSION,
            request_id: &request_id,
            command_type,
            payload,
        };
        let mut line = serde_json::to_vec(&envelope).map_err(|error| {
            WorkerError::new(
                WorkerErrorKind::Protocol,
                format!("Unable to serialize the worker request: {error}"),
            )
        })?;
        if line.len() > MAX_STDIN_LINE_BYTES {
            return Err(WorkerError::new(
                WorkerErrorKind::Protocol,
                format!(
                    "The worker request exceeds the JSONL line limit ({} bytes).",
                    MAX_STDIN_LINE_BYTES
                ),
            ));
        }
        line.push(b'\n');

        let (sender, receiver) = oneshot::channel();
        let write_tx = {
            let mut state = self.state.lock().await;
            let handle = state
                .active
                .as_ref()
                .filter(|handle| handle.generation == generation)
                .ok_or_else(|| {
                    WorkerError::new(
                        WorkerErrorKind::Crashed,
                        "The worker generation is no longer active.",
                    )
                    .for_command(command_type)
                })?;
            let write_tx = handle.write_tx.clone();
            if state
                .pending
                .insert(
                    request_id.clone(),
                    PendingRequest {
                        generation,
                        command_type: command_type.to_owned(),
                        job_id,
                        expected,
                        sender,
                    },
                )
                .is_some()
            {
                return Err(WorkerError::new(
                    WorkerErrorKind::Protocol,
                    "Internal error: a duplicate requestId was generated.",
                ));
            }
            write_tx
        };

        if let Err(error) = write_tx.try_send(line) {
            self.remove_pending(&request_id, generation).await;
            return match error {
                mpsc::error::TrySendError::Full(_) => Err(WorkerError::new(
                    WorkerErrorKind::Overloaded,
                    "The worker write queue is full; the request was not queued or retried automatically.",
                )),
                mpsc::error::TrySendError::Closed(_) => {
                    let error = WorkerError::new(
                        WorkerErrorKind::Crashed,
                        "The worker stdin channel is closed; this generation failed closed and the request was not retried automatically.",
                    )
                    .for_command(command_type);
                    self.fail_generation(generation, error.clone()).await;
                    Err(error)
                }
            };
        }

        match timeout(request_timeout, receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(WorkerError::new(
                WorkerErrorKind::Crashed,
                "The worker request correlation channel closed unexpectedly.",
            )
            .for_command(command_type)),
            Err(_) => {
                self.remove_pending(&request_id, generation).await;
                let error = WorkerError::new(
                    WorkerErrorKind::Timeout,
                    format!(
                        "Worker request {request_id} did not return an explicit response within {} ms; this generation failed closed.",
                        request_timeout.as_millis()
                    ),
                )
                .for_command(command_type);
                self.fail_generation(generation, error.clone()).await;
                Err(error)
            }
        }
    }

    async fn remove_pending(&self, request_id: &str, generation: u64) {
        let mut state = self.state.lock().await;
        if state
            .pending
            .get(request_id)
            .is_some_and(|pending| pending.generation == generation)
        {
            state.pending.remove(request_id);
        }
    }

    async fn handle_output_line(
        self: &Arc<Self>,
        generation: u64,
        line: &str,
    ) -> Result<(), WorkerError> {
        if !self.is_current_generation(generation).await {
            return Ok(());
        }
        match parse_output_line(line)? {
            OutputEnvelope::Event(event) => self.route_event(generation, event).await,
            OutputEnvelope::Control(control) => self.route_control(generation, control).await,
        }
    }

    async fn route_event(
        self: &Arc<Self>,
        generation: u64,
        event: EventEnvelope,
    ) -> Result<(), WorkerError> {
        {
            let mut state = self.state.lock().await;
            if !state
                .active
                .as_ref()
                .is_some_and(|handle| handle.generation == generation)
            {
                return Ok(());
            }
            let (job_generation, next_sequence) = state
                .jobs
                .get(&event.job_id)
                .map(|cursor| (cursor.generation, cursor.next_sequence))
                .ok_or_else(|| {
                    WorkerError::new(
                        WorkerErrorKind::Protocol,
                        format!(
                            "Worker event {} referenced unregistered jobId {:?}.",
                            event.event_id, event.job_id
                        ),
                    )
                })?;
            if job_generation != generation {
                return Err(WorkerError::new(
                    WorkerErrorKind::Protocol,
                    "A worker event crossed process-generation boundaries and was rejected.",
                ));
            }
            if event.sequence != next_sequence {
                return Err(WorkerError::new(
                    WorkerErrorKind::Protocol,
                    format!(
                        "Worker event sequence for job {:?} must be exactly {}; received {}.",
                        event.job_id, next_sequence, event.sequence
                    ),
                ));
            }
            if !state.seen_event_ids.insert(event.event_id.clone()) {
                return Err(WorkerError::new(
                    WorkerErrorKind::Protocol,
                    format!(
                        "Worker eventId {:?} was duplicated or replayed.",
                        event.event_id
                    ),
                ));
            }
            let cursor = state.jobs.get_mut(&event.job_id).ok_or_else(|| {
                WorkerError::new(
                    WorkerErrorKind::Protocol,
                    "The registered worker-event cursor disappeared during validation.",
                )
            })?;
            cursor.next_sequence = cursor.next_sequence.checked_add(1).ok_or_else(|| {
                WorkerError::new(
                    WorkerErrorKind::Protocol,
                    "Worker event sequence overflowed and the generation failed closed.",
                )
            })?;
            if is_terminal_event_type(&event.event_type) {
                state.jobs.remove(&event.job_id);
            }
        }

        let _ = self.event_tx.send(WorkerNotification::Event(WorkerEvent {
            generation,
            event_id: event.event_id,
            job_id: event.job_id,
            sequence: event.sequence,
            timestamp: event.timestamp,
            event_type: event.event_type,
            payload: event.payload,
        }));
        Ok(())
    }

    async fn route_control(
        self: &Arc<Self>,
        generation: u64,
        control: ControlEnvelope,
    ) -> Result<(), WorkerError> {
        if control.response_type == "worker.startup.failed" {
            if control.request_id != "startup" {
                return Err(WorkerError::new(
                    WorkerErrorKind::Protocol,
                    "worker.startup.failed must use requestId=startup.",
                ));
            }
            let rejection = parse_rejection_payload(control.payload)?;
            return Err(WorkerError::new(
                WorkerErrorKind::Spawn,
                format!(
                    "Worker preflight failed [{}]: {} (retryable={})",
                    rejection.code, rejection.message, rejection.retryable
                ),
            ));
        }

        let mut state = self.state.lock().await;
        let Some(pending) = state.pending.remove(&control.request_id) else {
            return Err(WorkerError::new(
                WorkerErrorKind::Protocol,
                format!(
                    "The worker returned an unknown or expired requestId: {}.",
                    control.request_id
                ),
            ));
        };
        if pending.generation != generation {
            state.pending.insert(control.request_id.clone(), pending);
            return Err(WorkerError::new(
                WorkerErrorKind::Protocol,
                "The worker response crossed generation boundaries and was rejected.",
            ));
        }

        let result = match control.response_type.as_str() {
            "command.accepted" => {
                if pending.expected != ResponseKind::Accepted {
                    Err(WorkerError::new(
                        WorkerErrorKind::Protocol,
                        format!(
                            "{} expected command.completed but received command.accepted.",
                            pending.command_type
                        ),
                    ))
                } else if pending.command_type == "job.start" {
                    let expected_job_id = pending.job_id.as_deref().ok_or_else(|| {
                        WorkerError::new(
                            WorkerErrorKind::Protocol,
                            "Internal error: job.start lost its registered jobId.",
                        )
                    })?;
                    let accepted_job_id = control
                        .payload
                        .get("jobId")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            WorkerError::new(
                                WorkerErrorKind::Protocol,
                                "Accepted job.start response must contain a string jobId.",
                            )
                        })?;
                    if accepted_job_id != expected_job_id {
                        return Err(WorkerError::new(
                            WorkerErrorKind::Protocol,
                            format!(
                                "Accepted job.start response jobId mismatch: expected {expected_job_id:?}, received {accepted_job_id:?}."
                            ),
                        ));
                    }
                    if state.jobs.contains_key(expected_job_id) {
                        return Err(WorkerError::new(
                            WorkerErrorKind::Protocol,
                            format!(
                                "Accepted job.start attempted to register duplicate jobId {expected_job_id:?}."
                            ),
                        ));
                    }
                    state.jobs.insert(
                        expected_job_id.to_owned(),
                        JobEventCursor {
                            generation,
                            next_sequence: 0,
                        },
                    );
                    Ok(WorkerResponse {
                        kind: ResponseKind::Accepted,
                        payload: control.payload,
                    })
                } else {
                    Ok(WorkerResponse {
                        kind: ResponseKind::Accepted,
                        payload: control.payload,
                    })
                }
            }
            "command.completed" => {
                if pending.expected != ResponseKind::Completed {
                    Err(WorkerError::new(
                        WorkerErrorKind::Protocol,
                        format!(
                            "{} expected command.accepted but received command.completed.",
                            pending.command_type
                        ),
                    ))
                } else {
                    Ok(WorkerResponse {
                        kind: ResponseKind::Completed,
                        payload: control.payload,
                    })
                }
            }
            "command.rejected" => match parse_rejection_payload(control.payload) {
                Ok(rejection) => Err(WorkerError::new(
                    WorkerErrorKind::Rejected,
                    format!(
                        "The worker rejected {} [{}]: {} (retryable={})",
                        pending.command_type,
                        rejection.code,
                        rejection.message,
                        rejection.retryable
                    ),
                )),
                Err(error) => Err(error),
            },
            _ => Err(WorkerError::new(
                WorkerErrorKind::Protocol,
                format!(
                    "Unsupported worker control response type: {}.",
                    control.response_type
                ),
            )),
        };
        let command_type = pending.command_type.clone();
        let protocol_mismatch = result
            .as_ref()
            .err()
            .is_some_and(|error| error.kind == WorkerErrorKind::Protocol);
        let _ = pending
            .sender
            .send(result.map_err(|error| error.for_command(&command_type)));
        drop(state);

        if protocol_mismatch {
            return Err(WorkerError::new(
                WorkerErrorKind::Protocol,
                format!("{command_type} returned a protocol-invalid response type."),
            ));
        }
        Ok(())
    }

    async fn is_current_generation(&self, generation: u64) -> bool {
        let state = self.state.lock().await;
        state
            .active
            .as_ref()
            .is_some_and(|handle| handle.generation == generation)
    }

    async fn fail_generation(&self, generation: u64, base_error: WorkerError) {
        let (handle, pending, job_ids) = {
            let mut state = self.state.lock().await;
            let handle = if state
                .active
                .as_ref()
                .is_some_and(|handle| handle.generation == generation)
            {
                state.active.take()
            } else {
                None
            };
            let request_ids = state
                .pending
                .iter()
                .filter(|(_, pending)| pending.generation == generation)
                .map(|(request_id, _)| request_id.clone())
                .collect::<Vec<_>>();
            let pending = request_ids
                .into_iter()
                .filter_map(|request_id| state.pending.remove(&request_id))
                .collect::<Vec<_>>();
            let job_ids = state
                .jobs
                .iter()
                .filter(|(_, cursor)| cursor.generation == generation)
                .map(|(job_id, _)| job_id.clone())
                .collect::<Vec<_>>();
            state
                .jobs
                .retain(|_, cursor| cursor.generation != generation);
            state.seen_event_ids.clear();
            (handle, pending, job_ids)
        };

        let mut error = base_error;
        if let Some(handle) = handle.as_ref() {
            let stderr = stderr_snapshot(&handle.stderr_tail);
            if !stderr.trim().is_empty() {
                error.message.push_str("\nworker stderr tail:\n");
                error.message.push_str(&stderr);
            }
            let _ = handle.kill_tx.try_send(());
        }
        for pending in pending {
            let _ = pending
                .sender
                .send(Err(error.clone().for_command(&pending.command_type)));
        }
        if handle.is_some() || !job_ids.is_empty() {
            let _ = self.event_tx.send(WorkerNotification::GenerationFailed {
                generation,
                job_ids,
                message: error.message,
            });
        }
    }
}

fn sanitize_python_environment(command: &mut Command) {
    // AppImage's AppRun wrapper points these variables at its own minimal
    // filesystem. The worker may use an external Python, whose standard
    // library must be resolved from that interpreter rather than the AppDir.
    command.env_remove("PYTHONHOME").env_remove("PYTHONPATH");
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CommandEnvelope<'a> {
    schema_version: &'static str,
    request_id: &'a str,
    #[serde(rename = "type")]
    command_type: &'a str,
    payload: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ControlEnvelope {
    schema_version: String,
    request_id: String,
    timestamp: String,
    #[serde(rename = "type")]
    response_type: String,
    payload: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EventEnvelope {
    schema_version: String,
    event_id: String,
    job_id: String,
    sequence: u64,
    timestamp: String,
    #[serde(rename = "type")]
    event_type: String,
    payload: Map<String, Value>,
}

#[derive(Debug)]
enum OutputEnvelope {
    Control(ControlEnvelope),
    Event(EventEnvelope),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RejectionPayload {
    code: String,
    message: String,
    retryable: bool,
    #[serde(default)]
    details: Option<Map<String, Value>>,
}

fn parse_output_line(line: &str) -> Result<OutputEnvelope, WorkerError> {
    if line.is_empty() {
        return Err(WorkerError::new(
            WorkerErrorKind::Protocol,
            "Worker stdout contained an empty JSONL line.",
        ));
    }
    let control_result = serde_json::from_str::<ControlEnvelope>(line);
    let event_result = serde_json::from_str::<EventEnvelope>(line);
    match (control_result, event_result) {
        (Ok(control), Err(_)) => {
        validate_protocol_version(&control.schema_version)?;
        validate_identifier(&control.request_id, "requestId")?;
        validate_timestamp(&control.timestamp)?;
        if !matches!(
            control.response_type.as_str(),
            "command.accepted"
                | "command.completed"
                | "command.rejected"
                | "worker.startup.failed"
        ) {
            return Err(WorkerError::new(
                WorkerErrorKind::Protocol,
                format!("Unknown control type: {}.", control.response_type),
            ));
        }
        Ok(OutputEnvelope::Control(control))
        }
        (Err(_), Ok(event)) => {
        validate_protocol_version(&event.schema_version)?;
        validate_identifier(&event.event_id, "eventId")?;
        validate_identifier(&event.job_id, "jobId")?;
        validate_timestamp(&event.timestamp)?;
            validate_identifier(&event.event_type, "event type")?;
            validate_event_type(&event.event_type)?;
            Ok(OutputEnvelope::Event(event))
        }
        (Ok(_), Ok(_)) => Err(WorkerError::new(
            WorkerErrorKind::Protocol,
            "Worker stdout matched both control and job-event envelopes; the generation failed closed.",
        )),
        (Err(control_error), Err(event_error)) => Err(WorkerError::new(
            WorkerErrorKind::Protocol,
            format!(
                "Worker stdout did not match the strict control or job-event schema; control={control_error}; event={event_error}"
            ),
        )),
    }
}

fn parse_rejection_payload(payload: Map<String, Value>) -> Result<RejectionPayload, WorkerError> {
    let rejection =
        serde_json::from_value::<RejectionPayload>(Value::Object(payload)).map_err(|error| {
            WorkerError::new(
                WorkerErrorKind::Protocol,
                format!("Invalid command.rejected payload schema: {error}"),
            )
        })?;
    validate_identifier(&rejection.code, "rejection.code")?;
    validate_nonempty_text(&rejection.message, "rejection.message", 4096)?;
    if let Some(details) = rejection.details.as_ref() {
        let _ = details.len();
    }
    Ok(rejection)
}

fn validate_protocol_version(version: &str) -> Result<(), WorkerError> {
    if version != SCHEMA_VERSION {
        return Err(WorkerError::new(
            WorkerErrorKind::Protocol,
            format!("Worker schemaVersion must be {SCHEMA_VERSION}; received {version:?}."),
        ));
    }
    Ok(())
}

fn validate_command_type(command_type: &str) -> Result<(), WorkerError> {
    if !matches!(
        command_type,
        "job.start"
            | "job.cancel"
            | "job.status"
            | "review.queue"
            | "review.submit"
            | "speaker.rename"
            | "job.resume"
            | "job.rerender"
            | "worker.health"
            | "worker.shutdown"
    ) {
        return Err(WorkerError::new(
            WorkerErrorKind::Protocol,
            format!("Unsupported worker command type: {command_type:?}."),
        ));
    }
    Ok(())
}

fn validate_event_type(event_type: &str) -> Result<(), WorkerError> {
    if !matches!(
        event_type,
        "job.started"
            | "stage.started"
            | "stage.progress"
            | "artifact.created"
            | "review.required"
            | "review.decision.persisted"
            | "warning"
            | "job.failed"
            | "job.completed"
            | "job.cancelled"
    ) {
        return Err(WorkerError::new(
            WorkerErrorKind::Protocol,
            format!("Unsupported worker event type: {event_type:?}."),
        ));
    }
    Ok(())
}

fn is_terminal_event_type(event_type: &str) -> bool {
    matches!(event_type, "job.failed" | "job.completed" | "job.cancelled")
}

fn validate_identifier(value: &str, field: &str) -> Result<(), WorkerError> {
    validate_nonempty_text(value, field, MAX_IDENTIFIER_CHARS)
}

fn validate_timestamp(value: &str) -> Result<(), WorkerError> {
    validate_nonempty_text(value, "timestamp", MAX_TIMESTAMP_CHARS)
}

fn validate_nonempty_text(value: &str, field: &str, max_chars: usize) -> Result<(), WorkerError> {
    if value.trim().is_empty()
        || value.chars().count() > max_chars
        || value.chars().any(char::is_control)
    {
        return Err(WorkerError::new(
            WorkerErrorKind::Protocol,
            format!("{field} is empty, too long, or contains control characters."),
        ));
    }
    Ok(())
}

async fn writer_loop(
    mut stdin: ChildStdin,
    mut receiver: mpsc::Receiver<Vec<u8>>,
    generation: u64,
    inner: Arc<Inner>,
) {
    while let Some(line) = receiver.recv().await {
        if let Err(error) = stdin.write_all(&line).await {
            inner
                .fail_generation(
                    generation,
                    WorkerError::new(
                        WorkerErrorKind::Crashed,
                        format!("Failed to write to worker stdin: {error}"),
                    ),
                )
                .await;
            return;
        }
        if let Err(error) = stdin.flush().await {
            inner
                .fail_generation(
                    generation,
                    WorkerError::new(
                        WorkerErrorKind::Crashed,
                        format!("Failed to flush worker stdin: {error}"),
                    ),
                )
                .await;
            return;
        }
    }
}

async fn stdout_loop(stdout: ChildStdout, generation: u64, inner: Arc<Inner>) {
    let mut reader = BufReader::new(stdout);
    loop {
        let line = match read_bounded_utf8_line(&mut reader, MAX_STDOUT_LINE_BYTES).await {
            Ok(Some(line)) => line,
            Ok(None) => {
                inner
                    .fail_generation(
                        generation,
                        WorkerError::new(
                            WorkerErrorKind::Crashed,
                            "Worker stdout reached EOF; this generation failed closed.",
                        ),
                    )
                    .await;
                return;
            }
            Err(error) => {
                inner.fail_generation(generation, error).await;
                return;
            }
        };
        if let Err(error) = inner.handle_output_line(generation, &line).await {
            inner.fail_generation(generation, error).await;
            return;
        }
    }
}

async fn stderr_loop(
    mut stderr: ChildStderr,
    generation: u64,
    tail: Arc<StdMutex<BoundedTail>>,
    inner: Arc<Inner>,
) {
    let mut buffer = [0_u8; 4096];
    loop {
        match stderr.read(&mut buffer).await {
            Ok(0) => return,
            Ok(read) => with_tail(&tail, |bounded| bounded.push(&buffer[..read])),
            Err(error) => {
                inner
                    .fail_generation(
                        generation,
                        WorkerError::new(
                            WorkerErrorKind::Crashed,
                            format!("Failed to read worker stderr: {error}"),
                        ),
                    )
                    .await;
                return;
            }
        }
    }
}

async fn child_waiter(
    mut child: Child,
    mut kill_rx: mpsc::Receiver<()>,
    generation: u64,
    expected_exit: Arc<AtomicBool>,
    exit_tx: watch::Sender<Option<ExitReport>>,
    inner: Arc<Inner>,
) {
    let status_result = tokio::select! {
        status = child.wait() => status,
        _ = kill_rx.recv() => {
            let _ = child.start_kill();
            child.wait().await
        }
    };
    let description = match status_result {
        Ok(status) => format!("exit status {status}"),
        Err(error) => format!("wait error: {error}"),
    };
    let _ = exit_tx.send(Some(ExitReport {
        description: description.clone(),
    }));

    let kind = if expected_exit.load(Ordering::Acquire) {
        WorkerErrorKind::ShuttingDown
    } else {
        WorkerErrorKind::Crashed
    };
    inner
        .fail_generation(
            generation,
            WorkerError::new(kind, format!("The worker process exited: {description}.")),
        )
        .await;
}

async fn read_bounded_utf8_line<R>(
    reader: &mut R,
    max_bytes: usize,
) -> Result<Option<String>, WorkerError>
where
    R: AsyncBufRead + Unpin,
{
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().await.map_err(|error| {
            WorkerError::new(
                WorkerErrorKind::Crashed,
                format!("Failed to read worker stdout: {error}"),
            )
        })?;
        if available.is_empty() {
            if line.is_empty() {
                return Ok(None);
            }
            return Err(WorkerError::new(
                WorkerErrorKind::Protocol,
                "Worker stdout reached EOF before the JSONL line ended.",
            ));
        }

        if let Some(newline) = available.iter().position(|byte| *byte == b'\n') {
            if line.len().saturating_add(newline) > max_bytes {
                return Err(WorkerError::new(
                    WorkerErrorKind::Protocol,
                    format!("Worker stdout JSONL line exceeds {max_bytes} bytes."),
                ));
            }
            line.extend_from_slice(&available[..newline]);
            reader.consume(newline + 1);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let text = String::from_utf8(line).map_err(|error| {
                WorkerError::new(
                    WorkerErrorKind::Protocol,
                    format!("Worker stdout contains invalid UTF-8: {error}"),
                )
            })?;
            return Ok(Some(text));
        }

        if line.len().saturating_add(available.len()) > max_bytes {
            return Err(WorkerError::new(
                WorkerErrorKind::Protocol,
                format!("Worker stdout JSONL line exceeds {max_bytes} bytes."),
            ));
        }
        let consumed = available.len();
        line.extend_from_slice(available);
        reader.consume(consumed);
    }
}

async fn wait_for_exit(
    mut receiver: watch::Receiver<Option<ExitReport>>,
    wait_timeout: Duration,
) -> bool {
    if receiver.borrow().is_some() {
        return true;
    }
    timeout(wait_timeout, async move {
        loop {
            if receiver.changed().await.is_err() {
                return false;
            }
            if let Some(report) = receiver.borrow().as_ref() {
                let _ = &report.description;
                return true;
            }
        }
    })
    .await
    .unwrap_or(false)
}

fn repository_root(runtime_resource_directory: Option<&Path>) -> Result<PathBuf, WorkerError> {
    let development_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or_else(|| {
            WorkerError::new(
                WorkerErrorKind::Configuration,
                "Unable to resolve the repository root from CARGO_MANIFEST_DIR.",
            )
        })?;

    if let Some(path) = env::var_os(RUNTIME_ROOT_ENV) {
        return resolve_runtime_root_candidate(Path::new(&path), RUNTIME_ROOT_ENV);
    }

    let executable = env::current_exe().map_err(|error| {
        WorkerError::new(
            WorkerErrorKind::Configuration,
            format!("Unable to resolve the installed application path: {error}"),
        )
    })?;
    resolve_runtime_root_from(&executable, development_root, runtime_resource_directory)
}

fn resolve_runtime_root_from(
    executable: &Path,
    development_root: &Path,
    runtime_resource_directory: Option<&Path>,
) -> Result<PathBuf, WorkerError> {
    if let Some(resource_directory) = runtime_resource_directory {
        for candidate in [
            resource_directory.to_path_buf(),
            resource_directory.join("mts-runtime"),
        ] {
            if runtime_root_marker(&candidate).is_file() {
                return resolve_runtime_root_candidate(&candidate, "Tauri resource directory");
            }
        }
    }

    let mut cursor = executable.parent().map(Path::to_path_buf);
    for depth in 0..=3 {
        let Some(parent) = cursor.take() else {
            break;
        };
        let candidates = [
            parent.clone(),
            parent.join("resources"),
            // Tauri uses the canonical macOS bundle directory name
            // `Contents/Resources`; keep the case-sensitive path explicit so
            // Linux-built fixtures and case-sensitive APFS behave like macOS.
            parent.join("Resources"),
            parent.join("payload"),
            parent.join("app"),
            parent.join("resources").join("mts-runtime"),
            parent.join("Resources").join("mts-runtime"),
        ];
        for candidate in candidates {
            if runtime_root_marker(&candidate).is_file() {
                return resolve_runtime_root_candidate(
                    &candidate,
                    if depth == 0 {
                        "installed executable"
                    } else {
                        "installed resources"
                    },
                );
            }
        }
        cursor = parent.parent().map(Path::to_path_buf);
    }
    resolve_runtime_root_candidate(development_root, "CARGO_MANIFEST_DIR")
}

fn runtime_root_marker(root: &Path) -> PathBuf {
    root.join("backend").join("worker.py")
}

fn resolve_runtime_root_candidate(candidate: &Path, source: &str) -> Result<PathBuf, WorkerError> {
    let canonical = fs::canonicalize(candidate).map_err(|error| {
        WorkerError::new(
            WorkerErrorKind::Configuration,
            format!(
                "Runtime root from {source} does not exist or cannot be resolved: {} ({error}).",
                candidate.display()
            ),
        )
    })?;
    if !canonical.is_dir() || !runtime_root_marker(&canonical).is_file() {
        return Err(WorkerError::new(
            WorkerErrorKind::Configuration,
            format!(
                "Runtime root from {source} must contain backend/worker.py: {}. Set {RUNTIME_ROOT_ENV} to a complete release payload.",
                canonical.display()
            ),
        ));
    }
    Ok(canonical)
}

fn resolve_production_config(repository_root: &Path) -> Result<PathBuf, WorkerError> {
    if let Some(path) = env::var_os(PRODUCTION_CONFIG_ENV) {
        return resolve_required_file(
            PathBuf::from(path),
            "production config",
            PRODUCTION_CONFIG_ENV,
        );
    }

    let local_app_data = env::var_os("LOCALAPPDATA").map(PathBuf::from);
    resolve_production_config_from(repository_root, local_app_data.as_deref())
}

fn resolve_production_config_from(
    repository_root: &Path,
    local_app_data: Option<&Path>,
) -> Result<PathBuf, WorkerError> {
    let mut candidates = vec![repository_root.join("production.config.json")];
    if let Some(data_root) = env::var_os("MTS_DATA_ROOT").map(PathBuf::from) {
        candidates.push(data_root.join("config").join("production.config.json"));
    }
    if let Some(local_app_data) = local_app_data {
        candidates.push(
            local_app_data
                .join("MediaTranscribeStudio")
                .join("config")
                .join("production.config.json"),
        );
    }
    if cfg!(target_os = "macos") {
        if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
            candidates.push(
                home.join("Library")
                    .join("Application Support")
                    .join("MediaTranscribeStudio")
                    .join("config")
                    .join("production.config.json"),
            );
        }
    }
    if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
        let xdg_data_home = env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local").join("share"));
        candidates.push(
            xdg_data_home
                .join("MediaTranscribeStudio")
                .join("config")
                .join("production.config.json"),
        );
    }
    candidates.push(repository_root.join("production.config.example.json"));

    for candidate in candidates {
        if candidate.is_file() {
            return resolve_required_file(candidate, "production config", PRODUCTION_CONFIG_ENV);
        }
    }
    Err(WorkerError::new(
        WorkerErrorKind::Configuration,
        format!(
            "No production configuration is available under {}, MTS_DATA_ROOT, LOCALAPPDATA, or the macOS Application Support directory. Run the runtime bootstrap or set {PRODUCTION_CONFIG_ENV}.",
            repository_root.display()
        ),
    ))
}

fn resolve_worker_python(
    repository_root: &Path,
    config_directory: Option<&Path>,
) -> Result<PathBuf, WorkerError> {
    if let Some(path) = env::var_os(PYTHON_ENV) {
        return resolve_required_file(PathBuf::from(path), "media-asr Python", PYTHON_ENV);
    }

    if let Some(config_directory) = config_directory {
        let hint = config_directory.join(PYTHON_HINT_FILE);
        if hint.is_file() {
            let bytes = fs::read(&hint).map_err(|error| {
                WorkerError::new(
                    WorkerErrorKind::Configuration,
                    format!(
                        "Unable to read worker Python hint {}: {error}",
                        hint.display()
                    ),
                )
            })?;
            if bytes.len() > 4096 {
                return Err(WorkerError::new(
                    WorkerErrorKind::Configuration,
                    format!("Worker Python hint is too large: {}", hint.display()),
                ));
            }
            let value = String::from_utf8(bytes).map_err(|error| {
                WorkerError::new(
                    WorkerErrorKind::Configuration,
                    format!(
                        "Worker Python hint is not UTF-8: {} ({error})",
                        hint.display()
                    ),
                )
            })?;
            let nonempty_lines = value.lines().filter(|line| !line.trim().is_empty()).count();
            if nonempty_lines != 1 {
                return Err(WorkerError::new(
                    WorkerErrorKind::Configuration,
                    format!(
                        "Worker Python hint must contain exactly one non-empty path: {}",
                        hint.display()
                    ),
                ));
            }
            return resolve_required_file(
                PathBuf::from(value.trim()),
                "media-asr Python hint",
                PYTHON_ENV,
            );
        }
    }

    for relative_path in python_relative_paths() {
        let candidate = repository_root
            .join("runtime")
            .join("media-asr")
            .join(relative_path);
        if candidate.is_file() {
            return resolve_required_file(candidate, "media-asr Python", PYTHON_ENV);
        }
    }

    for root_env in ["CONDA_PREFIX", "VIRTUAL_ENV"] {
        let Some(root) = env::var_os(root_env).map(PathBuf::from) else {
            continue;
        };
        for relative_path in python_relative_paths() {
            let candidate = root.join(relative_path);
            if candidate.is_file() {
                return fs::canonicalize(&candidate).map_err(|error| {
                    WorkerError::new(
                        WorkerErrorKind::Configuration,
                        format!(
                            "Unable to resolve Python from {root_env}: {} ({error}).",
                            candidate.display()
                        ),
                    )
                });
            }
        }
    }

    if let Some(path_value) = env::var_os("PATH") {
        for directory in env::split_paths(&path_value) {
            for executable_name in python_executable_names() {
                let candidate = directory.join(executable_name);
                if candidate.is_file() {
                    return fs::canonicalize(&candidate).map_err(|error| {
                        WorkerError::new(
                            WorkerErrorKind::Configuration,
                            format!(
                                "Unable to resolve Python from PATH: {} ({error}).",
                                candidate.display()
                            ),
                        )
                    });
                }
            }
        }
    }

    Err(WorkerError::new(
        WorkerErrorKind::Configuration,
        format!(
            "No usable media-asr Python was found in the release runtime, the user config hint, an active Conda/virtual environment, or PATH. Run the runtime bootstrap or set {PYTHON_ENV} to the absolute Python executable path."
        ),
    ))
}

#[cfg(windows)]
fn python_relative_paths() -> &'static [&'static str] {
    &["python.exe", "Scripts/python.exe"]
}

#[cfg(not(windows))]
fn python_relative_paths() -> &'static [&'static str] {
    &["bin/python3", "bin/python"]
}

#[cfg(windows)]
fn python_executable_names() -> &'static [&'static str] {
    &["python.exe", "python3.exe"]
}

#[cfg(not(windows))]
fn python_executable_names() -> &'static [&'static str] {
    &["python3", "python"]
}

fn resolve_required_file(
    path: PathBuf,
    label: &str,
    override_env: &str,
) -> Result<PathBuf, WorkerError> {
    if !path.is_absolute() {
        return Err(WorkerError::new(
            WorkerErrorKind::Configuration,
            format!(
                "{label} must be an absolute path: {}. Override it with {override_env}.",
                path.display()
            ),
        ));
    }
    let canonical = fs::canonicalize(&path).map_err(|error| {
        WorkerError::new(
            WorkerErrorKind::Configuration,
            format!(
                "{label} does not exist or cannot be resolved: {} ({error}). Set {override_env}.",
                path.display()
            ),
        )
    })?;
    if !canonical.is_file() {
        return Err(WorkerError::new(
            WorkerErrorKind::Configuration,
            format!(
                "{label} must point to a regular file: {}. Set {override_env}.",
                canonical.display()
            ),
        ));
    }
    Ok(canonical)
}

fn duration_from_env(name: &str, default: Duration) -> Duration {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .map(Duration::from_millis)
        .unwrap_or(default)
}

#[derive(Debug)]
struct BoundedTail {
    capacity: usize,
    bytes: VecDeque<u8>,
}

impl BoundedTail {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            bytes: VecDeque::with_capacity(capacity),
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        if self.capacity == 0 {
            return;
        }
        if chunk.len() >= self.capacity {
            self.bytes.clear();
            self.bytes
                .extend(chunk[chunk.len() - self.capacity..].iter().copied());
            return;
        }
        let overflow = self
            .bytes
            .len()
            .saturating_add(chunk.len())
            .saturating_sub(self.capacity);
        self.bytes.drain(..overflow);
        self.bytes.extend(chunk.iter().copied());
    }

    fn to_lossy_string(&self) -> String {
        let bytes = self.bytes.iter().copied().collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn with_tail<T>(
    tail: &Arc<StdMutex<BoundedTail>>,
    operation: impl FnOnce(&mut BoundedTail) -> T,
) -> T {
    match tail.lock() {
        Ok(mut guard) => operation(&mut guard),
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            operation(&mut guard)
        }
    }
}

fn stderr_snapshot(tail: &Arc<StdMutex<BoundedTail>>) -> String {
    with_tail(tail, |bounded| bounded.to_lossy_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::io::{duplex, AsyncWriteExt};

    struct MockHarness {
        supervisor: WorkerSupervisor,
        write_rx: mpsc::Receiver<Vec<u8>>,
        kill_rx: mpsc::Receiver<()>,
        _exit_tx: watch::Sender<Option<ExitReport>>,
        generation: u64,
    }

    async fn mock_supervisor() -> MockHarness {
        let supervisor = WorkerSupervisor::new();
        let generation = 7;
        let (write_tx, write_rx) = mpsc::channel(WRITE_QUEUE_CAPACITY);
        let (kill_tx, kill_rx) = mpsc::channel(KILL_QUEUE_CAPACITY);
        let (exit_tx, exit_rx) = watch::channel(None);
        supervisor.inner.state.lock().await.active = Some(ProcessHandle {
            generation,
            write_tx,
            kill_tx,
            exit_rx,
            expected_exit: Arc::new(AtomicBool::new(false)),
            stderr_tail: Arc::new(StdMutex::new(BoundedTail::new(64))),
            ready: true,
        });
        MockHarness {
            supervisor,
            write_rx,
            kill_rx,
            _exit_tx: exit_tx,
            generation,
        }
    }

    fn response_line(request_id: &str, response_type: &str, marker: &str) -> String {
        serde_json::json!({
            "schemaVersion": SCHEMA_VERSION,
            "requestId": request_id,
            "timestamp": "2026-07-22T12:00:00Z",
            "type": response_type,
            "payload": {"marker": marker}
        })
        .to_string()
    }

    fn accepted_job_start_line(request_id: &str, job_id: &str) -> String {
        serde_json::json!({
            "schemaVersion": SCHEMA_VERSION,
            "requestId": request_id,
            "timestamp": "2026-07-22T12:00:00Z",
            "type": "command.accepted",
            "payload": {
                "jobId": job_id,
                "status": "queued"
            }
        })
        .to_string()
    }

    fn event_line(event_id: &str, job_id: &str, sequence: u64, event_type: &str) -> String {
        serde_json::json!({
            "schemaVersion": SCHEMA_VERSION,
            "eventId": event_id,
            "jobId": job_id,
            "sequence": sequence,
            "timestamp": "2026-07-22T12:00:00Z",
            "type": event_type,
            "payload": {}
        })
        .to_string()
    }

    fn request_id(line: &[u8]) -> String {
        let value: Value = serde_json::from_slice(line).expect("valid outbound JSONL");
        value["requestId"].as_str().expect("requestId").to_owned()
    }

    fn job_start_payload(job_id: &str) -> Map<String, Value> {
        Map::from_iter([("jobId".to_owned(), Value::String(job_id.to_owned()))])
    }

    fn temporary_test_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        env::temp_dir().join(format!(
            "mts-worker-supervisor-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn write_runtime_marker(root: &Path) {
        let marker = runtime_root_marker(root);
        fs::create_dir_all(marker.parent().expect("backend parent")).expect("create backend");
        fs::write(marker, b"# packaged worker\n").expect("write worker marker");
    }

    #[test]
    fn installed_executable_runtime_root_wins_over_build_machine_path() {
        let root = temporary_test_root("installed-root");
        let installed = root.join("installed");
        let development = root.join("development");
        fs::create_dir_all(&installed).expect("create installed root");
        write_runtime_marker(&installed);
        let executable = installed.join("media-transcribe-studio.exe");
        fs::write(&executable, b"fixture").expect("write executable");

        let resolved = resolve_runtime_root_from(&executable, &development, None)
            .expect("installed payload should resolve");
        assert_eq!(
            resolved,
            fs::canonicalize(&installed).expect("canonical root")
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn adjacent_resources_payload_is_used_when_executable_directory_is_thin() {
        let root = temporary_test_root("resources-root");
        let executable_root = root.join("release").join("app");
        let resources_root = root.join("release").join("resources");
        let development = root.join("development");
        fs::create_dir_all(&executable_root).expect("create executable root");
        write_runtime_marker(&resources_root);
        let executable = executable_root.join("media-transcribe-studio");
        fs::write(&executable, b"fixture").expect("write executable");

        let resolved = resolve_runtime_root_from(&executable, &development, None)
            .expect("adjacent resources payload should resolve");
        assert_eq!(
            resolved,
            fs::canonicalize(&resources_root).expect("canonical resources root")
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn macos_contents_resources_runtime_payload_is_discovered() {
        let root = temporary_test_root("macos-resources-root");
        let executable_root = root
            .join("MediaTranscribe Studio.app")
            .join("Contents")
            .join("MacOS");
        let resources_root = root
            .join("MediaTranscribe Studio.app")
            .join("Contents")
            .join("Resources")
            .join("mts-runtime");
        let development = root.join("development");
        fs::create_dir_all(&executable_root).expect("create app executable root");
        write_runtime_marker(&resources_root);
        let executable = executable_root.join("media-transcribe-studio");
        fs::write(&executable, b"fixture").expect("write executable");

        let resolved = resolve_runtime_root_from(&executable, &development, None)
            .expect("Contents/Resources runtime payload should resolve");
        assert_eq!(
            resolved,
            fs::canonicalize(&resources_root).expect("canonical macOS runtime root")
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn tauri_linux_resource_directory_runtime_payload_is_discovered() {
        let root = temporary_test_root("linux-tauri-resources-root");
        let executable_root = root.join("AppDir").join("usr").join("bin");
        let resource_directory = root
            .join("AppDir")
            .join("usr")
            .join("lib")
            .join("media-transcribe-studio");
        let runtime_root = resource_directory.join("mts-runtime");
        let development = root.join("development");
        fs::create_dir_all(&executable_root).expect("create executable root");
        write_runtime_marker(&runtime_root);
        let executable = executable_root.join("media-transcribe-studio");
        fs::write(&executable, b"fixture").expect("write executable");

        let resolved =
            resolve_runtime_root_from(&executable, &development, Some(&resource_directory))
                .expect("Tauri Linux resource directory should resolve");
        assert_eq!(
            resolved,
            fs::canonicalize(&runtime_root).expect("canonical Linux runtime root")
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn production_config_prefers_operator_config_then_packaged_example() {
        let root = temporary_test_root("config");
        let local = root.join("local-app-data");
        write_runtime_marker(&root);
        let example = root.join("production.config.example.json");
        fs::write(&example, b"{}\n").expect("write example");

        let fallback = resolve_production_config_from(&root, Some(&local))
            .expect("packaged example should resolve");
        assert_eq!(
            fallback,
            fs::canonicalize(&example).expect("canonical example")
        );

        let operator = local
            .join("MediaTranscribeStudio")
            .join("config")
            .join("production.config.json");
        fs::create_dir_all(operator.parent().expect("operator config parent"))
            .expect("create operator config directory");
        fs::write(&operator, b"{}\n").expect("write operator config");
        let selected = resolve_production_config_from(&root, Some(&local))
            .expect("operator config should resolve");
        assert_eq!(
            selected,
            fs::canonicalize(&operator).expect("canonical operator")
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn worker_python_hint_survives_finder_style_environment() {
        let root = temporary_test_root("python-hint");
        let runtime = root.join("runtime-root");
        let config = root.join("user-data").join("config");
        let python = root.join("external-python").join("bin").join("python3");
        write_runtime_marker(&runtime);
        fs::create_dir_all(&config).expect("create config directory");
        fs::create_dir_all(python.parent().expect("python parent"))
            .expect("create python directory");
        fs::write(&python, b"fixture python\n").expect("write python fixture");
        fs::write(
            config.join(PYTHON_HINT_FILE),
            format!("{}\n", python.display()),
        )
        .expect("write Python hint");

        let resolved = resolve_worker_python(&runtime, Some(&config))
            .expect("user-owned Python hint should resolve");
        assert_eq!(
            resolved,
            fs::canonicalize(&python).expect("canonical Python")
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn worker_command_clears_packager_python_environment() {
        let mut command = Command::new("python3");
        command
            .env("PYTHONHOME", "/tmp/appdir/usr")
            .env("PYTHONPATH", "/tmp/appdir/usr/share/pyshared")
            .env("MTS_TEST_SENTINEL", "preserved");

        sanitize_python_environment(&mut command);

        let mut python_home_removed = false;
        let mut python_path_removed = false;
        let mut sentinel_preserved = false;
        for (name, value) in command.as_std().get_envs() {
            if name == std::ffi::OsStr::new("PYTHONHOME") {
                python_home_removed = value.is_none();
            } else if name == std::ffi::OsStr::new("PYTHONPATH") {
                python_path_removed = value.is_none();
            } else if name == std::ffi::OsStr::new("MTS_TEST_SENTINEL") {
                sentinel_preserved = value == Some(std::ffi::OsStr::new("preserved"));
            }
        }
        assert!(python_home_removed);
        assert!(python_path_removed);
        assert!(sentinel_preserved);
    }

    #[test]
    fn parses_exact_control_and_event_envelopes() {
        let control = response_line("req-1", "command.completed", "health");
        assert!(matches!(
            parse_output_line(&control).expect("control"),
            OutputEnvelope::Control(_)
        ));
        let event = serde_json::json!({
            "schemaVersion": SCHEMA_VERSION,
            "eventId": "evt-1",
            "jobId": "job-1",
            "sequence": 3,
            "timestamp": "2026-07-22T12:00:00Z",
            "type": "stage.progress",
            "payload": {"progress": 20}
        })
        .to_string();
        assert!(matches!(
            parse_output_line(&event).expect("event"),
            OutputEnvelope::Event(_)
        ));
    }

    #[test]
    fn rejects_unknown_event_type() {
        let event = serde_json::json!({
            "schemaVersion": SCHEMA_VERSION,
            "eventId": "evt-unknown",
            "jobId": "job-1",
            "sequence": 4,
            "timestamp": "2026-07-22T12:00:00Z",
            "type": "review.decision.unknown",
            "payload": {}
        })
        .to_string();

        let error = parse_output_line(&event).expect_err("unknown event type must fail closed");
        assert_eq!(error.kind, WorkerErrorKind::Protocol);
        assert!(error.message.contains("review.decision.unknown"));
    }

    #[test]
    fn rejects_wrong_version_unknown_fields_and_duplicate_fields() {
        let wrong_version =
            response_line("req-1", "command.completed", "ok").replace(SCHEMA_VERSION, "2.0.0");
        assert_eq!(
            parse_output_line(&wrong_version)
                .expect_err("version must fail")
                .kind,
            WorkerErrorKind::Protocol
        );

        let unknown = serde_json::json!({
            "schemaVersion": SCHEMA_VERSION,
            "requestId": "req-1",
            "timestamp": "2026-07-22T12:00:00Z",
            "type": "command.completed",
            "payload": {},
            "extra": true
        })
        .to_string();
        assert!(parse_output_line(&unknown).is_err());

        let duplicate = format!(
            "{{\"schemaVersion\":\"{SCHEMA_VERSION}\",\"requestId\":\"req-1\",\
             \"requestId\":\"req-2\",\"timestamp\":\"2026-07-22T12:00:00Z\",\
             \"type\":\"command.completed\",\"payload\":{{}}}}"
        );
        assert!(parse_output_line(&duplicate).is_err());
    }

    #[tokio::test]
    async fn bounded_reader_rejects_invalid_utf8_oversize_and_partial_eof() {
        let (mut writer, reader) = duplex(64);
        writer
            .write_all(&[0xff, b'\n'])
            .await
            .expect("write invalid UTF-8");
        drop(writer);
        let mut reader = BufReader::new(reader);
        assert_eq!(
            read_bounded_utf8_line(&mut reader, 16)
                .await
                .expect_err("invalid UTF-8")
                .kind,
            WorkerErrorKind::Protocol
        );

        let (mut writer, reader) = duplex(64);
        writer
            .write_all(b"12345\n")
            .await
            .expect("write oversized line");
        drop(writer);
        let mut reader = BufReader::new(reader);
        assert!(read_bounded_utf8_line(&mut reader, 4).await.is_err());

        let (mut writer, reader) = duplex(64);
        writer
            .write_all(b"{\"partial\":true}")
            .await
            .expect("write partial line");
        drop(writer);
        let mut reader = BufReader::new(reader);
        assert!(read_bounded_utf8_line(&mut reader, 64).await.is_err());
    }

    #[test]
    fn stderr_tail_is_strictly_bounded_and_keeps_newest_bytes() {
        let mut tail = BoundedTail::new(5);
        tail.push(b"abc");
        tail.push(b"defg");
        assert_eq!(tail.to_lossy_string(), "cdefg");
        tail.push(b"0123456789");
        assert_eq!(tail.to_lossy_string(), "56789");
    }

    #[tokio::test]
    async fn correlates_concurrent_requests_with_out_of_order_responses() {
        let mut harness = mock_supervisor().await;
        let first_inner = Arc::clone(&harness.supervisor.inner);
        let second_inner = Arc::clone(&harness.supervisor.inner);
        let generation = harness.generation;
        let first = tokio::spawn(async move {
            first_inner
                .request_generation(
                    generation,
                    "job.status",
                    Map::new(),
                    ResponseKind::Completed,
                    Duration::from_secs(1),
                )
                .await
        });
        let second = tokio::spawn(async move {
            second_inner
                .request_generation(
                    generation,
                    "review.queue",
                    Map::new(),
                    ResponseKind::Completed,
                    Duration::from_secs(1),
                )
                .await
        });

        let first_line = harness.write_rx.recv().await.expect("first write");
        let second_line = harness.write_rx.recv().await.expect("second write");
        let first_id = request_id(&first_line);
        let second_id = request_id(&second_line);
        harness
            .supervisor
            .inner
            .handle_output_line(
                generation,
                &response_line(&second_id, "command.completed", "second"),
            )
            .await
            .expect("second response");
        harness
            .supervisor
            .inner
            .handle_output_line(
                generation,
                &response_line(&first_id, "command.completed", "first"),
            )
            .await
            .expect("first response");

        let first_response = first.await.expect("first task").expect("first result");
        let second_response = second.await.expect("second task").expect("second result");
        assert_eq!(first_response.payload["marker"], "first");
        assert_eq!(second_response.payload["marker"], "second");
    }

    #[tokio::test]
    async fn terminal_event_retires_only_its_job_route() {
        let mut harness = mock_supervisor().await;
        for job_id in ["job-a", "job-b"] {
            let inner = Arc::clone(&harness.supervisor.inner);
            let generation = harness.generation;
            let owned_job_id = job_id.to_owned();
            let request = tokio::spawn(async move {
                inner
                    .request_generation(
                        generation,
                        "job.start",
                        job_start_payload(&owned_job_id),
                        ResponseKind::Accepted,
                        Duration::from_secs(1),
                    )
                    .await
            });
            let line = harness.write_rx.recv().await.expect("job.start write");
            let id = request_id(&line);
            harness
                .supervisor
                .inner
                .handle_output_line(generation, &accepted_job_start_line(&id, job_id))
                .await
                .expect("job.start acceptance");
            request
                .await
                .expect("job.start request task")
                .expect("job.start response");
        }

        assert!(harness.supervisor.routes_job_events("job-a").await);
        assert!(harness.supervisor.routes_job_events("job-b").await);
        harness
            .supervisor
            .inner
            .handle_output_line(
                harness.generation,
                &event_line("evt-a-terminal", "job-a", 0, "job.completed"),
            )
            .await
            .expect("terminal event");

        assert!(!harness.supervisor.routes_job_events("job-a").await);
        assert!(harness.supervisor.routes_job_events("job-b").await);
        let late_error = harness
            .supervisor
            .inner
            .handle_output_line(
                harness.generation,
                &event_line("evt-a-late", "job-a", 1, "warning"),
            )
            .await
            .expect_err("events after a terminal event must be rejected");
        assert_eq!(late_error.kind, WorkerErrorKind::Protocol);
        assert!(late_error.message.contains("unregistered jobId"));

        harness
            .supervisor
            .inner
            .handle_output_line(
                harness.generation,
                &event_line("evt-b-progress", "job-b", 0, "warning"),
            )
            .await
            .expect("the other job route must remain usable");
        assert!(harness.supervisor.routes_job_events("job-b").await);
    }

    #[tokio::test]
    async fn unknown_request_id_is_a_protocol_violation() {
        let harness = mock_supervisor().await;
        let error = harness
            .supervisor
            .inner
            .handle_output_line(
                harness.generation,
                &response_line("req-unknown", "command.completed", "bad"),
            )
            .await
            .expect_err("unknown requestId must fail");
        assert_eq!(error.kind, WorkerErrorKind::Protocol);
    }

    #[tokio::test]
    async fn timeout_fails_generation_and_never_replays_job_start() {
        let mut harness = mock_supervisor().await;
        let result_future = harness.supervisor.inner.request_generation(
            harness.generation,
            "job.start",
            job_start_payload("job-timeout"),
            ResponseKind::Accepted,
            Duration::from_millis(20),
        );
        let (_, result) = tokio::join!(harness.write_rx.recv(), result_future);
        let error = result.expect_err("must timeout");
        assert_eq!(error.kind, WorkerErrorKind::Timeout);
        assert!(error.message.contains("never replay"));
        assert!(harness.kill_rx.recv().await.is_some());
        assert!(matches!(
            harness.write_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected)
        ));
        assert!(harness.supervisor.inner.state.lock().await.active.is_none());
    }

    #[tokio::test]
    async fn generation_failure_rejects_all_pending_requests() {
        let mut harness = mock_supervisor().await;
        let first_inner = Arc::clone(&harness.supervisor.inner);
        let second_inner = Arc::clone(&harness.supervisor.inner);
        let generation = harness.generation;
        let first = tokio::spawn(async move {
            first_inner
                .request_generation(
                    generation,
                    "job.start",
                    job_start_payload("job-generation-failure"),
                    ResponseKind::Accepted,
                    Duration::from_secs(1),
                )
                .await
        });
        let second = tokio::spawn(async move {
            second_inner
                .request_generation(
                    generation,
                    "job.cancel",
                    Map::new(),
                    ResponseKind::Accepted,
                    Duration::from_secs(1),
                )
                .await
        });
        harness.write_rx.recv().await.expect("first write");
        harness.write_rx.recv().await.expect("second write");
        harness
            .supervisor
            .inner
            .fail_generation(
                generation,
                WorkerError::new(WorkerErrorKind::Crashed, "controlled crash"),
            )
            .await;

        let first_error = first.await.expect("first task").expect_err("first fails");
        let second_error = second
            .await
            .expect("second task")
            .expect_err("second fails");
        assert_eq!(first_error.kind, WorkerErrorKind::Crashed);
        assert_eq!(second_error.kind, WorkerErrorKind::Crashed);
        assert!(first_error.message.contains("never replay"));
    }

    #[tokio::test]
    async fn rejected_response_is_explicit_and_does_not_fake_success() {
        let mut harness = mock_supervisor().await;
        let inner = Arc::clone(&harness.supervisor.inner);
        let generation = harness.generation;
        let request = tokio::spawn(async move {
            inner
                .request_generation(
                    generation,
                    "job.cancel",
                    Map::new(),
                    ResponseKind::Accepted,
                    Duration::from_secs(1),
                )
                .await
        });
        let line = harness.write_rx.recv().await.expect("write");
        let id = request_id(&line);
        let rejected = serde_json::json!({
            "schemaVersion": SCHEMA_VERSION,
            "requestId": id,
            "timestamp": "2026-07-22T12:00:00Z",
            "type": "command.rejected",
            "payload": {
                "code": "JOB_NOT_FOUND",
                "message": "missing",
                "retryable": false,
                "details": {}
            }
        })
        .to_string();
        harness
            .supervisor
            .inner
            .handle_output_line(generation, &rejected)
            .await
            .expect("valid rejection");
        let error = request.await.expect("request task").expect_err("rejected");
        assert_eq!(error.kind, WorkerErrorKind::Rejected);
        assert!(harness.supervisor.inner.state.lock().await.active.is_some());
    }

    #[test]
    fn accepts_worker_backed_human_mutation_command_types() {
        assert!(validate_command_type("review.submit").is_ok());
        assert!(validate_command_type("speaker.rename").is_ok());
        assert!(validate_command_type("review.submit.legacy").is_err());
        assert!(validate_command_type("speaker.update").is_err());
    }

    #[test]
    fn mutation_failures_report_unknown_outcome_and_never_replay() {
        for command_type in ["review.submit", "speaker.rename"] {
            let error = WorkerError::new(WorkerErrorKind::Timeout, "controlled timeout")
                .for_command(command_type);
            assert!(error.message.contains("outcome may be unknown"));
            assert!(error.message.contains("failed closed"));
            assert!(error.message.contains("never replay"));
        }
    }
}
