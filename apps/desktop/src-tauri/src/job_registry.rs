//! Runtime-independent safety primitives for managing multiple jobs.
//!
//! This module intentionally does not know about Tauri commands, windows, or
//! worker processes. It provides the state and routing guarantees required
//! before the existing single-job production path can be migrated.

use std::{
    collections::HashMap,
    error::Error,
    fmt,
    sync::{Arc, Mutex, MutexGuard},
};

pub(crate) const DEFAULT_DISPATCH_LIMIT: usize = 1;
pub(crate) const MIN_DISPATCH_LIMIT: usize = 1;
pub(crate) const MAX_DISPATCH_LIMIT: usize = 4;
const MAX_JOB_ID_BYTES: usize = 256;
const MAX_IDEMPOTENCY_KEY_BYTES: usize = 1_024;

#[derive(Clone)]
struct RegistryIdentity(Arc<()>);

impl RegistryIdentity {
    fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl fmt::Debug for RegistryIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RegistryIdentity(<opaque>)")
    }
}

impl PartialEq for RegistryIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for RegistryIdentity {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct JobId(String);

impl JobId {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, RegistryError> {
        let value = value.into();
        if !is_valid_identifier(&value, MAX_JOB_ID_BYTES) {
            return Err(RegistryError::InvalidJobId);
        }
        Ok(Self(value))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct IdempotencyKey(String);

impl IdempotencyKey {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, RegistryError> {
        let value = value.into();
        if !is_valid_identifier(&value, MAX_IDEMPOTENCY_KEY_BYTES) {
            return Err(RegistryError::InvalidIdempotencyKey);
        }
        Ok(Self(value))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for IdempotencyKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegistryJobStatus {
    Registered,
    Queued,
    Running,
    ReviewRequired,
    Completed,
    Failed,
    Cancelled,
}

impl RegistryJobStatus {
    fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    fn can_transition_to(self, next: Self) -> bool {
        match self {
            Self::Registered => matches!(next, Self::Queued | Self::Failed | Self::Cancelled),
            Self::Queued => matches!(next, Self::Running | Self::Failed | Self::Cancelled),
            Self::Running => matches!(
                next,
                Self::ReviewRequired | Self::Completed | Self::Failed | Self::Cancelled
            ),
            Self::ReviewRequired => matches!(
                next,
                Self::Queued | Self::Running | Self::Completed | Self::Failed | Self::Cancelled
            ),
            Self::Completed | Self::Failed | Self::Cancelled => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JobToken {
    registry_identity: RegistryIdentity,
    job_id: JobId,
    revision: u64,
}

impl JobToken {
    pub(crate) fn job_id(&self) -> &JobId {
        &self.job_id
    }

    #[cfg(test)]
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EventTargetToken {
    registry_identity: RegistryIdentity,
    job_id: JobId,
    revision: u64,
}

#[cfg(test)]
impl EventTargetToken {
    pub(crate) fn job_id(&self) -> &JobId {
        &self.job_id
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProjectionToken {
    registry_identity: RegistryIdentity,
    job_id: JobId,
    epoch: u64,
}

#[cfg(test)]
impl ProjectionToken {
    pub(crate) fn job_id(&self) -> &JobId {
        &self.job_id
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RegisterOutcome {
    Created { job_id: JobId, token: JobToken },
    Existing { job_id: JobId, token: JobToken },
}

impl RegisterOutcome {
    pub(crate) fn job_id(&self) -> &JobId {
        match self {
            Self::Created { job_id, .. } | Self::Existing { job_id, .. } => job_id,
        }
    }

    #[cfg(test)]
    pub(crate) fn token(&self) -> &JobToken {
        match self {
            Self::Created { token, .. } | Self::Existing { token, .. } => token,
        }
    }

    #[cfg(test)]
    pub(crate) fn was_created(&self) -> bool {
        matches!(self, Self::Created { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JobSnapshot<S> {
    pub(crate) job_id: JobId,
    pub(crate) status: RegistryJobStatus,
    pub(crate) revision: u64,
    pub(crate) runtime_state: S,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DispatcherSnapshot {
    pub(crate) limit: usize,
    pub(crate) in_use: usize,
    pub(crate) available: usize,
    pub(crate) active_job_ids: Vec<JobId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DispatchPermitToken {
    registry_identity: RegistryIdentity,
    permit_id: u64,
    job_id: JobId,
}

impl DispatchPermitToken {
    #[cfg(test)]
    pub(crate) fn permit_id(&self) -> u64 {
        self.permit_id
    }

    #[cfg(test)]
    pub(crate) fn job_id(&self) -> &JobId {
        &self.job_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RegistryError {
    InvalidJobId,
    InvalidIdempotencyKey,
    InvalidDispatcherLimit {
        requested: usize,
        minimum: usize,
        maximum: usize,
    },
    DuplicateJob {
        job_id: JobId,
    },
    UnknownJob {
        job_id: JobId,
    },
    ForeignToken {
        token_kind: &'static str,
        job_id: JobId,
    },
    StaleJobToken {
        job_id: JobId,
        token_revision: u64,
        current_revision: u64,
    },
    #[cfg(test)]
    EventTargetMismatch {
        event_job_id: JobId,
        target_job_id: JobId,
    },
    InvalidStatusTransition {
        job_id: JobId,
        from: RegistryJobStatus,
        to: RegistryJobStatus,
    },
    #[cfg(test)]
    StaleProjectionToken {
        token_job_id: JobId,
        token_epoch: u64,
        active_job_id: Option<JobId>,
        active_epoch: u64,
    },
    DispatcherSaturated {
        limit: usize,
        in_use: usize,
    },
    JobAlreadyDispatched {
        job_id: JobId,
    },
    StaleDispatchPermit {
        permit_id: u64,
        job_id: JobId,
    },
    SequenceExhausted {
        sequence: &'static str,
    },
    StatePoisoned {
        component: &'static str,
    },
    InvariantViolation {
        message: String,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJobId => write!(
                formatter,
                "job id must be non-empty, trimmed, free of control characters, and at most {MAX_JOB_ID_BYTES} bytes"
            ),
            Self::InvalidIdempotencyKey => {
                write!(
                    formatter,
                    "idempotency key must be non-empty, trimmed, free of control characters, and at most {MAX_IDEMPOTENCY_KEY_BYTES} bytes"
                )
            }
            Self::InvalidDispatcherLimit {
                requested,
                minimum,
                maximum,
            } => write!(
                formatter,
                "dispatcher limit {requested} is outside the supported range {minimum}..={maximum}"
            ),
            Self::DuplicateJob { job_id } => write!(formatter, "job {job_id} already exists"),
            Self::UnknownJob { job_id } => write!(formatter, "job {job_id} is not registered"),
            Self::ForeignToken { token_kind, job_id } => write!(
                formatter,
                "{token_kind} token for job {job_id} belongs to a different registry instance"
            ),
            Self::StaleJobToken {
                job_id,
                token_revision,
                current_revision,
            } => write!(
                formatter,
                "job {job_id} token revision {token_revision} is stale; current revision is {current_revision}"
            ),
            #[cfg(test)]
            Self::EventTargetMismatch {
                event_job_id,
                target_job_id,
            } => write!(
                formatter,
                "event for job {event_job_id} cannot mutate target job {target_job_id}"
            ),
            Self::InvalidStatusTransition { job_id, from, to } => write!(
                formatter,
                "job {job_id} cannot transition from {from:?} to {to:?}"
            ),
            #[cfg(test)]
            Self::StaleProjectionToken {
                token_job_id,
                token_epoch,
                active_job_id,
                active_epoch,
            } => write!(
                formatter,
                "projection token for job {token_job_id} at epoch {token_epoch} is stale; active projection is {:?} at epoch {active_epoch}",
                active_job_id.as_ref().map(JobId::as_str)
            ),
            Self::DispatcherSaturated { limit, in_use } => write!(
                formatter,
                "dispatcher is saturated with {in_use} active permits (limit {limit})"
            ),
            Self::JobAlreadyDispatched { job_id } => {
                write!(formatter, "job {job_id} already owns a dispatcher permit")
            }
            Self::StaleDispatchPermit { permit_id, job_id } => write!(
                formatter,
                "dispatcher permit {permit_id} for job {job_id} is stale or already released"
            ),
            Self::SequenceExhausted { sequence } => {
                write!(formatter, "{sequence} sequence is exhausted")
            }
            Self::StatePoisoned { component } => {
                write!(formatter, "{component} state lock is poisoned")
            }
            Self::InvariantViolation { message } => {
                write!(formatter, "registry invariant violated: {message}")
            }
        }
    }
}

impl Error for RegistryError {}

struct JobEntry<S> {
    runtime_state: S,
    status: RegistryJobStatus,
    revision: u64,
}

#[derive(Debug, Clone)]
struct ProjectionSelection {
    job_id: JobId,
    #[cfg(test)]
    epoch: u64,
}

struct RegistryInner<S> {
    jobs: HashMap<JobId, JobEntry<S>>,
    idempotency: HashMap<IdempotencyKey, JobId>,
    projection: Option<ProjectionSelection>,
    #[cfg(test)]
    projection_epoch: u64,
}

impl<S> Default for RegistryInner<S> {
    fn default() -> Self {
        Self {
            jobs: HashMap::new(),
            idempotency: HashMap::new(),
            projection: None,
            #[cfg(test)]
            projection_epoch: 0,
        }
    }
}

struct DispatcherState {
    limit: usize,
    next_permit_id: u64,
    active_by_permit: HashMap<u64, JobId>,
    active_by_job: HashMap<JobId, u64>,
}

impl DispatcherState {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            next_permit_id: 1,
            active_by_permit: HashMap::new(),
            active_by_job: HashMap::new(),
        }
    }

    fn validate(&self) -> Result<(), RegistryError> {
        if !(MIN_DISPATCH_LIMIT..=MAX_DISPATCH_LIMIT).contains(&self.limit) {
            return Err(RegistryError::InvariantViolation {
                message: format!(
                    "dispatcher limit {} is outside {}..={}",
                    self.limit, MIN_DISPATCH_LIMIT, MAX_DISPATCH_LIMIT
                ),
            });
        }
        if self.next_permit_id == 0 {
            return Err(RegistryError::InvariantViolation {
                message: "dispatcher next permit id must never be zero".to_owned(),
            });
        }
        if self.active_by_permit.len() != self.active_by_job.len() {
            return Err(RegistryError::InvariantViolation {
                message: format!(
                    "dispatcher indexes disagree: {} permit entries versus {} job entries",
                    self.active_by_permit.len(),
                    self.active_by_job.len()
                ),
            });
        }
        if self.active_by_permit.len() > self.limit {
            return Err(RegistryError::InvariantViolation {
                message: format!(
                    "dispatcher tracks {} active permits above limit {}",
                    self.active_by_permit.len(),
                    self.limit
                ),
            });
        }
        for (permit_id, job_id) in &self.active_by_permit {
            if *permit_id == 0 || *permit_id >= self.next_permit_id {
                return Err(RegistryError::InvariantViolation {
                    message: format!(
                        "dispatcher active permit {permit_id} is outside the allocated sequence"
                    ),
                });
            }
            if self.active_by_job.get(job_id) != Some(permit_id) {
                return Err(RegistryError::InvariantViolation {
                    message: format!(
                        "dispatcher permit {permit_id} for job {job_id} has no matching job index"
                    ),
                });
            }
        }
        Ok(())
    }
}

struct DispatcherCore {
    registry_identity: RegistryIdentity,
    state: Mutex<DispatcherState>,
}

impl DispatcherCore {
    fn lock(&self) -> Result<MutexGuard<'_, DispatcherState>, RegistryError> {
        self.state.lock().map_err(|_| RegistryError::StatePoisoned {
            component: "dispatcher",
        })
    }

    fn acquire(self: &Arc<Self>, job_id: JobId) -> Result<DispatchPermit, RegistryError> {
        let token = {
            let mut state = self.lock()?;
            state.validate()?;
            if state.active_by_job.contains_key(&job_id) {
                return Err(RegistryError::JobAlreadyDispatched { job_id });
            }
            let in_use = state.active_by_permit.len();
            if in_use >= state.limit {
                return Err(RegistryError::DispatcherSaturated {
                    limit: state.limit,
                    in_use,
                });
            }

            let permit_id = state.next_permit_id;
            state.next_permit_id =
                permit_id
                    .checked_add(1)
                    .ok_or(RegistryError::SequenceExhausted {
                        sequence: "dispatcher permit",
                    })?;
            state.active_by_permit.insert(permit_id, job_id.clone());
            state.active_by_job.insert(job_id.clone(), permit_id);

            DispatchPermitToken {
                registry_identity: self.registry_identity.clone(),
                permit_id,
                job_id,
            }
        };

        Ok(DispatchPermit {
            core: Arc::clone(self),
            token,
            released: false,
        })
    }

    fn release(&self, token: &DispatchPermitToken) -> Result<(), RegistryError> {
        if token.registry_identity != self.registry_identity {
            return Err(RegistryError::ForeignToken {
                token_kind: "dispatch permit",
                job_id: token.job_id.clone(),
            });
        }
        let mut state = self.lock()?;
        state.validate()?;
        let Some(active_job_id) = state.active_by_permit.get(&token.permit_id) else {
            return Err(RegistryError::StaleDispatchPermit {
                permit_id: token.permit_id,
                job_id: token.job_id.clone(),
            });
        };
        if active_job_id != &token.job_id {
            return Err(RegistryError::StaleDispatchPermit {
                permit_id: token.permit_id,
                job_id: token.job_id.clone(),
            });
        }
        if state.active_by_job.get(&token.job_id) != Some(&token.permit_id) {
            return Err(RegistryError::InvariantViolation {
                message: format!(
                    "dispatcher job {} does not point back to permit {}",
                    token.job_id, token.permit_id
                ),
            });
        }

        state.active_by_permit.remove(&token.permit_id);
        state.active_by_job.remove(&token.job_id);
        Ok(())
    }

    fn snapshot(&self) -> Result<DispatcherSnapshot, RegistryError> {
        let state = self.lock()?;
        state.validate()?;
        let mut active_permits = state
            .active_by_permit
            .iter()
            .map(|(permit_id, job_id)| (*permit_id, job_id.clone()))
            .collect::<Vec<_>>();
        active_permits.sort_unstable_by_key(|(permit_id, _)| *permit_id);
        let active_job_ids = active_permits
            .into_iter()
            .map(|(_, job_id)| job_id)
            .collect::<Vec<_>>();
        let in_use = active_job_ids.len();

        Ok(DispatcherSnapshot {
            limit: state.limit,
            in_use,
            available: state.limit.saturating_sub(in_use),
            active_job_ids,
        })
    }
}

#[must_use = "the permit must be retained for the entire dispatched job lifetime"]
pub(crate) struct DispatchPermit {
    core: Arc<DispatcherCore>,
    token: DispatchPermitToken,
    released: bool,
}

impl DispatchPermit {
    #[cfg(test)]
    pub(crate) fn token(&self) -> &DispatchPermitToken {
        &self.token
    }

    pub(crate) fn release(mut self) -> Result<(), RegistryError> {
        let result = self.core.release(&self.token);
        self.released = true;
        result
    }
}

impl Drop for DispatchPermit {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        let _ = self.core.release(&self.token);
        self.released = true;
    }
}

pub(crate) struct JobRegistry<S> {
    registry_identity: RegistryIdentity,
    inner: Mutex<RegistryInner<S>>,
    dispatcher: Arc<DispatcherCore>,
}

impl<S> Default for JobRegistry<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> JobRegistry<S> {
    pub(crate) fn new() -> Self {
        let registry_identity = RegistryIdentity::new();
        Self {
            registry_identity: registry_identity.clone(),
            inner: Mutex::new(RegistryInner::default()),
            dispatcher: Arc::new(DispatcherCore {
                registry_identity,
                state: Mutex::new(DispatcherState::new(DEFAULT_DISPATCH_LIMIT)),
            }),
        }
    }

    pub(crate) fn with_dispatch_limit(limit: usize) -> Result<Self, RegistryError> {
        if !(MIN_DISPATCH_LIMIT..=MAX_DISPATCH_LIMIT).contains(&limit) {
            return Err(RegistryError::InvalidDispatcherLimit {
                requested: limit,
                minimum: MIN_DISPATCH_LIMIT,
                maximum: MAX_DISPATCH_LIMIT,
            });
        }

        let registry_identity = RegistryIdentity::new();
        Ok(Self {
            registry_identity: registry_identity.clone(),
            inner: Mutex::new(RegistryInner::default()),
            dispatcher: Arc::new(DispatcherCore {
                registry_identity,
                state: Mutex::new(DispatcherState::new(limit)),
            }),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, RegistryInner<S>>, RegistryError> {
        self.inner.lock().map_err(|_| RegistryError::StatePoisoned {
            component: "job registry",
        })
    }

    pub(crate) fn register_or_get(
        &self,
        job_id: JobId,
        idempotency_key: IdempotencyKey,
        runtime_state: S,
    ) -> Result<RegisterOutcome, RegistryError> {
        let mut inner = self.lock()?;

        if let Some(existing_job_id) = inner.idempotency.get(&idempotency_key).cloned() {
            let existing_revision = inner
                .jobs
                .get(&existing_job_id)
                .map(|entry| entry.revision)
                .ok_or_else(|| RegistryError::InvariantViolation {
                    message: format!("an idempotency key references missing job {existing_job_id}"),
                })?;
            if job_id != existing_job_id && inner.jobs.contains_key(&job_id) {
                return Err(RegistryError::DuplicateJob { job_id });
            }
            let token = JobToken {
                registry_identity: self.registry_identity.clone(),
                job_id: existing_job_id.clone(),
                revision: existing_revision,
            };
            return Ok(RegisterOutcome::Existing {
                job_id: existing_job_id,
                token,
            });
        }

        if inner.jobs.contains_key(&job_id) {
            return Err(RegistryError::DuplicateJob { job_id });
        }

        let token = JobToken {
            registry_identity: self.registry_identity.clone(),
            job_id: job_id.clone(),
            revision: 1,
        };
        inner.jobs.insert(
            job_id.clone(),
            JobEntry {
                runtime_state,
                status: RegistryJobStatus::Registered,
                revision: token.revision,
            },
        );
        inner.idempotency.insert(idempotency_key, job_id.clone());

        Ok(RegisterOutcome::Created { job_id, token })
    }

    #[cfg(test)]
    pub(crate) fn job_count(&self) -> Result<usize, RegistryError> {
        Ok(self.lock()?.jobs.len())
    }

    pub(crate) fn job_ids(&self) -> Result<Vec<JobId>, RegistryError> {
        let inner = self.lock()?;
        let mut job_ids = inner.jobs.keys().cloned().collect::<Vec<_>>();
        job_ids.sort();
        Ok(job_ids)
    }

    #[cfg(test)]
    pub(crate) fn idempotent_job(
        &self,
        idempotency_key: &IdempotencyKey,
    ) -> Result<Option<JobId>, RegistryError> {
        Ok(self.lock()?.idempotency.get(idempotency_key).cloned())
    }

    #[cfg(test)]
    pub(crate) fn token(&self, job_id: &JobId) -> Result<JobToken, RegistryError> {
        let inner = self.lock()?;
        let entry = inner
            .jobs
            .get(job_id)
            .ok_or_else(|| RegistryError::UnknownJob {
                job_id: job_id.clone(),
            })?;
        Ok(JobToken {
            registry_identity: self.registry_identity.clone(),
            job_id: job_id.clone(),
            revision: entry.revision,
        })
    }

    pub(crate) fn status(&self, job_id: &JobId) -> Result<RegistryJobStatus, RegistryError> {
        let inner = self.lock()?;
        inner
            .jobs
            .get(job_id)
            .map(|entry| entry.status)
            .ok_or_else(|| RegistryError::UnknownJob {
                job_id: job_id.clone(),
            })
    }

    pub(crate) fn snapshot(&self, job_id: &JobId) -> Result<JobSnapshot<S>, RegistryError>
    where
        S: Clone,
    {
        let inner = self.lock()?;
        Self::snapshot_from_inner(&inner, job_id)
    }

    pub(crate) fn snapshot_with_token(
        &self,
        job_id: &JobId,
    ) -> Result<(JobSnapshot<S>, JobToken), RegistryError>
    where
        S: Clone,
    {
        let inner = self.lock()?;
        let snapshot = Self::snapshot_from_inner(&inner, job_id)?;
        let token = JobToken {
            registry_identity: self.registry_identity.clone(),
            job_id: job_id.clone(),
            revision: snapshot.revision,
        };
        Ok((snapshot, token))
    }

    #[cfg(test)]
    pub(crate) fn mutate_runtime<R>(
        &self,
        token: &JobToken,
        mutate: impl FnOnce(&mut S) -> R,
    ) -> Result<(R, JobToken), RegistryError> {
        // The callback runs under the registry mutex so the revision check,
        // mutation, and revision advance remain atomic. Callers must keep it
        // short and non-reentrant. A callback panic intentionally poisons the
        // mutex, preventing partially-mutated state from being observed.
        let mut inner = self.lock()?;
        let entry = self.entry_for_token(&mut inner, token)?;
        let next_revision = next_sequence(entry.revision, "job revision")?;
        let result = mutate(&mut entry.runtime_state);
        entry.revision = next_revision;
        let next_token = JobToken {
            registry_identity: self.registry_identity.clone(),
            job_id: token.job_id.clone(),
            revision: entry.revision,
        };
        Ok((result, next_token))
    }

    pub(crate) fn try_mutate_runtime<R, E>(
        &self,
        token: &JobToken,
        mutate: impl FnOnce(&mut S) -> Result<R, E>,
    ) -> Result<Result<(R, JobToken), E>, RegistryError>
    where
        S: Clone,
    {
        let mut inner = self.lock()?;
        let entry = self.entry_for_token(&mut inner, token)?;
        let next_revision = next_sequence(entry.revision, "job revision")?;
        let mut candidate = entry.runtime_state.clone();
        let result = match mutate(&mut candidate) {
            Ok(result) => result,
            Err(error) => return Ok(Err(error)),
        };
        entry.runtime_state = candidate;
        entry.revision = next_revision;
        Ok(Ok((
            result,
            JobToken {
                registry_identity: self.registry_identity.clone(),
                job_id: token.job_id.clone(),
                revision: entry.revision,
            },
        )))
    }

    #[cfg(test)]
    pub(crate) fn transition_status(
        &self,
        token: &JobToken,
        next_status: RegistryJobStatus,
    ) -> Result<JobToken, RegistryError> {
        self.transition_status_with(token, next_status, |_| ())
            .map(|(_, token)| token)
    }

    pub(crate) fn transition_status_with<R>(
        &self,
        token: &JobToken,
        next_status: RegistryJobStatus,
        mutate: impl FnOnce(&mut S) -> R,
    ) -> Result<(R, JobToken), RegistryError> {
        let mut inner = self.lock()?;
        let entry = self.entry_for_token(&mut inner, token)?;
        if entry.status == next_status {
            let next_revision = next_sequence(entry.revision, "job revision")?;
            let result = mutate(&mut entry.runtime_state);
            entry.revision = next_revision;
            return Ok((
                result,
                JobToken {
                    registry_identity: self.registry_identity.clone(),
                    job_id: token.job_id.clone(),
                    revision: entry.revision,
                },
            ));
        }
        if entry.status.is_terminal() || !entry.status.can_transition_to(next_status) {
            return Err(RegistryError::InvalidStatusTransition {
                job_id: token.job_id.clone(),
                from: entry.status,
                to: next_status,
            });
        }

        let next_revision = next_sequence(entry.revision, "job revision")?;
        let result = mutate(&mut entry.runtime_state);
        entry.status = next_status;
        entry.revision = next_revision;
        Ok((
            result,
            JobToken {
                registry_identity: self.registry_identity.clone(),
                job_id: token.job_id.clone(),
                revision: entry.revision,
            },
        ))
    }

    pub(crate) fn try_transition_status_with<R, E>(
        &self,
        token: &JobToken,
        next_status: RegistryJobStatus,
        mutate: impl FnOnce(&mut S) -> Result<R, E>,
    ) -> Result<Result<(R, JobToken), E>, RegistryError>
    where
        S: Clone,
    {
        let mut inner = self.lock()?;
        let entry = self.entry_for_token(&mut inner, token)?;
        if entry.status != next_status
            && (entry.status.is_terminal() || !entry.status.can_transition_to(next_status))
        {
            return Err(RegistryError::InvalidStatusTransition {
                job_id: token.job_id.clone(),
                from: entry.status,
                to: next_status,
            });
        }

        let next_revision = next_sequence(entry.revision, "job revision")?;
        let mut candidate = entry.runtime_state.clone();
        let result = match mutate(&mut candidate) {
            Ok(result) => result,
            Err(error) => return Ok(Err(error)),
        };
        entry.runtime_state = candidate;
        entry.status = next_status;
        entry.revision = next_revision;
        Ok(Ok((
            result,
            JobToken {
                registry_identity: self.registry_identity.clone(),
                job_id: token.job_id.clone(),
                revision: entry.revision,
            },
        )))
    }

    #[cfg(test)]
    pub(crate) fn event_target(
        &self,
        event_job_id: &JobId,
    ) -> Result<EventTargetToken, RegistryError> {
        let inner = self.lock()?;
        let entry = inner
            .jobs
            .get(event_job_id)
            .ok_or_else(|| RegistryError::UnknownJob {
                job_id: event_job_id.clone(),
            })?;
        Ok(EventTargetToken {
            registry_identity: self.registry_identity.clone(),
            job_id: event_job_id.clone(),
            revision: entry.revision,
        })
    }

    #[cfg(test)]
    pub(crate) fn apply_event<R>(
        &self,
        event_job_id: &JobId,
        target: &EventTargetToken,
        apply: impl FnOnce(&mut S) -> R,
    ) -> Result<(R, EventTargetToken), RegistryError> {
        if event_job_id != target.job_id() {
            return Err(RegistryError::EventTargetMismatch {
                event_job_id: event_job_id.clone(),
                target_job_id: target.job_id.clone(),
            });
        }
        if target.registry_identity != self.registry_identity {
            return Err(RegistryError::ForeignToken {
                token_kind: "event target",
                job_id: target.job_id.clone(),
            });
        }

        // As with mutate_runtime, fail closed if caller code panics while the
        // revision-checked event mutation is in progress.
        let mut inner = self.lock()?;
        let entry = inner
            .jobs
            .get_mut(event_job_id)
            .ok_or_else(|| RegistryError::UnknownJob {
                job_id: event_job_id.clone(),
            })?;
        if entry.revision != target.revision {
            return Err(RegistryError::StaleJobToken {
                job_id: event_job_id.clone(),
                token_revision: target.revision,
                current_revision: entry.revision,
            });
        }

        let next_revision = next_sequence(entry.revision, "job revision")?;
        let result = apply(&mut entry.runtime_state);
        entry.revision = next_revision;
        Ok((
            result,
            EventTargetToken {
                registry_identity: self.registry_identity.clone(),
                job_id: event_job_id.clone(),
                revision: entry.revision,
            },
        ))
    }

    #[cfg(test)]
    pub(crate) fn select_projection(
        &self,
        job_id: &JobId,
    ) -> Result<ProjectionToken, RegistryError> {
        let mut inner = self.lock()?;
        if !inner.jobs.contains_key(job_id) {
            return Err(RegistryError::UnknownJob {
                job_id: job_id.clone(),
            });
        }
        inner.projection_epoch = next_sequence(inner.projection_epoch, "projection epoch")?;
        let token = ProjectionToken {
            registry_identity: self.registry_identity.clone(),
            job_id: job_id.clone(),
            epoch: inner.projection_epoch,
        };
        inner.projection = Some(ProjectionSelection {
            job_id: token.job_id.clone(),
            epoch: token.epoch,
        });
        Ok(token)
    }

    #[cfg(not(test))]
    pub(crate) fn select_projection(&self, job_id: &JobId) -> Result<(), RegistryError> {
        let mut inner = self.lock()?;
        if !inner.jobs.contains_key(job_id) {
            return Err(RegistryError::UnknownJob {
                job_id: job_id.clone(),
            });
        }
        inner.projection = Some(ProjectionSelection {
            job_id: job_id.clone(),
        });
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn clear_projection(&self, token: &ProjectionToken) -> Result<(), RegistryError> {
        if token.registry_identity != self.registry_identity {
            return Err(RegistryError::ForeignToken {
                token_kind: "projection",
                job_id: token.job_id.clone(),
            });
        }
        let mut inner = self.lock()?;
        let is_current = inner.projection.as_ref().is_some_and(|selection| {
            selection.job_id == token.job_id && selection.epoch == token.epoch
        });
        if !is_current {
            return Err(RegistryError::StaleProjectionToken {
                token_job_id: token.job_id.clone(),
                token_epoch: token.epoch,
                active_job_id: inner
                    .projection
                    .as_ref()
                    .map(|selection| selection.job_id.clone()),
                active_epoch: inner.projection_epoch,
            });
        }
        inner.projection = None;
        Ok(())
    }

    pub(crate) fn projected_job_id(&self) -> Result<Option<JobId>, RegistryError> {
        Ok(self
            .lock()?
            .projection
            .as_ref()
            .map(|selection| selection.job_id.clone()))
    }

    pub(crate) fn projected_snapshot(&self) -> Result<Option<JobSnapshot<S>>, RegistryError>
    where
        S: Clone,
    {
        let inner = self.lock()?;
        inner
            .projection
            .as_ref()
            .map(|selection| Self::snapshot_from_inner(&inner, &selection.job_id))
            .transpose()
    }

    pub(crate) fn try_acquire_dispatch(
        &self,
        job_id: &JobId,
    ) -> Result<DispatchPermit, RegistryError> {
        {
            let inner = self.lock()?;
            if !inner.jobs.contains_key(job_id) {
                return Err(RegistryError::UnknownJob {
                    job_id: job_id.clone(),
                });
            }
        }
        self.dispatcher.acquire(job_id.clone())
    }

    #[cfg(test)]
    pub(crate) fn release_dispatch_permit(
        &self,
        token: &DispatchPermitToken,
    ) -> Result<(), RegistryError> {
        self.dispatcher.release(token)
    }

    pub(crate) fn dispatcher_snapshot(&self) -> Result<DispatcherSnapshot, RegistryError> {
        self.dispatcher.snapshot()
    }

    fn entry_for_token<'a>(
        &self,
        inner: &'a mut RegistryInner<S>,
        token: &JobToken,
    ) -> Result<&'a mut JobEntry<S>, RegistryError> {
        if token.registry_identity != self.registry_identity {
            return Err(RegistryError::ForeignToken {
                token_kind: "job",
                job_id: token.job_id.clone(),
            });
        }
        let entry =
            inner
                .jobs
                .get_mut(token.job_id())
                .ok_or_else(|| RegistryError::UnknownJob {
                    job_id: token.job_id.clone(),
                })?;
        if entry.revision != token.revision {
            return Err(RegistryError::StaleJobToken {
                job_id: token.job_id.clone(),
                token_revision: token.revision,
                current_revision: entry.revision,
            });
        }
        Ok(entry)
    }

    fn snapshot_from_inner(
        inner: &RegistryInner<S>,
        job_id: &JobId,
    ) -> Result<JobSnapshot<S>, RegistryError>
    where
        S: Clone,
    {
        let entry = inner
            .jobs
            .get(job_id)
            .ok_or_else(|| RegistryError::UnknownJob {
                job_id: job_id.clone(),
            })?;
        Ok(JobSnapshot {
            job_id: job_id.clone(),
            status: entry.status,
            revision: entry.revision,
            runtime_state: entry.runtime_state.clone(),
        })
    }
}

fn next_sequence(current: u64, sequence: &'static str) -> Result<u64, RegistryError> {
    current
        .checked_add(1)
        .ok_or(RegistryError::SequenceExhausted { sequence })
}

fn is_valid_identifier(value: &str, maximum_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum_bytes
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc, Barrier,
        },
        thread,
        time::Duration,
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestRuntime {
        label: String,
        event_count: usize,
    }

    fn job_id(value: &str) -> JobId {
        JobId::new(value).expect("valid job id")
    }

    fn key(value: &str) -> IdempotencyKey {
        IdempotencyKey::new(value).expect("valid idempotency key")
    }

    fn state(label: &str) -> TestRuntime {
        TestRuntime {
            label: label.to_owned(),
            event_count: 0,
        }
    }

    fn register(
        registry: &JobRegistry<TestRuntime>,
        job: &str,
        idempotency_key: &str,
    ) -> RegisterOutcome {
        registry
            .register_or_get(job_id(job), key(idempotency_key), state(job))
            .expect("job registration should succeed")
    }

    #[test]
    fn rejects_blank_identifiers_and_invalid_dispatch_limits() {
        assert_eq!(JobId::new("  "), Err(RegistryError::InvalidJobId));
        assert_eq!(JobId::new(" job"), Err(RegistryError::InvalidJobId));
        assert_eq!(
            JobId::new("job\ninjection"),
            Err(RegistryError::InvalidJobId)
        );
        assert_eq!(
            JobId::new("x".repeat(MAX_JOB_ID_BYTES + 1)),
            Err(RegistryError::InvalidJobId)
        );
        assert_eq!(
            IdempotencyKey::new(""),
            Err(RegistryError::InvalidIdempotencyKey)
        );
        assert_eq!(
            IdempotencyKey::new("request\tinjection"),
            Err(RegistryError::InvalidIdempotencyKey)
        );
        assert_eq!(
            IdempotencyKey::new("x".repeat(MAX_IDEMPOTENCY_KEY_BYTES + 1)),
            Err(RegistryError::InvalidIdempotencyKey)
        );
        assert!(matches!(
            JobRegistry::<TestRuntime>::with_dispatch_limit(0),
            Err(RegistryError::InvalidDispatcherLimit {
                requested: 0,
                minimum: MIN_DISPATCH_LIMIT,
                maximum: MAX_DISPATCH_LIMIT,
            })
        ));
        assert!(matches!(
            JobRegistry::<TestRuntime>::with_dispatch_limit(5),
            Err(RegistryError::InvalidDispatcherLimit {
                requested: 5,
                minimum: MIN_DISPATCH_LIMIT,
                maximum: MAX_DISPATCH_LIMIT,
            })
        ));
    }

    #[test]
    fn duplicate_job_ids_are_rejected() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");

        assert_eq!(
            registry.register_or_get(job_id("job-a"), key("request-b"), state("replacement")),
            Err(RegistryError::DuplicateJob {
                job_id: job_id("job-a")
            })
        );
        assert_eq!(registry.job_count().expect("job count"), 1);
    }

    #[test]
    fn repeated_idempotency_key_returns_existing_job_without_creating_another() {
        let registry = JobRegistry::new();
        let first = register(&registry, "job-a", "same-request");
        assert!(first.was_created());

        let repeated = registry
            .register_or_get(
                job_id("job-b"),
                key("same-request"),
                state("must-not-be-inserted"),
            )
            .expect("idempotent replay should succeed");

        assert!(matches!(repeated, RegisterOutcome::Existing { .. }));
        assert_eq!(repeated.job_id(), &job_id("job-a"));
        assert_eq!(repeated.token(), first.token());
        assert_eq!(registry.job_count().expect("job count"), 1);
        assert_eq!(
            registry
                .idempotent_job(&key("same-request"))
                .expect("idempotency lookup"),
            Some(job_id("job-a"))
        );
        assert_eq!(
            registry.snapshot(&job_id("job-b")),
            Err(RegistryError::UnknownJob {
                job_id: job_id("job-b")
            })
        );
    }

    #[test]
    fn idempotent_replay_cannot_alias_a_different_existing_job() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        register(&registry, "job-b", "request-b");

        assert_eq!(
            registry.register_or_get(job_id("job-b"), key("request-a"), state("replacement")),
            Err(RegistryError::DuplicateJob {
                job_id: job_id("job-b")
            })
        );
        assert_eq!(
            registry
                .idempotent_job(&key("request-a"))
                .expect("request a mapping"),
            Some(job_id("job-a"))
        );
        assert_eq!(
            registry
                .idempotent_job(&key("request-b"))
                .expect("request b mapping"),
            Some(job_id("job-b"))
        );
        assert_eq!(registry.job_count().expect("job count"), 2);
    }

    #[test]
    fn concurrent_idempotent_registration_creates_exactly_one_job() {
        const ATTEMPTS: usize = 16;

        let registry = Arc::new(JobRegistry::new());
        let barrier = Arc::new(Barrier::new(ATTEMPTS));
        let mut workers = Vec::new();
        for index in 0..ATTEMPTS {
            let registry = Arc::clone(&registry);
            let barrier = Arc::clone(&barrier);
            workers.push(thread::spawn(move || {
                barrier.wait();
                registry
                    .register_or_get(
                        job_id(&format!("job-{index}")),
                        key("same-request"),
                        state(&format!("state-{index}")),
                    )
                    .expect("concurrent idempotent registration")
            }));
        }

        let outcomes = workers
            .into_iter()
            .map(|worker| worker.join().expect("registration worker should not panic"))
            .collect::<Vec<_>>();
        let winning_job_id = outcomes[0].job_id().clone();
        assert!(outcomes
            .iter()
            .all(|outcome| outcome.job_id() == &winning_job_id));
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| outcome.was_created())
                .count(),
            1
        );
        assert_eq!(registry.job_count().expect("job count"), 1);
    }

    #[test]
    fn stores_generic_runtime_state_and_rejects_stale_job_tokens() {
        let registry = JobRegistry::new();
        let created = register(&registry, "job-a", "request-a");
        let original_token = created.token().clone();

        let (_, current_token) = registry
            .mutate_runtime(&original_token, |runtime| {
                runtime.event_count += 3;
            })
            .expect("runtime mutation");
        assert_eq!(current_token.revision(), original_token.revision() + 1);
        assert_eq!(
            registry
                .snapshot(&job_id("job-a"))
                .expect("job snapshot")
                .runtime_state
                .event_count,
            3
        );

        assert_eq!(
            registry.mutate_runtime(&original_token, |runtime| {
                runtime.event_count = 999;
            }),
            Err(RegistryError::StaleJobToken {
                job_id: job_id("job-a"),
                token_revision: original_token.revision(),
                current_revision: current_token.revision(),
            })
        );
        assert_eq!(
            registry
                .snapshot(&job_id("job-a"))
                .expect("job snapshot")
                .runtime_state
                .event_count,
            3
        );
    }

    #[test]
    fn concurrent_mutations_with_one_revision_token_commit_exactly_once() {
        const ATTEMPTS: usize = 16;

        let registry = Arc::new(JobRegistry::new());
        let created = register(registry.as_ref(), "job-a", "request-a");
        let token = created.token().clone();
        let barrier = Arc::new(Barrier::new(ATTEMPTS));
        let mut workers = Vec::new();
        for _ in 0..ATTEMPTS {
            let registry = Arc::clone(&registry);
            let token = token.clone();
            let barrier = Arc::clone(&barrier);
            workers.push(thread::spawn(move || {
                barrier.wait();
                registry.mutate_runtime(&token, |runtime| {
                    runtime.event_count += 1;
                })
            }));
        }

        let outcomes = workers
            .into_iter()
            .map(|worker| worker.join().expect("mutation worker should not panic"))
            .collect::<Vec<_>>();
        assert_eq!(outcomes.iter().filter(|outcome| outcome.is_ok()).count(), 1);
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, Err(RegistryError::StaleJobToken { .. })))
                .count(),
            ATTEMPTS - 1
        );
        let snapshot = registry.snapshot(&job_id("job-a")).expect("job snapshot");
        assert_eq!(snapshot.revision, 2);
        assert_eq!(snapshot.runtime_state.event_count, 1);
    }

    #[test]
    fn status_changes_are_revision_checked_and_terminal_jobs_cannot_reopen() {
        let registry = JobRegistry::new();
        let created = register(&registry, "job-a", "request-a");
        assert_eq!(
            registry.status(&job_id("job-a")).expect("status"),
            RegistryJobStatus::Registered
        );

        let queued = registry
            .transition_status(created.token(), RegistryJobStatus::Queued)
            .expect("queue transition");
        let running = registry
            .transition_status(&queued, RegistryJobStatus::Running)
            .expect("running transition");
        let completed = registry
            .transition_status(&running, RegistryJobStatus::Completed)
            .expect("complete transition");
        assert_eq!(
            registry.status(&job_id("job-a")).expect("status"),
            RegistryJobStatus::Completed
        );
        assert_eq!(
            registry.transition_status(&completed, RegistryJobStatus::Running),
            Err(RegistryError::InvalidStatusTransition {
                job_id: job_id("job-a"),
                from: RegistryJobStatus::Completed,
                to: RegistryJobStatus::Running,
            })
        );
    }

    #[test]
    fn status_transitions_reject_skips_and_backwards_moves() {
        let registry = JobRegistry::new();
        let created = register(&registry, "job-a", "request-a");
        assert_eq!(
            registry.transition_status(created.token(), RegistryJobStatus::Completed),
            Err(RegistryError::InvalidStatusTransition {
                job_id: job_id("job-a"),
                from: RegistryJobStatus::Registered,
                to: RegistryJobStatus::Completed,
            })
        );

        let queued = registry
            .transition_status(created.token(), RegistryJobStatus::Queued)
            .expect("queue transition");
        let running = registry
            .transition_status(&queued, RegistryJobStatus::Running)
            .expect("running transition");
        assert_eq!(
            registry.transition_status(&running, RegistryJobStatus::Queued),
            Err(RegistryError::InvalidStatusTransition {
                job_id: job_id("job-a"),
                from: RegistryJobStatus::Running,
                to: RegistryJobStatus::Queued,
            })
        );
    }

    #[test]
    fn fallible_status_transition_rolls_back_runtime_status_and_revision() {
        let registry = JobRegistry::new();
        let created = register(&registry, "job-a", "request-a");
        let error = registry
            .try_transition_status_with(
                created.token(),
                RegistryJobStatus::Queued,
                |runtime| -> Result<(), &'static str> {
                    runtime.label = "must-not-commit".to_owned();
                    runtime.event_count = 99;
                    Err("controlled mutation failure")
                },
            )
            .expect("registry transaction should execute")
            .expect_err("fallible mutation must be returned to the caller");
        assert_eq!(error, "controlled mutation failure");

        let snapshot = registry.snapshot(&job_id("job-a")).expect("job snapshot");
        assert_eq!(snapshot.status, RegistryJobStatus::Registered);
        assert_eq!(snapshot.revision, created.token().revision());
        assert_eq!(snapshot.runtime_state, state("job-a"));
    }

    #[test]
    fn fallible_status_transition_rejects_reopening_terminal_jobs_before_mutation() {
        let registry = JobRegistry::new();
        let created = register(&registry, "job-a", "request-a");
        let failed = registry
            .transition_status(created.token(), RegistryJobStatus::Failed)
            .expect("failure transition");
        let mutation_called = AtomicBool::new(false);

        assert_eq!(
            registry.try_transition_status_with(
                &failed,
                RegistryJobStatus::Queued,
                |runtime| -> Result<(), &'static str> {
                    mutation_called.store(true, Ordering::SeqCst);
                    runtime.event_count += 1;
                    Ok(())
                },
            ),
            Err(RegistryError::InvalidStatusTransition {
                job_id: job_id("job-a"),
                from: RegistryJobStatus::Failed,
                to: RegistryJobStatus::Queued,
            })
        );
        assert!(!mutation_called.load(Ordering::SeqCst));
        let snapshot = registry.snapshot(&job_id("job-a")).expect("job snapshot");
        assert_eq!(snapshot.status, RegistryJobStatus::Failed);
        assert_eq!(snapshot.revision, failed.revision());
        assert_eq!(snapshot.runtime_state, state("job-a"));
    }

    #[test]
    fn review_jobs_can_resume_and_failure_and_cancellation_are_terminal() {
        let review_registry = JobRegistry::new();
        let created = register(&review_registry, "review-job", "review-request");
        let queued = review_registry
            .transition_status(created.token(), RegistryJobStatus::Queued)
            .expect("queue transition");
        let running = review_registry
            .transition_status(&queued, RegistryJobStatus::Running)
            .expect("running transition");
        let review = review_registry
            .transition_status(&running, RegistryJobStatus::ReviewRequired)
            .expect("review transition");
        review_registry
            .transition_status(&review, RegistryJobStatus::Running)
            .expect("reviewed job can resume");

        for (job, request, terminal) in [
            ("failed-job", "failed-request", RegistryJobStatus::Failed),
            (
                "cancelled-job",
                "cancelled-request",
                RegistryJobStatus::Cancelled,
            ),
        ] {
            let registry = JobRegistry::new();
            let created = register(&registry, job, request);
            let terminal_token = registry
                .transition_status(created.token(), terminal)
                .expect("terminal transition");
            assert_eq!(
                registry.transition_status(&terminal_token, RegistryJobStatus::Queued),
                Err(RegistryError::InvalidStatusTransition {
                    job_id: job_id(job),
                    from: terminal,
                    to: RegistryJobStatus::Queued,
                })
            );
        }
    }

    #[test]
    fn projection_selection_and_event_target_lookup_are_independent() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        register(&registry, "job-b", "request-b");
        registry
            .select_projection(&job_id("job-a"))
            .expect("projection selection");

        let event_target = registry
            .event_target(&job_id("job-b"))
            .expect("event target");
        registry
            .apply_event(&job_id("job-b"), &event_target, |runtime| {
                runtime.event_count += 1;
            })
            .expect("event application");

        assert_eq!(
            registry.projected_job_id().expect("projection lookup"),
            Some(job_id("job-a"))
        );
        assert_eq!(
            registry
                .projected_snapshot()
                .expect("projected snapshot")
                .expect("active projection")
                .runtime_state
                .event_count,
            0
        );
        assert_eq!(
            registry
                .snapshot(&job_id("job-b"))
                .expect("event target snapshot")
                .runtime_state
                .event_count,
            1
        );
    }

    #[test]
    fn event_token_for_one_job_cannot_mutate_another_job() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        register(&registry, "job-b", "request-b");
        let target_a = registry
            .event_target(&job_id("job-a"))
            .expect("event target");

        assert_eq!(
            registry.apply_event(&job_id("job-b"), &target_a, |runtime| {
                runtime.event_count += 1;
            }),
            Err(RegistryError::EventTargetMismatch {
                event_job_id: job_id("job-b"),
                target_job_id: job_id("job-a"),
            })
        );
        assert_eq!(
            registry
                .snapshot(&job_id("job-a"))
                .expect("job a snapshot")
                .runtime_state
                .event_count,
            0
        );
        assert_eq!(
            registry
                .snapshot(&job_id("job-b"))
                .expect("job b snapshot")
                .runtime_state
                .event_count,
            0
        );
    }

    #[test]
    fn tokens_from_another_registry_instance_are_rejected() {
        let first_registry = JobRegistry::new();
        let second_registry = JobRegistry::new();
        let first = register(&first_registry, "job-a", "request-a");
        register(&second_registry, "job-a", "request-a");

        assert!(matches!(
            second_registry.mutate_runtime(first.token(), |runtime| {
                runtime.event_count += 1;
            }),
            Err(RegistryError::ForeignToken {
                token_kind: "job",
                job_id: foreign_job_id,
            }) if foreign_job_id == job_id("job-a")
        ));

        let foreign_event_target = first_registry
            .event_target(&job_id("job-a"))
            .expect("first event target");
        assert!(matches!(
            second_registry.apply_event(
                &job_id("job-a"),
                &foreign_event_target,
                |runtime| runtime.event_count += 1,
            ),
            Err(RegistryError::ForeignToken {
                token_kind: "event target",
                job_id: foreign_job_id,
            }) if foreign_job_id == job_id("job-a")
        ));

        let foreign_projection = first_registry
            .select_projection(&job_id("job-a"))
            .expect("first projection");
        second_registry
            .select_projection(&job_id("job-a"))
            .expect("second projection");
        assert!(matches!(
            second_registry.clear_projection(&foreign_projection),
            Err(RegistryError::ForeignToken {
                token_kind: "projection",
                job_id: foreign_job_id,
            }) if foreign_job_id == job_id("job-a")
        ));
        assert_eq!(
            second_registry.projected_job_id().expect("projection"),
            Some(job_id("job-a"))
        );

        let first_permit = first_registry
            .try_acquire_dispatch(&job_id("job-a"))
            .expect("first registry permit");
        let foreign_permit_token = first_permit.token().clone();
        let second_permit = second_registry
            .try_acquire_dispatch(&job_id("job-a"))
            .expect("second registry permit");
        assert!(matches!(
            second_registry.release_dispatch_permit(&foreign_permit_token),
            Err(RegistryError::ForeignToken {
                token_kind: "dispatch permit",
                job_id: foreign_job_id,
            }) if foreign_job_id == job_id("job-a")
        ));
        assert_eq!(
            second_registry
                .dispatcher_snapshot()
                .expect("second dispatcher snapshot")
                .in_use,
            1
        );
        drop((first_permit, second_permit));
    }

    #[test]
    fn stale_event_tokens_are_rejected() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        let stale_target = registry
            .event_target(&job_id("job-a"))
            .expect("event target");
        assert_eq!(stale_target.revision(), 1);
        let token = registry.token(&job_id("job-a")).expect("job token");
        registry
            .mutate_runtime(&token, |runtime| runtime.event_count += 1)
            .expect("out-of-band mutation");

        assert!(matches!(
            registry.apply_event(&job_id("job-a"), &stale_target, |runtime| {
                runtime.event_count += 100;
            }),
            Err(RegistryError::StaleJobToken {
                job_id: stale_job_id,
                token_revision: 1,
                current_revision: 2,
            }) if stale_job_id == job_id("job-a")
        ));
    }

    #[test]
    fn exhausted_revisions_fail_before_mutating_state_or_status() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        {
            let mut inner = registry.lock().expect("registry lock");
            inner
                .jobs
                .get_mut(&job_id("job-a"))
                .expect("registered job")
                .revision = u64::MAX;
        }

        let token = JobToken {
            registry_identity: registry.registry_identity.clone(),
            job_id: job_id("job-a"),
            revision: u64::MAX,
        };
        let event_target = EventTargetToken {
            registry_identity: registry.registry_identity.clone(),
            job_id: job_id("job-a"),
            revision: u64::MAX,
        };
        let mutation_called = AtomicBool::new(false);
        assert_eq!(
            registry.mutate_runtime(&token, |runtime| {
                mutation_called.store(true, Ordering::SeqCst);
                runtime.event_count += 1;
            }),
            Err(RegistryError::SequenceExhausted {
                sequence: "job revision"
            })
        );
        assert!(!mutation_called.load(Ordering::SeqCst));

        let event_called = AtomicBool::new(false);
        assert_eq!(
            registry.apply_event(&job_id("job-a"), &event_target, |runtime| {
                event_called.store(true, Ordering::SeqCst);
                runtime.event_count += 1;
            }),
            Err(RegistryError::SequenceExhausted {
                sequence: "job revision"
            })
        );
        assert!(!event_called.load(Ordering::SeqCst));
        assert_eq!(
            registry.transition_status(&token, RegistryJobStatus::Queued),
            Err(RegistryError::SequenceExhausted {
                sequence: "job revision"
            })
        );

        let snapshot = registry.snapshot(&job_id("job-a")).expect("job snapshot");
        assert_eq!(snapshot.revision, u64::MAX);
        assert_eq!(snapshot.status, RegistryJobStatus::Registered);
        assert_eq!(snapshot.runtime_state.event_count, 0);
    }

    #[test]
    fn panicking_mutation_fail_closes_the_registry() {
        let registry = JobRegistry::new();
        let created = register(&registry, "job-a", "request-a");

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = registry.mutate_runtime(created.token(), |runtime| {
                runtime.event_count = 999;
                panic!("simulated mutation panic");
            });
        }));
        assert!(panic.is_err());
        assert!(matches!(
            registry.job_count(),
            Err(RegistryError::StatePoisoned {
                component: "job registry"
            })
        ));
    }

    #[test]
    fn projection_tokens_are_epoch_checked() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        register(&registry, "job-b", "request-b");
        let stale = registry
            .select_projection(&job_id("job-a"))
            .expect("first projection");
        let current = registry
            .select_projection(&job_id("job-b"))
            .expect("second projection");
        assert_eq!(stale.job_id(), &job_id("job-a"));
        assert_eq!(stale.epoch(), 1);
        assert_eq!(current.job_id(), &job_id("job-b"));
        assert_eq!(current.epoch(), 2);

        assert!(matches!(
            registry.clear_projection(&stale),
            Err(RegistryError::StaleProjectionToken {
                token_job_id,
                token_epoch: 1,
                active_job_id: Some(active_job_id),
                active_epoch: 2,
            }) if token_job_id == job_id("job-a") && active_job_id == job_id("job-b")
        ));
        registry
            .clear_projection(&current)
            .expect("current projection can be cleared");
        assert_eq!(registry.projected_job_id().expect("projection"), None);
    }

    #[test]
    fn unknown_jobs_fail_all_targeted_operations() {
        let registry = JobRegistry::<TestRuntime>::new();
        let missing = job_id("missing");

        assert!(matches!(
            registry.token(&missing),
            Err(RegistryError::UnknownJob { job_id }) if job_id == missing
        ));
        assert!(matches!(
            registry.event_target(&missing),
            Err(RegistryError::UnknownJob { job_id }) if job_id == missing
        ));
        assert!(matches!(
            registry.select_projection(&missing),
            Err(RegistryError::UnknownJob { job_id }) if job_id == missing
        ));
        assert!(matches!(
            registry.try_acquire_dispatch(&missing),
            Err(RegistryError::UnknownJob { job_id }) if job_id == missing
        ));
    }

    #[test]
    fn default_dispatcher_allows_one_permit_and_drop_releases_it() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        register(&registry, "job-b", "request-b");

        let permit_a = registry
            .try_acquire_dispatch(&job_id("job-a"))
            .expect("first permit");
        assert_eq!(
            registry.dispatcher_snapshot().expect("dispatcher state"),
            DispatcherSnapshot {
                limit: 1,
                in_use: 1,
                available: 0,
                active_job_ids: vec![job_id("job-a")],
            }
        );
        assert!(matches!(
            registry.try_acquire_dispatch(&job_id("job-b")),
            Err(RegistryError::DispatcherSaturated {
                limit: 1,
                in_use: 1,
            })
        ));

        drop(permit_a);
        let permit_b = registry
            .try_acquire_dispatch(&job_id("job-b"))
            .expect("permit after RAII release");
        assert_eq!(permit_b.token().job_id(), &job_id("job-b"));
    }

    #[test]
    fn dispatcher_limit_is_configurable_up_to_four_without_overbooking() {
        let registry = JobRegistry::with_dispatch_limit(4).expect("valid dispatcher limit");
        let mut permits = Vec::new();
        for index in 0..5 {
            register(
                &registry,
                &format!("job-{index}"),
                &format!("request-{index}"),
            );
        }

        for index in 0..4 {
            permits.push(
                registry
                    .try_acquire_dispatch(&job_id(&format!("job-{index}")))
                    .expect("permit within limit"),
            );
        }
        assert!(matches!(
            registry.try_acquire_dispatch(&job_id("job-4")),
            Err(RegistryError::DispatcherSaturated {
                limit: 4,
                in_use: 4,
            })
        ));
        assert_eq!(
            registry
                .dispatcher_snapshot()
                .expect("dispatcher snapshot")
                .in_use,
            4
        );
        drop(permits);
        assert_eq!(
            registry
                .dispatcher_snapshot()
                .expect("dispatcher snapshot")
                .in_use,
            0
        );
    }

    #[test]
    fn a_job_cannot_hold_two_dispatch_permits() {
        let registry = JobRegistry::with_dispatch_limit(2).expect("valid dispatcher limit");
        register(&registry, "job-a", "request-a");
        let _permit = registry
            .try_acquire_dispatch(&job_id("job-a"))
            .expect("first permit");

        assert!(matches!(
            registry.try_acquire_dispatch(&job_id("job-a")),
            Err(RegistryError::JobAlreadyDispatched {
                job_id: duplicate_job_id
            })
            if duplicate_job_id == job_id("job-a")
        ));
    }

    #[test]
    fn explicitly_released_permit_becomes_stale_and_cannot_over_release() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        let permit = registry
            .try_acquire_dispatch(&job_id("job-a"))
            .expect("permit");
        let token = permit.token().clone();

        registry
            .release_dispatch_permit(&token)
            .expect("explicit release");
        assert_eq!(
            registry.release_dispatch_permit(&token),
            Err(RegistryError::StaleDispatchPermit {
                permit_id: token.permit_id(),
                job_id: job_id("job-a"),
            })
        );
        assert_eq!(
            permit.release(),
            Err(RegistryError::StaleDispatchPermit {
                permit_id: token.permit_id(),
                job_id: job_id("job-a"),
            })
        );
        assert_eq!(
            registry
                .dispatcher_snapshot()
                .expect("dispatcher snapshot")
                .in_use,
            0
        );
    }

    #[test]
    fn dispatch_permit_drop_releases_during_unwind() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        register(&registry, "job-b", "request-b");

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _permit = registry
                .try_acquire_dispatch(&job_id("job-a"))
                .expect("permit");
            panic!("simulated dispatched task panic");
        }));
        assert!(panic.is_err());
        let permit = registry
            .try_acquire_dispatch(&job_id("job-b"))
            .expect("RAII should release during unwind");
        assert_eq!(permit.token().job_id(), &job_id("job-b"));
    }

    #[test]
    fn exhausted_dispatch_sequence_never_creates_a_permit() {
        let registry = JobRegistry::new();
        register(&registry, "job-a", "request-a");
        registry
            .dispatcher
            .lock()
            .expect("dispatcher lock")
            .next_permit_id = u64::MAX;

        assert!(matches!(
            registry.try_acquire_dispatch(&job_id("job-a")),
            Err(RegistryError::SequenceExhausted {
                sequence: "dispatcher permit"
            })
        ));
        assert_eq!(
            registry.dispatcher_snapshot().expect("dispatcher snapshot"),
            DispatcherSnapshot {
                limit: 1,
                in_use: 0,
                available: 1,
                active_job_ids: Vec::new(),
            }
        );
    }

    #[test]
    fn dispatcher_detects_cross_index_corruption_without_overbooking() {
        let registry = JobRegistry::with_dispatch_limit(2).expect("valid dispatcher limit");
        register(&registry, "job-a", "request-a");
        {
            let mut state = registry.dispatcher.lock().expect("dispatcher lock");
            state.active_by_permit.insert(1, job_id("job-a"));
        }

        assert!(matches!(
            registry.try_acquire_dispatch(&job_id("job-a")),
            Err(RegistryError::InvariantViolation { message })
                if message.contains("indexes disagree")
        ));
        assert!(matches!(
            registry.dispatcher_snapshot(),
            Err(RegistryError::InvariantViolation { message })
                if message.contains("indexes disagree")
        ));
    }

    #[test]
    fn concurrent_dispatch_attempts_never_exceed_the_configured_limit() {
        const JOBS: usize = 16;
        const LIMIT: usize = 4;

        let registry = Arc::new(
            JobRegistry::with_dispatch_limit(LIMIT).expect("valid concurrent dispatcher limit"),
        );
        for index in 0..JOBS {
            register(
                registry.as_ref(),
                &format!("job-{index}"),
                &format!("request-{index}"),
            );
        }

        let barrier = Arc::new(Barrier::new(JOBS));
        let maximum_observed = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::new();
        for index in 0..JOBS {
            let registry = Arc::clone(&registry);
            let barrier = Arc::clone(&barrier);
            let maximum_observed = Arc::clone(&maximum_observed);
            workers.push(thread::spawn(move || {
                barrier.wait();
                if let Ok(_permit) = registry.try_acquire_dispatch(&job_id(&format!("job-{index}")))
                {
                    let in_use = registry
                        .dispatcher_snapshot()
                        .expect("dispatcher snapshot")
                        .in_use;
                    maximum_observed.fetch_max(in_use, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(10));
                }
            }));
        }
        for worker in workers {
            worker.join().expect("dispatcher worker should not panic");
        }

        assert!(maximum_observed.load(Ordering::SeqCst) <= LIMIT);
        assert_eq!(
            registry
                .dispatcher_snapshot()
                .expect("final dispatcher snapshot")
                .in_use,
            0
        );
    }
}
