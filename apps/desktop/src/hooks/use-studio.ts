import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createJobBatch as coordinateCreateJobBatch,
  type CreateJobBatchResult,
} from "../bridge/batch-coordinator";
import {
  desktopBackend,
  isTauriRuntime,
} from "../bridge/desktop-backend";
import { parseStudioSnapshot } from "../contracts/runtime-validation";
import type {
  AppSection,
  CreateJobRequest,
  JobRuntimeStatus,
  ModelStrategyId,
  ReviewDecision,
  StudioJobItem,
  StudioSnapshot,
  UpdateSpeakerRequest,
} from "../contracts/studio";
import {
  useI18n,
  type MessageKey,
  type MessageParams,
} from "../i18n";

interface ToastState {
  id: number;
  tone: "success" | "warning" | "error" | "info";
  title: string;
  detail: string;
}

const RECONCILIATION_INTERVAL_MS = 2_000;
const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "cancelled"]);

export function useStudio() {
  const { t } = useI18n();
  const translationRef = useRef(t);
  const [snapshot, setSnapshot] = useState<StudioSnapshot | null>(null);
  const [jobStatuses, setJobStatuses] = useState<JobRuntimeStatus[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobMetadata, setJobMetadata] = useState<
    ReadonlyMap<
      string,
      Pick<StudioJobItem, "title" | "sourcePath" | "progress">
    >
  >(() => new Map());
  const [activeSection, setActiveSection] = useState<AppSection>("overview");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastState | null>(null);
  const toastSequence = useRef(1);
  const loadSequence = useRef(0);
  const snapshotRevision = useRef(0);
  const selectedJobIdRef = useRef<string | null>(null);
  const pollFailureCount = useRef(0);
  const pollWarningShown = useRef(false);

  useEffect(() => {
    translationRef.current = t;
  }, [t]);

  const translateMessage = useCallback(
    (key: MessageKey, params?: MessageParams) =>
      translationRef.current(key, params),
    [],
  );

  const notify = useCallback(
    (tone: ToastState["tone"], title: string, detail: string) => {
      toastSequence.current += 1;
      setToast({ id: toastSequence.current, tone, title, detail });
    },
    [],
  );

  const notifyLocalized = useCallback(
    (
      tone: ToastState["tone"],
      titleKey: MessageKey,
      detailKey: MessageKey,
      params?: MessageParams,
    ) => {
      notify(
        tone,
        translateMessage(titleKey, params),
        translateMessage(detailKey, params),
      );
    },
    [notify, translateMessage],
  );

  const commitSnapshot = useCallback((next: StudioSnapshot) => {
    snapshotRevision.current += 1;
    setSnapshot(next);
    selectedJobIdRef.current = next.job.id;
    setSelectedJobId(next.job.id);
    setJobMetadata((current) => {
      const updated = new Map(current);
      updated.set(next.job.id, {
        title: next.job.title,
        sourcePath: next.job.sourcePath,
        progress: next.job.progress,
      });
      return updated;
    });
    setLoadError(null);
  }, []);

  const commitJobStatuses = useCallback(
    (next: JobRuntimeStatus[]) => {
      setJobStatuses(next);
      const projected = next.find((job) => job.projected);
      if (projected && selectedJobIdRef.current === null) {
        selectedJobIdRef.current = projected.jobId;
        setSelectedJobId(projected.jobId);
      }
    },
    [],
  );

  const refreshJobs = useCallback(async () => {
    const next = await desktopBackend.listJobs();
    commitJobStatuses(next);
    return next;
  }, [commitJobStatuses]);

  const refresh = useCallback(async () => {
    const revisionAtStart = snapshotRevision.current;
    const [next, nextJobs] = await Promise.all([
      desktopBackend.getSnapshot(),
      desktopBackend.listJobs(),
    ]);
    if (snapshotRevision.current === revisionAtStart) {
      commitSnapshot(next);
    }
    commitJobStatuses(nextJobs);
  }, [commitJobStatuses, commitSnapshot]);

  const loadSnapshot = useCallback(async () => {
    const sequence = loadSequence.current + 1;
    const revisionAtStart = snapshotRevision.current;
    loadSequence.current = sequence;
    setLoading(true);
    setLoadError(null);

    try {
      const [next, nextJobs] = await Promise.all([
        desktopBackend.getSnapshot(),
        desktopBackend.listJobs(),
      ]);
      if (loadSequence.current !== sequence) {
        return;
      }
      if (snapshotRevision.current === revisionAtStart) {
        commitSnapshot(next);
      }
      commitJobStatuses(nextJobs);
    } catch (error: unknown) {
      if (loadSequence.current !== sequence) {
        return;
      }
      const detail =
        error instanceof Error
          ? error.message
          : translateMessage("notification.workspace.unknownError");
      setSnapshot(null);
      setJobStatuses([]);
      selectedJobIdRef.current = null;
      setSelectedJobId(null);
      setLoadError(detail);
      notify(
        "error",
        translateMessage("notification.workspace.loadFailedTitle"),
        detail,
      );
    } finally {
      if (loadSequence.current === sequence) {
        setLoading(false);
      }
    }
  }, [
    commitJobStatuses,
    commitSnapshot,
    notify,
    translateMessage,
  ]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      loadSnapshot().catch((error: unknown) => {
        console.error("Failed to read the local workspace", error);
      });
    }, 0);

    return () => {
      window.clearTimeout(timer);
      loadSequence.current += 1;
    };
  }, [loadSnapshot]);

  useEffect(() => {
    if (!isTauriRuntime()) {
      return undefined;
    }

    let disposed = false;
    let unlisten: (() => void) | undefined;

    listen<unknown>("snapshot-updated", (event) => {
      if (disposed) {
        return;
      }
      try {
        commitSnapshot(parseStudioSnapshot(event.payload));
        pollFailureCount.current = 0;
        pollWarningShown.current = false;
      } catch (error) {
        notifyLocalized(
          "error",
          "notification.live.rejectedTitle",
          "notification.live.rejectedDetail",
          {
            error:
              error instanceof Error
                ? error.message
                : translateMessage(
                    "notification.live.invalidPayload",
                  ),
          },
        );
      }
    })
      .then((disposeListener) => {
        if (disposed) {
          disposeListener();
          return;
        }
        unlisten = disposeListener;
      })
      .catch((error: unknown) => {
        if (disposed) {
          return;
        }
        notifyLocalized(
          "warning",
          "notification.live.unavailableTitle",
          "notification.live.unavailableDetail",
          {
            error:
              error instanceof Error
                ? error.message
                : translateMessage(
                    "notification.live.listenerFailed",
                  ),
          },
        );
      });

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [commitSnapshot, notifyLocalized, translateMessage]);

  const hasActiveJobs = useMemo(
    () =>
      jobStatuses.some(
        (job) => !TERMINAL_JOB_STATUSES.has(job.status),
      ) ||
      Boolean(
        snapshot &&
          !TERMINAL_JOB_STATUSES.has(snapshot.job.status),
      ),
    [jobStatuses, snapshot],
  );

  useEffect(() => {
    if (!isTauriRuntime() || !snapshot || !hasActiveJobs) {
      return undefined;
    }

    let disposed = false;
    let timer: number | undefined;

    const reconcile = async () => {
      try {
        await refresh();
        pollFailureCount.current = 0;
        pollWarningShown.current = false;
      } catch (error) {
        pollFailureCount.current += 1;
        if (
          pollFailureCount.current >= 3 &&
          !pollWarningShown.current &&
          !disposed
        ) {
          pollWarningShown.current = true;
          notifyLocalized(
            "warning",
            "notification.sync.delayedTitle",
            "notification.sync.delayedDetail",
            {
              error:
                error instanceof Error
                  ? error.message
                  : translateMessage("notification.sync.readFailed"),
            },
          );
        }
      } finally {
        if (!disposed) {
          timer = window.setTimeout(() => {
            reconcile().catch((error: unknown) => {
              console.error("Snapshot reconciliation failed", error);
            });
          }, RECONCILIATION_INTERVAL_MS);
        }
      }
    };

    timer = window.setTimeout(() => {
      reconcile().catch((error: unknown) => {
        console.error("Snapshot reconciliation failed", error);
      });
    }, RECONCILIATION_INTERVAL_MS);
    return () => {
      disposed = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [
    hasActiveJobs,
    notifyLocalized,
    refresh,
    snapshot,
    translateMessage,
  ]);

  const chooseStrategy = useCallback((strategyId: ModelStrategyId) => {
    setSnapshot((current) =>
      current
        ? {
            ...current,
            job: { ...current.job, activeStrategyId: strategyId },
          }
        : current,
    );
  }, []);

  const updateSpeaker = useCallback(
    async (request: UpdateSpeakerRequest) => {
      setBusyAction(request.speakerId);
      try {
        await desktopBackend.updateSpeaker(request);
        try {
          await refresh();
          notify(
            "success",
            "Speaker settings saved",
            `${request.label.trim()}'s name, lock, and review state were saved locally.`,
          );
        } catch (syncError) {
          notify(
            "warning",
            "Speaker settings saved; refresh delayed",
            `The worker accepted the change. Live synchronization will reconcile it. ${
              syncError instanceof Error ? syncError.message : ""
            }`.trim(),
          );
        }
      } catch (error) {
        notify(
          "error",
          "Speaker settings could not be saved",
          error instanceof Error
            ? error.message
            : "Check the input and try again.",
        );
        throw error;
      } finally {
        setBusyAction(null);
      }
    },
    [notify, refresh],
  );

  const createJob = useCallback(
    async (request: CreateJobRequest) => {
      setBusyAction("create-job");
      try {
        const result = await desktopBackend.createJob(request);
        if (result.accepted) {
          setJobMetadata((current) => {
            const updated = new Map(current);
            updated.set(result.jobId, {
              title: request.title.trim(),
              sourcePath: request.mediaPath.trim(),
              progress: 0,
            });
            return updated;
          });
        }
        try {
          await refresh();
          notify("success", "Task created safely", result.message);
        } catch (syncError) {
          notify(
            "warning",
            "Task created; refresh delayed",
            `${result.message} Live synchronization will reconcile the workspace. ${
              syncError instanceof Error ? syncError.message : ""
            }`.trim(),
          );
        }
        setActiveSection("overview");
        return result;
      } catch (error) {
        notify(
          "error",
          "Task could not be created",
          error instanceof Error
            ? error.message
            : "Check the input and try again.",
        );
        throw error;
      } finally {
        setBusyAction(null);
      }
    },
    [notify, refresh],
  );

  const createJobBatch = useCallback(
    async (
      requests: readonly CreateJobRequest[],
    ): Promise<CreateJobBatchResult> => {
      setBusyAction("create-job-batch");
      try {
        const result = await coordinateCreateJobBatch(
          requests,
          async (request) => await desktopBackend.createJob(request),
        );
        const acceptedItems = result.items.filter(
          (item) => item.status === "accepted",
        );

        if (acceptedItems.length > 0) {
          setJobMetadata((current) => {
            const updated = new Map(current);
            acceptedItems.forEach((item) => {
              updated.set(item.result.jobId, {
                title: item.request.title.trim(),
                sourcePath: item.request.mediaPath.trim(),
                progress: 0,
              });
            });
            return updated;
          });

          const target = acceptedItems.at(-1);
          if (!target) {
            return result;
          }

          try {
            const next = await desktopBackend.selectJob(
              target.result.jobId,
            );
            commitSnapshot(next);
            await refreshJobs();
            setActiveSection("overview");
            if (result.failedCount === 0) {
              notifyLocalized(
                "success",
                "notification.batch.createdTitle",
                "notification.batch.createdDetail",
                { count: result.acceptedCount },
              );
            } else {
              notifyLocalized(
                "warning",
                "notification.batch.partialTitle",
                "notification.batch.partialDetail",
                {
                  accepted: result.acceptedCount,
                  failed: result.failedCount,
                },
              );
            }
          } catch (syncError) {
            try {
              await refreshJobs();
            } catch {
              // The accepted items remain valid even if task-list refresh lags.
            }
            notifyLocalized(
              "warning",
              "notification.batch.syncDelayedTitle",
              "notification.batch.syncDelayedDetail",
              {
                accepted: result.acceptedCount,
                failed: result.failedCount,
                error:
                  syncError instanceof Error
                    ? syncError.message
                    : translateMessage("notification.sync.readFailed"),
              },
            );
          }
        } else {
          notifyLocalized(
            "error",
            "notification.batch.failedTitle",
            "notification.batch.failedDetail",
            { count: result.failedCount },
          );
        }

        return result;
      } finally {
        setBusyAction(null);
      }
    },
    [
      commitSnapshot,
      notifyLocalized,
      refreshJobs,
      translateMessage,
    ],
  );

  const selectJob = useCallback(
    async (jobId: string) => {
      if (jobId === selectedJobIdRef.current) {
        return;
      }

      setBusyAction(`select-job:${jobId}`);
      try {
        const next = await desktopBackend.selectJob(jobId);
        commitSnapshot(next);
        setActiveSection("overview");
        try {
          await refreshJobs();
          notifyLocalized(
            "success",
            "notification.task.selectedTitle",
            "notification.task.selectedDetail",
            { title: next.job.title },
          );
        } catch (syncError) {
          notifyLocalized(
            "warning",
            "notification.task.selectedSyncDelayedTitle",
            "notification.task.selectedSyncDelayedDetail",
            {
              title: next.job.title,
              error:
                syncError instanceof Error
                  ? syncError.message
                  : translateMessage("notification.sync.readFailed"),
            },
          );
        }
      } catch (error) {
        notifyLocalized(
          "error",
          "notification.task.selectFailedTitle",
          "notification.task.selectFailedDetail",
          {
            error:
              error instanceof Error
                ? error.message
                : translateMessage(
                    "notification.task.selectRejected",
                  ),
          },
        );
        throw error;
      } finally {
        setBusyAction(null);
      }
    },
    [
      commitSnapshot,
      notifyLocalized,
      refreshJobs,
      translateMessage,
    ],
  );

  const applyReview = useCallback(
    async (decision: ReviewDecision) => {
      setBusyAction(decision.reviewId);
      try {
        await desktopBackend.applyReviewDecision(decision);
        try {
          await refresh();
          notify(
            "success",
            "Review decision locked",
            "The decision was written to the local audit trail. You can continue to the next segment.",
          );
        } catch (syncError) {
          notify(
            "warning",
            "Review saved; refresh delayed",
            `The worker persisted the human decision. Live synchronization will reconcile it. ${
              syncError instanceof Error ? syncError.message : ""
            }`.trim(),
          );
        }
      } catch (error) {
        notify(
          "error",
          "Review could not be saved",
          error instanceof Error
            ? error.message
            : "Your current input was preserved. Try again.",
        );
      } finally {
        setBusyAction(null);
      }
    },
    [notify, refresh],
  );

  const cancelJob = useCallback(
    async (jobId: string) => {
      const cancellingSelectedJob =
        jobId === selectedJobIdRef.current;
      setBusyAction(`cancel-job:${jobId}`);
      try {
        await desktopBackend.cancelJob(jobId);
        try {
          if (cancellingSelectedJob) {
            await refresh();
          } else {
            await refreshJobs();
          }
          notifyLocalized(
            "success",
            "notification.cancel.acceptedTitle",
            "notification.cancel.acceptedDetail",
          );
        } catch (syncError) {
          notifyLocalized(
            "warning",
            "notification.cancel.syncDelayedTitle",
            "notification.cancel.syncDelayedDetail",
            {
              error:
                syncError instanceof Error
                  ? syncError.message
                  : translateMessage("notification.sync.readFailed"),
            },
          );
        }
      } catch (error) {
        notifyLocalized(
          "error",
          "notification.cancel.failedTitle",
          "notification.cancel.failedDetail",
          {
            error:
              error instanceof Error
                ? error.message
                : translateMessage("notification.cancel.rejected"),
          },
        );
        throw error;
      } finally {
        setBusyAction(null);
      }
    },
    [
      notifyLocalized,
      refresh,
      refreshJobs,
      translateMessage,
    ],
  );

  const openArtifact = useCallback(
    async (artifactId: string) => {
      setBusyAction(artifactId);
      try {
        const result = await desktopBackend.openArtifact(artifactId);
        notify(
          result.opened ? "success" : "info",
          result.opened
            ? "Artifact opened locally"
            : "Artifact path verified safely",
          result.message,
        );
      } catch (error) {
        notify(
          "error",
          "Artifact could not be opened",
          error instanceof Error ? error.message : "The artifact is not ready yet.",
        );
      } finally {
        setBusyAction(null);
      }
    },
    [notify],
  );

  const openReviews = useMemo(
    () => snapshot?.reviews.filter((item) => !item.reviewed) ?? [],
    [snapshot],
  );
  const jobs = useMemo<StudioJobItem[]>(
    () =>
      jobStatuses.map((status) => {
        const metadata = jobMetadata.get(status.jobId);
        return {
          ...status,
          title: metadata?.title ?? null,
          sourcePath: metadata?.sourcePath ?? null,
          progress: metadata?.progress ?? null,
        };
      }),
    [jobMetadata, jobStatuses],
  );

  return {
    snapshot,
    jobs,
    selectedJobId,
    activeSection,
    setActiveSection,
    loading,
    loadError,
    retryLoad: loadSnapshot,
    busyAction,
    toast,
    clearToast: () => setToast(null),
    openReviews,
    chooseStrategy,
    updateSpeaker,
    createJob,
    createJobBatch,
    selectJob,
    cancelJob,
    applyReview,
    openArtifact,
    refreshJobs,
    notify,
  };
}
