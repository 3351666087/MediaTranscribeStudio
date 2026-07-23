import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  desktopBackend,
  isTauriRuntime,
} from "../bridge/desktop-backend";
import { parseStudioSnapshot } from "../contracts/runtime-validation";
import type {
  AppSection,
  CreateJobRequest,
  ModelStrategyId,
  ReviewDecision,
  StudioSnapshot,
  UpdateSpeakerRequest,
} from "../contracts/studio";

interface ToastState {
  id: number;
  tone: "success" | "warning" | "error" | "info";
  title: string;
  detail: string;
}

const RECONCILIATION_INTERVAL_MS = 2_000;
const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "cancelled"]);

export function useStudio() {
  const [snapshot, setSnapshot] = useState<StudioSnapshot | null>(null);
  const [activeSection, setActiveSection] = useState<AppSection>("overview");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastState | null>(null);
  const toastSequence = useRef(1);
  const loadSequence = useRef(0);
  const snapshotRevision = useRef(0);
  const pollFailureCount = useRef(0);
  const pollWarningShown = useRef(false);

  const notify = useCallback(
    (tone: ToastState["tone"], title: string, detail: string) => {
      toastSequence.current += 1;
      setToast({ id: toastSequence.current, tone, title, detail });
    },
    [],
  );

  const commitSnapshot = useCallback((next: StudioSnapshot) => {
    snapshotRevision.current += 1;
    setSnapshot(next);
    setLoadError(null);
  }, []);

  const refresh = useCallback(async () => {
    const revisionAtStart = snapshotRevision.current;
    const next = await desktopBackend.getSnapshot();
    if (snapshotRevision.current === revisionAtStart) {
      commitSnapshot(next);
    }
  }, [commitSnapshot]);

  const loadSnapshot = useCallback(async () => {
    const sequence = loadSequence.current + 1;
    const revisionAtStart = snapshotRevision.current;
    loadSequence.current = sequence;
    setLoading(true);
    setLoadError(null);

    try {
      const next = await desktopBackend.getSnapshot();
      if (loadSequence.current !== sequence) {
        return;
      }
      if (snapshotRevision.current === revisionAtStart) {
        commitSnapshot(next);
      }
    } catch (error: unknown) {
      if (loadSequence.current !== sequence) {
        return;
      }
      const detail =
        error instanceof Error
          ? error.message
          : "The local workspace returned an unknown error.";
      setSnapshot(null);
      setLoadError(detail);
      toastSequence.current += 1;
      setToast({
        id: toastSequence.current,
        tone: "error",
        title: "Workspace failed to load",
        detail,
      });
    } finally {
      if (loadSequence.current === sequence) {
        setLoading(false);
      }
    }
  }, [commitSnapshot]);

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
        notify(
          "error",
          "Unsafe live update rejected",
          error instanceof Error
            ? error.message
            : "A local worker update failed strict contract validation.",
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
        notify(
          "warning",
          "Live updates unavailable",
          `${
            error instanceof Error
              ? error.message
              : "The Tauri event listener could not be established."
          } Bounded snapshot reconciliation remains active.`,
        );
      });

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [commitSnapshot, notify]);

  useEffect(() => {
    if (
      !isTauriRuntime() ||
      !snapshot ||
      TERMINAL_JOB_STATUSES.has(snapshot.job.status)
    ) {
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
          notify(
            "warning",
            "Workspace synchronization is delayed",
            error instanceof Error
              ? error.message
              : "The bounded reconciliation poll could not read the local snapshot.",
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
  }, [notify, refresh, snapshot]);

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
      setBusyAction("cancel-job");
      try {
        await desktopBackend.cancelJob(jobId);
        try {
          await refresh();
          notify(
            "success",
            "Cancellation requested",
            "The local worker accepted the cancellation request.",
          );
        } catch (syncError) {
          notify(
            "warning",
            "Cancellation accepted; refresh delayed",
            `Live synchronization will reconcile the final task state. ${
              syncError instanceof Error ? syncError.message : ""
            }`.trim(),
          );
        }
      } catch (error) {
        notify(
          "error",
          "Task could not be cancelled",
          error instanceof Error
            ? error.message
            : "The local worker rejected the cancellation request.",
        );
        throw error;
      } finally {
        setBusyAction(null);
      }
    },
    [notify, refresh],
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

  return {
    snapshot,
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
    cancelJob,
    applyReview,
    openArtifact,
    notify,
  };
}
