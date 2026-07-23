import { useRef } from "react";
import type { StudioJobItem } from "../contracts/studio";
import { useI18n } from "../i18n";
import { Icon } from "./Icon";
import { StatusBadge } from "./StatusBadge";

interface TaskCenterProps {
  jobs: readonly StudioJobItem[];
  selectedJobId: string | null;
  busyAction: string | null;
  onSelect: (jobId: string) => Promise<void>;
  onCancel: (jobId: string) => Promise<void>;
}

function sourceName(path: string | null): string | null {
  if (!path) {
    return null;
  }
  return path.split(/[\\/]/u).filter(Boolean).at(-1) ?? path;
}

export function TaskCenter({
  jobs,
  selectedJobId,
  busyAction,
  onSelect,
  onCancel,
}: TaskCenterProps) {
  const { t } = useI18n();
  const detailsRef = useRef<HTMLDetailsElement>(null);
  const switchingJobId = busyAction?.startsWith("select-job:")
    ? busyAction.slice("select-job:".length)
    : null;
  const cancellingJobId = busyAction?.startsWith("cancel-job:")
    ? busyAction.slice("cancel-job:".length)
    : null;
  const taskCenterBusy =
    switchingJobId !== null ||
    cancellingJobId !== null ||
    busyAction === "create-job-batch";

  return (
    <details className="task-center" ref={detailsRef}>
      <summary
        className="task-center__trigger"
        aria-label={t("tasks.center.trigger", { count: jobs.length })}
      >
        <span className="task-center__trigger-icon" aria-hidden="true">
          <Icon name="wave" size={17} />
        </span>
        <span>{t("tasks.center.title")}</span>
        <span className="task-center__count" aria-hidden="true">
          {jobs.length}
        </span>
        <Icon name="chevron-down" size={15} />
      </summary>

      <section
        className="task-center__panel"
        aria-label={t("tasks.center.panelLabel")}
      >
        <header className="task-center__header">
          <div>
            <span className="panel__eyebrow">
              {t("tasks.center.volatile")}
            </span>
            <h2>{t("tasks.center.title")}</h2>
          </div>
          <p>{t("tasks.center.description")}</p>
        </header>

        {jobs.length === 0 ? (
          <div className="task-center__empty">
            <span aria-hidden="true">
              <Icon name="sparkles" size={22} />
            </span>
            <strong>{t("tasks.center.emptyTitle")}</strong>
            <small>{t("tasks.center.emptyDetail")}</small>
          </div>
        ) : (
          <ul className="task-center__list">
            {jobs.map((job) => {
              const normalizedTitle = job.title?.trim();
              const title =
                normalizedTitle && normalizedTitle.length > 0
                  ? normalizedTitle
                  : job.jobId;
              const current = job.jobId === selectedJobId;
              const switching = job.jobId === switchingJobId;
              const cancelling = job.jobId === cancellingJobId;
              const progress =
                job.progress === null
                  ? null
                  : Math.min(100, Math.max(0, Math.round(job.progress)));
              return (
                <li
                  className="task-center__item"
                  data-current={current ? "true" : "false"}
                  data-status={job.status}
                  key={job.jobId}
                >
                  <button
                    className="task-center__select"
                    type="button"
                    disabled={current || taskCenterBusy}
                    aria-current={current ? "true" : undefined}
                    aria-label={
                      switching
                        ? t("tasks.center.selecting", { title })
                        : t("tasks.center.select", { title })
                    }
                    onClick={() => {
                      void onSelect(job.jobId)
                        .then(() => {
                          if (detailsRef.current) {
                            detailsRef.current.open = false;
                          }
                        })
                        .catch(() => undefined);
                    }}
                  >
                    <span className="task-center__identity">
                      <strong>{title}</strong>
                      <small>
                        {sourceName(job.sourcePath) ??
                          t("tasks.center.sourceUnknown")}
                      </small>
                    </span>
                    <span className="task-center__state">
                      {current ? (
                        <span className="task-center__current">
                          {t("tasks.center.current")}
                        </span>
                      ) : null}
                      <StatusBadge status={job.status} subtle />
                    </span>
                  </button>

                  {job.cancellable ? (
                    <button
                      className="task-center__cancel"
                      type="button"
                      disabled={taskCenterBusy}
                      aria-label={
                        cancelling
                          ? t("tasks.center.cancelling", { title })
                          : t("tasks.center.cancel", { title })
                      }
                      onClick={() => {
                        void onCancel(job.jobId).catch(() => undefined);
                      }}
                    >
                      <Icon name="x" size={15} />
                    </button>
                  ) : null}

                  <span className="task-center__progress-copy">
                    {progress === null
                      ? t("tasks.center.progressUnknown")
                      : t("tasks.center.progress", { progress })}
                  </span>
                  <span
                    className="task-center__progress"
                    role={progress === null ? undefined : "progressbar"}
                    aria-label={
                      progress === null
                        ? undefined
                        : t("tasks.center.progress", { progress })
                    }
                    aria-valuemin={progress === null ? undefined : 0}
                    aria-valuemax={progress === null ? undefined : 100}
                    aria-valuenow={progress ?? undefined}
                    data-indeterminate={progress === null ? "true" : "false"}
                  >
                    <span
                      style={{
                        inlineSize:
                          progress === null ? "34%" : `${progress}%`,
                      }}
                    />
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </details>
  );
}
