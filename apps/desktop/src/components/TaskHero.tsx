import type {
  JobStatus,
  JobSummary,
  ModelStrategy,
} from "../contracts/studio";
import { useI18n, type MessageKey, type Translate } from "../i18n";
import { Icon } from "./Icon";
import { ProgressRing } from "./ProgressRing";
import { StatusBadge } from "./StatusBadge";

interface TaskHeroProps {
  job: JobSummary;
  strategy?: ModelStrategy;
  onReview: () => void;
}

interface JobStatusCopy {
  title: MessageKey;
  lead: MessageKey;
  progressTitle: MessageKey;
  progressDetail: MessageKey;
  guardrail: MessageKey;
}

const JOB_STATUS_COPY: Record<JobStatus, JobStatusCopy> = {
  draft: {
    title: "hero.draft.title",
    lead: "hero.draft.lead",
    progressTitle: "hero.draft.progressTitle",
    progressDetail: "hero.draft.progressDetail",
    guardrail: "hero.draft.guardrail",
  },
  queued: {
    title: "hero.queued.title",
    lead: "hero.queued.lead",
    progressTitle: "hero.queued.progressTitle",
    progressDetail: "hero.queued.progressDetail",
    guardrail: "hero.queued.guardrail",
  },
  running: {
    title: "hero.running.title",
    lead: "hero.running.lead",
    progressTitle: "hero.running.progressTitle",
    progressDetail: "hero.running.progressDetail",
    guardrail: "hero.running.guardrail",
  },
  review_required: {
    title: "hero.review.title",
    lead: "hero.review.lead",
    progressTitle: "hero.review.progressTitle",
    progressDetail: "hero.review.progressDetail",
    guardrail: "hero.review.guardrail",
  },
  completed: {
    title: "hero.completed.title",
    lead: "hero.completed.lead",
    progressTitle: "hero.completed.progressTitle",
    progressDetail: "hero.completed.progressDetail",
    guardrail: "hero.completed.guardrail",
  },
  failed: {
    title: "hero.failed.title",
    lead: "hero.failed.lead",
    progressTitle: "hero.failed.progressTitle",
    progressDetail: "hero.failed.progressDetail",
    guardrail: "hero.failed.guardrail",
  },
  cancelled: {
    title: "hero.cancelled.title",
    lead: "hero.cancelled.lead",
    progressTitle: "hero.cancelled.progressTitle",
    progressDetail: "hero.cancelled.progressDetail",
    guardrail: "hero.cancelled.guardrail",
  },
};

function speakerPolicyLabel(job: JobSummary, t: Translate): string {
  if (job.speakerPolicy.mode === "manual") {
    return t("hero.manualPolicy", { count: job.speakerPolicy.count });
  }
  if (job.speakerPolicy.mode === "hybrid") {
    return t("hero.hybridPolicy", {
      min: job.speakerPolicy.minSpeakers,
      max: job.speakerPolicy.maxSpeakers,
      prior: job.speakerPolicy.priorCount,
    });
  }
  if (
    (job.status === "review_required" || job.status === "completed") &&
    job.speakerDetection
  ) {
    return t("hero.autoResolved", {
      count: job.speakerDetection.estimatedCount,
    });
  }
  return t("hero.autoWaiting");
}

export function TaskHero({ job, strategy, onReview }: TaskHeroProps) {
  const { t } = useI18n();
  const copy = JOB_STATUS_COPY[job.status];
  const canInspectReviews =
    job.reviewOpenCount > 0 &&
    job.status !== "draft" &&
    job.status !== "queued" &&
    job.status !== "cancelled";
  const reviewActionLabel =
    job.status === "review_required"
      ? t("hero.reviewPending", { count: job.reviewOpenCount })
      : t("hero.inspectPending", { count: job.reviewOpenCount });

  return (
    <section
      className={`task-hero task-hero--${job.status}`}
      aria-labelledby="task-hero-title"
      data-job-status={job.status}
    >
      <div className="task-hero__content">
        <div className="task-hero__eyebrow">
          <StatusBadge status={job.status} />
          <span>
            {speakerPolicyLabel(job, t)} · {t("hero.sourcePreserved")} ·{" "}
            {t("hero.timestamps")}
          </span>
        </div>
        <h1 id="task-hero-title">{t(copy.title)}</h1>
        <p className="task-hero__lead">
          {t("hero.task")} <strong>{job.title}</strong> {t(copy.lead)}
        </p>

        <dl className="task-hero__facts">
          <div>
            <dt>{t("hero.inputMedia")}</dt>
            <dd title={job.sourcePath}>{job.sourcePath}</dd>
          </div>
          <div>
            <dt>{t("hero.duration")}</dt>
            <dd>{job.durationLabel}</dd>
          </div>
          <div>
            <dt>{t("hero.modelStrategy")}</dt>
            <dd>{strategy?.label ?? t("hero.notSelected")}</dd>
          </div>
        </dl>

        <div className="task-hero__actions">
          {canInspectReviews ? (
            <button
              className="button button--dark"
              type="button"
              onClick={onReview}
            >
              <Icon name="headphones" size={18} />
              {reviewActionLabel}
            </button>
          ) : null}
          <span className="task-hero__guardrail">
            <Icon name="lock" size={15} />
            {t(copy.guardrail)}
          </span>
        </div>
      </div>

      <div
        className="task-hero__visual"
        aria-label={`${t("hero.taskProgress")} ${job.progress}%`}
      >
        <div className="scene-orb" aria-hidden="true">
          <span className="scene-orb__spark scene-orb__spark--one" />
          <span className="scene-orb__spark scene-orb__spark--two" />
        </div>
        <ProgressRing
          value={job.progress}
          label={t("hero.taskProgress")}
          size={118}
        />
        <div className="task-hero__progress-copy">
          <strong>{t(copy.progressTitle)}</strong>
          <span>{t(copy.progressDetail)}</span>
        </div>
      </div>
    </section>
  );
}
