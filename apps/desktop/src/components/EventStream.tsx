import type { Severity, StudioEvent } from "../contracts/studio";
import { useI18n, type MessageKey } from "../i18n";
import { Icon, type IconName } from "./Icon";

const eventIcon: Record<Severity, IconName> = {
  info: "wave",
  success: "check",
  warning: "alert",
  error: "x",
};

const eventTypeLabels: Record<StudioEvent["type"], MessageKey> = {
  "job.started": "workbench.events.type.jobStarted",
  "stage.started": "workbench.events.type.stageStarted",
  "stage.progress": "workbench.events.type.stageProgress",
  "artifact.created": "workbench.events.type.artifactCreated",
  "review.required": "workbench.events.type.reviewRequired",
  "review.decision.persisted":
    "workbench.events.type.reviewDecisionPersisted",
  warning: "workbench.events.type.warning",
  "job.failed": "workbench.events.type.jobFailed",
  "job.completed": "workbench.events.type.jobCompleted",
  "job.cancelled": "workbench.events.type.jobCancelled",
};

interface EventStreamProps {
  events: StudioEvent[];
}

export function EventStream({ events }: EventStreamProps) {
  const { t } = useI18n();

  return (
    <section className="panel event-panel" aria-labelledby="event-stream-title">
      <div className="panel__header">
        <div>
          <span className="panel__eyebrow">
            {t("workbench.events.eyebrow")}
          </span>
          <h2 id="event-stream-title">
            {t("workbench.events.title")}
          </h2>
        </div>
        <span className="live-indicator">
          <span aria-hidden="true" />
          {t("workbench.events.live")}
        </span>
      </div>

      <ol
        className="event-list"
        aria-label={t("workbench.events.recentAria")}
      >
        {events.map((event) => (
          <li className={`event-item event-item--${event.severity}`} key={event.id}>
            <span className="event-item__icon" aria-hidden="true">
              <Icon name={eventIcon[event.severity]} size={16} />
            </span>
            <div className="event-item__copy">
              <div>
                <strong>{event.title}</strong>
                <time dateTime={event.timestamp}>{event.timestamp}</time>
              </div>
              <p>{event.detail}</p>
              <small>
                {t("workbench.events.sequence", {
                  sequence: event.sequence,
                  type: t(eventTypeLabels[event.type]),
                })}
              </small>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
