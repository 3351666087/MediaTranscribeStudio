import type { JobStatus, Severity, StageStatus } from "../contracts/studio";
import { useI18n, type MessageKey } from "../i18n";
import { cx } from "../lib/format";
import { Icon } from "./Icon";

type Status = JobStatus | StageStatus | Severity | "verified" | "pending" | "failed";

const labelKeys: Record<Status, MessageKey> = {
  draft: "status.draft",
  queued: "status.queued",
  running: "status.running",
  review_required: "status.reviewRequired",
  completed: "status.completed",
  failed: "status.failed",
  cancelled: "status.cancelled",
  pending: "status.pending",
  warning: "status.warning",
  blocked: "status.blocked",
  info: "status.info",
  success: "status.success",
  error: "status.error",
  verified: "status.verified",
};

export function StatusBadge({
  status,
  label,
  subtle = false,
}: {
  status: Status;
  label?: string;
  subtle?: boolean;
}) {
  const { t } = useI18n();
  const successStates: Status[] = ["completed", "success", "verified"];
  const warningStates: Status[] = ["review_required", "warning", "pending"];
  const errorStates: Status[] = ["failed", "error", "blocked", "cancelled"];
  const tone = successStates.includes(status)
    ? "success"
    : warningStates.includes(status)
      ? "warning"
      : errorStates.includes(status)
        ? "error"
        : "neutral";

  return (
    <span className={cx("status-badge", `status-badge--${tone}`, subtle && "status-badge--subtle")}>
      <span className="status-badge__dot" aria-hidden="true" />
      {label ?? t(labelKeys[status])}
    </span>
  );
}

export function StateIcon({ status }: { status: StageStatus }) {
  const icon = status === "completed" ? "check" : status === "warning" ? "alert" : "clock";
  return (
    <span className={cx("state-icon", `state-icon--${status}`)}>
      <Icon name={icon} size={15} />
    </span>
  );
}
