import type {
  AppSection,
  StudioJobItem,
  SystemStatus,
} from "../contracts/studio";
import { useI18n, type MessageKey } from "../i18n";
import { Icon } from "./Icon";
import { PreferencesControl } from "./PreferencesControl";
import { StatusBadge } from "./StatusBadge";
import { TaskCenter } from "./TaskCenter";

const sectionCopy: Record<
  AppSection,
  { eyebrow: MessageKey; title: MessageKey }
> = {
  overview: { eyebrow: "top.overviewEyebrow", title: "top.overviewTitle" },
  review: { eyebrow: "top.reviewEyebrow", title: "top.reviewTitle" },
  artifacts: { eyebrow: "top.artifactsEyebrow", title: "top.artifactsTitle" },
  "pdf-qa": { eyebrow: "top.pdfEyebrow", title: "top.pdfTitle" },
};

interface TopBarProps {
  activeSection: AppSection;
  system: SystemStatus;
  jobs: readonly StudioJobItem[];
  selectedJobId: string | null;
  busyAction: string | null;
  onSelectJob: (jobId: string) => Promise<void>;
  onCancelJob: (jobId: string) => Promise<void>;
  onCreateTask: () => void;
}

export function TopBar({
  activeSection,
  system,
  jobs,
  selectedJobId,
  busyAction,
  onSelectJob,
  onCancelJob,
  onCreateTask,
}: TopBarProps) {
  const { t } = useI18n();
  const copy = sectionCopy[activeSection];
  const backendLabel =
    system.backendMode === "tauri-ipc"
      ? t("top.strictTauri")
      : t("top.safeMock");
  return (
    <header className="top-bar">
      <div className="top-bar__title">
        <span>{t(copy.eyebrow)}</span>
        <strong>{t(copy.title)}</strong>
      </div>

      <div className="top-bar__actions">
        <TaskCenter
          jobs={jobs}
          selectedJobId={selectedJobId}
          busyAction={busyAction}
          onSelect={onSelectJob}
          onCancel={onCancelJob}
        />
        <PreferencesControl />
        <div className="system-pill" aria-label={t("top.runtime")}>
          <Icon name="cloud-off" size={17} />
          <span>{t("top.offline")}</span>
          <StatusBadge
            status={system.backendMode === "mock" ? "info" : "success"}
            label={backendLabel}
            subtle
          />
        </div>
        <button
          className="button button--primary top-bar__create"
          type="button"
          aria-label={t("top.createTask")}
          onClick={onCreateTask}
        >
          <Icon name="plus" size={18} />
          <span>{t("top.createTask")}</span>
        </button>
      </div>
    </header>
  );
}
