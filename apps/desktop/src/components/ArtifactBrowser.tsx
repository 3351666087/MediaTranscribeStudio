import type { ArtifactItem, ArtifactKind, SystemStatus } from "../contracts/studio";
import { useI18n, type MessageKey } from "../i18n";
import { Icon, type IconName } from "./Icon";
import { StatusBadge } from "./StatusBadge";

const artifactIcons: Record<ArtifactKind, IconName> = {
  "transcript-json": "code",
  "transcript-text": "file",
  pdf: "pdf",
  "page-image": "image",
  "contact-sheet": "image",
  "quality-report": "shield",
  "repair-queue": "wand",
  log: "file",
};

const artifactTypeLabels: Record<ArtifactKind, MessageKey> = {
  "transcript-json": "workbench.artifacts.kind.transcriptJson",
  "transcript-text": "workbench.artifacts.kind.transcriptText",
  pdf: "workbench.artifacts.kind.pdf",
  "page-image": "workbench.artifacts.kind.pageImage",
  "contact-sheet": "workbench.artifacts.kind.contactSheet",
  "quality-report": "workbench.artifacts.kind.qualityReport",
  "repair-queue": "workbench.artifacts.kind.repairQueue",
  log: "workbench.artifacts.kind.log",
};

interface ArtifactBrowserProps {
  artifacts: ArtifactItem[];
  backendMode: SystemStatus["backendMode"];
  busyAction: string | null;
  onOpen: (artifactId: string) => Promise<void>;
}

export function ArtifactBrowser({
  artifacts,
  backendMode,
  busyAction,
  onOpen,
}: ArtifactBrowserProps) {
  const { t } = useI18n();
  const verified = artifacts.filter((artifact) => artifact.integrity === "verified").length;
  const tauriMode = backendMode === "tauri-ipc";

  return (
    <section className="artifact-page" aria-labelledby="artifacts-title">
      <div className="page-heading">
        <div>
          <span className="panel__eyebrow">
            {t("workbench.artifacts.eyebrow")}
          </span>
          <h1 id="artifacts-title">
            {t("workbench.artifacts.title")}
          </h1>
          <p>
            {tauriMode
              ? t("workbench.artifacts.description.tauri")
              : t("workbench.artifacts.description.mock")}
          </p>
        </div>
        <div className="page-heading__metric">
          <strong>{verified}</strong>
          <span>
            {t("workbench.artifacts.verifiedSummary", {
              total: artifacts.length,
            })}
          </span>
        </div>
      </div>

      <div className="artifact-banner">
        <span className="artifact-banner__icon" aria-hidden="true">
          <Icon name="folder" size={22} />
        </span>
        <div>
          <strong>
            {tauriMode
              ? t("workbench.artifacts.banner.title.tauri")
              : t("workbench.artifacts.banner.title.mock")}
          </strong>
          <span>
            {tauriMode
              ? t("workbench.artifacts.banner.detail.tauri")
              : t("workbench.artifacts.banner.detail.mock")}
          </span>
        </div>
        <StatusBadge
          status={tauriMode ? "success" : "info"}
          label={
            tauriMode
              ? t("workbench.artifacts.banner.status.tauri")
              : t("workbench.artifacts.banner.status.mock")
          }
          subtle
        />
      </div>

      <div className="artifact-grid">
        {artifacts.map((artifact) => (
          <article className="artifact-card" key={artifact.id}>
            <div className="artifact-card__top">
              <span className={`artifact-card__icon artifact-card__icon--${artifact.kind}`} aria-hidden="true">
                <Icon name={artifactIcons[artifact.kind]} size={22} />
              </span>
              <StatusBadge status={artifact.integrity} subtle />
            </div>
            <div className="artifact-card__copy">
              <span>{t(artifactTypeLabels[artifact.kind])}</span>
              <h2>{artifact.name}</h2>
              <code title={artifact.relativePath}>{artifact.relativePath}</code>
            </div>
            <dl className="artifact-card__facts">
              <div>
                <dt>{t("workbench.artifacts.fact.size")}</dt>
                <dd>{artifact.sizeLabel}</dd>
              </div>
              <div>
                <dt>{t("workbench.artifacts.fact.created")}</dt>
                <dd>{artifact.createdAt}</dd>
              </div>
              <div>
                <dt>{t("workbench.artifacts.fact.digest")}</dt>
                <dd>{artifact.sha256 ?? t("common.pending")}</dd>
              </div>
            </dl>
            <button
              className="button button--soft button--full"
              type="button"
              disabled={busyAction === artifact.id}
              onClick={() => {
                onOpen(artifact.id).catch((error: unknown) => {
                  console.error("Failed to open artifact", error);
                });
              }}
            >
              <Icon name="external" size={17} />
              {busyAction === artifact.id
                ? t("common.checking")
                : t("common.openLocally")}
            </button>
          </article>
        ))}
      </div>
    </section>
  );
}
