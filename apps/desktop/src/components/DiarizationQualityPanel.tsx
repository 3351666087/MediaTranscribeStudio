import type {
  DiarizationQualityMetrics,
  PercentMetric,
} from "../contracts/studio";
import { useI18n, type MessageKey } from "../i18n";
import { Icon } from "./Icon";

interface DiarizationQualityPanelProps {
  metrics: DiarizationQualityMetrics;
}

const qualityItems: Array<{
  key: keyof DiarizationQualityMetrics;
  label: MessageKey;
  shortLabel: MessageKey;
  description: MessageKey;
  needsReference: boolean;
}> = [
  {
    key: "der",
    label: "workbench.quality.metric.der.label",
    shortLabel: "workbench.quality.metric.der.short",
    description: "workbench.quality.metric.der.description",
    needsReference: true,
  },
  {
    key: "jer",
    label: "workbench.quality.metric.jer.label",
    shortLabel: "workbench.quality.metric.jer.short",
    description: "workbench.quality.metric.jer.description",
    needsReference: true,
  },
  {
    key: "confusion",
    label: "workbench.quality.metric.confusion.label",
    shortLabel: "workbench.quality.metric.confusion.short",
    description: "workbench.quality.metric.confusion.description",
    needsReference: true,
  },
  {
    key: "overlapF1",
    label: "workbench.quality.metric.overlapF1.label",
    shortLabel: "workbench.quality.metric.overlapF1.short",
    description: "workbench.quality.metric.overlapF1.description",
    needsReference: true,
  },
  {
    key: "reviewRate",
    label: "workbench.quality.metric.reviewRate.label",
    shortLabel: "workbench.quality.metric.reviewRate.short",
    description: "workbench.quality.metric.reviewRate.description",
    needsReference: false,
  },
];

function formatMetricValue(metric: Extract<PercentMetric, { status: "available" }>): string {
  const fractionDigits = metric.value > 0 && metric.value < 10 ? 2 : 1;
  return `${metric.value.toFixed(fractionDigits)}%`;
}

export function DiarizationQualityPanel({
  metrics,
}: DiarizationQualityPanelProps) {
  const { t } = useI18n();

  return (
    <section className="panel quality-panel" aria-labelledby="quality-panel-title">
      <header className="quality-panel__header">
        <div>
          <span className="panel__eyebrow">
            {t("workbench.quality.eyebrow")}
          </span>
          <h2 id="quality-panel-title">
            {t("workbench.quality.title")}
          </h2>
          <p>{t("workbench.quality.description")}</p>
        </div>
        <span className="quality-panel__mascot" aria-hidden="true">
          <Icon name="shield" size={22} />
        </span>
      </header>

      <div className="quality-panel__grid">
        {qualityItems.map((item) => {
          const metric = metrics[item.key];
          const available = metric.status === "available";
          return (
            <article
              className={`metric-card ${
                available ? "metric-card--available" : "metric-card--unavailable"
              }`}
              key={item.key}
            >
              <div className="metric-card__heading">
                <span>{t(item.shortLabel)}</span>
                <span
                  className={`metric-card__state ${
                    available
                      ? "metric-card__state--available"
                      : "metric-card__state--unavailable"
                  }`}
                >
                  {available ? t("common.measured") : t("common.unavailable")}
                </span>
              </div>
              <strong>{available ? formatMetricValue(metric) : "—"}</strong>
              <h3>{t(item.label)}</h3>
              <p>{t(item.description)}</p>
              <small>
                {available
                  ? metric.source
                  : item.needsReference
                    ? t("workbench.quality.referenceMissing", {
                        reason: metric.reason,
                      })
                    : metric.reason}
              </small>
            </article>
          );
        })}
      </div>

      <footer className="quality-panel__note">
        <Icon name="alert" size={17} />
        <p>
          <strong>{t("workbench.quality.note.title")}</strong>{" "}
          {t("workbench.quality.note.detail")}
        </p>
      </footer>
    </section>
  );
}
