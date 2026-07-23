import type {
  PerformanceMetrics,
  PipelineStage,
  ReviewSegment,
} from "../contracts/studio";
import { useI18n } from "../i18n";
import { deriveEscalationCoverage } from "../lib/escalation";
import { EscalationTrace } from "./EscalationTrace";
import { Icon } from "./Icon";

interface PerformanceWorkbenchProps {
  metrics: PerformanceMetrics;
  stages: PipelineStage[];
  reviews?: ReviewSegment[];
}

function formatPercent(value: number): string {
  return `${value.toFixed(value < 10 ? 1 : 0)}%`;
}

function formatLatency(milliseconds: number): string {
  if (milliseconds >= 1_000) {
    return `${(milliseconds / 1_000).toFixed(2)} s`;
  }
  return `${Math.round(milliseconds)} ms`;
}

export function PerformanceWorkbench({
  metrics,
  stages,
  reviews = [],
}: PerformanceWorkbenchProps) {
  const { t } = useI18n();
  const stageLabels = new Map(stages.map((stage) => [stage.id, stage.label]));
  const coverage = deriveEscalationCoverage(reviews);

  return (
    <section
      className="panel performance-workbench"
      aria-labelledby="performance-workbench-title"
    >
      <header className="performance-workbench__header">
        <div>
          <span className="panel__eyebrow">
            {t("workbench.performance.eyebrow")}
          </span>
          <h2 id="performance-workbench-title">
            {t("workbench.performance.title")}
          </h2>
          <p>{t("workbench.performance.description")}</p>
        </div>
        <span className="performance-workbench__mascot" aria-hidden="true">
          <Icon name="sparkles" size={22} />
        </span>
      </header>

      <EscalationTrace reviews={reviews} />

      <div className="performance-provenance">
        <Icon name="shield" size={16} />
        <p>
          {metrics.status === "measured"
            ? t("workbench.performance.provenance.measured", {
                source: metrics.sourceLabel,
              })
            : t("workbench.performance.provenance.unavailable")}
        </p>
      </div>

      <dl
        className="escalation-coverage-kpis"
        aria-label={t("workbench.performance.coverageAria")}
      >
        <div>
          <dt>{t("workbench.performance.coverage.eres2net.title")}</dt>
          <dd>{coverage.eres2net}</dd>
          <small>{t("workbench.performance.coverage.eres2net.detail")}</small>
        </div>
        <div>
          <dt>{t("workbench.performance.coverage.pyannote.title")}</dt>
          <dd>{coverage.pyannote}</dd>
          <small>{t("workbench.performance.coverage.pyannote.detail")}</small>
        </div>
        <div>
          <dt>{t("workbench.performance.coverage.lowMargin.title")}</dt>
          <dd>{coverage.lowMargin}</dd>
          <small>{t("workbench.performance.coverage.lowMargin.detail")}</small>
        </div>
        <div>
          <dt>{t("workbench.performance.coverage.human.title")}</dt>
          <dd>{coverage.reviewRequired}</dd>
          <small>REVIEW_REQUIRED</small>
        </div>
      </dl>

      {metrics.status === "unavailable" ? (
        <div className="performance-workbench__empty" role="status">
          <span aria-hidden="true">
            <Icon name="clock" size={23} />
          </span>
          <div>
            <strong>{t("workbench.performance.unavailable.title")}</strong>
            <p>{metrics.reason}</p>
            <small>{t("workbench.performance.unavailable.detail")}</small>
          </div>
        </div>
      ) : (
        <>
          <div className="performance-workbench__headline">
            <article className="rtf-card">
              <span>{t("workbench.performance.rtf.title")}</span>
              <strong>{metrics.rtf.toFixed(2)}×</strong>
              <small>
                {metrics.rtf <= 1
                  ? t("workbench.performance.rtf.faster")
                  : t("workbench.performance.rtf.slower")}
              </small>
            </article>

            <dl className="performance-kpis">
              <div>
                <dt>{t("workbench.performance.kpi.cacheHit")}</dt>
                <dd>{formatPercent(metrics.cacheHitRate)}</dd>
              </div>
              <div>
                <dt>{t("workbench.performance.kpi.escalation")}</dt>
                <dd>{formatPercent(metrics.selectiveEscalationRate)}</dd>
              </div>
              <div>
                <dt>{t("workbench.performance.kpi.recompute")}</dt>
                <dd>{formatPercent(metrics.recomputeRate)}</dd>
              </div>
            </dl>
          </div>

          <div className="performance-stage-table">
            <div className="performance-stage-table__heading">
              <h3>{t("workbench.performance.latency.title")}</h3>
              <span>{metrics.sourceLabel}</span>
            </div>
            <table>
              <caption className="sr-only">
                {t("workbench.performance.latency.caption")}
              </caption>
              <thead>
                <tr>
                  <th scope="col">
                    {t("workbench.performance.latency.stage")}
                  </th>
                  <th scope="col">p50</th>
                  <th scope="col">p95</th>
                  <th scope="col">
                    {t("workbench.performance.latency.tailDelta")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {metrics.stageLatency.map((stage) => (
                  <tr key={stage.stageId}>
                    <th scope="row">{stageLabels.get(stage.stageId) ?? stage.stageId}</th>
                    <td data-label="p50">{formatLatency(stage.p50Ms)}</td>
                    <td data-label="p95">{formatLatency(stage.p95Ms)}</td>
                    <td
                      data-label={t(
                        "workbench.performance.latency.tailDelta",
                      )}
                    >
                      {formatLatency(stage.p95Ms - stage.p50Ms)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="peak-resource-card">
            <div>
              <span className="peak-resource-card__icon" aria-hidden="true">
                <Icon name="cpu" size={18} />
              </span>
              <span>
                <strong>{t("workbench.performance.resources.title")}</strong>
                <small>{t("workbench.performance.resources.detail")}</small>
              </span>
            </div>
            <dl>
              <div>
                <dt>CPU</dt>
                <dd>{formatPercent(metrics.peakResources.cpuPercent)}</dd>
              </div>
              <div>
                <dt>RAM</dt>
                <dd>{metrics.peakResources.ramGb.toFixed(1)} GB</dd>
              </div>
              <div>
                <dt>VRAM</dt>
                <dd>{metrics.peakResources.vramGb.toFixed(1)} GB</dd>
              </div>
            </dl>
          </div>
        </>
      )}
    </section>
  );
}
