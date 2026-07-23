import type { PdfQualityReport } from "../contracts/studio";
import { useI18n } from "../i18n";
import { cx } from "../lib/format";
import { Icon } from "./Icon";
import { ProgressRing } from "./ProgressRing";
import { StatusBadge } from "./StatusBadge";

interface PdfQaPanelProps {
  report: PdfQualityReport;
}

export function PdfQaPanel({ report }: PdfQaPanelProps) {
  const { t } = useI18n();
  const hardGatePassCount = report.hardGates.filter((gate) => gate.status === "passed").length;
  const facetPassCount = report.facets.filter((facet) => facet.status === "passed").length;
  const scoreState =
    report.status === "pending"
      ? {
          badgeStatus: "pending" as const,
          badgeLabel: t("workbench.pdf.score.pending.badge"),
          message: t("workbench.pdf.score.pending.message"),
        }
      : report.score >= report.minimumScore
        ? {
            badgeStatus: report.status === "passed" ? ("verified" as const) : ("warning" as const),
            badgeLabel:
              report.status === "passed"
                ? t("workbench.pdf.score.passed.badge")
                : t("workbench.pdf.score.hardGateReview.badge"),
            message: t(
              "workbench.pdf.score.thresholdMet.message",
            ),
          }
        : {
            badgeStatus: report.status === "blocked" ? ("blocked" as const) : ("warning" as const),
            badgeLabel:
              report.status === "blocked"
                ? t("workbench.pdf.score.blocked.badge")
                : t("workbench.pdf.score.repair.badge"),
            message: t(
              "workbench.pdf.score.thresholdMissed.message",
            ),
          };

  return (
    <section className="pdf-qa-page" aria-labelledby="pdf-qa-title">
      <div className="page-heading">
        <div>
          <span className="panel__eyebrow">
            {t("workbench.pdf.eyebrow")}
          </span>
          <h1 id="pdf-qa-title">{t("workbench.pdf.title")}</h1>
          <p>{t("workbench.pdf.description")}</p>
        </div>
        <StatusBadge status={scoreState.badgeStatus} label={scoreState.badgeLabel} />
      </div>

      <div className="qa-summary-grid">
        <article className="qa-score-card">
          <div className="qa-score-card__ring">
            <ProgressRing
              value={report.score}
              label={t("workbench.pdf.visualScoreAria")}
              size={126}
            />
          </div>
          <div>
            <span className="panel__eyebrow">
              {t("workbench.pdf.pass", {
                current: report.passNumber,
                total: 5,
              })}
            </span>
            <h2>
              {t("workbench.pdf.visualScore", {
                score: report.score,
              })}
            </h2>
            <p>
              {t("workbench.pdf.minimumScore", {
                score: report.minimumScore,
              })}{" "}
              {scoreState.message}
            </p>
            <small>{report.renderedAt}</small>
          </div>
        </article>

        <article className="qa-policy-card">
          <span className="qa-policy-card__icon" aria-hidden="true">
            <Icon name="shield" size={24} />
          </span>
          <div>
            <span className="panel__eyebrow">
              {t("workbench.pdf.policy.eyebrow")}
            </span>
            <h2>{t("workbench.pdf.policy.title")}</h2>
            <p>{t("workbench.pdf.policy.detail")}</p>
          </div>
          <dl>
            <div>
              <dt>{t("workbench.pdf.pages")}</dt>
              <dd>{report.pageCount}</dd>
            </div>
            <div>
              <dt>{t("workbench.pdf.evidenceDigest")}</dt>
              <dd>{report.evidenceDigest}</dd>
            </div>
          </dl>
        </article>
      </div>

      <div className="qa-content-grid">
        <section className="panel hard-gates-panel" aria-labelledby="hard-gates-title">
          <div className="panel__header">
            <div>
              <span className="panel__eyebrow">
                {t("workbench.pdf.hardGates.eyebrow")}
              </span>
              <h2 id="hard-gates-title">
                {t("workbench.pdf.hardGates.title", {
                  count: report.hardGates.length,
                })}
              </h2>
            </div>
            <span className="count-chip">{hardGatePassCount} / {report.hardGates.length}</span>
          </div>
          <ul className="hard-gate-list">
            {report.hardGates.map((gate, index) => (
              <li key={gate.id}>
                <span className={cx("hard-gate-list__index", `hard-gate-list__index--${gate.status}`)}>
                  {gate.status === "passed" ? <Icon name="check" size={15} /> : index + 1}
                </span>
                <span className="hard-gate-list__copy">
                  <strong>{gate.label}</strong>
                  <small>{gate.id}</small>
                  <span>{gate.detail}</span>
                </span>
                <StatusBadge
                  status={gate.status === "passed" ? "verified" : gate.status === "failed" ? "failed" : "pending"}
                  subtle
                />
              </li>
            ))}
          </ul>
        </section>

        <section className="panel facet-panel" aria-labelledby="facet-title">
          <div className="panel__header">
            <div>
              <span className="panel__eyebrow">
                {t("workbench.pdf.facets.eyebrow")}
              </span>
              <h2 id="facet-title">
                {t("workbench.pdf.facets.title", {
                  count: report.facets.length,
                })}
              </h2>
            </div>
            <span className="count-chip">{facetPassCount} / {report.facets.length}</span>
          </div>
          <div className="facet-list">
            {report.facets.map((facet) => (
              <div className="facet-row" key={facet.id}>
                <div className="facet-row__title">
                  <strong>{facet.label}</strong>
                  <span>{facet.score}</span>
                </div>
                <div
                  className="facet-row__bar"
                  role="progressbar"
                  aria-label={t("workbench.pdf.facets.scoreAria", {
                    label: facet.label,
                  })}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={facet.score}
                >
                  <span style={{ width: `${facet.score}%` }} />
                </div>
                <small title={facet.id}>{facet.evidence}</small>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="panel repair-panel" aria-labelledby="repair-title">
        <div className="panel__header">
          <div>
            <span className="panel__eyebrow">
              {t("workbench.pdf.repairs.eyebrow")}
            </span>
            <h2 id="repair-title">
              {t("workbench.pdf.repairs.title")}
            </h2>
          </div>
          <StatusBadge
            status="success"
            label={t("workbench.pdf.repairs.protection")}
            subtle
          />
        </div>
        <div className="repair-guardrail">
          <Icon name="lock" size={18} />
          <p>
            {t("workbench.pdf.repairs.guardrail.scope")}{" "}
            <strong>
              {t("workbench.pdf.repairs.guardrail.protected")}
            </strong>
          </p>
        </div>
        <div className="repair-list">
          {report.repairQueue.map((repair) => (
            <article key={repair.id}>
              <span className="repair-list__priority">P{repair.priority}</span>
              <div>
                <span>{repair.sourceId}</span>
                <h3>{repair.title}</h3>
                <p>{repair.detail}</p>
              </div>
              <div className="repair-list__state">
                <code>{repair.safeScope}</code>
                <StatusBadge
                  status={repair.status === "applied" ? "verified" : repair.status === "blocked" ? "blocked" : "pending"}
                  label={
                    repair.status === "applied"
                      ? t("workbench.pdf.repairs.status.applied")
                      : repair.status === "blocked"
                        ? t("workbench.pdf.repairs.status.blocked")
                        : t("workbench.pdf.repairs.status.pending")
                  }
                  subtle
                />
              </div>
            </article>
          ))}
        </div>
      </section>
    </section>
  );
}
