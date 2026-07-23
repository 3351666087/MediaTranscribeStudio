import type {
  AestheticFacetId,
  PdfHardGateId,
  PdfQualityReport,
} from "../contracts/studio";
import { useI18n, type MessageKey } from "../i18n";
import { cx } from "../lib/format";
import { Icon } from "./Icon";
import { ProgressRing } from "./ProgressRing";
import { StatusBadge } from "./StatusBadge";

interface PdfQaPanelProps {
  report: PdfQualityReport;
}

interface LocalizedQaItem {
  label: MessageKey;
  detail: MessageKey;
}

const HARD_GATE_COPY = {
  "PDF-OPENABLE": {
    label: "workbench.pdf.hardGates.item.openable.label",
    detail: "workbench.pdf.hardGates.item.openable.detail",
  },
  "PDF-PAGE-COUNT": {
    label: "workbench.pdf.hardGates.item.pageCount.label",
    detail: "workbench.pdf.hardGates.item.pageCount.detail",
  },
  "PDF-PAGE-SIZE": {
    label: "workbench.pdf.hardGates.item.pageSize.label",
    detail: "workbench.pdf.hardGates.item.pageSize.detail",
  },
  "PDF-TRANSCRIPT-TEXT-INTEGRITY": {
    label: "workbench.pdf.hardGates.item.transcriptText.label",
    detail: "workbench.pdf.hardGates.item.transcriptText.detail",
  },
  "PDF-SEGMENT-COUNT": {
    label: "workbench.pdf.hardGates.item.segmentCount.label",
    detail: "workbench.pdf.hardGates.item.segmentCount.detail",
  },
  "PDF-TIMESTAMP-INTEGRITY": {
    label: "workbench.pdf.hardGates.item.timestamps.label",
    detail: "workbench.pdf.hardGates.item.timestamps.detail",
  },
  "PDF-SPEAKER-SET-INTEGRITY": {
    label: "workbench.pdf.hardGates.item.speakerSet.label",
    detail: "workbench.pdf.hardGates.item.speakerSet.detail",
  },
  "PDF-FONT-EMBEDDED": {
    label: "workbench.pdf.hardGates.item.fonts.label",
    detail: "workbench.pdf.hardGates.item.fonts.detail",
  },
  "PDF-NO-BLANK-PAGES": {
    label: "workbench.pdf.hardGates.item.noBlankPages.label",
    detail: "workbench.pdf.hardGates.item.noBlankPages.detail",
  },
  "PDF-NO-CONTENT-OVERFLOW": {
    label: "workbench.pdf.hardGates.item.noOverflow.label",
    detail: "workbench.pdf.hardGates.item.noOverflow.detail",
  },
  "PDF-OFFLINE-ASSETS": {
    label: "workbench.pdf.hardGates.item.offlineAssets.label",
    detail: "workbench.pdf.hardGates.item.offlineAssets.detail",
  },
  "PDF-PAGE-EVIDENCE": {
    label: "workbench.pdf.hardGates.item.pageEvidence.label",
    detail: "workbench.pdf.hardGates.item.pageEvidence.detail",
  },
  "PDF-IMMUTABLE-CONTENT-HASH": {
    label: "workbench.pdf.hardGates.item.contentHash.label",
    detail: "workbench.pdf.hardGates.item.contentHash.detail",
  },
} as const satisfies Record<PdfHardGateId, LocalizedQaItem>;

const FACET_COPY = {
  "AESTHETIC-COHERENCE": {
    label: "workbench.pdf.facets.item.coherence.label",
    detail: "workbench.pdf.facets.item.coherence.detail",
  },
  "AESTHETIC-DISTINCTION": {
    label: "workbench.pdf.facets.item.distinction.label",
    detail: "workbench.pdf.facets.item.distinction.detail",
  },
  "AESTHETIC-REFINEMENT": {
    label: "workbench.pdf.facets.item.refinement.label",
    detail: "workbench.pdf.facets.item.refinement.detail",
  },
  "AESTHETIC-PROPORTION": {
    label: "workbench.pdf.facets.item.proportion.label",
    detail: "workbench.pdf.facets.item.proportion.detail",
  },
  "AESTHETIC-HIERARCHY": {
    label: "workbench.pdf.facets.item.hierarchy.label",
    detail: "workbench.pdf.facets.item.hierarchy.detail",
  },
  "AESTHETIC-TYPOGRAPHY": {
    label: "workbench.pdf.facets.item.typography.label",
    detail: "workbench.pdf.facets.item.typography.detail",
  },
  "AESTHETIC-COLOR-RELATIONSHIPS": {
    label: "workbench.pdf.facets.item.color.label",
    detail: "workbench.pdf.facets.item.color.detail",
  },
  "AESTHETIC-RHYTHM": {
    label: "workbench.pdf.facets.item.rhythm.label",
    detail: "workbench.pdf.facets.item.rhythm.detail",
  },
  "AESTHETIC-DENSITY": {
    label: "workbench.pdf.facets.item.density.label",
    detail: "workbench.pdf.facets.item.density.detail",
  },
  "AESTHETIC-RESTRAINT": {
    label: "workbench.pdf.facets.item.restraint.label",
    detail: "workbench.pdf.facets.item.restraint.detail",
  },
  "AESTHETIC-REAL-CONTENT-STRESS": {
    label: "workbench.pdf.facets.item.realContent.label",
    detail: "workbench.pdf.facets.item.realContent.detail",
  },
  "AESTHETIC-FONT-FAILURE": {
    label: "workbench.pdf.facets.item.fontFailure.label",
    detail: "workbench.pdf.facets.item.fontFailure.detail",
  },
  "AESTHETIC-IMAGE-FAILURE": {
    label: "workbench.pdf.facets.item.imageFailure.label",
    detail: "workbench.pdf.facets.item.imageFailure.detail",
  },
  "AESTHETIC-SCRIPT-FAILURE": {
    label: "workbench.pdf.facets.item.scriptFailure.label",
    detail: "workbench.pdf.facets.item.scriptFailure.detail",
  },
} as const satisfies Record<AestheticFacetId, LocalizedQaItem>;

export function PdfQaPanel({ report }: PdfQaPanelProps) {
  const { t } = useI18n();
  const hardGatePassCount = report.hardGates.filter((gate) => gate.status === "passed").length;
  const facetPassCount = report.facets.filter((facet) => facet.status === "passed").length;
  const scoreIsAvailable = report.status !== "pending";
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
      </div>

      <div className="qa-summary-grid">
        <article
          className="qa-score-card pdf-qa-score-hero"
          aria-labelledby="pdf-qa-score-title"
          aria-live="polite"
        >
          <div className="qa-score-card__ring pdf-qa-score-hero__meter">
            {scoreIsAvailable ? (
              <ProgressRing
                value={report.score}
                label={t("workbench.pdf.visualScoreAria")}
                size={144}
              />
            ) : (
              <div
                className="pdf-qa-score-hero__unavailable"
                role="img"
                aria-label={`${t("workbench.pdf.visualScoreAria")} —`}
              >
                <strong aria-hidden="true">—</strong>
                <small>{scoreState.badgeLabel}</small>
              </div>
            )}
          </div>
          <div className="pdf-qa-score-hero__copy">
            <span className="panel__eyebrow">
              {t("workbench.pdf.pass", {
                current: report.passNumber,
                total: 5,
              })}
            </span>
            <h2 id="pdf-qa-score-title">
              {t("workbench.pdf.visualScoreAria")}
            </h2>
            <div className="pdf-qa-score-hero__status">
              <StatusBadge
                status={scoreState.badgeStatus}
                label={scoreState.badgeLabel}
              />
            </div>
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

      <div className="qa-content-grid pdf-qa-disclosures">
        <details
          className="panel hard-gates-panel pdf-qa-disclosure pdf-qa-disclosure--hard-gates"
          aria-labelledby="hard-gates-title"
        >
          <summary className="pdf-qa-disclosure__summary">
            <span className="pdf-qa-disclosure__heading">
              <span className="panel__eyebrow">
                {t("workbench.pdf.hardGates.eyebrow")}
              </span>
              <strong id="hard-gates-title">
                {t("workbench.pdf.hardGates.title", {
                  count: report.hardGates.length,
                })}
              </strong>
            </span>
            <span className="pdf-qa-disclosure__meta">
              <span className="count-chip">
                {hardGatePassCount} / {report.hardGates.length}
              </span>
              <Icon
                className="pdf-qa-disclosure__chevron"
                name="chevron-down"
                size={18}
              />
            </span>
          </summary>
          <div className="pdf-qa-disclosure__body">
            <ul className="hard-gate-list">
              {report.hardGates.map((gate, index) => {
                const copy = HARD_GATE_COPY[gate.id];
                return (
                  <li key={gate.id}>
                    <span
                      className={cx(
                        "hard-gate-list__index",
                        `hard-gate-list__index--${gate.status}`,
                      )}
                    >
                      {gate.status === "passed" ? (
                        <Icon name="check" size={15} />
                      ) : (
                        index + 1
                      )}
                    </span>
                    <span className="hard-gate-list__copy">
                      <strong>{t(copy.label)}</strong>
                      <small>{gate.id}</small>
                      <span>{t(copy.detail)}</span>
                    </span>
                    <StatusBadge
                      status={
                        gate.status === "passed"
                          ? "verified"
                          : gate.status === "failed"
                            ? "failed"
                            : "pending"
                      }
                      subtle
                    />
                  </li>
                );
              })}
            </ul>
          </div>
        </details>

        <details
          className="panel facet-panel pdf-qa-disclosure pdf-qa-disclosure--facets"
          aria-labelledby="facet-title"
        >
          <summary className="pdf-qa-disclosure__summary">
            <span className="pdf-qa-disclosure__heading">
              <span className="panel__eyebrow">
                {t("workbench.pdf.facets.eyebrow")}
              </span>
              <strong id="facet-title">
                {t("workbench.pdf.facets.title", {
                  count: report.facets.length,
                })}
              </strong>
            </span>
            <span className="pdf-qa-disclosure__meta">
              <span className="count-chip">
                {facetPassCount} / {report.facets.length}
              </span>
              <Icon
                className="pdf-qa-disclosure__chevron"
                name="chevron-down"
                size={18}
              />
            </span>
          </summary>
          <div className="pdf-qa-disclosure__body">
            <div className="facet-list">
              {report.facets.map((facet) => {
                const copy = FACET_COPY[facet.id];
                const label = t(copy.label);
                return (
                  <div className="facet-row" key={facet.id}>
                    <div className="facet-row__title">
                      <strong>{label}</strong>
                      <span>{facet.score}</span>
                    </div>
                    <div
                      className="facet-row__bar"
                      role="progressbar"
                      aria-label={t("workbench.pdf.facets.scoreAria", {
                        label,
                      })}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={facet.score}
                    >
                      <span style={{ width: `${facet.score}%` }} />
                    </div>
                    <small title={facet.id}>{t(copy.detail)}</small>
                  </div>
                );
              })}
            </div>
          </div>
        </details>
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
