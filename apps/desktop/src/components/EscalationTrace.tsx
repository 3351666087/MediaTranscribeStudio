import type { ReviewSegment } from "../contracts/studio";
import {
  deriveEscalationCoverage,
  eres2NetReasons,
  pyannoteReasons,
} from "../lib/escalation";
import { useI18n } from "../i18n";
import { cx } from "../lib/format";
import { Icon } from "./Icon";

interface EscalationTraceProps {
  reviews: readonly ReviewSegment[];
  review?: ReviewSegment;
  compact?: boolean;
}

export function EscalationTrace({
  reviews,
  review,
  compact = false,
}: EscalationTraceProps) {
  const { t } = useI18n();
  const coverage = deriveEscalationCoverage(reviews);
  const eres2netActive =
    review?.reasons.some((reason) => eres2NetReasons.has(reason)) ?? false;
  const pyannoteActive =
    review?.reasons.some((reason) => pyannoteReasons.has(reason)) ?? false;

  const stages = [
    {
      id: "campp",
      label: "CAM++",
      scope: review
        ? t("workbench.escalation.campp.segment")
        : t("workbench.escalation.campp.queue", {
            count: coverage.total,
          }),
      active: true,
      tone: "base",
    },
    {
      id: "eres2net",
      label: "ERes2NetV2",
      scope: review
        ? eres2netActive
          ? t("workbench.escalation.eres2net.segmentActive")
          : t("workbench.escalation.eres2net.segmentInactive")
        : t("workbench.escalation.eres2net.queue", {
            count: coverage.eres2net,
          }),
      active: review ? eres2netActive : coverage.eres2net > 0,
      tone: "secondary",
    },
    {
      id: "pyannote",
      label: "pyannote",
      scope: review
        ? pyannoteActive
          ? t("workbench.escalation.pyannote.segmentActive")
          : t("workbench.escalation.pyannote.segmentInactive")
        : t("workbench.escalation.pyannote.queue", {
            count: coverage.pyannote,
          }),
      active: review ? pyannoteActive : coverage.pyannote > 0,
      tone: "fallback",
    },
    {
      id: "human",
      label: "REVIEW_REQUIRED",
      scope: review
        ? t("workbench.escalation.human.segment")
        : t("workbench.escalation.human.queue", {
            count: coverage.reviewRequired,
          }),
      active: true,
      tone: "human",
    },
  ] as const;

  return (
    <section
      className={cx("escalation-trace", compact && "escalation-trace--compact")}
      aria-label={
        review
          ? t("workbench.escalation.aria.segment")
          : t("workbench.escalation.aria.queue")
      }
    >
      <header className="escalation-trace__header">
        <span>
          <Icon name="sparkles" size={15} />
          {t("workbench.escalation.title")}
        </span>
        <small>{t("workbench.escalation.disclaimer")}</small>
      </header>
      <ol className="escalation-trace__stages">
        {stages.map((stage, index) => (
          <li
            className={cx(
              "escalation-stage",
              `escalation-stage--${stage.tone}`,
              stage.active && "escalation-stage--active",
            )}
            key={stage.id}
          >
            <span className="escalation-stage__index">{index + 1}</span>
            <span className="escalation-stage__copy">
              <strong>{stage.label}</strong>
              <small>{stage.scope}</small>
            </span>
            {index < stages.length - 1 ? (
              <Icon
                className="escalation-stage__arrow"
                name="arrow-right"
                size={14}
              />
            ) : null}
          </li>
        ))}
      </ol>
    </section>
  );
}
