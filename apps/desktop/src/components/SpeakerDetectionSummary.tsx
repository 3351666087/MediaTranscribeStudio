import type {
  SpeakerCountDetection,
  SpeakerCountPolicy,
} from "../contracts/studio";
import { useI18n } from "../i18n";
import { Icon } from "./Icon";

interface SpeakerDetectionSummaryProps {
  detection: SpeakerCountDetection | null;
  policy: SpeakerCountPolicy;
  compact?: boolean;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;

export function SpeakerDetectionSummary({
  detection,
  policy,
  compact = false,
}: SpeakerDetectionSummaryProps) {
  const { t } = useI18n();

  if (policy.mode === "manual") {
    return (
      <div className="speaker-detection speaker-detection--manual" role="status">
        <Icon name="lock" size={17} />
        <span>
          <strong>
            {t("workbench.speakerDetection.manual.title", {
              count: policy.count,
            })}
          </strong>
          <small>
            {t("workbench.speakerDetection.manual.detail", {
              count: policy.count,
            })}
          </small>
        </span>
      </div>
    );
  }

  if (!detection) {
    return (
      <div className="speaker-detection speaker-detection--waiting" role="status">
        <Icon name="wand" size={17} />
        <span>
          <strong>
            {t("workbench.speakerDetection.waiting.title")}
          </strong>
          <small>
            {t("workbench.speakerDetection.waiting.detail")}
          </small>
        </span>
      </div>
    );
  }

  return (
    <section
      className={compact ? "speaker-detection speaker-detection--compact" : "speaker-detection"}
      aria-label={t("workbench.speakerDetection.resultAria")}
    >
      <div className="speaker-detection__headline">
        <span className="speaker-detection__spark" aria-hidden="true">
          <Icon name="sparkles" size={17} />
        </span>
        <span>
          <small>
            {t("workbench.speakerDetection.estimatedCount")}
          </small>
          <strong>
            {t("workbench.speakerDetection.speakerCount", {
              count: detection.estimatedCount,
            })}
          </strong>
        </span>
        <span>
          <small>{t("workbench.speakerDetection.confidence")}</small>
          <strong>{percent(detection.confidence)}</strong>
        </span>
        <meter
          min={0}
          max={1}
          low={0.58}
          high={0.82}
          optimum={1}
          value={detection.confidence}
          aria-label={t(
            "workbench.speakerDetection.confidenceAria",
            {
              confidence: percent(detection.confidence),
            },
          )}
        >
          {percent(detection.confidence)}
        </meter>
      </div>

      <div
        className="speaker-detection__candidates"
        aria-label={t("workbench.speakerDetection.candidatesAria")}
      >
        {detection.candidates.map((candidate, index) => (
          <span
            className={index === 0 ? "candidate-chip candidate-chip--primary" : "candidate-chip"}
            key={candidate.count}
          >
            {t("workbench.speakerDetection.candidate", {
              count: candidate.count,
              confidence: percent(candidate.confidence),
            })}
          </span>
        ))}
      </div>

      <ol
        className="speaker-cascade"
        aria-label={t("workbench.speakerDetection.cascadeAria")}
      >
        <li className="speaker-cascade__step speaker-cascade__step--done">
          {t("workbench.speakerDetection.cascade.count")}
        </li>
        <li className="speaker-cascade__step speaker-cascade__step--done">
          {t("workbench.speakerDetection.cascade.voiceprint")}
        </li>
        <li className="speaker-cascade__step speaker-cascade__step--active">
          {t("workbench.speakerDetection.cascade.audioReview")}
        </li>
        <li className="speaker-cascade__step">
          {t("workbench.speakerDetection.cascade.humanLock")}
        </li>
      </ol>

      {detection.provider ? (
        <small className="speaker-detection__provider">
          {t("workbench.speakerDetection.provider", {
            provider: detection.provider,
          })}
        </small>
      ) : null}
    </section>
  );
}
