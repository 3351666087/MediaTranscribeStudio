import type { PipelineStage } from "../contracts/studio";
import { useI18n } from "../i18n";
import { cx } from "../lib/format";
import { StateIcon } from "./StatusBadge";

interface PipelineOverviewProps {
  stages: PipelineStage[];
}

export function PipelineOverview({ stages }: PipelineOverviewProps) {
  const { t } = useI18n();
  const completed = stages.filter((stage) => stage.status === "completed").length;

  return (
    <section className="panel pipeline-panel" aria-labelledby="pipeline-title">
      <div className="panel__header">
        <div>
          <span className="panel__eyebrow">
            {t("workbench.pipeline.eyebrow")}
          </span>
          <h2 id="pipeline-title">
            {t("workbench.pipeline.title", {
              count: stages.length,
            })}
          </h2>
        </div>
        <span className="count-chip">
          {t("common.complete", {
            complete: completed,
            total: stages.length,
          })}
        </span>
      </div>

      <ol className="pipeline-list">
        {stages.map((stage, index) => (
          <li className={cx("pipeline-stage", `pipeline-stage--${stage.status}`)} key={stage.id}>
            <div className="pipeline-stage__rail" aria-hidden="true">
              <StateIcon status={stage.status} />
              {index < stages.length - 1 ? <span /> : null}
            </div>
            <div className="pipeline-stage__body">
              <div className="pipeline-stage__heading">
                <span>
                  <small>{String(index + 1).padStart(2, "0")}</small>
                  <strong>{stage.label}</strong>
                </span>
                <span className="pipeline-stage__meta">
                  {stage.durationLabel ? <time>{stage.durationLabel}</time> : null}
                  <strong>{stage.progress}%</strong>
                </span>
              </div>
              <p>{stage.detail}</p>
              <div
                className="linear-progress"
                role="progressbar"
                aria-label={t("workbench.pipeline.stageProgress", {
                  stage: stage.label,
                })}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={stage.progress}
              >
                <span style={{ width: `${stage.progress}%` }} />
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
