import type { ModelStrategy, ModelStrategyId } from "../contracts/studio";
import { useI18n } from "../i18n";
import { cx } from "../lib/format";
import { Icon } from "./Icon";

interface ModelStrategyPanelProps {
  strategies: ModelStrategy[];
  selectedId: ModelStrategyId;
  onSelect: (strategyId: ModelStrategyId) => void;
}

export function ModelStrategyPanel({
  strategies,
  selectedId,
  onSelect,
}: ModelStrategyPanelProps) {
  const { t } = useI18n();
  const selected = strategies.find((strategy) => strategy.id === selectedId);

  return (
    <section className="panel strategy-panel" aria-labelledby="strategy-panel-title">
      <div className="panel__header">
        <div>
          <span className="panel__eyebrow">
            {t("workbench.strategy.eyebrow")}
          </span>
          <h2 id="strategy-panel-title">
            {t("workbench.strategy.title")}
          </h2>
        </div>
        <Icon name="cpu" size={22} />
      </div>

      <fieldset className="strategy-options">
        <legend className="sr-only">
          {t("workbench.strategy.legend")}
        </legend>
        {strategies.map((strategy) => (
          <label
            className={cx("strategy-option", selectedId === strategy.id && "strategy-option--active")}
            key={strategy.id}
          >
            <input
              type="radio"
              name="model-strategy"
              value={strategy.id}
              checked={selectedId === strategy.id}
              onChange={() => onSelect(strategy.id)}
            />
            <span className="strategy-option__radio" aria-hidden="true" />
            <span className="strategy-option__copy">
              <strong>
                {strategy.label}
                {strategy.recommended ? <em>{t("common.recommended")}</em> : null}
              </strong>
              <small>{strategy.description}</small>
            </span>
            <span className="strategy-option__vram">{strategy.estimatedVramGb.toFixed(1)} GB</span>
          </label>
        ))}
      </fieldset>

      {selected ? (
        <div className="strategy-detail">
          <div>
            <span>{t("workbench.strategy.recognition")}</span>
            <strong>{selected.asrModel}</strong>
          </div>
          <div>
            <span>{t("workbench.strategy.speakerIdentity")}</span>
            <strong>{selected.diarizationModel}</strong>
          </div>
          <div>
            <span>{t("workbench.strategy.semanticModel")}</span>
            <strong>{selected.semanticModel}</strong>
            <code className="strategy-detail__status-code">
              {selected.semanticModelStatus}
            </code>
          </div>
          <p className="strategy-detail__evaluation">
            <Icon name="lock" size={15} />
            {selected.semanticModelEvaluation}
          </p>
          <p className="strategy-detail__guardrail">{selected.semanticGuardrail}</p>
        </div>
      ) : null}
    </section>
  );
}
