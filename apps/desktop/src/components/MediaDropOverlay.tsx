import { useI18n } from "../i18n";
import { Icon } from "./Icon";

interface MediaDropOverlayProps {
  visible: boolean;
  resolving: boolean;
}

export function MediaDropOverlay({
  visible,
  resolving,
}: MediaDropOverlayProps) {
  const { t } = useI18n();

  if (!visible && !resolving) {
    return null;
  }

  return (
    <div
      className="media-drop-overlay"
      role="status"
      aria-live="polite"
      aria-busy={resolving}
      data-state={resolving ? "resolving" : "ready"}
    >
      <div className="media-drop-overlay__card">
        <span className="media-drop-overlay__icon" aria-hidden="true">
          <Icon name={resolving ? "sparkles" : "upload"} size={34} />
        </span>
        <span className="panel__eyebrow">{t("drop.eyebrow")}</span>
        <strong>
          {resolving ? t("drop.processingTitle") : t("drop.readyTitle")}
        </strong>
        <p>
          {resolving ? t("drop.processingDetail") : t("drop.readyDetail")}
        </p>
        <small>{t("drop.supportedTypes")}</small>
      </div>
    </div>
  );
}
