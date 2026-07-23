import { useEffect, useState } from "react";
import { useI18n } from "../i18n";
import { Icon } from "./Icon";

interface MediaDropOverlayProps {
  visible: boolean;
  resolving: boolean;
}

type OverlayLifecycle = "mounted" | "visible" | "exiting";

interface OverlayState {
  lifecycle: OverlayLifecycle | null;
  shouldShow: boolean;
}

const EXIT_DURATION_MS = 140;

export function MediaDropOverlay({
  visible,
  resolving,
}: MediaDropOverlayProps) {
  const { t } = useI18n();
  const shouldShow = visible || resolving;
  const [overlayState, setOverlayState] = useState<OverlayState>(() => ({
    lifecycle: shouldShow ? "mounted" : null,
    shouldShow,
  }));

  if (overlayState.shouldShow !== shouldShow) {
    setOverlayState({
      lifecycle: shouldShow
        ? "mounted"
        : overlayState.lifecycle === null
          ? null
          : "exiting",
      shouldShow,
    });
  }

  const { lifecycle } = overlayState;

  useEffect(() => {
    if (lifecycle === "mounted") {
      const frame = window.requestAnimationFrame(() => {
        setOverlayState((current) =>
          current.shouldShow && current.lifecycle === "mounted"
            ? { ...current, lifecycle: "visible" }
            : current,
        );
      });

      return () => {
        window.cancelAnimationFrame(frame);
      };
    }

    if (lifecycle === "exiting") {
      const timeout = window.setTimeout(() => {
        setOverlayState((current) =>
          !current.shouldShow && current.lifecycle === "exiting"
            ? { ...current, lifecycle: null }
            : current,
        );
      }, EXIT_DURATION_MS);

      return () => {
        window.clearTimeout(timeout);
      };
    }
  }, [lifecycle]);

  if (lifecycle === null) {
    return null;
  }

  return (
    <div
      className="media-drop-overlay"
      role="status"
      aria-live="polite"
      aria-atomic="true"
      aria-busy={resolving}
      aria-hidden={lifecycle === "exiting"}
      data-lifecycle={lifecycle}
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
