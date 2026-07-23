import { useEffect } from "react";
import { useI18n } from "../i18n";
import { Icon, type IconName } from "./Icon";

interface ToastProps {
  toast: {
    id: number;
    tone: "success" | "warning" | "error" | "info";
    title: string;
    detail: string;
  } | null;
  onDismiss: () => void;
}

const toneIcons: Record<NonNullable<ToastProps["toast"]>["tone"], IconName> = {
  success: "check",
  warning: "alert",
  error: "x",
  info: "bell",
};

export function Toast({ toast, onDismiss }: ToastProps) {
  const { t } = useI18n();
  useEffect(() => {
    if (!toast) {
      return undefined;
    }
    const timer = window.setTimeout(onDismiss, 5200);
    return () => window.clearTimeout(timer);
  }, [onDismiss, toast]);

  if (!toast) {
    return null;
  }

  return (
    <div
      className={`toast toast--${toast.tone}`}
      role={toast.tone === "error" ? "alert" : "status"}
      aria-live={toast.tone === "error" ? "assertive" : "polite"}
    >
      <span className="toast__icon" aria-hidden="true">
        <Icon name={toneIcons[toast.tone]} size={19} />
      </span>
      <span className="toast__copy">
        <strong>{toast.title}</strong>
        <small>{toast.detail}</small>
      </span>
      <button
        type="button"
        aria-label={t("common.dismiss")}
        onClick={onDismiss}
      >
        <Icon name="x" size={17} />
      </button>
    </div>
  );
}
