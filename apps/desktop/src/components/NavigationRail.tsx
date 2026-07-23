import type { AppSection } from "../contracts/studio";
import { useI18n, type MessageKey } from "../i18n";
import { cx } from "../lib/format";
import { Icon, type IconName } from "./Icon";

interface NavigationItem {
  id: AppSection;
  label: MessageKey;
  description: MessageKey;
  icon: IconName;
}

const navigationItems: NavigationItem[] = [
  {
    id: "overview",
    label: "nav.overview",
    description: "nav.overviewDescription",
    icon: "home",
  },
  {
    id: "review",
    label: "nav.review",
    description: "nav.reviewDescription",
    icon: "review",
  },
  {
    id: "artifacts",
    label: "nav.artifacts",
    description: "nav.artifactsDescription",
    icon: "folder",
  },
  {
    id: "pdf-qa",
    label: "nav.pdfQa",
    description: "nav.pdfQaDescription",
    icon: "shield",
  },
];

interface NavigationRailProps {
  activeSection: AppSection;
  onNavigate: (section: AppSection) => void;
  reviewCount: number;
}

export function NavigationRail({
  activeSection,
  onNavigate,
  reviewCount,
}: NavigationRailProps) {
  const { t } = useI18n();
  return (
    <aside className="navigation-rail" aria-label={t("nav.label")}>
      <div className="brand-lockup" aria-label="MediaTranscribe Studio">
        <span className="brand-lockup__mark" aria-hidden="true">
          <Icon name="sparkles" size={23} />
        </span>
        <span className="brand-lockup__copy">
          <strong>WhisperNote</strong>
          <small>MediaTranscribe Studio</small>
        </span>
      </div>

      <nav className="primary-navigation" aria-label={t("nav.primary")}>
        {navigationItems.map((item) => {
          const active = item.id === activeSection;
          return (
            <button
              key={item.id}
              className={cx("nav-item", active && "nav-item--active")}
              type="button"
              aria-label={t(item.label)}
              aria-current={active ? "page" : undefined}
              onClick={() => onNavigate(item.id)}
            >
              <span className="nav-item__icon" aria-hidden="true">
                <Icon name={item.icon} size={20} />
              </span>
              <span className="nav-item__copy">
                <strong>{t(item.label)}</strong>
                <small>{t(item.description)}</small>
              </span>
              {item.id === "review" && reviewCount > 0 ? (
                <span
                  className="nav-item__count"
                  aria-label={t("nav.pendingReviews", { count: reviewCount })}
                >
                  {reviewCount}
                </span>
              ) : null}
            </button>
          );
        })}
      </nav>

      <div className="navigation-rail__note">
        <span className="navigation-rail__note-icon" aria-hidden="true">
          <Icon name="cloud-off" size={17} />
        </span>
        <span>
          <strong>{t("nav.localFirst")}</strong>
          <small>{t("nav.localFirstDetail")}</small>
        </span>
      </div>
    </aside>
  );
}
