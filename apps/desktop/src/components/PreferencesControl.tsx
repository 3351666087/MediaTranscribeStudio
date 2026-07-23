import { useI18n } from "../i18n";
import { THEME_MODES, useTheme, type ThemeMode } from "../theme";
import { Icon, type IconName } from "./Icon";

const themeIcons: Record<ThemeMode, IconName> = {
  system: "monitor",
  light: "sun",
  dark: "moon",
};

const themeLabelKeys = {
  system: "theme.system",
  light: "theme.light",
  dark: "theme.dark",
} as const;

const themeDescriptionKeys = {
  system: "theme.systemDescription",
  light: "theme.lightDescription",
  dark: "theme.darkDescription",
} as const;

export function PreferencesControl() {
  const { locale, localeOptions, setLocale, t } = useI18n();
  const { mode, setMode } = useTheme();

  return (
    <details className="preferences-control">
      <summary
        className="preferences-control__trigger"
        aria-label={t("preferences.label")}
        title={t("preferences.label")}
      >
        <Icon name="settings" size={18} />
        <span className="preferences-control__trigger-label">
          {localeOptions.find((option) => option.code === locale)?.shortLabel ??
            "EN"}
        </span>
      </summary>

      <div className="preferences-control__panel">
        <div className="preferences-control__heading">
          <span aria-hidden="true">
            <Icon name="sparkles" size={18} />
          </span>
          <strong>{t("preferences.label")}</strong>
        </div>

        <label className="preferences-control__field">
          <span>
            <Icon name="globe" size={16} />
            {t("preferences.language")}
          </span>
          <select
            value={locale}
            aria-label={t("preferences.language")}
            onChange={(event) => {
              const nextLocale = localeOptions.find(
                (option) => option.code === event.target.value,
              );
              if (nextLocale) {
                setLocale(nextLocale.code);
              }
            }}
          >
            {localeOptions.map((option) => (
              <option value={option.code} key={option.code}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <fieldset className="preferences-control__themes">
          <legend>{t("preferences.theme")}</legend>
          <div>
            {THEME_MODES.map((themeMode) => (
              <button
                className={
                  mode === themeMode
                    ? "theme-choice theme-choice--active"
                    : "theme-choice"
                }
                type="button"
                aria-pressed={mode === themeMode}
                title={t(themeDescriptionKeys[themeMode])}
                onClick={() => setMode(themeMode)}
                key={themeMode}
              >
                <Icon name={themeIcons[themeMode]} size={17} />
                <span>{t(themeLabelKeys[themeMode])}</span>
              </button>
            ))}
          </div>
        </fieldset>
      </div>
    </details>
  );
}
