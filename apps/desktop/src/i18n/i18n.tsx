import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { LOCALE_OPTIONS, type Locale } from "./catalog";
import { I18nContext, type I18nContextValue } from "./context";
import {
  LOCALE_STORAGE_KEY,
  resolveInitialLocale,
  translate,
  type Translate,
} from "./core";

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() =>
    resolveInitialLocale(window.localStorage),
  );

  const setLocale = useCallback((nextLocale: Locale) => {
    try {
      window.localStorage.setItem(LOCALE_STORAGE_KEY, nextLocale);
    } catch {
      // Storage can be unavailable in hardened WebViews; keep the live choice.
    }
    setLocaleState(nextLocale);
  }, []);

  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.dataset.locale = locale;
    document.title = translate(locale, "app.windowTitle");
  }, [locale]);

  const t = useCallback<Translate>(
    (key, params) => translate(locale, key, params),
    [locale],
  );

  const value = useMemo<I18nContextValue>(
    () => ({ locale, localeOptions: LOCALE_OPTIONS, setLocale, t }),
    [locale, setLocale, t],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}
