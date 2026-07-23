import { createContext, useContext } from "react";
import type { Locale, LOCALE_OPTIONS } from "./catalog";
import { DEFAULT_LOCALE, translate, type Translate } from "./core";
import { LOCALE_OPTIONS as localeOptions } from "./catalog";

export interface I18nContextValue {
  locale: Locale;
  localeOptions: typeof LOCALE_OPTIONS;
  setLocale: (locale: Locale) => void;
  t: Translate;
}

const englishFallback: I18nContextValue = {
  locale: DEFAULT_LOCALE,
  localeOptions,
  setLocale: () => undefined,
  t: (key, params) => translate(DEFAULT_LOCALE, key, params),
};

export const I18nContext =
  createContext<I18nContextValue>(englishFallback);

export function useI18n(): I18nContextValue {
  return useContext(I18nContext);
}
