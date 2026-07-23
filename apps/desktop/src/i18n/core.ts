import {
  MESSAGE_CATALOGS,
  SUPPORTED_LOCALES,
  type Locale,
  type MessageKey,
  type MessageParams,
} from "./catalog";

export const LOCALE_STORAGE_KEY = "media-transcribe-studio.ui-locale";
export const DEFAULT_LOCALE: Locale = "en";

export type Translate = (key: MessageKey, params?: MessageParams) => string;

export function isLocale(value: string | null): value is Locale {
  return (
    value !== null &&
    (SUPPORTED_LOCALES as readonly string[]).includes(value)
  );
}

export function resolveInitialLocale(
  storage: Pick<Storage, "getItem">,
): Locale {
  try {
    const stored = storage.getItem(LOCALE_STORAGE_KEY);
    return isLocale(stored) ? stored : DEFAULT_LOCALE;
  } catch {
    return DEFAULT_LOCALE;
  }
}

export function translate(
  locale: Locale,
  key: MessageKey,
  params: MessageParams = {},
): string {
  const template: string = MESSAGE_CATALOGS[locale][key];
  if (typeof template !== "string" || template.trim().length === 0) {
    throw new Error(`Missing localized message "${key}" for locale "${locale}".`);
  }

  return template.replace(
    /\{([a-zA-Z][a-zA-Z0-9]*)\}/gu,
    (match, name: string) => {
      const value = params[name];
      return value === undefined ? match : String(value);
    },
  );
}
