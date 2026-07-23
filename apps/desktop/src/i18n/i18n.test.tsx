import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  CORE_MESSAGE_KEYS,
  DEFAULT_LOCALE,
  ENGLISH_MESSAGES,
  I18nProvider,
  LOCALE_OPTIONS,
  LOCALE_STORAGE_KEY,
  MESSAGE_CATALOGS,
  SUPPORTED_LOCALES,
  resolveInitialLocale,
  translate,
  useI18n,
} from ".";

function LocaleProbe() {
  const { locale, localeOptions, setLocale, t } = useI18n();
  return (
    <>
      <output aria-label="active locale">{locale}</output>
      <output aria-label="translated overview">{t("nav.overview")}</output>
      <select
        aria-label="locale"
        value={locale}
        onChange={(event) => {
          const option = localeOptions.find(
            (candidate) => candidate.code === event.target.value,
          );
          if (option) {
            setLocale(option.code);
          }
        }}
      >
        {localeOptions.map((option) => (
          <option value={option.code} key={option.code}>
            {option.label}
          </option>
        ))}
      </select>
    </>
  );
}

describe("desktop interface localization", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("ships exactly the required first-wave locale choices", () => {
    expect(SUPPORTED_LOCALES).toEqual([
      "en",
      "zh-Hans",
      "zh-Hant",
      "ja",
      "ko",
      "es",
      "fr",
      "de",
      "pt-BR",
    ]);
    expect(LOCALE_OPTIONS).toHaveLength(9);
    expect(LOCALE_OPTIONS.map((option) => option.label)).toEqual([
      "English",
      "简体中文",
      "繁體中文",
      "日本語",
      "한국어",
      "Español",
      "Français",
      "Deutsch",
      "Português (Brasil)",
    ]);
  });

  it("defaults to English regardless of the browser language", () => {
    Object.defineProperty(window.navigator, "language", {
      configurable: true,
      value: "zh-CN",
    });

    expect(resolveInitialLocale(window.localStorage)).toBe(DEFAULT_LOCALE);
    render(
      <I18nProvider>
        <LocaleProbe />
      </I18nProvider>,
    );

    expect(screen.getByLabelText("active locale")).toHaveTextContent("en");
    expect(screen.getByLabelText("translated overview")).toHaveTextContent(
      "Overview",
    );
  });

  it("persists an explicit locale and rejects invalid stored values", async () => {
    const user = userEvent.setup();
    window.localStorage.setItem(LOCALE_STORAGE_KEY, "not-a-locale");

    render(
      <I18nProvider>
        <LocaleProbe />
      </I18nProvider>,
    );

    expect(screen.getByLabelText("active locale")).toHaveTextContent("en");
    await user.selectOptions(screen.getByLabelText("locale"), "zh-Hans");
    expect(screen.getByLabelText("active locale")).toHaveTextContent(
      "zh-Hans",
    );
    expect(screen.getByLabelText("translated overview")).toHaveTextContent(
      "概览",
    );
    expect(window.localStorage.getItem(LOCALE_STORAGE_KEY)).toBe("zh-Hans");
    expect(document.documentElement).toHaveAttribute("lang", "zh-Hans");
    expect(document.title).toBe("MediaTranscribe Studio");
  });

  it("keeps English and live locale changes when preference storage fails", async () => {
    const throwingStorage: Pick<Storage, "getItem"> = {
      getItem() {
        throw new Error("Storage is disabled.");
      },
    };
    expect(resolveInitialLocale(throwingStorage)).toBe("en");

    const user = userEvent.setup();
    vi.spyOn(Storage.prototype, "setItem").mockImplementationOnce(() => {
      throw new Error("Storage is read-only.");
    });
    render(
      <I18nProvider>
        <LocaleProbe />
      </I18nProvider>,
    );

    await user.selectOptions(screen.getByLabelText("locale"), "ja");
    expect(screen.getByLabelText("active locale")).toHaveTextContent("ja");
    expect(screen.getByLabelText("translated overview")).toHaveTextContent(
      "概要",
    );
  });

  it("uses a complete locale catalog without runtime English fallback", () => {
    expect(MESSAGE_CATALOGS.fr["app.windowTitle"]).toBe(
      ENGLISH_MESSAGES["app.windowTitle"],
    );
    expect(translate("fr", "top.createTask")).toBe("Créer une tâche");
  });

  it.each(SUPPORTED_LOCALES)(
    "contains every core interaction key for %s",
    (locale) => {
      CORE_MESSAGE_KEYS.forEach((key) => {
        expect(MESSAGE_CATALOGS[locale][key]).toBeTypeOf("string");
        expect(MESSAGE_CATALOGS[locale][key].trim().length).toBeGreaterThan(0);
      });
    },
  );
});
