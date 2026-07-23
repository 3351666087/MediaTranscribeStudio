import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  fireEvent,
  render,
  screen,
  within,
  type RenderResult,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import {
  DEFAULT_OUTPUT_CUSTOMIZATION,
  EXPORT_FORMAT_IDS,
  OutputCustomizationValidationError,
  cloneOutputCustomization,
  normalizeOutputCustomization,
  speakerColorForIndex,
  validateOutputCustomization,
  type OutputCustomization,
} from "../contracts/output-customization";
import {
  LOCALE_OPTIONS,
  type MessageKey,
  type MessageParams,
} from "../i18n";
import { I18nContext } from "../i18n/context";
import { translate } from "../i18n/core";
import { OutputCustomizationPanel } from "./OutputCustomizationPanel";

interface RuntimeRecipe {
  schemaVersion: unknown;
  report: Record<string, unknown>;
  subtitles: Record<string, unknown>;
  delivery: Record<string, unknown>;
  finishing: Record<string, unknown>;
  [key: string]: unknown;
}

function renderPanel(ui: ReactElement): RenderResult {
  return render(
    <I18nContext.Provider
      value={{
        locale: "en",
        localeOptions: LOCALE_OPTIONS,
        setLocale: () => undefined,
        t: (key: MessageKey, params?: MessageParams) =>
          translate("en", key, params),
      }}
    >
      {ui}
    </I18nContext.Provider>,
  );
}

function cloneDefaults(): OutputCustomization {
  return cloneOutputCustomization(DEFAULT_OUTPUT_CUSTOMIZATION);
}

function runtimeRecipe(): RuntimeRecipe {
  return JSON.parse(
    JSON.stringify(DEFAULT_OUTPUT_CUSTOMIZATION),
  ) as RuntimeRecipe;
}

function expectInvalid(value: unknown, path: string): void {
  try {
    validateOutputCustomization(value);
  } catch (error) {
    expect(error).toBeInstanceOf(OutputCustomizationValidationError);
    expect((error as OutputCustomizationValidationError).path).toBe(path);
    return;
  }
  throw new Error(`Expected validation to fail at ${path}`);
}

describe("native output recipe contract", () => {
  it("mirrors the backend export allowlist and excludes docx", () => {
    expect(EXPORT_FORMAT_IDS).toEqual([
      "pdf",
      "html",
      "markdown",
      "txt",
      "json",
      "srt",
      "webvtt",
      "ass",
    ]);
    expect(EXPORT_FORMAT_IDS).not.toContain("docx");
  });

  it("ships a valid recursively frozen default without mutating validated input", () => {
    expect(validateOutputCustomization(DEFAULT_OUTPUT_CUSTOMIZATION)).toEqual(
      DEFAULT_OUTPUT_CUSTOMIZATION,
    );
    expect(Object.isFrozen(DEFAULT_OUTPUT_CUSTOMIZATION)).toBe(true);
    expect(Object.isFrozen(DEFAULT_OUTPUT_CUSTOMIZATION.report)).toBe(true);
    expect(Object.isFrozen(DEFAULT_OUTPUT_CUSTOMIZATION.subtitles)).toBe(true);
    expect(Object.isFrozen(DEFAULT_OUTPUT_CUSTOMIZATION.delivery)).toBe(true);
    expect(Object.isFrozen(DEFAULT_OUTPUT_CUSTOMIZATION.delivery.formats)).toBe(
      true,
    );
    expect(
      Object.isFrozen(DEFAULT_OUTPUT_CUSTOMIZATION.delivery.subtitleModes),
    ).toBe(true);
    expect(Object.isFrozen(DEFAULT_OUTPUT_CUSTOMIZATION.finishing)).toBe(true);

    const input = runtimeRecipe();
    const before = structuredClone(input);
    const validated = validateOutputCustomization(input);

    expect(input).toEqual(before);
    expect(validated).toEqual(before);
    expect(validated).not.toBe(input);
    expect(normalizeOutputCustomization(input)).toEqual(validated);
  });

  it("rejects missing and unknown fields instead of dropping them", () => {
    const missing = runtimeRecipe();
    delete missing.report.template;
    expectInvalid(missing, "outputCustomization.report");

    const unknown = runtimeRecipe();
    unknown.delivery.container = "mp4";
    expectInvalid(unknown, "outputCustomization.delivery");

    const unknownRoot = runtimeRecipe();
    unknownRoot.extra = true;
    expectInvalid(unknownRoot, "outputCustomization");
  });

  it.each([
    ["report.template", "cinematic"],
    ["report.font", "downloaded-font"],
    ["report.pageSize", "tabloid"],
    ["subtitles.theme", "karaoke"],
    ["subtitles.position", "left"],
    ["finishing.timestampStyle", "word"],
  ])("rejects unsupported enum %s=%s", (field, invalidValue) => {
    const recipe = runtimeRecipe();
    const [section, property] = field.split(".");
    (recipe[section] as Record<string, unknown>)[property] = invalidValue;
    expectInvalid(recipe, `outputCustomization.${field}`);
  });

  it.each([-1, 101, 1.5, Number.NaN, Number.POSITIVE_INFINITY])(
    "rejects backgroundOpacity=%s instead of clamping it",
    (invalidValue) => {
      const recipe = runtimeRecipe();
      recipe.subtitles.backgroundOpacity = invalidValue;
      expectInvalid(
        recipe,
        "outputCustomization.subtitles.backgroundOpacity",
      );
    },
  );

  it.each([0, 4, 1.5, Number.NaN, Number.NEGATIVE_INFINITY])(
    "rejects maximumLines=%s instead of replacing it",
    (invalidValue) => {
      const recipe = runtimeRecipe();
      recipe.subtitles.maximumLines = invalidValue;
      expectInvalid(recipe, "outputCustomization.subtitles.maximumLines");
    },
  );

  it("rejects empty, duplicate, unsupported, and unsafe delivery values", () => {
    const emptyFormats = runtimeRecipe();
    emptyFormats.delivery.formats = [];
    expectInvalid(emptyFormats, "outputCustomization.delivery.formats");

    const duplicateFormats = runtimeRecipe();
    duplicateFormats.delivery.formats = ["pdf", "pdf"];
    expectInvalid(
      duplicateFormats,
      "outputCustomization.delivery.formats[1]",
    );

    const docx = runtimeRecipe();
    docx.delivery.formats = ["pdf", "docx"];
    expectInvalid(docx, "outputCustomization.delivery.formats[1]");

    const mutableSource = runtimeRecipe();
    mutableSource.delivery.preserveSourceMedia = false;
    expectInvalid(
      mutableSource,
      "outputCustomization.delivery.preserveSourceMedia",
    );
  });

  it("rejects backend-incompatible subtitle relationships and word progress", () => {
    const noMode = runtimeRecipe();
    noMode.delivery.subtitleModes = [];
    expectInvalid(noMode, "outputCustomization.delivery.subtitleModes");

    const modesWhileDisabled = runtimeRecipe();
    modesWhileDisabled.subtitles.enabled = false;
    modesWhileDisabled.delivery.formats = ["pdf"];
    expectInvalid(
      modesWhileDisabled,
      "outputCustomization.delivery.subtitleModes",
    );

    const subtitleFormatWhileDisabled = runtimeRecipe();
    subtitleFormatWhileDisabled.subtitles.enabled = false;
    subtitleFormatWhileDisabled.delivery.subtitleModes = [];
    subtitleFormatWhileDisabled.delivery.formats = ["pdf", "srt"];
    expectInvalid(
      subtitleFormatWhileDisabled,
      "outputCustomization.delivery.formats",
    );

    const wordProgress = runtimeRecipe();
    wordProgress.subtitles.wordProgressHighlight = true;
    expectInvalid(
      wordProgress,
      "outputCustomization.subtitles.wordProgressHighlight",
    );
  });

  it("rejects dependent report and finishing states rather than repairing them", () => {
    const emptyCustomFont = runtimeRecipe();
    emptyCustomFont.report.font = "custom";
    emptyCustomFont.report.customFontFamily = "";
    expectInvalid(
      emptyCustomFont,
      "outputCustomization.report.customFontFamily",
    );

    const enabledWithoutStyle = runtimeRecipe();
    enabledWithoutStyle.finishing.chapterStyle = "none";
    expectInvalid(
      enabledWithoutStyle,
      "outputCustomization.finishing.chapterStyle",
    );

    const disabledWithStyle = runtimeRecipe();
    disabledWithStyle.finishing.includeChapters = false;
    expectInvalid(
      disabledWithStyle,
      "outputCustomization.finishing.chapterStyle",
    );
  });

  it("rejects values that the Python parser would silently normalize", () => {
    const uppercaseColor = runtimeRecipe();
    uppercaseColor.report.accentColor = "#AABBCC";
    expectInvalid(
      uppercaseColor,
      "outputCustomization.report.accentColor",
    );

    const paddedTitle = runtimeRecipe();
    paddedTitle.finishing.customTitle = " Padded title ";
    expectInvalid(
      paddedTitle,
      "outputCustomization.finishing.customTitle",
    );

    const decomposedTitle = runtimeRecipe();
    decomposedTitle.finishing.customTitle = "Cafe\u0301";
    expectInvalid(
      decomposedTitle,
      "outputCustomization.finishing.customTitle",
    );
  });

  it("uses Unicode code-point limits matching Python and JSON Schema", () => {
    const atLimit = runtimeRecipe();
    atLimit.finishing.customTitle = "😀".repeat(160);
    expect(validateOutputCustomization(atLimit).finishing.customTitle).toBe(
      "😀".repeat(160),
    );

    const beyondLimit = runtimeRecipe();
    beyondLimit.finishing.customTitle = "😀".repeat(161);
    expectInvalid(
      beyondLimit,
      "outputCustomization.finishing.customTitle",
    );
  });

  it.each([
    "",
    "plain-name",
    "../{sourceStem}",
    "folder\\{sourceStem}",
    "{unknown}",
    "{sourceStem",
    "{sourceStem}\u0000",
    " {sourceStem}",
    "{sourceStem} ",
  ])("rejects unsafe filename template %j", (fileNamePattern) => {
    const recipe = runtimeRecipe();
    recipe.delivery.fileNamePattern = fileNamePattern;
    expectInvalid(
      recipe,
      "outputCustomization.delivery.fileNamePattern",
    );
  });

  it("accepts every documented filename token without rewriting the template", () => {
    const recipe = runtimeRecipe();
    const fileNamePattern =
      "{sourceStem}-{artifact}-{language}-{date}-{speakerCount}";
    recipe.delivery.fileNamePattern = fileNamePattern;

    expect(
      validateOutputCustomization(recipe).delivery.fileNamePattern,
    ).toBe(fileNamePattern);
  });
});

describe("OutputCustomizationPanel fail-closed editing", () => {
  it("uses four progressive rooms and never renders a docx control", async () => {
    const user = userEvent.setup();
    renderPanel(<OutputCustomizationPanel />);

    expect(
      screen.getByRole("tablist", { name: "Output customization rooms" }),
    ).toHaveAttribute("aria-orientation", "vertical");
    expect(screen.getAllByRole("tabpanel", { hidden: true })).toHaveLength(4);
    expect(screen.getByRole("tabpanel")).toHaveAccessibleName(/Report/u);
    expect(
      screen.getByRole("heading", {
        name: "Set the report's visual voice",
      }),
    ).toBeVisible();

    await user.click(screen.getByRole("tab", { name: /Delivery/u }));

    expect(screen.getByRole("tabpanel")).toHaveAccessibleName(/Delivery/u);
    expect(
      screen.queryByRole("checkbox", { name: /Word document/u }),
    ).not.toBeInTheDocument();
    expect(
      within(screen.getByRole("group", { name: "Export formats" })).getAllByRole(
        "checkbox",
      ),
    ).toHaveLength(EXPORT_FORMAT_IDS.length);
  });

  it("keeps an invalid custom-font draft visible and emits only after it is valid", async () => {
    const user = userEvent.setup();
    const onChange =
      vi.fn<(value: OutputCustomization) => void>();
    renderPanel(<OutputCustomizationPanel onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: "Custom" }));

    const customFont = screen.getByRole("textbox", {
      name: "Local font family",
    });
    expect(customFont).toHaveValue("");
    expect(customFont).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("alert")).toHaveTextContent(
      "customFontFamily: is required",
    );
    expect(
      screen.getByRole("button", { name: "Use these settings" }),
    ).toBeDisabled();
    expect(onChange).not.toHaveBeenCalled();

    await user.type(customFont, "Noto Sans CJK");

    expect(customFont).toHaveValue("Noto Sans CJK");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Use these settings" }),
    ).toBeEnabled();
    expect(onChange.mock.lastCall?.[0].report).toMatchObject({
      font: "custom",
      customFontFamily: "Noto Sans CJK",
    });
  });

  it("keeps an invalid naming draft visible and blocks apply without emitting it", () => {
    const onChange =
      vi.fn<(value: OutputCustomization) => void>();
    renderPanel(
      <OutputCustomizationPanel initialLevel="delivery" onChange={onChange} />,
    );

    const input = screen.getByRole("textbox", { name: "File naming" });
    fireEvent.change(input, { target: { value: "" } });

    expect(input).toHaveValue("");
    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("alert")).toHaveTextContent(
      "fileNamePattern: must not be empty",
    );
    expect(
      screen.getByRole("button", { name: "Use these settings" }),
    ).toBeDisabled();
    expect(onChange).not.toHaveBeenCalled();

    fireEvent.change(input, {
      target: { value: "{sourceStem}-{date}" },
    });

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(onChange.mock.lastCall?.[0].delivery.fileNamePattern).toBe(
      "{sourceStem}-{date}",
    );
  });

  it("blocks removing the final format without inventing a fallback", async () => {
    const user = userEvent.setup();
    const onChange =
      vi.fn<(value: OutputCustomization) => void>();
    const value = cloneDefaults();
    value.subtitles.enabled = false;
    value.delivery.formats = ["pdf"];
    value.delivery.subtitleModes = [];
    renderPanel(
      <OutputCustomizationPanel
        value={value}
        initialLevel="delivery"
        onChange={onChange}
      />,
    );

    const pdf = screen.getByRole("checkbox", { name: "Verified PDF" });
    await user.click(pdf);

    expect(pdf).toBeChecked();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "At least one output format is required.",
    );
    expect(onChange).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Use these settings" }),
    ).toBeEnabled();
  });

  it("blocks removing the final subtitle mode without repairing the selection", async () => {
    const user = userEvent.setup();
    const onChange =
      vi.fn<(value: OutputCustomization) => void>();
    renderPanel(
      <OutputCustomizationPanel initialLevel="delivery" onChange={onChange} />,
    );

    const sidecar = screen.getByRole("checkbox", {
      name: /^Sidecar files/u,
    });
    await user.click(sidecar);

    expect(sidecar).toBeChecked();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "At least one subtitle delivery mode is required",
    );
    expect(onChange).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Use these settings" }),
    ).toBeEnabled();
  });

  it("explicitly clears incompatible subtitle outputs when subtitles are disabled", async () => {
    const user = userEvent.setup();
    const onChange =
      vi.fn<(value: OutputCustomization) => void>();
    renderPanel(<OutputCustomizationPanel onChange={onChange} />);

    await user.click(screen.getByRole("tab", { name: /Subtitles/u }));
    await user.click(
      screen.getByRole("checkbox", { name: "Include subtitles" }),
    );

    expect(onChange.mock.lastCall?.[0]).toMatchObject({
      subtitles: {
        enabled: false,
        wordProgressHighlight: false,
      },
      delivery: {
        formats: ["pdf"],
        subtitleModes: [],
        preserveSourceMedia: true,
      },
    });

    await user.click(screen.getByRole("tab", { name: /Delivery/u }));
    expect(
      screen.getByRole("checkbox", { name: "SRT subtitles" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("checkbox", { name: "WebVTT subtitles" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("checkbox", { name: /^Sidecar files/u }),
    ).toBeDisabled();
  });

  it("never offers a mutable word-progress value", async () => {
    const user = userEvent.setup();
    renderPanel(<OutputCustomizationPanel initialLevel="subtitles" />);

    const wordProgress = screen.getByRole("checkbox", {
      name: /Word progress highlight/u,
    });
    expect(wordProgress).toBeDisabled();
    expect(wordProgress).not.toBeChecked();

    await user.click(wordProgress);
    expect(wordProgress).not.toBeChecked();
  });

  it("covers sidecar, soft-mux, and burn-in while exposing real container mappings", async () => {
    const user = userEvent.setup();
    const onChange =
      vi.fn<(value: OutputCustomization) => void>();
    renderPanel(
      <OutputCustomizationPanel initialLevel="delivery" onChange={onChange} />,
    );

    await user.click(
      screen.getByRole("checkbox", { name: /^Selectable subtitle track/u }),
    );
    await user.click(
      screen.getByRole("checkbox", { name: /^Burn into a new video/u }),
    );

    expect(onChange.mock.lastCall?.[0].delivery.subtitleModes).toEqual([
      "sidecar",
      "soft-mux",
      "burn-in",
    ]);

    const mapping = screen.getByRole("note", {
      name: "Delivery container mapping",
    });
    expect(mapping).toHaveTextContent("Sidecar · SRT / WebVTT / ASS");
    expect(mapping).toHaveTextContent("Soft mux · Matroska + ASS");
    expect(mapping).toHaveTextContent("Burn-in · MP4 + H.264");
    expect(mapping).toHaveTextContent(
      "unsupported values are never sent from this UI",
    );
  });

  it("supports arbitrary detected speaker counts with a bounded preview", () => {
    renderPanel(
      <OutputCustomizationPanel
        initialLevel="subtitles"
        speakerCount={25}
        speakerLabels={Array.from(
          { length: 25 },
          (_, index) => `Speaker ${String(index + 1)}`,
        )}
      />,
    );

    const preview = screen.getByRole("list", {
      name: "Speaker color preview",
    });
    expect(within(preview).getAllByRole("listitem")).toHaveLength(13);
    expect(
      within(preview).getByRole("listitem", {
        name: "and 13 more speakers",
      }),
    ).toHaveTextContent("+13");
    expect(speakerColorForIndex(24, "adaptive-spectrum")).toMatch(/^hsl\(/u);
  });
});

describe("OutputCustomizationPanel navigation and quality", () => {
  it("supports arrow, Home, and End navigation with managed focus", () => {
    renderPanel(<OutputCustomizationPanel />);

    const reportTab = screen.getByRole("tab", { name: /Report/u });
    reportTab.focus();
    fireEvent.keyDown(reportTab, { key: "ArrowDown" });
    expect(screen.getByRole("tab", { name: /Subtitles/u })).toHaveFocus();

    fireEvent.keyDown(screen.getByRole("tab", { name: /Subtitles/u }), {
      key: "End",
    });
    expect(screen.getByRole("tab", { name: /Finishing/u })).toHaveFocus();

    fireEvent.keyDown(screen.getByRole("tab", { name: /Finishing/u }), {
      key: "Home",
    });
    expect(reportTab).toHaveFocus();
    expect(reportTab).toHaveAttribute("aria-selected", "true");
  });

  it("uses a compact score ring and keeps hard-gate evidence closed by default", () => {
    const { container } = renderPanel(
      <OutputCustomizationPanel
        quality={{
          score: 93,
          minimumScore: 90,
          status: "review-required",
          gates: [
            {
              id: "readability",
              label: "Reading comfort",
              detail: "Verified line length and subtitle pace.",
              status: "passed",
            },
          ],
        }}
      />,
    );

    expect(
      screen.getByRole("img", { name: "Output visual quality score: 93%" }),
    ).toHaveTextContent("93");
    expect(container.querySelector(".output-diy-score-ring")).toBeInTheDocument();
    expect(screen.queryByText(/Visual score 93/iu)).not.toBeInTheDocument();

    const details = screen
      .getByText("Advanced quality gates")
      .closest("details");
    expect(details).not.toBeNull();
    expect(details).not.toHaveAttribute("open");
    expect(
      within(details as HTMLElement).getByText("Reading comfort"),
    ).not.toBeVisible();
  });
});

describe("schema and visual contract", () => {
  it("parses the native recipe schema with exact fail-closed invariants", () => {
    const schemaPath = resolve(
      process.cwd(),
      "../../contracts/output-customization.schema.json",
    );
    const schemaText = readFileSync(schemaPath, "utf8");
    const schema = JSON.parse(schemaText) as {
      additionalProperties: boolean;
      required: string[];
      $defs: {
        report: {
          additionalProperties: boolean;
          properties: {
            accentColor: { pattern: string };
          };
        };
        subtitles: {
          additionalProperties: boolean;
          properties: {
            wordProgressHighlight: { const: boolean };
          };
        };
        delivery: {
          additionalProperties: boolean;
          properties: {
            formats: {
              minItems: number;
              uniqueItems: boolean;
              items: { enum: string[] };
            };
            preserveSourceMedia: { const: boolean };
          };
        };
        finishing: {
          additionalProperties: boolean;
        };
      };
    };

    expect(schema.additionalProperties).toBe(false);
    expect(schema.required).toEqual([
      "schemaVersion",
      "report",
      "subtitles",
      "delivery",
      "finishing",
    ]);
    expect(schema.$defs.delivery.properties.formats.items.enum).toEqual([
      ...EXPORT_FORMAT_IDS,
    ]);
    expect(schemaText).not.toContain('"docx"');
    expect(
      schema.$defs.subtitles.properties.wordProgressHighlight.const,
    ).toBe(false);
    expect(
      schema.$defs.delivery.properties.preserveSourceMedia.const,
    ).toBe(true);
    expect(schema.$defs.delivery.properties.formats.minItems).toBe(1);
    expect(schema.$defs.delivery.properties.formats.uniqueItems).toBe(true);
    expect(schema.$defs.report.properties.accentColor.pattern).toBe(
      "^#[0-9a-f]{6}$",
    );
    expect(schema.$defs.report.additionalProperties).toBe(false);
    expect(schema.$defs.subtitles.additionalProperties).toBe(false);
    expect(schema.$defs.delivery.additionalProperties).toBe(false);
    expect(schema.$defs.finishing.additionalProperties).toBe(false);
  });

  it("keeps validation, container mapping, reduced-motion, and forced-color styles", () => {
    const stylesheet = readFileSync(
      resolve(process.cwd(), "src/styles/output-customization.css"),
      "utf8",
    );

    expect(stylesheet).toContain(".output-diy__validation");
    expect(stylesheet).toContain(".output-diy-container-map");
    expect(stylesheet).toContain("@media (prefers-reduced-motion: reduce)");
    expect(stylesheet).toContain("@media (forced-colors: active)");
  });
});
