/**
 * Native desktop output recipe.
 *
 * This is intentionally the small, presentation-only contract accepted by
 * `backend/output_recipe.py::parse_output_recipe`. The backend compiles one
 * evidence-bearing OutputCustomization snapshot for every selected delivery
 * mode. Do not add a UI field here until the Python recipe parser accepts and
 * compiles it.
 */

export const OUTPUT_CUSTOMIZATION_SCHEMA_VERSION = "1.0.0" as const;

export const REPORT_TEMPLATE_IDS = [
  "soft-glass",
  "editorial",
  "academic",
  "compact",
] as const;
export type ReportTemplateId = (typeof REPORT_TEMPLATE_IDS)[number];

export const REPORT_FONT_IDS = [
  "system-sans",
  "humanist",
  "serif",
  "mono-accent",
  "custom",
] as const;
export type ReportFontId = (typeof REPORT_FONT_IDS)[number];

export const REPORT_PAGE_SIZES = ["a4", "letter", "legal", "screen"] as const;
export type ReportPageSize = (typeof REPORT_PAGE_SIZES)[number];

export const REPORT_DENSITIES = ["airy", "balanced", "compact"] as const;
export type ReportDensity = (typeof REPORT_DENSITIES)[number];

export const SUBTITLE_THEME_IDS = [
  "youtube-clean",
  "soft-bubble",
  "cinema",
  "high-contrast",
] as const;
export type SubtitleThemeId = (typeof SUBTITLE_THEME_IDS)[number];

export const SUBTITLE_SIZE_IDS = ["small", "medium", "large", "x-large"] as const;
export type SubtitleSizeId = (typeof SUBTITLE_SIZE_IDS)[number];

export const SUBTITLE_SAFE_AREA_IDS = [
  "standard",
  "broadcast",
  "generous",
] as const;
export type SubtitleSafeAreaId = (typeof SUBTITLE_SAFE_AREA_IDS)[number];

export const SUBTITLE_POSITION_IDS = ["bottom", "smart", "top"] as const;
export type SubtitlePositionId = (typeof SUBTITLE_POSITION_IDS)[number];

export const SPEAKER_PALETTE_IDS = [
  "adaptive-spectrum",
  "candy",
  "ocean",
  "high-contrast",
  "monochrome",
] as const;
export type SpeakerPaletteId = (typeof SPEAKER_PALETTE_IDS)[number];

/**
 * Exact `_EXPORT_FORMATS` parity with `backend/output_recipe.py`.
 * DOCX is deliberately absent because no production exporter/gate exists.
 */
export const EXPORT_FORMAT_IDS = [
  "pdf",
  "html",
  "markdown",
  "txt",
  "json",
  "srt",
  "webvtt",
  "ass",
] as const;
export type ExportFormatId = (typeof EXPORT_FORMAT_IDS)[number];

export const SUBTITLE_EXPORT_FORMAT_IDS = [
  "srt",
  "webvtt",
  "ass",
] as const satisfies readonly ExportFormatId[];

export const SUBTITLE_DELIVERY_MODE_IDS = [
  "sidecar",
  "soft-mux",
  "burn-in",
] as const;
export type SubtitleDeliveryModeId =
  (typeof SUBTITLE_DELIVERY_MODE_IDS)[number];

export const CHAPTER_STYLE_IDS = ["semantic", "interval", "none"] as const;
export type ChapterStyleId = (typeof CHAPTER_STYLE_IDS)[number];

export const TIMESTAMP_STYLE_IDS = ["segment", "paragraph", "chapter"] as const;
export type TimestampStyleId = (typeof TIMESTAMP_STYLE_IDS)[number];

export const OUTPUT_CUSTOMIZATION_LEVELS = [
  "report",
  "subtitles",
  "delivery",
  "finishing",
] as const;
export type OutputCustomizationLevel =
  (typeof OUTPUT_CUSTOMIZATION_LEVELS)[number];

export const OUTPUT_CUSTOMIZATION_LIMITS = {
  customText: 160,
  fileNamePattern: 160,
} as const;

export const FILE_NAME_TEMPLATE_TOKENS = [
  "sourceStem",
  "artifact",
  "language",
  "date",
  "speakerCount",
] as const;
export type FileNameTemplateToken =
  (typeof FILE_NAME_TEMPLATE_TOKENS)[number];

export interface ReportCustomization {
  template: ReportTemplateId;
  font: ReportFontId;
  customFontFamily: string;
  pageSize: ReportPageSize;
  density: ReportDensity;
  accentColor: string;
}

export interface SubtitleCustomization {
  enabled: boolean;
  theme: SubtitleThemeId;
  size: SubtitleSizeId;
  safeArea: SubtitleSafeAreaId;
  position: SubtitlePositionId;
  speakerPalette: SpeakerPaletteId;
  backgroundOpacity: number;
  maximumLines: 1 | 2 | 3;
  avoidVisualCollisions: boolean;
  /**
   * The current backend rejects word-progress without verified word evidence.
   * The native recipe therefore cannot express `true`.
   */
  wordProgressHighlight: false;
}

export interface DeliveryCustomization {
  formats: readonly ExportFormatId[];
  subtitleModes: readonly SubtitleDeliveryModeId[];
  includeMediaMetadata: boolean;
  preserveSourceMedia: true;
  fileNamePattern: string;
}

export interface FinishingCustomization {
  includeCover: boolean;
  includeChapters: boolean;
  includeTimestamps: boolean;
  includeHeader: boolean;
  includeFooter: boolean;
  includeSpeakerIndex: boolean;
  includeConfidenceNotes: boolean;
  chapterStyle: ChapterStyleId;
  timestampStyle: TimestampStyleId;
  customTitle: string;
}

export interface OutputCustomization {
  schemaVersion: typeof OUTPUT_CUSTOMIZATION_SCHEMA_VERSION;
  report: ReportCustomization;
  subtitles: SubtitleCustomization;
  delivery: DeliveryCustomization;
  finishing: FinishingCustomization;
}

export type OutputQualityStatus =
  | "pending"
  | "passed"
  | "review-required"
  | "blocked";

export interface OutputQualityGate {
  id: string;
  label: string;
  detail?: string;
  status: "pending" | "passed" | "failed";
}

export interface OutputCustomizationQuality {
  score: number | null;
  minimumScore?: number;
  status: OutputQualityStatus;
  gates?: readonly OutputQualityGate[];
}

export class OutputCustomizationValidationError extends Error {
  readonly path: string;

  constructor(path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = "OutputCustomizationValidationError";
    this.path = path;
  }
}

function freezeOutputCustomization(
  value: OutputCustomization,
): Readonly<OutputCustomization> {
  Object.freeze(value.report);
  Object.freeze(value.subtitles);
  Object.freeze(value.delivery.formats);
  Object.freeze(value.delivery.subtitleModes);
  Object.freeze(value.delivery);
  Object.freeze(value.finishing);
  return Object.freeze(value);
}

export const DEFAULT_OUTPUT_CUSTOMIZATION = freezeOutputCustomization({
  schemaVersion: OUTPUT_CUSTOMIZATION_SCHEMA_VERSION,
  report: {
    template: "soft-glass",
    font: "system-sans",
    customFontFamily: "",
    pageSize: "a4",
    density: "balanced",
    accentColor: "#6959d2",
  },
  subtitles: {
    enabled: true,
    theme: "youtube-clean",
    size: "medium",
    safeArea: "broadcast",
    position: "smart",
    speakerPalette: "adaptive-spectrum",
    backgroundOpacity: 72,
    maximumLines: 2,
    avoidVisualCollisions: true,
    wordProgressHighlight: false,
  },
  delivery: {
    formats: ["pdf", "srt", "webvtt"],
    subtitleModes: ["sidecar"],
    includeMediaMetadata: true,
    preserveSourceMedia: true,
    fileNamePattern: "{sourceStem}-{artifact}",
  },
  finishing: {
    includeCover: true,
    includeChapters: true,
    includeTimestamps: true,
    includeHeader: true,
    includeFooter: true,
    includeSpeakerIndex: true,
    includeConfidenceNotes: false,
    chapterStyle: "semantic",
    timestampStyle: "segment",
    customTitle: "",
  },
});

const CONTROL_CHARACTER_PATTERN = /[\u0000-\u001f\u007f]/u;
const BIDI_CONTROL_PATTERN =
  /[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]/u;
const LOWERCASE_HEX_COLOR_PATTERN = /^#[0-9a-f]{6}$/u;
const TEMPLATE_TOKEN_PATTERN = /\{([A-Za-z][A-Za-z0-9]*)\}/gu;
const PATH_SEPARATOR_PATTERN = /[\\/]/u;
const SUBTITLE_EXPORT_FORMAT_SET = new Set<ExportFormatId>(
  SUBTITLE_EXPORT_FORMAT_IDS,
);

function fail(path: string, message: string): never {
  throw new OutputCustomizationValidationError(path, message);
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return fail(path, "must be an object");
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  path: string,
  expected: readonly string[],
): void {
  const expectedSet = new Set(expected);
  const actual = Object.keys(value);
  const missing = expected.filter((key) => !Object.hasOwn(value, key));
  const unknown = actual.filter((key) => !expectedSet.has(key));
  if (missing.length === 0 && unknown.length === 0) {
    return;
  }
  const details = [
    missing.length === 0 ? "" : `missing ${missing.join(", ")}`,
    unknown.length === 0 ? "" : `unknown ${unknown.join(", ")}`,
  ].filter(Boolean);
  fail(path, `has invalid fields: ${details.join("; ")}`);
}

function enumeration<T extends string>(
  value: unknown,
  path: string,
  choices: readonly T[],
): T {
  if (
    typeof value !== "string" ||
    !(choices as readonly string[]).includes(value)
  ) {
    fail(path, `must be one of ${choices.map((item) => `'${item}'`).join(", ")}`);
  }
  return value as T;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") {
    fail(path, "must be a boolean");
  }
  return value;
}

function integer(
  value: unknown,
  path: string,
  minimum: number,
  maximum: number,
): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    !Number.isInteger(value)
  ) {
    fail(path, "must be a finite integer");
  }
  if (value < minimum || value > maximum) {
    fail(path, `must be between ${minimum} and ${maximum}`);
  }
  return value;
}

function exactText(
  value: unknown,
  path: string,
  maximum: number,
  allowEmpty: boolean,
): string {
  if (typeof value !== "string") {
    fail(path, "must be a string");
  }
  if (!allowEmpty && value.length === 0) {
    fail(path, "must not be empty");
  }
  // Python's len(str) and JSON Schema maxLength count Unicode code points,
  // whereas JavaScript's String.length counts UTF-16 code units. Keep the
  // native contract aligned for non-BMP titles and font family names.
  if ([...value].length > maximum) {
    fail(path, `must contain at most ${maximum} characters`);
  }
  if (CONTROL_CHARACTER_PATTERN.test(value) || BIDI_CONTROL_PATTERN.test(value)) {
    fail(path, "contains unsafe control characters");
  }
  if (value.normalize("NFC") !== value) {
    fail(path, "must use Unicode NFC normalization");
  }
  if (value.trim() !== value) {
    fail(path, "must not start or end with whitespace");
  }
  return value;
}

function color(value: unknown, path: string): string {
  if (typeof value !== "string" || !LOWERCASE_HEX_COLOR_PATTERN.test(value)) {
    fail(path, "must be a lowercase six-digit hexadecimal color");
  }
  return value;
}

function uniqueEnumList<T extends string>(
  value: unknown,
  path: string,
  choices: readonly T[],
  minimum: number,
): T[] {
  if (!Array.isArray(value)) {
    fail(path, "must be an array");
  }
  if (value.length < minimum || value.length > choices.length) {
    fail(path, `must contain between ${minimum} and ${choices.length} entries`);
  }
  const result: T[] = [];
  value.forEach((item, index) => {
    const selected = enumeration(item, `${path}[${index}]`, choices);
    if (result.includes(selected)) {
      fail(`${path}[${index}]`, `duplicates '${selected}'`);
    }
    result.push(selected);
  });
  return result;
}

function fileNamePattern(value: unknown): string {
  const path = "outputCustomization.delivery.fileNamePattern";
  const template = exactText(
    value,
    path,
    OUTPUT_CUSTOMIZATION_LIMITS.fileNamePattern,
    false,
  );
  if (PATH_SEPARATOR_PATTERN.test(template)) {
    fail(path, "cannot contain path separators");
  }
  const tokenMatches = [...template.matchAll(TEMPLATE_TOKEN_PATTERN)];
  const tokens = tokenMatches.map((match) => match[1]);
  const unknown = tokens.filter(
    (token) =>
      !(FILE_NAME_TEMPLATE_TOKENS as readonly string[]).includes(token),
  );
  if (unknown.length > 0) {
    fail(path, `has unknown tokens: ${[...new Set(unknown)].join(", ")}`);
  }
  if (tokens.length === 0) {
    fail(path, "must contain at least one supported token");
  }
  const withoutTokens = template.replace(TEMPLATE_TOKEN_PATTERN, "");
  if (/[{}]/u.test(withoutTokens)) {
    fail(path, "contains malformed token braces");
  }
  return template;
}

/**
 * Strict fail-closed validation. This function never drops fields, clamps
 * numbers, injects fallback formats, rewrites text, or repairs relationships.
 */
export function validateOutputCustomization(
  value: unknown,
): OutputCustomization {
  const root = record(value, "outputCustomization");
  exactKeys(root, "outputCustomization", [
    "schemaVersion",
    "report",
    "subtitles",
    "delivery",
    "finishing",
  ]);
  if (root.schemaVersion !== OUTPUT_CUSTOMIZATION_SCHEMA_VERSION) {
    fail(
      "outputCustomization.schemaVersion",
      `must be exactly '${OUTPUT_CUSTOMIZATION_SCHEMA_VERSION}'`,
    );
  }

  const report = record(root.report, "outputCustomization.report");
  exactKeys(report, "outputCustomization.report", [
    "template",
    "font",
    "customFontFamily",
    "pageSize",
    "density",
    "accentColor",
  ]);
  const reportFont = enumeration(
    report.font,
    "outputCustomization.report.font",
    REPORT_FONT_IDS,
  );
  const customFontFamily = exactText(
    report.customFontFamily,
    "outputCustomization.report.customFontFamily",
    OUTPUT_CUSTOMIZATION_LIMITS.customText,
    true,
  );
  if (reportFont === "custom" && customFontFamily.length === 0) {
    fail(
      "outputCustomization.report.customFontFamily",
      "is required when font is 'custom'",
    );
  }
  const normalizedReport: ReportCustomization = {
    template: enumeration(
      report.template,
      "outputCustomization.report.template",
      REPORT_TEMPLATE_IDS,
    ),
    font: reportFont,
    customFontFamily,
    pageSize: enumeration(
      report.pageSize,
      "outputCustomization.report.pageSize",
      REPORT_PAGE_SIZES,
    ),
    density: enumeration(
      report.density,
      "outputCustomization.report.density",
      REPORT_DENSITIES,
    ),
    accentColor: color(
      report.accentColor,
      "outputCustomization.report.accentColor",
    ),
  };

  const subtitles = record(root.subtitles, "outputCustomization.subtitles");
  exactKeys(subtitles, "outputCustomization.subtitles", [
    "enabled",
    "theme",
    "size",
    "safeArea",
    "position",
    "speakerPalette",
    "backgroundOpacity",
    "maximumLines",
    "avoidVisualCollisions",
    "wordProgressHighlight",
  ]);
  const subtitlesEnabled = boolean(
    subtitles.enabled,
    "outputCustomization.subtitles.enabled",
  );
  if (subtitles.wordProgressHighlight !== false) {
    fail(
      "outputCustomization.subtitles.wordProgressHighlight",
      "must remain false until verified word-level timing evidence is supported",
    );
  }
  const normalizedSubtitles: SubtitleCustomization = {
    enabled: subtitlesEnabled,
    theme: enumeration(
      subtitles.theme,
      "outputCustomization.subtitles.theme",
      SUBTITLE_THEME_IDS,
    ),
    size: enumeration(
      subtitles.size,
      "outputCustomization.subtitles.size",
      SUBTITLE_SIZE_IDS,
    ),
    safeArea: enumeration(
      subtitles.safeArea,
      "outputCustomization.subtitles.safeArea",
      SUBTITLE_SAFE_AREA_IDS,
    ),
    position: enumeration(
      subtitles.position,
      "outputCustomization.subtitles.position",
      SUBTITLE_POSITION_IDS,
    ),
    speakerPalette: enumeration(
      subtitles.speakerPalette,
      "outputCustomization.subtitles.speakerPalette",
      SPEAKER_PALETTE_IDS,
    ),
    backgroundOpacity: integer(
      subtitles.backgroundOpacity,
      "outputCustomization.subtitles.backgroundOpacity",
      0,
      100,
    ),
    maximumLines: integer(
      subtitles.maximumLines,
      "outputCustomization.subtitles.maximumLines",
      1,
      3,
    ) as 1 | 2 | 3,
    avoidVisualCollisions: boolean(
      subtitles.avoidVisualCollisions,
      "outputCustomization.subtitles.avoidVisualCollisions",
    ),
    wordProgressHighlight: false,
  };

  const delivery = record(root.delivery, "outputCustomization.delivery");
  exactKeys(delivery, "outputCustomization.delivery", [
    "formats",
    "subtitleModes",
    "includeMediaMetadata",
    "preserveSourceMedia",
    "fileNamePattern",
  ]);
  const formats = uniqueEnumList(
    delivery.formats,
    "outputCustomization.delivery.formats",
    EXPORT_FORMAT_IDS,
    1,
  );
  const subtitleModes = uniqueEnumList(
    delivery.subtitleModes,
    "outputCustomization.delivery.subtitleModes",
    SUBTITLE_DELIVERY_MODE_IDS,
    subtitlesEnabled ? 1 : 0,
  );
  if (!subtitlesEnabled && subtitleModes.length > 0) {
    fail(
      "outputCustomization.delivery.subtitleModes",
      "must be empty while subtitles are disabled",
    );
  }
  if (
    !subtitlesEnabled &&
    formats.some((format) => SUBTITLE_EXPORT_FORMAT_SET.has(format))
  ) {
    fail(
      "outputCustomization.delivery.formats",
      "cannot contain subtitle formats while subtitles are disabled",
    );
  }
  if (delivery.preserveSourceMedia !== true) {
    fail(
      "outputCustomization.delivery.preserveSourceMedia",
      "must remain exactly true",
    );
  }
  const normalizedDelivery: DeliveryCustomization = {
    formats,
    subtitleModes,
    includeMediaMetadata: boolean(
      delivery.includeMediaMetadata,
      "outputCustomization.delivery.includeMediaMetadata",
    ),
    preserveSourceMedia: true,
    fileNamePattern: fileNamePattern(delivery.fileNamePattern),
  };

  const finishing = record(root.finishing, "outputCustomization.finishing");
  exactKeys(finishing, "outputCustomization.finishing", [
    "includeCover",
    "includeChapters",
    "includeTimestamps",
    "includeHeader",
    "includeFooter",
    "includeSpeakerIndex",
    "includeConfidenceNotes",
    "chapterStyle",
    "timestampStyle",
    "customTitle",
  ]);
  const includeChapters = boolean(
    finishing.includeChapters,
    "outputCustomization.finishing.includeChapters",
  );
  const chapterStyle = enumeration(
    finishing.chapterStyle,
    "outputCustomization.finishing.chapterStyle",
    CHAPTER_STYLE_IDS,
  );
  if (includeChapters && chapterStyle === "none") {
    fail(
      "outputCustomization.finishing.chapterStyle",
      "cannot be 'none' while chapters are enabled",
    );
  }
  if (!includeChapters && chapterStyle !== "none") {
    fail(
      "outputCustomization.finishing.chapterStyle",
      "must be 'none' while chapters are disabled",
    );
  }
  const normalizedFinishing: FinishingCustomization = {
    includeCover: boolean(
      finishing.includeCover,
      "outputCustomization.finishing.includeCover",
    ),
    includeChapters,
    includeTimestamps: boolean(
      finishing.includeTimestamps,
      "outputCustomization.finishing.includeTimestamps",
    ),
    includeHeader: boolean(
      finishing.includeHeader,
      "outputCustomization.finishing.includeHeader",
    ),
    includeFooter: boolean(
      finishing.includeFooter,
      "outputCustomization.finishing.includeFooter",
    ),
    includeSpeakerIndex: boolean(
      finishing.includeSpeakerIndex,
      "outputCustomization.finishing.includeSpeakerIndex",
    ),
    includeConfidenceNotes: boolean(
      finishing.includeConfidenceNotes,
      "outputCustomization.finishing.includeConfidenceNotes",
    ),
    chapterStyle,
    timestampStyle: enumeration(
      finishing.timestampStyle,
      "outputCustomization.finishing.timestampStyle",
      TIMESTAMP_STYLE_IDS,
    ),
    customTitle: exactText(
      finishing.customTitle,
      "outputCustomization.finishing.customTitle",
      OUTPUT_CUSTOMIZATION_LIMITS.customText,
      true,
    ),
  };

  return {
    schemaVersion: OUTPUT_CUSTOMIZATION_SCHEMA_VERSION,
    report: normalizedReport,
    subtitles: normalizedSubtitles,
    delivery: normalizedDelivery,
    finishing: normalizedFinishing,
  };
}

/**
 * Compatibility name retained for existing callers. Unlike the former
 * implementation, this is strict validation rather than lossy normalization.
 */
export function normalizeOutputCustomization(
  value: unknown,
): OutputCustomization {
  return validateOutputCustomization(value);
}

export function cloneOutputCustomization(
  value: OutputCustomization,
): OutputCustomization {
  const validated = validateOutputCustomization(value);
  return {
    schemaVersion: OUTPUT_CUSTOMIZATION_SCHEMA_VERSION,
    report: { ...validated.report },
    subtitles: { ...validated.subtitles },
    delivery: {
      ...validated.delivery,
      formats: [...validated.delivery.formats],
      subtitleModes: [...validated.delivery.subtitleModes],
      preserveSourceMedia: true,
    },
    finishing: { ...validated.finishing },
  };
}

export function outputCustomizationValidationMessage(
  value: unknown,
): string | null {
  try {
    validateOutputCustomization(value);
    return null;
  } catch (error) {
    return error instanceof OutputCustomizationValidationError
      ? error.message
      : "outputCustomization: validation failed";
  }
}

export function speakerColorForIndex(
  index: number,
  palette: SpeakerPaletteId,
): string {
  const safeIndex = Number.isFinite(index)
    ? Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.trunc(index)))
    : 0;
  const hue = (safeIndex * 137.508 + 252) % 360;

  switch (palette) {
    case "adaptive-spectrum":
      return `hsl(${hue.toFixed(1)} 68% 52%)`;
    case "candy":
      return `hsl(${((safeIndex * 47 + 322) % 360).toFixed(1)} 72% 64%)`;
    case "ocean":
      return `hsl(${((safeIndex * 31 + 174) % 102 + 174).toFixed(1)} 62% 48%)`;
    case "high-contrast":
      return `hsl(${((safeIndex * 151 + 24) % 360).toFixed(1)} 82% 42%)`;
    case "monochrome":
      return `hsl(260 8% ${String(24 + (safeIndex * 13) % 58)}%)`;
  }
}
