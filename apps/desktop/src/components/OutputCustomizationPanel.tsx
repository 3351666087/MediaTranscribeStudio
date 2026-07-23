import {
  Fragment,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import {
  DEFAULT_OUTPUT_CUSTOMIZATION,
  EXPORT_FORMAT_IDS,
  OUTPUT_CUSTOMIZATION_LIMITS,
  OUTPUT_CUSTOMIZATION_LEVELS,
  REPORT_DENSITIES,
  REPORT_FONT_IDS,
  REPORT_PAGE_SIZES,
  REPORT_TEMPLATE_IDS,
  SPEAKER_PALETTE_IDS,
  SUBTITLE_DELIVERY_MODE_IDS,
  SUBTITLE_EXPORT_FORMAT_IDS,
  SUBTITLE_POSITION_IDS,
  SUBTITLE_SAFE_AREA_IDS,
  SUBTITLE_SIZE_IDS,
  SUBTITLE_THEME_IDS,
  cloneOutputCustomization,
  outputCustomizationValidationMessage,
  speakerColorForIndex,
  validateOutputCustomization,
  type DeliveryCustomization,
  type ExportFormatId,
  type FinishingCustomization,
  type OutputCustomization,
  type OutputCustomizationLevel,
  type OutputCustomizationQuality,
  type ReportCustomization,
  type SpeakerPaletteId,
  type SubtitleCustomization,
  type SubtitleDeliveryModeId,
} from "../contracts/output-customization";
import { useI18n } from "../i18n";
import outputCustomizationMessages from "../i18n/fragments/output-customization.json";
import { cx } from "../lib/format";
import { Icon, type IconName } from "./Icon";
import "../styles/output-customization.css";

type OutputCustomizationMessageKey =
  keyof typeof outputCustomizationMessages.en;
type OutputCustomizationMessageParams = Readonly<
  Partial<Record<string, string | number>>
>;

const LEVEL_ICONS: Record<OutputCustomizationLevel, IconName> = {
  report: "pdf",
  subtitles: "review",
  delivery: "download",
  finishing: "sparkles",
};

const LEVEL_COPY = {
  report: {
    label: "output.level.report.title",
    detail: "output.level.report.detail",
  },
  subtitles: {
    label: "output.level.subtitles.title",
    detail: "output.level.subtitles.detail",
  },
  delivery: {
    label: "output.level.delivery.title",
    detail: "output.level.delivery.detail",
  },
  finishing: {
    label: "output.level.finishing.title",
    detail: "output.level.finishing.detail",
  },
} as const satisfies Record<
  OutputCustomizationLevel,
  {
    label: OutputCustomizationMessageKey;
    detail: OutputCustomizationMessageKey;
  }
>;

const REPORT_TEMPLATE_COPY = {
  "soft-glass": {
    label: "output.report.template.softGlass",
    detail: "output.report.template.softGlassDetail",
  },
  editorial: {
    label: "output.report.template.editorial",
    detail: "output.report.template.editorialDetail",
  },
  academic: {
    label: "output.report.template.academic",
    detail: "output.report.template.academicDetail",
  },
  compact: {
    label: "output.report.template.compact",
    detail: "output.report.template.compactDetail",
  },
} as const;

const REPORT_FONT_COPY = {
  "system-sans": "output.report.font.system",
  humanist: "output.report.font.humanist",
  serif: "output.report.font.serif",
  "mono-accent": "output.report.font.monoAccent",
  custom: "output.report.font.custom",
} as const satisfies Record<
  (typeof REPORT_FONT_IDS)[number],
  OutputCustomizationMessageKey
>;

const REPORT_PAGE_COPY = {
  a4: "output.report.page.a4",
  letter: "output.report.page.letter",
  legal: "output.report.page.legal",
  screen: "output.report.page.screen",
} as const satisfies Record<
  (typeof REPORT_PAGE_SIZES)[number],
  OutputCustomizationMessageKey
>;

const REPORT_DENSITY_COPY = {
  airy: "output.report.density.airy",
  balanced: "output.report.density.balanced",
  compact: "output.report.density.compact",
} as const satisfies Record<
  (typeof REPORT_DENSITIES)[number],
  OutputCustomizationMessageKey
>;

const SUBTITLE_THEME_COPY = {
  "youtube-clean": "output.subtitles.theme.youtube",
  "soft-bubble": "output.subtitles.theme.bubble",
  cinema: "output.subtitles.theme.cinema",
  "high-contrast": "output.subtitles.theme.highContrast",
} as const satisfies Record<
  (typeof SUBTITLE_THEME_IDS)[number],
  OutputCustomizationMessageKey
>;

const SUBTITLE_SIZE_COPY = {
  small: "output.subtitles.size.small",
  medium: "output.subtitles.size.medium",
  large: "output.subtitles.size.large",
  "x-large": "output.subtitles.size.xLarge",
} as const satisfies Record<
  (typeof SUBTITLE_SIZE_IDS)[number],
  OutputCustomizationMessageKey
>;

const SUBTITLE_SAFE_AREA_COPY = {
  standard: "output.subtitles.safeArea.standard",
  broadcast: "output.subtitles.safeArea.broadcast",
  generous: "output.subtitles.safeArea.generous",
} as const satisfies Record<
  (typeof SUBTITLE_SAFE_AREA_IDS)[number],
  OutputCustomizationMessageKey
>;

const SUBTITLE_POSITION_COPY = {
  bottom: "output.subtitles.position.bottom",
  smart: "output.subtitles.position.smart",
  top: "output.subtitles.position.top",
} as const satisfies Record<
  (typeof SUBTITLE_POSITION_IDS)[number],
  OutputCustomizationMessageKey
>;

const SPEAKER_PALETTE_COPY = {
  "adaptive-spectrum": "output.subtitles.palette.adaptive",
  candy: "output.subtitles.palette.candy",
  ocean: "output.subtitles.palette.ocean",
  "high-contrast": "output.subtitles.palette.highContrast",
  monochrome: "output.subtitles.palette.monochrome",
} as const satisfies Record<
  SpeakerPaletteId,
  OutputCustomizationMessageKey
>;

const EXPORT_FORMAT_COPY = {
  pdf: "output.delivery.format.pdf",
  html: "output.delivery.format.html",
  markdown: "output.delivery.format.markdown",
  txt: "output.delivery.format.txt",
  json: "output.delivery.format.json",
  srt: "output.delivery.format.srt",
  webvtt: "output.delivery.format.vtt",
  ass: "output.delivery.format.ass",
} as const satisfies Record<
  ExportFormatId,
  OutputCustomizationMessageKey
>;

const SUBTITLE_EXPORT_FORMAT_SET: ReadonlySet<ExportFormatId> = new Set(
  SUBTITLE_EXPORT_FORMAT_IDS,
);

const SUBTITLE_MODE_COPY = {
  sidecar: {
    label: "output.delivery.mode.sidecar",
    detail: "output.delivery.mode.sidecarDetail",
  },
  "soft-mux": {
    label: "output.delivery.mode.softMux",
    detail: "output.delivery.mode.softMuxDetail",
  },
  "burn-in": {
    label: "output.delivery.mode.burnIn",
    detail: "output.delivery.mode.burnInDetail",
  },
} as const satisfies Record<
  SubtitleDeliveryModeId,
  {
    label: OutputCustomizationMessageKey;
    detail: OutputCustomizationMessageKey;
  }
>;

const DEFAULT_QUALITY_GATES = [
  {
    id: "readability",
    label: "output.score.gate.readability",
    detail: "output.score.gate.readabilityDetail",
  },
  {
    id: "safe-area",
    label: "output.score.gate.safeArea",
    detail: "output.score.gate.safeAreaDetail",
  },
  {
    id: "font-evidence",
    label: "output.score.gate.fonts",
    detail: "output.score.gate.fontsDetail",
  },
  {
    id: "source-safety",
    label: "output.score.gate.sourceSafety",
    detail: "output.score.gate.sourceSafetyDetail",
  },
] as const;

const FINISHING_TOGGLES = [
  {
    field: "includeCover",
    label: "output.finishing.cover",
    detail: "output.finishing.coverDetail",
  },
  {
    field: "includeChapters",
    label: "output.finishing.chapters",
    detail: "output.finishing.chaptersDetail",
  },
  {
    field: "includeTimestamps",
    label: "output.finishing.timestamps",
    detail: "output.finishing.timestampsDetail",
  },
  {
    field: "includeHeader",
    label: "output.finishing.header",
    detail: "output.finishing.headerDetail",
  },
  {
    field: "includeFooter",
    label: "output.finishing.footer",
    detail: "output.finishing.footerDetail",
  },
  {
    field: "includeSpeakerIndex",
    label: "output.finishing.speakerIndex",
    detail: "output.finishing.speakerIndexDetail",
  },
  {
    field: "includeConfidenceNotes",
    label: "output.finishing.confidence",
    detail: "output.finishing.confidenceDetail",
  },
] as const satisfies ReadonlyArray<{
  field: keyof Pick<
    FinishingCustomization,
    | "includeCover"
    | "includeChapters"
    | "includeTimestamps"
    | "includeHeader"
    | "includeFooter"
    | "includeSpeakerIndex"
    | "includeConfidenceNotes"
  >;
  label: OutputCustomizationMessageKey;
  detail: OutputCustomizationMessageKey;
}>;

export interface OutputCustomizationPanelProps {
  value?: OutputCustomization;
  defaultValue?: OutputCustomization;
  initialLevel?: OutputCustomizationLevel;
  speakerCount?: number;
  speakerLabels?: readonly string[];
  quality?: OutputCustomizationQuality;
  disabled?: boolean;
  showActions?: boolean;
  className?: string;
  onChange?: (value: OutputCustomization) => void;
  onApply?: (value: OutputCustomization) => void;
}

interface ChoiceOption<T extends string> {
  value: T;
  label: string;
  detail?: string;
  swatch?: ReactNode;
}

function interpolate(
  template: string,
  params: OutputCustomizationMessageParams = {},
): string {
  return template.replace(
    /\{([a-zA-Z][a-zA-Z0-9]*)\}/gu,
    (match, name: string) => {
      const value = params[name];
      return value === undefined ? match : String(value);
    },
  );
}

function useOutputCustomizationI18n() {
  const { locale } = useI18n();
  const localeMessages: Partial<
    Record<OutputCustomizationMessageKey, string>
  > = outputCustomizationMessages[locale];

  return useCallback(
    (
      key: OutputCustomizationMessageKey,
      params: OutputCustomizationMessageParams = {},
    ): string => {
      const template =
        localeMessages[key] ?? outputCustomizationMessages.en[key];
      return interpolate(template, params);
    },
    [localeMessages],
  );
}

function ChoiceStrip<T extends string>({
  ariaLabel,
  disabled,
  options,
  value,
  onChange,
}: {
  ariaLabel: string;
  disabled: boolean;
  options: ReadonlyArray<ChoiceOption<T>>;
  value: T;
  onChange: (value: T) => void;
}) {
  const descriptionsId = useId();

  return (
    <div className="output-diy-choice-strip" role="group" aria-label={ariaLabel}>
      {options.map((option, index) => {
        const selected = option.value === value;
        const descriptionId =
          option.detail === undefined
            ? undefined
            : `${descriptionsId}-description-${String(index)}`;
        return (
          <Fragment key={option.value}>
            <button
              className={cx(
                "output-diy-choice",
                selected && "output-diy-choice--active",
              )}
              type="button"
              aria-pressed={selected}
              aria-describedby={descriptionId}
              disabled={disabled}
              title={option.detail}
              onClick={() => {
                onChange(option.value);
              }}
            >
              {option.swatch}
              <span>{option.label}</span>
            </button>
            {option.detail === undefined ? null : (
              <span className="output-diy-sr-only" id={descriptionId}>
                {option.detail}
              </span>
            )}
          </Fragment>
        );
      })}
    </div>
  );
}

function ToggleRow({
  checked,
  detail,
  disabled,
  label,
  onChange,
}: {
  checked: boolean;
  detail: string;
  disabled: boolean;
  label: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="output-diy-toggle">
      <span className="output-diy-toggle__copy">
        <strong>{label}</strong>
        <small>{detail}</small>
      </span>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => {
          onChange(event.target.checked);
        }}
      />
      <span className="output-diy-toggle__track" aria-hidden="true">
        <span />
      </span>
    </label>
  );
}

function normalizeQualityScore(score: number | null): number | null {
  return (
    score === null || !Number.isFinite(score)
      ? null
      : Math.min(100, Math.max(0, Math.round(score)))
  );
}

function QualityRing({
  label,
  score,
}: {
  label: string;
  score: number | null;
}) {
  const normalized = normalizeQualityScore(score);
  const style = {
    "--output-diy-score-angle":
      normalized === null ? "0deg" : `${normalized * 3.6}deg`,
  } as CSSProperties;

  return (
    <div
      className="output-diy-score-ring"
      role="img"
      aria-label={
        normalized === null ? `${label}: —` : `${label}: ${normalized}%`
      }
      data-available={normalized === null ? "false" : "true"}
      style={style}
    >
      <span aria-hidden="true">{normalized ?? "—"}</span>
    </div>
  );
}

function updateArraySelection<T extends string>(
  current: readonly T[],
  value: T,
  selected: boolean,
): T[] {
  if (selected) {
    return current.includes(value) ? [...current] : [...current, value];
  }
  return current.filter((item) => item !== value);
}

export function OutputCustomizationPanel({
  value,
  defaultValue = cloneOutputCustomization(DEFAULT_OUTPUT_CUSTOMIZATION),
  initialLevel = "report",
  speakerCount = 0,
  speakerLabels = [],
  quality = {
    score: null,
    status: "pending",
  },
  disabled = false,
  showActions = true,
  className,
  onChange,
  onApply,
}: OutputCustomizationPanelProps) {
  const t = useOutputCustomizationI18n();
  const [current, setCurrent] = useState<OutputCustomization>(() =>
    cloneOutputCustomization(value ?? defaultValue),
  );
  const [interactionError, setInteractionError] = useState<string | null>(null);
  const [activeLevel, setActiveLevel] =
    useState<OutputCustomizationLevel>(initialLevel);
  const panelId = useId();
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  useEffect(() => {
    if (value === undefined) {
      return;
    }
    setCurrent(cloneOutputCustomization(value));
    setInteractionError(null);
  }, [value]);
  const validationMessage = useMemo(
    () => outputCustomizationValidationMessage(current),
    [current],
  );
  const blockingMessage = interactionError ?? validationMessage;
  const requestedSpeakerCount =
    Number.isFinite(speakerCount) && speakerCount > 0
      ? Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(speakerCount))
      : 0;
  const safeSpeakerCount = Math.max(
    requestedSpeakerCount,
    speakerLabels.length,
  );
  const visibleSpeakerCount = Math.min(safeSpeakerCount, 12);
  const normalizedQualityScore = normalizeQualityScore(quality.score);
  const minimumQualityScore =
    quality.minimumScore === undefined ||
    !Number.isFinite(quality.minimumScore)
      ? 90
      : Math.min(100, Math.max(0, Math.round(quality.minimumScore)));

  const qualityGates = useMemo(() => {
    if (quality.gates !== undefined) {
      return quality.gates;
    }
    return DEFAULT_QUALITY_GATES.map((gate) => ({
      id: gate.id,
      label: t(gate.label),
      detail: t(gate.detail),
      status: "pending" as const,
    }));
  }, [quality.gates, t]);

  const qualityPassedCount = qualityGates.filter(
    (gate) => gate.status === "passed",
  ).length;
  const qualityStatusKey = {
    pending: "output.score.status.pending",
    passed: "output.score.status.passed",
    "review-required": "output.score.status.review",
    blocked: "output.score.status.blocked",
  } as const satisfies Record<
    OutputCustomizationQuality["status"],
    OutputCustomizationMessageKey
  >;
  const qualityGateStatusKey = {
    pending: "output.score.gateStatus.pending",
    passed: "output.score.gateStatus.passed",
    failed: "output.score.gateStatus.failed",
  } as const satisfies Record<
    NonNullable<OutputCustomizationQuality["gates"]>[number]["status"],
    OutputCustomizationMessageKey
  >;

  const emit = (nextValue: OutputCustomization) => {
    setCurrent(nextValue);
    const nextValidationMessage =
      outputCustomizationValidationMessage(nextValue);
    setInteractionError(null);
    if (nextValidationMessage !== null) {
      return;
    }
    onChange?.(validateOutputCustomization(nextValue));
  };

  const updateReport = (patch: Partial<ReportCustomization>) => {
    emit({
      ...current,
      report: {
        ...current.report,
        ...patch,
      },
    });
  };

  const updateSubtitles = (patch: Partial<SubtitleCustomization>) => {
    emit({
      ...current,
      subtitles: {
        ...current.subtitles,
        ...patch,
      },
    });
  };

  const updateDelivery = (patch: Partial<DeliveryCustomization>) => {
    emit({
      ...current,
      delivery: {
        ...current.delivery,
        ...patch,
      },
    });
  };

  const updateFinishing = (patch: Partial<FinishingCustomization>) => {
    emit({
      ...current,
      finishing: {
        ...current.finishing,
        ...patch,
      },
    });
  };

  const activateLevel = (level: OutputCustomizationLevel) => {
    setActiveLevel(level);
  };

  const handleTabKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = (index + 1) % OUTPUT_CUSTOMIZATION_LEVELS.length;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex =
        (index - 1 + OUTPUT_CUSTOMIZATION_LEVELS.length) %
        OUTPUT_CUSTOMIZATION_LEVELS.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = OUTPUT_CUSTOMIZATION_LEVELS.length - 1;
    }

    if (nextIndex === null) {
      return;
    }
    event.preventDefault();
    const nextLevel = OUTPUT_CUSTOMIZATION_LEVELS[nextIndex];
    activateLevel(nextLevel);
    tabRefs.current[nextIndex]?.focus();
  };

  const reportTemplateOptions = REPORT_TEMPLATE_IDS.map((template) => ({
    value: template,
    label: t(REPORT_TEMPLATE_COPY[template].label),
    detail: t(REPORT_TEMPLATE_COPY[template].detail),
    swatch: (
      <span
        className={`output-diy-template-swatch output-diy-template-swatch--${template}`}
        aria-hidden="true"
      >
        <i />
        <i />
        <i />
      </span>
    ),
  }));

  const reportFontOptions = REPORT_FONT_IDS.map((font) => ({
    value: font,
    label: t(REPORT_FONT_COPY[font]),
    swatch: (
      <span
        className={`output-diy-font-swatch output-diy-font-swatch--${font}`}
        aria-hidden="true"
      >
        Aa
      </span>
    ),
  }));

  const subtitleThemeOptions = SUBTITLE_THEME_IDS.map((theme) => ({
    value: theme,
    label: t(SUBTITLE_THEME_COPY[theme]),
    swatch: (
      <span
        className={`output-diy-caption-swatch output-diy-caption-swatch--${theme}`}
        aria-hidden="true"
      >
        Aa
      </span>
    ),
  }));

  return (
    <section
      className={cx("output-diy", className)}
      aria-labelledby={`${panelId}-title`}
      data-level={activeLevel}
    >
      <header className="output-diy__header">
        <div className="output-diy__heading">
          <span className="output-diy__eyebrow">
            <Icon name="wand" size={17} />
            {t("output.eyebrow")}
          </span>
          <h2 id={`${panelId}-title`}>{t("output.title")}</h2>
          <p>{t("output.description")}</p>
        </div>

        <div
          className="output-diy__quality"
          role="status"
          aria-live="polite"
          data-status={quality.status}
        >
          <QualityRing
            label={t("output.score.aria")}
            score={normalizedQualityScore}
          />
          <div>
            <strong>{t(qualityStatusKey[quality.status])}</strong>
            <small>
              {normalizedQualityScore === null
                ? t("output.score.pending")
                : t("output.score.threshold", {
                    score: minimumQualityScore,
                  })}
            </small>
          </div>
        </div>
      </header>

      <div className="output-diy__body">
        <nav
          className="output-diy__levels"
          aria-label={t("output.navigation.aria")}
        >
          <div
            role="tablist"
            aria-label={t("output.navigation.aria")}
            aria-orientation="vertical"
          >
            {OUTPUT_CUSTOMIZATION_LEVELS.map((level, index) => {
              const active = level === activeLevel;
              return (
                <button
                  ref={(element) => {
                    tabRefs.current[index] = element;
                  }}
                  className={cx(
                    "output-diy__level",
                    active && "output-diy__level--active",
                  )}
                  type="button"
                  role="tab"
                  id={`${panelId}-tab-${level}`}
                  aria-controls={`${panelId}-panel-${level}`}
                  aria-selected={active}
                  tabIndex={active ? 0 : -1}
                  disabled={disabled}
                  onClick={() => {
                    activateLevel(level);
                  }}
                  onKeyDown={(event) => {
                    handleTabKeyDown(event, index);
                  }}
                  key={level}
                >
                  <span className="output-diy__level-icon" aria-hidden="true">
                    <Icon name={LEVEL_ICONS[level]} size={19} />
                  </span>
                  <span>
                    <strong>{t(LEVEL_COPY[level].label)}</strong>
                    <small>{t(LEVEL_COPY[level].detail)}</small>
                  </span>
                  <Icon name="arrow-right" size={15} />
                </button>
              );
            })}
          </div>
        </nav>

        <div className="output-diy__workspace">
          <section
            className="output-diy__scene"
            role="tabpanel"
            id={`${panelId}-panel-report`}
            aria-labelledby={`${panelId}-tab-report`}
            hidden={activeLevel !== "report"}
          >
            <div className="output-diy__scene-heading">
              <div>
                <span>{t("output.level.report.title")}</span>
                <h3>{t("output.report.heading")}</h3>
                <p>{t("output.report.detail")}</p>
              </div>
              <span
                className="output-diy__accent-orb"
                style={{ backgroundColor: current.report.accentColor }}
                aria-hidden="true"
              />
            </div>

            <div className="output-diy__section">
              <div className="output-diy__section-label">
                <strong>{t("output.report.template.label")}</strong>
                <small>
                  {t(REPORT_TEMPLATE_COPY[current.report.template].detail)}
                </small>
              </div>
              <ChoiceStrip
                ariaLabel={t("output.report.template.label")}
                disabled={disabled}
                options={reportTemplateOptions}
                value={current.report.template}
                onChange={(template) => {
                  updateReport({ template });
                }}
              />
            </div>

            <div className="output-diy__section">
              <div className="output-diy__section-label">
                <strong>{t("output.report.font.label")}</strong>
                <small>{t("output.report.font.help")}</small>
              </div>
              <ChoiceStrip
                ariaLabel={t("output.report.font.label")}
                disabled={disabled}
                options={reportFontOptions}
                value={current.report.font}
                onChange={(font) => {
                  updateReport({ font });
                }}
              />
              {current.report.font === "custom" ? (
                <label className="output-diy-field">
                  <span id={`${panelId}-custom-font-label`}>
                    {t("output.report.font.customLabel")}
                  </span>
                  <input
                    type="text"
                    value={current.report.customFontFamily}
                    disabled={disabled}
                    maxLength={OUTPUT_CUSTOMIZATION_LIMITS.customText}
                    placeholder={t("output.report.font.customPlaceholder")}
                    autoCapitalize="none"
                    spellCheck={false}
                    aria-labelledby={`${panelId}-custom-font-label`}
                    aria-describedby={`${panelId}-custom-font-help`}
                    aria-invalid={
                      current.report.font === "custom" &&
                      current.report.customFontFamily.length === 0
                    }
                    onChange={(event) => {
                      updateReport({ customFontFamily: event.target.value });
                    }}
                  />
                  <small id={`${panelId}-custom-font-help`}>
                    {t("output.report.font.customHelp")}
                  </small>
                </label>
              ) : null}
            </div>

            <div className="output-diy__compact-grid">
              <div className="output-diy__section">
                <div className="output-diy__section-label">
                  <strong>{t("output.report.page.label")}</strong>
                </div>
                <ChoiceStrip
                  ariaLabel={t("output.report.page.label")}
                  disabled={disabled}
                  options={REPORT_PAGE_SIZES.map((pageSize) => ({
                    value: pageSize,
                    label: t(REPORT_PAGE_COPY[pageSize]),
                  }))}
                  value={current.report.pageSize}
                  onChange={(pageSize) => {
                    updateReport({ pageSize });
                  }}
                />
              </div>

              <div className="output-diy__section">
                <div className="output-diy__section-label">
                  <strong>{t("output.report.density.label")}</strong>
                </div>
                <ChoiceStrip
                  ariaLabel={t("output.report.density.label")}
                  disabled={disabled}
                  options={REPORT_DENSITIES.map((density) => ({
                    value: density,
                    label: t(REPORT_DENSITY_COPY[density]),
                  }))}
                  value={current.report.density}
                  onChange={(density) => {
                    updateReport({ density });
                  }}
                />
              </div>
            </div>

            <label className="output-diy-color-field">
              <span>
                <strong>{t("output.report.accent.label")}</strong>
                <small>{t("output.report.accent.help")}</small>
              </span>
              <input
                type="color"
                value={current.report.accentColor}
                disabled={disabled}
                aria-label={t("output.report.accent.label")}
                onChange={(event) => {
                  updateReport({ accentColor: event.target.value });
                }}
              />
            </label>
          </section>

          <section
            className="output-diy__scene"
            role="tabpanel"
            id={`${panelId}-panel-subtitles`}
            aria-labelledby={`${panelId}-tab-subtitles`}
            hidden={activeLevel !== "subtitles"}
          >
            <div className="output-diy__scene-heading">
              <div>
                <span>{t("output.level.subtitles.title")}</span>
                <h3>{t("output.subtitles.heading")}</h3>
                <p>{t("output.subtitles.detail")}</p>
              </div>
              <label className="output-diy-inline-switch">
                <span>{t("output.subtitles.enable")}</span>
                <input
                  type="checkbox"
                  checked={current.subtitles.enabled}
                  disabled={disabled}
                  onChange={(event) => {
                    const enabled = event.target.checked;
                    if (!enabled) {
                      const formats = current.delivery.formats.filter(
                        (format) => !SUBTITLE_EXPORT_FORMAT_SET.has(format),
                      );
                      if (formats.length === 0) {
                        setInteractionError(
                          "Select at least one report, transcript, or data format before disabling subtitles.",
                        );
                        return;
                      }
                      emit({
                        ...current,
                        subtitles: {
                          ...current.subtitles,
                          enabled: false,
                          wordProgressHighlight: false,
                        },
                        delivery: {
                          ...current.delivery,
                          formats,
                          subtitleModes: [],
                        },
                      });
                      return;
                    }
                    emit({
                      ...current,
                      subtitles: {
                        ...current.subtitles,
                        enabled: true,
                        wordProgressHighlight: false,
                      },
                      delivery: {
                        ...current.delivery,
                        subtitleModes:
                          current.delivery.subtitleModes.length === 0
                            ? ["sidecar"]
                            : current.delivery.subtitleModes,
                      },
                    });
                  }}
                />
                <span aria-hidden="true">
                  <span />
                </span>
              </label>
            </div>

            <div
              className="output-diy-subtitle-preview"
              data-theme={current.subtitles.theme}
              data-size={current.subtitles.size}
              data-disabled={current.subtitles.enabled ? "false" : "true"}
            >
              <span>{t("output.subtitles.preview.label")}</span>
              <div>
                <strong>{t("output.subtitles.preview.sample")}</strong>
                <small>{t("output.subtitles.preview.timestamp")}</small>
              </div>
            </div>

            <fieldset
              className="output-diy__section"
              disabled={disabled || !current.subtitles.enabled}
            >
              <legend>{t("output.subtitles.theme.label")}</legend>
              <ChoiceStrip
                ariaLabel={t("output.subtitles.theme.label")}
                disabled={disabled || !current.subtitles.enabled}
                options={subtitleThemeOptions}
                value={current.subtitles.theme}
                onChange={(theme) => {
                  updateSubtitles({ theme });
                }}
              />
            </fieldset>

            <div className="output-diy__compact-grid">
              <fieldset
                className="output-diy__section"
                disabled={disabled || !current.subtitles.enabled}
              >
                <legend>{t("output.subtitles.size.label")}</legend>
                <ChoiceStrip
                  ariaLabel={t("output.subtitles.size.label")}
                  disabled={disabled || !current.subtitles.enabled}
                  options={SUBTITLE_SIZE_IDS.map((size) => ({
                    value: size,
                    label: t(SUBTITLE_SIZE_COPY[size]),
                  }))}
                  value={current.subtitles.size}
                  onChange={(size) => {
                    updateSubtitles({ size });
                  }}
                />
              </fieldset>

              <fieldset
                className="output-diy__section"
                disabled={disabled || !current.subtitles.enabled}
              >
                <legend>{t("output.subtitles.safeArea.label")}</legend>
                <ChoiceStrip
                  ariaLabel={t("output.subtitles.safeArea.label")}
                  disabled={disabled || !current.subtitles.enabled}
                  options={SUBTITLE_SAFE_AREA_IDS.map((safeArea) => ({
                    value: safeArea,
                    label: t(SUBTITLE_SAFE_AREA_COPY[safeArea]),
                  }))}
                  value={current.subtitles.safeArea}
                  onChange={(safeArea) => {
                    updateSubtitles({ safeArea });
                  }}
                />
              </fieldset>
            </div>

            <div className="output-diy__compact-grid">
              <fieldset
                className="output-diy__section"
                disabled={disabled || !current.subtitles.enabled}
              >
                <legend>{t("output.subtitles.position.label")}</legend>
                <ChoiceStrip
                  ariaLabel={t("output.subtitles.position.label")}
                  disabled={disabled || !current.subtitles.enabled}
                  options={SUBTITLE_POSITION_IDS.map((position) => ({
                    value: position,
                    label: t(SUBTITLE_POSITION_COPY[position]),
                  }))}
                  value={current.subtitles.position}
                  onChange={(position) => {
                    updateSubtitles({ position });
                  }}
                />
              </fieldset>

              <div className="output-diy__section output-diy__sliders">
                <label>
                  <span>
                    {t("output.subtitles.backgroundOpacity")}
                    <strong>{current.subtitles.backgroundOpacity}%</strong>
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="4"
                    value={current.subtitles.backgroundOpacity}
                    disabled={disabled || !current.subtitles.enabled}
                    onChange={(event) => {
                      updateSubtitles({
                        backgroundOpacity: Number(event.target.value),
                      });
                    }}
                  />
                </label>
                <label>
                  <span>
                    {t("output.subtitles.maximumLines")}
                    <strong>{current.subtitles.maximumLines}</strong>
                  </span>
                  <input
                    type="range"
                    min="1"
                    max="3"
                    step="1"
                    value={current.subtitles.maximumLines}
                    disabled={disabled || !current.subtitles.enabled}
                    onChange={(event) => {
                      updateSubtitles({
                        maximumLines: Number(event.target.value) as 1 | 2 | 3,
                      });
                    }}
                  />
                </label>
              </div>
            </div>

            <div className="output-diy__section">
              <div className="output-diy__section-label">
                <strong>{t("output.subtitles.palette.label")}</strong>
                <small>
                  {t("output.subtitles.palette.detail", {
                    count: safeSpeakerCount,
                  })}
                </small>
              </div>
              <ChoiceStrip
                ariaLabel={t("output.subtitles.palette.label")}
                disabled={disabled || !current.subtitles.enabled}
                options={SPEAKER_PALETTE_IDS.map((palette) => ({
                  value: palette,
                  label: t(SPEAKER_PALETTE_COPY[palette]),
                  swatch: (
                    <span className="output-diy-palette-swatch" aria-hidden="true">
                      {[0, 1, 2].map((index) => (
                        <i
                          style={{
                            backgroundColor: speakerColorForIndex(
                              index,
                              palette,
                            ),
                          }}
                          key={index}
                        />
                      ))}
                    </span>
                  ),
                }))}
                value={current.subtitles.speakerPalette}
                onChange={(speakerPalette) => {
                  updateSubtitles({ speakerPalette });
                }}
              />

              <div
                className="output-diy-speaker-colors"
                aria-label={t("output.subtitles.palette.preview")}
                role={safeSpeakerCount === 0 ? "status" : "list"}
              >
                {Array.from({ length: visibleSpeakerCount }, (_, index) => {
                  const speakerLabel =
                    speakerLabels[index] ??
                    t("output.subtitles.speakerFallback", {
                      number: index + 1,
                    });
                  return (
                    <span
                      style={{
                        "--output-diy-speaker-color": speakerColorForIndex(
                          index,
                          current.subtitles.speakerPalette,
                        ),
                      } as CSSProperties}
                      role="listitem"
                      aria-label={speakerLabel}
                      title={speakerLabel}
                      key={index}
                    >
                      {speakerLabels[index]?.slice(0, 2) ?? index + 1}
                    </span>
                  );
                })}
                {safeSpeakerCount > visibleSpeakerCount ? (
                  <span
                    className="output-diy-speaker-colors__more"
                    role="listitem"
                    aria-label={t("output.subtitles.palette.more", {
                      count: safeSpeakerCount - visibleSpeakerCount,
                    })}
                  >
                    +{safeSpeakerCount - visibleSpeakerCount}
                  </span>
                ) : null}
                {safeSpeakerCount === 0 ? (
                  <small>{t("output.subtitles.palette.autoCount")}</small>
                ) : null}
              </div>
            </div>

            <div className="output-diy-toggle-list">
              <ToggleRow
                checked={current.subtitles.avoidVisualCollisions}
                disabled={disabled || !current.subtitles.enabled}
                label={t("output.subtitles.collision.label")}
                detail={t("output.subtitles.collision.detail")}
                onChange={(avoidVisualCollisions) => {
                  updateSubtitles({ avoidVisualCollisions });
                }}
              />
              <ToggleRow
                checked={false}
                disabled
                label={t("output.subtitles.wordProgress.label")}
                detail={t("output.subtitles.wordProgress.detail")}
                onChange={() => undefined}
              />
            </div>
            <p className="output-diy__evidence-note" role="note">
              <Icon name="shield" size={16} />
              {t("output.subtitles.wordProgress.evidence")}
            </p>
          </section>

          <section
            className="output-diy__scene"
            role="tabpanel"
            id={`${panelId}-panel-delivery`}
            aria-labelledby={`${panelId}-tab-delivery`}
            hidden={activeLevel !== "delivery"}
          >
            <div className="output-diy__scene-heading">
              <div>
                <span>{t("output.level.delivery.title")}</span>
                <h3>{t("output.delivery.heading")}</h3>
                <p>{t("output.delivery.detail")}</p>
              </div>
              <Icon name="download" size={28} />
            </div>

            <fieldset className="output-diy__section">
              <legend>{t("output.delivery.formats.label")}</legend>
              <p
                className="output-diy-sr-only"
                id={`${panelId}-subtitle-format-help`}
              >
                {t("output.delivery.formats.requiresSubtitles")}
              </p>
              <div
                className="output-diy-format-grid"
                aria-label={t("output.delivery.formats.label")}
              >
                {EXPORT_FORMAT_IDS.map((format) => {
                  const checked = current.delivery.formats.includes(format);
                  const requiresSubtitles =
                    SUBTITLE_EXPORT_FORMAT_SET.has(format);
                  const formatDisabled =
                    disabled ||
                    (requiresSubtitles && !current.subtitles.enabled);
                  return (
                    <label
                      className={cx(
                        "output-diy-format",
                        checked && "output-diy-format--active",
                      )}
                      key={format}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={formatDisabled}
                        aria-describedby={
                          requiresSubtitles && !current.subtitles.enabled
                            ? `${panelId}-subtitle-format-help`
                            : undefined
                        }
                        onChange={(event) => {
                          if (
                            !event.target.checked &&
                            current.delivery.formats.length === 1
                          ) {
                            setInteractionError(
                              "At least one output format is required.",
                            );
                            return;
                          }
                          updateDelivery({
                            formats: updateArraySelection(
                              current.delivery.formats,
                              format,
                              event.target.checked,
                            ),
                          });
                        }}
                      />
                      <span>{t(EXPORT_FORMAT_COPY[format])}</span>
                      <Icon name={checked ? "check" : "file"} size={16} />
                    </label>
                  );
                })}
              </div>
            </fieldset>

            <fieldset
              className="output-diy__section"
              disabled={!current.subtitles.enabled}
            >
              <legend>{t("output.delivery.modes.label")}</legend>
              <div className="output-diy-mode-list">
                {SUBTITLE_DELIVERY_MODE_IDS.map((mode) => {
                  const checked =
                    current.delivery.subtitleModes.includes(mode);
                  return (
                    <ToggleRow
                      checked={checked}
                      disabled={disabled || !current.subtitles.enabled}
                      label={t(SUBTITLE_MODE_COPY[mode].label)}
                      detail={t(SUBTITLE_MODE_COPY[mode].detail)}
                      onChange={(selected) => {
                        if (
                          !selected &&
                          current.delivery.subtitleModes.length === 1
                        ) {
                          setInteractionError(
                            "At least one subtitle delivery mode is required while subtitles are enabled.",
                          );
                          return;
                        }
                        updateDelivery({
                          subtitleModes: updateArraySelection(
                            current.delivery.subtitleModes,
                            mode,
                            selected,
                          ),
                        });
                      }}
                      key={mode}
                    />
                  );
                })}
              </div>
            </fieldset>

            <div
              className="output-diy-container-map"
              role="note"
              aria-label="Delivery container mapping"
            >
              <strong>Backend-verified container mapping</strong>
              <span data-active={current.delivery.subtitleModes.includes("sidecar")}>
                Sidecar · SRT / WebVTT / ASS
              </span>
              <span
                data-active={current.delivery.subtitleModes.includes("soft-mux")}
              >
                Soft mux · Matroska + ASS
              </span>
              <span data-active={current.delivery.subtitleModes.includes("burn-in")}>
                Burn-in · MP4 + H.264
              </span>
              <small>
                Container and codec are compiled by the backend; unsupported
                values are never sent from this UI.
              </small>
            </div>

            <div className="output-diy-source-safety" role="note">
              <span aria-hidden="true">
                <Icon name="lock" size={22} />
              </span>
              <div>
                <strong>{t("output.delivery.sourceSafety.title")}</strong>
                <p>{t("output.delivery.sourceSafety.detail")}</p>
              </div>
            </div>

            <div className="output-diy-toggle-list">
              <ToggleRow
                checked={current.delivery.includeMediaMetadata}
                disabled={disabled}
                label={t("output.delivery.metadata.label")}
                detail={t("output.delivery.metadata.detail")}
                onChange={(includeMediaMetadata) => {
                  updateDelivery({ includeMediaMetadata });
                }}
              />
            </div>

            <label className="output-diy-field">
              <span id={`${panelId}-file-name-label`}>
                {t("output.delivery.fileName.label")}
              </span>
              <input
                type="text"
                value={current.delivery.fileNamePattern}
                disabled={disabled}
                maxLength={OUTPUT_CUSTOMIZATION_LIMITS.fileNamePattern}
                autoCapitalize="none"
                spellCheck={false}
                aria-labelledby={`${panelId}-file-name-label`}
                aria-describedby={`${panelId}-file-name-help`}
                aria-invalid={validationMessage?.includes("fileNamePattern")}
                onChange={(event) => {
                  updateDelivery({ fileNamePattern: event.target.value });
                }}
              />
              <small id={`${panelId}-file-name-help`}>
                {t("output.delivery.fileName.help")}
              </small>
            </label>
          </section>

          <section
            className="output-diy__scene"
            role="tabpanel"
            id={`${panelId}-panel-finishing`}
            aria-labelledby={`${panelId}-tab-finishing`}
            hidden={activeLevel !== "finishing"}
          >
            <div className="output-diy__scene-heading">
              <div>
                <span>{t("output.level.finishing.title")}</span>
                <h3>{t("output.finishing.heading")}</h3>
                <p>{t("output.finishing.detail")}</p>
              </div>
              <Icon name="sparkles" size={28} />
            </div>

            <label className="output-diy-field">
              <span id={`${panelId}-custom-title-label`}>
                {t("output.finishing.customTitle.label")}
              </span>
              <input
                type="text"
                value={current.finishing.customTitle}
                disabled={disabled}
                maxLength={OUTPUT_CUSTOMIZATION_LIMITS.customText}
                placeholder={t("output.finishing.customTitle.placeholder")}
                aria-labelledby={`${panelId}-custom-title-label`}
                aria-describedby={`${panelId}-custom-title-help`}
                onChange={(event) => {
                  updateFinishing({ customTitle: event.target.value });
                }}
              />
              <small id={`${panelId}-custom-title-help`}>
                {t("output.finishing.customTitle.help")}
              </small>
            </label>

            <div className="output-diy-toggle-list output-diy-toggle-list--columns">
              {FINISHING_TOGGLES.map((toggle) => (
                <ToggleRow
                  checked={current.finishing[toggle.field]}
                  disabled={disabled}
                  label={t(toggle.label)}
                  detail={t(toggle.detail)}
                  onChange={(checked) => {
                    if (toggle.field === "includeChapters") {
                      updateFinishing({
                        includeChapters: checked,
                        chapterStyle: checked ? "semantic" : "none",
                      });
                      return;
                    }
                    updateFinishing({ [toggle.field]: checked });
                  }}
                  key={toggle.field}
                />
              ))}
            </div>

            <div className="output-diy__compact-grid">
              <fieldset
                className="output-diy__section"
                disabled={disabled || !current.finishing.includeChapters}
              >
                <legend>{t("output.finishing.chapterStyle.label")}</legend>
                <ChoiceStrip
                  ariaLabel={t("output.finishing.chapterStyle.label")}
                  disabled={disabled || !current.finishing.includeChapters}
                  options={[
                    {
                      value: "semantic",
                      label: t("output.finishing.chapterStyle.semantic"),
                    },
                    {
                      value: "interval",
                      label: t("output.finishing.chapterStyle.interval"),
                    },
                  ]}
                  value={current.finishing.chapterStyle}
                  onChange={(chapterStyle) => {
                    updateFinishing({ chapterStyle });
                  }}
                />
              </fieldset>

              <fieldset
                className="output-diy__section"
                disabled={disabled || !current.finishing.includeTimestamps}
              >
                <legend>{t("output.finishing.timestampStyle.label")}</legend>
                <ChoiceStrip
                  ariaLabel={t("output.finishing.timestampStyle.label")}
                  disabled={disabled || !current.finishing.includeTimestamps}
                  options={[
                    {
                      value: "segment",
                      label: t("output.finishing.timestampStyle.segment"),
                    },
                    {
                      value: "paragraph",
                      label: t("output.finishing.timestampStyle.paragraph"),
                    },
                    {
                      value: "chapter",
                      label: t("output.finishing.timestampStyle.chapter"),
                    },
                  ]}
                  value={current.finishing.timestampStyle}
                  onChange={(timestampStyle) => {
                    updateFinishing({ timestampStyle });
                  }}
                />
              </fieldset>
            </div>
          </section>
        </div>
      </div>

      <details className="output-diy__quality-details">
        <summary>
          <span>
            <Icon name="shield" size={17} />
            <strong>{t("output.score.details")}</strong>
          </span>
          <small>
            {qualityGates.length === 0
              ? t("output.score.noGateEvidence")
              : t("output.score.passCount", {
                  passed: qualityPassedCount,
                  total: qualityGates.length,
                })}
          </small>
          <Icon name="chevron-down" size={16} />
        </summary>
        <ul>
          {qualityGates.map((gate) => (
            <li data-status={gate.status} key={gate.id}>
              <span aria-hidden="true">
                <Icon
                  name={
                    gate.status === "passed"
                      ? "check"
                      : gate.status === "failed"
                        ? "alert"
                        : "clock"
                  }
                  size={16}
                />
              </span>
              <div>
                <strong>{gate.label}</strong>
                <span className="output-diy-sr-only">
                  {t(qualityGateStatusKey[gate.status])}
                </span>
                {gate.detail === undefined ? null : <small>{gate.detail}</small>}
              </div>
            </li>
          ))}
        </ul>
      </details>

      {blockingMessage === null ? null : (
        <div className="output-diy__validation" role="alert">
          <Icon name="alert" size={17} />
          <div>
            <strong>These settings cannot be applied yet.</strong>
            <small>{blockingMessage}</small>
          </div>
        </div>
      )}

      {showActions ? (
        <footer className="output-diy__actions">
          <button
            className="button button--soft"
            type="button"
            disabled={disabled}
            onClick={() => {
              const defaults = cloneOutputCustomization(
                DEFAULT_OUTPUT_CUSTOMIZATION,
              );
              setCurrent(defaults);
              setInteractionError(null);
              onChange?.(defaults);
            }}
          >
            {t("output.actions.reset")}
          </button>
          <button
            className="button button--primary"
            type="button"
            disabled={disabled || validationMessage !== null}
            onClick={() => {
              onApply?.(
                cloneOutputCustomization(validateOutputCustomization(current)),
              );
            }}
          >
            <Icon name="check" size={17} />
            {t("output.actions.apply")}
          </button>
        </footer>
      ) : null}
    </section>
  );
}
