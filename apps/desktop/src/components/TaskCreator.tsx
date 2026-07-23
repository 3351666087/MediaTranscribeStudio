import { useEffect, useRef, useState, type SyntheticEvent } from "react";
import {
  pathsAreDistinct,
  type MediaSelection,
} from "../bridge/media-drop";
import type {
  CreateJobRequest,
  ModelStrategy,
  ModelStrategyId,
  SpeakerCountPolicy,
  SpeakerProfile,
  SystemStatus,
} from "../contracts/studio";
import { isPracticalLanguageTag } from "../contracts/runtime-validation";
import {
  mediaDropErrorMessageKey,
  useI18n,
  type MessageKey,
  type MessageParams,
} from "../i18n";
import { Icon } from "./Icon";
import { SpeakerDetectionSummary } from "./SpeakerDetectionSummary";

export interface InitialMediaSelection extends MediaSelection {
  sequence: number;
}

interface TaskCreatorProps {
  open: boolean;
  initialMediaSelection?: InitialMediaSelection | null;
  resolveMediaPath?: (path: string) => Promise<MediaSelection>;
  speakers: SpeakerProfile[];
  initialSpeakerPolicy: SpeakerCountPolicy;
  strategies: ModelStrategy[];
  selectedStrategyId: ModelStrategyId;
  backendMode: SystemStatus["backendMode"];
  busy: boolean;
  onClose: () => void;
  onCreate: (request: CreateJobRequest) => Promise<unknown>;
}

export const MAX_INLINE_SPEAKER_EDITORS = 32;
const CUSTOM_LANGUAGE_VALUE = "__custom__";
const DEFAULT_LOCAL_MODEL = "qwen3.5:4b";
const DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:11434";
const DEFAULT_SOURCE_LANGUAGE = "auto";
const DEFAULT_OUTPUT_LOCALE = "en-US";
type TaskCreatorMessageKey = Extract<MessageKey, `creator.${string}`>;
const LANGUAGE_PRESETS = [
  ["auto", "creator.language.auto"],
  ["en-US", "creator.language.enUs"],
  ["en-GB", "creator.language.enGb"],
  ["es-ES", "creator.language.esEs"],
  ["es-419", "creator.language.es419"],
  ["fr-FR", "creator.language.fr"],
  ["de-DE", "creator.language.de"],
  ["pt-BR", "creator.language.ptBr"],
  ["ja-JP", "creator.language.ja"],
  ["ko-KR", "creator.language.ko"],
  ["zh-Hans", "creator.language.zhHans"],
  ["zh-Hant", "creator.language.zhHant"],
  ["ar", "creator.language.ar"],
  ["hi-IN", "creator.language.hi"],
  ["ru-RU", "creator.language.ru"],
] as const satisfies ReadonlyArray<readonly [string, TaskCreatorMessageKey]>;
const OUTPUT_LOCALE_PRESETS = LANGUAGE_PRESETS.filter(
  ([tag]) => tag !== "auto",
);
function isConcreteLanguageTag(value: string): boolean {
  return isPracticalLanguageTag(value);
}

function parseLanguageTags(value: string): string[] {
  return value
    .split(/[,;\n]/u)
    .map((tag) => tag.trim())
    .filter((tag) => tag.length > 0);
}

function materializeLabels(
  count: number | null,
  current: readonly string[],
  labelForNumber: (number: number) => string,
): string[] {
  if (count === null || count > MAX_INLINE_SPEAKER_EDITORS) {
    return [];
  }
  return Array.from(
    { length: count },
    (_, index) => current.at(index) ?? labelForNumber(index + 1),
  );
}

function hybridLabelCount(
  minimum: number | null,
  maximum: number | null,
  prior: number | null,
): number | null {
  if (
    minimum === null ||
    maximum === null ||
    prior === null ||
    minimum > maximum ||
    prior < minimum ||
    prior > maximum
  ) {
    return null;
  }
  return prior;
}

export function TaskCreator({
  open,
  initialMediaSelection,
  resolveMediaPath,
  speakers,
  initialSpeakerPolicy,
  strategies,
  selectedStrategyId,
  backendMode,
  busy,
  onClose,
  onCreate,
}: TaskCreatorProps) {
  const { t } = useI18n();
  const taskT = (
    key: TaskCreatorMessageKey,
    params?: MessageParams,
  ): string => t(key, params);
  const defaultSpeakerLabel = (number: number): string =>
    taskT("creator.speaker.defaultName", { number });
  const titleInputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const busyRef = useRef(busy);
  const previousOpenRef = useRef(false);
  const appliedSelectionRef = useRef<number | null>(null);
  const outputEditedRef = useRef(false);
  const mediaPathValueRef = useRef("");
  const pathResolutionRef = useRef(0);
  const [title, setTitle] = useState(() =>
    taskT("creator.defaultJobTitle"),
  );
  const [mediaPath, setMediaPath] = useState("");
  const [outputDirectory, setOutputDirectory] = useState("");
  const [mediaPathError, setMediaPathError] = useState<string | null>(null);
  const [resolvingMediaPath, setResolvingMediaPath] = useState(false);
  const [strategyId, setStrategyId] = useState<ModelStrategyId>(selectedStrategyId);
  const [speakerMode, setSpeakerMode] =
    useState<SpeakerCountPolicy["mode"]>(initialSpeakerPolicy.mode);
  const [manualCount, setManualCount] = useState(
    initialSpeakerPolicy.mode === "manual"
      ? String(initialSpeakerPolicy.count)
      : String(Math.max(1, speakers.length)),
  );
  const [hybridMin, setHybridMin] = useState(
    initialSpeakerPolicy.mode === "hybrid"
      ? String(initialSpeakerPolicy.minSpeakers)
      : "2",
  );
  const [hybridMax, setHybridMax] = useState(
    initialSpeakerPolicy.mode === "hybrid"
      ? String(initialSpeakerPolicy.maxSpeakers)
      : String(Math.max(8, speakers.length)),
  );
  const [hybridPrior, setHybridPrior] = useState(
    initialSpeakerPolicy.mode === "hybrid"
      ? String(initialSpeakerPolicy.priorCount)
      : String(Math.max(1, speakers.length)),
  );
  const [speakerLabels, setSpeakerLabels] = useState<string[]>([]);
  const [sourceLanguageChoice, setSourceLanguageChoice] = useState(
    DEFAULT_SOURCE_LANGUAGE,
  );
  const [customSourceLanguage, setCustomSourceLanguage] = useState("");
  const [businessEnabled, setBusinessEnabled] = useState(false);
  const [translationEnabled, setTranslationEnabled] = useState(false);
  const [translationTargetsText, setTranslationTargetsText] = useState("");
  const [polishEnabled, setPolishEnabled] = useState(false);
  const [summaryEnabled, setSummaryEnabled] = useState(false);
  const [outputLocaleChoice, setOutputLocaleChoice] = useState(
    DEFAULT_OUTPUT_LOCALE,
  );
  const [customOutputLocale, setCustomOutputLocale] = useState("");
  const [localLlmModel, setLocalLlmModel] = useState(DEFAULT_LOCAL_MODEL);
  const [localLlmEndpoint, setLocalLlmEndpoint] = useState(
    DEFAULT_LOCAL_ENDPOINT,
  );

  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  useEffect(() => {
    const opening = open && !previousOpenRef.current;
    previousOpenRef.current = open;
    if (!opening) {
      return;
    }

    const selectedMediaPath = initialMediaSelection?.sourcePath ?? "";
    const selectedOutputDirectory =
      initialMediaSelection?.outputDirectory ?? "";
    setTitle(t("creator.defaultJobTitle"));
    setMediaPath(selectedMediaPath);
    mediaPathValueRef.current = selectedMediaPath;
    setOutputDirectory(selectedOutputDirectory);
    outputEditedRef.current = false;
    appliedSelectionRef.current = initialMediaSelection?.sequence ?? null;
    pathResolutionRef.current += 1;
    setMediaPathError(null);
    setResolvingMediaPath(false);
    setStrategyId(selectedStrategyId);
    setSpeakerMode(initialSpeakerPolicy.mode);
    setManualCount(
      initialSpeakerPolicy.mode === "manual"
        ? String(initialSpeakerPolicy.count)
        : String(Math.max(1, speakers.length)),
    );
    setHybridMin(
      initialSpeakerPolicy.mode === "hybrid"
        ? String(initialSpeakerPolicy.minSpeakers)
        : "2",
    );
    setHybridMax(
      initialSpeakerPolicy.mode === "hybrid"
        ? String(initialSpeakerPolicy.maxSpeakers)
        : String(Math.max(8, speakers.length)),
    );
    setHybridPrior(
      initialSpeakerPolicy.mode === "hybrid"
        ? String(initialSpeakerPolicy.priorCount)
        : String(Math.max(1, speakers.length)),
    );
    const initialCount =
      initialSpeakerPolicy.mode === "manual"
        ? initialSpeakerPolicy.count
        : initialSpeakerPolicy.mode === "hybrid"
          ? initialSpeakerPolicy.priorCount
          : null;
    setSpeakerLabels(
      materializeLabels(initialCount, [], (number) =>
        t("creator.speaker.defaultName", { number }),
      ),
    );
    setSourceLanguageChoice(DEFAULT_SOURCE_LANGUAGE);
    setCustomSourceLanguage("");
    setBusinessEnabled(false);
    setTranslationEnabled(false);
    setTranslationTargetsText("");
    setPolishEnabled(false);
    setSummaryEnabled(false);
    setOutputLocaleChoice(DEFAULT_OUTPUT_LOCALE);
    setCustomOutputLocale("");
    setLocalLlmModel(DEFAULT_LOCAL_MODEL);
    setLocalLlmEndpoint(DEFAULT_LOCAL_ENDPOINT);
  }, [
    initialMediaSelection,
    initialSpeakerPolicy,
    open,
    selectedStrategyId,
    speakers,
    t,
  ]);

  useEffect(() => {
    if (
      !open ||
      !initialMediaSelection ||
      appliedSelectionRef.current === initialMediaSelection.sequence
    ) {
      return;
    }

    appliedSelectionRef.current = initialMediaSelection.sequence;
    pathResolutionRef.current += 1;
    mediaPathValueRef.current = initialMediaSelection.sourcePath;
    outputEditedRef.current = false;
    setMediaPath(initialMediaSelection.sourcePath);
    setOutputDirectory(initialMediaSelection.outputDirectory);
    setMediaPathError(null);
    setResolvingMediaPath(false);
  }, [initialMediaSelection, open]);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    restoreFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    window.requestAnimationFrame(() => titleInputRef.current?.focus());

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busyRef.current) {
        event.preventDefault();
        onClose();
        return;
      }

      if (event.key !== "Tab" || !dialogRef.current) {
        return;
      }
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      );
      if (focusable.length === 0) {
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    document.body.classList.add("modal-open");
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.classList.remove("modal-open");
      restoreFocusRef.current?.focus();
    };
  }, [onClose, open]);

  const positiveInteger = (value: string): number | null => {
    if (!/^[1-9]\d*$/u.test(value)) {
      return null;
    }
    const parsed = Number(value);
    return Number.isSafeInteger(parsed) ? parsed : null;
  };

  const manualSpeakerCount = positiveInteger(manualCount);
  const hybridMinCount = positiveInteger(hybridMin);
  const hybridMaxCount = positiveInteger(hybridMax);
  const hybridPriorCount = positiveInteger(hybridPrior);
  const hybridValid =
    hybridMinCount !== null &&
    hybridMaxCount !== null &&
    hybridPriorCount !== null &&
    hybridMinCount <= hybridMaxCount &&
    hybridPriorCount >= hybridMinCount &&
    hybridPriorCount <= hybridMaxCount;
  const configuredSpeakerCount =
    speakerMode === "manual"
      ? manualSpeakerCount
      : speakerMode === "hybrid" && hybridValid
        ? hybridPriorCount
        : null;
  const language =
    sourceLanguageChoice === CUSTOM_LANGUAGE_VALUE
      ? customSourceLanguage.trim()
      : sourceLanguageChoice;
  const sourceLanguageValid =
    language === "auto" || isConcreteLanguageTag(language);
  const translationTargets =
    businessEnabled && translationEnabled
      ? parseLanguageTags(translationTargetsText)
      : [];
  const translationTargetsValid =
    !businessEnabled ||
    !translationEnabled ||
    (translationTargets.length > 0 &&
      translationTargets.every(isConcreteLanguageTag) &&
      new Set(
        translationTargets.map((target) => target.toLocaleLowerCase("en-US")),
      ).size === translationTargets.length);
  const outputLocale =
    outputLocaleChoice === CUSTOM_LANGUAGE_VALUE
      ? customOutputLocale.trim()
      : outputLocaleChoice;
  const outputLocaleValid =
    !businessEnabled || isConcreteLanguageTag(outputLocale);
  const localRuntimeValid =
    !businessEnabled ||
    (localLlmModel.trim().length > 0 &&
      /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:\/.*)?$/u.test(
        localLlmEndpoint.trim(),
      ));
  const outputPathDistinct = pathsAreDistinct(mediaPath, outputDirectory);

  const deriveOutputDirectory = async () => {
    const candidate = mediaPath.trim();
    if (
      !resolveMediaPath ||
      candidate.length === 0 ||
      outputEditedRef.current
    ) {
      return;
    }

    const operation = pathResolutionRef.current + 1;
    pathResolutionRef.current = operation;
    setResolvingMediaPath(true);
    setMediaPathError(null);

    try {
      const selection = await resolveMediaPath(candidate);
      if (
        pathResolutionRef.current !== operation ||
        mediaPathValueRef.current.trim() !== candidate
      ) {
        return;
      }
      mediaPathValueRef.current = selection.sourcePath;
      setMediaPath(selection.sourcePath);
      setOutputDirectory(selection.outputDirectory);
    } catch (error: unknown) {
      if (pathResolutionRef.current === operation) {
        setMediaPathError(t(mediaDropErrorMessageKey(error)));
      }
    } finally {
      if (pathResolutionRef.current === operation) {
        setResolvingMediaPath(false);
      }
    }
  };

  if (!open) {
    return null;
  }

  const countPolicyValid =
    speakerMode === "auto" ||
    (speakerMode === "manual" && manualSpeakerCount !== null) ||
    (speakerMode === "hybrid" && hybridValid);
  const shouldMaterializeLabels =
    configuredSpeakerCount !== null &&
    configuredSpeakerCount <= MAX_INLINE_SPEAKER_EDITORS;
  const speakerLabelsValid =
    speakerMode === "auto" ||
    (configuredSpeakerCount !== null &&
      (!shouldMaterializeLabels ||
        (speakerLabels.length === configuredSpeakerCount &&
          speakerLabels.every((label) => label.trim().length > 0))));
  const canSubmit =
    title.trim().length > 0 &&
    mediaPath.trim().length > 0 &&
    outputDirectory.trim().length > 0 &&
    mediaPathError === null &&
    outputPathDistinct &&
    countPolicyValid &&
    speakerLabelsValid &&
    sourceLanguageValid &&
    translationTargetsValid &&
    outputLocaleValid &&
    localRuntimeValid;
  let previewPolicy: SpeakerCountPolicy | null = null;
  if (speakerMode === "auto") {
    previewPolicy = { mode: "auto" };
  } else if (speakerMode === "manual" && manualSpeakerCount !== null) {
    previewPolicy = { mode: "manual", count: manualSpeakerCount };
  } else if (
    speakerMode === "hybrid" &&
    hybridMinCount !== null &&
    hybridMaxCount !== null &&
    hybridPriorCount !== null &&
    hybridValid
  ) {
    previewPolicy = {
      mode: "hybrid",
      minSpeakers: hybridMinCount,
      maxSpeakers: hybridMaxCount,
      priorCount: hybridPriorCount,
    };
  }

  const updateSpeakerLabel = (index: number, value: string) => {
    setSpeakerLabels((current) => {
      const next = [...current];
      next[index] = value;
      return next;
    });
  };

  const submit = async (event: SyntheticEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit || busy) {
      return;
    }
    let speakerPolicy: SpeakerCountPolicy;
    if (speakerMode === "auto") {
      speakerPolicy = { mode: "auto" };
    } else if (speakerMode === "manual") {
      if (manualSpeakerCount === null) {
        return;
      }
      speakerPolicy = { mode: "manual", count: manualSpeakerCount };
    } else {
      if (
        hybridMinCount === null ||
        hybridMaxCount === null ||
        hybridPriorCount === null ||
        !hybridValid
      ) {
        return;
      }
      speakerPolicy = {
        mode: "hybrid",
        minSpeakers: hybridMinCount,
        maxSpeakers: hybridMaxCount,
        priorCount: hybridPriorCount,
      };
    }
    await onCreate({
      title: title.trim(),
      mediaPath: mediaPath.trim(),
      outputDirectory: outputDirectory.trim(),
      strategyId,
      speakerPolicy,
      speakerLabels:
        speakerMode === "auto" || !shouldMaterializeLabels
          ? []
          : speakerLabels.map((label) => label.trim()),
      language,
      localLlmMode:
        businessEnabled &&
        (translationTargets.length > 0 || polishEnabled || summaryEnabled)
          ? "business"
          : "disabled",
      localLlmModel: localLlmModel.trim(),
      localLlmEndpoint: localLlmEndpoint.trim(),
      localLlmEndpointPolicy: "loopback-only",
      localLlmAutoApply: false,
      translationTargets,
      polish: businessEnabled && polishEnabled,
      summary: businessEnabled && summaryEnabled,
      outputLocale,
      businessPromptVersion: "business-v1",
    });
    onClose();
  };

  return (
    <div className="modal-layer">
      <button
        className="modal-backdrop"
        type="button"
        aria-label={t("common.close")}
        onClick={() => {
          if (!busy) {
            onClose();
          }
        }}
      />
      <div
        className="task-dialog"
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="task-dialog-title"
        aria-describedby="task-dialog-description"
      >
        <div className="task-dialog__header">
          <div>
            <span className="panel__eyebrow">{t("creator.eyebrow")}</span>
            <h2 id="task-dialog-title">{t("creator.title")}</h2>
            <p id="task-dialog-description">
              {t("creator.description")}
            </p>
          </div>
          <button
            className="icon-button"
            type="button"
            aria-label={t("common.close")}
            disabled={busy}
            onClick={onClose}
          >
            <Icon name="x" size={20} />
          </button>
        </div>

        <form
          className="task-form"
          onSubmit={(event) => {
            submit(event).catch((error: unknown) => {
              console.error("Failed to create job", error);
            });
          }}
        >
          <div className="form-section">
            <div className="form-section__heading">
              <span>01</span>
              <div>
                <strong>{t("creator.pathsTitle")}</strong>
                <small>
                  {backendMode === "tauri-ipc"
                    ? t("creator.pathsTauriDetail")
                    : t("creator.pathsMockDetail")}
                </small>
              </div>
            </div>
            <div className="form-grid">
              <label className="field field--full">
                <span>{t("creator.jobName")}</span>
                <input
                  ref={titleInputRef}
                  value={title}
                  maxLength={80}
                  required
                  onChange={(event) => setTitle(event.target.value)}
                />
              </label>
              <label className="field field--full">
                <span>{t("creator.mediaPath")}</span>
                <span className="field__with-icon">
                  <Icon name="file" size={17} />
                  <input
                    value={mediaPath}
                    placeholder={t("creator.mediaPlaceholder")}
                    required
                    spellCheck={false}
                    autoComplete="off"
                    aria-invalid={mediaPathError !== null}
                    aria-describedby="media-path-help"
                    onChange={(event) => {
                      mediaPathValueRef.current = event.target.value;
                      pathResolutionRef.current += 1;
                      setMediaPath(event.target.value);
                      setMediaPathError(null);
                    }}
                    onBlur={() => {
                      deriveOutputDirectory().catch((error: unknown) => {
                        console.error(
                          "Failed to derive the local output directory",
                          error,
                        );
                      });
                    }}
                  />
                </span>
                <small
                  id="media-path-help"
                  className={mediaPathError ? "field__error" : undefined}
                >
                  {mediaPathError ??
                    (resolvingMediaPath
                      ? t("common.checking")
                      : t("creator.mediaHint"))}
                </small>
              </label>
              <label className="field field--full">
                <span>{t("creator.outputDirectory")}</span>
                <span className="field__with-icon">
                  <Icon name="folder" size={17} />
                  <input
                    value={outputDirectory}
                    placeholder={t("creator.outputPlaceholder")}
                    required
                    spellCheck={false}
                    autoComplete="off"
                    aria-invalid={!outputPathDistinct}
                    aria-describedby="output-directory-help"
                    onChange={(event) => {
                      outputEditedRef.current = true;
                      pathResolutionRef.current += 1;
                      setOutputDirectory(event.target.value);
                    }}
                  />
                </span>
                <small
                  id="output-directory-help"
                  className={!outputPathDistinct ? "field__error" : undefined}
                >
                  {outputPathDistinct
                    ? t("creator.outputHint")
                    : t("creator.outputMatchesSource")}
                </small>
              </label>
            </div>
          </div>

          <div className="form-section">
            <div className="form-section__heading">
              <span>02</span>
              <div>
                <strong>{taskT("creator.speakerPolicy.title")}</strong>
                <small>
                  {taskT("creator.speakerPolicy.description")}
                </small>
              </div>
            </div>

            <fieldset className="speaker-policy-picker">
              <legend className="sr-only">
                {taskT("creator.speakerPolicy.legend")}
              </legend>
              {(
                [
                  [
                    "auto",
                    "creator.speakerPolicy.auto.label",
                    "creator.speakerPolicy.auto.description",
                  ],
                  [
                    "manual",
                    "creator.speakerPolicy.manual.label",
                    "creator.speakerPolicy.manual.description",
                  ],
                  [
                    "hybrid",
                    "creator.speakerPolicy.hybrid.label",
                    "creator.speakerPolicy.hybrid.description",
                  ],
                ] as const
              ).map(([mode, labelKey, descriptionKey]) => (
                <label
                  className={
                    speakerMode === mode
                      ? "speaker-policy-option speaker-policy-option--active"
                      : "speaker-policy-option"
                  }
                  key={mode}
                >
                  <input
                    type="radio"
                    name="speaker-count-policy"
                    value={mode}
                    checked={speakerMode === mode}
                    onChange={() => {
                      setSpeakerMode(mode);
                      const count =
                        mode === "manual"
                          ? manualSpeakerCount
                          : mode === "hybrid"
                            ? hybridLabelCount(
                                hybridMinCount,
                                hybridMaxCount,
                                hybridPriorCount,
                              )
                            : null;
                      setSpeakerLabels((current) =>
                        materializeLabels(
                          count,
                          current,
                          defaultSpeakerLabel,
                        ),
                      );
                    }}
                  />
                  <span>
                    <strong>{taskT(labelKey)}</strong>
                    <small>{taskT(descriptionKey)}</small>
                  </span>
                  <Icon name="check" size={16} />
                </label>
              ))}
            </fieldset>

            {previewPolicy ? (
              <SpeakerDetectionSummary
                detection={null}
                policy={previewPolicy}
                compact
              />
            ) : null}

            {speakerMode === "manual" ? (
              <div className="speaker-count-controls speaker-count-controls--single">
                <label className="field">
                  <span>{taskT("creator.speakerPolicy.exactCount")}</span>
                  <input
                    inputMode="numeric"
                    pattern="[1-9][0-9]*"
                    value={manualCount}
                    aria-invalid={manualSpeakerCount === null}
                    onChange={(event) => {
                      const { value } = event.target;
                      setManualCount(value);
                      setSpeakerLabels((current) =>
                        materializeLabels(
                          positiveInteger(value),
                          current,
                          defaultSpeakerLabel,
                        ),
                      );
                    }}
                  />
                  {manualSpeakerCount === null ? (
                    <small className="field__error">
                      {taskT("creator.validation.positiveSafeInteger")}
                    </small>
                  ) : null}
                </label>
              </div>
            ) : null}

            {speakerMode === "hybrid" ? (
              <div className="speaker-count-controls">
                <label className="field">
                  <span>{taskT("creator.speakerPolicy.minimum")}</span>
                  <input
                    inputMode="numeric"
                    pattern="[1-9][0-9]*"
                    value={hybridMin}
                    aria-invalid={hybridMinCount === null}
                    onChange={(event) => {
                      const { value } = event.target;
                      setHybridMin(value);
                      setSpeakerLabels((current) =>
                        materializeLabels(
                          hybridLabelCount(
                            positiveInteger(value),
                            hybridMaxCount,
                            hybridPriorCount,
                          ),
                          current,
                          defaultSpeakerLabel,
                        ),
                      );
                    }}
                  />
                </label>
                <label className="field">
                  <span>{taskT("creator.speakerPolicy.maximum")}</span>
                  <input
                    inputMode="numeric"
                    pattern="[1-9][0-9]*"
                    value={hybridMax}
                    aria-invalid={hybridMaxCount === null}
                    onChange={(event) => {
                      const { value } = event.target;
                      setHybridMax(value);
                      setSpeakerLabels((current) =>
                        materializeLabels(
                          hybridLabelCount(
                            hybridMinCount,
                            positiveInteger(value),
                            hybridPriorCount,
                          ),
                          current,
                          defaultSpeakerLabel,
                        ),
                      );
                    }}
                  />
                </label>
                <label className="field">
                  <span>{taskT("creator.speakerPolicy.prior")}</span>
                  <input
                    inputMode="numeric"
                    pattern="[1-9][0-9]*"
                    value={hybridPrior}
                    aria-invalid={hybridPriorCount === null || !hybridValid}
                    onChange={(event) => {
                      const { value } = event.target;
                      setHybridPrior(value);
                      setSpeakerLabels((current) =>
                        materializeLabels(
                          hybridLabelCount(
                            hybridMinCount,
                            hybridMaxCount,
                            positiveInteger(value),
                          ),
                          current,
                          defaultSpeakerLabel,
                        ),
                      );
                    }}
                  />
                </label>
                {!hybridValid ? (
                  <small className="field__error speaker-count-controls__error">
                    {taskT("creator.validation.hybridOrder")}
                  </small>
                ) : null}
              </div>
            ) : null}

            {shouldMaterializeLabels ? (
              <div className="speaker-label-editor">
                <div className="speaker-label-editor__header">
                  <span>
                    <strong>{taskT("creator.speaker.initialNames")}</strong>
                    <small>
                      {taskT("creator.speaker.profilesReady", {
                        count: configuredSpeakerCount,
                      })}
                    </small>
                  </span>
                  <span className="count-chip">{speakerLabels.length}</span>
                </div>
                <div className="speaker-form-grid">
                  {speakerLabels.map((label, index) => (
                    <label className="field" key={`new-speaker-${index + 1}`}>
                      <span>
                        {taskT("creator.speaker.defaultName", {
                          number: index + 1,
                        })}
                      </span>
                      <input
                        value={label}
                        maxLength={32}
                        required
                        aria-label={taskT(
                          "creator.speaker.initialNameAria",
                          { number: index + 1 },
                        )}
                        onChange={(event) =>
                          updateSpeakerLabel(index, event.target.value)
                        }
                      />
                    </label>
                  ))}
                </div>
              </div>
            ) : null}

            {configuredSpeakerCount !== null &&
            configuredSpeakerCount > MAX_INLINE_SPEAKER_EDITORS ? (
              <div className="speaker-policy-note" role="status">
                <Icon name="review" size={18} />
                <span>
                  <strong>{taskT("creator.speaker.largeModeTitle")}</strong>
                  <small>
                    {taskT("creator.speaker.largeModeDetail")}
                  </small>
                </span>
              </div>
            ) : null}
          </div>

          <div className="form-section">
            <div className="form-section__heading">
              <span>03</span>
              <div>
                <strong>{taskT("creator.languageSection.title")}</strong>
                <small>
                  {taskT("creator.languageSection.description")}
                </small>
              </div>
            </div>

            <div className="language-settings">
              <label className="field">
                <span>{taskT("creator.sourceLanguage.label")}</span>
                <select
                  value={sourceLanguageChoice}
                  aria-describedby="source-language-help"
                  aria-invalid={!sourceLanguageValid}
                  onChange={(event) =>
                    setSourceLanguageChoice(event.target.value)
                  }
                >
                  {LANGUAGE_PRESETS.map(([tag, labelKey]) => (
                    <option value={tag} key={tag}>
                      {taskT(labelKey)} ({tag})
                    </option>
                  ))}
                  <option value={CUSTOM_LANGUAGE_VALUE}>
                    {taskT("creator.language.customOption")}
                  </option>
                </select>
                <small id="source-language-help">
                  {taskT("creator.sourceLanguage.help")}
                </small>
              </label>
              {sourceLanguageChoice === CUSTOM_LANGUAGE_VALUE ? (
                <label className="field">
                  <span>{taskT("creator.sourceLanguage.customLabel")}</span>
                  <input
                    value={customSourceLanguage}
                    placeholder={taskT(
                      "creator.sourceLanguage.customPlaceholder",
                    )}
                    spellCheck={false}
                    autoComplete="off"
                    aria-invalid={!sourceLanguageValid}
                    onChange={(event) =>
                      setCustomSourceLanguage(event.target.value)
                    }
                  />
                  {!sourceLanguageValid ? (
                    <small className="field__error">
                      {taskT("creator.validation.sourceLanguageTag")}
                    </small>
                  ) : null}
                </label>
              ) : null}
            </div>

            <section
              className={
                businessEnabled
                  ? "business-processing business-processing--active"
                  : "business-processing"
              }
              aria-labelledby="business-processing-title"
            >
              <label className="business-master-switch">
                <input
                  type="checkbox"
                  role="switch"
                  checked={businessEnabled}
                  onChange={(event) =>
                    setBusinessEnabled(event.target.checked)
                  }
                />
                <span>
                  <strong id="business-processing-title">
                    {taskT("creator.business.enable")}
                  </strong>
                  <small>
                    {taskT("creator.business.enableDescription")}
                  </small>
                </span>
                <span className="switch-track" aria-hidden="true">
                  <span />
                </span>
              </label>

              <div
                className="business-options"
                aria-disabled={!businessEnabled}
              >
                <label className="business-option">
                  <input
                    type="checkbox"
                    checked={translationEnabled}
                    disabled={!businessEnabled}
                    onChange={(event) =>
                      setTranslationEnabled(event.target.checked)
                    }
                  />
                  <span>
                    <strong>
                      {taskT("creator.business.translationTitle")}
                    </strong>
                    <small>
                      {taskT("creator.business.translationDescription")}
                    </small>
                  </span>
                  <Icon name="check" size={16} />
                </label>
                <label className="business-option">
                  <input
                    type="checkbox"
                    checked={polishEnabled}
                    disabled={!businessEnabled}
                    onChange={(event) =>
                      setPolishEnabled(event.target.checked)
                    }
                  />
                  <span>
                    <strong>{taskT("creator.business.polishTitle")}</strong>
                    <small>
                      {taskT("creator.business.polishDescription")}
                    </small>
                  </span>
                  <Icon name="check" size={16} />
                </label>
                <label className="business-option">
                  <input
                    type="checkbox"
                    checked={summaryEnabled}
                    disabled={!businessEnabled}
                    onChange={(event) =>
                      setSummaryEnabled(event.target.checked)
                    }
                  />
                  <span>
                    <strong>{taskT("creator.business.summaryTitle")}</strong>
                    <small>
                      {taskT("creator.business.summaryDescription")}
                    </small>
                  </span>
                  <Icon name="check" size={16} />
                </label>
              </div>

              {businessEnabled && translationEnabled ? (
                <label className="field field--full translation-target-field">
                  <span>{taskT("creator.business.translationTargets")}</span>
                  <input
                    value={translationTargetsText}
                    placeholder={taskT(
                      "creator.business.translationTargetsPlaceholder",
                    )}
                    spellCheck={false}
                    autoComplete="off"
                    aria-invalid={!translationTargetsValid}
                    onChange={(event) =>
                      setTranslationTargetsText(event.target.value)
                    }
                  />
                  <small>
                    {taskT("creator.business.translationTargetsHelp")}
                  </small>
                  {translationTargets.length > 0 ? (
                    <span
                      className="language-chips"
                      aria-label={taskT(
                        "creator.business.parsedTargetsAria",
                      )}
                    >
                      {translationTargets.map((target, index) => (
                        <span key={`${target}-${index}`}>{target}</span>
                      ))}
                    </span>
                  ) : null}
                  {!translationTargetsValid ? (
                    <small className="field__error">
                      {taskT("creator.validation.translationTargets")}
                    </small>
                  ) : null}
                </label>
              ) : null}

              {businessEnabled ? (
                <>
                  <div className="language-settings language-settings--output">
                    <label className="field">
                      <span>{taskT("creator.business.outputLocale")}</span>
                      <select
                        value={outputLocaleChoice}
                        aria-invalid={!outputLocaleValid}
                        onChange={(event) =>
                          setOutputLocaleChoice(event.target.value)
                        }
                      >
                        {OUTPUT_LOCALE_PRESETS.map(([tag, labelKey]) => (
                          <option value={tag} key={tag}>
                            {taskT(labelKey)} ({tag})
                          </option>
                        ))}
                        <option value={CUSTOM_LANGUAGE_VALUE}>
                          {taskT("creator.language.customOption")}
                        </option>
                      </select>
                    </label>
                    {outputLocaleChoice === CUSTOM_LANGUAGE_VALUE ? (
                      <label className="field">
                        <span>
                          {taskT("creator.business.customOutputLocale")}
                        </span>
                        <input
                          value={customOutputLocale}
                          placeholder={taskT(
                            "creator.business.customOutputPlaceholder",
                          )}
                          spellCheck={false}
                          autoComplete="off"
                          aria-invalid={!outputLocaleValid}
                          onChange={(event) =>
                            setCustomOutputLocale(event.target.value)
                          }
                        />
                      </label>
                    ) : null}
                  </div>

                  <details className="runtime-details">
                    <summary>{taskT("creator.runtime.advanced")}</summary>
                    <div className="runtime-details__grid">
                      <label className="field">
                        <span>{taskT("creator.runtime.model")}</span>
                        <input
                          value={localLlmModel}
                          spellCheck={false}
                          autoComplete="off"
                          onChange={(event) =>
                            setLocalLlmModel(event.target.value)
                          }
                        />
                      </label>
                      <label className="field">
                        <span>{taskT("creator.runtime.endpoint")}</span>
                        <input
                          value={localLlmEndpoint}
                          spellCheck={false}
                          autoComplete="off"
                          aria-invalid={!localRuntimeValid}
                          onChange={(event) =>
                            setLocalLlmEndpoint(event.target.value)
                          }
                        />
                        <small>
                          {taskT("creator.runtime.endpointHelp")}
                        </small>
                      </label>
                    </div>
                  </details>
                </>
              ) : null}

              <div className="immutable-transcript-callout" role="note">
                <Icon name="shield" size={19} />
                <span>
                  <strong>{taskT("creator.business.immutableTitle")}</strong>
                  <small>
                    {taskT("creator.business.immutableDescription")}
                  </small>
                </span>
              </div>
            </section>
          </div>

          <div className="form-section">
            <div className="form-section__heading">
              <span>04</span>
              <div>
                <strong>{taskT("creator.strategy.title")}</strong>
                <small>{taskT("creator.strategy.description")}</small>
              </div>
            </div>
            <div className="compact-strategies">
              {strategies.map((strategy) => (
                <label
                  className={strategyId === strategy.id ? "compact-strategy compact-strategy--active" : "compact-strategy"}
                  key={strategy.id}
                >
                  <input
                    type="radio"
                    name="new-task-strategy"
                    value={strategy.id}
                    checked={strategyId === strategy.id}
                    onChange={() => setStrategyId(strategy.id)}
                  />
                  <span>
                    <strong>{strategy.label}</strong>
                    <small>{strategy.asrModel} · {strategy.estimatedVramGb.toFixed(1)} GB</small>
                  </span>
                  <Icon name="check" size={17} />
                </label>
              ))}
            </div>
          </div>

          <div className="offline-callout">
            <Icon name="cloud-off" size={20} />
            <div>
              <strong>{taskT("creator.offline.title")}</strong>
              <p>
                {backendMode === "tauri-ipc"
                  ? taskT("creator.offline.tauri")
                  : taskT("creator.offline.mock")}
              </p>
            </div>
          </div>

          <div className="task-dialog__actions">
            <button className="button button--soft" type="button" disabled={busy} onClick={onClose}>
              {t("common.cancel")}
            </button>
            <button className="button button--primary" type="submit" disabled={!canSubmit || busy}>
              <Icon name="sparkles" size={18} />
              {busy
                ? taskT("creator.actions.creating")
                : taskT("creator.actions.create")}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
