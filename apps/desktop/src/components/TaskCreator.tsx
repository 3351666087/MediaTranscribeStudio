import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type SyntheticEvent,
} from "react";
import {
  pathsAreDistinct,
  type MediaSelection,
} from "../bridge/media-drop";
import { planMediaIntake } from "../bridge/media-capabilities";
import {
  selectNativeMediaFiles,
  selectNativeOutputDirectory,
} from "../bridge/native-path-picker";
import {
  DEFAULT_OUTPUT_CUSTOMIZATION,
  cloneOutputCustomization,
  normalizeOutputCustomization,
  type OutputCustomization,
  type OutputCustomizationQuality,
} from "../contracts/output-customization";
import type {
  CreateJobRequest,
  CreateJobResult,
  ModelStrategy,
  ModelStrategyId,
  SpeakerCountPolicy,
  SpeakerProfile,
  SystemStatus,
} from "../contracts/studio";
import {
  mediaDropErrorMessageKey,
  useI18n,
  type MessageKey,
  type MessageParams,
} from "../i18n";
import { Icon } from "./Icon";
import { OutputCustomizationPanel } from "./OutputCustomizationPanel";
import { SpeakerDetectionSummary } from "./SpeakerDetectionSummary";

export interface InitialMediaSelection extends MediaSelection {
  sequence: number;
}

export interface InitialMediaBatch {
  sequence: number;
  selections: readonly MediaSelection[];
}

export type MediaQueueItemStatus =
  | "pending"
  | "creating"
  | "accepted"
  | "failed";

export interface MediaQueueItem {
  id: string;
  sourcePath: string;
  outputDirectory: string;
  outputEdited: boolean;
  mediaError: string | null;
  resolving: boolean;
  status: MediaQueueItemStatus;
  submitError: string | null;
}

export interface CreateJobBatchItemResult {
  index: number;
  request: CreateJobRequest;
  status: "accepted" | "failed";
  result?: CreateJobResult;
  error?: string;
}

export interface CreateJobBatchResult {
  items: CreateJobBatchItemResult[];
  acceptedCount: number;
  failedCount: number;
}

interface TaskCreatorProps {
  open: boolean;
  initialMediaBatch?: InitialMediaBatch | null;
  /** Compatibility input for callers that have not migrated to batches yet. */
  initialMediaSelection?: InitialMediaSelection | null;
  resolveMediaPath?: (path: string) => Promise<MediaSelection>;
  selectMediaFiles?: () => Promise<readonly string[]>;
  /** Compatibility input for single-file picker integrations. */
  selectMediaFile?: () => Promise<string | null>;
  selectOutputDirectory?: () => Promise<string | null>;
  speakers: SpeakerProfile[];
  initialSpeakerPolicy: SpeakerCountPolicy;
  strategies: ModelStrategy[];
  selectedStrategyId: ModelStrategyId;
  backendMode: SystemStatus["backendMode"];
  outputQuality?: OutputCustomizationQuality;
  busy: boolean;
  onClose: () => void;
  onCreate: (request: CreateJobRequest) => Promise<unknown>;
  onCreateBatch?: (
    requests: readonly CreateJobRequest[],
  ) => Promise<CreateJobBatchResult>;
}

export const MAX_INLINE_SPEAKER_EDITORS = 32;
const DEFAULT_LOCAL_MODEL = "qwen3.5:9b";
const DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:11434";
const DEFAULT_SOURCE_LANGUAGE = "auto";
const DEFAULT_OUTPUT_LOCALE = "en-US";
const MAX_MEDIA_QUEUE_ITEMS = 32;
const TASK_CREATOR_STEPS = [
  "media",
  "speakers",
  "language",
  "strategy",
  "output",
] as const;
type TaskCreatorStep = (typeof TASK_CREATOR_STEPS)[number];
const TASK_CREATOR_STEP_COPY = {
  media: {
    label: "creator.steps.media.label",
    description: "creator.steps.media.description",
  },
  speakers: {
    label: "creator.steps.speakers.label",
    description: "creator.steps.speakers.description",
  },
  language: {
    label: "creator.steps.language.label",
    description: "creator.steps.language.description",
  },
  strategy: {
    label: "creator.steps.strategy.label",
    description: "creator.steps.strategy.description",
  },
  output: {
    label: "creator.steps.output.label",
    description: "creator.steps.output.description",
  },
} as const;
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
const TRANSLATION_TARGET_PRESETS = OUTPUT_LOCALE_PRESETS;
function isListedLanguage(
  value: string,
  presets: ReadonlyArray<readonly [string, TaskCreatorMessageKey]>,
): boolean {
  return presets.some(([tag]) => tag === value);
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

function queueItemFromSelection(
  id: string,
  selection?: Partial<MediaSelection>,
  mediaError: string | null = null,
): MediaQueueItem {
  return {
    id,
    sourcePath: selection?.sourcePath ?? "",
    outputDirectory: selection?.outputDirectory ?? "",
    outputEdited: false,
    mediaError,
    resolving: false,
    status: mediaError === null ? "pending" : "failed",
    submitError: null,
  };
}

function comparableLocalPath(path: string): string {
  return path
    .trim()
    .replace(/\//gu, "\\")
    .replace(/\\+$/gu, "")
    .toLocaleLowerCase("en-US");
}

function deduplicateLocalPaths(paths: readonly string[]): string[] {
  const seen = new Set<string>();
  return paths.filter((path) => {
    const comparable = comparableLocalPath(path);
    if (seen.has(comparable)) {
      return false;
    }
    seen.add(comparable);
    return true;
  });
}

function deduplicateMediaSelections(
  selections: readonly MediaSelection[],
): MediaSelection[] {
  const seen = new Set<string>();
  return selections.filter((selection) => {
    const comparable = comparableLocalPath(selection.sourcePath);
    if (seen.has(comparable)) {
      return false;
    }
    seen.add(comparable);
    return true;
  });
}

export function TaskCreator({
  open,
  initialMediaBatch,
  initialMediaSelection,
  resolveMediaPath,
  selectMediaFiles,
  selectMediaFile,
  selectOutputDirectory,
  speakers,
  initialSpeakerPolicy,
  strategies,
  selectedStrategyId,
  backendMode,
  outputQuality,
  busy,
  onClose,
  onCreate,
  onCreateBatch,
}: TaskCreatorProps) {
  const { t } = useI18n();
  const taskT = (
    key: TaskCreatorMessageKey,
    params?: MessageParams,
  ): string => t(key, params);
  const defaultSpeakerLabel = (number: number): string =>
    taskT("creator.speaker.defaultName", { number });
  const languageLabel = (tag: string): string => {
    const preset = LANGUAGE_PRESETS.find(([presetTag]) => presetTag === tag);
    return preset ? taskT(preset[1]) : "";
  };
  const titleInputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const stepPanelRef = useRef<HTMLDivElement>(null);
  const stepButtonRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const busyRef = useRef(busy);
  const previousOpenRef = useRef(false);
  const appliedMediaBatchRef = useRef<number | null>(null);
  const queueSequenceRef = useRef(0);
  const pathResolutionRef = useRef(new Map<string, number>());
  const submissionLockRef = useRef(false);
  const [title, setTitle] = useState(() =>
    taskT("creator.defaultJobTitle"),
  );
  const [mediaQueue, setMediaQueue] = useState<MediaQueueItem[]>([]);
  const [pathPickerError, setPathPickerError] = useState<string | null>(null);
  const [activePathPicker, setActivePathPicker] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [activeStep, setActiveStep] =
    useState<TaskCreatorStep>("media");
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
  const [businessEnabled, setBusinessEnabled] = useState(false);
  const [translationEnabled, setTranslationEnabled] = useState(false);
  const [selectedTranslationTargets, setSelectedTranslationTargets] = useState<
    string[]
  >([]);
  const [summaryEnabled, setSummaryEnabled] = useState(false);
  const [outputLocaleChoice, setOutputLocaleChoice] = useState(
    DEFAULT_OUTPUT_LOCALE,
  );
  const [localLlmModel, setLocalLlmModel] = useState(DEFAULT_LOCAL_MODEL);
  const [localLlmEndpoint, setLocalLlmEndpoint] = useState(
    DEFAULT_LOCAL_ENDPOINT,
  );
  const [outputCustomization, setOutputCustomization] =
    useState<OutputCustomization>(() =>
      cloneOutputCustomization(DEFAULT_OUTPUT_CUSTOMIZATION),
    );
  const incomingMediaBatch = useMemo<InitialMediaBatch | null>(() => {
    if (initialMediaBatch) {
      return initialMediaBatch;
    }
    if (initialMediaSelection) {
      return {
        sequence: initialMediaSelection.sequence,
        selections: [initialMediaSelection],
      };
    }
    return null;
  }, [initialMediaBatch, initialMediaSelection]);
  const openNativeMediaPicker = useMemo(
    () =>
      selectMediaFiles ??
      (selectMediaFile
        ? async (): Promise<readonly string[]> => {
            const selected = await selectMediaFile();
            return selected === null ? [] : [selected];
          }
        : async () =>
            selectNativeMediaFiles({
              title: t("creator.mediaDialogTitle"),
            })),
    [selectMediaFile, selectMediaFiles, t],
  );
  const openNativeOutputPicker = useMemo(
    () =>
      selectOutputDirectory ??
      (async () =>
        selectNativeOutputDirectory({
          title: t("creator.outputDialogTitle"),
        })),
    [selectOutputDirectory, t],
  );

  const nextQueueId = useCallback((): string => {
    queueSequenceRef.current += 1;
    return `media-queue-${queueSequenceRef.current}`;
  }, []);

  const materializeInitialQueue = useCallback(
    (selections: readonly MediaSelection[]): MediaQueueItem[] => {
      if (selections.length === 0) {
        return [queueItemFromSelection(nextQueueId())];
      }
      return deduplicateMediaSelections(selections)
        .slice(0, MAX_MEDIA_QUEUE_ITEMS)
        .map((selection) =>
          queueItemFromSelection(nextQueueId(), selection),
        );
    },
    [nextQueueId],
  );

  useEffect(() => {
    busyRef.current = busy || submitting;
  }, [busy, submitting]);

  useEffect(() => {
    const opening = open && !previousOpenRef.current;
    previousOpenRef.current = open;
    if (!opening) {
      return;
    }

    setTitle(t("creator.defaultJobTitle"));
    setMediaQueue(
      materializeInitialQueue(incomingMediaBatch?.selections ?? []),
    );
    appliedMediaBatchRef.current = incomingMediaBatch?.sequence ?? null;
    pathResolutionRef.current.clear();
    setPathPickerError(null);
    setActivePathPicker(null);
    setSubmitting(false);
    setActiveStep("media");
    submissionLockRef.current = false;
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
    setBusinessEnabled(false);
    setTranslationEnabled(false);
    setSelectedTranslationTargets([]);
    setSummaryEnabled(false);
    setOutputLocaleChoice(DEFAULT_OUTPUT_LOCALE);
    setLocalLlmModel(DEFAULT_LOCAL_MODEL);
    setLocalLlmEndpoint(DEFAULT_LOCAL_ENDPOINT);
    setOutputCustomization(
      cloneOutputCustomization(DEFAULT_OUTPUT_CUSTOMIZATION),
    );
  }, [
    incomingMediaBatch,
    initialSpeakerPolicy,
    materializeInitialQueue,
    open,
    selectedStrategyId,
    speakers,
    t,
  ]);

  useEffect(() => {
    if (
      !open ||
      !incomingMediaBatch ||
      appliedMediaBatchRef.current === incomingMediaBatch.sequence
    ) {
      return;
    }

    appliedMediaBatchRef.current = incomingMediaBatch.sequence;
    setMediaQueue((current) => {
      const existingPaths = new Set(
        current
          .map((item) => comparableLocalPath(item.sourcePath))
          .filter((path) => path.length > 0),
      );
      const additions = incomingMediaBatch.selections
        .filter((selection) => {
          const comparable = comparableLocalPath(selection.sourcePath);
          if (existingPaths.has(comparable)) {
            return false;
          }
          existingPaths.add(comparable);
          return true;
        })
        .map((selection) =>
          queueItemFromSelection(nextQueueId(), selection),
        );

      if (additions.length === 0) {
        return current;
      }

      const untouchedBlank =
        current.length === 1 &&
        current[0].sourcePath.trim().length === 0 &&
        current[0].outputDirectory.trim().length === 0 &&
        !current[0].outputEdited &&
        current[0].status === "pending";
      const base = untouchedBlank ? [] : current;
      return [...base, ...additions].slice(0, MAX_MEDIA_QUEUE_ITEMS);
    });
    setPathPickerError(null);
  }, [incomingMediaBatch, nextQueueId, open]);

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
  const language = sourceLanguageChoice;
  const sourceLanguageValid = isListedLanguage(language, LANGUAGE_PRESETS);
  const translationTargets =
    businessEnabled && translationEnabled
      ? selectedTranslationTargets
      : [];
  const normalizedTranslationTargets = translationTargets.map((target) =>
    target.toLocaleLowerCase("en-US"),
  );
  const visibleTranslationTargets = translationTargets;
  const translationTargetsValid =
    !businessEnabled ||
    !translationEnabled ||
    (translationTargets.length > 0 &&
      translationTargets.every((target) =>
        isListedLanguage(target, TRANSLATION_TARGET_PRESETS),
      ) &&
      new Set(normalizedTranslationTargets).size === translationTargets.length);
  const outputLocale = outputLocaleChoice;
  const outputLocaleValid =
    !businessEnabled ||
    isListedLanguage(outputLocale, OUTPUT_LOCALE_PRESETS);
  const localRuntimeValid =
    !businessEnabled ||
    (localLlmModel.trim().length > 0 &&
      /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:\/.*)?$/u.test(
        localLlmEndpoint.trim(),
      ));
  const actionableMediaItems = mediaQueue.filter(
    (item) => item.status !== "accepted",
  );
  const submittableMediaItems = actionableMediaItems.filter(
    (item) => item.mediaError === null,
  );
  const outputDirectoryCounts = new Map<string, number>();
  submittableMediaItems.forEach((item) => {
    const comparable = comparableLocalPath(item.outputDirectory);
    if (comparable.length > 0) {
      outputDirectoryCounts.set(
        comparable,
        (outputDirectoryCounts.get(comparable) ?? 0) + 1,
      );
    }
  });
  const mediaQueueValid =
    submittableMediaItems.length > 0 &&
    submittableMediaItems.every((item) => {
      const comparableOutput = comparableLocalPath(item.outputDirectory);
      return (
        item.sourcePath.trim().length > 0 &&
        item.outputDirectory.trim().length > 0 &&
        item.mediaError === null &&
        !item.resolving &&
        pathsAreDistinct(item.sourcePath, item.outputDirectory) &&
        (outputDirectoryCounts.get(comparableOutput) ?? 0) === 1
      );
    });

  const updateQueueItem = (
    itemId: string,
    update:
      | Partial<MediaQueueItem>
      | ((item: MediaQueueItem) => MediaQueueItem),
  ) => {
    setMediaQueue((current) =>
      current.map((item) => {
        if (item.id !== itemId || item.status === "accepted") {
          return item;
        }
        return typeof update === "function"
          ? update(item)
          : { ...item, ...update };
      }),
    );
  };

  const resolveAndApplyMediaPath = async (
    itemId: string,
    rawPath: string,
  ) => {
    const candidate = rawPath.trim();
    if (!resolveMediaPath || candidate.length === 0) {
      return;
    }

    const operation =
      (pathResolutionRef.current.get(itemId) ?? 0) + 1;
    pathResolutionRef.current.set(itemId, operation);
    updateQueueItem(itemId, {
      resolving: true,
      mediaError: null,
      submitError: null,
      status: "pending",
    });
    setPathPickerError(null);

    try {
      const selection = await resolveMediaPath(candidate);
      if (
        pathResolutionRef.current.get(itemId) !== operation
      ) {
        return;
      }
      updateQueueItem(itemId, (item) => {
        if (item.sourcePath.trim() !== candidate) {
          return item;
        }
        return {
          ...item,
          sourcePath: selection.sourcePath,
          outputDirectory: item.outputEdited
            ? item.outputDirectory
            : selection.outputDirectory,
          mediaError: null,
          resolving: false,
          status: "pending",
        };
      });
    } catch (error: unknown) {
      if (pathResolutionRef.current.get(itemId) === operation) {
        updateQueueItem(itemId, {
          resolving: false,
          mediaError: t(mediaDropErrorMessageKey(error)),
          status: "failed",
        });
      }
    } finally {
      if (pathResolutionRef.current.get(itemId) === operation) {
        updateQueueItem(itemId, { resolving: false });
      }
    }
  };

  const deriveOutputDirectory = async (item: MediaQueueItem) => {
    await resolveAndApplyMediaPath(item.id, item.sourcePath);
  };

  const chooseMedia = async () => {
    if (activePathPicker !== null) {
      return;
    }
    setActivePathPicker("media");
    setPathPickerError(null);
    try {
      const selectedPaths = deduplicateLocalPaths(
        await openNativeMediaPicker(),
      ).slice(0, MAX_MEDIA_QUEUE_ITEMS);
      if (selectedPaths.length === 0) {
        return;
      }

      const resolved = await Promise.all(
        selectedPaths.map(async (selectedPath) => {
          if (!resolveMediaPath) {
            return {
              selection: {
                sourcePath: selectedPath,
                outputDirectory: "",
              },
              error: null,
            };
          }
          try {
            return {
              selection: await resolveMediaPath(selectedPath),
              error: null,
            };
          } catch (error: unknown) {
            return {
              selection: {
                sourcePath: selectedPath,
                outputDirectory: "",
              },
              error: t(mediaDropErrorMessageKey(error)),
            };
          }
        }),
      );

      setMediaQueue((current) => {
        const existingPaths = new Set(
          current
            .map((item) => comparableLocalPath(item.sourcePath))
            .filter((path) => path.length > 0),
        );
        const additions = resolved
          .filter(({ selection }) => {
            const comparable = comparableLocalPath(selection.sourcePath);
            if (existingPaths.has(comparable)) {
              return false;
            }
            existingPaths.add(comparable);
            return true;
          })
          .map(({ selection, error }) =>
            queueItemFromSelection(
              nextQueueId(),
              selection,
              error,
            ),
          );
        const untouchedBlank =
          current.length === 1 &&
          current[0].sourcePath.trim().length === 0 &&
          current[0].outputDirectory.trim().length === 0 &&
          !current[0].outputEdited &&
          current[0].status === "pending";
        const base = untouchedBlank ? [] : current;
        return [...base, ...additions].slice(0, MAX_MEDIA_QUEUE_ITEMS);
      });
    } catch {
      setPathPickerError(taskT("creator.pathPickerError"));
    } finally {
      setActivePathPicker(null);
    }
  };

  const chooseOutputDirectory = async (itemId: string) => {
    if (activePathPicker !== null) {
      return;
    }
    setActivePathPicker(`output:${itemId}`);
    setPathPickerError(null);
    try {
      const selectedPath = await openNativeOutputPicker();
      if (selectedPath === null) {
        return;
      }
      pathResolutionRef.current.set(
        itemId,
        (pathResolutionRef.current.get(itemId) ?? 0) + 1,
      );
      updateQueueItem(itemId, {
        outputDirectory: selectedPath,
        outputEdited: true,
        submitError: null,
      });
    } catch {
      setPathPickerError(taskT("creator.pathPickerError"));
    } finally {
      setActivePathPicker(null);
    }
  };

  const removeMediaItem = (itemId: string) => {
    pathResolutionRef.current.set(
      itemId,
      (pathResolutionRef.current.get(itemId) ?? 0) + 1,
    );
    setMediaQueue((current) => {
      const remaining = current.filter(
        (item) => item.id !== itemId || item.status === "accepted",
      );
      return remaining.length > 0
        ? remaining
        : [queueItemFromSelection(nextQueueId())];
    });
  };

  const activeStepIndex = TASK_CREATOR_STEPS.indexOf(activeStep);
  const moveToStep = (
    step: TaskCreatorStep,
    focusPanel = false,
  ) => {
    setActiveStep(step);
    if (focusPanel) {
      window.requestAnimationFrame(() => stepPanelRef.current?.focus());
    }
  };
  const moveByStep = (offset: -1 | 1) => {
    const nextIndex = Math.min(
      TASK_CREATOR_STEPS.length - 1,
      Math.max(0, activeStepIndex + offset),
    );
    moveToStep(TASK_CREATOR_STEPS[nextIndex], true);
  };
  const handleStepNavigationKeyDown = (
    event: ReactKeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = (index + 1) % TASK_CREATOR_STEPS.length;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex =
        (index - 1 + TASK_CREATOR_STEPS.length) %
        TASK_CREATOR_STEPS.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = TASK_CREATOR_STEPS.length - 1;
    }
    if (nextIndex === null) {
      return;
    }
    event.preventDefault();
    const nextStep = TASK_CREATOR_STEPS[nextIndex];
    moveToStep(nextStep);
    stepButtonRefs.current[nextIndex]?.focus();
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
    mediaQueueValid &&
    countPolicyValid &&
    speakerLabelsValid &&
    sourceLanguageValid &&
    translationTargetsValid &&
    outputLocaleValid &&
    localRuntimeValid;
  const stepCompletion: Record<TaskCreatorStep, boolean> = {
    media: title.trim().length > 0 && mediaQueueValid,
    speakers: countPolicyValid && speakerLabelsValid,
    language:
      sourceLanguageValid &&
      translationTargetsValid &&
      outputLocaleValid &&
      localRuntimeValid,
    strategy: strategies.some((strategy) => strategy.id === strategyId),
    output: true,
  };
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
    if (
      !canSubmit ||
      busy ||
      submitting ||
      submissionLockRef.current
    ) {
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
    submissionLockRef.current = true;
    setSubmitting(true);
    const submittedItems = mediaQueue.filter(
      (item) => item.status !== "accepted" && item.mediaError === null,
    );
    const hasIsolatedIntakeFailures = mediaQueue.some(
      (item) => item.status !== "accepted" && item.mediaError !== null,
    );
    const requests = submittedItems.map(
      (item, index): CreateJobRequest => ({
        title:
          submittedItems.length === 1
            ? title.trim()
            : `${title.trim()} — ${index + 1}`,
        mediaPath: item.sourcePath.trim(),
        outputDirectory: item.outputDirectory.trim(),
        strategyId,
        speakerPolicy,
        speakerLabels:
          speakerMode === "auto" || !shouldMaterializeLabels
            ? []
            : speakerLabels.map((label) => label.trim()),
        language,
        localLlmMode:
          businessEnabled &&
          (translationTargets.length > 0 || summaryEnabled)
            ? "business"
            : "disabled",
        localLlmModel: localLlmModel.trim(),
        localLlmEndpoint: localLlmEndpoint.trim(),
        localLlmEndpointPolicy: "loopback-only",
        localLlmAutoApply: false,
        translationTargets,
        summary: businessEnabled && summaryEnabled,
        outputLocale,
        businessPromptVersion: "business-v1",
        outputCustomization: cloneOutputCustomization(
          normalizeOutputCustomization(outputCustomization),
        ),
      }),
    );

    setMediaQueue((current) =>
      current.map((item) =>
        submittedItems.some((submitted) => submitted.id === item.id)
          ? {
              ...item,
              status: "creating",
              submitError: null,
            }
          : item,
      ),
    );

    try {
      let batchResult: CreateJobBatchResult;
      if (onCreateBatch) {
        batchResult = await onCreateBatch(requests);
      } else {
        const items: CreateJobBatchItemResult[] = [];
        for (const [index, request] of requests.entries()) {
          try {
            const rawResult = await onCreate(request);
            const result =
              rawResult &&
              typeof rawResult === "object" &&
              "accepted" in rawResult
                ? (rawResult as CreateJobResult)
                : undefined;
            const accepted = result?.accepted !== false;
            items.push(
              accepted
                ? {
                    index,
                    request,
                    status: "accepted",
                    result,
                  }
                : {
                    index,
                    request,
                    status: "failed",
                    result,
                    error:
                      result.message.trim().length > 0
                        ? result.message
                        : taskT("creator.batch.partialFailure"),
                  },
            );
          } catch (error: unknown) {
            items.push({
              index,
              request,
              status: "failed",
              error:
                error instanceof Error && error.message.trim().length > 0
                  ? error.message
                  : taskT("creator.batch.partialFailure"),
            });
          }
        }
        const acceptedCount = items.filter(
          (item) => item.status === "accepted",
        ).length;
        batchResult = {
          items,
          acceptedCount,
          failedCount: items.length - acceptedCount,
        };
      }

      setMediaQueue((current) =>
        current.map((item) => {
          const submittedIndex = submittedItems.findIndex(
            (submitted) => submitted.id === item.id,
          );
          if (submittedIndex < 0) {
            return item;
          }
          const result = batchResult.items.find(
            (candidate) => candidate.index === submittedIndex,
          );
          if (result?.status === "accepted") {
            return {
              ...item,
              status: "accepted",
              submitError: null,
            };
          }
          return {
            ...item,
            status: "failed",
            submitError:
              result?.error ?? taskT("creator.batch.partialFailure"),
          };
        }),
      );

      if (
        batchResult.failedCount === 0 &&
        batchResult.acceptedCount === submittedItems.length &&
        !hasIsolatedIntakeFailures
      ) {
        onClose();
      }
    } finally {
      submissionLockRef.current = false;
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-layer">
      <button
        className="modal-backdrop"
        type="button"
        aria-label={t("common.close")}
        onClick={() => {
          if (!busy && !submitting) {
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
            disabled={busy || submitting}
            onClick={onClose}
          >
            <Icon name="x" size={20} />
          </button>
        </div>

        <nav
          className="task-steps"
          aria-label={taskT("creator.steps.label")}
        >
          <div className="task-steps__rail" role="tablist">
            {TASK_CREATOR_STEPS.map((step, index) => {
              const copy = TASK_CREATOR_STEP_COPY[step];
              const selected = step === activeStep;
              return (
                <button
                  className="task-step-tab"
                  type="button"
                  role="tab"
                  id={`task-step-tab-${step}`}
                  aria-controls="task-step-panel"
                  aria-selected={selected}
                  tabIndex={selected ? 0 : -1}
                  data-state={
                    selected
                      ? "current"
                      : stepCompletion[step]
                        ? "complete"
                        : "upcoming"
                  }
                  key={step}
                  ref={(element) => {
                    stepButtonRefs.current[index] = element;
                  }}
                  onClick={() => moveToStep(step)}
                  onKeyDown={(event) =>
                    handleStepNavigationKeyDown(event, index)
                  }
                >
                  <span className="task-step-tab__number" aria-hidden="true">
                    {stepCompletion[step] && !selected ? (
                      <Icon name="check" size={14} />
                    ) : (
                      String(index + 1).padStart(2, "0")
                    )}
                  </span>
                  <span className="task-step-tab__copy">
                    <strong>{taskT(copy.label)}</strong>
                    <small>{taskT(copy.description)}</small>
                  </span>
                </button>
              );
            })}
          </div>
        </nav>

        <form
          className="task-form"
          onSubmit={(event) => {
            submit(event).catch((error: unknown) => {
              console.error("Failed to create job", error);
            });
          }}
        >
          <fieldset
            className="task-form__controls"
            disabled={busy || submitting}
          >
          <div
            className="task-form__viewport"
            ref={stepPanelRef}
            role="tabpanel"
            id="task-step-panel"
            aria-labelledby={`task-step-tab-${activeStep}`}
            tabIndex={-1}
          >
          {activeStep === "media" ? (
          <section className="task-step-panel task-step-panel--media">
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
              <section
                className="media-batch-picker field--full"
                aria-labelledby="creator-media-batch-title"
              >
                <div className="media-batch-picker__invitation">
                  <span
                    className="media-batch-picker__orb"
                    aria-hidden="true"
                  >
                    <Icon name="upload" size={24} />
                  </span>
                  <div className="media-batch-picker__copy">
                    <strong id="creator-media-batch-title">
                      {taskT("creator.batch.dropTitle")}
                    </strong>
                    <small>{taskT("creator.batch.dropDetail")}</small>
                    <small className="media-batch-picker__native-hint">
                      {taskT("creator.batch.nativePickerHint")}
                    </small>
                  </div>
                  <button
                    className="button button--primary media-batch-picker__button"
                    type="button"
                    disabled={
                      busy ||
                      submitting ||
                      activePathPicker !== null ||
                      mediaQueue.length >= MAX_MEDIA_QUEUE_ITEMS
                    }
                    onClick={() => {
                      chooseMedia().catch((error: unknown) => {
                        console.error(
                          "Failed to open the native media picker",
                          error,
                        );
                      });
                    }}
                  >
                    <Icon name="folder" size={17} />
                    {activePathPicker === "media"
                      ? t("common.checking")
                      : taskT("creator.batch.addMedia")}
                  </button>
                </div>

                <div className="media-batch-picker__summary" role="status">
                  <span>
                    {taskT("creator.batch.selectedCount", {
                      count: mediaQueue.filter(
                        (item) => item.sourcePath.trim().length > 0,
                      ).length,
                    })}
                  </span>
                  <small>{taskT("creator.batch.appendPromise")}</small>
                </div>

                <div
                  className="media-queue"
                  role="list"
                  aria-label={taskT("creator.batch.queueLabel")}
                >
                  {mediaQueue.map((item, index) => {
                    const mediaInputId =
                      index === 0
                        ? "creator-media-path"
                        : `creator-media-path-${item.id}`;
                    const outputInputId =
                      index === 0
                        ? "creator-output-directory"
                        : `creator-output-directory-${item.id}`;
                    const mediaHelpId = `${mediaInputId}-help`;
                    const outputHelpId = `${outputInputId}-help`;
                    const mediaIntakePlan =
                      item.sourcePath.trim().length > 0
                        ? planMediaIntake(item.sourcePath)
                        : null;
                    const idleMediaHint =
                      mediaIntakePlan?.extensionHint.status === "recognized"
                        ? taskT("creator.batch.extensionHint", {
                            extension: `.${mediaIntakePlan.extensionHint.extension.toLocaleUpperCase(
                              "en-US",
                            )}`,
                          })
                        : mediaIntakePlan
                          ? taskT("creator.batch.contentProbeRequired")
                          : t("creator.mediaHint");
                    const outputPathDistinct = pathsAreDistinct(
                      item.sourcePath,
                      item.outputDirectory,
                    );
                    const comparableOutput = comparableLocalPath(
                      item.outputDirectory,
                    );
                    const outputCollision =
                      comparableOutput.length > 0 &&
                      (outputDirectoryCounts.get(comparableOutput) ?? 0) > 1;
                    const statusLabel =
                      item.status === "accepted"
                        ? taskT("creator.batch.accepted")
                        : item.status === "creating"
                          ? taskT("creator.batch.creating")
                          : item.status === "failed"
                            ? taskT("creator.batch.failed")
                            : taskT("creator.batch.pending");
                    const itemLocked =
                      busy || submitting || item.status === "accepted";

                    return (
                      <article
                        className={`media-queue__item media-queue__item--${item.status}`}
                        data-output-origin={
                          item.outputEdited ? "user" : "suggested"
                        }
                        key={item.id}
                        role="listitem"
                      >
                        <header className="media-queue__header">
                          <span className="media-queue__number">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          <span
                            className={`media-queue__status media-queue__status--${item.status}`}
                          >
                            {statusLabel}
                          </span>
                          <button
                            className="icon-button media-queue__remove"
                            type="button"
                            aria-label={taskT("creator.batch.remove", {
                              number: index + 1,
                            })}
                            disabled={itemLocked}
                            onClick={() => removeMediaItem(item.id)}
                          >
                            <Icon name="x" size={16} />
                          </button>
                        </header>

                        <div className="media-queue__fields">
                          <div className="field field--full">
                            <label htmlFor={mediaInputId}>
                              {t("creator.mediaPath")}
                            </label>
                            <span className="field__with-icon">
                              <Icon name="file" size={17} />
                              <input
                                id={mediaInputId}
                                value={item.sourcePath}
                                placeholder={t("creator.mediaPlaceholder")}
                                required
                                disabled={itemLocked}
                                spellCheck={false}
                                autoComplete="off"
                                aria-invalid={item.mediaError !== null}
                                aria-describedby={mediaHelpId}
                                onChange={(event) => {
                                  pathResolutionRef.current.set(
                                    item.id,
                                    (pathResolutionRef.current.get(item.id) ??
                                      0) + 1,
                                  );
                                  updateQueueItem(item.id, (currentItem) => ({
                                    ...currentItem,
                                    sourcePath: event.target.value,
                                    outputDirectory:
                                      currentItem.outputEdited
                                        ? currentItem.outputDirectory
                                        : "",
                                    mediaError: null,
                                    submitError: null,
                                    status: "pending",
                                    resolving: false,
                                  }));
                                  setPathPickerError(null);
                                }}
                                onBlur={() => {
                                  deriveOutputDirectory(item).catch(
                                    (error: unknown) => {
                                      console.error(
                                        "Failed to derive the local output directory",
                                        error,
                                      );
                                    },
                                  );
                                }}
                              />
                            </span>
                            <small
                              id={mediaHelpId}
                              className={
                                item.mediaError
                                  ? "field__error"
                                  : undefined
                              }
                            >
                              {item.mediaError ??
                                (item.resolving
                                  ? t("common.checking")
                                  : idleMediaHint)}
                            </small>
                          </div>

                          <div className="field field--full">
                            <label htmlFor={outputInputId}>
                              {t("creator.outputDirectory")}
                            </label>
                            <span className="field__with-icon">
                              <Icon name="folder" size={17} />
                              <input
                                id={outputInputId}
                                value={item.outputDirectory}
                                placeholder={t("creator.outputPlaceholder")}
                                required
                                disabled={itemLocked}
                                spellCheck={false}
                                autoComplete="off"
                                aria-invalid={
                                  !outputPathDistinct || outputCollision
                                }
                                aria-describedby={outputHelpId}
                                onChange={(event) => {
                                  pathResolutionRef.current.set(
                                    item.id,
                                    (pathResolutionRef.current.get(item.id) ??
                                      0) + 1,
                                  );
                                  updateQueueItem(item.id, {
                                    outputDirectory: event.target.value,
                                    outputEdited: true,
                                    submitError: null,
                                    status: "pending",
                                  });
                                  setPathPickerError(null);
                                }}
                              />
                              <button
                                className="button button--soft field__path-picker"
                                type="button"
                                disabled={
                                  itemLocked || activePathPicker !== null
                                }
                                onClick={() => {
                                  chooseOutputDirectory(item.id).catch(
                                    (error: unknown) => {
                                      console.error(
                                        "Failed to open the native output folder picker",
                                        error,
                                      );
                                    },
                                  );
                                }}
                              >
                                <Icon name="folder" size={16} />
                                {activePathPicker === `output:${item.id}`
                                  ? t("common.checking")
                                  : taskT("creator.outputBrowse")}
                              </button>
                            </span>
                            <small
                              id={outputHelpId}
                              className={
                                !outputPathDistinct || outputCollision
                                  ? "field__error"
                                  : undefined
                              }
                            >
                              {!outputPathDistinct
                                ? t("creator.outputMatchesSource")
                                : outputCollision
                                  ? taskT(
                                      "creator.batch.outputCollision",
                                    )
                                  : t("creator.outputHint")}
                            </small>
                          </div>
                        </div>

                        {item.submitError ? (
                          <div
                            className="media-queue__error"
                            role="alert"
                          >
                            <Icon name="alert" size={16} />
                            <span>{item.submitError}</span>
                          </div>
                        ) : null}
                      </article>
                    );
                  })}
                </div>

                {pathPickerError ? (
                  <small className="field__error" role="alert">
                    {pathPickerError}
                  </small>
                ) : null}
              </section>
            </div>
          </div>
          </section>
          ) : null}

          {activeStep === "speakers" ? (
          <section className="task-step-panel task-step-panel--speakers">
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
          </section>
          ) : null}

          {activeStep === "language" ? (
          <section className="task-step-panel task-step-panel--language">
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
                  onChange={(event) => {
                    setSourceLanguageChoice(event.target.value);
                  }}
                >
                  {LANGUAGE_PRESETS.map(([tag, labelKey]) => (
                    <option value={tag} key={tag}>
                      {taskT(labelKey)}
                    </option>
                  ))}
                </select>
                <small id="source-language-help">
                  {taskT("creator.sourceLanguage.help")}
                </small>
              </label>
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
                  <select
                    value=""
                    aria-invalid={!translationTargetsValid}
                    onChange={(event) => {
                      const target = event.target.value;
                      if (
                        target.length > 0 &&
                        !normalizedTranslationTargets.includes(
                          target.toLocaleLowerCase("en-US"),
                        )
                      ) {
                        setSelectedTranslationTargets((current) => [
                          ...current,
                          target,
                        ]);
                      }
                    }}
                  >
                    <option value="">
                      {taskT("creator.business.chooseTranslationTarget")}
                    </option>
                    {TRANSLATION_TARGET_PRESETS.map(([tag, labelKey]) => (
                      <option
                        value={tag}
                        key={tag}
                        disabled={normalizedTranslationTargets.includes(
                          tag.toLocaleLowerCase("en-US"),
                        )}
                      >
                        {taskT(labelKey)}
                      </option>
                    ))}
                  </select>
                  <small>
                    {taskT("creator.business.translationTargetsHelp")}
                  </small>
                  {visibleTranslationTargets.length > 0 ? (
                    <span
                      className="language-chips"
                      aria-label={taskT(
                        "creator.business.parsedTargetsAria",
                      )}
                    >
                      {visibleTranslationTargets.map((target) => (
                        <span key={target.toLocaleLowerCase("en-US")}>
                          {languageLabel(target)}
                          <button
                            type="button"
                            aria-label={taskT(
                              "creator.business.removeTranslationTarget",
                              { language: languageLabel(target) },
                            )}
                            onClick={() => {
                              const normalizedTarget =
                                target.toLocaleLowerCase("en-US");
                              setSelectedTranslationTargets((current) =>
                                current.filter(
                                  (item) =>
                                    item.toLocaleLowerCase("en-US") !==
                                    normalizedTarget,
                                ),
                              );
                            }}
                          >
                            <Icon name="x" size={13} />
                          </button>
                        </span>
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
                        onChange={(event) => {
                          setOutputLocaleChoice(event.target.value);
                        }}
                        >
                          {OUTPUT_LOCALE_PRESETS.map(([tag, labelKey]) => (
                            <option value={tag} key={tag}>
                              {taskT(labelKey)}
                            </option>
                          ))}
                        </select>
                      </label>
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
          </section>
          ) : null}

          {activeStep === "strategy" ? (
          <section className="task-step-panel task-step-panel--strategy">
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
                    <small>
                      {strategy.asrModel} ·{" "}
                      {strategy.estimatedVramGb.toFixed(1)} GB
                    </small>
                  </span>
                  <Icon name="check" size={17} />
                </label>
              ))}
            </div>
          </div>
          </section>
          ) : null}

          {activeStep === "output" ? (
          <section className="task-step-panel task-step-panel--output">
          <OutputCustomizationPanel
            className="task-output-diy"
            value={outputCustomization}
            speakerCount={configuredSpeakerCount ?? speakers.length}
            speakerLabels={
              speakerMode === "auto"
                ? speakers.map((speaker) => speaker.label)
                : speakerLabels
            }
            quality={outputQuality}
            disabled={busy || submitting}
            showActions={false}
            onChange={(nextValue) => {
              setOutputCustomization(
                cloneOutputCustomization(nextValue),
              );
            }}
          />

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
          </section>
          ) : null}
          </div>

          </fieldset>
          <div className="task-dialog__actions">
            <button
              className="button button--soft"
              type="button"
              disabled={busy || submitting}
              onClick={onClose}
            >
              {t("common.cancel")}
            </button>
            <div className="task-dialog__actions-primary">
              {activeStepIndex > 0 ? (
                <button
                  className="button button--soft"
                  type="button"
                  disabled={busy || submitting}
                  onClick={() => moveByStep(-1)}
                >
                  {taskT("creator.actions.back")}
                </button>
              ) : null}
              {activeStepIndex < TASK_CREATOR_STEPS.length - 1 ? (
                <button
                  className="button button--primary"
                  type="button"
                  disabled={busy || submitting}
                  onClick={() => moveByStep(1)}
                >
                  {taskT("creator.actions.continue")}
                  <span aria-hidden="true">→</span>
                </button>
              ) : (
                <button
                  className="button button--primary"
                  type="submit"
                  disabled={!canSubmit || busy || submitting}
                >
                  <Icon name="sparkles" size={18} />
                  {busy || submitting
                    ? taskT("creator.actions.creating")
                    : taskT("creator.actions.create")}
                </button>
              )}
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
