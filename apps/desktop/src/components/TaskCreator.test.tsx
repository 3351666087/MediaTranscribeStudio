import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import {
  MediaDropError,
  type MediaSelection,
} from "../bridge/media-drop";
import type {
  CreateJobRequest,
  SpeakerCountPolicy,
  SpeakerProfile,
} from "../contracts/studio";
import {
  ENGLISH_MESSAGES,
  LOCALE_OPTIONS,
  MESSAGE_CATALOGS,
  type I18nContextValue,
  type MessageParams,
} from "../i18n";
import { I18nContext } from "../i18n/context";
import taskCreatorMessages from "../i18n/fragments/task-creator.json";
import { modelStrategies } from "../mocks/studio-fixture";
import { MAX_INLINE_SPEAKER_EDITORS, TaskCreator } from "./TaskCreator";

const dynamicCounts = [1, 2, 5, 8, 13] as const;
const taskCreatorLocales = [
  "en",
  "zh-Hans",
  "zh-Hant",
  "ja",
  "ko",
  "es",
  "fr",
  "de",
  "pt-BR",
] as const;
const placeholderPattern = /\{([a-zA-Z][a-zA-Z0-9]*)\}/gu;
const finiteFormatListPattern =
  /MOV.*MP4.*(?:M4V|MKV)|WAV.*MP3.*(?:M4A|FLAC)/iu;
const multiFileCopyPatterns = {
  en: [/one or more/iu, /one or more|multi-file/iu, /one or more/iu],
  "zh-Hans": [/一个或多个/u, /一个或多个|多文件/u, /一个或多个/u],
  "zh-Hant": [/一個或多個/u, /一個或多個|多檔/u, /一個或多個/u],
  ja: [/1件以上/u, /1件以上|複数/u, /1つ以上/u],
  ko: [/하나 이상의/u, /하나 이상의|다중/u, /하나 이상의/u],
  es: [/uno o varios/iu, /uno o varios|selección múltiple/iu, /uno o varios/iu],
  fr: [
    /un ou plusieurs/iu,
    /un ou plusieurs|sélection multiple/iu,
    /un ou plusieurs/iu,
  ],
  de: [
    /eine oder mehrere/iu,
    /eine oder mehrere|Mehrfachauswahl/iu,
    /eine oder mehrere/iu,
  ],
  "pt-BR": [
    /um ou mais/iu,
    /um ou mais|seleção múltipla/iu,
    /uma ou mais/iu,
  ],
} as const;
const testMessages: Readonly<Record<string, string>> = {
  ...ENGLISH_MESSAGES,
  ...taskCreatorMessages.en,
  "workbench.speakerDetection.manual.title":
    "Manual count locked at {count}",
  "workbench.speakerDetection.manual.detail":
    "{count} speaker profiles will be prepared when the job starts.",
  "workbench.speakerDetection.waiting.title":
    "Waiting for local speaker-count detection",
  "workbench.speakerDetection.waiting.detail":
    "The estimated count, confidence, and candidate distribution will appear after the job starts.",
};
const testI18n: I18nContextValue = {
  locale: "en",
  localeOptions: LOCALE_OPTIONS,
  setLocale: () => undefined,
  t: (key, params: MessageParams = {}) => {
    const template = testMessages[key];
    if (typeof template !== "string" || template.trim().length === 0) {
      throw new Error(`Missing TaskCreator test message: ${key}`);
    }
    return template.replace(
      /\{([a-zA-Z][a-zA-Z0-9]*)\}/gu,
      (match, name: string) => {
        const value = params[name];
        return value === undefined ? match : String(value);
      },
    );
  },
};

function TaskCreatorI18n({ children }: { children: ReactNode }) {
  return (
    <I18nContext.Provider value={testI18n}>{children}</I18nContext.Provider>
  );
}

function renderTaskCreator(ui: ReactNode) {
  return render(ui, { wrapper: TaskCreatorI18n });
}

function createSpeakers(count: number): SpeakerProfile[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `speaker-${index + 1}`,
    label: `Previous job participant ${index + 1}`,
    shortLabel: `S${index + 1}`,
    color: `hsl(${(index * 137 + 252) % 360} 52% 49%)`,
    roleHint: "Previous job role",
    sampleStatus: "missing",
    locked: false,
    reviewStatus: "pending",
  }));
}

function taskCreatorProps({
  count,
  policy,
  onCreate = vi.fn<(request: CreateJobRequest) => Promise<void>>(),
}: {
  count: number;
  policy: SpeakerCountPolicy;
  onCreate?: ReturnType<
    typeof vi.fn<(request: CreateJobRequest) => Promise<void>>
  >;
}) {
  onCreate.mockResolvedValue(undefined);
  return {
    speakers: createSpeakers(count),
    initialSpeakerPolicy: policy,
    strategies: modelStrategies,
    selectedStrategyId: "balanced" as const,
    backendMode: "mock" as const,
    busy: false,
    onClose: vi.fn(),
    onCreate,
  };
}

function renderCreator({
  count,
  policy,
}: {
  count: number;
  policy: SpeakerCountPolicy;
}) {
  const props = taskCreatorProps({ count, policy });
  renderTaskCreator(<TaskCreator open {...props} />);
  fireEvent.change(
    screen.getByPlaceholderText("Enter the absolute path to a media file"),
    {
      target: { value: "C:\\Media\\fictional-meeting.mov" },
    },
  );
  fireEvent.change(
    screen.getByPlaceholderText(
      "Enter the absolute path to an output directory",
    ),
    {
      target: { value: "C:\\Output\\fictional-job" },
    },
  );
  return props;
}

type CreatorStepLabel =
  | "Media"
  | "Speakers"
  | "Language"
  | "Models"
  | "Output";

function openCreatorStep(label: CreatorStepLabel) {
  fireEvent.click(
    screen.getByRole("tab", {
      name: new RegExp(`^${label}\\b`, "u"),
    }),
  );
}

describe("TaskCreator locale fragment", () => {
  it.each(taskCreatorLocales)(
    "keeps the %s catalog complete and placeholder-compatible",
    (locale) => {
      const englishEntries = Object.entries(taskCreatorMessages.en);
      const localized = taskCreatorMessages[locale];

      expect(Object.keys(localized).sort()).toEqual(
        Object.keys(taskCreatorMessages.en).sort(),
      );
      for (const [key, english] of englishEntries) {
        const value = localized[key as keyof typeof localized];
        const localizedPlaceholders = [
          ...value.matchAll(placeholderPattern),
        ].map((match) => match[1]);
        const englishPlaceholders = [
          ...english.matchAll(placeholderPattern),
        ].map((match) => match[1]);

        expect(value.trim()).not.toHaveLength(0);
        expect(value).not.toContain("\uFFFD");
        expect(localizedPlaceholders.sort()).toEqual(
          englishPlaceholders.sort(),
        );
      }
      expect(Object.values(localized).join(" ")).not.toMatch(/BCP[-‑]47/iu);

      if (locale !== "en") {
        const copiedValues = englishEntries.filter(
          ([key, value]) =>
            localized[key as keyof typeof localized] === value,
        );
        expect(copiedValues.length).toBeLessThanOrEqual(4);
      }
    },
  );

  it.each(taskCreatorLocales)(
    "describes extension-agnostic multi-file FFmpeg intake in %s",
    (locale) => {
      const catalog = MESSAGE_CATALOGS[locale];
      const multiFileMessages = [
        catalog["drop.readyTitle"],
        catalog["creator.pathsTauriDetail"],
        catalog["creator.batch.dropTitle"],
      ] as const;

      expect(catalog["drop.supportedTypes"]).toContain("FFmpeg");
      expect(catalog["creator.batch.dropDetail"]).toContain("FFmpeg");
      expect(catalog["drop.error.unsupportedExtension"]).toContain("FFmpeg");
      expect(catalog["creator.mediaHint"]).toContain("FFmpeg");
      expect(catalog["drop.supportedTypes"]).not.toMatch(
        finiteFormatListPattern,
      );
      expect(catalog["drop.error.unsupportedExtension"]).not.toMatch(
        finiteFormatListPattern,
      );
      multiFileMessages.forEach((message, index) => {
        expect(message).toMatch(multiFileCopyPatterns[locale][index]);
      });
    },
  );
});

describe("TaskCreator dynamic speaker policies", () => {
  it("provides five focused setup levels with keyboard and footer navigation", async () => {
    const user = userEvent.setup();
    renderCreator({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });

    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(5);
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(
      tabs.every(
        (tab) => tab.getAttribute("aria-controls") === "task-step-panel",
      ),
    ).toBe(true);
    expect(
      screen.getByPlaceholderText("Enter the absolute path to a media file"),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("radio", { name: /Manual/u }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Continue" }));
    const panel = screen.getByRole("tabpanel", {
      name: /^Speakers\b/u,
    });
    await waitFor(() => expect(panel).toHaveFocus());
    expect(
      screen.getByRole("radio", { name: /Manual/u }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Back" }));
    await waitFor(() =>
      expect(
        screen.getByRole("tabpanel", { name: /^Media\b/u }),
      ).toHaveFocus(),
    );

    const mediaTab = screen.getByRole("tab", { name: /^Media\b/u });
    mediaTab.focus();
    fireEvent.keyDown(mediaTab, { key: "End" });
    expect(screen.getByRole("tab", { name: /^Output\b/u })).toHaveFocus();
    expect(
      screen.getByRole("tab", { name: /^Output\b/u }),
    ).toHaveAttribute("aria-selected", "true");
    expect(
      screen.getByRole("button", { name: "Create job" }),
    ).toBeInTheDocument();

    fireEvent.keyDown(
      screen.getByRole("tab", { name: /^Output\b/u }),
      { key: "Home" },
    );
    expect(screen.getByRole("tab", { name: /^Media\b/u })).toHaveFocus();
    expect(
      screen.getByRole("tab", { name: /^Media\b/u }),
    ).toHaveAttribute("aria-selected", "true");

    fireEvent.keyDown(mediaTab, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: /^Speakers\b/u })).toHaveFocus();
    expect(
      screen.getByRole("tab", { name: /^Speakers\b/u }),
    ).toHaveAttribute("aria-selected", "true");
  });

  it("initializes native source and output paths while keeping output editable", async () => {
    const user = userEvent.setup();
    const props = taskCreatorProps({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });
    renderTaskCreator(
      <TaskCreator
        open
        initialMediaSelection={{
          sequence: 1,
          sourcePath: "D:\\Media\\meeting.mov",
          outputDirectory: "D:\\Media\\meeting-MediaTranscribeStudio",
        }}
        {...props}
      />,
    );

    const sourceInput = screen.getByPlaceholderText(
      "Enter the absolute path to a media file",
    );
    const outputInput = screen.getByPlaceholderText(
      "Enter the absolute path to an output directory",
    );
    expect(sourceInput).toHaveValue("D:\\Media\\meeting.mov");
    expect(outputInput).toHaveValue(
      "D:\\Media\\meeting-MediaTranscribeStudio",
    );

    await user.clear(outputInput);
    await user.type(outputInput, "D:\\Media\\editable-output");
    expect(outputInput).toHaveValue("D:\\Media\\editable-output");
  });

  it("blocks a source-overwriting output path", async () => {
    const user = userEvent.setup();
    renderCreator({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });

    const sourceInput = screen.getByPlaceholderText(
      "Enter the absolute path to a media file",
    );
    const outputInput = screen.getByPlaceholderText(
      "Enter the absolute path to an output directory",
    );
    await user.clear(outputInput);
    await user.type(outputInput, (sourceInput as HTMLInputElement).value);

    expect(outputInput).toHaveAttribute("aria-invalid", "true");
    expect(
      screen.getByText(
        "The output directory must not equal or overwrite the source media path.",
      ),
    ).toBeInTheDocument();
    openCreatorStep("Output");
    expect(screen.getByRole("button", { name: "Create job" })).toBeDisabled();
  });

  it("derives a default on blur but preserves a later manual output edit", async () => {
    const user = userEvent.setup();
    const resolver = vi
      .fn<(path: string) => Promise<{
        sourcePath: string;
        outputDirectory: string;
      }>>()
      .mockResolvedValueOnce({
        sourcePath: "D:\\Media\\first.mov",
        outputDirectory: "D:\\Media\\first-MediaTranscribeStudio",
      })
      .mockResolvedValueOnce({
        sourcePath: "D:\\Media\\second.wav",
        outputDirectory: "D:\\Media\\second-MediaTranscribeStudio",
      });
    const props = taskCreatorProps({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });
    renderTaskCreator(
      <TaskCreator open resolveMediaPath={resolver} {...props} />,
    );

    const sourceInput = screen.getByPlaceholderText(
      "Enter the absolute path to a media file",
    );
    const outputInput = screen.getByPlaceholderText(
      "Enter the absolute path to an output directory",
    );
    await user.type(sourceInput, "D:\\Media\\first.mov");
    await user.tab();
    await waitFor(() => {
      expect(resolver).toHaveBeenCalledWith("D:\\Media\\first.mov");
      expect(outputInput).toHaveValue(
        "D:\\Media\\first-MediaTranscribeStudio",
      );
    });

    await user.clear(outputInput);
    await user.type(outputInput, "D:\\Media\\my-custom-output");
    await user.clear(sourceInput);
    await user.type(sourceInput, "D:\\Media\\second.wav");
    await user.tab();
    await waitFor(() => expect(outputInput).toHaveValue(
      "D:\\Media\\my-custom-output",
    ));
    expect(resolver).toHaveBeenCalledTimes(2);
    expect(resolver).toHaveBeenLastCalledWith("D:\\Media\\second.wav");
  });

  it("keeps picker cancellation silent and preserves the current paths", async () => {
    const user = userEvent.setup();
    const selectMediaFile = vi.fn<() => Promise<string | null>>();
    const selectOutputDirectory = vi.fn<() => Promise<string | null>>();
    selectMediaFile.mockResolvedValue(null);
    selectOutputDirectory.mockResolvedValue(null);
    const props = taskCreatorProps({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });

    renderTaskCreator(
      <TaskCreator
        open
        initialMediaSelection={{
          sequence: 1,
          sourcePath: "D:\\Media\\meeting.mov",
          outputDirectory: "D:\\Media\\meeting-MediaTranscribeStudio",
        }}
        selectMediaFile={selectMediaFile}
        selectOutputDirectory={selectOutputDirectory}
        {...props}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Choose files" }));
    await user.click(screen.getByRole("button", { name: "Choose folder" }));

    expect(selectMediaFile).toHaveBeenCalledTimes(1);
    expect(selectOutputDirectory).toHaveBeenCalledTimes(1);
    expect(
      screen.getByPlaceholderText("Enter the absolute path to a media file"),
    ).toHaveValue("D:\\Media\\meeting.mov");
    expect(
      screen.getByPlaceholderText(
        "Enter the absolute path to an output directory",
      ),
    ).toHaveValue("D:\\Media\\meeting-MediaTranscribeStudio");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps unknown and extensionless files from a native multi-selection in the editable queue", async () => {
    const user = userEvent.setup();
    const selectedPaths = [
      "D:\\Media\\meeting.futuremedia",
      "D:\\Media\\extensionless",
    ] as const;
    const selectMediaFiles = vi
      .fn<() => Promise<readonly string[]>>()
      .mockResolvedValue(selectedPaths);
    const resolver = vi
      .fn<(path: string) => Promise<MediaSelection>>()
      .mockImplementation(async (path) => {
        await Promise.resolve();
        return {
          sourcePath: path,
          outputDirectory: `${path}-MediaTranscribeStudio`,
        };
      });
    const props = taskCreatorProps({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });

    renderTaskCreator(
      <TaskCreator
        open
        resolveMediaPath={resolver}
        selectMediaFiles={selectMediaFiles}
        {...props}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Choose files" }));

    await waitFor(() => {
      const sourceInputs = screen.getAllByPlaceholderText(
        "Enter the absolute path to a media file",
      );
      expect(sourceInputs).toHaveLength(2);
      expect(sourceInputs[0]).toHaveValue(selectedPaths[0]);
      expect(sourceInputs[1]).toHaveValue(selectedPaths[1]);
    });

    expect(selectMediaFiles).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenCalledTimes(2);
    expect(resolver).toHaveBeenNthCalledWith(1, selectedPaths[0]);
    expect(resolver).toHaveBeenNthCalledWith(2, selectedPaths[1]);
    expect(
      screen.getAllByText(
        "Unknown or missing filename extensions are accepted. Local FFmpeg will probe the actual content before processing.",
      ),
    ).toHaveLength(2);
    screen
      .getAllByPlaceholderText("Enter the absolute path to a media file")
      .forEach((input) => {
        expect(input).not.toHaveAttribute("aria-invalid", "true");
      });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("appends successive drag-drop batches in stable order and deduplicates paths case-insensitively", async () => {
    const props = taskCreatorProps({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });
    const { rerender } = renderTaskCreator(
      <TaskCreator
        open
        initialMediaBatch={{
          sequence: 1,
          selections: [
            {
              sourcePath: "D:\\Media\\FIRST.MOV",
              outputDirectory: "D:\\Output\\first",
            },
            {
              sourcePath: "d:\\media\\first.mov",
              outputDirectory: "D:\\Output\\duplicate",
            },
            {
              sourcePath: "D:\\Media\\extensionless",
              outputDirectory: "D:\\Output\\extensionless",
            },
          ],
        }}
        {...props}
      />,
    );

    expect(
      screen
        .getAllByPlaceholderText("Enter the absolute path to a media file")
        .map((input) => (input as HTMLInputElement).value),
    ).toEqual([
      "D:\\Media\\FIRST.MOV",
      "D:\\Media\\extensionless",
    ]);

    rerender(
      <TaskCreator
        open
        initialMediaBatch={{
          sequence: 2,
          selections: [
            {
              sourcePath: "D:\\MEDIA\\EXTENSIONLESS",
              outputDirectory: "D:\\Output\\duplicate-extensionless",
            },
            {
              sourcePath: "D:\\Media\\unknown.futuremedia",
              outputDirectory: "D:\\Output\\unknown",
            },
            {
              sourcePath: "D:\\Media\\CAMERA.M4A",
              outputDirectory: "D:\\Output\\camera",
            },
          ],
        }}
        {...props}
      />,
    );

    await waitFor(() => {
      expect(
        screen
          .getAllByPlaceholderText("Enter the absolute path to a media file")
          .map((input) => (input as HTMLInputElement).value),
      ).toEqual([
        "D:\\Media\\FIRST.MOV",
        "D:\\Media\\extensionless",
        "D:\\Media\\unknown.futuremedia",
        "D:\\Media\\CAMERA.M4A",
      ]);
    });
    expect(screen.getByRole("status")).toHaveTextContent(/\b4\b/u);
  });

  it("keeps picker order, skips duplicates, and isolates an unsupported directory from valid job payloads", async () => {
    const user = userEvent.setup();
    const selectedPaths = [
      "D:\\Media\\FIRST.MOV",
      "d:\\media\\first.mov",
      "D:\\Media\\not-a-media-folder",
      "D:\\Media\\second.futuremedia",
    ] as const;
    const selectMediaFiles = vi
      .fn<() => Promise<readonly string[]>>()
      .mockResolvedValue(selectedPaths);
    const resolver = vi
      .fn<(path: string) => Promise<MediaSelection>>()
      .mockImplementation(async (path) => {
        await Promise.resolve();
        if (path.endsWith("\\not-a-media-folder")) {
          throw new MediaDropError(
            "absolutePath",
            "Directories are not supported as media inputs.",
          );
        }
        return {
          sourcePath: path,
          outputDirectory: `${path}-MediaTranscribeStudio`,
        };
      });
    const onCreate =
      vi.fn<(request: CreateJobRequest) => Promise<void>>();
    onCreate.mockResolvedValue(undefined);
    const props = taskCreatorProps({
      count: 2,
      policy: { mode: "manual", count: 2 },
      onCreate,
    });

    renderTaskCreator(
      <TaskCreator
        open
        resolveMediaPath={resolver}
        selectMediaFiles={selectMediaFiles}
        {...props}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Choose files" }));

    await waitFor(() => {
      expect(
        screen
          .getAllByPlaceholderText("Enter the absolute path to a media file")
          .map((input) => (input as HTMLInputElement).value),
      ).toEqual([
        "D:\\Media\\FIRST.MOV",
        "D:\\Media\\not-a-media-folder",
        "D:\\Media\\second.futuremedia",
      ]);
    });
    expect(resolver).toHaveBeenCalledTimes(3);
    expect(screen.getByText("Needs attention")).toBeInTheDocument();
    expect(
      screen.getByText("Select an absolute local media path."),
    ).toBeInTheDocument();

    openCreatorStep("Output");
    const createButton = screen.getByRole("button", { name: "Create job" });
    await waitFor(() => expect(createButton).toBeEnabled());
    await user.click(createButton);

    await waitFor(() => expect(onCreate).toHaveBeenCalledTimes(2));
    expect(onCreate.mock.calls.map(([request]) => request.mediaPath)).toEqual([
      "D:\\Media\\FIRST.MOV",
      "D:\\Media\\second.futuremedia",
    ]);
    expect(onCreate.mock.calls.map(([request]) => request.title)).toEqual([
      "Meeting transcription — 1",
      "Meeting transcription — 2",
    ]);
    expect(props.onClose).not.toHaveBeenCalled();

    openCreatorStep("Media");
    expect(screen.getByText("Needs attention")).toBeInTheDocument();
    openCreatorStep("Output");
    expect(screen.getByRole("button", { name: "Create job" })).toBeDisabled();
  });

  it("never overwrites an output folder explicitly chosen by the user", async () => {
    const user = userEvent.setup();
    const resolver = vi
      .fn<(path: string) => Promise<MediaSelection>>()
      .mockResolvedValue({
        sourcePath: "D:\\Media\\second.mov",
        outputDirectory: "D:\\Media\\second-MediaTranscribeStudio",
      });
    const selectMediaFile = vi
      .fn<() => Promise<string | null>>()
      .mockResolvedValue("D:\\Media\\second.mov");
    const selectOutputDirectory = vi
      .fn<() => Promise<string | null>>()
      .mockResolvedValue("D:\\Projects\\Final");
    const props = taskCreatorProps({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });

    renderTaskCreator(
      <TaskCreator
        open
        resolveMediaPath={resolver}
        selectMediaFile={selectMediaFile}
        selectOutputDirectory={selectOutputDirectory}
        {...props}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    await waitFor(() => {
      expect(
        screen.getByPlaceholderText(
          "Enter the absolute path to an output directory",
        ),
      ).toHaveValue("D:\\Projects\\Final");
    });

    await user.click(screen.getByRole("button", { name: "Choose files" }));
    await waitFor(() => {
      const sourceInputs = screen.getAllByPlaceholderText(
        "Enter the absolute path to a media file",
      );
      const outputInputs = screen.getAllByPlaceholderText(
        "Enter the absolute path to an output directory",
      );
      expect(sourceInputs).toHaveLength(2);
      expect(outputInputs).toHaveLength(2);
      expect(sourceInputs[0]).toHaveValue("");
      expect(outputInputs[0]).toHaveValue("D:\\Projects\\Final");
      expect(sourceInputs[1]).toHaveValue("D:\\Media\\second.mov");
      expect(outputInputs[1]).toHaveValue(
        "D:\\Media\\second-MediaTranscribeStudio",
      );
    });
  });

  it("keeps translation opt-in, human-readable, removable, and suggestion-only", async () => {
    const user = userEvent.setup();
    const { onCreate } = renderCreator({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });
    openCreatorStep("Language");
    const businessSwitch = screen.getByRole("switch", {
      name: /Enable local business processing/u,
    });
    const translation = screen.getByRole("checkbox", {
      name: /Translation artifacts/u,
    });

    expect(businessSwitch).not.toBeChecked();
    expect(translation).not.toBeChecked();
    expect(translation).toBeDisabled();
    expect(
      screen.queryByText("Advanced language settings"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("textbox", { name: /^Languages not listed/u }),
    ).not.toBeInTheDocument();

    await user.click(businessSwitch);
    await user.click(translation);

    const translateTo = screen.getByRole("combobox", {
      name: /^Translate to/u,
    });
    expect(translateTo).toHaveValue("");
    expect(
      within(translateTo).getByRole("option", { name: "Japanese" }),
    ).toHaveValue("ja-JP");
    expect(
      within(translateTo).queryByRole("option", { name: "ja-JP" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Create job" }),
    ).not.toBeInTheDocument();
    expect(onCreate).not.toHaveBeenCalled();

    await user.selectOptions(translateTo, "ja-JP");
    expect(onCreate).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Remove Japanese" }),
    ).toBeInTheDocument();

    openCreatorStep("Output");
    const createButton = screen.getByRole("button", { name: "Create job" });
    await waitFor(() => expect(createButton).toBeEnabled());
    await user.click(createButton);

    await waitFor(() => expect(onCreate).toHaveBeenCalledTimes(1));
    expect(onCreate.mock.calls[0][0]).toMatchObject({
      translationTargets: ["ja-JP"],
      localLlmMode: "business",
      localLlmAutoApply: false,
    });
  });

  it("deduplicates preset translation chips without exposing raw language-tag input", async () => {
    const user = userEvent.setup();
    renderCreator({
      count: 2,
      policy: { mode: "manual", count: 2 },
    });
    openCreatorStep("Language");

    await user.click(
      screen.getByRole("switch", {
        name: /Enable local business processing/u,
      }),
    );
    await user.click(
      screen.getByRole("checkbox", { name: /Translation artifacts/u }),
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: /^Translate to/u }),
      "ja-JP",
    );

    const chips = screen.getByLabelText("Parsed translation targets");
    expect(within(chips).getAllByText("Japanese")).toHaveLength(1);
    expect(
      within(
        screen.getByRole("combobox", { name: /^Translate to/u }),
      ).getByRole("option", { name: "Japanese" }),
    ).toBeDisabled();
    expect(
      screen.queryByText("Advanced language settings"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("textbox", { name: /^Languages not listed/u }),
    ).not.toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "Remove Japanese" }),
    );
    expect(
      screen.queryByLabelText("Parsed translation targets"),
    ).not.toBeInTheDocument();
    expect(
      within(
        screen.getByRole("combobox", { name: /^Translate to/u }),
      ).getByRole("option", { name: "Japanese" }),
    ).toBeEnabled();
  });

  it.each(dynamicCounts)(
    "materializes and submits %i manual speaker labels",
    async (count) => {
      const user = userEvent.setup();
      const { onCreate } = renderCreator({
        count,
        policy: { mode: "manual", count },
      });
      openCreatorStep("Speakers");
      const dialog = screen.getByRole("dialog", {
        name: "Create transcription job",
      });

      await waitFor(() => {
        expect(
          within(dialog).getAllByRole("textbox", {
            name: /Initial name for speaker-\d+/u,
          }),
        ).toHaveLength(count);
      });
      expect(
        within(dialog).getByRole("textbox", {
          name: `Initial name for speaker-${count}`,
        }),
      ).toHaveValue(`Speaker ${count}`);
      expect(
        within(dialog).queryByDisplayValue(
          `Previous job participant ${count}`,
        ),
      ).not.toBeInTheDocument();

      openCreatorStep("Output");
      const submitButton = within(dialog).getByRole("button", {
        name: "Create job",
      });
      await waitFor(() => expect(submitButton).toBeEnabled());
      await user.click(submitButton);

      await waitFor(() => expect(onCreate).toHaveBeenCalledTimes(1));
      const request = onCreate.mock.calls[0][0];
      expect(request.speakerPolicy).toEqual({ mode: "manual", count });
      expect(request.speakerLabels).toHaveLength(count);
      expect(request.speakerLabels.at(-1)).toBe(`Speaker ${count}`);
    },
  );

  it("uses the hybrid prior without exposing another job's detection evidence", async () => {
    const user = userEvent.setup();
    const policy: SpeakerCountPolicy = {
      mode: "hybrid",
      minSpeakers: 2,
      priorCount: 13,
      maxSpeakers: 21,
    };
    const { onCreate } = renderCreator({
      count: 13,
      policy,
    });
    openCreatorStep("Speakers");

    expect(
      screen.getByText("Waiting for local speaker-count detection"),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Speaker-count detection result"),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Speaker-count candidates"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText(
        "The estimated count, confidence, and candidate distribution will appear after the job starts.",
      ),
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(
        screen.getAllByRole("textbox", {
          name: /Initial name for speaker-\d+/u,
        }),
      ).toHaveLength(13);
    });
    expect(
      screen.getByRole("textbox", { name: "Initial name for speaker-13" }),
    ).toHaveValue("Speaker 13");

    openCreatorStep("Output");
    await user.click(screen.getByRole("button", { name: "Create job" }));
    await waitFor(() => expect(onCreate).toHaveBeenCalledTimes(1));
    const request = onCreate.mock.calls[0][0];
    expect(request.speakerPolicy).toEqual(policy);
    expect(request.speakerLabels).toContain("Speaker 1");
    expect(request.speakerLabels).toContain("Speaker 13");
    expect(request.speakerLabels).toHaveLength(13);
  });

  it("submits auto mode without labels and waits for current-media analysis", async () => {
    const user = userEvent.setup();
    const { onCreate } = renderCreator({
      count: 8,
      policy: { mode: "auto" },
    });
    openCreatorStep("Speakers");

    expect(
      screen.queryByRole("textbox", {
        name: /Initial name for speaker-\d+/u,
      }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText("Waiting for local speaker-count detection"),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Speaker-count detection result"),
    ).not.toBeInTheDocument();

    openCreatorStep("Output");
    await user.click(screen.getByRole("button", { name: "Create job" }));
    await waitFor(() => expect(onCreate).toHaveBeenCalledTimes(1));
    expect(onCreate.mock.calls[0][0]).toMatchObject({
      speakerPolicy: { mode: "auto" },
      speakerLabels: [],
    });
  });

  it("keeps complete inline editing at the safe threshold", async () => {
    const user = userEvent.setup();
    const count = MAX_INLINE_SPEAKER_EDITORS;
    const { onCreate } = renderCreator({
      count,
      policy: { mode: "manual", count },
    });
    openCreatorStep("Speakers");

    await waitFor(() => {
      expect(
        screen.getAllByRole("textbox", {
          name: /Initial name for speaker-\d+/u,
        }),
      ).toHaveLength(count);
    });
    openCreatorStep("Output");
    await user.click(screen.getByRole("button", { name: "Create job" }));

    await waitFor(() => expect(onCreate).toHaveBeenCalledTimes(1));
    expect(onCreate.mock.calls[0][0].speakerLabels).toHaveLength(count);
  });

  it("uses empty labels above the inline-editor threshold", async () => {
    const user = userEvent.setup();
    const count = MAX_INLINE_SPEAKER_EDITORS + 1;
    const { onCreate } = renderCreator({
      count: 0,
      policy: { mode: "manual", count },
    });
    openCreatorStep("Speakers");

    expect(
      screen.queryByRole("textbox", {
        name: /Initial name for speaker-\d+/u,
      }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Large-count safe mode enabled")).toBeInTheDocument();
    expect(
      screen.getByText(
        /default names will be generated after the job is created/u,
      ),
    ).toBeInTheDocument();

    openCreatorStep("Output");
    await user.click(screen.getByRole("button", { name: "Create job" }));
    await waitFor(() => expect(onCreate).toHaveBeenCalledTimes(1));
    expect(onCreate.mock.calls[0][0]).toMatchObject({
      speakerPolicy: { mode: "manual", count },
      speakerLabels: [],
    });
  });

  it("does not allocate a MAX_SAFE_INTEGER-sized array", async () => {
    const user = userEvent.setup();
    const count = Number.MAX_SAFE_INTEGER;
    const { onCreate } = renderCreator({
      count: 0,
      policy: { mode: "manual", count },
    });
    openCreatorStep("Speakers");

    expect(screen.getByDisplayValue(String(count))).toBeInTheDocument();
    expect(
      screen.queryByRole("textbox", {
        name: /Initial name for speaker-\d+/u,
      }),
    ).not.toBeInTheDocument();

    openCreatorStep("Output");
    expect(screen.getByRole("button", { name: "Create job" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Create job" }));
    await waitFor(() => expect(onCreate).toHaveBeenCalledTimes(1));
    expect(onCreate.mock.calls[0][0]).toMatchObject({
      speakerPolicy: { mode: "manual", count },
      speakerLabels: [],
    });
  });

  it("uses the hybrid prior as the dynamic count in large-speaker safe mode", async () => {
    const user = userEvent.setup();
    const priorCount = MAX_INLINE_SPEAKER_EDITORS + 1;
    const policy: SpeakerCountPolicy = {
      mode: "hybrid",
      minSpeakers: 2,
      priorCount,
      maxSpeakers: 64,
    };
    const { onCreate } = renderCreator({
      count: 0,
      policy,
    });
    openCreatorStep("Speakers");

    expect(screen.getByText("Large-count safe mode enabled")).toBeInTheDocument();
    openCreatorStep("Output");
    await user.click(screen.getByRole("button", { name: "Create job" }));
    await waitFor(() => expect(onCreate).toHaveBeenCalledTimes(1));
    expect(onCreate.mock.calls[0][0]).toMatchObject({
      speakerPolicy: policy,
      speakerLabels: [],
    });
  });

  it("resets paths, business settings, runtime overrides, and edits whenever it reopens", async () => {
    const user = userEvent.setup();
    const props = taskCreatorProps({
      count: 5,
      policy: { mode: "manual", count: 5 },
    });
    const { rerender } = renderTaskCreator(<TaskCreator open {...props} />);

    fireEvent.change(screen.getByRole("textbox", { name: "Job name" }), {
      target: { value: "Sensitive customer session" },
    });
    fireEvent.change(
      screen.getByPlaceholderText("Enter the absolute path to a media file"),
      {
        target: { value: "C:\\Sensitive\\customer.mov" },
      },
    );
    fireEvent.change(
      screen.getByPlaceholderText(
        "Enter the absolute path to an output directory",
      ),
      {
        target: { value: "C:\\Sensitive\\output" },
      },
    );

    openCreatorStep("Speakers");
    fireEvent.change(
      screen.getByRole("textbox", {
        name: "Initial name for speaker-1",
      }),
      {
        target: { value: "Confidential participant" },
      },
    );
    await user.click(screen.getByRole("radio", { name: /Automatic/u }));

    openCreatorStep("Language");
    await user.selectOptions(
      screen.getByRole("combobox", { name: /Source language/u }),
      "fr-FR",
    );
    await user.click(
      screen.getByRole("switch", {
        name: /Enable local business processing/u,
      }),
    );
    await user.click(
      screen.getByRole("checkbox", { name: /Translation artifacts/u }),
    );
    await user.click(
      screen.getByRole("checkbox", {
        name: /Structured meeting intelligence/u,
      }),
    );
    await user.selectOptions(
      screen.getByRole("combobox", {
        name: /^Translate to/u,
      }),
      "de-DE",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: /Writing language/u }),
      "ja-JP",
    );
    await user.click(screen.getByText("Advanced local runtime"));
    fireEvent.change(
      screen.getByRole("textbox", { name: "Local model" }),
      {
        target: { value: "private-model" },
      },
    );
    fireEvent.change(
      screen.getByRole("textbox", { name: /^Loopback endpoint/u }),
      {
        target: { value: "http://localhost:9999" },
      },
    );
    openCreatorStep("Output");

    rerender(<TaskCreator open={false} {...props} />);
    rerender(<TaskCreator open {...props} />);

    await waitFor(() => {
      expect(screen.getByRole("textbox", { name: "Job name" })).toHaveValue(
        "Meeting transcription",
      );
    });
    expect(
      screen.getByRole("tab", { name: /^Media\b/u }),
    ).toHaveAttribute("aria-selected", "true");
    expect(
      screen.getByPlaceholderText("Enter the absolute path to a media file"),
    ).toHaveValue("");
    expect(
      screen.getByPlaceholderText(
        "Enter the absolute path to an output directory",
      ),
    ).toHaveValue("");

    openCreatorStep("Speakers");
    expect(screen.getByRole("radio", { name: /Manual/u })).toBeChecked();
    expect(
      screen.getByRole("textbox", { name: "Initial name for speaker-1" }),
    ).toHaveValue("Speaker 1");

    openCreatorStep("Language");
    expect(
      screen.getByRole("combobox", { name: /Source language/u }),
    ).toHaveValue("auto");
    expect(
      screen.getByRole("switch", {
        name: /Enable local business processing/u,
      }),
    ).not.toBeChecked();

    await user.click(
      screen.getByRole("switch", {
        name: /Enable local business processing/u,
      }),
    );
    expect(
      screen.getByRole("checkbox", { name: /Translation artifacts/u }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: /Structured meeting intelligence/u,
      }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("combobox", { name: /Writing language/u }),
    ).toHaveValue("en-US");
    await user.click(screen.getByText("Advanced local runtime"));
    expect(screen.getByRole("textbox", { name: "Local model" })).toHaveValue(
      "qwen3.5:9b",
    );
    expect(
      screen.getByRole("textbox", { name: /^Loopback endpoint/u }),
    ).toHaveValue("http://127.0.0.1:11434");
  });
});
