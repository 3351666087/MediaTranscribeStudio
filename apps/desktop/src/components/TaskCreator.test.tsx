import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import type {
  CreateJobRequest,
  SpeakerCountPolicy,
  SpeakerProfile,
} from "../contracts/studio";
import {
  ENGLISH_MESSAGES,
  LOCALE_OPTIONS,
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
        expect(localizedPlaceholders.sort()).toEqual(
          englishPlaceholders.sort(),
        );
      }

      if (locale !== "en") {
        const copiedValues = englishEntries.filter(
          ([key, value]) =>
            localized[key as keyof typeof localized] === value,
        );
        expect(copiedValues.length).toBeLessThanOrEqual(4);
      }
    },
  );
});

describe("TaskCreator dynamic speaker policies", () => {
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
    expect(resolver).toHaveBeenCalledTimes(1);
  });

  it.each(dynamicCounts)(
    "materializes and submits %i manual speaker labels",
    async (count) => {
      const user = userEvent.setup();
      const { onCreate } = renderCreator({
        count,
        policy: { mode: "manual", count },
      });
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

    await waitFor(() => {
      expect(
        screen.getAllByRole("textbox", {
          name: /Initial name for speaker-\d+/u,
        }),
      ).toHaveLength(count);
    });
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

    expect(screen.getByDisplayValue(String(count))).toBeInTheDocument();
    expect(
      screen.queryByRole("textbox", {
        name: /Initial name for speaker-\d+/u,
      }),
    ).not.toBeInTheDocument();
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

    expect(screen.getByText("Large-count safe mode enabled")).toBeInTheDocument();
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

    await user.clear(screen.getByRole("textbox", { name: "Job name" }));
    await user.type(
      screen.getByRole("textbox", { name: "Job name" }),
      "Sensitive customer session",
    );
    await user.type(
      screen.getByPlaceholderText("Enter the absolute path to a media file"),
      "C:\\Sensitive\\customer.mov",
    );
    await user.type(
      screen.getByPlaceholderText(
        "Enter the absolute path to an output directory",
      ),
      "C:\\Sensitive\\output",
    );
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
        name: /Source-language semantic polish/u,
      }),
    );
    await user.click(
      screen.getByRole("checkbox", {
        name: /Structured meeting intelligence/u,
      }),
    );
    await user.type(
      screen.getByRole("textbox", {
        name: /Translation target languages/u,
      }),
      "de-DE",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: /Business output locale/u }),
      "ja-JP",
    );
    await user.click(screen.getByText("Advanced local runtime"));
    await user.clear(screen.getByRole("textbox", { name: "Local model" }));
    await user.type(
      screen.getByRole("textbox", { name: "Local model" }),
      "private-model",
    );
    await user.clear(
      screen.getByRole("textbox", { name: /^Loopback endpoint/u }),
    );
    await user.type(
      screen.getByRole("textbox", { name: /^Loopback endpoint/u }),
      "http://localhost:9999",
    );
    await user.clear(
      screen.getByRole("textbox", { name: "Initial name for speaker-1" }),
    );
    await user.type(
      screen.getByRole("textbox", { name: "Initial name for speaker-1" }),
      "Confidential participant",
    );
    await user.click(screen.getByRole("radio", { name: /Automatic/u }));

    rerender(<TaskCreator open={false} {...props} />);
    rerender(<TaskCreator open {...props} />);

    await waitFor(() => {
      expect(screen.getByRole("textbox", { name: "Job name" })).toHaveValue(
        "Meeting transcription",
      );
    });
    expect(
      screen.getByPlaceholderText("Enter the absolute path to a media file"),
    ).toHaveValue("");
    expect(
      screen.getByPlaceholderText(
        "Enter the absolute path to an output directory",
      ),
    ).toHaveValue("");
    expect(
      screen.getByRole("combobox", { name: /Source language/u }),
    ).toHaveValue("auto");
    expect(
      screen.getByRole("switch", {
        name: /Enable local business processing/u,
      }),
    ).not.toBeChecked();
    expect(screen.getByRole("radio", { name: /Manual/u })).toBeChecked();
    expect(
      screen.getByRole("textbox", { name: "Initial name for speaker-1" }),
    ).toHaveValue("Speaker 1");

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
        name: /Source-language semantic polish/u,
      }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: /Structured meeting intelligence/u,
      }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("combobox", { name: /Business output locale/u }),
    ).toHaveValue("en-US");
    await user.click(screen.getByText("Advanced local runtime"));
    expect(screen.getByRole("textbox", { name: "Local model" })).toHaveValue(
      "qwen3.5:4b",
    );
    expect(
      screen.getByRole("textbox", { name: /^Loopback endpoint/u }),
    ).toHaveValue("http://127.0.0.1:11434");
  });
});
