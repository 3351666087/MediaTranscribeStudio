import { useState } from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type {
  ReviewSegment,
  SpeakerProfile,
  UpdateSpeakerRequest,
} from "../contracts/studio";
import speakerSetupCatalogs from "../i18n/fragments/speaker-setup.json";
import { SpeakerSetupPanel } from "./SpeakerSetupPanel";

vi.mock("../i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../i18n")>();
  const catalogs = (
    await import("../i18n/fragments/speaker-setup.json")
  ).default;
  const messages: Partial<Record<string, string>> = catalogs.en;

  return {
    ...actual,
    useI18n: () => ({
      locale: "en" as const,
      localeOptions: actual.LOCALE_OPTIONS,
      setLocale: () => undefined,
      t: (
        key: Parameters<typeof actual.translate>[1],
        params: Parameters<typeof actual.translate>[2] = {},
      ) => {
        const template = messages[key];
        if (template === undefined) {
          return key;
        }
        return template.replace(
          /\{([a-zA-Z][a-zA-Z0-9]*)\}/gu,
          (match, name: string) => {
            const value = params[name];
            return value === undefined ? match : String(value);
          },
        );
      },
    }),
  };
});

const PAGE_SIZE = 40;
const scaleCounts = [129, 500, 1000] as const;
const expectedLocales = [
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

function placeholders(value: string): string[] {
  return Array.from(
    value.matchAll(/\{([a-zA-Z][a-zA-Z0-9]*)\}/gu),
    (match) => match[1],
  ).sort();
}

function createSpeakers(count: number): SpeakerProfile[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `speaker-${index + 1}`,
    label: `Speaker ${index + 1}`,
    shortLabel: `S${index + 1}`,
    color: `hsl(${(index * 137 + 252) % 360} 52% 49%)`,
    roleHint: `Unassigned role ${index + 1}`,
    sampleStatus: index % 3 === 0 ? "ready" : "missing",
    locked: index > 0 && index % 97 === 0,
    reviewStatus:
      index > 0 && index % 113 === 0 ? "needs_review" : "pending",
  }));
}

function createReview(id = "review-1"): ReviewSegment {
  return {
    id,
    startMs: 1_000,
    endMs: 3_500,
    timestampLabel: "00:00:01.000–00:00:03.500",
    rawText: "Source-language recognition evidence.",
    normalizedText: "Source-language recognition evidence.",
    currentSpeakerId: "speaker-1",
    candidates: [
      {
        speakerId: "speaker-1",
        score: 0.62,
        evidence: "Primary voiceprint candidate.",
      },
      {
        speakerId: "speaker-2",
        score: 0.59,
        evidence: "Close secondary voiceprint candidate.",
      },
    ],
    reasons: ["speaker_close_score"],
    confidence: 0.62,
    confidenceBand: "low",
    waveform: [20, 48, 72, 35],
    locked: false,
    reviewed: false,
    auditTrail: [],
  };
}

function SpeakerHarness({
  initialSpeakers,
  reviews = [],
  jobId = "job-speaker-harness",
  onUpdateSpy,
}: {
  initialSpeakers: SpeakerProfile[];
  reviews?: ReviewSegment[];
  jobId?: string;
  onUpdateSpy?: (request: UpdateSpeakerRequest) => void;
}) {
  const [speakers, setSpeakers] = useState(initialSpeakers);

  const update = async (request: UpdateSpeakerRequest): Promise<void> => {
    onUpdateSpy?.(request);
    setSpeakers((current) =>
      current.map((speaker) =>
        speaker.id === request.speakerId
          ? {
              ...speaker,
              label: request.label,
              locked: request.locked,
              reviewStatus: request.reviewStatus,
            }
          : speaker,
      ),
    );
    await Promise.resolve();
  };

  return (
    <SpeakerSetupPanel
      jobId={jobId}
      speakers={speakers}
      reviews={reviews}
      speakerPolicy={{ mode: "manual", count: speakers.length }}
      speakerDetection={null}
      busyAction={null}
      onUpdate={update}
    />
  );
}

function getMountedSpeakerNameInputs(): HTMLInputElement[] {
  return Array.from(
    document.querySelectorAll<HTMLInputElement>(
      'input[id^="speaker-label-speaker-"]',
    ),
  );
}

function openGovernancePanel(): void {
  const toggle = screen.getByRole("button", {
    name: /Speaker-governance audit drafts/u,
  });
  expect(toggle).toHaveAttribute("aria-expanded", "false");
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute("aria-expanded", "true");
}

describe("SpeakerSetupPanel scalable Dynamic-N management", () => {
  it("ships complete speaker-workspace copy for all nine locales", () => {
    const englishKeys = Object.keys(speakerSetupCatalogs.en).sort();

    expect(englishKeys).toHaveLength(95);
    expect(Object.keys(speakerSetupCatalogs)).toEqual(expectedLocales);

    for (const locale of expectedLocales) {
      const messages = speakerSetupCatalogs[locale];
      expect(Object.keys(messages).sort()).toEqual(englishKeys);

      for (const key of englishKeys) {
        const localized = messages[key as keyof typeof messages];
        expect(localized.trim()).not.toHaveLength(0);
        expect(placeholders(localized)).toEqual(
          placeholders(
            speakerSetupCatalogs.en[
              key as keyof typeof speakerSetupCatalogs.en
            ],
          ),
        );
      }
    }
  });

  it.each(scaleCounts)(
    "mounts one 40-track page and preserves the total for %i speakers",
    (count) => {
      render(<SpeakerHarness initialSpeakers={createSpeakers(count)} />);

      const totalPages = Math.ceil(count / PAGE_SIZE);
      const finalPageSize = count % PAGE_SIZE || PAGE_SIZE;
      const finalPageStart = (totalPages - 1) * PAGE_SIZE + 1;
      const countChip = document.querySelector<HTMLElement>(
        `[aria-label="${count} speakers currently detected"]`,
      );

      expect(countChip).toHaveTextContent(`${count} speakers`);
      expect(getMountedSpeakerNameInputs()).toHaveLength(PAGE_SIZE);
      expect(document.getElementById("speaker-label-speaker-1")).toHaveValue(
        "Speaker 1",
      );
      expect(
        document.getElementById("speaker-label-speaker-41"),
      ).not.toBeInTheDocument();
      expect(screen.getByText(`Showing ${count} of ${count}`)).toBeInTheDocument();
      expect(
        screen.getByText(`Showing 1–${PAGE_SIZE} of ${count}`),
      ).toBeInTheDocument();
      expect(screen.getByText(`of ${totalPages}`)).toBeInTheDocument();

      fireEvent.change(
        document.getElementById("speaker-page-input") as HTMLInputElement,
        { target: { value: String(totalPages) } },
      );

      expect(getMountedSpeakerNameInputs()).toHaveLength(finalPageSize);
      expect(
        document.getElementById(`speaker-label-speaker-${finalPageStart}`),
      ).toBeInTheDocument();
      expect(
        document.getElementById(`speaker-label-speaker-${count}`),
      ).toHaveValue(`Speaker ${count}`);
      expect(
        screen.getByText(`Showing ${finalPageStart}–${count} of ${count}`),
      ).toBeInTheDocument();
    },
  );

  it("supports previous, next, direct-page, and clamped page navigation", async () => {
    const user = userEvent.setup();
    render(<SpeakerHarness initialSpeakers={createSpeakers(129)} />);

    const pageInput = screen.getByRole("spinbutton", {
      name: "Speaker page number",
    });
    const previous = screen.getByRole("button", {
      name: "Previous page of speakers",
    });
    const next = screen.getByRole("button", {
      name: "Next page of speakers",
    });

    expect(previous).toBeDisabled();
    await user.click(next);
    expect(pageInput).toHaveValue(2);
    expect(
      screen.getByRole("textbox", { name: "speaker-41 name" }),
    ).toBeInTheDocument();

    await user.click(previous);
    expect(pageInput).toHaveValue(1);

    fireEvent.change(pageInput, { target: { value: "3" } });
    expect(pageInput).toHaveValue(3);
    expect(
      screen.getByRole("textbox", { name: "speaker-81 name" }),
    ).toBeInTheDocument();

    fireEvent.change(pageInput, { target: { value: "999" } });
    expect(pageInput).toHaveValue(4);
    expect(next).toBeDisabled();

    fireEvent.change(pageInput, { target: { value: "0" } });
    expect(pageInput).toHaveValue(1);
    expect(previous).toBeDisabled();
  });

  it("resets pagination for a high-number search without changing the 1000-speaker total", async () => {
    const user = userEvent.setup();
    render(<SpeakerHarness initialSpeakers={createSpeakers(1000)} />);

    const pageInput = screen.getByRole("spinbutton", {
      name: "Speaker page number",
    });
    fireEvent.change(pageInput, { target: { value: "25" } });
    expect(pageInput).toHaveValue(25);

    await user.type(
      screen.getByRole("searchbox", { name: "Search speaker tracks" }),
      "speaker-1000",
    );

    expect(pageInput).toHaveValue(1);
    expect(getMountedSpeakerNameInputs()).toHaveLength(1);
    expect(
      screen.getByRole("textbox", { name: "speaker-1000 name" }),
    ).toHaveValue("Speaker 1000");
    expect(screen.getByText("Showing 1 of 1000")).toBeInTheDocument();
    expect(screen.getByText("Showing 1–1 of 1")).toBeInTheDocument();
    expect(
      screen.getByLabelText("1000 speakers currently detected"),
    ).toHaveTextContent("1000 speakers");
  });

  it("resets to page one when filtering and keeps the invariant total count", async () => {
    const user = userEvent.setup();
    const speakers = createSpeakers(500);
    render(<SpeakerHarness initialSpeakers={speakers} />);

    const pageInput = screen.getByRole("spinbutton", {
      name: "Speaker page number",
    });
    fireEvent.change(pageInput, { target: { value: "7" } });
    expect(pageInput).toHaveValue(7);

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Speaker view filter" }),
      "needs_review",
    );

    const expectedFilteredCount = speakers.filter(
      (speaker) => speaker.reviewStatus === "needs_review",
    ).length;
    expect(pageInput).toHaveValue(1);
    expect(getMountedSpeakerNameInputs()).toHaveLength(expectedFilteredCount);
    expect(
      screen.getByText(`Showing ${expectedFilteredCount} of 500`),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("500 speakers currently detected"),
    ).toHaveTextContent("500 speakers");
    expect(
      screen.getByText(
        "Filters affect this view only; they never change the Dynamic-N speaker set, continuous IDs, or detected count.",
      ),
    ).toBeInTheDocument();
  });

  it("keeps names and review-state actions accessible and auditable", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(
      <SpeakerHarness
        initialSpeakers={createSpeakers(13)}
        onUpdateSpy={onUpdate}
      />,
    );

    const search = screen.getByRole("searchbox", {
      name: "Search speaker tracks",
    });
    const nameInput = screen.getByRole("textbox", {
      name: "speaker-13 name",
    });
    expect(search).toHaveAttribute("id", "speaker-search-input");
    expect(nameInput).toHaveAttribute("id", "speaker-label-speaker-13");

    await user.clear(nameInput);
    await user.type(nameInput, "Interview guest");
    await user.click(
      screen.getByRole("button", { name: "Save name for speaker-13" }),
    );
    await waitFor(() =>
      expect(onUpdate).toHaveBeenLastCalledWith({
        speakerId: "speaker-13",
        label: "Interview guest",
        locked: false,
        reviewStatus: "pending",
      }),
    );

    await user.click(screen.getByRole("button", { name: "Lock speaker-13" }));
    expect(
      await screen.findByRole("button", { name: "Unlock speaker-13" }),
    ).toHaveAttribute("aria-pressed", "true");

    await user.click(
      screen.getByRole("button", {
        name: "speaker-13 Mark as reviewed",
      }),
    );
    expect(
      await screen.findByRole("button", {
        name: "speaker-13 Mark as unreviewed",
      }),
    ).toHaveAttribute("aria-pressed", "true");

    await user.click(
      screen.getByRole("button", {
        name: "speaker-13 Require review",
      }),
    );
    expect(
      await screen.findByRole("button", {
        name: "speaker-13 Clear review requirement",
      }),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("records Dynamic-N merge and split operations as local DRAFT_ONLY audit entries", () => {
    const onUpdate = vi.fn();
    render(
      <SpeakerHarness
        initialSpeakers={createSpeakers(129)}
        reviews={[createReview("review-1000")]}
        onUpdateSpy={onUpdate}
      />,
    );

    openGovernancePanel();
    const mergeForm = screen.getByRole("group", { name: "Merge draft" });
    expect(
      within(mergeForm).getByRole("combobox", {
        name: "Merge source track",
      }),
    ).toHaveAttribute("id", "speaker-merge-source");
    expect(
      within(mergeForm).getByRole("combobox", {
        name: "Merge target track",
      }),
    ).toHaveAttribute("id", "speaker-merge-target");
    expect(
      within(mergeForm).getByRole("textbox", { name: "Merge rationale" }),
    ).toHaveAttribute("id", "speaker-merge-reason");
    expect(
      within(mergeForm).getByRole("textbox", {
        name: "Voiceprint or context evidence for merge",
      }),
    ).toHaveAttribute("id", "speaker-merge-evidence");

    fireEvent.change(
      within(mergeForm).getByRole("combobox", {
        name: "Merge source track",
      }),
      { target: { value: "speaker-129" } },
    );
    fireEvent.change(
      within(mergeForm).getByRole("combobox", {
        name: "Merge target track",
      }),
      { target: { value: "speaker-128" } },
    );
    fireEvent.change(
      within(mergeForm).getByRole("textbox", { name: "Merge rationale" }),
      {
        target: {
          value: "Both tracks belong to the same reviewer-verified voice.",
        },
      },
    );
    fireEvent.change(
      within(mergeForm).getByRole("textbox", {
        name: "Voiceprint or context evidence for merge",
      }),
      {
        target: {
          value:
            "Voiceprint continuity and adjacent turns were reviewed locally.",
        },
      },
    );
    fireEvent.change(
      within(mergeForm).getByRole("spinbutton", {
        name: "Human confidence for merge draft",
      }),
      { target: { value: "0.94" } },
    );
    fireEvent.click(
      within(mergeForm).getByRole("button", {
        name: "Create merge audit draft",
      }),
    );

    const splitForm = screen.getByRole("group", { name: "Split draft" });
    expect(
      within(splitForm).getByRole("combobox", { name: "Track to split" }),
    ).toHaveAttribute("id", "speaker-split-track");
    expect(
      within(splitForm).getByRole("combobox", {
        name: "Split-segment anchor",
      }),
    ).toHaveAttribute("id", "speaker-split-anchor");

    fireEvent.change(
      within(splitForm).getByRole("combobox", { name: "Track to split" }),
      { target: { value: "speaker-129" } },
    );
    fireEvent.change(
      within(splitForm).getByRole("combobox", {
        name: "Split-segment anchor",
      }),
      { target: { value: "review-1000" } },
    );
    fireEvent.change(
      within(splitForm).getByRole("textbox", { name: "Split rationale" }),
      {
        target: {
          value:
            "The track contains two stable and mutually exclusive voice clusters.",
        },
      },
    );
    fireEvent.change(
      within(splitForm).getByRole("textbox", {
        name: "Voiceprint or timestamp-boundary evidence for split",
      }),
      {
        target: {
          value:
            "A voiceprint transition aligns with the reviewed timestamp boundary.",
        },
      },
    );
    fireEvent.change(
      within(splitForm).getByRole("spinbutton", {
        name: "Human confidence for split draft",
      }),
      { target: { value: "0.91" } },
    );
    fireEvent.click(
      within(splitForm).getByRole("button", {
        name: "Create split audit draft",
      }),
    );

    const drafts = screen.getByRole("list", {
      name: "Speaker-governance drafts",
    });
    expect(within(drafts).getAllByRole("listitem")).toHaveLength(2);
    expect(within(drafts).getByText("MERGE · DRAFT_ONLY")).toBeInTheDocument();
    expect(within(drafts).getByText("SPLIT · DRAFT_ONLY")).toBeInTheDocument();
    expect(onUpdate).not.toHaveBeenCalled();
    expect(
      screen.getByLabelText("129 speakers currently detected"),
    ).toHaveTextContent("129 speakers");
  });

  it("isolates unsaved speaker drafts when the job or roster identity changes", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn<
      (request: UpdateSpeakerRequest) => Promise<void>
    >();
    onUpdate.mockResolvedValue(undefined);
    const jobASpeakers = createSpeakers(2);
    const jobBSpeakers = createSpeakers(2).map((speaker, index) => ({
      ...speaker,
      label: `Job B speaker ${index + 1}`,
    }));
    const { rerender } = render(
      <SpeakerSetupPanel
        jobId="job-a"
        speakers={jobASpeakers}
        reviews={[]}
        speakerPolicy={{ mode: "manual", count: 2 }}
        speakerDetection={null}
        busyAction={null}
        onUpdate={onUpdate}
      />,
    );

    const nameInput = screen.getByRole("textbox", {
      name: "speaker-1 name",
    });
    await user.clear(nameInput);
    await user.type(nameInput, "Unsaved job A name");
    await user.type(
      screen.getByRole("searchbox", { name: "Search speaker tracks" }),
      "speaker-2",
    );

    rerender(
      <SpeakerSetupPanel
        jobId="job-b"
        speakers={jobBSpeakers}
        reviews={[]}
        speakerPolicy={{ mode: "manual", count: 2 }}
        speakerDetection={null}
        busyAction={null}
        onUpdate={onUpdate}
      />,
    );

    await waitFor(() => {
      expect(
        screen.getByRole("textbox", { name: "speaker-1 name" }),
      ).toHaveValue("Job B speaker 1");
    });
    expect(
      screen.getByRole("searchbox", { name: "Search speaker tracks" }),
    ).toHaveValue("");

    const jobBDraft = screen.getByRole("textbox", {
      name: "speaker-1 name",
    });
    await user.clear(jobBDraft);
    await user.type(jobBDraft, "Unsaved job B name");
    const expandedRoster = [
      {
        ...jobBSpeakers[0],
        label: "Roster revision speaker 1",
      },
      jobBSpeakers[1],
      {
        ...createSpeakers(3)[2],
        label: "Roster revision speaker 3",
      },
    ];

    rerender(
      <SpeakerSetupPanel
        jobId="job-b"
        speakers={expandedRoster}
        reviews={[]}
        speakerPolicy={{ mode: "manual", count: 3 }}
        speakerDetection={null}
        busyAction={null}
        onUpdate={onUpdate}
      />,
    );

    await waitFor(() => {
      expect(
        screen.getByRole("textbox", { name: "speaker-1 name" }),
      ).toHaveValue("Roster revision speaker 1");
    });
    expect(onUpdate).not.toHaveBeenCalled();
  });
});
