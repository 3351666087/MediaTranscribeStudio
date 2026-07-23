import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type {
  ReviewDecision,
  ReviewReason,
  ReviewSegment,
  SpeakerProfile,
} from "../contracts/studio";
import {
  ENGLISH_MESSAGES,
  LOCALE_OPTIONS,
  SUPPORTED_LOCALES,
  type MessageParams,
} from "../i18n";
import { I18nContext } from "../i18n/context";
import reviewQueueMessages from "../i18n/fragments/review-queue.json";
import { ReviewQueue } from "./ReviewQueue";

const PAGE_SIZE = 40;
const scaleCounts = [129, 500, 1000] as const;
const englishReviewMessages = reviewQueueMessages.en as Partial<
  Record<string, string>
>;
const intentionalLocaleInvariants = new Set([
  "review.audit.confidencePlaceholder",
  "review.audio.duration",
  "review.editor.requiredCode",
  "review.speaker.option",
  "review.split.code",
  "review.split.status",
]);
const reviewReasons: ReviewReason[] = [
  "speaker_close_score",
  "speaker_count_uncertain",
  "overlap_detected",
  "timestamp_boundary",
  "speaker_outlier",
  "local_audio_review",
];

function interpolate(
  template: string,
  params: MessageParams = {},
): string {
  return template.replace(
    /\{([a-zA-Z][a-zA-Z0-9]*)\}/gu,
    (match, name: string) => {
      const value = params[name];
      return value === undefined ? match : String(value);
    },
  );
}

function translateForReviewTest(
  key: string,
  params: MessageParams = {},
): string {
  const template =
    englishReviewMessages[key] ??
    (ENGLISH_MESSAGES as Partial<Record<string, string>>)[key] ??
    key;
  return interpolate(template, params);
}

function placeholders(value: string): string[] {
  return [...value.matchAll(/\{([a-zA-Z][a-zA-Z0-9]*)\}/gu)]
    .map((match) => match[1])
    .sort();
}

function englishReviewMessage(key: string): string {
  const value = englishReviewMessages[key];
  if (value === undefined) {
    throw new Error(`Missing review-queue English message: ${key}`);
  }
  return value;
}

function createSpeakers(count: number): SpeakerProfile[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `speaker-${index + 1}`,
    label: `Speaker ${index + 1}`,
    shortLabel: `S${index + 1}`,
    color: `hsl(${(index * 137 + 252) % 360} 52% 49%)`,
    roleHint: `Unassigned role ${index + 1}`,
    sampleStatus: "ready",
    locked: false,
    reviewStatus: "pending",
  }));
}

function formatTimestamp(totalMs: number): string {
  const hours = Math.floor(totalMs / 3_600_000);
  const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
  const seconds = Math.floor((totalMs % 60_000) / 1_000);
  const milliseconds = totalMs % 1_000;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(milliseconds).padStart(3, "0")}`;
}

function createReviews(
  count: number,
  speakers: SpeakerProfile[],
): ReviewSegment[] {
  return Array.from({ length: count }, (_, index) => {
    const startMs = index * 2_000;
    const endMs = startMs + 1_500;
    const primarySpeaker = speakers[index % speakers.length];
    const secondarySpeaker = speakers[(index + 1) % speakers.length];
    const reason = reviewReasons[index % reviewReasons.length];

    return {
      id: `review-${index + 1}`,
      startMs,
      endMs,
      timestampLabel: `${formatTimestamp(startMs)}–${formatTimestamp(endMs)}`,
      rawText: `Immutable source-language utterance ${index + 1}.`,
      normalizedText: `Confirmed source-language utterance ${index + 1}.`,
      currentSpeakerId: primarySpeaker.id,
      candidates: [
        {
          speakerId: primarySpeaker.id,
          score: 0.62,
          evidence: "Primary local voiceprint candidate.",
        },
        {
          speakerId: secondarySpeaker.id,
          score: 0.59,
          evidence: "Close secondary local voiceprint candidate.",
        },
      ],
      reasons: [reason],
      confidence: 0.62,
      confidenceBand: "low",
      waveform: [22, 47, 71, 34],
      locked: index % 17 === 16,
      reviewed: false,
      auditTrail: [],
    };
  });
}

function renderQueue({
  reviewCount = 8,
  speakerCount = 8,
  mutateReviews,
}: {
  reviewCount?: number;
  speakerCount?: number;
  mutateReviews?: (reviews: ReviewSegment[]) => ReviewSegment[];
} = {}) {
  const speakers = createSpeakers(speakerCount);
  const baseReviews = createReviews(reviewCount, speakers);
  const reviews = mutateReviews?.(baseReviews) ?? baseReviews;
  const onApply = vi.fn<(decision: ReviewDecision) => Promise<void>>();
  onApply.mockResolvedValue(undefined);
  const onNotify = vi.fn();

  render(
    <I18nContext.Provider
      value={{
        locale: "en",
        localeOptions: LOCALE_OPTIONS,
        setLocale: () => undefined,
        t: translateForReviewTest,
      }}
    >
      <ReviewQueue
        reviews={reviews}
        speakers={speakers}
        busyAction={null}
        onApply={onApply}
        onNotify={onNotify}
      />
    </I18nContext.Provider>,
  );

  return { onApply, onNotify, reviews, speakers };
}

function getMountedReviewTabs(): HTMLElement[] {
  return screen.getAllByRole("button", {
    name: /^Open review segment \d+:/u,
  });
}

describe("ReviewQueue scalable review and lock semantics", () => {
  it("ships complete review copy for every supported locale", () => {
    const englishKeys = Object.keys(reviewQueueMessages.en).sort();
    expect(englishKeys).toHaveLength(91);
    expect(Object.keys(reviewQueueMessages).sort()).toEqual(
      [...SUPPORTED_LOCALES].sort(),
    );

    for (const locale of SUPPORTED_LOCALES) {
      const messages = reviewQueueMessages[locale] as Record<string, string>;
      expect(Object.keys(messages).sort()).toEqual(englishKeys);

      for (const key of englishKeys) {
        expect(messages[key].trim()).not.toBe("");
        const englishMessage = englishReviewMessage(key);
        expect(placeholders(messages[key])).toEqual(placeholders(englishMessage));
        const englishWordCount =
          englishMessage.match(/[A-Za-z0-9+.-]+/gu)?.length ?? 0;
        if (
          locale !== "en" &&
          !intentionalLocaleInvariants.has(key) &&
          englishWordCount >= 4
        ) {
          expect(messages[key]).not.toBe(englishMessage);
        }
      }
    }
  });

  it.each(scaleCounts)(
    "mounts one 40-segment page and preserves all %i review records",
    (count) => {
      const { reviews } = renderQueue({ reviewCount: count });
      const totalPages = Math.ceil(count / PAGE_SIZE);
      const finalPageSize = count % PAGE_SIZE || PAGE_SIZE;
      const finalPageStart = (totalPages - 1) * PAGE_SIZE + 1;

      expect(
        screen.getByRole("heading", {
          level: 1,
          name: "Low-confidence review queue",
        }),
      ).toBeInTheDocument();
      expect(screen.getByText(`${count} items`)).toBeInTheDocument();
      expect(getMountedReviewTabs()).toHaveLength(PAGE_SIZE);
      expect(
        screen.getByText(`Showing 1–${PAGE_SIZE} of ${count}`),
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("button", {
          name: new RegExp(`^Open review segment ${PAGE_SIZE + 1}:`, "u"),
        }),
      ).not.toBeInTheDocument();

      fireEvent.change(
        screen.getByRole("spinbutton", { name: "Review page number" }),
        { target: { value: String(totalPages) } },
      );

      expect(getMountedReviewTabs()).toHaveLength(finalPageSize);
      expect(
        screen.getByRole("button", {
          name: new RegExp(`^Open review segment ${finalPageStart}:`, "u"),
        }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", {
          name: new RegExp(`^Open review segment ${count}:`, "u"),
        }),
      ).toBeInTheDocument();
      expect(
        screen.getByText(`Showing ${finalPageStart}–${count} of ${count}`),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("heading", {
          level: 2,
          name: reviews[finalPageStart - 1]?.timestampLabel,
        }),
      ).toBeInTheDocument();
    },
  );

  it("supports previous, next, direct-page, and clamped review navigation", async () => {
    const user = userEvent.setup();
    renderQueue({ reviewCount: 129 });

    const pageInput = screen.getByRole("spinbutton", {
      name: "Review page number",
    });
    const previous = screen.getByRole("button", {
      name: "Previous page of review segments",
    });
    const next = screen.getByRole("button", {
      name: "Next page of review segments",
    });

    expect(previous).toBeDisabled();
    await user.click(next);
    expect(pageInput).toHaveValue(2);
    expect(
      screen.getByRole("button", {
        name: /^Open review segment 41:/u,
      }),
    ).toHaveAttribute("aria-pressed", "true");

    await user.click(previous);
    expect(pageInput).toHaveValue(1);

    fireEvent.change(pageInput, { target: { value: "3" } });
    expect(pageInput).toHaveValue(3);
    expect(
      screen.getByRole("button", {
        name: /^Open review segment 81:/u,
      }),
    ).toHaveAttribute("aria-pressed", "true");

    fireEvent.change(pageInput, { target: { value: "999" } });
    expect(pageInput).toHaveValue(4);
    expect(next).toBeDisabled();

    fireEvent.change(pageInput, { target: { value: "0" } });
    expect(pageInput).toHaveValue(1);
    expect(previous).toBeDisabled();
  });

  it("aggregates review reasons and human locks once without changing queue semantics", () => {
    const { reviews } = renderQueue({ reviewCount: 129 });
    const summary = screen
      .getByRole("heading", { level: 1, name: "Low-confidence review queue" })
      .closest("section");
    expect(summary).not.toBeNull();

    const expectedLocked = reviews.filter((review) => review.locked).length;
    expect(
      within(summary as HTMLElement).getByText(
        `${expectedLocked} human-locked`,
      ),
    ).toBeInTheDocument();

    const expectedByReason = new Map<ReviewReason, number>();
    for (const review of reviews) {
      for (const reason of review.reasons) {
        expectedByReason.set(reason, (expectedByReason.get(reason) ?? 0) + 1);
      }
    }
    const summaryItems = within(summary as HTMLElement).getAllByRole("listitem");
    const signalExpectations: Array<[string, ReviewReason]> = [
      ["Acoustic conflict", "speaker_close_score"],
      ["Count uncertainty", "speaker_count_uncertain"],
      ["Overlap", "overlap_detected"],
      ["Audio review", "local_audio_review"],
    ];
    for (const [label, reason] of signalExpectations) {
      const item = summaryItems.find((candidate) =>
        within(candidate).queryByText(label),
      );
      expect(item).toBeDefined();
      expect(
        within(item as HTMLElement).getByText(
          String(expectedByReason.get(reason) ?? 0),
        ),
      ).toBeInTheDocument();
    }
  });

  it("preserves the human lock while allowing source-language transcript review", () => {
    renderQueue({
      reviewCount: 1,
      mutateReviews: (reviews) =>
        reviews.map((review) => ({ ...review, locked: true })),
    });

    expect(screen.getByText("Speaker human-locked")).toBeInTheDocument();
    screen.getAllByRole("radio").forEach((radio) => {
      expect(radio).toBeDisabled();
    });
    expect(
      screen.getByRole("combobox", { name: "All speaker tracks" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("textbox", { name: "Human correction draft" }),
    ).toBeEnabled();
    expect(
      screen.getByText(
        "A reviewer locked this speaker; only boundaries and source-language text may be reviewed.",
      ),
    ).toBeInTheDocument();
  });

  it("keeps rawText immutable and requires explicit acceptance before submitting a correction", async () => {
    const user = userEvent.setup();
    const { onApply, reviews } = renderQueue({ reviewCount: 1 });
    const activeReview = reviews[0];
    const revisedText =
      "This corrected source-language transcript was explicitly accepted.";

    const rawTextNode = screen.getByText(activeReview.rawText);
    expect(rawTextNode.tagName).toBe("P");
    expect(
      screen.queryByDisplayValue(activeReview.rawText),
    ).not.toBeInTheDocument();
    expect(screen.getByText("rawText · immutable")).toBeInTheDocument();

    const suggestion = screen.getByRole("textbox", {
      name: "Human correction draft",
    });
    const reason = screen.getByRole("textbox", {
      name: "Decision rationale",
    });
    const evidence = screen.getByRole("textbox", { name: "Evidence" });
    const confidence = screen.getByRole("spinbutton", {
      name: "Human confidence",
    });
    const submit = screen.getByRole("button", {
      name: "Confirm and lock",
    });

    fireEvent.change(reason, {
      target: {
        value: "The reviewer verified punctuation and segmentation.",
      },
    });
    fireEvent.change(evidence, {
      target: {
        value:
          "The immutable recognition evidence and local context were compared.",
      },
    });
    fireEvent.change(confidence, { target: { value: "0.95" } });
    expect(submit).toBeEnabled();

    fireEvent.change(suggestion, { target: { value: revisedText } });
    expect(screen.getByText("Awaiting human acceptance")).toBeInTheDocument();
    expect(submit).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Discard draft" }));
    expect(suggestion).toHaveValue(activeReview.normalizedText);
    expect(submit).toBeEnabled();

    fireEvent.change(suggestion, { target: { value: revisedText } });
    await user.click(
      screen.getByRole("button", {
        name: "Accept as human-corrected transcript",
      }),
    );
    expect(
      screen.getByText("Explicitly accepted by the current reviewer"),
    ).toBeInTheDocument();
    expect(submit).toBeEnabled();

    await user.click(submit);
    expect(onApply).toHaveBeenCalledWith({
      reviewId: activeReview.id,
      speakerId: activeReview.currentSpeakerId,
      normalizedText: revisedText,
      reason: "The reviewer verified punctuation and segmentation.",
      evidence:
        "The immutable recognition evidence and local context were compared.",
      confidence: 0.95,
    });
  });

  it("keeps all 1000 Dynamic-N speaker tracks available beyond top candidates", () => {
    const { onApply } = renderQueue({ reviewCount: 1, speakerCount: 1000 });
    const trackPicker = screen.getByRole("combobox", {
      name: "All speaker tracks",
    });

    expect(trackPicker).toHaveAttribute(
      "id",
      "review-speaker-select-review-1",
    );
    expect(within(trackPicker).getAllByRole("option")).toHaveLength(1000);
    fireEvent.change(trackPicker, { target: { value: "speaker-1000" } });
    expect(trackPicker).toHaveValue("speaker-1000");
    expect(
      screen.getByText(
        "All 1000 Dynamic-N tracks remain available. The native selector supports keyboard navigation beyond the top two candidates.",
      ),
    ).toBeInTheDocument();

    fireEvent.change(
      screen.getByRole("textbox", { name: "Decision rationale" }),
      {
        target: {
          value:
            "The reviewer selected a speaker from the complete Dynamic-N set.",
        },
      },
    );
    fireEvent.change(screen.getByRole("textbox", { name: "Evidence" }), {
      target: {
        value:
          "Local voiceprint continuity supports the selected high-number track.",
      },
    });
    fireEvent.change(
      screen.getByRole("spinbutton", { name: "Human confidence" }),
      { target: { value: "0.91" } },
    );
    fireEvent.click(screen.getByRole("button", { name: "Confirm and lock" }));

    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({
        reviewId: "review-1",
        speakerId: "speaker-1000",
      }),
    );
  });

  it("exposes stable IDs and records split data as DRAFT_ONLY without mutating the review", async () => {
    const user = userEvent.setup();
    const { onApply, onNotify } = renderQueue({ reviewCount: 1 });

    expect(
      screen.getByRole("button", { name: /^Open review segment 1:/u }),
    ).toHaveAttribute("id", "review-tab-review-1");
    expect(
      screen.getByRole("radio", {
        name: /^Speaker 1, candidate confidence/u,
      }),
    ).toHaveAttribute(
      "id",
      "review-candidate-review-1-speaker-1",
    );
    expect(
      screen.getByRole("textbox", { name: "Human correction draft" }),
    ).toHaveAttribute("id", "review-suggestion-review-1");
    expect(
      screen.getByRole("textbox", { name: "Decision rationale" }),
    ).toHaveAttribute("id", "review-reason-review-1");
    expect(
      screen.getByRole("textbox", { name: "Evidence" }),
    ).toHaveAttribute("id", "review-evidence-review-1");
    expect(
      screen.getByRole("spinbutton", { name: "Human confidence" }),
    ).toHaveAttribute("id", "review-confidence-review-1");

    const splitSummary = screen
      .getByText("Segment split audit draft")
      .closest("summary");
    expect(splitSummary).not.toBeNull();
    if (!splitSummary) {
      throw new Error("Expected the split-audit summary element.");
    }
    await user.click(splitSummary);
    const splitOffset = screen.getByRole("spinbutton", {
      name: "Split point within segment in seconds",
    });
    const splitConfidence = screen.getByRole("spinbutton", {
      name: "Human confidence for the split draft",
    });
    const splitReason = screen.getByRole("textbox", {
      name: "Split rationale",
    });
    const splitEvidence = screen.getByRole("textbox", {
      name: "Audio-review evidence for the split draft",
    });

    expect(splitOffset).toHaveAttribute("id", "review-split-offset-review-1");
    expect(splitConfidence).toHaveAttribute(
      "id",
      "review-split-confidence-review-1",
    );
    expect(splitReason).toHaveAttribute(
      "id",
      "review-split-reason-review-1",
    );
    expect(splitEvidence).toHaveAttribute(
      "id",
      "review-split-evidence-review-1",
    );

    await user.clear(splitOffset);
    await user.type(splitOffset, "0.75");
    await user.type(splitConfidence, "0.92");
    await user.type(
      splitReason,
      "Two speaker turns meet inside the current segment.",
    );
    await user.type(
      splitEvidence,
      "The reviewed waveform and voiceprint transition align at 0.75 seconds.",
    );
    await user.click(
      screen.getByRole("button", { name: "Create split audit draft" }),
    );

    expect(screen.getByText("DRAFT_ONLY · +0.75 s")).toBeInTheDocument();
    expect(onApply).not.toHaveBeenCalled();
    expect(onNotify).toHaveBeenCalledWith(
      "info",
      "Split audit draft recorded",
      "DRAFT_ONLY: backend support is unavailable, so no segment or speaker set was changed.",
    );
  });

  it("disables simulated playback and requires structured human audit fields", async () => {
    const user = userEvent.setup();
    renderQueue({ reviewCount: 1 });

    expect(
      screen.getByRole("button", { name: "Play segment audio" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", {
        name: "Play when the audio sidecar is available",
      }),
    ).toBeDisabled();
    expect(
      screen.getByText(
        "Audio sidecar unavailable · simulated playback is disabled",
      ),
    ).toBeInTheDocument();

    const reason = screen.getByRole("textbox", {
      name: "Decision rationale",
    });
    const evidence = screen.getByRole("textbox", { name: "Evidence" });
    const confidence = screen.getByRole("spinbutton", {
      name: "Human confidence",
    });
    const submit = screen.getByRole("button", {
      name: "Confirm and lock",
    });

    expect(reason).toHaveValue("");
    expect(evidence).toHaveValue("");
    expect(confidence).toHaveValue(null);
    expect(submit).toBeDisabled();

    await user.type(reason, "A reviewer made the final assignment.");
    await user.type(evidence, "Local audio and adjacent context were reviewed.");
    await user.type(confidence, "1.01");
    expect(submit).toBeDisabled();
    await user.clear(confidence);
    await user.type(confidence, "0.93");
    expect(submit).toBeEnabled();
  });
});
