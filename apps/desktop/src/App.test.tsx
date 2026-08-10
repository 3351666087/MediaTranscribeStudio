import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { desktopBackend } from "./bridge/desktop-backend";
import { createStudioFixture } from "./mocks/studio-fixture";

describe("MediaTranscribe Studio desktop scaffold", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a recoverable, offline-safe load failure instead of an endless boot screen", async () => {
    const user = userEvent.setup();
    vi.spyOn(desktopBackend, "getSnapshot")
      .mockRejectedValueOnce(new Error("Local IPC is temporarily unavailable."))
      .mockResolvedValueOnce(createStudioFixture(8));

    render(<App />);

    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "The local workspace is not ready yet",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Local IPC is temporarily unavailable.")).toBeInTheDocument();
    expect(screen.getByText("Still fully offline")).toBeInTheDocument();
    expect(screen.getByText("Fail closed")).toBeInTheDocument();
    expect(
      screen.getByRole("alert"),
    ).toHaveAttribute("data-evidence-state", "load-failure");

    await user.click(
      screen.getByRole("button", { name: "Reload local workspace" }),
    );

    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Return every sentence to the right speaker.",
      }),
    ).toBeInTheDocument();
    expect(document.querySelector(".app-shell")).toHaveAttribute(
      "data-speaker-count",
      "8",
    );
  });

  it("supports the complete offline dynamic-speaker review and QA journey", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Return every sentence to the right speaker.",
      }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByText("Fully offline")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        level: 2,
        name: "Choose your next workspace",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Speaker & strategy studio/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Confidence & performance lab/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Pipeline & event observatory/ }),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("8 speakers currently detected"),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Model runtime")).toBeInTheDocument();
    expect(screen.queryByText(/Python worker/i)).not.toBeInTheDocument();
    const companionLayer = document.querySelector(".desktop-companion-layer");
    expect(companionLayer).not.toBeNull();
    expect(companionLayer).not.toHaveAttribute("inert");
    expect(
      within(companionLayer as HTMLElement).getByRole("complementary"),
    ).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: /Speaker & strategy studio/ }),
    );
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Speaker & strategy studio",
      }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", {
        level: 1,
        name: "Return every sentence to the right speaker.",
      }),
    ).not.toBeInTheDocument();
    expect(screen.getByLabelText("8 speakers currently detected")).toHaveTextContent("8 speakers");
    expect(
      screen.getAllByRole("textbox", { name: /speaker-\d+ name/ }),
    ).toHaveLength(8);

    await user.click(
      screen.getByRole("button", { name: "Back to overview" }),
    );
    await user.click(
      screen.getByRole("button", { name: /Confidence & performance lab/ }),
    );
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Confidence & performance lab",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Speaker separation quality" }),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Unavailable")).toHaveLength(4);
    expect(screen.getByText("1.48%")).toBeInTheDocument();
    expect(screen.getByText("No fabricated benchmarks.")).toBeInTheDocument();

    expect(
      screen.getByRole("heading", { level: 2, name: "Performance workbench" }),
    ).toBeInTheDocument();
    expect(screen.getByText("0.37×")).toBeInTheDocument();
    expect(screen.getByText("78%")).toBeInTheDocument();
    expect(screen.getByText("6.8%")).toBeInTheDocument();
    expect(screen.getByText("2.2%")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "p50" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "p95" })).toBeInTheDocument();
    expect(screen.getByText("Peak resources")).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "Back to overview" }),
    );
    await user.click(
      screen.getByRole("button", { name: /Speaker & strategy studio/ }),
    );
    const qualityStrategy = screen.getByRole("radio", { name: /Quality first/ });
    await user.click(qualityStrategy);
    expect(qualityStrategy).toBeChecked();
    const strategyPanel = screen
      .getByRole("heading", { level: 2, name: "Model strategy" })
      .closest("section");
    expect(strategyPanel).not.toBeNull();
    expect(
      within(strategyPanel as HTMLElement).getByText("qwen3.5:27b-q4_K_M"),
    ).toBeInTheDocument();
    expect(
      within(strategyPanel as HTMLElement).getByText("suggestion_only"),
    ).toBeInTheDocument();
    expect(
      within(strategyPanel as HTMLElement).getByText(
        /qwen3\.5:27b-q4_K_M won the current multilingual semantic challenge/i,
      ),
    ).toBeInTheDocument();

    const createTaskButton = screen.getByRole("button", { name: "Create task" });
    await user.click(createTaskButton);
    const dialog = screen.getByRole("dialog", { name: "Create transcription job" });
    expect(companionLayer).toHaveAttribute("inert");
    expect(companionLayer).toHaveAttribute("aria-hidden", "true");
    expect(dialog).toHaveAccessibleDescription(
      "Register one or more local paths only. Media is not uploaded and remote models are never contacted.",
    );
    await user.click(
      within(dialog).getByRole("tab", { name: /^Speakers\b/u }),
    );
    expect(
      within(dialog).getAllByRole("textbox", {
        name: /Initial name for speaker-\d+/u,
      }),
    ).toHaveLength(8);
    await user.click(
      within(dialog).getByRole("tab", { name: /^Output\b/u }),
    );
    expect(within(dialog).getByText("Offline guarantee")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Create transcription job" })).not.toBeInTheDocument();
    expect(createTaskButton).toHaveFocus();
    expect(companionLayer).not.toHaveAttribute("inert");
    expect(companionLayer).not.toHaveAttribute("aria-hidden");

    await user.click(
      screen.getByRole("button", { name: "Back to overview" }),
    );
    await user.click(
      screen.getByRole("button", { name: /Pipeline & event observatory/ }),
    );
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Pipeline & event observatory",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "8-stage progress" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Event stream" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Review queue" }));
    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Low-confidence review queue",
      }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("4 reviews pending")).toBeInTheDocument();
    const reviewSummary = screen.getByText("Queue signals").closest("section");
    expect(reviewSummary).not.toBeNull();
    expect(
      within(reviewSummary as HTMLElement).getByText("Acoustic conflict"),
    ).toBeInTheDocument();
    expect(
      within(reviewSummary as HTMLElement).getByText("Count uncertainty"),
    ).toBeInTheDocument();
    expect(
      within(reviewSummary as HTMLElement).getByText("Overlap"),
    ).toBeInTheDocument();
    expect(
      within(reviewSummary as HTMLElement).getByText("Audio review"),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Low CAM++ margin")).not.toHaveLength(0);
    expect(screen.getAllByText("Timestamp or speaker-boundary conflict")).not.toHaveLength(0);
    expect(screen.getAllByText("Overlapping speech")).not.toHaveLength(0);
    expect(screen.getAllByText("Speaker outlier escalation")).not.toHaveLength(0);
    expect(screen.getByText("1 human-locked")).toBeInTheDocument();
    expect(
      screen.getByRole("textbox", { name: "Human correction draft" }),
    ).toHaveValue("那这个边界我觉得可以先锁。然后我补充一下，不是这个意思。");

    await user.click(screen.getByRole("radio", { name: /Speaker 2 · Product/ }));
    await user.type(
      screen.getByRole("textbox", { name: "Decision rationale" }),
      "Human review assigns this segment to the second speaker.",
    );
    await user.type(
      screen.getByRole("textbox", { name: "Evidence" }),
      "Local listening confirms that the second speaker's voiceprint and boundary match the audio.",
    );
    await user.type(
      screen.getByRole("spinbutton", { name: "Human confidence" }),
      "0.96",
    );
    await user.click(screen.getByRole("button", { name: "Confirm and lock" }));

    await waitFor(() => {
      expect(screen.getByLabelText("3 reviews pending")).toBeInTheDocument();
    });
    expect(
      screen.getByRole("heading", {
        level: 2,
        name: "00:24:15.060–00:24:22.740",
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Artifacts" }));
    expect(
      await screen.findByRole("heading", { level: 1, name: "Artifacts" }),
    ).toBeInTheDocument();
    expect(screen.getByText("source-transcript.zh-Hans.pdf")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "PDF quality" }));
    expect(
      await screen.findByRole("heading", { level: 1, name: "PDF quality" }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);

    const hardGateSection = screen
      .getByText("13 PDF hard gates")
      .closest("details");
    expect(hardGateSection).not.toBeNull();
    expect(hardGateSection).not.toHaveAttribute("open");
    await user.click(
      within(hardGateSection as HTMLElement).getByText(
        "13 PDF hard gates",
      ),
    );
    expect(within(hardGateSection as HTMLElement).getAllByRole("listitem")).toHaveLength(13);

    const facetSection = screen
      .getByText("14 visual facets")
      .closest("details");
    expect(facetSection).not.toBeNull();
    expect(facetSection).not.toHaveAttribute("open");
    await user.click(
      within(facetSection as HTMLElement).getByText("14 visual facets"),
    );
    expect(
      within(facetSection as HTMLElement).getAllByRole("progressbar"),
    ).toHaveLength(14);
    expect(
      screen.getByText("Transcript text, speaker identity, and timestamps are never auto-repaired."),
    ).toBeInTheDocument();

    screen.getAllByRole("button").forEach((button) => {
      expect(button).toHaveAccessibleName();
    });
  }, 15_000);

  it("switches among the nine interface locales and persists explicit choice", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByRole("heading", {
      level: 1,
      name: "Return every sentence to the right speaker.",
    });
    const preferences = screen.getByText("Appearance and language", {
      selector: "strong",
    }).closest("details");
    expect(preferences).not.toBeNull();
    const languageSelect = within(preferences as HTMLElement).getByRole(
      "combobox",
      { name: "Interface language" },
    );
    expect(within(languageSelect).getAllByRole("option")).toHaveLength(9);

    await user.selectOptions(languageSelect, "zh-Hans");
    expect(
      screen.getByRole("button", { name: "概览" }),
    ).toBeInTheDocument();
    expect(window.localStorage.getItem("media-transcribe-studio.ui-locale")).toBe(
      "zh-Hans",
    );

    await user.selectOptions(languageSelect, "en");
    expect(
      screen.getByRole("button", { name: "Overview" }),
    ).toBeInTheDocument();
  });
});
