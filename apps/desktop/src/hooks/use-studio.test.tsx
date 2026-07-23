import {
  act,
  cleanup,
  render,
  renderHook,
  screen,
} from "@testing-library/react";
import type {
  CreateJobRequest,
  CreateJobResult,
  ReviewDecision,
  ReviewSegment,
  SpeakerProfile,
  StudioSnapshot,
  UpdateSpeakerRequest,
} from "../contracts/studio";
import { createStudioFixture } from "../mocks/studio-fixture";
import { useStudio } from "./use-studio";

type SnapshotEventListener = (event: { payload: unknown }) => void;

const mocks = vi.hoisted(() => ({
  listen: vi.fn<
    (
      eventName: string,
      listener: SnapshotEventListener,
    ) => Promise<() => void>
  >(),
  unlisten: vi.fn<() => void>(),
  isTauriRuntime: vi.fn<() => boolean>(),
  backend: {
    getSnapshot: vi.fn<() => Promise<StudioSnapshot>>(),
    createJob: vi.fn<
      (request: CreateJobRequest) => Promise<CreateJobResult>
    >(),
    cancelJob: vi.fn<(jobId: string) => Promise<void>>(),
    updateSpeaker: vi.fn<
      (request: UpdateSpeakerRequest) => Promise<SpeakerProfile>
    >(),
    applyReviewDecision: vi.fn<
      (decision: ReviewDecision) => Promise<ReviewSegment>
    >(),
    openArtifact: vi.fn(),
  },
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: mocks.listen,
}));

vi.mock("../bridge/desktop-backend", () => ({
  desktopBackend: mocks.backend,
  isTauriRuntime: mocks.isTauriRuntime,
}));

const createRequest: CreateJobRequest = {
  title: "Global product review",
  mediaPath: "D:\\media\\meeting.mov",
  outputDirectory: "D:\\output",
  strategyId: "balanced",
  speakerPolicy: { mode: "auto" },
  speakerLabels: [],
  language: "auto",
  localLlmMode: "disabled",
  localLlmModel: "qwen3.5:4b",
  localLlmEndpoint: "http://127.0.0.1:11434",
  localLlmEndpointPolicy: "loopback-only",
  localLlmAutoApply: false,
  translationTargets: [],
  polish: false,
  summary: false,
  outputLocale: "en-US",
  businessPromptVersion: "business-v1",
};

const updateRequest: UpdateSpeakerRequest = {
  speakerId: "speaker-1",
  label: "Facilitator",
  locked: true,
  reviewStatus: "confirmed",
};

const reviewDecision: ReviewDecision = {
  reviewId: "review-181",
  speakerId: "speaker-1",
  normalizedText: "Human-reviewed source-language transcript.",
  reason: "Local listening confirms the speaker assignment.",
  evidence: "Voiceprint, boundary, and context evidence agree.",
  confidence: 0.97,
};

function completedSnapshot(): StudioSnapshot {
  const snapshot = createStudioFixture(8);
  snapshot.job.status = "completed";
  return snapshot;
}

async function renderLoadedHook() {
  const rendered = renderHook(() => useStudio());

  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });

  expect(rendered.result.current.loading).toBe(false);
  expect(rendered.result.current.snapshot).not.toBeNull();
  return rendered;
}

function emitSnapshot(payload: unknown): void {
  const listener = mocks.listen.mock.calls.at(-1)?.[1];
  if (!listener) {
    throw new Error("The snapshot-updated listener was not registered.");
  }

  act(() => {
    listener({ payload });
  });
}

function ToastProbe() {
  const { snapshot, toast } = useStudio();

  return (
    <>
      <div data-testid="current-task">{snapshot?.job.title ?? "Loading"}</div>
      {toast ? (
        <div role="alert" data-tone={toast.tone}>
          <strong>{toast.title}</strong>
          <span>{toast.detail}</span>
        </div>
      ) : null}
    </>
  );
}

describe("useStudio synchronization", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.resetAllMocks();

    const initialSnapshot = completedSnapshot();
    mocks.isTauriRuntime.mockReturnValue(true);
    mocks.listen.mockResolvedValue(mocks.unlisten);
    mocks.backend.getSnapshot.mockResolvedValue(
      structuredClone(initialSnapshot),
    );
    mocks.backend.createJob.mockResolvedValue({
      accepted: true,
      jobId: "job-accepted",
      message: "The worker accepted the task.",
    });
    mocks.backend.cancelJob.mockResolvedValue();
    mocks.backend.updateSpeaker.mockResolvedValue(
      structuredClone(initialSnapshot.speakers[0]),
    );
    mocks.backend.applyReviewDecision.mockResolvedValue(
      structuredClone(initialSnapshot.reviews[0]),
    );
  });

  afterEach(() => {
    cleanup();
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it("does not register a Tauri listener or reconciliation poll in browser mock mode", async () => {
    mocks.isTauriRuntime.mockReturnValue(false);

    await renderLoadedHook();

    expect(mocks.listen).not.toHaveBeenCalled();
    expect(mocks.backend.getSnapshot).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(8_000);
    });

    expect(mocks.listen).not.toHaveBeenCalled();
    expect(mocks.backend.getSnapshot).toHaveBeenCalledTimes(1);
  });

  it("strictly parses and applies a valid snapshot-updated payload in Tauri mode", async () => {
    const { result } = await renderLoadedHook();
    const liveSnapshot = createStudioFixture(5);
    liveSnapshot.job.title = "Projected worker snapshot";

    expect(mocks.listen).toHaveBeenCalledWith(
      "snapshot-updated",
      expect.any(Function),
    );

    emitSnapshot(structuredClone(liveSnapshot));

    expect(result.current.snapshot).toEqual(liveSnapshot);
    expect(result.current.snapshot?.job.speakerCount).toBe(5);
    expect(result.current.snapshot?.speakers).toHaveLength(5);
    expect(result.current.toast).toBeNull();
  });

  it("rejects an invalid live payload and exposes a visible error toast", async () => {
    render(<ToastProbe />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const originalTitle = screen.getByTestId("current-task").textContent;
    const invalidPayload = {
      ...createStudioFixture(5),
      unsafeWorkerMutation: true,
    };

    emitSnapshot(invalidPayload);

    const alert = screen.getByRole("alert");
    expect(alert).toBeVisible();
    expect(alert).toHaveAttribute("data-tone", "error");
    expect(alert).toHaveTextContent("Unsafe live update rejected");
    expect(alert).toHaveTextContent(/unknown fields/u);
    expect(screen.getByTestId("current-task")).toHaveTextContent(
      originalTitle,
    );
  });

  const acceptedCommandCases = [
    {
      name: "create",
      backendMethod: "createJob",
      warningTitle: "Task created; refresh delayed",
      invoke: async (studio: ReturnType<typeof useStudio>) =>
        studio.createJob(createRequest),
    },
    {
      name: "update",
      backendMethod: "updateSpeaker",
      warningTitle: "Speaker settings saved; refresh delayed",
      invoke: async (studio: ReturnType<typeof useStudio>) =>
        studio.updateSpeaker(updateRequest),
    },
    {
      name: "review",
      backendMethod: "applyReviewDecision",
      warningTitle: "Review saved; refresh delayed",
      invoke: async (studio: ReturnType<typeof useStudio>) =>
        studio.applyReview(reviewDecision),
    },
    {
      name: "cancel",
      backendMethod: "cancelJob",
      warningTitle: "Cancellation accepted; refresh delayed",
      invoke: async (studio: ReturnType<typeof useStudio>) =>
        studio.cancelJob("job-dynamic-8"),
    },
  ] as const;

  it.each(acceptedCommandCases)(
    "does not report an accepted $name command as failed when its refresh fails",
    async ({ backendMethod, invoke, warningTitle }) => {
      const initialSnapshot = completedSnapshot();
      mocks.backend.getSnapshot
        .mockReset()
        .mockResolvedValueOnce(structuredClone(initialSnapshot))
        .mockRejectedValueOnce(
          new Error("The follow-up snapshot is temporarily unavailable."),
        );
      const { result } = await renderLoadedHook();

      await act(async () => {
        await invoke(result.current);
      });

      expect(mocks.backend[backendMethod]).toHaveBeenCalledTimes(1);
      expect(mocks.backend.getSnapshot).toHaveBeenCalledTimes(2);
      expect(result.current.toast).toMatchObject({
        tone: "warning",
        title: warningTitle,
      });
      expect(result.current.toast?.detail).toContain(
        "The follow-up snapshot is temporarily unavailable.",
      );
      expect(result.current.toast?.title).not.toMatch(/could not|failed/u);
      expect(result.current.busyAction).toBeNull();
    },
  );
});
