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
  JobRuntimeStatus,
  ReviewDecision,
  ReviewSegment,
  SpeakerProfile,
  StudioSnapshot,
  UpdateSpeakerRequest,
} from "../contracts/studio";
import {
  I18nProvider,
  LOCALE_STORAGE_KEY,
} from "../i18n";
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
    listJobs: vi.fn<() => Promise<JobRuntimeStatus[]>>(),
    selectJob: vi.fn<(jobId: string) => Promise<StudioSnapshot>>(),
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
  localLlmModel: "qwen3.5:9b",
  localLlmEndpoint: "http://127.0.0.1:11434",
  localLlmEndpointPolicy: "loopback-only",
  localLlmAutoApply: false,
  translationTargets: [],
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

function runtimeStatus(
  jobId: string,
  status: JobRuntimeStatus["status"],
  projected: boolean,
): JobRuntimeStatus {
  const active =
    status === "draft" ||
    status === "queued" ||
    status === "running" ||
    status === "review_required";
  return {
    jobId,
    status,
    revision: 1,
    acceptedByWorker: true,
    projected,
    inFlight: active,
    workerEventRouteRegistered: active,
    cancellable:
      status === "queued" ||
      status === "running" ||
      status === "review_required",
    volatileOnly: true,
  };
}

function statusesFor(
  snapshot: StudioSnapshot,
  siblings: readonly JobRuntimeStatus[] = [],
): JobRuntimeStatus[] {
  return [
    runtimeStatus(snapshot.job.id, snapshot.job.status, true),
    ...siblings,
  ];
}

async function renderLoadedHook(options?: {
  wrapper?: typeof I18nProvider;
}) {
  const rendered = renderHook(() => useStudio(), options);

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
    mocks.backend.listJobs.mockResolvedValue(
      statusesFor(initialSnapshot),
    );
    mocks.backend.selectJob.mockResolvedValue(
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
    window.localStorage.removeItem(LOCALE_STORAGE_KEY);
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
        studio.cancelJob("job-demo-001"),
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

  it("isolates batch failures and deterministically opens the last accepted item", async () => {
    const initialSnapshot = completedSnapshot();
    const selectedSnapshot = completedSnapshot();
    selectedSnapshot.job.id = "job-c";
    selectedSnapshot.job.title = "Third accepted task";
    selectedSnapshot.job.sourcePath = "D:\\media\\third.wav";

    mocks.backend.createJob
      .mockReset()
      .mockResolvedValueOnce({
        accepted: true,
        jobId: "job-a",
        message: "First accepted.",
      })
      .mockRejectedValueOnce(new Error("Second item is invalid."))
      .mockResolvedValueOnce({
        accepted: true,
        jobId: "job-c",
        message: "Third accepted.",
      });
    mocks.backend.selectJob.mockResolvedValue(
      structuredClone(selectedSnapshot),
    );
    mocks.backend.listJobs
      .mockReset()
      .mockResolvedValueOnce(statusesFor(initialSnapshot))
      .mockResolvedValueOnce([
        runtimeStatus("job-a", "queued", false),
        runtimeStatus("job-c", "completed", true),
      ]);

    const { result } = await renderLoadedHook();
    const requests = [
      {
        ...createRequest,
        title: "First task",
        mediaPath: "D:\\media\\first.wav",
      },
      {
        ...createRequest,
        title: "Invalid task",
        mediaPath: "D:\\media\\invalid.bin",
      },
      {
        ...createRequest,
        title: "Third accepted task",
        mediaPath: "D:\\media\\third.wav",
      },
    ];

    let batchResult:
      | Awaited<ReturnType<typeof result.current.createJobBatch>>
      | undefined;
    await act(async () => {
      batchResult = await result.current.createJobBatch(requests);
    });

    expect(batchResult).toMatchObject({
      acceptedCount: 2,
      failedCount: 1,
    });
    expect(batchResult?.items.map((item) => item.status)).toEqual([
      "accepted",
      "failed",
      "accepted",
    ]);
    expect(mocks.backend.createJob).toHaveBeenCalledTimes(3);
    expect(mocks.backend.selectJob).toHaveBeenCalledWith("job-c");
    expect(result.current.snapshot?.job.id).toBe("job-c");
    expect(result.current.selectedJobId).toBe("job-c");
    expect(result.current.toast).toMatchObject({
      tone: "warning",
      title: "Batch completed with isolated failures",
    });
    expect(result.current.busyAction).toBeNull();
  });

  it("switches the projected snapshot and selected task by exact job ID", async () => {
    const initialSnapshot = completedSnapshot();
    const selectedSnapshot = completedSnapshot();
    selectedSnapshot.job.id = "job-second";
    selectedSnapshot.job.title = "Second workspace";
    selectedSnapshot.job.sourcePath = "D:\\media\\second.mov";
    mocks.backend.listJobs
      .mockReset()
      .mockResolvedValueOnce(
        statusesFor(initialSnapshot, [
          runtimeStatus("job-second", "running", false),
        ]),
      )
      .mockResolvedValueOnce([
        runtimeStatus(initialSnapshot.job.id, "completed", false),
        runtimeStatus("job-second", "completed", true),
      ]);
    mocks.backend.selectJob.mockResolvedValue(
      structuredClone(selectedSnapshot),
    );

    const { result } = await renderLoadedHook();

    await act(async () => {
      await result.current.selectJob("job-second");
    });

    expect(mocks.backend.selectJob).toHaveBeenCalledTimes(1);
    expect(mocks.backend.selectJob).toHaveBeenCalledWith("job-second");
    expect(result.current.snapshot?.job.title).toBe("Second workspace");
    expect(result.current.selectedJobId).toBe("job-second");
    expect(result.current.activeSection).toBe("overview");
  });

  it("keeps a freshly selected snapshot authoritative over a stale task-list refresh", async () => {
    const initialSnapshot = completedSnapshot();
    const selectedSnapshot = completedSnapshot();
    selectedSnapshot.job.id = "job-fresh";
    selectedSnapshot.job.title = "Fresh workspace";
    selectedSnapshot.job.sourcePath = "D:\\media\\fresh.mov";
    let resolveStaleRefresh:
      | ((statuses: JobRuntimeStatus[]) => void)
      | undefined;
    const staleRefresh = new Promise<JobRuntimeStatus[]>((resolve) => {
      resolveStaleRefresh = resolve;
    });

    mocks.backend.listJobs
      .mockReset()
      .mockResolvedValueOnce(
        statusesFor(initialSnapshot, [
          runtimeStatus("job-fresh", "running", false),
        ]),
      )
      .mockImplementationOnce(async () => await staleRefresh)
      .mockResolvedValueOnce([
        runtimeStatus(initialSnapshot.job.id, "completed", false),
        runtimeStatus("job-fresh", "completed", true),
      ]);
    mocks.backend.selectJob.mockResolvedValue(
      structuredClone(selectedSnapshot),
    );

    const { result } = await renderLoadedHook();
    let pendingRefresh: Promise<JobRuntimeStatus[]> | undefined;
    act(() => {
      pendingRefresh = result.current.refreshJobs();
    });

    await act(async () => {
      await result.current.selectJob("job-fresh");
    });

    await act(async () => {
      resolveStaleRefresh?.(statusesFor(initialSnapshot, [
        runtimeStatus("job-fresh", "running", false),
      ]));
      await pendingRefresh;
    });

    expect(result.current.snapshot?.job.id).toBe("job-fresh");
    expect(result.current.selectedJobId).toBe("job-fresh");
  });

  it("cancels a non-current task without selecting or refreshing its snapshot", async () => {
    const initialSnapshot = completedSnapshot();
    mocks.backend.listJobs
      .mockReset()
      .mockResolvedValueOnce(
        statusesFor(initialSnapshot, [
          runtimeStatus("job-background", "running", false),
        ]),
      )
      .mockResolvedValueOnce(
        statusesFor(initialSnapshot, [
          runtimeStatus("job-background", "cancelled", false),
        ]),
      );

    const { result } = await renderLoadedHook();

    await act(async () => {
      await result.current.cancelJob("job-background");
    });

    expect(mocks.backend.cancelJob).toHaveBeenCalledWith(
      "job-background",
    );
    expect(mocks.backend.selectJob).not.toHaveBeenCalled();
    expect(mocks.backend.getSnapshot).toHaveBeenCalledTimes(1);
    expect(mocks.backend.listJobs).toHaveBeenCalledTimes(2);
    expect(result.current.snapshot?.job.id).toBe(
      initialSnapshot.job.id,
    );
    expect(result.current.selectedJobId).toBe(initialSnapshot.job.id);
  });

  it("keeps reconciliation active when a sibling runs behind a terminal selected task", async () => {
    const initialSnapshot = completedSnapshot();
    mocks.backend.listJobs.mockResolvedValue(
      statusesFor(initialSnapshot, [
        runtimeStatus("job-background", "running", false),
      ]),
    );

    await renderLoadedHook();

    expect(mocks.backend.getSnapshot).toHaveBeenCalledTimes(1);
    expect(mocks.backend.listJobs).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });

    expect(mocks.backend.getSnapshot).toHaveBeenCalledTimes(2);
    expect(mocks.backend.listJobs).toHaveBeenCalledTimes(2);
  });

  it("localizes new task-switch notifications through message keys", async () => {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, "zh-Hans");
    const initialSnapshot = completedSnapshot();
    const selectedSnapshot = completedSnapshot();
    selectedSnapshot.job.id = "job-zh";
    selectedSnapshot.job.title = "中文访谈";
    mocks.backend.listJobs
      .mockReset()
      .mockResolvedValueOnce(
        statusesFor(initialSnapshot, [
          runtimeStatus("job-zh", "completed", false),
        ]),
      )
      .mockResolvedValueOnce([
        runtimeStatus(initialSnapshot.job.id, "completed", false),
        runtimeStatus("job-zh", "completed", true),
      ]);
    mocks.backend.selectJob.mockResolvedValue(
      structuredClone(selectedSnapshot),
    );

    const { result } = await renderLoadedHook({
      wrapper: I18nProvider,
    });

    await act(async () => {
      await result.current.selectJob("job-zh");
    });

    expect(result.current.toast).toMatchObject({
      tone: "success",
      title: "任务已打开",
      detail: "“中文访谈”现在是当前工作区。",
    });
  });
});
