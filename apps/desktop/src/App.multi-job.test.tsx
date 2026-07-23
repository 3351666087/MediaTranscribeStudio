import {
  act,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { ControlledMediaDropAdapter } from "./bridge/media-drop";
import type {
  ArtifactOpenResult,
  CreateJobRequest,
  CreateJobResult,
  JobRuntimeStatus,
  ReviewDecision,
  ReviewSegment,
  SpeakerProfile,
  StudioSnapshot,
  UpdateSpeakerRequest,
} from "./contracts/studio";
import { createStudioFixture } from "./mocks/studio-fixture";

const mocks = vi.hoisted(() => ({
  backend: {
    getSnapshot: vi.fn<() => Promise<StudioSnapshot>>(),
    listJobs: vi.fn<() => Promise<JobRuntimeStatus[]>>(),
    selectJob: vi.fn<(jobId: string) => Promise<StudioSnapshot>>(),
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
    openArtifact: vi.fn<
      (artifactId: string) => Promise<ArtifactOpenResult>
    >(),
  },
  isTauriRuntime: vi.fn<() => boolean>(),
}));

vi.mock("./bridge/desktop-backend", () => ({
  desktopBackend: mocks.backend,
  isTauriRuntime: mocks.isTauriRuntime,
}));

function runtimeStatus(
  jobId: string,
  status: JobRuntimeStatus["status"],
  projected: boolean,
): JobRuntimeStatus {
  const active =
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
    cancellable: active,
    volatileOnly: true,
  };
}

function createAdapter(): ControlledMediaDropAdapter {
  return new ControlledMediaDropAdapter(async (path) => {
    await Promise.resolve();
    return {
      sourcePath: path,
      outputDirectory: path.replace(
        /\.[^.\\/]+$/u,
        "-MediaTranscribeStudio",
      ),
    };
  });
}

async function settleNativeListener(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
  });
}

describe("App multi-task integration", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    const initialSnapshot = createStudioFixture(5);
    mocks.isTauriRuntime.mockReturnValue(false);
    mocks.backend.getSnapshot.mockResolvedValue(
      structuredClone(initialSnapshot),
    );
    mocks.backend.listJobs.mockResolvedValue([
      runtimeStatus(initialSnapshot.job.id, "review_required", true),
      runtimeStatus("job-second", "running", false),
    ]);
    mocks.backend.selectJob.mockResolvedValue(
      structuredClone(initialSnapshot),
    );
    mocks.backend.createJob.mockResolvedValue({
      accepted: true,
      jobId: "job-created",
      message: "Accepted.",
    });
    mocks.backend.cancelJob.mockResolvedValue();
    mocks.backend.updateSpeaker.mockResolvedValue(
      structuredClone(initialSnapshot.speakers[0]),
    );
    mocks.backend.applyReviewDecision.mockResolvedValue(
      structuredClone(initialSnapshot.reviews[0]),
    );
    mocks.backend.openArtifact.mockResolvedValue({
      artifactId: "artifact-1",
      canonicalPath: "D:\\output\\artifact.pdf",
      opened: true,
      message: "Opened.",
    });
  });

  it("shows the task center and switches the current workspace by exact ID", async () => {
    const user = userEvent.setup();
    const secondSnapshot = createStudioFixture(3);
    secondSnapshot.job.id = "job-second";
    secondSnapshot.job.title = "Second workspace";
    secondSnapshot.job.sourcePath = "D:\\Media\\second.wav";
    secondSnapshot.job.status = "review_required";
    mocks.backend.selectJob.mockResolvedValue(
      structuredClone(secondSnapshot),
    );
    mocks.backend.listJobs
      .mockReset()
      .mockResolvedValueOnce([
        runtimeStatus("job-demo-001", "review_required", true),
        runtimeStatus("job-second", "running", false),
      ])
      .mockResolvedValueOnce([
        runtimeStatus("job-demo-001", "review_required", false),
        runtimeStatus("job-second", "review_required", true),
      ]);

    render(<App mediaDropAdapter={createAdapter()} />);

    await screen.findByRole("heading", {
      level: 1,
      name: "Return every sentence to the right speaker.",
    });
    const trigger = screen.getByLabelText("Tasks · 2");
    expect(trigger).toBeVisible();

    await user.click(trigger);
    await user.click(
      screen.getByRole("button", { name: "Open job-second" }),
    );

    await waitFor(() => {
      expect(mocks.backend.selectJob).toHaveBeenCalledWith("job-second");
    });
    const hero = document.querySelector(".task-hero");
    expect(hero).not.toBeNull();
    expect(within(hero as HTMLElement).getByText("Second workspace")).toBeVisible();
    expect(document.querySelector(".app-shell")).toHaveAttribute(
      "data-speaker-count",
      "3",
    );
  });

  it("routes a two-file drop through the batch coordinator and opens the last accepted task", async () => {
    const user = userEvent.setup();
    const adapter = createAdapter();
    const selectedSnapshot = createStudioFixture(5);
    selectedSnapshot.job.id = "job-drop-two";
    selectedSnapshot.job.title = "Meeting transcription — 2";
    selectedSnapshot.job.sourcePath = "D:\\Media\\two.wav";
    mocks.backend.createJob
      .mockReset()
      .mockResolvedValueOnce({
        accepted: true,
        jobId: "job-drop-one",
        message: "First accepted.",
      })
      .mockResolvedValueOnce({
        accepted: true,
        jobId: "job-drop-two",
        message: "Second accepted.",
      });
    mocks.backend.selectJob.mockResolvedValue(
      structuredClone(selectedSnapshot),
    );
    mocks.backend.listJobs
      .mockReset()
      .mockResolvedValueOnce([
        runtimeStatus("job-demo-001", "review_required", true),
      ])
      .mockResolvedValueOnce([
        runtimeStatus("job-demo-001", "review_required", false),
        runtimeStatus("job-drop-one", "queued", false),
        runtimeStatus("job-drop-two", "review_required", true),
      ]);

    render(<App mediaDropAdapter={adapter} />);
    await screen.findByRole("heading", {
      level: 1,
      name: "Return every sentence to the right speaker.",
    });
    await settleNativeListener();

    act(() => {
      adapter.emit({
        type: "drop",
        paths: ["D:\\Media\\one.mov", "D:\\Media\\two.wav"],
      });
    });

    const dialog = await screen.findByRole("dialog", {
      name: "Create transcription job",
    });
    await waitFor(() => {
      expect(
        within(dialog).getAllByPlaceholderText(
          "Enter the absolute path to a media file",
        ),
      ).toHaveLength(2);
    });
    await user.click(
      within(dialog).getByRole("tab", { name: /^Output\b/u }),
    );
    const createButton = within(dialog).getByRole("button", {
      name: "Create job",
    });
    await waitFor(() => expect(createButton).toBeEnabled());
    await user.click(createButton);

    await waitFor(() => {
      expect(mocks.backend.createJob).toHaveBeenCalledTimes(2);
      expect(mocks.backend.selectJob).toHaveBeenCalledWith(
        "job-drop-two",
      );
    });
    expect(
      mocks.backend.createJob.mock.calls.map(
        ([request]) => request.mediaPath,
      ),
    ).toEqual(["D:\\Media\\one.mov", "D:\\Media\\two.wav"]);
    await waitFor(() => {
      expect(
        screen.queryByRole("dialog", {
          name: "Create transcription job",
        }),
      ).not.toBeInTheDocument();
    });
    const hero = document.querySelector(".task-hero");
    expect(hero).not.toBeNull();
    expect(
      within(hero as HTMLElement).getByText("Meeting transcription — 2"),
    ).toBeVisible();
  });
});
