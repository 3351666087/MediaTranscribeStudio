import type { JobRuntimeStatus } from "../contracts/studio";
import { createStudioFixture } from "../mocks/studio-fixture";
import { TauriDesktopBackend } from "./tauri-backend";

type InvokeCommand = <T>(
  command: string,
  args?: Record<string, unknown>,
) => Promise<T>;

function runtimeStatus(
  overrides: Partial<JobRuntimeStatus> = {},
): JobRuntimeStatus {
  return {
    jobId: "job-a",
    status: "running",
    revision: 2,
    acceptedByWorker: true,
    projected: true,
    inFlight: true,
    workerEventRouteRegistered: true,
    cancellable: true,
    volatileOnly: true,
    ...overrides,
  };
}

function createBackend(
  implementation: (
    command: string,
    args?: Record<string, unknown>,
  ) => Promise<unknown>,
) {
  const invokeMock = vi.fn(implementation);
  return {
    backend: new TauriDesktopBackend(invokeMock as InvokeCommand),
    invokeMock,
  };
}

describe("TauriDesktopBackend multi-task IPC", () => {
  it("uses exact list_jobs and select_job command envelopes", async () => {
    const selected = createStudioFixture(5);
    selected.job.id = "job-b";
    const responses: Record<string, unknown> = {
      list_jobs: [
        runtimeStatus(),
        runtimeStatus({
          jobId: "job-b",
          status: "completed",
          projected: false,
          inFlight: false,
          workerEventRouteRegistered: false,
          cancellable: false,
        }),
      ],
      select_job: selected,
    };
    const { backend, invokeMock } = createBackend(async (command) => {
      await Promise.resolve();
      return structuredClone(responses[command]);
    });

    await expect(backend.listJobs()).resolves.toHaveLength(2);
    await expect(backend.selectJob("job-b")).resolves.toMatchObject({
      job: { id: "job-b" },
    });

    expect(invokeMock.mock.calls).toEqual([
      ["list_jobs", undefined],
      ["select_job", { jobId: "job-b" }],
    ]);
  });

  it("rejects malformed task lists at the IPC boundary", async () => {
    const { backend } = createBackend(async () => {
      await Promise.resolve();
      return [
        {
          ...runtimeStatus(),
          persistent: true,
        },
      ];
    });

    await expect(backend.listJobs()).rejects.toThrow(/unknown fields/u);
  });

  it("rejects an invalid selection ID before invoking Rust", async () => {
    const { backend, invokeMock } = createBackend(async () => {
      await Promise.resolve();
      return createStudioFixture(5);
    });

    await expect(backend.selectJob("")).rejects.toThrow(/jobId/u);
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it("strictly validates the selected snapshot returned by Rust", async () => {
    const malformed = {
      ...createStudioFixture(5),
      unexpectedProjection: true,
    };
    const { backend } = createBackend(async () => {
      await Promise.resolve();
      return malformed;
    });

    await expect(backend.selectJob("job-b")).rejects.toThrow(
      /unknown fields/u,
    );
  });
});
