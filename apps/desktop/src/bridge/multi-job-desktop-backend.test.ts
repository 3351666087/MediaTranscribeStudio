import type { CreateJobRequest } from "../contracts/studio";
import { MockDesktopBackend } from "./desktop-backend";

function createRequest(index: number): CreateJobRequest {
  return {
    title: `Concurrent task ${index}`,
    mediaPath: `D:\\Media\\session-${index}.mov`,
    outputDirectory: `D:\\Output\\session-${index}`,
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
    summary: false,
    outputLocale: "en-US",
    businessPromptVersion: "business-v1",
  };
}

async function advance<T>(
  promise: Promise<T>,
  milliseconds = 400,
): Promise<T> {
  const observed = promise.then(
    (value) => ({ status: "fulfilled" as const, value }),
    (reason: unknown) => ({ status: "rejected" as const, reason }),
  );
  await vi.advanceTimersByTimeAsync(milliseconds);
  const result = await observed;
  if (result.status === "rejected") {
    throw result.reason;
  }
  return result.value;
}

describe("MockDesktopBackend multi-task registry", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps concurrent siblings registered and projects the last accepted task", async () => {
    const backend = new MockDesktopBackend();
    const firstPromise = backend.createJob(createRequest(1));
    const secondPromise = backend.createJob(createRequest(2));

    const [first, second] = await advance(
      Promise.all([firstPromise, secondPromise]),
    );
    const jobs = await advance(backend.listJobs(), 100);

    expect(first.jobId).not.toBe(second.jobId);
    expect(jobs.map(({ jobId }) => jobId)).toEqual(
      expect.arrayContaining([
        "job-demo-001",
        first.jobId,
        second.jobId,
      ]),
    );
    expect(jobs).toHaveLength(3);
    expect(jobs.find(({ projected }) => projected)?.jobId).toBe(
      second.jobId,
    );
  });

  it("switches snapshots without losing siblings", async () => {
    const backend = new MockDesktopBackend();
    const first = await advance(backend.createJob(createRequest(1)));
    const second = await advance(backend.createJob(createRequest(2)));

    const selected = await advance(backend.selectJob(first.jobId), 100);
    const jobs = await advance(backend.listJobs(), 100);

    expect(selected.job).toMatchObject({
      id: first.jobId,
      title: "Concurrent task 1",
    });
    expect(jobs).toHaveLength(3);
    expect(jobs.find(({ jobId }) => jobId === first.jobId)?.projected).toBe(
      true,
    );
    expect(jobs.some(({ jobId }) => jobId === second.jobId)).toBe(true);
  });

  it("cancels a non-current task without changing the projection", async () => {
    const backend = new MockDesktopBackend();
    const first = await advance(backend.createJob(createRequest(1)));
    const second = await advance(backend.createJob(createRequest(2)));
    await advance(backend.selectJob(first.jobId), 100);

    await advance(backend.cancelJob(second.jobId), 220);
    const jobs = await advance(backend.listJobs(), 100);

    expect(jobs.find(({ jobId }) => jobId === second.jobId)).toMatchObject({
      status: "cancelled",
      projected: false,
      cancellable: false,
    });
    expect(jobs.find(({ projected }) => projected)?.jobId).toBe(first.jobId);
    await expect(
      advance(backend.getSnapshot(), 100),
    ).resolves.toMatchObject({
      job: { id: first.jobId },
    });
  });
});
