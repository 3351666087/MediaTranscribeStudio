import type {
  CreateJobRequest,
  CreateJobResult,
} from "../contracts/studio";
import {
  DEFAULT_BATCH_CONCURRENCY,
  MAX_BATCH_CONCURRENCY,
  MIN_BATCH_CONCURRENCY,
  clampBatchConcurrency,
  createJobBatch,
  type CreateJobFunction,
} from "./batch-coordinator";

interface Deferred {
  promise: Promise<void>;
  resolve: () => void;
}

function createDeferred(): Deferred {
  let resolvePromise!: () => void;
  const promise = new Promise<void>((resolve) => {
    resolvePromise = resolve;
  });
  return {
    promise,
    resolve: resolvePromise,
  };
}

function createRequest(index: number): CreateJobRequest {
  return {
    title: `Batch job ${index}`,
    mediaPath: `D:\\media\\meeting-${index}.mov`,
    outputDirectory: `D:\\output\\meeting-${index}`,
    strategyId: "balanced",
    speakerPolicy: {
      mode: "auto",
    },
    speakerLabels: [],
    language: "zh-CN",
    localLlmMode: "disabled",
    localLlmModel: "",
    localLlmEndpoint: "http://127.0.0.1:11434",
    localLlmEndpointPolicy: "loopback-only",
    localLlmAutoApply: false,
    translationTargets: [],
    polish: false,
    summary: false,
    outputLocale: "zh-CN",
    businessPromptVersion: "business-v1",
  };
}

function acceptedResult(index: number): CreateJobResult {
  return {
    accepted: true,
    jobId: `job-${index}`,
    message: "Accepted.",
  };
}

describe("frontend batch job coordinator", () => {
  it("returns an empty result without calling create for an empty batch", async () => {
    const create = vi.fn<CreateJobFunction>();

    await expect(createJobBatch([], create)).resolves.toEqual({
      items: [],
      acceptedCount: 0,
      failedCount: 0,
    });
    expect(create).not.toHaveBeenCalled();
  });

  it("bounds concurrency and preserves request order after out-of-order completion", async () => {
    const requests = Array.from({ length: 5 }, (_, index) =>
      createRequest(index),
    );
    const gates = requests.map(() => createDeferred());
    let activeCount = 0;
    let maximumActiveCount = 0;

    const create = vi.fn<CreateJobFunction>(async (request) => {
      const index = requests.indexOf(request);
      activeCount += 1;
      maximumActiveCount = Math.max(maximumActiveCount, activeCount);
      await gates[index].promise;
      activeCount -= 1;
      return acceptedResult(index);
    });

    const batchPromise = createJobBatch(requests, create);

    await vi.waitFor(() => {
      expect(create).toHaveBeenCalledTimes(DEFAULT_BATCH_CONCURRENCY);
    });
    expect(maximumActiveCount).toBe(DEFAULT_BATCH_CONCURRENCY);

    gates[1].resolve();
    await vi.waitFor(() => {
      expect(create).toHaveBeenCalledTimes(3);
    });
    gates[2].resolve();
    await vi.waitFor(() => {
      expect(create).toHaveBeenCalledTimes(4);
    });
    gates[3].resolve();
    await vi.waitFor(() => {
      expect(create).toHaveBeenCalledTimes(5);
    });
    gates[4].resolve();
    gates[0].resolve();

    const result = await batchPromise;

    expect(maximumActiveCount).toBe(DEFAULT_BATCH_CONCURRENCY);
    expect(result.items.map((item) => item.request)).toEqual(requests);
    expect(
      result.items.map((item) =>
        item.status === "accepted" ? item.result.jobId : item.error,
      ),
    ).toEqual(["job-0", "job-1", "job-2", "job-3", "job-4"]);
    expect(result.acceptedCount).toBe(5);
    expect(result.failedCount).toBe(0);
  });

  it("preserves successes across rejected, synchronously thrown, and declined items", async () => {
    const requests = Array.from({ length: 4 }, (_, index) =>
      createRequest(index),
    );
    const create: CreateJobFunction = vi.fn(
      // Intentionally non-async: this verifies a throw before any Promise exists.
      // eslint-disable-next-line @typescript-eslint/promise-function-async
      (request: CreateJobRequest): Promise<CreateJobResult> => {
        const index = requests.indexOf(request);
        if (index === 1) {
          return Promise.reject(new Error("Worker rejected the request."));
        }
        if (index === 2) {
          throw new Error("Synchronous adapter failure.");
        }
        if (index === 3) {
          return Promise.resolve({
            accepted: false,
            jobId: "",
            message: "Backend declined the request.",
          });
        }
        return Promise.resolve(acceptedResult(index));
      },
    );

    const result = await createJobBatch(requests, create, 4);

    expect(result.acceptedCount).toBe(1);
    expect(result.failedCount).toBe(3);
    expect(result.items.map((item) => item.status)).toEqual([
      "accepted",
      "failed",
      "failed",
      "failed",
    ]);
    expect(result.items[0]).toMatchObject({
      status: "accepted",
      result: {
        jobId: "job-0",
      },
    });
    expect(result.items[1]).toMatchObject({
      status: "failed",
      error: "Worker rejected the request.",
    });
    expect(result.items[2]).toMatchObject({
      status: "failed",
      error: "Synchronous adapter failure.",
    });
    expect(result.items[3]).toMatchObject({
      status: "failed",
      error: "Backend declined the request.",
      result: {
        accepted: false,
      },
    });
  });

  it("invokes each request once and never retries an accepted item after another item fails", async () => {
    const requests = [createRequest(0), createRequest(1), createRequest(2)];
    const attempts = new Map<string, number>();
    const create = vi.fn<CreateJobFunction>(async (request) => {
      attempts.set(
        request.mediaPath,
        (attempts.get(request.mediaPath) ?? 0) + 1,
      );
      if (request === requests[1]) {
        throw new Error("Expected isolated failure.");
      }
      return await Promise.resolve(
        acceptedResult(requests.indexOf(request)),
      );
    });

    const result = await createJobBatch(requests, create, 2);

    expect(result.acceptedCount).toBe(2);
    expect(result.failedCount).toBe(1);
    expect(create).toHaveBeenCalledTimes(requests.length);
    expect([...attempts.values()]).toEqual([1, 1, 1]);
  });

  it("clamps invalid concurrency values to the supported range", () => {
    expect(clampBatchConcurrency(0)).toBe(MIN_BATCH_CONCURRENCY);
    expect(clampBatchConcurrency(-20)).toBe(MIN_BATCH_CONCURRENCY);
    expect(clampBatchConcurrency(99)).toBe(MAX_BATCH_CONCURRENCY);
    expect(clampBatchConcurrency(Number.POSITIVE_INFINITY)).toBe(
      MAX_BATCH_CONCURRENCY,
    );
    expect(clampBatchConcurrency(Number.NaN)).toBe(
      DEFAULT_BATCH_CONCURRENCY,
    );
    expect(clampBatchConcurrency(3.9)).toBe(3);
  });
});
