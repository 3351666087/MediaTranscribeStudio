import type {
  CreateJobRequest,
  CreateJobResult,
} from "../contracts/studio";

export const MIN_BATCH_CONCURRENCY = 1;
export const MAX_BATCH_CONCURRENCY = 4;
export const DEFAULT_BATCH_CONCURRENCY = 2;

export type CreateJobFunction = (
  request: CreateJobRequest,
) => Promise<CreateJobResult>;

export interface AcceptedCreateJobBatchItemResult {
  index: number;
  request: CreateJobRequest;
  status: "accepted";
  result: CreateJobResult;
}

export interface FailedCreateJobBatchItemResult {
  index: number;
  request: CreateJobRequest;
  status: "failed";
  result?: CreateJobResult;
  error: string;
}

export type CreateJobBatchItemResult =
  | AcceptedCreateJobBatchItemResult
  | FailedCreateJobBatchItemResult;

export interface CreateJobBatchResult {
  items: CreateJobBatchItemResult[];
  acceptedCount: number;
  failedCount: number;
}

export function clampBatchConcurrency(concurrency: number): number {
  if (Number.isNaN(concurrency)) {
    return DEFAULT_BATCH_CONCURRENCY;
  }

  return Math.min(
    MAX_BATCH_CONCURRENCY,
    Math.max(MIN_BATCH_CONCURRENCY, Math.trunc(concurrency)),
  );
}

function describeFailure(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }

  if (typeof error === "string" && error.trim().length > 0) {
    return error;
  }

  return "The job could not be created.";
}

export async function createJobBatch(
  requests: readonly CreateJobRequest[],
  create: CreateJobFunction,
  concurrency = DEFAULT_BATCH_CONCURRENCY,
): Promise<CreateJobBatchResult> {
  const batch = [...requests];
  if (batch.length === 0) {
    return {
      items: [],
      acceptedCount: 0,
      failedCount: 0,
    };
  }

  const limit = clampBatchConcurrency(concurrency);
  const workerCount = Math.min(limit, batch.length);
  const items = new Array<CreateJobBatchItemResult>(batch.length);
  let nextIndex = 0;

  async function runWorker(): Promise<void> {
    while (nextIndex < batch.length) {
      const index = nextIndex;
      nextIndex += 1;
      const request = batch[index];

      try {
        const result = await create(request);
        if (result.accepted) {
          items[index] = {
            index,
            request,
            status: "accepted",
            result,
          };
        } else {
          items[index] = {
            index,
            request,
            status: "failed",
            result,
            error:
              result.message.trim().length > 0
                ? result.message
                : "The job was not accepted.",
          };
        }
      } catch (error: unknown) {
        items[index] = {
          index,
          request,
          status: "failed",
          error: describeFailure(error),
        };
      }
    }
  }

  const workers: Array<Promise<void>> = [];
  for (let workerIndex = 0; workerIndex < workerCount; workerIndex += 1) {
    workers.push(runWorker());
  }
  await Promise.all(workers);

  const acceptedCount = items.filter(
    (item) => item.status === "accepted",
  ).length;

  return {
    items,
    acceptedCount,
    failedCount: items.length - acceptedCount,
  };
}
