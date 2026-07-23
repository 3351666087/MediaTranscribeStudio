import type { JobRuntimeStatus } from "./studio";
import {
  ContractValidationError,
  parseJobRuntimeStatus,
  parseJobRuntimeStatuses,
} from "./runtime-validation";

function runtimeStatus(
  overrides: Partial<JobRuntimeStatus> = {},
): JobRuntimeStatus {
  return {
    jobId: "job-runtime-001",
    status: "running",
    revision: 3,
    acceptedByWorker: true,
    projected: true,
    inFlight: true,
    workerEventRouteRegistered: true,
    cancellable: true,
    volatileOnly: true,
    ...overrides,
  };
}

describe("job runtime status validation", () => {
  it("accepts the exact Rust task-list contract", () => {
    const statuses = [
      runtimeStatus(),
      runtimeStatus({
        jobId: "job-runtime-002",
        status: "completed",
        revision: 9,
        projected: false,
        inFlight: false,
        workerEventRouteRegistered: false,
        cancellable: false,
      }),
    ];

    expect(parseJobRuntimeStatuses(statuses)).toEqual(statuses);
  });

  it("rejects unknown fields rather than trusting an extended payload", () => {
    expect(() =>
      parseJobRuntimeStatus({
        ...runtimeStatus(),
        persisted: true,
      }),
    ).toThrow(/unknown fields/u);
  });

  it("rejects duplicate task IDs and multiple projected tasks", () => {
    expect(() =>
      parseJobRuntimeStatuses([
        runtimeStatus(),
        runtimeStatus({ projected: false }),
      ]),
    ).toThrow(/must be unique/u);

    expect(() =>
      parseJobRuntimeStatuses([
        runtimeStatus(),
        runtimeStatus({ jobId: "job-runtime-002" }),
      ]),
    ).toThrow(/more than one projected task/u);
  });

  it("rejects terminal tasks that claim to be cancellable", () => {
    expect(() =>
      parseJobRuntimeStatus(
        runtimeStatus({
          status: "completed",
          inFlight: false,
          workerEventRouteRegistered: false,
        }),
      ),
    ).toThrow(ContractValidationError);
    expect(() =>
      parseJobRuntimeStatus(
        runtimeStatus({
          status: "cancelled",
          inFlight: false,
          workerEventRouteRegistered: false,
        }),
      ),
    ).toThrow(/cancellable/u);
  });
});
