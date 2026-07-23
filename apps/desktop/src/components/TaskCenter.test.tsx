import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { StudioJobItem } from "../contracts/studio";
import { TaskCenter } from "./TaskCenter";

function job(
  jobId: string,
  overrides: Partial<StudioJobItem> = {},
): StudioJobItem {
  return {
    jobId,
    status: "running",
    revision: 1,
    acceptedByWorker: true,
    projected: false,
    inFlight: true,
    workerEventRouteRegistered: true,
    cancellable: true,
    volatileOnly: true,
    title: `Task ${jobId}`,
    sourcePath: `D:\\Media\\${jobId}.mov`,
    progress: 42,
    ...overrides,
  };
}

describe("TaskCenter", () => {
  it("renders multiple tasks and marks the current task", () => {
    render(
      <TaskCenter
        jobs={[
          job("job-a", { projected: true }),
          job("job-b", {
            status: "completed",
            inFlight: false,
            workerEventRouteRegistered: false,
            cancellable: false,
            progress: 100,
          }),
        ]}
        selectedJobId="job-a"
        busyAction={null}
        onSelect={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(
      screen.getByLabelText("Tasks · 2"),
    ).toHaveTextContent("2");
    const current = screen.getByRole("button", {
      name: "Open Task job-a",
    });
    expect(current).toBeDisabled();
    expect(current).toHaveAttribute("aria-current", "true");
    expect(screen.getByText("Current task")).toBeInTheDocument();
    expect(
      screen.getByRole("progressbar", { name: "42% complete" }),
    ).toHaveAttribute("aria-valuenow", "42");
    expect(
      screen.queryByRole("button", { name: "Cancel Task job-b" }),
    ).not.toBeInTheDocument();
  });

  it("switches and cancels exact task IDs, including a non-current task", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn().mockResolvedValue(undefined);
    const onCancel = vi.fn().mockResolvedValue(undefined);
    render(
      <TaskCenter
        jobs={[
          job("job-a", { projected: true }),
          job("job-b", { progress: null }),
        ]}
        selectedJobId="job-a"
        busyAction={null}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
    );

    const trigger = screen.getByLabelText("Tasks · 2");
    const details = trigger.closest("details");
    expect(details).not.toBeNull();
    await user.click(trigger);
    expect(details).toHaveAttribute("open");

    await user.click(
      screen.getByRole("button", { name: "Cancel Task job-b" }),
    );
    expect(onCancel).toHaveBeenCalledWith("job-b");
    expect(onSelect).not.toHaveBeenCalled();

    await user.click(
      screen.getByRole("button", { name: "Open Task job-b" }),
    );
    await waitFor(() => expect(onSelect).toHaveBeenCalledWith("job-b"));
    await waitFor(() => expect(details).not.toHaveAttribute("open"));
    expect(screen.getByText("Progress pending")).toBeInTheDocument();
  });

  it("keeps the panel open when selection fails", async () => {
    const user = userEvent.setup();
    const onSelect = vi
      .fn()
      .mockRejectedValue(new Error("Selection rejected."));
    render(
      <TaskCenter
        jobs={[job("job-a"), job("job-b")]}
        selectedJobId="job-a"
        busyAction={null}
        onSelect={onSelect}
        onCancel={vi.fn()}
      />,
    );

    const trigger = screen.getByLabelText("Tasks · 2");
    const details = trigger.closest("details");
    await user.click(trigger);
    await user.click(
      screen.getByRole("button", { name: "Open Task job-b" }),
    );

    await waitFor(() => expect(onSelect).toHaveBeenCalledWith("job-b"));
    expect(details).toHaveAttribute("open");
  });
});
