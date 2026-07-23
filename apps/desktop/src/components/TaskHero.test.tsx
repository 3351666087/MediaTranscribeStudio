import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { JobStatus, JobSummary } from "../contracts/studio";
import {
  createStudioFixture,
  modelStrategies,
} from "../mocks/studio-fixture";
import { TaskHero } from "./TaskHero";

const statusCases = [
  {
    status: "draft",
    title: "Configure a trustworthy offline transcription.",
    progressTitle: "Waiting for configuration",
    guardrail: "No acoustic evidence has been produced",
  },
  {
    status: "queued",
    title: "Your local job is waiting to start.",
    progressTitle: "Queued for local processing",
    guardrail: "Waiting for current-job worker evidence",
  },
  {
    status: "running",
    title: "Local acoustic analysis is in progress.",
    progressTitle: "Processing local evidence",
    guardrail: "Source evidence remains protected",
  },
  {
    status: "review_required",
    title: "Return every sentence to the right speaker.",
    progressTitle: "Waiting for human review",
    guardrail: "Unresolved decisions remain fail-closed",
  },
  {
    status: "completed",
    title: "Processing completed; verify the evidence.",
    progressTitle: "Local processing completed",
    guardrail: "Source and derived artifacts remain separate",
  },
  {
    status: "failed",
    title: "This local job needs attention.",
    progressTitle: "Processing failed",
    guardrail: "Existing source data is preserved",
  },
  {
    status: "cancelled",
    title: "This local job was cancelled.",
    progressTitle: "Processing cancelled",
    guardrail: "Existing source data is preserved",
  },
] as const satisfies ReadonlyArray<{
  status: JobStatus;
  title: string;
  progressTitle: string;
  guardrail: string;
}>;

function jobFor(status: JobStatus, reviewOpenCount = 0): JobSummary {
  return {
    ...createStudioFixture(5).job,
    status,
    reviewOpenCount,
    progress:
      status === "draft"
        ? 0
        : status === "queued"
          ? 4
          : status === "completed"
            ? 100
            : 62,
  };
}

describe("TaskHero job-status messaging", () => {
  it.each(statusCases)(
    "renders truthful $status copy",
    ({ status, title, progressTitle, guardrail }) => {
      render(
        <TaskHero
          job={jobFor(status)}
          strategy={modelStrategies[0]}
          onReview={vi.fn()}
        />,
      );

      expect(
        screen.getByRole("heading", { level: 1, name: title }),
      ).toBeInTheDocument();
      expect(screen.getByText(progressTitle)).toBeInTheDocument();
      expect(screen.getByText(guardrail)).toBeInTheDocument();
      expect(
        document.querySelector(`[data-job-status="${status}"]`),
      ).toBeInTheDocument();
      expect(
        screen.queryByText("Reviewing acoustic evidence"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByText("High-confidence CAM++ results are locked"),
      ).not.toBeInTheDocument();
    },
  );

  it("does not present auto speaker counts as verified before current-job evidence is ready", () => {
    const job = {
      ...jobFor("queued"),
      speakerPolicy: { mode: "auto" } as const,
      speakerCount: 99,
      speakerDetection: {
        estimatedCount: 99,
        confidence: 0.99,
        candidates: [{ count: 99, confidence: 0.99 }],
        provider: "stale provider",
      },
    };

    render(
      <TaskHero
        job={job}
        strategy={modelStrategies[0]}
        onReview={vi.fn()}
      />,
    );

    expect(
      screen.getByText(
        /Auto-detect · awaiting current-job worker evidence/u,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/99 speakers reported/u)).not.toBeInTheDocument();
  });

  it("offers review navigation only when unresolved evidence is actionable", async () => {
    const user = userEvent.setup();
    const onReview = vi.fn();
    const { rerender } = render(
      <TaskHero
        job={jobFor("review_required", 3)}
        strategy={modelStrategies[0]}
        onReview={onReview}
      />,
    );

    await user.click(
      screen.getByRole("button", {
        name: "Review 3 pending segments",
      }),
    );
    expect(onReview).toHaveBeenCalledTimes(1);

    rerender(
      <TaskHero
        job={jobFor("cancelled", 3)}
        strategy={modelStrategies[0]}
        onReview={onReview}
      />,
    );
    expect(
      screen.queryByRole("button", {
        name: /pending|unresolved/u,
      }),
    ).not.toBeInTheDocument();
  });
});
