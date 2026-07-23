import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, render, screen } from "@testing-library/react";
import { MediaDropOverlay } from "./MediaDropOverlay";

const copy: Readonly<Record<string, string>> = {
  "drop.eyebrow": "Native media drop",
  "drop.processingTitle": "Preparing media",
  "drop.processingDetail": "Checking the selected media.",
  "drop.readyTitle": "Drop media here",
  "drop.readyDetail": "Release to add media to the queue.",
  "drop.supportedTypes": "MOV, MP4, WAV, MP3",
};

vi.mock("../i18n", () => ({
  useI18n: () => ({
    t: (key: string) => copy[key] ?? key,
  }),
}));

afterEach(() => {
  vi.useRealTimers();
});

describe("MediaDropOverlay", () => {
  it("keeps the overlay mounted through a visible-to-exiting lifecycle", () => {
    vi.useFakeTimers();
    const { rerender } = render(
      <MediaDropOverlay visible={false} resolving={false} />,
    );

    expect(screen.queryByRole("status")).not.toBeInTheDocument();

    rerender(<MediaDropOverlay visible resolving={false} />);

    const mountedOverlay = screen.getByRole("status");
    expect(mountedOverlay).toHaveAttribute("data-lifecycle", "mounted");
    expect(mountedOverlay).toHaveAttribute("aria-atomic", "true");

    act(() => {
      vi.runOnlyPendingTimers();
    });
    expect(screen.getByRole("status")).toHaveAttribute(
      "data-lifecycle",
      "visible",
    );

    rerender(<MediaDropOverlay visible={false} resolving={false} />);

    const exitingOverlay = document.querySelector(".media-drop-overlay");
    expect(exitingOverlay).toHaveAttribute("data-lifecycle", "exiting");
    expect(exitingOverlay).toHaveAttribute("aria-hidden", "true");

    act(() => {
      vi.advanceTimersByTime(139);
    });
    expect(document.querySelector(".media-drop-overlay")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(document.querySelector(".media-drop-overlay")).not.toBeInTheDocument();
  });

  it("cancels an in-flight exit when drag activity resumes", () => {
    vi.useFakeTimers();
    const { rerender } = render(
      <MediaDropOverlay visible resolving={false} />,
    );

    act(() => {
      vi.runOnlyPendingTimers();
    });
    expect(screen.getByRole("status")).toHaveAttribute(
      "data-lifecycle",
      "visible",
    );

    rerender(<MediaDropOverlay visible={false} resolving={false} />);
    expect(document.querySelector(".media-drop-overlay")).toHaveAttribute(
      "data-lifecycle",
      "exiting",
    );

    act(() => {
      vi.advanceTimersByTime(70);
    });
    rerender(<MediaDropOverlay visible={false} resolving />);
    expect(screen.getByRole("status")).toHaveAttribute(
      "data-lifecycle",
      "mounted",
    );
    expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true");

    act(() => {
      vi.runOnlyPendingTimers();
    });
    expect(screen.getByRole("status")).toHaveAttribute(
      "data-lifecycle",
      "visible",
    );

    act(() => {
      vi.advanceTimersByTime(200);
    });
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("locks the stylesheet to bounded reduced motion and high-contrast rings", () => {
    const stylesheet = readFileSync(
      resolve(process.cwd(), "src/styles/global.css"),
      "utf8",
    );

    expect(stylesheet).toContain("--mts-motion-enter: 200ms;");
    expect(stylesheet).toContain("--mts-motion-exit: 140ms;");
    expect(stylesheet).toContain(
      "--mts-ease-enter: cubic-bezier(0.23, 1, 0.32, 1);",
    );
    expect(stylesheet).not.toContain("emil-reduced-pulse");
    expect(stylesheet).not.toMatch(
      /prefers-reduced-motion[\s\S]*?infinite/gu,
    );
    expect(stylesheet).toMatch(
      /\.progress-ring__track\s*\{[\s\S]*?stroke:\s*GrayText;/u,
    );
    expect(stylesheet).toMatch(
      /\.progress-ring__value\s*\{[\s\S]*?stroke:\s*Highlight;/u,
    );
    expect(stylesheet).toMatch(
      /\.progress-ring__label\s*\{[\s\S]*?color:\s*CanvasText;/u,
    );
  });
});
