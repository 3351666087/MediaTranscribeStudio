import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { DesktopCompanion } from "./DesktopCompanion";

const labels = {
  collapseLabel: "Collapse companion",
  expandLabel: "Expand companion",
  imageAlt: "Animated local companion",
  regionLabel: "Desktop companion",
  statusText: "Local processing is ready",
};

function createRect({
  height,
  left,
  top,
  width,
}: {
  height: number;
  left: number;
  top: number;
  width: number;
}): DOMRect {
  return {
    bottom: top + height,
    height,
    left,
    right: left + width,
    top,
    width,
    x: left,
    y: top,
    toJSON: () => ({}),
  };
}

function renderCompanion(
  overrides: Partial<ComponentProps<typeof DesktopCompanion>> = {},
) {
  return render(<DesktopCompanion {...labels} {...overrides} />);
}

function dispatchPointer(
  element: Element,
  type: string,
  {
    button = 0,
    clientX,
    clientY,
    isPrimary = true,
    pointerId,
    pointerType,
  }: {
    button?: number;
    clientX: number;
    clientY: number;
    isPrimary?: boolean;
    pointerId: number;
    pointerType: string;
  },
) {
  const event = new MouseEvent(type, {
    bubbles: true,
    button,
    cancelable: true,
    clientX,
    clientY,
  });

  Object.defineProperties(event, {
    isPrimary: {
      configurable: true,
      value: isPrimary,
    },
    pointerId: {
      configurable: true,
      value: pointerId,
    },
    pointerType: {
      configurable: true,
      value: pointerType,
    },
  });

  fireEvent(element, event);
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("DesktopCompanion", () => {
  it("renders the bundled GIF and receives every visible label through props", () => {
    renderCompanion();

    const region = screen.getByRole("complementary", {
      name: labels.regionLabel,
    });
    const toggle = screen.getByRole("button", {
      name: labels.collapseLabel,
    });

    expect(region).toHaveAttribute("data-state", "expanded");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(toggle).toHaveAttribute(
      "aria-keyshortcuts",
      "ArrowUp ArrowDown ArrowLeft ArrowRight Enter Space",
    );
    expect(screen.getByRole("status")).toHaveTextContent(labels.statusText);
    expect(screen.getByAltText(labels.imageAlt)).toHaveAttribute(
      "src",
      expect.stringContaining("companion.gif"),
    );
  });

  it("collapses and expands on click while reporting state changes", () => {
    const onCollapsedChange = vi.fn();
    renderCompanion({ onCollapsedChange });

    fireEvent.click(
      screen.getByRole("button", { name: labels.collapseLabel }),
    );

    expect(
      screen.getByRole("button", { name: labels.expandLabel }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.getByRole("complementary", { name: labels.regionLabel }),
    ).toHaveAttribute("data-state", "collapsed");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(onCollapsedChange).toHaveBeenLastCalledWith(true);

    fireEvent.click(
      screen.getByRole("button", { name: labels.expandLabel }),
    );

    expect(
      screen.getByRole("button", { name: labels.collapseLabel }),
    ).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("status")).toHaveTextContent(labels.statusText);
    expect(onCollapsedChange).toHaveBeenLastCalledWith(false);
  });

  it("preserves the grab offset, damps overflow, captures one pointer, and clamps on release", () => {
    vi.useFakeTimers();
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(1000);
    vi.spyOn(window, "innerHeight", "get").mockReturnValue(700);
    renderCompanion({ boundaryPadding: 20 });

    const region = screen.getByRole("complementary", {
      name: labels.regionLabel,
    });
    const toggle = screen.getByRole("button", {
      name: labels.collapseLabel,
    });
    vi.spyOn(region, "getBoundingClientRect").mockReturnValue(
      createRect({ height: 120, left: 700, top: 500, width: 280 }),
    );
    const setPointerCapture = vi.fn();
    const releasePointerCapture = vi.fn();
    Object.defineProperties(toggle, {
      hasPointerCapture: {
        configurable: true,
        value: vi.fn(() => true),
      },
      releasePointerCapture: {
        configurable: true,
        value: releasePointerCapture,
      },
      setPointerCapture: {
        configurable: true,
        value: setPointerCapture,
      },
    });

    dispatchPointer(toggle, "pointerdown", {
      button: 0,
      clientX: 850,
      clientY: 590,
      pointerId: 7,
      pointerType: "touch",
    });
    dispatchPointer(toggle, "pointermove", {
      clientX: 940,
      clientY: 650,
      pointerId: 7,
      pointerType: "touch",
    });

    expect(setPointerCapture).toHaveBeenCalledWith(7);
    expect(region).toHaveClass("is-dragging");
    expect(region).toHaveStyle({
      transform: "translate3d(721.6px, 560px, 0)",
    });

    dispatchPointer(toggle, "pointerup", {
      clientX: 940,
      clientY: 650,
      pointerId: 7,
      pointerType: "touch",
    });

    expect(releasePointerCapture).toHaveBeenCalledWith(7);
    expect(region).not.toHaveClass("is-dragging");
    expect(region).toHaveStyle({
      transform: "translate3d(700px, 560px, 0)",
    });

    fireEvent.click(toggle);
    expect(region).toHaveAttribute("data-state", "expanded");
  });

  it("ignores additional pointers until the active pointer finishes", () => {
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(1200);
    vi.spyOn(window, "innerHeight", "get").mockReturnValue(800);
    renderCompanion();

    const region = screen.getByRole("complementary", {
      name: labels.regionLabel,
    });
    const toggle = screen.getByRole("button", {
      name: labels.collapseLabel,
    });
    vi.spyOn(region, "getBoundingClientRect").mockReturnValue(
      createRect({ height: 104, left: 900, top: 620, width: 104 }),
    );
    const setPointerCapture = vi.fn();
    Object.defineProperty(toggle, "setPointerCapture", {
      configurable: true,
      value: setPointerCapture,
    });

    dispatchPointer(toggle, "pointerdown", {
      button: 0,
      clientX: 940,
      clientY: 660,
      pointerId: 1,
      pointerType: "touch",
    });
    dispatchPointer(toggle, "pointerdown", {
      button: 0,
      clientX: 960,
      clientY: 680,
      isPrimary: false,
      pointerId: 2,
      pointerType: "touch",
    });
    dispatchPointer(toggle, "pointermove", {
      clientX: 200,
      clientY: 200,
      pointerId: 2,
      pointerType: "touch",
    });

    expect(setPointerCapture).toHaveBeenCalledTimes(1);
    expect(setPointerCapture).toHaveBeenCalledWith(1);
    expect(region).toHaveStyle({
      transform: "translate3d(900px, 620px, 0)",
    });
  });

  it("moves with arrow keys, uses a larger shift step, and stays inside the viewport", () => {
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(900);
    vi.spyOn(window, "innerHeight", "get").mockReturnValue(700);
    renderCompanion({ boundaryPadding: 20 });

    const region = screen.getByRole("complementary", {
      name: labels.regionLabel,
    });
    const toggle = screen.getByRole("button", {
      name: labels.collapseLabel,
    });
    vi.spyOn(region, "getBoundingClientRect").mockReturnValue(
      createRect({ height: 104, left: 760, top: 560, width: 104 }),
    );

    fireEvent.keyDown(toggle, { key: "ArrowLeft" });
    fireEvent.keyDown(toggle, { key: "ArrowUp", shiftKey: true });

    expect(region).toHaveStyle({
      transform: "translate3d(748px, 528px, 0)",
    });
    expect(region).toHaveClass("is-keyboard-moving");

    for (let index = 0; index < 100; index += 1) {
      fireEvent.keyDown(toggle, { key: "ArrowRight", shiftKey: true });
      fireEvent.keyDown(toggle, { key: "ArrowDown", shiftKey: true });
    }

    expect(region).toHaveStyle({
      transform: "translate3d(776px, 576px, 0)",
    });
  });

  it("keeps motion accessibility and pointer-specific hover rules in the component stylesheet", () => {
    const stylesheet = readFileSync(
      resolve("src/components/DesktopCompanion.css"),
      "utf8",
    );

    expect(stylesheet).toContain("@media (prefers-reduced-motion: reduce)");
    expect(stylesheet).toMatch(
      /@media \(prefers-reduced-motion: reduce\)[\s\S]*transition: none !important;/u,
    );
    expect(stylesheet).toContain(
      "@media (hover: hover) and (pointer: fine)",
    );
    expect(stylesheet).toMatch(
      /\.desktop-companion\.is-keyboard-moving[\s\S]*transition: none/u,
    );
  });
});
