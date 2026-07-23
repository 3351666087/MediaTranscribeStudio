import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
} from "react";
import "./DesktopCompanion.css";

const companionSource = new URL("../assets/companion.gif", import.meta.url).href;
const dragThresholdSquared = 36;
const edgeDamping = 0.24;
const keyboardStep = 12;
const keyboardLargeStep = 32;

interface Point {
  x: number;
  y: number;
}

interface ViewportBounds {
  maxX: number;
  maxY: number;
  minX: number;
  minY: number;
}

export interface DesktopCompanionProps {
  collapseLabel: string;
  expandLabel: string;
  imageAlt: string;
  regionLabel: string;
  statusText: string;
  boundaryPadding?: number;
  initialCollapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
}

function getViewportBounds(
  element: HTMLElement,
  boundaryPadding: number,
): ViewportBounds {
  const rect = element.getBoundingClientRect();
  const padding = Math.max(0, boundaryPadding);

  return {
    maxX: Math.max(padding, window.innerWidth - rect.width - padding),
    maxY: Math.max(padding, window.innerHeight - rect.height - padding),
    minX: padding,
    minY: padding,
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function clampPoint(point: Point, bounds: ViewportBounds): Point {
  return {
    x: clamp(point.x, bounds.minX, bounds.maxX),
    y: clamp(point.y, bounds.minY, bounds.maxY),
  };
}

function damp(value: number, minimum: number, maximum: number): number {
  if (value < minimum) {
    return minimum + (value - minimum) * edgeDamping;
  }

  if (value > maximum) {
    return maximum + (value - maximum) * edgeDamping;
  }

  return value;
}

function dampPoint(point: Point, bounds: ViewportBounds): Point {
  return {
    x: damp(point.x, bounds.minX, bounds.maxX),
    y: damp(point.y, bounds.minY, bounds.maxY),
  };
}

export function DesktopCompanion({
  boundaryPadding = 20,
  collapseLabel,
  expandLabel,
  imageAlt,
  initialCollapsed = false,
  onCollapsedChange,
  regionLabel,
  statusText,
}: DesktopCompanionProps) {
  const [collapsed, setCollapsed] = useState(initialCollapsed);
  const [dragging, setDragging] = useState(false);
  const [keyboardMoving, setKeyboardMoving] = useState(false);
  const [position, setPosition] = useState<Point | null>(null);
  const rootRef = useRef<HTMLElement>(null);
  const positionRef = useRef<Point | null>(null);
  const activePointerRef = useRef<number | null>(null);
  const grabOffsetRef = useRef<Point>({ x: 0, y: 0 });
  const pointerOriginRef = useRef<Point>({ x: 0, y: 0 });
  const didDragRef = useRef(false);
  const suppressClickRef = useRef(false);
  const clickResetTimerRef = useRef<number | null>(null);
  const keyboardFrameRef = useRef<number | null>(null);
  const statusId = useId();

  const applyPosition = useCallback((nextPosition: Point) => {
    positionRef.current = nextPosition;
    setPosition(nextPosition);
  }, []);

  const keepOnScreen = useCallback(() => {
    const root = rootRef.current;
    const currentPosition = positionRef.current;

    if (root === null || currentPosition === null) {
      return;
    }

    applyPosition(
      clampPoint(currentPosition, getViewportBounds(root, boundaryPadding)),
    );
  }, [applyPosition, boundaryPadding]);

  useLayoutEffect(() => {
    keepOnScreen();
  }, [collapsed, keepOnScreen]);

  useEffect(() => {
    window.addEventListener("resize", keepOnScreen);

    return () => {
      window.removeEventListener("resize", keepOnScreen);

      if (clickResetTimerRef.current !== null) {
        window.clearTimeout(clickResetTimerRef.current);
      }

      if (keyboardFrameRef.current !== null) {
        window.cancelAnimationFrame(keyboardFrameRef.current);
      }
    };
  }, [keepOnScreen]);

  const toggleCollapsed = () => {
    const nextCollapsed = !collapsed;
    setCollapsed(nextCollapsed);
    onCollapsedChange?.(nextCollapsed);
  };

  const handlePointerDown = (event: PointerEvent<HTMLButtonElement>) => {
    if (
      activePointerRef.current !== null ||
      !event.isPrimary ||
      (event.pointerType === "mouse" && event.button !== 0)
    ) {
      return;
    }

    const root = rootRef.current;
    if (root === null) {
      return;
    }

    const rect = root.getBoundingClientRect();
    const nextPosition = { x: rect.left, y: rect.top };

    activePointerRef.current = event.pointerId;
    grabOffsetRef.current = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
    pointerOriginRef.current = {
      x: event.clientX,
      y: event.clientY,
    };
    didDragRef.current = false;
    suppressClickRef.current = false;
    setKeyboardMoving(false);
    applyPosition(nextPosition);
    setDragging(true);

    if (typeof event.currentTarget.setPointerCapture === "function") {
      event.currentTarget.setPointerCapture(event.pointerId);
    }
  };

  const handlePointerMove = (event: PointerEvent<HTMLButtonElement>) => {
    if (activePointerRef.current !== event.pointerId) {
      return;
    }

    const root = rootRef.current;
    if (root === null) {
      return;
    }

    const horizontalTravel = event.clientX - pointerOriginRef.current.x;
    const verticalTravel = event.clientY - pointerOriginRef.current.y;

    if (
      horizontalTravel * horizontalTravel +
        verticalTravel * verticalTravel >
      dragThresholdSquared
    ) {
      didDragRef.current = true;
    }

    applyPosition(
      dampPoint(
        {
          x: event.clientX - grabOffsetRef.current.x,
          y: event.clientY - grabOffsetRef.current.y,
        },
        getViewportBounds(root, boundaryPadding),
      ),
    );
    event.preventDefault();
  };

  const finishPointerInteraction = (
    event: PointerEvent<HTMLButtonElement>,
    cancelled: boolean,
  ) => {
    if (activePointerRef.current !== event.pointerId) {
      return;
    }

    const root = rootRef.current;
    const currentPosition = positionRef.current;

    activePointerRef.current = null;
    setDragging(false);

    if (root !== null && currentPosition !== null) {
      applyPosition(
        clampPoint(currentPosition, getViewportBounds(root, boundaryPadding)),
      );
    }

    if (
      typeof event.currentTarget.hasPointerCapture === "function" &&
      event.currentTarget.hasPointerCapture(event.pointerId) &&
      typeof event.currentTarget.releasePointerCapture === "function"
    ) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    if (didDragRef.current || cancelled) {
      suppressClickRef.current = true;

      if (clickResetTimerRef.current !== null) {
        window.clearTimeout(clickResetTimerRef.current);
      }

      clickResetTimerRef.current = window.setTimeout(() => {
        suppressClickRef.current = false;
        clickResetTimerRef.current = null;
      }, 0);
    }
  };

  const handleClick = () => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }

    toggleCollapsed();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const directions: Partial<Record<string, Point>> = {
      ArrowDown: { x: 0, y: 1 },
      ArrowLeft: { x: -1, y: 0 },
      ArrowRight: { x: 1, y: 0 },
      ArrowUp: { x: 0, y: -1 },
    };
    const direction = directions[event.key];

    if (direction === undefined) {
      return;
    }

    const root = rootRef.current;
    if (root === null) {
      return;
    }

    const rect = root.getBoundingClientRect();
    const currentPosition = positionRef.current ?? {
      x: rect.left,
      y: rect.top,
    };
    const step = event.shiftKey ? keyboardLargeStep : keyboardStep;

    setKeyboardMoving(true);
    applyPosition(
      clampPoint(
        {
          x: currentPosition.x + direction.x * step,
          y: currentPosition.y + direction.y * step,
        },
        getViewportBounds(root, boundaryPadding),
      ),
    );

    if (keyboardFrameRef.current !== null) {
      window.cancelAnimationFrame(keyboardFrameRef.current);
    }
    keyboardFrameRef.current = window.requestAnimationFrame(() => {
      setKeyboardMoving(false);
      keyboardFrameRef.current = null;
    });
    event.preventDefault();
  };

  const positionStyle =
    position === null
      ? undefined
      : ({
          transform: `translate3d(${position.x}px, ${position.y}px, 0)`,
        } as CSSProperties);
  const statusVisible = !collapsed && statusText.length > 0;
  const stateClass = collapsed ? "is-collapsed" : "is-expanded";

  return (
    <aside
      ref={rootRef}
      aria-label={regionLabel}
      className={[
        "desktop-companion",
        stateClass,
        dragging ? "is-dragging" : "",
        keyboardMoving ? "is-keyboard-moving" : "",
        position === null ? "" : "is-positioned",
      ]
        .filter(Boolean)
        .join(" ")}
      data-state={collapsed ? "collapsed" : "expanded"}
      style={positionStyle}
    >
      {statusVisible ? (
        <span
          className="desktop-companion__status"
          id={statusId}
          role="status"
          aria-live="polite"
        >
          {statusText}
        </span>
      ) : null}

      <button
        className="desktop-companion__button"
        type="button"
        aria-describedby={statusVisible ? statusId : undefined}
        aria-expanded={!collapsed}
        aria-keyshortcuts="ArrowUp ArrowDown ArrowLeft ArrowRight Enter Space"
        aria-label={collapsed ? expandLabel : collapseLabel}
        onClick={handleClick}
        onDragStart={(event) => {
          event.preventDefault();
        }}
        onKeyDown={handleKeyDown}
        onLostPointerCapture={(event) => {
          finishPointerInteraction(event, true);
        }}
        onPointerCancel={(event) => {
          finishPointerInteraction(event, true);
        }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={(event) => {
          finishPointerInteraction(event, false);
        }}
      >
        <span className="desktop-companion__aura" aria-hidden="true" />
        <span className="desktop-companion__visual">
          <img
            className="desktop-companion__image"
            src={companionSource}
            alt={imageAlt}
            draggable={false}
          />
        </span>
        <span className="desktop-companion__glint" aria-hidden="true" />
      </button>
    </aside>
  );
}
