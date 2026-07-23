import { useEffect, useRef, useState } from "react";
import {
  validateDroppedPaths,
  type MediaDropAdapter,
  type MediaSelection,
} from "../bridge/media-drop";

interface NativeMediaDropOptions {
  onSelection: (selection: MediaSelection) => void;
  onError: (error: unknown) => void;
}

export interface NativeMediaDropState {
  dragging: boolean;
  resolving: boolean;
}

export function useNativeMediaDrop(
  adapter: MediaDropAdapter,
  { onSelection, onError }: NativeMediaDropOptions,
): NativeMediaDropState {
  const [dragging, setDragging] = useState(false);
  const [resolving, setResolving] = useState(false);
  const onSelectionRef = useRef(onSelection);
  const onErrorRef = useRef(onError);
  const operationRef = useRef(0);

  useEffect(() => {
    onSelectionRef.current = onSelection;
    onErrorRef.current = onError;
  }, [onError, onSelection]);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;

    adapter
      .listen((event) => {
        if (disposed) {
          return;
        }
        if (event.type === "enter") {
          setDragging(true);
          return;
        }
        if (event.type === "over") {
          return;
        }
        if (event.type === "leave") {
          setDragging(false);
          return;
        }

        setDragging(false);
        setResolving(true);
        const operation = operationRef.current + 1;
        operationRef.current = operation;

        Promise.resolve()
          .then(() => validateDroppedPaths(event.paths))
          .then(async (path) => await adapter.resolve(path))
          .then((selection) => {
            if (!disposed && operationRef.current === operation) {
              onSelectionRef.current(selection);
            }
          })
          .catch((error: unknown) => {
            if (!disposed && operationRef.current === operation) {
              onErrorRef.current(error);
            }
          })
          .finally(() => {
            if (!disposed && operationRef.current === operation) {
              setResolving(false);
            }
          });
      })
      .then((disposeListener) => {
        if (disposed) {
          disposeListener();
        } else {
          unlisten = disposeListener;
        }
      })
      .catch((error: unknown) => {
        if (!disposed) {
          onErrorRef.current(error);
        }
      });

    return () => {
      disposed = true;
      operationRef.current += 1;
      unlisten?.();
    };
  }, [adapter]);

  return { dragging, resolving };
}
