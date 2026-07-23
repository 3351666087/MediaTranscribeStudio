import { useEffect, useRef, useState } from "react";
import {
  resolveMediaDropBatch,
  type MediaDropAdapter,
  type MediaDropBatchResult,
} from "../bridge/media-drop";

interface NativeMediaDropOptions {
  onSelection: (result: MediaDropBatchResult) => void;
  onError: (error: unknown) => void;
}

export interface NativeMediaDropState {
  dragging: boolean;
  resolving: boolean;
}

export function useNativeMediaDrop(
  adapter: MediaDropAdapter,
  options: NativeMediaDropOptions,
): NativeMediaDropState {
  const [dragging, setDragging] = useState(false);
  const [resolving, setResolving] = useState(false);
  const callbacksRef = useRef<NativeMediaDropOptions>(options);
  const operationRef = useRef(0);

  useEffect(() => {
    callbacksRef.current = options;
  }, [options]);

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
          .then(async () => await resolveMediaDropBatch(adapter, event.paths))
          .then((result) => {
            if (!disposed && operationRef.current === operation) {
              callbacksRef.current.onSelection(result);
            }
          })
          .catch((error: unknown) => {
            if (!disposed && operationRef.current === operation) {
              callbacksRef.current.onError(error);
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
          callbacksRef.current.onError(error);
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
