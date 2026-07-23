import { act, renderHook, waitFor } from "@testing-library/react";
import {
  ControlledMediaDropAdapter,
  MediaDropError,
  type MediaSelection,
} from "../bridge/media-drop";
import { useNativeMediaDrop } from "./use-native-media-drop";

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolvePromise!: (value: T) => void;
  let rejectPromise!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolve, reject) => {
    resolvePromise = resolve;
    rejectPromise = reject;
  });
  return {
    promise,
    resolve: resolvePromise,
    reject: rejectPromise,
  };
}

function selection(path: string): MediaSelection {
  return {
    sourcePath: path,
    outputDirectory: `${path}-output`,
  };
}

async function installListener(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
  });
}

describe("useNativeMediaDrop", () => {
  it("reports a whole batch with successes, failures, and duplicates", async () => {
    const onSelection = vi.fn();
    const onError = vi.fn();
    const adapter = new ControlledMediaDropAdapter(async (path) => {
      await Promise.resolve();
      if (path.endsWith(".txt")) {
        throw new MediaDropError(
          "unsupportedExtension",
          "Unsupported media extension.",
        );
      }
      return selection(path);
    });
    const { result } = renderHook(() =>
      useNativeMediaDrop(adapter, { onSelection, onError }),
    );
    await installListener();

    act(() => {
      adapter.emit({
        type: "drop",
        paths: [
          "D:\\Media\\one.mov",
          "D:\\Media\\bad.txt",
          "d:/media/ONE.mov",
          "D:\\Media\\two.wav",
        ],
      });
    });

    expect(result.current.resolving).toBe(true);
    await waitFor(() => expect(onSelection).toHaveBeenCalledTimes(1));
    expect(onSelection).toHaveBeenCalledWith({
      selections: [
        selection("D:\\Media\\one.mov"),
        selection("D:\\Media\\two.wav"),
      ],
      failures: [
        {
          index: 1,
          path: "D:\\Media\\bad.txt",
          code: "unsupportedExtension",
          message: "Unsupported media extension.",
        },
        {
          index: 2,
          path: "d:/media/ONE.mov",
          code: "duplicatePath",
          message:
            "This media path duplicates an earlier item in the same drop.",
          duplicateOf: "D:\\Media\\one.mov",
        },
      ],
    });
    expect(onError).not.toHaveBeenCalled();
    await waitFor(() => expect(result.current.resolving).toBe(false));
  });

  it("fails closed for an empty drop without publishing an empty batch", async () => {
    const onSelection = vi.fn();
    const onError = vi.fn();
    const adapter = new ControlledMediaDropAdapter(async (path) => {
      await Promise.resolve();
      return selection(path);
    });
    renderHook(() => useNativeMediaDrop(adapter, { onSelection, onError }));
    await installListener();

    act(() => {
      adapter.emit({ type: "drop", paths: [] });
    });

    await waitFor(() => expect(onError).toHaveBeenCalledTimes(1));
    expect(onError.mock.calls[0]?.[0]).toMatchObject({
      name: "MediaDropError",
      code: "singleFile",
    });
    expect(onSelection).not.toHaveBeenCalled();
  });

  it("invalidates an older resolution when a newer drop starts", async () => {
    const first = deferred<MediaSelection>();
    const onSelection = vi.fn();
    const onError = vi.fn();
    const adapter = new ControlledMediaDropAdapter(async (path) => {
      if (path.endsWith("old.mov")) {
        return await first.promise;
      }
      return selection(path);
    });
    const { result } = renderHook(() =>
      useNativeMediaDrop(adapter, { onSelection, onError }),
    );
    await installListener();

    act(() => {
      adapter.emit({
        type: "drop",
        paths: ["D:\\Media\\old.mov"],
      });
    });
    expect(result.current.resolving).toBe(true);

    act(() => {
      adapter.emit({
        type: "drop",
        paths: ["D:\\Media\\new.mov"],
      });
    });

    await waitFor(() => expect(onSelection).toHaveBeenCalledTimes(1));
    expect(onSelection).toHaveBeenLastCalledWith({
      selections: [selection("D:\\Media\\new.mov")],
      failures: [],
    });

    await act(async () => {
      first.resolve(selection("D:\\Media\\old.mov"));
      await first.promise;
    });
    await Promise.resolve();

    expect(onSelection).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
    await waitFor(() => expect(result.current.resolving).toBe(false));
  });

  it("tracks drag state and removes the native listener on unmount", async () => {
    const adapter = new ControlledMediaDropAdapter(async (path) => {
      await Promise.resolve();
      return selection(path);
    });
    const { result, unmount } = renderHook(() =>
      useNativeMediaDrop(adapter, {
        onSelection: vi.fn(),
        onError: vi.fn(),
      }),
    );
    await installListener();

    act(() => {
      adapter.emit({ type: "enter", paths: ["D:\\Media\\one.mov"] });
    });
    expect(result.current.dragging).toBe(true);

    act(() => {
      adapter.emit({ type: "leave" });
    });
    expect(result.current.dragging).toBe(false);

    unmount();
    expect(() => {
      adapter.emit({
        type: "drop",
        paths: ["D:\\Media\\ignored.mov"],
      });
    }).not.toThrow();
  });
});
