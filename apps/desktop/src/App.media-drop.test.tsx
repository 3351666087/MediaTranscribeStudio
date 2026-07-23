import {
  act,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import {
  ControlledMediaDropAdapter,
  MediaDropError,
} from "./bridge/media-drop";

async function settleNativeListener(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
  });
}

describe("Tauri desktop media-drop integration", () => {
  it("shows native drag state and opens an editable job with generated paths", async () => {
    const user = userEvent.setup();
    const adapter = new ControlledMediaDropAdapter(async (path) => {
      await Promise.resolve();
      return {
        sourcePath: path,
        outputDirectory:
          path === "D:\\Media\\second.wav"
            ? "D:\\Media\\second-MediaTranscribeStudio"
            : "D:\\Media\\meeting-MediaTranscribeStudio",
      };
    });

    render(<App mediaDropAdapter={adapter} />);
    await screen.findByRole("heading", {
      level: 1,
      name: "Return every sentence to the right speaker.",
    });
    await settleNativeListener();

    act(() => adapter.emit({ type: "enter", paths: ["D:\\Media\\meeting.mov"] }));
    expect(
      screen.getByText("Drop one local media file").closest(
        ".media-drop-overlay",
      ),
    ).toHaveTextContent(
      "Drop one local media file",
    );

    act(() => adapter.emit({ type: "leave" }));
    await waitFor(() => {
      expect(
        screen.queryByText("Drop one local media file"),
      ).not.toBeInTheDocument();
    });

    act(() => adapter.emit({ type: "drop", paths: ["D:\\Media\\meeting.mov"] }));
    const dialog = await screen.findByRole("dialog", {
      name: "Create transcription job",
    });
    const sourceInput = within(dialog).getByPlaceholderText(
      "Enter the absolute path to a media file",
    );
    const outputInput = within(dialog).getByPlaceholderText(
      "Enter the absolute path to an output directory",
    );
    await waitFor(() => {
      expect(sourceInput).toHaveValue("D:\\Media\\meeting.mov");
      expect(outputInput).toHaveValue(
        "D:\\Media\\meeting-MediaTranscribeStudio",
      );
    });

    await user.clear(outputInput);
    await user.type(outputInput, "D:\\Media\\custom-output");
    expect(outputInput).toHaveValue("D:\\Media\\custom-output");

    act(() => adapter.emit({ type: "drop", paths: ["D:\\Media\\second.wav"] }));
    await waitFor(() => {
      expect(sourceInput).toHaveValue("D:\\Media\\second.wav");
      expect(outputInput).toHaveValue(
        "D:\\Media\\second-MediaTranscribeStudio",
      );
    });
  });

  it("fails closed for multiple or unsupported native drops", async () => {
    const adapter = new ControlledMediaDropAdapter(async (path) => {
      await Promise.resolve();
      if (path.endsWith(".exe")) {
        throw new MediaDropError(
          "unsupportedExtension",
          "Unsupported media extension.",
        );
      }
      return {
        sourcePath: path,
        outputDirectory: "D:\\Media\\safe-MediaTranscribeStudio",
      };
    });

    render(<App mediaDropAdapter={adapter} />);
    await screen.findByRole("heading", {
      level: 1,
      name: "Return every sentence to the right speaker.",
    });
    await settleNativeListener();

    act(() =>
      adapter.emit({
        type: "drop",
        paths: ["D:\\Media\\one.mov", "D:\\Media\\two.mov"],
      }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Media drop rejected",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Drop exactly one local media file.",
    );
    expect(
      screen.queryByRole("dialog", { name: "Create transcription job" }),
    ).not.toBeInTheDocument();

    act(() => adapter.emit({ type: "drop", paths: ["D:\\Media\\tool.exe"] }));
    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        "That file type is not supported.",
      );
    });
    expect(
      screen.queryByRole("dialog", { name: "Create transcription job" }),
    ).not.toBeInTheDocument();
  });
});
