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

const OUTPUT_BY_SOURCE: Readonly<Record<string, string>> = {
  "D:\\Media\\one.mov": "D:\\Media\\one-MediaTranscribeStudio",
  "D:\\Media\\two.wav": "D:\\Media\\two-MediaTranscribeStudio",
  "D:\\Media\\three.mp4": "D:\\Media\\three-MediaTranscribeStudio",
};

function createAdapter(): ControlledMediaDropAdapter {
  return new ControlledMediaDropAdapter(async (path) => {
    await Promise.resolve();
    if (path.endsWith(".exe")) {
      throw new MediaDropError(
        "unsupportedExtension",
        "Unsupported media extension.",
      );
    }
    return {
      sourcePath: path,
      outputDirectory:
        OUTPUT_BY_SOURCE[path] ?? "D:\\Media\\safe-MediaTranscribeStudio",
    };
  });
}

async function settleNativeListener(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
  });
}

async function openDropDialog(
  adapter: ControlledMediaDropAdapter,
  paths: readonly string[],
): Promise<HTMLElement> {
  act(() => adapter.emit({ type: "drop", paths }));
  return await screen.findByRole("dialog", {
    name: "Create transcription job",
  });
}

describe("Tauri desktop media-drop integration", () => {
  it("keeps two dropped media files as two editable queue rows", async () => {
    const adapter = createAdapter();

    render(<App mediaDropAdapter={adapter} />);
    await screen.findByRole("heading", {
      level: 1,
      name: "Return every sentence to the right speaker.",
    });
    await settleNativeListener();

    const dialog = await openDropDialog(adapter, [
      "D:\\Media\\one.mov",
      "D:\\Media\\two.wav",
    ]);
    await waitFor(() => {
      expect(
        within(dialog).getAllByPlaceholderText(
          "Enter the absolute path to a media file",
        ),
      ).toHaveLength(2);
    });
    const sourceInputs = within(dialog).getAllByPlaceholderText(
      "Enter the absolute path to a media file",
    );
    const outputInputs = within(dialog).getAllByPlaceholderText(
      "Enter the absolute path to an output directory",
    );

    expect(sourceInputs).toHaveLength(2);
    expect(outputInputs).toHaveLength(2);
    expect(sourceInputs[0]).toHaveValue("D:\\Media\\one.mov");
    expect(sourceInputs[1]).toHaveValue("D:\\Media\\two.wav");
    expect(outputInputs[0]).toHaveValue(
      "D:\\Media\\one-MediaTranscribeStudio",
    );
    expect(outputInputs[1]).toHaveValue(
      "D:\\Media\\two-MediaTranscribeStudio",
    );
  });

  it("appends later drops without replacing existing rows or user output", async () => {
    const user = userEvent.setup();
    const adapter = createAdapter();

    render(<App mediaDropAdapter={adapter} />);
    await screen.findByRole("heading", {
      level: 1,
      name: "Return every sentence to the right speaker.",
    });
    await settleNativeListener();

    const dialog = await openDropDialog(adapter, ["D:\\Media\\one.mov"]);
    await waitFor(() => {
      expect(
        within(dialog).getByPlaceholderText(
          "Enter the absolute path to an output directory",
        ),
      ).toHaveValue("D:\\Media\\one-MediaTranscribeStudio");
    });
    const firstOutput = within(dialog).getByPlaceholderText(
      "Enter the absolute path to an output directory",
    );
    await user.clear(firstOutput);
    await user.type(firstOutput, "D:\\Projects\\Final");

    act(() =>
      adapter.emit({
        type: "drop",
        paths: ["D:\\Media\\two.wav", "D:\\Media\\three.mp4"],
      }),
    );

    await waitFor(() => {
      expect(
        within(dialog).getAllByPlaceholderText(
          "Enter the absolute path to a media file",
        ),
      ).toHaveLength(3);
    });
    const sourceInputs = within(dialog).getAllByPlaceholderText(
      "Enter the absolute path to a media file",
    );
    const outputInputs = within(dialog).getAllByPlaceholderText(
      "Enter the absolute path to an output directory",
    );

    expect(
      sourceInputs.map((input) => (input as HTMLInputElement).value),
    ).toEqual([
      "D:\\Media\\one.mov",
      "D:\\Media\\two.wav",
      "D:\\Media\\three.mp4",
    ]);
    expect(outputInputs[0]).toHaveValue("D:\\Projects\\Final");
    expect(outputInputs[1]).toHaveValue(
      "D:\\Media\\two-MediaTranscribeStudio",
    );
    expect(outputInputs[2]).toHaveValue(
      "D:\\Media\\three-MediaTranscribeStudio",
    );
  });

  it("retains supported files when a sibling path fails and reports the failure", async () => {
    const adapter = createAdapter();

    render(<App mediaDropAdapter={adapter} />);
    await screen.findByRole("heading", {
      level: 1,
      name: "Return every sentence to the right speaker.",
    });
    await settleNativeListener();

    const dialog = await openDropDialog(adapter, [
      "D:\\Media\\one.mov",
      "D:\\Media\\tool.exe",
      "D:\\Media\\two.wav",
    ]);

    await waitFor(() => {
      expect(
        within(dialog).getAllByPlaceholderText(
          "Enter the absolute path to a media file",
        ),
      ).toHaveLength(2);
    });
    const sourceInputs = within(dialog).getAllByPlaceholderText(
      "Enter the absolute path to a media file",
    );
    expect(sourceInputs[0]).toHaveValue("D:\\Media\\one.mov");
    expect(sourceInputs[1]).toHaveValue("D:\\Media\\two.wav");
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Media drop rejected",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "That file type is not supported.",
    );
  });
});
