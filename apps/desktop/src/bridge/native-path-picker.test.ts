import type { OpenDialogOptions } from "@tauri-apps/plugin-dialog";
import {
  selectNativeMediaFiles,
  selectNativeOutputDirectory,
} from "./native-path-picker";

describe("native path picker bridge", () => {
  it("fails safe in browser and test runtimes without opening a dialog", async () => {
    const openDialog = vi.fn();

    await expect(
      selectNativeMediaFiles({ openDialog, tauriRuntime: false }),
    ).resolves.toEqual([]);
    await expect(
      selectNativeOutputDirectory({ openDialog, tauriRuntime: false }),
    ).resolves.toBeNull();
    expect(openDialog).not.toHaveBeenCalled();
  });

  it("opens the native media picker with multi-select and no extension gate", async () => {
    const openDialog = vi
      .fn<
        (
          options: OpenDialogOptions,
        ) => Promise<string | string[] | null>
      >()
      .mockResolvedValue([
        "D:\\Media\\FIRST.MOV",
        "D:\\Media\\extensionless",
        "D:\\Media\\second.futuremedia",
      ]);

    await expect(
      selectNativeMediaFiles({
        openDialog,
        tauriRuntime: true,
      }),
    ).resolves.toEqual([
      "D:\\Media\\FIRST.MOV",
      "D:\\Media\\extensionless",
      "D:\\Media\\second.futuremedia",
    ]);

    expect(openDialog).toHaveBeenCalledWith({
      title: "Select media files",
      multiple: true,
      directory: false,
    });
  });

  it("opens the native Windows folder picker for one editable output path", async () => {
    const openDialog = vi
      .fn<
        (
          options: OpenDialogOptions,
        ) => Promise<string | string[] | null>
      >()
      .mockResolvedValue("D:\\Media\\meeting-output");

    await expect(
      selectNativeOutputDirectory({
        openDialog,
        tauriRuntime: true,
      }),
    ).resolves.toBe("D:\\Media\\meeting-output");

    expect(openDialog).toHaveBeenCalledWith({
      title: "Select output folder",
      multiple: false,
      directory: true,
      canCreateDirectories: true,
    });
  });

  it("passes localized titles to the native Windows dialogs", async () => {
    const openDialog = vi
      .fn<
        (
          options: OpenDialogOptions,
        ) => Promise<string | string[] | null>
      >()
      .mockResolvedValueOnce(["D:\\Media\\meeting.m4a"])
      .mockResolvedValueOnce("D:\\Media\\meeting-output");

    await selectNativeMediaFiles({
      openDialog,
      tauriRuntime: true,
      title: "Choose local media",
    });
    await selectNativeOutputDirectory({
      openDialog,
      tauriRuntime: true,
      title: "Choose local output folder",
    });

    expect(openDialog).toHaveBeenNthCalledWith(1, {
      title: "Choose local media",
      multiple: true,
      directory: false,
    });
    expect(openDialog).toHaveBeenNthCalledWith(2, {
      title: "Choose local output folder",
      multiple: false,
      directory: true,
      canCreateDirectories: true,
    });
  });

  it("treats media and directory cancellation as non-destructive results", async () => {
    const openDialog = vi.fn().mockResolvedValue(null);

    await expect(
      selectNativeMediaFiles({ openDialog, tauriRuntime: true }),
    ).resolves.toEqual([]);
    await expect(
      selectNativeOutputDirectory({ openDialog, tauriRuntime: true }),
    ).resolves.toBeNull();
  });

  it.each([
    undefined,
    42,
    "D:\\Media\\meeting.mov",
    [" D:\\Media\\meeting.mov"],
    ["D:\\Media\\bad\n.mov"],
  ])("rejects malformed media results", async (result) => {
    const openDialog = vi.fn().mockResolvedValue(result);

    await expect(
      selectNativeMediaFiles({ openDialog, tauriRuntime: true }),
    ).rejects.toThrow(/invalid|unsafe/u);
  });

  it.each([
    undefined,
    42,
    ["D:\\Media\\meeting-output"],
    " D:\\Media\\meeting-output",
    "D:\\Media\\bad\noutput",
  ])("rejects malformed directory results", async (result) => {
    const openDialog = vi.fn().mockResolvedValue(result);

    await expect(
      selectNativeOutputDirectory({ openDialog, tauriRuntime: true }),
    ).rejects.toThrow(/invalid|unsafe/u);
  });
});
