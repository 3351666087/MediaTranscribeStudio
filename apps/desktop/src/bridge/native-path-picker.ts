import {
  open,
  type OpenDialogOptions,
} from "@tauri-apps/plugin-dialog";
import { isTauriRuntime } from "./desktop-backend";
import { createMediaPickerFilters } from "./media-capabilities";

type OpenDialog = (
  options: OpenDialogOptions,
) => Promise<string | string[] | null>;

export interface NativePathPickerOptions {
  openDialog?: OpenDialog;
  tauriRuntime?: boolean;
}

function parseSafePath(value: unknown): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error("The native path picker returned an invalid path.");
  }
  if (value !== value.trim() || /[\u0000-\u001f\u007f]/u.test(value)) {
    throw new Error("The native path picker returned an unsafe path.");
  }
  return value;
}

function parseMediaPickerResult(value: unknown): string[] {
  if (value === null) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new Error("The native media picker returned an invalid result.");
  }
  return value.map(parseSafePath);
}

function parseDirectoryPickerResult(value: unknown): string | null {
  if (value === null) {
    return null;
  }
  if (Array.isArray(value)) {
    throw new Error("The native directory picker returned an invalid result.");
  }
  return parseSafePath(value);
}

export async function selectNativeMediaFiles(
  options: NativePathPickerOptions = {},
): Promise<string[]> {
  const tauriRuntime = options.tauriRuntime ?? isTauriRuntime();
  if (!tauriRuntime) {
    return [];
  }
  const openDialog = options.openDialog ?? open;
  const result = await openDialog({
    title: "Select media files",
    multiple: true,
    directory: false,
    filters: createMediaPickerFilters().map(({ name, extensions }) => ({
      name,
      extensions: [...extensions],
    })),
  });
  return parseMediaPickerResult(result);
}

export async function selectNativeOutputDirectory(
  options: NativePathPickerOptions = {},
): Promise<string | null> {
  const tauriRuntime = options.tauriRuntime ?? isTauriRuntime();
  if (!tauriRuntime) {
    return null;
  }
  const openDialog = options.openDialog ?? open;
  const result = await openDialog({
    title: "Select output folder",
    multiple: false,
    directory: true,
    canCreateDirectories: true,
  });
  return parseDirectoryPickerResult(result);
}
