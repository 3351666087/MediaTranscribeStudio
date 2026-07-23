import {
  MediaDropError,
  type MediaDropErrorCode,
} from "../bridge/media-drop";
import type { MessageKey } from "./catalog";

export const MEDIA_DROP_ERROR_MESSAGE_KEYS: Readonly<
  Record<MediaDropErrorCode, MessageKey>
> = {
  singleFile: "drop.error.singleFile",
  emptyPath: "drop.error.emptyPath",
  pathTooLong: "drop.error.pathTooLong",
  controlCharacter: "drop.error.controlCharacter",
  networkPath: "drop.error.networkPath",
  pathTraversal: "drop.error.pathTraversal",
  absolutePath: "drop.error.absolutePath",
  unsupportedExtension: "drop.error.unsupportedExtension",
  unsafeOutput: "drop.error.unsafeOutput",
  nativeUnavailable: "drop.error.nativeUnavailable",
};

export function mediaDropErrorMessageKey(error: unknown): MessageKey {
  return error instanceof MediaDropError
    ? MEDIA_DROP_ERROR_MESSAGE_KEYS[error.code]
    : "drop.error.unknown";
}
