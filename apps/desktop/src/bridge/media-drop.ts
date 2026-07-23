import {
  basename,
  dirname,
  extname,
  isAbsolute,
  join,
  normalize,
} from "@tauri-apps/api/path";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { isTauriRuntime } from "./desktop-backend";

export const SUPPORTED_MEDIA_EXTENSIONS = [
  "mov",
  "mp4",
  "m4v",
  "mkv",
  "webm",
  "wav",
  "mp3",
  "m4a",
  "flac",
  "aac",
  "ogg",
] as const;

export const MAX_MEDIA_DROP_PATHS = 32;
export const DEFAULT_MEDIA_DROP_CONCURRENCY = 4;

const SUPPORTED_EXTENSION_SET = new Set<string>(SUPPORTED_MEDIA_EXTENSIONS);
const MAX_LOCAL_PATH_LENGTH = 4_096;
const OUTPUT_SUFFIX = "-MediaTranscribeStudio";

export type MediaDropErrorCode =
  | "singleFile"
  | "emptyPath"
  | "pathTooLong"
  | "controlCharacter"
  | "networkPath"
  | "pathTraversal"
  | "absolutePath"
  | "unsupportedExtension"
  | "unsafeOutput"
  | "nativeUnavailable";

export class MediaDropError extends Error {
  readonly code: MediaDropErrorCode;

  constructor(code: MediaDropErrorCode, message: string) {
    super(message);
    this.name = "MediaDropError";
    this.code = code;
  }
}

export interface MediaSelection {
  sourcePath: string;
  outputDirectory: string;
}

export type MediaDropFailureCode =
  | MediaDropErrorCode
  | "duplicatePath"
  | "resolutionFailed";

export interface MediaDropFailure {
  index: number;
  path: string;
  code: MediaDropFailureCode;
  message: string;
  duplicateOf?: string;
}

export interface MediaDropBatchResult {
  selections: MediaSelection[];
  failures: MediaDropFailure[];
}

export type MediaDropEvent =
  | { type: "enter"; paths: readonly string[] }
  | { type: "over" }
  | { type: "drop"; paths: readonly string[] }
  | { type: "leave" };

export interface MediaDropAdapter {
  readonly kind: "tauri" | "mock";
  listen(handler: (event: MediaDropEvent) => void): Promise<() => void>;
  resolve(path: string): Promise<MediaSelection>;
}

export interface PathOperations {
  normalize(path: string): Promise<string>;
  join(...paths: string[]): Promise<string>;
  dirname(path: string): Promise<string>;
  extname(path: string): Promise<string>;
  basename(path: string, ext?: string): Promise<string>;
  isAbsolute(path: string): Promise<boolean>;
}

const tauriPathOperations: PathOperations = {
  normalize,
  join,
  dirname,
  extname,
  basename,
  isAbsolute,
};

function fail(code: MediaDropErrorCode, message: string): never {
  throw new MediaDropError(code, message);
}

function containsTraversalSegment(path: string): boolean {
  return /(?:^|[\\/])\.\.(?:[\\/]|$)/u.test(path);
}

function isNetworkOrDevicePath(path: string): boolean {
  return /^(?:\\\\|\/\/)/u.test(path);
}

function comparablePath(path: string): string {
  const normalizedSeparators = path.replace(/\\/gu, "/");
  const drivePrefix = /^[a-zA-Z]:\//u.exec(normalizedSeparators)?.[0] ?? "";
  const rootPrefix =
    drivePrefix || (normalizedSeparators.startsWith("/") ? "/" : "");
  const remainder = rootPrefix
    ? normalizedSeparators.slice(rootPrefix.length)
    : normalizedSeparators;
  const comparableSegments = remainder
    .split(/\/+/u)
    .filter((segment) => segment.length > 0 && segment !== ".");
  const withoutTrailingSeparators = `${rootPrefix}${comparableSegments.join("/")}`;
  return drivePrefix
    ? withoutTrailingSeparators.toLocaleLowerCase("en-US")
    : withoutTrailingSeparators;
}

export function pathsAreDistinct(sourcePath: string, outputPath: string): boolean {
  if (sourcePath.trim().length === 0 || outputPath.trim().length === 0) {
    return true;
  }
  return comparablePath(sourcePath.trim()) !== comparablePath(outputPath.trim());
}

export function createSafeMediaStem(rawStem: string): string {
  let safeStem = rawStem
    .normalize("NFKC")
    .replace(/[\u0000-\u001f\u007f]/gu, " ")
    .replace(/[<>:"/\\|?*]/gu, " ")
    .replace(/[\p{Z}\s]+/gu, "-")
    .replace(/[^\p{L}\p{M}\p{N}._-]+/gu, "-")
    .replace(/[._-]{2,}/gu, "-")
    .replace(/^[ ._-]+|[ ._-]+$/gu, "");

  if (safeStem.length === 0) {
    safeStem = "media";
  }

  if (/^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:[.-]|$)/iu.test(safeStem)) {
    safeStem = `media-${safeStem}`;
  }

  return safeStem.slice(0, 120).replace(/[ .]+$/gu, "") || "media";
}

export async function resolveMediaSelection(
  rawPath: string,
  pathOperations: PathOperations = tauriPathOperations,
): Promise<MediaSelection> {
  if (rawPath.length === 0 || rawPath.trim().length === 0) {
    fail("emptyPath", "The dropped media path is empty.");
  }
  if (rawPath !== rawPath.trim()) {
    fail("emptyPath", "The dropped media path contains unsafe outer whitespace.");
  }
  if (rawPath.length > MAX_LOCAL_PATH_LENGTH) {
    fail("pathTooLong", "The dropped media path is longer than the safety limit.");
  }
  if (/[\u0000-\u001f\u007f]/u.test(rawPath)) {
    fail("controlCharacter", "The dropped media path contains a control character.");
  }
  if (isNetworkOrDevicePath(rawPath)) {
    fail("networkPath", "Network, UNC, and device paths are not accepted.");
  }
  if (containsTraversalSegment(rawPath)) {
    fail("pathTraversal", "Parent-directory traversal segments are not accepted.");
  }
  if (!(await pathOperations.isAbsolute(rawPath))) {
    fail("absolutePath", "Select an absolute local media path.");
  }

  const sourcePath = await pathOperations.normalize(rawPath);
  if (
    sourcePath.length === 0 ||
    sourcePath.length > MAX_LOCAL_PATH_LENGTH ||
    isNetworkOrDevicePath(sourcePath) ||
    !(await pathOperations.isAbsolute(sourcePath))
  ) {
    fail("absolutePath", "The normalized source is not a safe absolute local path.");
  }

  const extension = (await pathOperations.extname(sourcePath))
    .replace(/^\./u, "")
    .toLocaleLowerCase("en-US");
  if (!SUPPORTED_EXTENSION_SET.has(extension)) {
    fail(
      "unsupportedExtension",
      `Supported media extensions are: ${SUPPORTED_MEDIA_EXTENSIONS.join(", ")}.`,
    );
  }

  const sourceParent = await pathOperations.normalize(
    await pathOperations.dirname(sourcePath),
  );
  const sourceName = await pathOperations.basename(sourcePath);
  const suffixLength = extension.length + 1;
  const rawStem =
    sourceName.toLocaleLowerCase("en-US").endsWith(`.${extension}`) &&
    sourceName.length > suffixLength
      ? sourceName.slice(0, -suffixLength)
      : sourceName;
  const outputLeaf = `${createSafeMediaStem(rawStem)}${OUTPUT_SUFFIX}`;
  const outputDirectory = await pathOperations.normalize(
    await pathOperations.join(sourceParent, outputLeaf),
  );
  const outputParent = await pathOperations.normalize(
    await pathOperations.dirname(outputDirectory),
  );

  if (
    comparablePath(outputParent) !== comparablePath(sourceParent) ||
    comparablePath(outputDirectory) === comparablePath(sourcePath) ||
    isNetworkOrDevicePath(outputDirectory) ||
    containsTraversalSegment(outputDirectory)
  ) {
    fail("unsafeOutput", "A safe sibling output directory could not be derived.");
  }

  return { sourcePath, outputDirectory };
}

export function validateDroppedPaths(paths: readonly string[]): readonly string[] {
  if (paths.length < 1 || paths.length > MAX_MEDIA_DROP_PATHS) {
    fail(
      "singleFile",
      `Drop between 1 and ${MAX_MEDIA_DROP_PATHS} local media files.`,
    );
  }
  return [...paths];
}

interface IndexedPath {
  index: number;
  path: string;
}

interface IndexedSelection extends IndexedPath {
  selection: MediaSelection;
}

function duplicateFailure(
  duplicate: IndexedPath,
  originalPath: string,
): MediaDropFailure {
  return {
    index: duplicate.index,
    path: duplicate.path,
    code: "duplicatePath",
    message: "This media path duplicates an earlier item in the same drop.",
    duplicateOf: originalPath,
  };
}

function resolutionFailure(
  item: IndexedPath,
  error: unknown,
): MediaDropFailure {
  if (error instanceof MediaDropError) {
    return {
      index: item.index,
      path: item.path,
      code: error.code,
      message: error.message,
    };
  }

  return {
    index: item.index,
    path: item.path,
    code: "resolutionFailed",
    message:
      error instanceof Error && error.message.length > 0
        ? error.message
        : "The media path could not be resolved.",
  };
}

function deduplicateComparablePaths(paths: readonly string[]): {
  unique: IndexedPath[];
  failures: MediaDropFailure[];
} {
  const firstPathByComparable = new Map<string, string>();
  const unique: IndexedPath[] = [];
  const failures: MediaDropFailure[] = [];

  paths.forEach((path, index) => {
    const comparable = comparablePath(path);
    const originalPath = firstPathByComparable.get(comparable);
    if (originalPath !== undefined) {
      failures.push(duplicateFailure({ index, path }, originalPath));
      return;
    }
    firstPathByComparable.set(comparable, path);
    unique.push({ index, path });
  });

  return { unique, failures };
}

async function mapWithBoundedConcurrencySettled<TInput, TOutput>(
  items: readonly TInput[],
  concurrency: number,
  mapper: (item: TInput) => Promise<TOutput>,
): Promise<Array<PromiseSettledResult<TOutput>>> {
  const results = new Array<PromiseSettledResult<TOutput>>(items.length);
  let nextIndex = 0;
  const workerCount = Math.min(concurrency, items.length);

  async function worker(): Promise<void> {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      const [result] = await Promise.allSettled([mapper(items[index])]);
      results[index] = result;
    }
  }

  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      await worker();
    }),
  );
  return results;
}

export async function resolveMediaDropBatch(
  adapter: Pick<MediaDropAdapter, "resolve">,
  rawPaths: readonly string[],
  concurrency = DEFAULT_MEDIA_DROP_CONCURRENCY,
): Promise<MediaDropBatchResult> {
  const paths = validateDroppedPaths(rawPaths);
  const boundedConcurrency = Math.max(
    1,
    Math.min(
      MAX_MEDIA_DROP_PATHS,
      Number.isFinite(concurrency) ? Math.floor(concurrency) : 1,
    ),
  );
  const {
    unique,
    failures: duplicateFailures,
  } = deduplicateComparablePaths(paths);
  const settled = await mapWithBoundedConcurrencySettled(
    unique,
    boundedConcurrency,
    async (item): Promise<IndexedSelection> => ({
      ...item,
      selection: await adapter.resolve(item.path),
    }),
  );
  const resolvedSelections: IndexedSelection[] = [];
  const failures = [...duplicateFailures];

  settled.forEach((result, resultIndex) => {
    const item = unique[resultIndex];
    if (result.status === "fulfilled") {
      resolvedSelections.push(result.value);
    } else {
      failures.push(resolutionFailure(item, result.reason));
    }
  });

  const firstResolvedPath = new Map<string, IndexedSelection>();
  const selections: IndexedSelection[] = [];
  resolvedSelections
    .sort((left, right) => left.index - right.index)
    .forEach((item) => {
      const comparable = comparablePath(item.selection.sourcePath);
      const original = firstResolvedPath.get(comparable);
      if (original !== undefined) {
        failures.push(duplicateFailure(item, original.path));
        return;
      }
      firstResolvedPath.set(comparable, item);
      selections.push(item);
    });

  return {
    selections: selections.map(({ selection }) => selection),
    failures: failures.sort((left, right) => left.index - right.index),
  };
}

export class TauriMediaDropAdapter implements MediaDropAdapter {
  readonly kind = "tauri" as const;

  async listen(handler: (event: MediaDropEvent) => void): Promise<() => void> {
    return getCurrentWindow().onDragDropEvent((event) => {
      const payload = event.payload;
      if (payload.type === "enter") {
        handler({ type: "enter", paths: payload.paths });
      } else if (payload.type === "drop") {
        handler({ type: "drop", paths: payload.paths });
      } else if (payload.type === "leave") {
        handler({ type: "leave" });
      } else {
        handler({ type: "over" });
      }
    });
  }

  async resolve(path: string): Promise<MediaSelection> {
    return resolveMediaSelection(path);
  }
}

export class ControlledMediaDropAdapter implements MediaDropAdapter {
  readonly kind = "mock" as const;
  private handler: ((event: MediaDropEvent) => void) | null = null;

  constructor(
    private readonly resolver: (path: string) => Promise<MediaSelection>,
  ) {}

  async listen(handler: (event: MediaDropEvent) => void): Promise<() => void> {
    await Promise.resolve();
    this.handler = handler;
    return () => {
      if (this.handler === handler) {
        this.handler = null;
      }
    };
  }

  emit(event: MediaDropEvent): void {
    this.handler?.(event);
  }

  async resolve(path: string): Promise<MediaSelection> {
    return await this.resolver(path);
  }
}

const unavailableAdapter: MediaDropAdapter = {
  kind: "mock",
  async listen() {
    await Promise.resolve();
    return () => undefined;
  },
  async resolve() {
    await Promise.resolve();
    throw new MediaDropError(
      "nativeUnavailable",
      "Native path validation is available only inside the Tauri desktop shell.",
    );
  },
};

export function createMediaDropAdapter(
  tauriRuntime = isTauriRuntime(),
): MediaDropAdapter {
  return tauriRuntime ? new TauriMediaDropAdapter() : unavailableAdapter;
}
