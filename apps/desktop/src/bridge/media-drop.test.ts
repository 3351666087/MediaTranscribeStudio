import { win32 } from "node:path";
import {
  DEFAULT_MEDIA_DROP_CONCURRENCY,
  MAX_MEDIA_DROP_PATHS,
  ControlledMediaDropAdapter,
  MediaDropError,
  createSafeMediaStem,
  pathsAreDistinct,
  resolveMediaDropBatch,
  resolveMediaSelection,
  validateDroppedPaths,
  type PathOperations,
} from "./media-drop";

async function resolved<T>(value: T): Promise<T> {
  await Promise.resolve();
  return value;
}

const windowsPathOperations: PathOperations = {
  async normalize(path) {
    return await resolved(win32.normalize(path));
  },
  async join(...paths) {
    return await resolved(win32.join(...paths));
  },
  async dirname(path) {
    return await resolved(win32.dirname(path));
  },
  async extname(path) {
    return await resolved(win32.extname(path));
  },
  async basename(path, extension) {
    return await resolved(win32.basename(path, extension));
  },
  async isAbsolute(path) {
    return await resolved(win32.isAbsolute(path));
  },
};

async function expectCode(
  promise: Promise<unknown>,
  code: MediaDropError["code"],
): Promise<void> {
  await expect(promise).rejects.toMatchObject({ code });
}

describe("native media-drop path policy", () => {
  it("derives an editable sibling output directory from a valid source", async () => {
    await expect(
      resolveMediaSelection(
        "D:\\Media\\Quarterly Meeting.mov",
        windowsPathOperations,
      ),
    ).resolves.toEqual({
      sourcePath: "D:\\Media\\Quarterly Meeting.mov",
      outputDirectory:
        "D:\\Media\\Quarterly-Meeting-MediaTranscribeStudio",
    });
  });

  it("accepts known uppercase aliases from the shared capability registry", async () => {
    await expect(
      resolveMediaSelection(
        "D:\\Media\\Legacy Capture.RMVB",
        windowsPathOperations,
      ),
    ).resolves.toEqual({
      sourcePath: "D:\\Media\\Legacy Capture.RMVB",
      outputDirectory:
        "D:\\Media\\Legacy-Capture-MediaTranscribeStudio",
    });

    await expect(
      resolveMediaSelection(
        "D:\\Media\\Camera Stream.H265",
        windowsPathOperations,
      ),
    ).resolves.toMatchObject({
      sourcePath: "D:\\Media\\Camera Stream.H265",
    });
  });

  it("preserves international letters while removing unsafe filename syntax", () => {
    expect(createSafeMediaStem(" 产品 评审：v2 / final ")).toBe(
      "产品-评审-v2-final",
    );
    expect(createSafeMediaStem("CON")).toBe("media-CON");
    expect(createSafeMediaStem("***")).toBe("media");
  });

  it("rejects unknown, extensionless, relative, network, traversal, and control paths", async () => {
    await expectCode(
      resolveMediaSelection("D:\\Media\\meeting.exe", windowsPathOperations),
      "unsupportedExtension",
    );
    await expectCode(
      resolveMediaSelection("D:\\Media\\meeting", windowsPathOperations),
      "unsupportedExtension",
    );
    await expectCode(
      resolveMediaSelection("Media\\meeting.mov", windowsPathOperations),
      "absolutePath",
    );
    await expectCode(
      resolveMediaSelection("\\\\server\\share\\meeting.mov", windowsPathOperations),
      "networkPath",
    );
    await expectCode(
      resolveMediaSelection(
        "D:\\Media\\..\\Secrets\\meeting.mov",
        windowsPathOperations,
      ),
      "pathTraversal",
    );
    await expectCode(
      resolveMediaSelection(
        "D:\\Media\\meeting\u0000.mov",
        windowsPathOperations,
      ),
      "controlCharacter",
    );
  });

  it("accepts between one and thirty-two dropped paths", () => {
    expect(() => validateDroppedPaths([])).toThrow(MediaDropError);
    expect(validateDroppedPaths(["D:\\Media\\one.mov"])).toEqual([
      "D:\\Media\\one.mov",
    ]);
    const maximumBatch = Array.from(
      { length: MAX_MEDIA_DROP_PATHS },
      (_, index) => `D:\\Media\\${index}.mov`,
    );
    expect(validateDroppedPaths(maximumBatch)).toEqual(maximumBatch);
    try {
      validateDroppedPaths([
        ...maximumBatch,
        "D:\\Media\\overflow.mov",
      ]);
      throw new Error("Expected an oversized batch to be rejected.");
    } catch (error: unknown) {
      expect(error).toBeInstanceOf(MediaDropError);
      expect((error as MediaDropError).code).toBe("singleFile");
    }
  });

  it("detects source/output equality across Windows case and separators", () => {
    expect(
      pathsAreDistinct(
        "D:\\Media\\Meeting.mov",
        "d:/media/meeting.mov",
      ),
    ).toBe(false);
    expect(
      pathsAreDistinct(
        "D:\\Media\\Meeting.mov",
        "D:\\Media\\Meeting-MediaTranscribeStudio",
      ),
    ).toBe(true);
  });

  it("fails closed if joining escapes the normalized source parent", async () => {
    const escapingOperations: PathOperations = {
      ...windowsPathOperations,
      async join() {
        return await resolved("D:\\Escaped\\output");
      },
    };

    await expectCode(
      resolveMediaSelection("D:\\Media\\meeting.mov", escapingOperations),
      "unsafeOutput",
    );
  });

  it("deduplicates comparable paths and reports the duplicate", async () => {
    const resolver = vi.fn(async (path: string) => {
      await Promise.resolve();
      return {
        sourcePath: win32.normalize(path),
        outputDirectory: `${win32.dirname(path)}\\output`,
      };
    });
    const adapter = new ControlledMediaDropAdapter(resolver);

    await expect(
      resolveMediaDropBatch(adapter, [
        "D:\\Media\\Meeting.mov",
        "d:/media/./meeting.mov",
        "D:\\Media\\Second.wav",
      ]),
    ).resolves.toEqual({
      selections: [
        {
          sourcePath: "D:\\Media\\Meeting.mov",
          outputDirectory: "D:\\Media\\output",
        },
        {
          sourcePath: "D:\\Media\\Second.wav",
          outputDirectory: "D:\\Media\\output",
        },
      ],
      failures: [
        {
          index: 1,
          path: "d:/media/./meeting.mov",
          code: "duplicatePath",
          message:
            "This media path duplicates an earlier item in the same drop.",
          duplicateOf: "D:\\Media\\Meeting.mov",
        },
      ],
    });
    expect(resolver.mock.calls.map(([path]) => path)).toEqual([
      "D:\\Media\\Meeting.mov",
      "D:\\Media\\Second.wav",
    ]);
  });

  it("deduplicates paths that resolve to the same canonical source", async () => {
    const adapter = new ControlledMediaDropAdapter(async (path) => {
      await Promise.resolve();
      return {
        sourcePath:
          path === "D:\\Alias\\meeting.mov"
            ? "D:\\Media\\meeting.mov"
            : path,
        outputDirectory: "D:\\Media\\meeting-MediaTranscribeStudio",
      };
    });

    const result = await resolveMediaDropBatch(adapter, [
      "D:\\Media\\meeting.mov",
      "D:\\Alias\\meeting.mov",
    ]);

    expect(result.selections).toHaveLength(1);
    expect(result.failures).toEqual([
      expect.objectContaining({
        index: 1,
        path: "D:\\Alias\\meeting.mov",
        code: "duplicatePath",
        duplicateOf: "D:\\Media\\meeting.mov",
      }),
    ]);
  });

  it("uses bounded concurrency and preserves successes when one item fails", async () => {
    let active = 0;
    let maximumActive = 0;
    const adapter = new ControlledMediaDropAdapter(async (path) => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      const delay = path.includes("one.mov")
        ? 20
        : path.includes("two.wav")
          ? 15
          : path.includes("bad.txt")
            ? 1
            : 5;
      await new Promise((resolve) => window.setTimeout(resolve, delay));
      active -= 1;
      if (path.endsWith(".txt")) {
        throw new MediaDropError(
          "unsupportedExtension",
          "Unsupported media extension.",
        );
      }
      return {
        sourcePath: path,
        outputDirectory: `${path}-output`,
      };
    });
    const paths = [
      "D:\\Media\\one.mov",
      "D:\\Media\\two.wav",
      "D:\\Media\\bad.txt",
      "D:\\Media\\three.mp4",
      "D:\\Media\\four.flac",
      "D:\\Media\\five.m4a",
    ];

    const result = await resolveMediaDropBatch(
      adapter,
      paths,
      DEFAULT_MEDIA_DROP_CONCURRENCY,
    );

    expect(maximumActive).toBe(DEFAULT_MEDIA_DROP_CONCURRENCY);
    expect(result.selections.map(({ sourcePath }) => sourcePath)).toEqual([
      paths[0],
      paths[1],
      paths[3],
      paths[4],
      paths[5],
    ]);
    expect(result.failures).toEqual([
      {
        index: 2,
        path: paths[2],
        code: "unsupportedExtension",
        message: "Unsupported media extension.",
      },
    ]);
  });

  it("keeps known formats when unknown and extensionless siblings fail", async () => {
    const adapter = new ControlledMediaDropAdapter(async (path) =>
      await resolveMediaSelection(path, windowsPathOperations),
    );
    const paths = [
      "D:\\Media\\voice.M4A",
      "D:\\Media\\legacy.rmvb",
      "D:\\Media\\unknown.xyz",
      "D:\\Media\\extensionless",
      "D:\\Media\\camera.M2TS",
    ];

    const result = await resolveMediaDropBatch(adapter, paths);

    expect(result.selections.map(({ sourcePath }) => sourcePath)).toEqual([
      paths[0],
      paths[1],
      paths[4],
    ]);
    expect(result.failures).toEqual([
      expect.objectContaining({
        index: 2,
        path: paths[2],
        code: "unsupportedExtension",
      }),
      expect.objectContaining({
        index: 3,
        path: paths[3],
        code: "unsupportedExtension",
      }),
    ]);
    expect(
      result.failures.every(({ message }) =>
        /probing is required/iu.test(message),
      ),
    ).toBe(true);
  });

  it("converts unexpected resolver rejection into an item failure", async () => {
    const adapter = new ControlledMediaDropAdapter(async () => {
      await Promise.resolve();
      throw new Error("resolver unavailable");
    });

    await expect(
      resolveMediaDropBatch(adapter, ["D:\\Media\\meeting.mov"]),
    ).resolves.toEqual({
      selections: [],
      failures: [
        {
          index: 0,
          path: "D:\\Media\\meeting.mov",
          code: "resolutionFailed",
          message: "resolver unavailable",
        },
      ],
    });
  });
});
