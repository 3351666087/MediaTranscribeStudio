import { win32 } from "node:path";
import {
  MediaDropError,
  createSafeMediaStem,
  pathsAreDistinct,
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

  it("preserves international letters while removing unsafe filename syntax", () => {
    expect(createSafeMediaStem(" 产品 评审：v2 / final ")).toBe(
      "产品-评审-v2-final",
    );
    expect(createSafeMediaStem("CON")).toBe("media-CON");
    expect(createSafeMediaStem("***")).toBe("media");
  });

  it("rejects unsupported, relative, network, traversal, and control paths", async () => {
    await expectCode(
      resolveMediaSelection("D:\\Media\\meeting.exe", windowsPathOperations),
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

  it("requires exactly one dropped path", () => {
    expect(() => validateDroppedPaths([])).toThrow(MediaDropError);
    try {
      validateDroppedPaths([
        "D:\\Media\\one.mov",
        "D:\\Media\\two.mov",
      ]);
      throw new Error("Expected multiple paths to be rejected.");
    } catch (error: unknown) {
      expect(error).toBeInstanceOf(MediaDropError);
      expect((error as MediaDropError).code).toBe("singleFile");
    }
    expect(validateDroppedPaths(["D:\\Media\\one.mov"])).toBe(
      "D:\\Media\\one.mov",
    );
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
});
