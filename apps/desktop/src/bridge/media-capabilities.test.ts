import {
  KNOWN_MEDIA_EXTENSIONS,
  MEDIA_CAPABILITY_GROUPS,
  MEDIA_CAPABILITY_REGISTRY,
  createMediaPickerFilters,
  getMediaExtensionsForGroup,
  inspectMediaExtension,
  inspectMediaPath,
  planMediaIntake,
} from "./media-capabilities";

function sorted(values: readonly string[]): string[] {
  return [...values].sort((left, right) =>
    left < right ? -1 : left > right ? 1 : 0,
  );
}

describe("media capability registry", () => {
  it("groups a broad, explicit set of known common and legacy formats", () => {
    expect(new Set(MEDIA_CAPABILITY_REGISTRY.map(({ group }) => group))).toEqual(
      new Set(MEDIA_CAPABILITY_GROUPS),
    );
    expect(KNOWN_MEDIA_EXTENSIONS.length).toBeGreaterThan(100);
    expect(KNOWN_MEDIA_EXTENSIONS).toEqual(
      expect.arrayContaining([
        "3gp",
        "aac",
        "aiff",
        "avi",
        "flac",
        "h264",
        "hevc",
        "m2ts",
        "m4a",
        "mkv",
        "mov",
        "mp3",
        "mp4",
        "mpeg",
        "mxf",
        "ogg",
        "opus",
        "rmvb",
        "ts",
        "wav",
        "webm",
        "wma",
        "wmv",
      ]),
    );
  });

  it("publishes one deduplicated, stable-sorted extension list", () => {
    expect(KNOWN_MEDIA_EXTENSIONS).toEqual(sorted(KNOWN_MEDIA_EXTENSIONS));
    expect(new Set(KNOWN_MEDIA_EXTENSIONS).size).toBe(
      KNOWN_MEDIA_EXTENSIONS.length,
    );

    const extensionsFromRegistry = MEDIA_CAPABILITY_REGISTRY.flatMap(
      ({ canonicalExtension, aliases }) => [canonicalExtension, ...aliases],
    );
    expect(sorted(extensionsFromRegistry)).toEqual(KNOWN_MEDIA_EXTENSIONS);
  });

  it("resolves uppercase extensions and aliases to their canonical record", () => {
    const m4bInspection = inspectMediaExtension(".M4B");
    expect(m4bInspection.status).toBe("recognized");
    if (m4bInspection.status !== "recognized") {
      throw new Error("Expected M4B to resolve through the MOV capability.");
    }
    expect(m4bInspection.extension).toBe("m4b");
    expect(m4bInspection.matchedAs).toBe("alias");
    expect(m4bInspection.capability.group).toBe("container");
    expect(m4bInspection.capability.canonicalExtension).toBe("mov");

    const h265Inspection = inspectMediaPath("D:\\Capture\\CAMERA.H265");
    expect(h265Inspection.status).toBe("recognized");
    if (h265Inspection.status !== "recognized") {
      throw new Error("Expected H265 to resolve through the HEVC capability.");
    }
    expect(h265Inspection.extension).toBe("h265");
    expect(h265Inspection.matchedAs).toBe("alias");
    expect(h265Inspection.capability.group).toBe("video");
    expect(h265Inspection.capability.canonicalExtension).toBe("hevc");
    expect(h265Inspection.capability.probeRequirement).toBe("required");
  });

  it("routes unknown or missing extensions to content probing instead of rejecting them", () => {
    expect(inspectMediaPath("D:\\Media\\recording")).toEqual({
      status: "probe-required",
      extension: null,
      reason: "missing-extension",
    });
    expect(inspectMediaPath("D:\\Media\\recording.UNKNOWN")).toEqual({
      status: "probe-required",
      extension: "unknown",
      reason: "unknown-extension",
    });
    expect(inspectMediaPath("D:\\Media\\.hidden")).toEqual({
      status: "probe-required",
      extension: null,
      reason: "missing-extension",
    });
    expect(inspectMediaPath("D:\\Media.with.dot\\recording")).toEqual({
      status: "probe-required",
      extension: null,
      reason: "missing-extension",
    });
    expect(inspectMediaPath("D:\\Media\\recording.")).toEqual({
      status: "probe-required",
      extension: null,
      reason: "missing-extension",
    });
  });

  it.each([
    ["recognized", "D:\\Media\\meeting.M4A", "recognized"],
    ["unknown", "D:\\Media\\meeting.futuremedia", "probe-required"],
    ["extensionless", "D:\\Media\\meeting", "probe-required"],
  ] as const)(
    "admits %s filenames to the same local FFmpeg content-probe route",
    (_label, path, expectedHintStatus) => {
      expect(planMediaIntake(path)).toMatchObject({
        admitted: true,
        verification: "local-ffmpeg-content-probe",
        extensionHint: {
          status: expectedHintStatus,
        },
      });
    },
  );

  it("derives sorted extension-hint groups without turning them into an allowlist", () => {
    const filters = createMediaPickerFilters();
    expect(filters.map(({ name }) => name)).toEqual([
      "Common media filename hints",
      "Audio filename hints",
      "Video filename hints",
      "Container filename hints",
    ]);
    expect(filters[0].extensions).toEqual(KNOWN_MEDIA_EXTENSIONS);
    expect(planMediaIntake("D:\\Media\\not-in-the-hints.custom").admitted).toBe(
      true,
    );

    MEDIA_CAPABILITY_GROUPS.forEach((group, index) => {
      const groupExtensions = getMediaExtensionsForGroup(group);
      expect(groupExtensions).toEqual(sorted(groupExtensions));
      expect(new Set(groupExtensions).size).toBe(groupExtensions.length);
      expect(filters[index + 1].extensions).toEqual(groupExtensions);
    });
  });
});
