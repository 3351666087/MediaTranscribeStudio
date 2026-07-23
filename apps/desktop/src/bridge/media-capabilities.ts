export const MEDIA_CAPABILITY_GROUPS = [
  "audio",
  "video",
  "container",
] as const;

export type MediaCapabilityGroup = (typeof MEDIA_CAPABILITY_GROUPS)[number];

export type MediaProbeRequirement = "recommended" | "required";

export interface MediaCapability {
  readonly group: MediaCapabilityGroup;
  readonly canonicalExtension: string;
  readonly aliases: readonly string[];
  readonly probeRequirement: MediaProbeRequirement;
}

export type MediaCapabilityInspection =
  | {
      readonly status: "recognized";
      readonly extension: string;
      readonly capability: MediaCapability;
      readonly matchedAs: "canonical" | "alias";
    }
  | {
      readonly status: "probe-required";
      readonly extension: null;
      readonly reason: "missing-extension";
    }
  | {
      readonly status: "probe-required";
      readonly extension: string;
      readonly reason: "unknown-extension";
    };

export interface MediaPickerFilter {
  readonly name: string;
  readonly extensions: readonly string[];
}

function compareExtensions(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function capability(
  group: MediaCapabilityGroup,
  canonicalExtension: string,
  aliases: readonly string[] = [],
  probeRequirement: MediaProbeRequirement = "recommended",
): MediaCapability {
  return Object.freeze({
    group,
    canonicalExtension,
    aliases: Object.freeze([...aliases].sort(compareExtensions)),
    probeRequirement,
  });
}

/**
 * Known, commonly encountered FFmpeg-compatible media filename extensions.
 *
 * This is intentionally an extension-level intake registry rather than a
 * promise that every FFmpeg build can decode every file. Codec availability,
 * damaged files, encrypted media, misleading extensions, and extensionless or
 * unknown inputs still need content probing before processing.
 */
export const MEDIA_CAPABILITY_REGISTRY: readonly MediaCapability[] =
  Object.freeze([
    capability("audio", "8svx"),
    capability("audio", "aa", ["aax"]),
    capability("audio", "aac", [], "required"),
    capability("audio", "ac3", [], "required"),
    capability("audio", "acm"),
    capability("audio", "adx"),
    capability("audio", "aea"),
    capability("audio", "afc"),
    capability("audio", "aiff", ["aif", "aifc"]),
    capability("audio", "aix"),
    capability("audio", "amr", ["amrnb", "amrwb"], "required"),
    capability("audio", "ape"),
    capability("audio", "aptx", ["aptxhd"], "required"),
    capability("audio", "ast"),
    capability("audio", "au", ["snd"]),
    capability("audio", "caf"),
    capability("audio", "dff"),
    capability("audio", "dsf"),
    capability("audio", "dts", ["dtshd"], "required"),
    capability("audio", "eac3", [], "required"),
    capability("audio", "flac"),
    capability("audio", "g722", [], "required"),
    capability("audio", "g723", ["rco", "tco"], "required"),
    capability("audio", "g726", ["g726le"], "required"),
    capability("audio", "gsm", [], "required"),
    capability("audio", "ircam", ["sf"]),
    capability("audio", "mlp", [], "required"),
    capability("audio", "mmf"),
    capability(
      "audio",
      "mod",
      ["669", "amf", "far", "it", "mtm", "okt", "s3m", "stm", "xm"],
      "required",
    ),
    capability("audio", "mp2", ["m2a", "mpa"], "required"),
    capability("audio", "mp3"),
    capability("audio", "mpc", ["mpp"]),
    capability("audio", "oma"),
    capability("audio", "opus"),
    capability(
      "audio",
      "pcm",
      [
        "alaw",
        "f32be",
        "f32le",
        "f64be",
        "f64le",
        "mulaw",
        "s16be",
        "s16le",
        "s24be",
        "s24le",
        "s32be",
        "s32le",
        "s8",
        "u16be",
        "u16le",
        "u24be",
        "u24le",
        "u32be",
        "u32le",
        "u8",
        "ulaw",
      ],
      "required",
    ),
    capability("audio", "pvf"),
    capability("audio", "qcp"),
    capability("audio", "rso"),
    capability("audio", "sbc", ["msbc"], "required"),
    capability("audio", "sds"),
    capability("audio", "sox"),
    capability("audio", "tak"),
    capability("audio", "truehd", ["thd"], "required"),
    capability("audio", "tta"),
    capability("audio", "voc"),
    capability("audio", "wav", ["bwf", "rf64", "w64", "wave"]),
    capability("audio", "wavpack", ["wv"]),
    capability("audio", "wve"),
    capability("audio", "xa"),

    capability("video", "apng"),
    capability("video", "av1", ["obu"], "required"),
    capability("video", "avs", ["avs2", "avs3", "cavs"], "required"),
    capability("video", "dirac", ["drc", "vc2"], "required"),
    capability("video", "dnxhd", ["dnxhr"], "required"),
    capability("video", "gif"),
    capability("video", "h261", [], "required"),
    capability("video", "h263", [], "required"),
    capability("video", "h264", ["264", "avc"], "required"),
    capability("video", "hevc", ["265", "h265"], "required"),
    capability("video", "ivf"),
    capability("video", "j2k", ["j2c"], "required"),
    capability("video", "m1v", [], "required"),
    capability("video", "m2v", ["mpv"], "required"),
    capability("video", "m4v", [], "required"),
    capability("video", "mjpeg", ["mjpg"], "required"),
    capability("video", "rawvideo", ["rgb", "yuv"], "required"),
    capability("video", "vc1", [], "required"),
    capability("video", "vvc", ["266", "h266"], "required"),
    capability("video", "y4m", [], "required"),

    capability("container", "4xm"),
    capability("container", "amv"),
    capability("container", "asf", ["wma", "wmv"]),
    capability("container", "avi", ["divx"]),
    capability("container", "bink", ["bik", "bk2"]),
    capability("container", "cine"),
    capability("container", "cpk"),
    capability("container", "dav"),
    capability("container", "dv"),
    capability("container", "flic", ["flc", "fli"]),
    capability("container", "flv"),
    capability("container", "gxf"),
    capability("container", "matroska", ["mka", "mkv", "webm"]),
    capability(
      "container",
      "mov",
      [
        "3g2",
        "3ga",
        "3gp",
        "3gp2",
        "f4v",
        "ism",
        "isma",
        "ismv",
        "m4a",
        "m4b",
        "mj2",
        "mp4",
        "psp",
        "qt",
      ],
    ),
    capability("container", "mpeg", ["m2p", "mpe", "mpg", "ps", "vro"]),
    capability(
      "container",
      "mpegts",
      ["m2t", "m2ts", "mts", "tod", "tp", "trp", "ts"],
      "required",
    ),
    capability("container", "mve"),
    capability("container", "mxf"),
    capability("container", "nsv"),
    capability("container", "nut"),
    capability("container", "ogg", ["oga", "ogm", "ogv", "ogx", "spx"]),
    capability("container", "r3d"),
    capability("container", "realmedia", ["ra", "ram", "rm", "rmvb"]),
    capability("container", "roq"),
    capability("container", "smk"),
    capability("container", "swf"),
    capability("container", "vob", ["evo"]),
    capability("container", "vqa"),
    capability("container", "wtv", ["dvr-ms"]),
  ]);

interface IndexedCapability {
  readonly capability: MediaCapability;
  readonly matchedAs: "canonical" | "alias";
}

function buildCapabilityIndex(): ReadonlyMap<string, IndexedCapability> {
  const index = new Map<string, IndexedCapability>();

  for (const entry of MEDIA_CAPABILITY_REGISTRY) {
    const extensions = [entry.canonicalExtension, ...entry.aliases];
    for (const extension of extensions) {
      if (!/^[a-z0-9][a-z0-9-]*$/u.test(extension)) {
        throw new Error(`Invalid media extension token in registry: ${extension}`);
      }
      if (index.has(extension)) {
        throw new Error(`Duplicate media extension in registry: ${extension}`);
      }
      index.set(extension, {
        capability: entry,
        matchedAs:
          extension === entry.canonicalExtension ? "canonical" : "alias",
      });
    }
  }

  return index;
}

const CAPABILITY_BY_EXTENSION = buildCapabilityIndex();

export const KNOWN_MEDIA_EXTENSIONS: readonly string[] = Object.freeze(
  [...CAPABILITY_BY_EXTENSION.keys()].sort(compareExtensions),
);

// Backward-compatible export for callers that previously imported the
// eleven-item list from media-drop.ts. The registry above is the only source.
export const SUPPORTED_MEDIA_EXTENSIONS = KNOWN_MEDIA_EXTENSIONS;

export function getMediaExtensionsForGroup(
  group: MediaCapabilityGroup,
): readonly string[] {
  return Object.freeze(
    MEDIA_CAPABILITY_REGISTRY.filter((entry) => entry.group === group)
      .flatMap((entry) => [entry.canonicalExtension, ...entry.aliases])
      .sort(compareExtensions),
  );
}

export function createMediaPickerFilters(): MediaPickerFilter[] {
  return [
    {
      name: "Known media",
      extensions: [...KNOWN_MEDIA_EXTENSIONS],
    },
    {
      name: "Audio",
      extensions: [...getMediaExtensionsForGroup("audio")],
    },
    {
      name: "Video streams",
      extensions: [...getMediaExtensionsForGroup("video")],
    },
    {
      name: "Media containers",
      extensions: [...getMediaExtensionsForGroup("container")],
    },
  ];
}

function normalizeExtension(rawExtension: string): string {
  return rawExtension
    .trim()
    .replace(/^\.+/u, "")
    .toLocaleLowerCase("en-US");
}

export function inspectMediaExtension(
  rawExtension: string | null | undefined,
): MediaCapabilityInspection {
  const extension =
    rawExtension === null || rawExtension === undefined
      ? ""
      : normalizeExtension(rawExtension);

  if (extension.length === 0) {
    return {
      status: "probe-required",
      extension: null,
      reason: "missing-extension",
    };
  }

  const indexed = CAPABILITY_BY_EXTENSION.get(extension);
  if (indexed === undefined) {
    return {
      status: "probe-required",
      extension,
      reason: "unknown-extension",
    };
  }

  return {
    status: "recognized",
    extension,
    capability: indexed.capability,
    matchedAs: indexed.matchedAs,
  };
}

export function inspectMediaPath(path: string): MediaCapabilityInspection {
  const leaf = path.split(/[\\/]/u).at(-1) ?? "";
  const finalDot = leaf.lastIndexOf(".");
  return inspectMediaExtension(
    finalDot > 0 && finalDot < leaf.length - 1
      ? leaf.slice(finalDot + 1)
      : null,
  );
}
