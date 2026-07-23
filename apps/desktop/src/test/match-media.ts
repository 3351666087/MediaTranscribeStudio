interface MatchMediaRecord {
  query: string;
  mediaQueryList: MediaQueryList;
  listeners: Set<EventListener>;
  legacyListeners: Set<(event: MediaQueryListEvent) => void>;
}

const records = new Set<MatchMediaRecord>();
let systemDark = false;

function queryMatches(query: string): boolean {
  return query === "(prefers-color-scheme: dark)" && systemDark;
}

export function installMatchMediaMock(initialDark = false): void {
  systemDark = initialDark;
  records.clear();

  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: (query: string): MediaQueryList => {
      const listeners = new Set<EventListener>();
      const legacyListeners = new Set<(event: MediaQueryListEvent) => void>();
      const record = {} as MatchMediaRecord;
      const mediaQueryList = {
        media: query,
        get matches() {
          return queryMatches(query);
        },
        onchange: null,
        addEventListener(
          type: string,
          listener: EventListenerOrEventListenerObject | null,
        ) {
          if (type === "change" && typeof listener === "function") {
            listeners.add(listener);
          }
        },
        removeEventListener(
          type: string,
          listener: EventListenerOrEventListenerObject | null,
        ) {
          if (type === "change" && typeof listener === "function") {
            listeners.delete(listener);
          }
        },
        addListener(listener: (event: MediaQueryListEvent) => void) {
          legacyListeners.add(listener);
        },
        removeListener(listener: (event: MediaQueryListEvent) => void) {
          legacyListeners.delete(listener);
        },
        dispatchEvent(event: Event) {
          listeners.forEach((listener) => listener(event));
          return true;
        },
      } as MediaQueryList;

      record.query = query;
      record.mediaQueryList = mediaQueryList;
      record.listeners = listeners;
      record.legacyListeners = legacyListeners;
      records.add(record);
      return mediaQueryList;
    },
  });
}

export function setSystemDarkMode(nextDark: boolean): void {
  if (systemDark === nextDark) {
    return;
  }
  systemDark = nextDark;

  records.forEach((record) => {
    if (record.query !== "(prefers-color-scheme: dark)") {
      return;
    }
    const event = Object.assign(new Event("change"), {
      matches: queryMatches(record.query),
      media: record.query,
    }) as MediaQueryListEvent;
    record.listeners.forEach((listener) => listener(event));
    record.legacyListeners.forEach((listener) => listener(event));
    record.mediaQueryList.onchange?.call(record.mediaQueryList, event);
  });
}
