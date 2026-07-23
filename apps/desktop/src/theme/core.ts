export const THEME_STORAGE_KEY = "media-transcribe-studio.ui-theme";
export const THEME_MODES = ["system", "light", "dark"] as const;

export type ThemeMode = (typeof THEME_MODES)[number];
export type ResolvedTheme = Exclude<ThemeMode, "system">;

export function isThemeMode(value: string | null): value is ThemeMode {
  return (
    value !== null &&
    (THEME_MODES as readonly string[]).includes(value)
  );
}

export function resolveInitialTheme(
  storage: Pick<Storage, "getItem">,
): ThemeMode {
  try {
    const stored = storage.getItem(THEME_STORAGE_KEY);
    return isThemeMode(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

export function resolveSystemTheme(
  mediaQuery: Pick<MediaQueryList, "matches">,
): ResolvedTheme {
  return mediaQuery.matches ? "dark" : "light";
}

export function applyTheme(
  mode: ThemeMode,
  resolvedTheme: ResolvedTheme,
): void {
  document.documentElement.dataset.themePreference = mode;
  document.documentElement.dataset.theme = resolvedTheme;
  document.documentElement.style.colorScheme = resolvedTheme;
}

export function initializeTheme(): void {
  const mode = resolveInitialTheme(window.localStorage);
  const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
  applyTheme(mode, mode === "system" ? resolveSystemTheme(mediaQuery) : mode);
}
