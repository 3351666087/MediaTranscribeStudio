import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { ThemeContext, type ThemeContextValue } from "./context";
import {
  THEME_STORAGE_KEY,
  applyTheme,
  resolveInitialTheme,
  resolveSystemTheme,
  type ResolvedTheme,
  type ThemeMode,
} from "./core";

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>(() =>
    resolveInitialTheme(window.localStorage),
  );
  const [systemResolvedTheme, setSystemResolvedTheme] = useState<ResolvedTheme>(
    () => resolveSystemTheme(window.matchMedia("(prefers-color-scheme: dark)")),
  );
  const resolvedTheme = mode === "system" ? systemResolvedTheme : mode;

  const setMode = useCallback((nextMode: ThemeMode) => {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, nextMode);
    } catch {
      // Storage can be unavailable in hardened WebViews; keep the live choice.
    }
    setModeState(nextMode);
  }, []);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event: MediaQueryListEvent) => {
      setSystemResolvedTheme(event.matches ? "dark" : "light");
    };
    mediaQuery.addEventListener("change", onChange);
    return () => mediaQuery.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    applyTheme(mode, resolvedTheme);
  }, [mode, resolvedTheme]);

  const value = useMemo<ThemeContextValue>(
    () => ({ mode, resolvedTheme, setMode }),
    [mode, resolvedTheme, setMode],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}
