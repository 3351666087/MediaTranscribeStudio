import { createContext, useContext } from "react";
import type { ResolvedTheme, ThemeMode } from "./core";

export interface ThemeContextValue {
  mode: ThemeMode;
  resolvedTheme: ResolvedTheme;
  setMode: (mode: ThemeMode) => void;
}

export const ThemeContext = createContext<ThemeContextValue>({
  mode: "system",
  resolvedTheme: "light",
  setMode: () => undefined,
});

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}
