import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { setSystemDarkMode } from "../test/match-media";
import {
  THEME_STORAGE_KEY,
  ThemeProvider,
  initializeTheme,
  resolveInitialTheme,
  useTheme,
} from ".";

function ThemeProbe() {
  const { mode, resolvedTheme, setMode } = useTheme();
  return (
    <>
      <output aria-label="theme mode">{mode}</output>
      <output aria-label="resolved theme">{resolvedTheme}</output>
      <button type="button" onClick={() => setMode("light")}>
        Choose light
      </button>
      <button type="button" onClick={() => setMode("dark")}>
        Choose dark
      </button>
      <button type="button" onClick={() => setMode("system")}>
        Choose system
      </button>
    </>
  );
}

describe("desktop appearance preferences", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("fails closed to system for an invalid stored value", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "sepia");
    expect(resolveInitialTheme(window.localStorage)).toBe("system");

    initializeTheme();
    expect(document.documentElement.dataset.themePreference).toBe("system");
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("persists explicit light and dark modes on the document", async () => {
    const user = userEvent.setup();
    render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>,
    );

    await user.click(screen.getByRole("button", { name: "Choose dark" }));
    expect(screen.getByLabelText("theme mode")).toHaveTextContent("dark");
    expect(screen.getByLabelText("resolved theme")).toHaveTextContent("dark");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("dark");
    expect(document.documentElement.dataset.themePreference).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.style.colorScheme).toBe("dark");

    await user.click(screen.getByRole("button", { name: "Choose light" }));
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
  });

  it("reacts live to operating-system changes while system mode is active", () => {
    setSystemDarkMode(true);
    render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>,
    );

    expect(screen.getByLabelText("theme mode")).toHaveTextContent("system");
    expect(screen.getByLabelText("resolved theme")).toHaveTextContent("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");

    act(() => setSystemDarkMode(false));
    expect(screen.getByLabelText("resolved theme")).toHaveTextContent("light");
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("keeps an explicit theme stable when the operating system changes", async () => {
    const user = userEvent.setup();
    render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Choose dark" }));

    act(() => setSystemDarkMode(true));
    act(() => setSystemDarkMode(false));
    expect(screen.getByLabelText("theme mode")).toHaveTextContent("dark");
    expect(screen.getByLabelText("resolved theme")).toHaveTextContent("dark");
  });

  it("uses system and still changes the live theme when storage is unavailable", async () => {
    const throwingStorage: Pick<Storage, "getItem"> = {
      getItem() {
        throw new Error("Storage is disabled.");
      },
    };
    expect(resolveInitialTheme(throwingStorage)).toBe("system");

    const user = userEvent.setup();
    vi.spyOn(Storage.prototype, "setItem").mockImplementationOnce(() => {
      throw new Error("Storage is read-only.");
    });
    render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>,
    );

    await user.click(screen.getByRole("button", { name: "Choose dark" }));
    expect(screen.getByLabelText("theme mode")).toHaveTextContent("dark");
    expect(screen.getByLabelText("resolved theme")).toHaveTextContent("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
  });
});
