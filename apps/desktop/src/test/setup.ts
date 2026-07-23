import "@testing-library/jest-dom/vitest";
import { installMatchMediaMock } from "./match-media";

window.requestAnimationFrame = (callback) =>
  window.setTimeout(() => callback(performance.now()), 0);
window.cancelAnimationFrame = (handle) => window.clearTimeout(handle);
Object.defineProperty(HTMLElement.prototype, "scrollTo", {
  configurable: true,
  value: () => undefined,
});

beforeEach(() => {
  installMatchMediaMock(false);
  window.localStorage.clear();
  document.documentElement.lang = "";
  delete document.documentElement.dataset.locale;
  delete document.documentElement.dataset.theme;
  delete document.documentElement.dataset.themePreference;
  document.documentElement.style.colorScheme = "";
  document.title = "";
});
