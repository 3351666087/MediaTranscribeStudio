import { render, screen, type RenderResult } from "@testing-library/react";
import type { ReactElement } from "react";
import {
  LOCALE_OPTIONS,
  type MessageKey,
  type MessageParams,
} from "../i18n";
import { I18nContext } from "../i18n/context";
import { translate } from "../i18n/core";
import workbenchMessages from "../i18n/fragments/workbench-panels.json";
import { createStudioFixture } from "../mocks/studio-fixture";
import { DiarizationQualityPanel } from "./DiarizationQualityPanel";

const englishWorkbenchMessages = workbenchMessages.en as Partial<
  Record<string, string>
>;

function interpolate(template: string, params: MessageParams = {}): string {
  return template.replace(
    /\{([a-zA-Z][a-zA-Z0-9]*)\}/gu,
    (match, name: string) => {
      const value = params[name];
      return value === undefined ? match : String(value);
    },
  );
}

function renderPanel(ui: ReactElement): RenderResult {
  return render(
    <I18nContext.Provider
      value={{
        locale: "en",
        localeOptions: LOCALE_OPTIONS,
        setLocale: () => undefined,
        t: (key: MessageKey, params?: MessageParams) => {
          const fragmentMessage = englishWorkbenchMessages[key];
          return fragmentMessage === undefined
            ? translate("en", key, params)
            : interpolate(fragmentMessage, params);
        },
      }}
    >
      {ui}
    </I18nContext.Provider>,
  );
}

describe("DiarizationQualityPanel", () => {
  it("shows reference-dependent metrics as unavailable instead of inventing values", () => {
    const fixture = createStudioFixture(8);

    renderPanel(<DiarizationQualityPanel metrics={fixture.diarizationQuality} />);

    expect(
      screen.getByRole("heading", { level: 2, name: "Speaker separation quality" }),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Unavailable")).toHaveLength(4);
    expect(screen.getAllByText(/Reference labels missing/u)).toHaveLength(4);
    expect(screen.getByText("1.48%")).toBeInTheDocument();
    expect(screen.getByText("No fabricated benchmarks.")).toBeInTheDocument();
    expect(screen.queryByText("7.40%")).not.toBeInTheDocument();
  });

  it("renders measured DER, JER, confusion, overlap F1, and review rate from their sources", () => {
    const fixture = createStudioFixture(5, { hasReferenceLabels: true });

    renderPanel(<DiarizationQualityPanel metrics={fixture.diarizationQuality} />);

    expect(screen.getAllByText("Measured")).toHaveLength(5);
    expect(screen.getByText("7.40%")).toBeInTheDocument();
    expect(screen.getByText("11.8%")).toBeInTheDocument();
    expect(screen.getByText("3.10%")).toBeInTheDocument();
    expect(screen.getByText("82.6%")).toBeInTheDocument();
    expect(screen.getByText("1.48%")).toBeInTheDocument();
    expect(screen.getAllByText("Local human reference labels")).toHaveLength(4);
  });
});
