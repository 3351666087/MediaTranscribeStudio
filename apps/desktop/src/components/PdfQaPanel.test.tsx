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
import { PdfQaPanel } from "./PdfQaPanel";

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

describe("PdfQaPanel", () => {
  it("does not claim that an unavailable score clears the threshold", () => {
    const fixture = createStudioFixture(8);

    renderPanel(<PdfQaPanel report={fixture.pdfQuality} />);

    expect(screen.getByText("Awaiting current render")).toBeInTheDocument();
    expect(
      screen.getByText(
        /No visual score is available until the current job produces verifiable page evidence\./u,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/clears the score threshold/u)).not.toBeInTheDocument();
  });

  it("keeps score and hard-gate outcomes independent", () => {
    const fixture = createStudioFixture(5);
    const report = {
      ...fixture.pdfQuality,
      status: "repair-required" as const,
      score: 93,
    };

    renderPanel(<PdfQaPanel report={report} />);

    expect(screen.getByText("Hard-gate review required")).toBeInTheDocument();
    expect(
      screen.getByText(
        /The visual score meets the threshold, but every PDF hard gate must also pass\./u,
      ),
    ).toBeInTheDocument();
  });
});
