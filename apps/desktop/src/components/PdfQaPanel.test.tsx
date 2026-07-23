import {
  render,
  screen,
  within,
  type RenderResult,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

    expect(screen.getAllByText("Awaiting current render")).toHaveLength(2);
    expect(
      screen.getByRole("img", { name: "PDF visual score —" }),
    ).toHaveTextContent("—");
    expect(screen.getByRole("heading", { name: "PDF visual score" })).toBeInTheDocument();
    expect(screen.queryByText("Visual score 0")).not.toBeInTheDocument();
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

  it("keeps hard-gate and visual detail closed until the user asks for it", async () => {
    const user = userEvent.setup();
    const fixture = createStudioFixture(5);
    const report = {
      ...fixture.pdfQuality,
      status: "passed" as const,
      score: 96,
    };

    renderPanel(<PdfQaPanel report={report} />);

    expect(
      screen.getByRole("img", { name: "PDF visual score 96%" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "PDF visual score" })).toBeInTheDocument();
    expect(screen.queryByText("Visual score 96")).not.toBeInTheDocument();

    const hardGates = screen
      .getByText("13 PDF hard gates")
      .closest("details");
    const facets = screen.getByText("14 visual facets").closest("details");
    expect(hardGates).not.toBeNull();
    expect(facets).not.toBeNull();
    expect(hardGates).not.toHaveAttribute("open");
    expect(facets).not.toHaveAttribute("open");
    expect(
      within(hardGates as HTMLElement).getByText("PDF opens successfully"),
    ).not.toBeVisible();
    expect(
      within(facets as HTMLElement).getByText("Coherence"),
    ).not.toBeVisible();

    await user.click(
      within(hardGates as HTMLElement).getByText("13 PDF hard gates"),
    );
    expect(hardGates).toHaveAttribute("open");
    expect(
      within(hardGates as HTMLElement).getAllByRole("listitem"),
    ).toHaveLength(13);
    expect(
      within(hardGates as HTMLElement).getByText("PDF opens successfully"),
    ).toBeInTheDocument();

    await user.click(
      within(facets as HTMLElement).getByText("14 visual facets"),
    );
    expect(facets).toHaveAttribute("open");
    expect(
      within(facets as HTMLElement).getAllByRole("progressbar"),
    ).toHaveLength(14);
    expect(
      within(facets as HTMLElement).getByText("Coherence"),
    ).toBeInTheDocument();
  });
});
