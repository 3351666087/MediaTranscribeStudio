import {
  render,
  screen,
  within,
  type RenderResult,
} from "@testing-library/react";
import type { ReactElement } from "react";
import type { PerformanceMetrics } from "../contracts/studio";
import {
  LOCALE_OPTIONS,
  type MessageKey,
  type MessageParams,
} from "../i18n";
import { I18nContext } from "../i18n/context";
import { translate } from "../i18n/core";
import workbenchMessages from "../i18n/fragments/workbench-panels.json";
import { createStudioFixture } from "../mocks/studio-fixture";
import { PerformanceWorkbench } from "./PerformanceWorkbench";

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

function renderWorkbench(ui: ReactElement): RenderResult {
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

describe("PerformanceWorkbench", () => {
  it("renders measured RTF, latency quantiles, cache/escalation/recompute, and peak resources", () => {
    const fixture = createStudioFixture(13);

    renderWorkbench(
      <PerformanceWorkbench
        metrics={fixture.performance}
        stages={fixture.stages}
        reviews={fixture.reviews}
      />,
    );

    expect(
      screen.getByRole("heading", { level: 2, name: "Performance workbench" }),
    ).toBeInTheDocument();
    expect(screen.getByText("0.37×")).toBeInTheDocument();
    expect(screen.getByText("78%")).toBeInTheDocument();
    expect(screen.getByText("6.8%")).toBeInTheDocument();
    expect(screen.getByText("2.2%")).toBeInTheDocument();
    expect(screen.getByText("Local demonstration sample (not a production benchmark)")).toBeInTheDocument();

    const table = screen.getByRole("table", {
      name: "p50 and p95 latency for each processing stage",
    });
    expect(within(table).getAllByRole("row")).toHaveLength(
      fixture.stages.length + 1,
    );
    expect(within(table).getByRole("columnheader", { name: "p50" })).toBeInTheDocument();
    expect(within(table).getByRole("columnheader", { name: "p95" })).toBeInTheDocument();
    expect(screen.getByText("84%")).toBeInTheDocument();
    expect(screen.getByText("10.6 GB")).toBeInTheDocument();
    expect(screen.getByText("7.4 GB")).toBeInTheDocument();

    const coverage = screen.getByLabelText("Selective cascade queue coverage");
    expect(within(coverage).getByText("ERes2NetV2-derived scope")).toBeInTheDocument();
    expect(within(coverage).getByText("pyannote fallback-derived scope")).toBeInTheDocument();
    expect(within(coverage).getByText("CAM++ conflict segments")).toBeInTheDocument();
    expect(within(coverage).getByText("REVIEW_REQUIRED")).toBeInTheDocument();
    const eresMetric = within(coverage)
      .getByText("ERes2NetV2-derived scope")
      .closest("div");
    const pyannoteMetric = within(coverage)
      .getByText("pyannote fallback-derived scope")
      .closest("div");
    const camMetric = within(coverage)
      .getByText("CAM++ conflict segments")
      .closest("div");
    const reviewMetric = within(coverage)
      .getByText("REVIEW_REQUIRED")
      .closest("div");

    expect(within(eresMetric as HTMLElement).getByText("3")).toBeInTheDocument();
    expect(within(pyannoteMetric as HTMLElement).getByText("2")).toBeInTheDocument();
    expect(within(camMetric as HTMLElement).getByText("2")).toBeInTheDocument();
    expect(within(reviewMetric as HTMLElement).getByText("4")).toBeInTheDocument();
    expect(screen.getAllByText("CAM++").length).toBeGreaterThan(0);
    expect(screen.getAllByText("ERes2NetV2").length).toBeGreaterThan(0);
    expect(screen.getAllByText("pyannote").length).toBeGreaterThan(0);
    expect(
      screen.getByText("Derived from review reasons; does not claim that model stages have run"),
    ).toBeInTheDocument();
  });

  it("shows an explicit unavailable state before real measurements exist", () => {
    const fixture = createStudioFixture(2);
    const unavailable: PerformanceMetrics = {
      status: "unavailable",
      reason: "The job has not run, so model-stage and resource measurements are unavailable.",
    };

    renderWorkbench(
      <PerformanceWorkbench
        metrics={unavailable}
        stages={fixture.stages}
        reviews={fixture.reviews}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent("Performance data unavailable");
    expect(screen.getByRole("status")).toHaveTextContent(
      "The job has not run, so model-stage and resource measurements are unavailable.",
    );
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.queryByText(/×$/u)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Selective cascade queue coverage")).toBeInTheDocument();
    expect(
      screen.getByText("The queue scope above is still derivable, but it is not a performance measurement."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Derived from review reasons; does not claim that model stages have run"),
    ).toBeInTheDocument();
  });
});
