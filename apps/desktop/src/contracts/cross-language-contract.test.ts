import rustDefaultSnapshot from "./fixtures/rust-default-snapshot.json";
import tsReviewDecision from "./fixtures/ts-review-decision.json";
import {
  assertReviewDecision,
  parseStudioSnapshot,
} from "./runtime-validation";
import {
  AESTHETIC_FACET_IDS,
  PDF_HARD_GATE_IDS,
  STUDIO_CONTRACT_VERSION,
} from "./studio";

describe("cross-language desktop contracts", () => {
  it("accepts the real Rust default_snapshot serde fixture in the TS parser", () => {
    const snapshot = parseStudioSnapshot(rustDefaultSnapshot);

    expect(snapshot.contractVersion).toBe(STUDIO_CONTRACT_VERSION);
    expect(snapshot.job.speakerCount).toBeNull();
    expect(snapshot.job.speakerDetection).toBeNull();
    expect(snapshot.speakers).toEqual([]);
    expect(snapshot.pdfQuality).toMatchObject({
      status: "pending",
      score: 0,
      pageCount: 0,
      repairQueue: [],
    });
    expect(snapshot.pdfQuality.hardGates.map(({ id }) => id)).toEqual(
      PDF_HARD_GATE_IDS,
    );
    expect(
      snapshot.pdfQuality.hardGates.every(
        ({ status }) => status === "pending",
      ),
    ).toBe(true);
    expect(snapshot.pdfQuality.facets.map(({ id }) => id)).toEqual(
      AESTHETIC_FACET_IDS,
    );
    expect(
      snapshot.pdfQuality.facets.every(
        ({ score, status }) => score === 0 && status === "pending",
      ),
    ).toBe(true);
  });

  it("accepts the real TS ReviewDecision JSON fixture in the TS validator", () => {
    expect(() => assertReviewDecision(tsReviewDecision)).not.toThrow();
    expect(JSON.parse(JSON.stringify(tsReviewDecision))).toEqual(
      tsReviewDecision,
    );
  });
});
