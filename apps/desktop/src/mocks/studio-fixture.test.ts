import { parseStudioSnapshot } from "../contracts/runtime-validation";
import {
  createSpeakerProfiles,
  createStudioFixture,
} from "./studio-fixture";

const requiredCounts = [1, 2, 5, 8, 13] as const;
const noFixedCapCounts = [21, 64] as const;

describe("studio fixture dynamic speaker generation", () => {
  it.each([...requiredCounts, ...noFixedCapCounts])(
    "generates and validates all %i speakers with contiguous IDs",
    (count) => {
      const fixture = createStudioFixture(count, {
        speakerPolicy: { mode: "manual", count },
      });
      const parsed = parseStudioSnapshot(fixture);

      expect(parsed.job.speakerCount).toBe(count);
      expect(parsed.speakers).toHaveLength(count);
      expect(parsed.speakers.map(({ id }) => id)).toEqual(
        Array.from({ length: count }, (_, index) => `speaker-${index + 1}`),
      );
      expect(parsed.speakers.at(-1)?.label).toContain(String(count));
      expect(
        parsed.reviews.every((review) =>
          review.candidates.every((candidate) =>
            parsed.speakers.some(({ id }) => id === candidate.speakerId),
          ),
        ),
      ).toBe(true);
    },
  );

  it("supports auto, manual, and hybrid count policies without changing the requested N", () => {
    const auto = parseStudioSnapshot(
      createStudioFixture(13, { speakerPolicy: { mode: "auto" } }),
    );
    const manual = parseStudioSnapshot(
      createStudioFixture(13, {
        speakerPolicy: { mode: "manual", count: 13 },
      }),
    );
    const hybrid = parseStudioSnapshot(
      createStudioFixture(13, {
        speakerPolicy: {
          mode: "hybrid",
          minSpeakers: 2,
          priorCount: 13,
          maxSpeakers: 64,
        },
      }),
    );

    expect(auto.job.speakerPolicy).toEqual({ mode: "auto" });
    expect(auto.job.speakerDetection?.estimatedCount).toBe(13);
    expect(manual.job.speakerPolicy).toEqual({ mode: "manual", count: 13 });
    expect(manual.job.speakerDetection).toBeNull();
    expect(hybrid.job.speakerPolicy).toEqual({
      mode: "hybrid",
      minSpeakers: 2,
      priorCount: 13,
      maxSpeakers: 64,
    });
    expect(hybrid.job.speakerDetection?.estimatedCount).toBe(13);
  });

  it("keeps N=1 review candidates valid without inventing a second speaker", () => {
    const fixture = createStudioFixture(1, {
      speakerPolicy: { mode: "manual", count: 1 },
    });

    expect(
      fixture.reviews.every(
        (review) =>
          review.currentSpeakerId === "speaker-1" &&
          review.candidates.length === 1 &&
          review.candidates[0].speakerId === "speaker-1",
      ),
    ).toBe(true);
  });

  it.each([0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "rejects invalid speaker count %s before allocation",
    (count) => {
      expect(() => createSpeakerProfiles(count)).toThrow(
        /positive safe integer/u,
      );
    },
  );
});
