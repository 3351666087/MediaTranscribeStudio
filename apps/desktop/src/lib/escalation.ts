import type { ReviewReason, ReviewSegment } from "../contracts/studio";

export const eres2NetReasons = new Set<ReviewReason>([
  "speaker_close_score",
  "speaker_count_uncertain",
  "speaker_outlier",
]);

export const pyannoteReasons = new Set<ReviewReason>([
  "overlap_detected",
  "timestamp_boundary",
]);

export interface EscalationCoverage {
  total: number;
  eres2net: number;
  pyannote: number;
  reviewRequired: number;
  overlap: number;
  lowMargin: number;
}

export function deriveEscalationCoverage(
  reviews: readonly ReviewSegment[],
): EscalationCoverage {
  return {
    total: reviews.length,
    eres2net: reviews.filter((review) =>
      review.reasons.some((reason) => eres2NetReasons.has(reason)),
    ).length,
    pyannote: reviews.filter((review) =>
      review.reasons.some((reason) => pyannoteReasons.has(reason)),
    ).length,
    reviewRequired: reviews.length,
    overlap: reviews.filter((review) =>
      review.reasons.includes("overlap_detected"),
    ).length,
    lowMargin: reviews.filter((review) =>
      review.reasons.includes("speaker_close_score"),
    ).length,
  };
}
