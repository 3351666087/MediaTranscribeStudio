import { useMemo, useState, type CSSProperties } from "react";
import type {
  ReviewDecision,
  ReviewReason,
  ReviewSegment,
  SpeakerProfile,
} from "../contracts/studio";
import { useI18n, type MessageKey } from "../i18n";
import { cx, formatPercent } from "../lib/format";
import { EscalationTrace } from "./EscalationTrace";
import { Icon } from "./Icon";
import { StatusBadge } from "./StatusBadge";

const reviewMessageKey = (key: MessageKey): MessageKey => key;

const reasonLabelKeys: Record<ReviewReason, MessageKey> = {
  speaker_close_score: reviewMessageKey("review.reason.speakerCloseScore"),
  speaker_count_uncertain: reviewMessageKey(
    "review.reason.speakerCountUncertain",
  ),
  overlap_detected: reviewMessageKey("review.reason.overlapDetected"),
  timestamp_boundary: reviewMessageKey("review.reason.timestampBoundary"),
  speaker_outlier: reviewMessageKey("review.reason.speakerOutlier"),
  local_audio_review: reviewMessageKey("review.reason.localAudioReview"),
};

const coreReviewSignals = [
  {
    reason: "speaker_close_score",
    labelKey: reviewMessageKey("review.signal.acousticConflict"),
  },
  {
    reason: "speaker_count_uncertain",
    labelKey: reviewMessageKey("review.signal.countUncertainty"),
  },
  {
    reason: "overlap_detected",
    labelKey: reviewMessageKey("review.signal.overlap"),
  },
  {
    reason: "local_audio_review",
    labelKey: reviewMessageKey("review.signal.audioReview"),
  },
] satisfies ReadonlyArray<{
  reason: ReviewReason;
  labelKey: MessageKey;
}>;

const REVIEW_PAGE_SIZE = 40;

function clampPage(value: number, totalPages: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.min(totalPages, Math.max(1, Math.trunc(value)));
}

interface ReviewQueueProps {
  reviews: ReviewSegment[];
  speakers: SpeakerProfile[];
  busyAction: string | null;
  onApply: (decision: ReviewDecision) => Promise<void>;
  onNotify: (
    tone: "success" | "warning" | "error" | "info",
    title: string,
    detail: string,
  ) => void;
}

export function ReviewQueue({
  reviews,
  speakers,
  busyAction,
  onApply,
  onNotify,
}: ReviewQueueProps) {
  const { t } = useI18n();
  const [selectedId, setSelectedId] = useState(reviews[0]?.id ?? "");
  const [page, setPage] = useState(1);
  const speakerById = useMemo(
    () => new Map(speakers.map((speaker) => [speaker.id, speaker])),
    [speakers],
  );
  const { reviewById, reasonCountsByReason, lockedCount } = useMemo(() => {
    const nextReviewById = new Map<ReviewSegment["id"], ReviewSegment>();
    const nextReasonCounts = new Map<ReviewReason, number>();
    let nextLockedCount = 0;

    for (const review of reviews) {
      nextReviewById.set(review.id, review);
      if (review.locked) {
        nextLockedCount += 1;
      }
      for (const reason of review.reasons) {
        nextReasonCounts.set(reason, (nextReasonCounts.get(reason) ?? 0) + 1);
      }
    }

    return {
      reviewById: nextReviewById,
      reasonCountsByReason: nextReasonCounts,
      lockedCount: nextLockedCount,
    };
  }, [reviews]);
  const activeReview = reviewById.get(selectedId) ?? reviews[0];
  const reasonCounts = coreReviewSignals.map((signal) => ({
    ...signal,
    count: reasonCountsByReason.get(signal.reason) ?? 0,
  }));
  const totalPages = Math.max(1, Math.ceil(reviews.length / REVIEW_PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageStartIndex = (currentPage - 1) * REVIEW_PAGE_SIZE;
  const pageReviews = reviews.slice(
    pageStartIndex,
    pageStartIndex + REVIEW_PAGE_SIZE,
  );
  const pageRangeStart = pageReviews.length > 0 ? pageStartIndex + 1 : 0;
  const pageRangeEnd = pageStartIndex + pageReviews.length;
  const goToPage = (nextPage: number) => {
    const clampedPage = clampPage(nextPage, totalPages);
    const firstReviewOnPage =
      reviews[(clampedPage - 1) * REVIEW_PAGE_SIZE];
    setPage(clampedPage);
    setSelectedId(firstReviewOnPage.id);
  };

  if (reviews.length === 0) {
    return (
      <section className="empty-state" aria-labelledby="review-empty-title">
        <span className="empty-state__illustration" aria-hidden="true">
          <Icon name="check" size={34} />
        </span>
        <span className="panel__eyebrow">
          {t(reviewMessageKey("review.empty.eyebrow"))}
        </span>
        <h1 id="review-empty-title">
          {t(reviewMessageKey("review.empty.title"))}
        </h1>
        <p>{t(reviewMessageKey("review.empty.detail"))}</p>
      </section>
    );
  }

  return (
    <div className="review-workspace">
      <section className="review-queue-list" aria-labelledby="review-queue-title">
        <div className="review-queue-list__header">
          <div>
            <span className="panel__eyebrow">
              {t(reviewMessageKey("review.queue.eyebrow"))}
            </span>
            <h1 id="review-queue-title">
              {t(reviewMessageKey("review.queue.title"))}
            </h1>
          </div>
          <span className="count-chip count-chip--warning">
            {t(reviewMessageKey("review.queue.itemCount"), {
              count: reviews.length,
            })}
          </span>
        </div>
        <p className="review-queue-list__intro">
          {t(reviewMessageKey("review.queue.intro"))}
        </p>

        <section className="review-summary" aria-labelledby="review-summary-title">
          <div className="review-summary__heading">
            <strong id="review-summary-title">
              {t(reviewMessageKey("review.summary.title"))}
            </strong>
            <span>
              {t(reviewMessageKey("review.summary.locked"), {
                count: lockedCount,
              })}
            </span>
          </div>
          <ul>
            {reasonCounts.map(({ reason, labelKey, count }) => (
              <li key={reason}>
                <span>{t(labelKey)}</span>
                <strong>{count}</strong>
              </li>
            ))}
          </ul>
        </section>

        <ol
          className="review-tabs"
          aria-label={t(reviewMessageKey("review.tabs.aria"))}
        >
          {pageReviews.map((review, index) => {
            const speaker = speakerById.get(review.currentSpeakerId);
            const active = activeReview.id === review.id;
            const globalIndex = pageStartIndex + index;
            return (
              <li key={review.id}>
                <button
                  id={`review-tab-${review.id}`}
                  type="button"
                  className={cx("review-tab", active && "review-tab--active")}
                  aria-pressed={active}
                  aria-label={t(reviewMessageKey("review.tabs.open"), {
                    index: globalIndex + 1,
                    timestamp: review.timestampLabel,
                  })}
                  aria-controls="review-editor"
                  onClick={() => setSelectedId(review.id)}
                >
                  <span className="review-tab__number">
                    {String(globalIndex + 1).padStart(2, "0")}
                  </span>
                  <span className="review-tab__copy">
                    <strong>{review.timestampLabel}</strong>
                    <small>
                      {t(reviewMessageKey("review.tabs.speakerConfidence"), {
                        speaker: speaker?.label ?? review.currentSpeakerId,
                        confidence: formatPercent(review.confidence),
                      })}
                    </small>
                    <span className="review-tab__reason-row">
                      {review.reasons.slice(0, 2).map((reason) => (
                        <span className="review-reason-chip" key={reason}>
                          {t(reasonLabelKeys[reason])}
                        </span>
                      ))}
                      {review.locked ? (
                        <span className="review-reason-chip review-reason-chip--locked">
                          {t(reviewMessageKey("review.tabs.humanLocked"))}
                        </span>
                      ) : null}
                    </span>
                    <span className="review-tab__excerpt">{review.normalizedText}</span>
                  </span>
                  <Icon name="arrow-right" size={17} />
                </button>
              </li>
            );
          })}
        </ol>
        <nav
          className="collection-pagination"
          aria-label={t(reviewMessageKey("review.pagination.aria"))}
        >
          <span className="collection-pagination__summary" role="status">
            {t(reviewMessageKey("review.pagination.summary"), {
              start: pageRangeStart,
              end: pageRangeEnd,
              total: reviews.length,
            })}
          </span>
          <div className="collection-pagination__controls">
            <button
              id="review-page-previous"
              className="collection-pagination__button"
              type="button"
              disabled={currentPage === 1}
              aria-label={t(reviewMessageKey("review.pagination.previousAria"))}
              onClick={() => goToPage(currentPage - 1)}
            >
              {t(reviewMessageKey("review.pagination.previous"))}
            </button>
            <label className="collection-pagination__page" htmlFor="review-page-input">
              <span>{t(reviewMessageKey("review.pagination.page"))}</span>
              <input
                id="review-page-input"
                type="number"
                min={1}
                max={totalPages}
                step={1}
                value={currentPage}
                aria-label={t(reviewMessageKey("review.pagination.pageAria"))}
                onChange={(event) => {
                  if (Number.isFinite(event.currentTarget.valueAsNumber)) {
                    goToPage(event.currentTarget.valueAsNumber);
                  }
                }}
              />
              <span aria-live="polite">
                {t(reviewMessageKey("review.pagination.of"), { totalPages })}
              </span>
            </label>
            <button
              id="review-page-next"
              className="collection-pagination__button"
              type="button"
              disabled={currentPage === totalPages}
              aria-label={t(reviewMessageKey("review.pagination.nextAria"))}
              onClick={() => goToPage(currentPage + 1)}
            >
              {t(reviewMessageKey("review.pagination.next"))}
            </button>
          </div>
        </nav>
      </section>

      <ReviewEditor
        key={activeReview.id}
        review={activeReview}
        speakers={speakers}
        speakerById={speakerById}
        reviews={reviews}
        busy={busyAction === activeReview.id}
        onNotify={onNotify}
        onApply={onApply}
      />
    </div>
  );
}

interface ReviewEditorProps {
  review: ReviewSegment;
  reviews: ReviewSegment[];
  speakers: SpeakerProfile[];
  speakerById: ReadonlyMap<SpeakerProfile["id"], SpeakerProfile>;
  busy: boolean;
  onApply: (decision: ReviewDecision) => Promise<void>;
  onNotify: ReviewQueueProps["onNotify"];
}

function ReviewEditor({
  review,
  reviews,
  speakers,
  speakerById,
  busy,
  onApply,
  onNotify,
}: ReviewEditorProps) {
  const { t } = useI18n();
  const [speakerId, setSpeakerId] = useState<SpeakerProfile["id"]>(review.currentSpeakerId);
  const [confirmedText, setConfirmedText] = useState(review.normalizedText);
  const [suggestionDraft, setSuggestionDraft] = useState(review.normalizedText);
  const [suggestionAccepted, setSuggestionAccepted] = useState(false);
  const [reason, setReason] = useState("");
  const [evidence, setEvidence] = useState("");
  const [confidenceInput, setConfidenceInput] = useState("");
  const [splitOffset, setSplitOffset] = useState(
    ((review.endMs - review.startMs) / 2_000).toFixed(2),
  );
  const [splitReason, setSplitReason] = useState("");
  const [splitEvidence, setSplitEvidence] = useState("");
  const [splitConfidence, setSplitConfidence] = useState("");
  const [splitDraft, setSplitDraft] = useState<{
    offset: number;
    reason: string;
    evidence: string;
    confidence: number;
  } | null>(null);
  const suggestionDirty = suggestionDraft !== confirmedText;
  const confidence =
    confidenceInput.trim().length === 0 ? Number.NaN : Number(confidenceInput);
  const canSubmit =
    confirmedText.trim().length > 0 &&
    !suggestionDirty &&
    reason.trim().length > 0 &&
    evidence.trim().length > 0 &&
    Number.isFinite(confidence) &&
    confidence >= 0 &&
    confidence <= 1;
  const durationSeconds = (review.endMs - review.startMs) / 1_000;
  const parsedSplitOffset = Number(splitOffset);
  const parsedSplitConfidence = Number(splitConfidence);
  const canCreateSplitDraft =
    Number.isFinite(parsedSplitOffset) &&
    parsedSplitOffset > 0 &&
    parsedSplitOffset < durationSeconds &&
    splitReason.trim().length > 0 &&
    splitEvidence.trim().length > 0 &&
    splitConfidence.trim().length > 0 &&
    Number.isFinite(parsedSplitConfidence) &&
    parsedSplitConfidence >= 0 &&
    parsedSplitConfidence <= 1;

  return (
    <section id="review-editor" className="review-editor" aria-labelledby="review-editor-title">
      <div className="review-required-ribbon" role="status">
        <Icon name="alert" size={16} />
        <strong>{t(reviewMessageKey("review.editor.requiredCode"))}</strong>
        <span>{t(reviewMessageKey("review.editor.requiredDetail"))}</span>
      </div>

      <div className="review-editor__header">
        <div>
          <span className="panel__eyebrow">
            {t(reviewMessageKey("review.editor.segmentLabel"), {
              segment: review.id.replace("review-", "#"),
            })}
          </span>
          <h2 id="review-editor-title">{review.timestampLabel}</h2>
        </div>
        <div className="review-editor__states">
          {review.locked ? (
            <StatusBadge
              status="success"
              label={t(reviewMessageKey("review.editor.lockedBadge"))}
            />
          ) : null}
          <StatusBadge
            status={review.confidenceBand === "low" ? "warning" : "info"}
            label={t(reviewMessageKey("review.editor.confidenceBadge"), {
              confidence: formatPercent(review.confidence),
            })}
          />
        </div>
      </div>

      <div
        className="reason-list"
        aria-label={t(reviewMessageKey("review.editor.reasonsAria"))}
      >
        {review.reasons.map((reason) => (
          <span key={reason}>{t(reasonLabelKeys[reason])}</span>
        ))}
      </div>

      <EscalationTrace reviews={reviews} review={review} compact />

      <div className="audio-review">
        <button
          id={`review-audio-preview-${review.id}`}
          className="audio-review__button"
          type="button"
          disabled
          aria-describedby={`audio-status-${review.id}`}
        >
          <Icon name="play" size={19} />
          <span className="sr-only">
            {t(reviewMessageKey("review.audio.play"))}
          </span>
        </button>
        <div
          className="waveform"
          aria-label={t(reviewMessageKey("review.audio.waveform"))}
          role="img"
        >
          {review.waveform.map((height, index) => (
            <span
              key={`${review.id}-${index}`}
              style={{ height: `${Math.max(16, height)}%` }}
              aria-hidden="true"
            />
          ))}
        </div>
        <span className="audio-review__time">
          {t(reviewMessageKey("review.audio.duration"), {
            seconds: ((review.endMs - review.startMs) / 1000).toFixed(2),
          })}
        </span>
        <span className="audio-review__unavailable" id={`audio-status-${review.id}`}>
          {t(reviewMessageKey("review.audio.unavailable"))}
        </span>
      </div>

      <div className="transcript-compare">
        <div className="transcript-compare__immutable">
          <span>
            {t(reviewMessageKey("review.transcript.rawLabel"))}
            <Icon name="lock" size={13} />
          </span>
          <p>{review.rawText}</p>
          <small>{t(reviewMessageKey("review.transcript.rawDetail"))}</small>
        </div>
        <div className="transcript-compare__confirmed">
          <span>{t(reviewMessageKey("review.transcript.confirmedLabel"))}</span>
          <p>{confirmedText}</p>
          <small>
            {t(reviewMessageKey("review.transcript.confirmedDetail"))}
          </small>
        </div>
        <label
          className={cx(suggestionDirty && "transcript-suggestion--dirty")}
          htmlFor={`review-suggestion-${review.id}`}
        >
          <span>
            {t(reviewMessageKey("review.transcript.draftLabel"))}
            {suggestionDirty ? (
              <em>
                {t(reviewMessageKey("review.transcript.awaitingAcceptance"))}
              </em>
            ) : null}
          </span>
          <textarea
            id={`review-suggestion-${review.id}`}
            value={suggestionDraft}
            rows={3}
            aria-label={t(reviewMessageKey("review.transcript.draftLabel"))}
            onChange={(event) => {
              setSuggestionDraft(event.target.value);
              setSuggestionAccepted(false);
            }}
          />
          <small>{t(reviewMessageKey("review.transcript.draftDetail"))}</small>
          {suggestionDirty ? (
            <span className="transcript-suggestion__actions">
              <button
                id={`review-accept-suggestion-${review.id}`}
                className="mini-action mini-action--active"
                type="button"
                disabled={!suggestionDraft.trim()}
                onClick={() => {
                  const nextText = suggestionDraft.trim();
                  setConfirmedText(nextText);
                  setSuggestionDraft(nextText);
                  setSuggestionAccepted(true);
                }}
              >
                {t(reviewMessageKey("review.transcript.accept"))}
              </button>
              <button
                id={`review-discard-suggestion-${review.id}`}
                className="mini-action"
                type="button"
                onClick={() => {
                  setSuggestionDraft(confirmedText);
                  setSuggestionAccepted(false);
                }}
              >
                {t(reviewMessageKey("review.transcript.discard"))}
              </button>
            </span>
          ) : suggestionAccepted ? (
            <span className="transcript-suggestion__accepted">
              <Icon name="check" size={13} />
              {t(reviewMessageKey("review.transcript.accepted"))}
            </span>
          ) : null}
        </label>
      </div>

      <fieldset className="candidate-fieldset">
        <legend>
          <span>{t(reviewMessageKey("review.speaker.confirmTitle"))}</span>
          <small>
            {review.locked
              ? t(reviewMessageKey("review.speaker.lockedDetail"))
              : t(reviewMessageKey("review.speaker.confirmDetail"))}
          </small>
        </legend>
        <div className="candidate-grid">
          {review.candidates.map((candidate) => {
            const speaker = speakerById.get(candidate.speakerId);
            if (!speaker) {
              return null;
            }
            const candidateId = `review-candidate-${review.id}-${candidate.speakerId}`;
            return (
              <label
                id={`review-candidate-label-${review.id}-${candidate.speakerId}`}
                className={cx(
                  "candidate-card",
                  speakerId === candidate.speakerId && "candidate-card--selected",
                )}
                htmlFor={candidateId}
                key={candidate.speakerId}
              >
                <input
                  id={candidateId}
                  type="radio"
                  name={`speaker-${review.id}`}
                  aria-label={t(
                    reviewMessageKey("review.speaker.candidateAria"),
                    {
                      speaker: speaker.label,
                      confidence: formatPercent(candidate.score),
                    },
                  )}
                  value={candidate.speakerId}
                  checked={speakerId === candidate.speakerId}
                  disabled={review.locked}
                  onChange={() => setSpeakerId(candidate.speakerId)}
                />
                <span
                  className="speaker-avatar speaker-avatar--small"
                  style={{ "--speaker-color": speaker.color } as CSSProperties}
                  aria-hidden="true"
                >
                  {speaker.shortLabel}
                </span>
                <span className="candidate-card__copy">
                  <strong>{speaker.label}</strong>
                  <small>{candidate.evidence}</small>
                </span>
                <span className="candidate-card__score">{formatPercent(candidate.score)}</span>
              </label>
            );
          })}
        </div>
        <label
          className="speaker-track-picker"
          htmlFor={`review-speaker-select-${review.id}`}
        >
          <span>{t(reviewMessageKey("review.speaker.allTracks"))}</span>
          <select
            id={`review-speaker-select-${review.id}`}
            aria-label={t(reviewMessageKey("review.speaker.allTracks"))}
            value={speakerId}
            disabled={review.locked}
            onChange={(event) =>
              setSpeakerId(event.target.value as SpeakerProfile["id"])
            }
          >
            {speakers.map((speaker) => (
              <option key={speaker.id} value={speaker.id}>
                {t(reviewMessageKey("review.speaker.option"), {
                  id: speaker.id,
                  speaker: speaker.label,
                })}
              </option>
            ))}
          </select>
          <small>
            {t(reviewMessageKey("review.speaker.allTracksDetail"), {
              count: speakers.length,
            })}
          </small>
        </label>
      </fieldset>

      <div
        className="audit-fields"
        aria-label={t(reviewMessageKey("review.audit.aria"))}
      >
        <label className="audit-note" htmlFor={`review-reason-${review.id}`}>
          <span>{t(reviewMessageKey("review.audit.reason"))}</span>
          <textarea
            id={`review-reason-${review.id}`}
            aria-label={t(reviewMessageKey("review.audit.reason"))}
            aria-required="true"
            value={reason}
            rows={2}
            onChange={(event) => setReason(event.target.value)}
            placeholder={t(
              reviewMessageKey("review.audit.reasonPlaceholder"),
            )}
          />
        </label>
        <label className="audit-note" htmlFor={`review-evidence-${review.id}`}>
          <span>{t(reviewMessageKey("review.audit.evidence"))}</span>
          <textarea
            id={`review-evidence-${review.id}`}
            aria-label={t(reviewMessageKey("review.audit.evidence"))}
            aria-required="true"
            value={evidence}
            rows={2}
            onChange={(event) => setEvidence(event.target.value)}
            placeholder={t(
              reviewMessageKey("review.audit.evidencePlaceholder"),
            )}
          />
        </label>
        <label
          className="audit-confidence"
          htmlFor={`review-confidence-${review.id}`}
        >
          <span>{t(reviewMessageKey("review.audit.confidence"))}</span>
          <input
            id={`review-confidence-${review.id}`}
            type="number"
            aria-label={t(reviewMessageKey("review.audit.confidence"))}
            aria-required="true"
            min="0"
            max="1"
            step="0.01"
            inputMode="decimal"
            value={confidenceInput}
            onChange={(event) => setConfidenceInput(event.target.value)}
            placeholder={t(
              reviewMessageKey("review.audit.confidencePlaceholder"),
            )}
          />
          <small>{t(reviewMessageKey("review.audit.confidenceDetail"))}</small>
        </label>
      </div>

      <details className="split-draft">
        <summary id={`review-split-summary-${review.id}`}>
          <span>
            <Icon name="wave" size={16} />
            {t(reviewMessageKey("review.split.title"))}
          </span>
          <strong>{t(reviewMessageKey("review.split.code"))}</strong>
        </summary>
        <p>{t(reviewMessageKey("review.split.detail"))}</p>
        <div className="split-draft__grid">
          <label htmlFor={`review-split-offset-${review.id}`}>
            <span>{t(reviewMessageKey("review.split.offset"))}</span>
            <input
              id={`review-split-offset-${review.id}`}
              type="number"
              aria-label={t(reviewMessageKey("review.split.offsetAria"))}
              min="0.01"
              max={Math.max(0.01, durationSeconds - 0.01)}
              step="0.01"
              value={splitOffset}
              onChange={(event) => setSplitOffset(event.target.value)}
            />
          </label>
          <label htmlFor={`review-split-confidence-${review.id}`}>
            <span>{t(reviewMessageKey("review.audit.confidence"))}</span>
            <input
              id={`review-split-confidence-${review.id}`}
              type="number"
              aria-label={t(
                reviewMessageKey("review.split.confidenceAria"),
              )}
              min="0"
              max="1"
              step="0.01"
              value={splitConfidence}
              onChange={(event) => setSplitConfidence(event.target.value)}
            />
          </label>
          <label htmlFor={`review-split-reason-${review.id}`}>
            <span>{t(reviewMessageKey("review.split.reason"))}</span>
            <textarea
              id={`review-split-reason-${review.id}`}
              aria-label={t(reviewMessageKey("review.split.reason"))}
              rows={2}
              value={splitReason}
              onChange={(event) => setSplitReason(event.target.value)}
            />
          </label>
          <label htmlFor={`review-split-evidence-${review.id}`}>
            <span>{t(reviewMessageKey("review.split.evidence"))}</span>
            <textarea
              id={`review-split-evidence-${review.id}`}
              aria-label={t(reviewMessageKey("review.split.evidenceAria"))}
              rows={2}
              value={splitEvidence}
              onChange={(event) => setSplitEvidence(event.target.value)}
            />
          </label>
        </div>
        <button
          id={`review-create-split-draft-${review.id}`}
          className="button button--soft"
          type="button"
          disabled={!canCreateSplitDraft}
          onClick={() => {
            setSplitDraft({
              offset: parsedSplitOffset,
              reason: splitReason.trim(),
              evidence: splitEvidence.trim(),
              confidence: parsedSplitConfidence,
            });
            onNotify(
              "info",
              t(reviewMessageKey("review.split.notifyTitle")),
              t(reviewMessageKey("review.split.notifyDetail")),
            );
          }}
        >
          {t(reviewMessageKey("review.split.create"))}
        </button>
        {splitDraft ? (
          <div className="draft-only-card" role="status">
            <strong>
              {t(reviewMessageKey("review.split.status"), {
                offset: splitDraft.offset.toFixed(2),
              })}
            </strong>
            <span>{splitDraft.reason}</span>
            <small>
              {t(reviewMessageKey("review.split.evidenceSummary"), {
                evidence: splitDraft.evidence,
                confidence: formatPercent(splitDraft.confidence),
              })}
            </small>
          </div>
        ) : null}
      </details>

      <div className="review-editor__guardrail">
        <Icon name="lock" size={18} />
        <div>
          <strong>{t(reviewMessageKey("review.guardrails.title"))}</strong>
          <ul className="guardrail-stack">
            <li>{t(reviewMessageKey("review.guardrails.localModel"))}</li>
            <li>{t(reviewMessageKey("review.guardrails.transcript"))}</li>
            <li>{t(reviewMessageKey("review.guardrails.speaker"))}</li>
            <li>{t(reviewMessageKey("review.guardrails.lock"))}</li>
          </ul>
        </div>
      </div>

      <div className="review-editor__actions">
        <button
          id={`review-audio-sidecar-${review.id}`}
          className="button button--soft"
          type="button"
          disabled
          aria-describedby={`audio-status-${review.id}`}
        >
          <Icon name="headphones" size={17} />
          {t(reviewMessageKey("review.actions.audio"))}
        </button>
        <button
          id={`review-submit-${review.id}`}
          className="button button--primary"
          type="button"
          disabled={busy || !canSubmit}
          onClick={() => {
            onApply({
              reviewId: review.id,
              speakerId,
              normalizedText: confirmedText.trim(),
              reason: reason.trim(),
              evidence: evidence.trim(),
              confidence,
            }).catch((error: unknown) => {
              onNotify(
                "error",
                t(reviewMessageKey("review.actions.saveErrorTitle")),
                error instanceof Error
                  ? error.message
                  : t(reviewMessageKey("review.actions.unknownError")),
              );
            });
          }}
        >
          <Icon name="lock" size={17} />
          {busy
            ? t(reviewMessageKey("review.actions.saving"))
            : t(reviewMessageKey("review.actions.confirm"))}
        </button>
      </div>
    </section>
  );
}
