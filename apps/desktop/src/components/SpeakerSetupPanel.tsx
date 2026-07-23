import { useMemo, useState, type CSSProperties } from "react";
import type {
  ReviewSegment,
  SpeakerCountDetection,
  SpeakerCountPolicy,
  SpeakerProfile,
  UpdateSpeakerRequest,
} from "../contracts/studio";
import {
  useI18n,
  type MessageKey,
  type MessageParams,
  type Translate,
} from "../i18n";
import { Icon } from "./Icon";
import { SpeakerDetectionSummary } from "./SpeakerDetectionSummary";
import { StatusBadge } from "./StatusBadge";

interface SpeakerSetupPanelProps {
  jobId: string;
  speakers: SpeakerProfile[];
  reviews?: ReviewSegment[];
  speakerPolicy: SpeakerCountPolicy;
  speakerDetection: SpeakerCountDetection | null;
  busyAction: string | null;
  onUpdate: (request: UpdateSpeakerRequest) => Promise<void>;
}

type SpeakerView = "all" | "needs_review" | "locked" | "pending";

const SPEAKER_PAGE_SIZE = 40;

function clampPage(value: number, totalPages: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.min(totalPages, Math.max(1, Math.trunc(value)));
}

interface GovernanceDraft {
  id: string;
  kind: "merge" | "split";
  title: string;
  detail: string;
  reason: string;
  evidence: string;
  confidence: number;
}

function speakerMessage(
  t: Translate,
  key: MessageKey,
  params?: MessageParams,
): string {
  return t(key, params);
}

function policySummary(t: Translate, policy: SpeakerCountPolicy): string {
  if (policy.mode === "auto") {
    return speakerMessage(t, "speakerSetup.policy.auto");
  }
  if (policy.mode === "manual") {
    return speakerMessage(t, "speakerSetup.policy.manual", {
      count: policy.count,
    });
  }
  return speakerMessage(t, "speakerSetup.policy.hybrid", {
    min: policy.minSpeakers,
    max: policy.maxSpeakers,
    prior: policy.priorCount,
  });
}

export function SpeakerSetupPanel({
  jobId,
  speakers,
  ...props
}: SpeakerSetupPanelProps) {
  const rosterIdentity = `${jobId}\u0000${speakers
    .map((speaker) => speaker.id)
    .join("\u0000")}`;

  return (
    <SpeakerSetupPanelEditor
      key={rosterIdentity}
      speakers={speakers}
      {...props}
    />
  );
}

function SpeakerSetupPanelEditor({
  speakers,
  reviews = [],
  speakerPolicy,
  speakerDetection,
  busyAction,
  onUpdate,
}: Omit<SpeakerSetupPanelProps, "jobId">) {
  const { t } = useI18n();
  const text = (key: MessageKey, params?: MessageParams) =>
    speakerMessage(t, key, params);
  const [draftLabels, setDraftLabels] = useState<Record<string, string>>({});
  const [query, setQuery] = useState("");
  const [view, setView] = useState<SpeakerView>("all");
  const [page, setPage] = useState(1);
  const [mergeSource, setMergeSource] = useState<SpeakerProfile["id"]>(
    speakers[0]?.id ?? "speaker-1",
  );
  const firstSpeakerId = speakers[0]?.id ?? "speaker-1";
  const secondSpeakerId =
    speakers.find((_, index) => index === 1)?.id ?? firstSpeakerId;
  const [mergeTarget, setMergeTarget] = useState<SpeakerProfile["id"]>(
    secondSpeakerId,
  );
  const [mergeReason, setMergeReason] = useState("");
  const [mergeEvidence, setMergeEvidence] = useState("");
  const [mergeConfidence, setMergeConfidence] = useState("");
  const [splitSpeaker, setSplitSpeaker] = useState<SpeakerProfile["id"]>(
    speakers[0]?.id ?? "speaker-1",
  );
  const [splitAnchor, setSplitAnchor] = useState("");
  const [splitReason, setSplitReason] = useState("");
  const [splitEvidence, setSplitEvidence] = useState("");
  const [splitConfidence, setSplitConfidence] = useState("");
  const [governanceDrafts, setGovernanceDrafts] = useState<GovernanceDraft[]>([]);
  const [governanceOpen, setGovernanceOpen] = useState(false);

  const speakerMetadata = useMemo(() => {
    const metadata = new Map<
      SpeakerProfile["id"],
      {
        speaker: SpeakerProfile;
        index: number;
        difficult: number;
        overlap: number;
        lowMargin: number;
      }
    >();

    for (const [index, speaker] of speakers.entries()) {
      metadata.set(speaker.id, {
        speaker,
        index,
        difficult: 0,
        overlap: 0,
        lowMargin: 0,
      });
    }

    for (const review of reviews) {
      const entry = metadata.get(review.currentSpeakerId);
      if (!entry) {
        continue;
      }
      entry.difficult += 1;
      if (review.reasons.includes("overlap_detected")) {
        entry.overlap += 1;
      }
      if (review.reasons.includes("speaker_close_score")) {
        entry.lowMargin += 1;
      }
    }

    return metadata;
  }, [reviews, speakers]);

  const effectiveMergeSource = speakerMetadata.has(mergeSource)
    ? mergeSource
    : firstSpeakerId;
  const effectiveMergeTarget = speakerMetadata.has(mergeTarget)
    ? mergeTarget
    : secondSpeakerId;
  const effectiveSplitSpeaker = speakerMetadata.has(splitSpeaker)
    ? splitSpeaker
    : firstSpeakerId;

  const visibleSpeakers = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    return speakers.filter((speaker) => {
      const matchesQuery =
        normalizedQuery.length === 0 ||
        [speaker.id, speaker.label, speaker.roleHint]
          .join(" ")
          .toLocaleLowerCase()
          .includes(normalizedQuery);
      const matchesView =
        view === "all" ||
        (view === "needs_review" && speaker.reviewStatus === "needs_review") ||
        (view === "locked" && speaker.locked) ||
        (view === "pending" && speaker.reviewStatus === "pending");
      return matchesQuery && matchesView;
    });
  }, [query, speakers, view]);

  const totalPages = Math.max(1, Math.ceil(visibleSpeakers.length / SPEAKER_PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageStartIndex = (currentPage - 1) * SPEAKER_PAGE_SIZE;
  const pageSpeakers = visibleSpeakers.slice(
    pageStartIndex,
    pageStartIndex + SPEAKER_PAGE_SIZE,
  );
  const pageRangeStart = pageSpeakers.length > 0 ? pageStartIndex + 1 : 0;
  const pageRangeEnd = pageStartIndex + pageSpeakers.length;
  const goToPage = (nextPage: number) => {
    setPage(clampPage(nextPage, totalPages));
  };

  const parsedMergeConfidence = Number(mergeConfidence);
  const canCreateMergeDraft =
    effectiveMergeSource !== effectiveMergeTarget &&
    mergeReason.trim().length > 0 &&
    mergeEvidence.trim().length > 0 &&
    mergeConfidence.trim().length > 0 &&
    Number.isFinite(parsedMergeConfidence) &&
    parsedMergeConfidence >= 0 &&
    parsedMergeConfidence <= 1;
  const parsedSplitConfidence = Number(splitConfidence);
  const canCreateSplitDraft =
    splitAnchor.trim().length > 0 &&
    splitReason.trim().length > 0 &&
    splitEvidence.trim().length > 0 &&
    splitConfidence.trim().length > 0 &&
    Number.isFinite(parsedSplitConfidence) &&
    parsedSplitConfidence >= 0 &&
    parsedSplitConfidence <= 1;

  const speakerLabel = (speakerId: SpeakerProfile["id"]) =>
    speakerMetadata.get(speakerId)?.speaker.label ?? speakerId;

  const submitUpdate = async (
    speaker: SpeakerProfile,
    patch: Partial<Pick<UpdateSpeakerRequest, "locked" | "reviewStatus">> = {},
  ) => {
    const label = (draftLabels[speaker.id] ?? speaker.label).trim();
    if (!label) {
      return;
    }
    await onUpdate({
      speakerId: speaker.id,
      label,
      locked: patch.locked ?? speaker.locked,
      reviewStatus: patch.reviewStatus ?? speaker.reviewStatus,
    });
  };
  const reportUpdateError = (error: unknown) => {
    console.error("Failed to update speaker", error);
  };

  return (
    <section className="panel speaker-panel" aria-labelledby="speaker-panel-title">
      <div className="panel__header">
        <div>
          <span className="panel__eyebrow">
            {text("speakerSetup.header.eyebrow")}
          </span>
          <h2 id="speaker-panel-title">{text("speakerSetup.header.title")}</h2>
        </div>
        <span
          className="count-chip"
          aria-label={text("speakerSetup.header.countAria", {
            count: speakers.length,
          })}
        >
          {text("speakerSetup.header.count", { count: speakers.length })}
        </span>
      </div>

      <p className="panel__intro">
        {policySummary(t, speakerPolicy)}.{" "}
        {text("speakerSetup.intro.continuousIds")}{" "}
        <strong>{text("speakerSetup.intro.idRange")}</strong>.{" "}
        {text("speakerSetup.intro.saveBoundary")}
      </p>

      <SpeakerDetectionSummary
        detection={speakerPolicy.mode === "manual" ? null : speakerDetection}
        policy={speakerPolicy}
        compact
      />

      <div className="speaker-toolbar">
        <label className="speaker-search" htmlFor="speaker-search-input">
          <span className="sr-only">{text("speakerSetup.search.label")}</span>
          <Icon name="review" size={15} />
          <input
            id="speaker-search-input"
            type="search"
            aria-label={text("speakerSetup.search.label")}
            value={query}
            placeholder={text("speakerSetup.search.placeholder")}
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(1);
            }}
          />
        </label>
        <label className="speaker-view-filter" htmlFor="speaker-view-filter">
          <span>{text("speakerSetup.filter.label")}</span>
          <select
            id="speaker-view-filter"
            aria-label={text("speakerSetup.filter.aria")}
            value={view}
            onChange={(event) => {
              setView(event.target.value as SpeakerView);
              setPage(1);
            }}
          >
            <option value="all">{text("speakerSetup.filter.all")}</option>
            <option value="needs_review">
              {text("speakerSetup.filter.needsReview")}
            </option>
            <option value="locked">
              {text("speakerSetup.filter.locked")}
            </option>
            <option value="pending">
              {text("speakerSetup.filter.pending")}
            </option>
          </select>
        </label>
        <span className="speaker-toolbar__count" role="status">
          {text("speakerSetup.filter.showing", {
            visible: visibleSpeakers.length,
            total: speakers.length,
          })}
        </span>
      </div>
      <p className="speaker-toolbar__guardrail">
        {text("speakerSetup.filter.guardrail")}
      </p>

      <div
        className="speaker-list"
        aria-label={text("speakerSetup.list.aria")}
        role="list"
      >
        {pageSpeakers.map((speaker) => {
          const metadata = speakerMetadata.get(speaker.id);
          const originalIndex = metadata?.index ?? 0;
          const counts = metadata ?? {
            difficult: 0,
            overlap: 0,
            lowMargin: 0,
          };
          return (
            <article
              className={speaker.locked ? "speaker-row speaker-row--locked" : "speaker-row"}
              key={speaker.id}
              role="listitem"
            >
              <span
                className="speaker-avatar"
                style={{ "--speaker-color": speaker.color } as CSSProperties}
                aria-hidden="true"
              >
                {speaker.shortLabel}
              </span>
              <label
                className="speaker-row__field"
                htmlFor={`speaker-label-${speaker.id}`}
              >
                <span className="sr-only">
                  {text("speakerSetup.row.nameForIndex", {
                    index: originalIndex + 1,
                  })}
                </span>
                <input
                  id={`speaker-label-${speaker.id}`}
                  value={draftLabels[speaker.id] ?? speaker.label}
                  maxLength={64}
                  aria-label={text("speakerSetup.row.nameAria", {
                    speakerId: speaker.id,
                  })}
                  disabled={speaker.locked || busyAction === speaker.id}
                  onChange={(event) =>
                    setDraftLabels((current) => ({
                      ...current,
                      [speaker.id]: event.target.value,
                    }))
                  }
                />
                <small>
                  {text("speakerSetup.row.identity", {
                    speakerId: speaker.id,
                    roleHint: speaker.roleHint,
                  })}
                </small>
                <span
                  className="speaker-row__evidence"
                  aria-label={text("speakerSetup.row.evidenceAria", {
                    speakerId: speaker.id,
                  })}
                >
                  <span>
                    {text("speakerSetup.row.difficult", {
                      count: counts.difficult,
                    })}
                  </span>
                  <span>
                    {text("speakerSetup.row.overlap", {
                      count: counts.overlap,
                    })}
                  </span>
                  <span>
                    {text("speakerSetup.row.lowMargin", {
                      count: counts.lowMargin,
                    })}
                  </span>
                </span>
              </label>

              <div className="speaker-row__states">
                <StatusBadge
                  status={speaker.sampleStatus === "ready" ? "verified" : "warning"}
                  label={
                    speaker.sampleStatus === "ready"
                      ? text("speakerSetup.row.voiceprintReady")
                      : text("speakerSetup.row.audioReviewAdvised")
                  }
                  subtle
                />
                <StatusBadge
                  status={
                    speaker.reviewStatus === "confirmed"
                      ? "verified"
                      : speaker.reviewStatus === "needs_review"
                        ? "warning"
                        : "pending"
                  }
                  label={
                    speaker.reviewStatus === "confirmed"
                      ? text("speakerSetup.row.reviewed")
                      : speaker.reviewStatus === "needs_review"
                        ? text("speakerSetup.row.needsReview")
                        : text("speakerSetup.row.pending")
                  }
                  subtle
                />
              </div>

              <div className="speaker-row__actions">
                <button
                  id={`speaker-save-${speaker.id}`}
                  className="mini-action"
                  type="button"
                  disabled={
                    busyAction === speaker.id ||
                    speaker.locked ||
                    !(draftLabels[speaker.id] ?? speaker.label).trim() ||
                    (draftLabels[speaker.id] ?? speaker.label).trim() === speaker.label
                  }
                  aria-label={text("speakerSetup.actions.saveAria", {
                    speakerId: speaker.id,
                  })}
                  onClick={() => {
                    submitUpdate(speaker).catch(reportUpdateError);
                  }}
                >
                  {text("speakerSetup.actions.save")}
                </button>
                <button
                  id={`speaker-lock-${speaker.id}`}
                  className={speaker.locked ? "mini-action mini-action--active" : "mini-action"}
                  type="button"
                  disabled={busyAction === speaker.id}
                  aria-pressed={speaker.locked}
                  aria-label={
                    speaker.locked
                      ? text("speakerSetup.actions.unlockAria", {
                          speakerId: speaker.id,
                        })
                      : text("speakerSetup.actions.lockAria", {
                          speakerId: speaker.id,
                        })
                  }
                  onClick={() => {
                    submitUpdate(speaker, { locked: !speaker.locked }).catch(
                      reportUpdateError,
                    );
                  }}
                >
                  <Icon name="lock" size={13} />
                  {speaker.locked
                    ? text("speakerSetup.actions.unlock")
                    : text("speakerSetup.actions.lock")}
                </button>
                <button
                  id={`speaker-review-confirm-${speaker.id}`}
                  className={
                    speaker.reviewStatus === "confirmed"
                      ? "mini-action mini-action--active"
                      : "mini-action"
                  }
                  type="button"
                  disabled={busyAction === speaker.id}
                  aria-pressed={speaker.reviewStatus === "confirmed"}
                  aria-label={
                    speaker.reviewStatus === "confirmed"
                      ? text("speakerSetup.actions.markUnreviewedAria", {
                          speakerId: speaker.id,
                        })
                      : text("speakerSetup.actions.markReviewedAria", {
                          speakerId: speaker.id,
                        })
                  }
                  onClick={() => {
                    submitUpdate(speaker, {
                      reviewStatus:
                        speaker.reviewStatus === "confirmed" ? "pending" : "confirmed",
                    }).catch(reportUpdateError);
                  }}
                >
                  <Icon name="check" size={13} />
                  {speaker.reviewStatus === "confirmed"
                    ? text("speakerSetup.actions.undoReview")
                    : text("speakerSetup.actions.markReviewed")}
                </button>
                <button
                  id={`speaker-review-required-${speaker.id}`}
                  className={
                    speaker.reviewStatus === "needs_review"
                      ? "mini-action mini-action--warning"
                      : "mini-action"
                  }
                  type="button"
                  disabled={busyAction === speaker.id}
                  aria-pressed={speaker.reviewStatus === "needs_review"}
                  aria-label={
                    speaker.reviewStatus === "needs_review"
                      ? text("speakerSetup.actions.clearReviewRequirementAria", {
                          speakerId: speaker.id,
                        })
                      : text("speakerSetup.actions.requireReviewAria", {
                          speakerId: speaker.id,
                        })
                  }
                  onClick={() => {
                    submitUpdate(speaker, {
                      reviewStatus:
                        speaker.reviewStatus === "needs_review" ? "pending" : "needs_review",
                    }).catch(reportUpdateError);
                  }}
                >
                  <Icon name="headphones" size={13} />
                  {speaker.reviewStatus === "needs_review"
                    ? text("speakerSetup.actions.clearReview")
                    : text("speakerSetup.actions.markForReview")}
                </button>
              </div>
            </article>
          );
        })}
        {visibleSpeakers.length === 0 ? (
          <div className="speaker-list__empty" role="status">
            <Icon name="alert" size={18} />
            {text("speakerSetup.empty", { count: speakers.length })}
          </div>
        ) : null}
      </div>

      {visibleSpeakers.length > 0 ? (
        <nav
          className="collection-pagination"
          aria-label={text("speakerSetup.pagination.aria")}
        >
          <span className="collection-pagination__summary" role="status">
            {text("speakerSetup.pagination.summary", {
              start: pageRangeStart,
              end: pageRangeEnd,
              total: visibleSpeakers.length,
            })}
          </span>
          <div className="collection-pagination__controls">
            <button
              id="speaker-page-previous"
              className="collection-pagination__button"
              type="button"
              disabled={currentPage === 1}
              aria-label={text("speakerSetup.pagination.previousAria")}
              onClick={() => goToPage(currentPage - 1)}
            >
              {text("speakerSetup.pagination.previous")}
            </button>
            <label className="collection-pagination__page" htmlFor="speaker-page-input">
              <span>{text("speakerSetup.pagination.page")}</span>
              <input
                id="speaker-page-input"
                type="number"
                min={1}
                max={totalPages}
                step={1}
                value={currentPage}
                aria-label={text("speakerSetup.pagination.pageAria")}
                onChange={(event) => {
                  if (Number.isFinite(event.currentTarget.valueAsNumber)) {
                    goToPage(event.currentTarget.valueAsNumber);
                  }
                }}
              />
              <span aria-live="polite">
                {text("speakerSetup.pagination.ofPages", { total: totalPages })}
              </span>
            </label>
            <button
              id="speaker-page-next"
              className="collection-pagination__button"
              type="button"
              disabled={currentPage === totalPages}
              aria-label={text("speakerSetup.pagination.nextAria")}
              onClick={() => goToPage(currentPage + 1)}
            >
              {text("speakerSetup.pagination.next")}
            </button>
          </div>
        </nav>
      ) : null}

      <section className="governance-panel" aria-labelledby="governance-panel-title">
        <button
          className="governance-panel__toggle"
          type="button"
          aria-expanded={governanceOpen}
          aria-controls="governance-panel-content"
          onClick={() => setGovernanceOpen((current) => !current)}
        >
          <span className="governance-panel__header">
            <span>
              <span className="panel__eyebrow">
                {text("speakerSetup.governance.eyebrow")}
              </span>
              <span id="governance-panel-title" className="governance-panel__title">
                {text("speakerSetup.governance.title")}
              </span>
            </span>
            <strong>{text("speakerSetup.governance.draftOnly")}</strong>
          </span>
          <span
            className={
              governanceOpen
                ? "governance-panel__chevron governance-panel__chevron--open"
                : "governance-panel__chevron"
            }
            aria-hidden="true"
          >
            <Icon name="chevron-down" size={17} />
          </span>
        </button>

        {governanceOpen ? (
          <div id="governance-panel-content" className="governance-panel__content">
            <p>{text("speakerSetup.governance.description")}</p>

            <div className="governance-panel__forms">
          <fieldset className="governance-form">
            <legend>{text("speakerSetup.merge.legend")}</legend>
            <div className="governance-form__pair">
              <label htmlFor="speaker-merge-source">
                <span>{text("speakerSetup.merge.sourceLabel")}</span>
                <select
                  id="speaker-merge-source"
                  aria-label={text("speakerSetup.merge.sourceAria")}
                  value={effectiveMergeSource}
                  onChange={(event) =>
                    setMergeSource(event.target.value as SpeakerProfile["id"])
                  }
                >
                  {speakers.map((speaker) => (
                    <option key={speaker.id} value={speaker.id}>
                      {text("speakerSetup.trackOption", {
                        speakerId: speaker.id,
                        label: speaker.label,
                      })}
                    </option>
                  ))}
                </select>
              </label>
              <label htmlFor="speaker-merge-target">
                <span>{text("speakerSetup.merge.targetLabel")}</span>
                <select
                  id="speaker-merge-target"
                  aria-label={text("speakerSetup.merge.targetAria")}
                  value={effectiveMergeTarget}
                  onChange={(event) =>
                    setMergeTarget(event.target.value as SpeakerProfile["id"])
                  }
                >
                  {speakers.map((speaker) => (
                    <option key={speaker.id} value={speaker.id}>
                      {text("speakerSetup.trackOption", {
                        speakerId: speaker.id,
                        label: speaker.label,
                      })}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <label htmlFor="speaker-merge-reason">
              <span>{text("speakerSetup.merge.reason")}</span>
              <textarea
                id="speaker-merge-reason"
                aria-label={text("speakerSetup.merge.reason")}
                rows={2}
                value={mergeReason}
                onChange={(event) => setMergeReason(event.target.value)}
              />
            </label>
            <label htmlFor="speaker-merge-evidence">
              <span>{text("speakerSetup.merge.evidenceLabel")}</span>
              <textarea
                id="speaker-merge-evidence"
                aria-label={text("speakerSetup.merge.evidenceAria")}
                rows={2}
                value={mergeEvidence}
                onChange={(event) => setMergeEvidence(event.target.value)}
              />
            </label>
            <label htmlFor="speaker-merge-confidence">
              <span>{text("speakerSetup.confidence.label")}</span>
              <input
                id="speaker-merge-confidence"
                type="number"
                aria-label={text("speakerSetup.merge.confidenceAria")}
                min="0"
                max="1"
                step="0.01"
                value={mergeConfidence}
                onChange={(event) => setMergeConfidence(event.target.value)}
              />
            </label>
            <button
              id="speaker-merge-create-draft"
              className="button button--soft"
              type="button"
              disabled={!canCreateMergeDraft}
              onClick={() =>
                setGovernanceDrafts((current) => [
                  ...current,
                  {
                    id: `merge-${current.length + 1}`,
                    kind: "merge",
                    title: text("speakerSetup.merge.draftTitle", {
                      source: speakerLabel(effectiveMergeSource),
                      target: speakerLabel(effectiveMergeTarget),
                    }),
                    detail: text("speakerSetup.merge.draftDetail", {
                      source: effectiveMergeSource,
                      target: effectiveMergeTarget,
                    }),
                    reason: mergeReason.trim(),
                    evidence: mergeEvidence.trim(),
                    confidence: parsedMergeConfidence,
                  },
                ])
              }
            >
              {text("speakerSetup.merge.create")}
            </button>
          </fieldset>

          <fieldset className="governance-form">
            <legend>{text("speakerSetup.split.legend")}</legend>
            <label htmlFor="speaker-split-track">
              <span>{text("speakerSetup.split.trackLabel")}</span>
              <select
                id="speaker-split-track"
                aria-label={text("speakerSetup.split.trackAria")}
                value={effectiveSplitSpeaker}
                onChange={(event) =>
                  setSplitSpeaker(event.target.value as SpeakerProfile["id"])
                }
              >
                {speakers.map((speaker) => (
                  <option key={speaker.id} value={speaker.id}>
                    {text("speakerSetup.trackOption", {
                      speakerId: speaker.id,
                      label: speaker.label,
                    })}
                  </option>
                ))}
              </select>
            </label>
            <label htmlFor="speaker-split-anchor">
              <span>{text("speakerSetup.split.anchorLabel")}</span>
              <input
                id="speaker-split-anchor"
                aria-label={text("speakerSetup.split.anchorAria")}
                list="review-anchor-options"
                value={splitAnchor}
                placeholder={text("speakerSetup.split.anchorPlaceholder")}
                onChange={(event) => setSplitAnchor(event.target.value)}
              />
              <datalist id="review-anchor-options">
                {reviews.map((review) => (
                  <option key={review.id} value={review.id}>
                    {review.timestampLabel}
                  </option>
                ))}
              </datalist>
            </label>
            <label htmlFor="speaker-split-reason">
              <span>{text("speakerSetup.split.reason")}</span>
              <textarea
                id="speaker-split-reason"
                aria-label={text("speakerSetup.split.reason")}
                rows={2}
                value={splitReason}
                onChange={(event) => setSplitReason(event.target.value)}
              />
            </label>
            <label htmlFor="speaker-split-evidence">
              <span>{text("speakerSetup.split.evidenceLabel")}</span>
              <textarea
                id="speaker-split-evidence"
                aria-label={text("speakerSetup.split.evidenceAria")}
                rows={2}
                value={splitEvidence}
                onChange={(event) => setSplitEvidence(event.target.value)}
              />
            </label>
            <label htmlFor="speaker-split-confidence">
              <span>{text("speakerSetup.confidence.label")}</span>
              <input
                id="speaker-split-confidence"
                type="number"
                aria-label={text("speakerSetup.split.confidenceAria")}
                min="0"
                max="1"
                step="0.01"
                value={splitConfidence}
                onChange={(event) => setSplitConfidence(event.target.value)}
              />
            </label>
            <button
              id="speaker-split-create-draft"
              className="button button--soft"
              type="button"
              disabled={!canCreateSplitDraft}
              onClick={() =>
                setGovernanceDrafts((current) => [
                  ...current,
                  {
                    id: `split-${current.length + 1}`,
                    kind: "split",
                    title: text("speakerSetup.split.draftTitle", {
                      speaker: speakerLabel(effectiveSplitSpeaker),
                      anchor: splitAnchor.trim(),
                    }),
                    detail: text("speakerSetup.split.draftDetail", {
                      speakerId: effectiveSplitSpeaker,
                      anchor: splitAnchor.trim(),
                    }),
                    reason: splitReason.trim(),
                    evidence: splitEvidence.trim(),
                    confidence: parsedSplitConfidence,
                  },
                ])
              }
            >
              {text("speakerSetup.split.create")}
            </button>
          </fieldset>
            </div>

            {governanceDrafts.length > 0 ? (
              <ol
                className="governance-drafts"
                aria-label={text("speakerSetup.governance.draftsAria")}
              >
                {governanceDrafts.map((draft) => (
                  <li key={draft.id}>
                    <span>
                      {text("speakerSetup.governance.draftBadge", {
                        kind:
                          draft.kind === "merge"
                            ? text("speakerSetup.governance.kindMerge")
                            : text("speakerSetup.governance.kindSplit"),
                        state: text("speakerSetup.governance.draftOnly"),
                      })}
                    </span>
                    <strong>{draft.title}</strong>
                    <p>
                      {text("speakerSetup.governance.draftReason", {
                        detail: draft.detail,
                        reason: draft.reason,
                      })}
                    </p>
                    <small>
                      {text("speakerSetup.governance.draftEvidence", {
                        evidence: draft.evidence,
                        confidence: (draft.confidence * 100).toFixed(0),
                      })}
                    </small>
                  </li>
                ))}
              </ol>
            ) : null}
          </div>
        ) : null}
      </section>

      <div className="panel__footnote">
        <Icon name="shield" size={16} />
        <span>{text("speakerSetup.footnote")}</span>
      </div>
    </section>
  );
}
