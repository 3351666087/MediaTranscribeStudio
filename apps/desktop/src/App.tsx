import { useCallback, useMemo, useRef, useState } from "react";
import {
  createMediaDropAdapter,
  MediaDropError,
  type MediaDropAdapter,
  type MediaDropBatchResult,
  type MediaDropFailure,
} from "./bridge/media-drop";
import { ArtifactBrowser } from "./components/ArtifactBrowser";
import { DiarizationQualityPanel } from "./components/DiarizationQualityPanel";
import { DesktopCompanion } from "./components/DesktopCompanion";
import { EventStream } from "./components/EventStream";
import { Icon } from "./components/Icon";
import { ModelStrategyPanel } from "./components/ModelStrategyPanel";
import { MediaDropOverlay } from "./components/MediaDropOverlay";
import { NavigationRail } from "./components/NavigationRail";
import {
  OverviewWorkspace,
  type OverviewRoom,
} from "./components/OverviewWorkspace";
import { PerformanceWorkbench } from "./components/PerformanceWorkbench";
import { PdfQaPanel } from "./components/PdfQaPanel";
import { PipelineOverview } from "./components/PipelineOverview";
import { ReviewQueue } from "./components/ReviewQueue";
import { SceneBackdrop } from "./components/SceneBackdrop";
import { SpeakerSetupPanel } from "./components/SpeakerSetupPanel";
import { StatusBadge } from "./components/StatusBadge";
import {
  TaskCreator,
  type InitialMediaBatch,
} from "./components/TaskCreator";
import { TaskHero } from "./components/TaskHero";
import { Toast } from "./components/Toast";
import { TopBar } from "./components/TopBar";
import type { AppSection } from "./contracts/studio";
import { useNativeMediaDrop } from "./hooks/use-native-media-drop";
import { useStudio } from "./hooks/use-studio";
import {
  I18nProvider,
  mediaDropErrorMessageKey,
  useI18n,
} from "./i18n";
import { ThemeProvider } from "./theme";

interface AppProps {
  mediaDropAdapter?: MediaDropAdapter;
}

type NavigationDirection = "forward" | "backward" | "neutral";

const SECTION_ORDER: readonly AppSection[] = [
  "overview",
  "review",
  "artifacts",
  "pdf-qa",
];

export default function App({ mediaDropAdapter }: AppProps) {
  return (
    <ThemeProvider>
      <I18nProvider>
        <StudioApp mediaDropAdapter={mediaDropAdapter} />
      </I18nProvider>
    </ThemeProvider>
  );
}

function StudioApp({ mediaDropAdapter: injectedAdapter }: AppProps) {
  const studio = useStudio();
  const { notify, setActiveSection } = studio;
  const { t } = useI18n();
  const mediaDropAdapter = useMemo(
    () => injectedAdapter ?? createMediaDropAdapter(),
    [injectedAdapter],
  );
  const [taskCreatorOpen, setTaskCreatorOpen] = useState(false);
  const [overviewRoom, setOverviewRoom] = useState<OverviewRoom>("home");
  const [sectionDirection, setSectionDirection] =
    useState<NavigationDirection>("neutral");
  const [roomDirection, setRoomDirection] =
    useState<NavigationDirection>("neutral");
  const [initialMediaBatch, setInitialMediaBatch] =
    useState<InitialMediaBatch | null>(null);
  const mediaSelectionSequence = useRef(0);
  const mainRef = useRef<HTMLElement>(null);
  const closeTaskCreator = useCallback(() => {
    setTaskCreatorOpen(false);
    setInitialMediaBatch(null);
  }, []);
  const openTaskCreator = useCallback(() => {
    setInitialMediaBatch(null);
    setTaskCreatorOpen(true);
  }, []);
  const resolveMediaPath = useCallback(
    async (path: string) => await mediaDropAdapter.resolve(path),
    [mediaDropAdapter],
  );
  const notifyMediaDropFailure = useCallback(
    (failure: MediaDropFailure) => {
      const detail =
        failure.code === "duplicatePath" ||
        failure.code === "resolutionFailed"
          ? failure.message
          : t(
              mediaDropErrorMessageKey(
                new MediaDropError(failure.code, failure.message),
              ),
            );
      notify("error", t("drop.errorTitle"), detail);
    },
    [notify, t],
  );
  const handleMediaSelection = useCallback(
    (result: MediaDropBatchResult) => {
      result.failures.forEach(notifyMediaDropFailure);
      if (result.selections.length === 0) {
        return;
      }

      mediaSelectionSequence.current += 1;
      setInitialMediaBatch({
        sequence: mediaSelectionSequence.current,
        selections: result.selections,
      });
      setTaskCreatorOpen(true);
    },
    [notifyMediaDropFailure],
  );
  const handleMediaDropError = useCallback(
    (error: unknown) => {
      notify(
        "error",
        t("drop.errorTitle"),
        t(mediaDropErrorMessageKey(error)),
      );
    },
    [notify, t],
  );
  const mediaDrop = useNativeMediaDrop(mediaDropAdapter, {
    onSelection: handleMediaSelection,
    onError: handleMediaDropError,
  });

  const navigate = useCallback(
    (section: AppSection) => {
      const currentIndex = SECTION_ORDER.indexOf(studio.activeSection);
      const nextIndex = SECTION_ORDER.indexOf(section);
      setSectionDirection(
        nextIndex === currentIndex
          ? "neutral"
          : nextIndex > currentIndex
            ? "forward"
            : "backward",
      );
      if (section === "overview") {
        setOverviewRoom("home");
        setRoomDirection("backward");
      }
      setActiveSection(section);
      window.requestAnimationFrame(() => {
        mainRef.current?.focus({ preventScroll: true });
        mainRef.current?.scrollTo({ top: 0, behavior: "auto" });
      });
    },
    [setActiveSection, studio.activeSection],
  );
  const changeOverviewRoom = useCallback(
    (room: OverviewRoom) => {
      setRoomDirection(
        room === overviewRoom
          ? "neutral"
          : room === "home"
            ? "backward"
            : "forward",
      );
      setOverviewRoom(room);
      window.requestAnimationFrame(() => {
        mainRef.current?.focus({ preventScroll: true });
        mainRef.current?.scrollTo({ top: 0, behavior: "auto" });
      });
    },
    [overviewRoom],
  );

  if (studio.loading) {
    return (
      <>
        <SceneBackdrop />
        <main
          className="boot-screen"
          aria-busy="true"
          data-evidence-state="loading"
        >
          <div className="boot-screen__mark" aria-hidden="true">
            <Icon name="sparkles" size={28} />
          </div>
          <span className="panel__eyebrow">{t("app.loading.eyebrow")}</span>
          <h1>{t("app.loading.title")}</h1>
          <p>{t("app.loading.detail")}</p>
          <div
            className="boot-screen__progress"
            aria-label={t("app.loading.aria")}
          >
            <span />
          </div>
        </main>
        <MediaDropOverlay
          visible={mediaDrop.dragging}
          resolving={mediaDrop.resolving}
        />
      </>
    );
  }

  if (studio.loadError || !studio.snapshot) {
    return (
      <>
        <SceneBackdrop />
        <main
          className="boot-screen boot-screen--error"
          role="alert"
          aria-live="assertive"
          aria-busy="false"
          data-evidence-state="load-failure"
        >
          <section
            className="boot-screen__error-card"
            aria-labelledby="load-failure-title"
          >
            <div className="boot-screen__error-header">
              <div
                className="boot-screen__mark boot-screen__mark--error"
                aria-hidden="true"
              >
                <Icon name="alert" size={28} />
              </div>
              <div className="boot-screen__error-copy">
                <span className="panel__eyebrow">
                  {t("app.error.eyebrow")}
                </span>
                <h1 id="load-failure-title">{t("app.error.title")}</h1>
                <p>{t("app.error.detail")}</p>
              </div>
            </div>

            <p className="boot-screen__error-detail">
              <strong>{t("app.error.loadDetails")}</strong>
              <span>
                {studio.loadError ?? t("app.error.noSnapshot")}
              </span>
            </p>

            <div
              className="boot-screen__safety-grid"
              aria-label={t("app.error.safeguards")}
            >
              <div className="boot-screen__safety-item">
                <span aria-hidden="true">
                  <Icon name="cloud-off" size={19} />
                </span>
                <p>
                  <strong>{t("app.error.offlineTitle")}</strong>
                  <small>{t("app.error.offlineDetail")}</small>
                </p>
              </div>
              <div className="boot-screen__safety-item">
                <span aria-hidden="true">
                  <Icon name="shield" size={19} />
                </span>
                <p>
                  <strong>{t("app.error.failClosedTitle")}</strong>
                  <small>{t("app.error.failClosedDetail")}</small>
                </p>
              </div>
            </div>

            <div className="boot-screen__error-actions">
              <button
                className="button button--primary"
                type="button"
                onClick={() => {
                  studio.retryLoad().catch((error: unknown) => {
                    console.error("Failed to reload local workspace", error);
                  });
                }}
              >
                {t("app.error.reload")}
                <Icon name="arrow-right" size={17} />
              </button>
              <small>{t("app.error.reloadHint")}</small>
            </div>
          </section>
        </main>
        <MediaDropOverlay
          visible={mediaDrop.dragging}
          resolving={mediaDrop.resolving}
        />
      </>
    );
  }

  const { snapshot } = studio;
  const selectedStrategy = snapshot.strategies.find(
    (strategy) => strategy.id === snapshot.job.activeStrategyId,
  );

  return (
    <>
      <SceneBackdrop />
      <a
        className="skip-link"
        href="#main-content"
        inert={taskCreatorOpen ? true : undefined}
        aria-hidden={taskCreatorOpen ? true : undefined}
      >
        {t("app.skipToContent")}
      </a>
      <div
        className="app-shell"
        data-evidence-state="ready"
        data-speaker-count={snapshot.speakers.length}
        data-active-section={studio.activeSection}
        data-overview-room={overviewRoom}
        data-section-direction={sectionDirection}
        data-room-direction={roomDirection}
        inert={taskCreatorOpen ? true : undefined}
        aria-hidden={taskCreatorOpen ? true : undefined}
      >
        <NavigationRail
          activeSection={studio.activeSection}
          reviewCount={studio.openReviews.length}
          onNavigate={navigate}
        />

        <div className="workspace-shell">
          <TopBar
            activeSection={studio.activeSection}
            system={snapshot.system}
            onCreateTask={openTaskCreator}
          />

          <main
            id="main-content"
            className="workspace-main"
            ref={mainRef}
            tabIndex={-1}
            aria-label={t("app.workspaceLabel")}
            data-active-section={studio.activeSection}
            data-overview-room={overviewRoom}
            data-section-direction={sectionDirection}
            data-room-direction={roomDirection}
          >
            {studio.activeSection === "overview" ? (
              <OverviewWorkspace
                room={overviewRoom}
                speakerCount={snapshot.speakers.length}
                reviewCount={studio.openReviews.length}
                stageCount={snapshot.stages.length}
                eventCount={snapshot.events.length}
                onRoomChange={changeOverviewRoom}
                hero={
                  <TaskHero
                    job={snapshot.job}
                    strategy={selectedStrategy}
                    onReview={() => navigate("review")}
                  />
                }
                runtime={
                  <section className="runtime-strip" aria-labelledby="runtime-title">
                    <div className="runtime-strip__heading">
                      <span className="runtime-strip__icon" aria-hidden="true">
                        <Icon name="cpu" size={19} />
                      </span>
                      <span>
                        <strong id="runtime-title">{t("runtime.title")}</strong>
                        <small>{t("runtime.detail")}</small>
                      </span>
                    </div>
                    <dl>
                      <div>
                        <dt>{t("runtime.gpu")}</dt>
                        <dd>{snapshot.system.gpuLabel}</dd>
                      </div>
                      <div>
                        <dt>{t("runtime.vram")}</dt>
                        <dd>{snapshot.system.vramLabel}</dd>
                      </div>
                      <div>
                        <dt>{t("runtime.model")}</dt>
                        <dd>
                          <StatusBadge
                            status={
                              snapshot.system.inferenceWorker === "ready"
                                ? "success"
                                : "warning"
                            }
                          />
                        </dd>
                      </div>
                      <div>
                        <dt>{t("runtime.java")}</dt>
                        <dd>
                          <StatusBadge
                            status={
                              snapshot.system.javaRenderer === "ready"
                                ? "success"
                                : "warning"
                            }
                          />
                        </dd>
                      </div>
                    </dl>
                  </section>
                }
                speakerStudio={
                  <div className="overview-grid overview-grid--setup">
                    <SpeakerSetupPanel
                      jobId={snapshot.job.id}
                      speakers={snapshot.speakers}
                      reviews={snapshot.reviews}
                      speakerPolicy={snapshot.job.speakerPolicy}
                      speakerDetection={snapshot.job.speakerDetection}
                      busyAction={studio.busyAction}
                      onUpdate={studio.updateSpeaker}
                    />
                    <ModelStrategyPanel
                      strategies={snapshot.strategies}
                      selectedId={snapshot.job.activeStrategyId}
                      onSelect={studio.chooseStrategy}
                    />
                  </div>
                }
                qualityLab={
                  <div className="overview-grid overview-grid--observability">
                    <DiarizationQualityPanel metrics={snapshot.diarizationQuality} />
                    <PerformanceWorkbench
                      metrics={snapshot.performance}
                      stages={snapshot.stages}
                      reviews={snapshot.reviews}
                    />
                  </div>
                }
                pipelineObservatory={
                  <div className="overview-grid overview-grid--pipeline">
                    <PipelineOverview stages={snapshot.stages} />
                    <EventStream events={snapshot.events} />
                  </div>
                }
              />
            ) : null}

            {studio.activeSection === "review" ? (
              <ReviewQueue
                reviews={studio.openReviews}
                speakers={snapshot.speakers}
                busyAction={studio.busyAction}
                onApply={studio.applyReview}
                onNotify={studio.notify}
              />
            ) : null}

            {studio.activeSection === "artifacts" ? (
              <ArtifactBrowser
                artifacts={snapshot.artifacts}
                backendMode={snapshot.system.backendMode}
                busyAction={studio.busyAction}
                onOpen={studio.openArtifact}
              />
            ) : null}

            {studio.activeSection === "pdf-qa" ? <PdfQaPanel report={snapshot.pdfQuality} /> : null}
          </main>
        </div>
      </div>

      <TaskCreator
        open={taskCreatorOpen}
        initialMediaBatch={initialMediaBatch}
        resolveMediaPath={resolveMediaPath}
        speakers={snapshot.speakers}
        initialSpeakerPolicy={snapshot.job.speakerPolicy}
        strategies={snapshot.strategies}
        selectedStrategyId={snapshot.job.activeStrategyId}
        backendMode={snapshot.system.backendMode}
        busy={studio.busyAction === "create-job"}
        onClose={closeTaskCreator}
        onCreate={studio.createJob}
      />
      <div
        className="desktop-companion-layer"
        inert={taskCreatorOpen ? true : undefined}
        aria-hidden={taskCreatorOpen ? true : undefined}
      >
        <DesktopCompanion
          collapseLabel={t("companion.collapse")}
          expandLabel={t("companion.expand")}
          imageAlt={t("companion.imageAlt")}
          regionLabel={t("companion.regionLabel")}
          statusText={t("companion.statusReady")}
        />
      </div>
      <Toast toast={studio.toast} onDismiss={studio.clearToast} />
      <MediaDropOverlay
        visible={mediaDrop.dragging}
        resolving={mediaDrop.resolving}
      />
    </>
  );
}
