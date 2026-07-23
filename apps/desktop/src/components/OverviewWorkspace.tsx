import type { CSSProperties, ReactNode } from "react";
import { useI18n } from "../i18n";
import { Icon, type IconName } from "./Icon";

export type OverviewRoom = "home" | "speakers" | "quality" | "pipeline";

interface OverviewWorkspaceProps {
  room: OverviewRoom;
  speakerCount: number;
  reviewCount: number;
  stageCount: number;
  eventCount: number;
  hero: ReactNode;
  runtime: ReactNode;
  speakerStudio: ReactNode;
  qualityLab: ReactNode;
  pipelineObservatory: ReactNode;
  onRoomChange: (room: OverviewRoom) => void;
}

interface RoomDefinition {
  id: Exclude<OverviewRoom, "home">;
  icon: IconName;
  tone: "violet" | "rose" | "mint";
}

const ROOM_DEFINITIONS: readonly RoomDefinition[] = [
  { id: "speakers", icon: "headphones", tone: "violet" },
  { id: "quality", icon: "wave", tone: "rose" },
  { id: "pipeline", icon: "sparkles", tone: "mint" },
];

export function OverviewWorkspace({
  room,
  speakerCount,
  reviewCount,
  stageCount,
  eventCount,
  hero,
  runtime,
  speakerStudio,
  qualityLab,
  pipelineObservatory,
  onRoomChange,
}: OverviewWorkspaceProps) {
  const { t } = useI18n();

  const roomMeta = (roomId: RoomDefinition["id"]) => {
    switch (roomId) {
      case "speakers":
        return t("overview.room.speakers.meta", {
          count: speakerCount,
          reviews: reviewCount,
        });
      case "quality":
        return t("overview.room.quality.meta", { stages: stageCount });
      case "pipeline":
        return t("overview.room.pipeline.meta", {
          stages: stageCount,
          events: eventCount,
        });
    }
  };

  if (room === "home") {
    return (
      <div className="overview-workspace overview-workspace--home">
        {hero}
        {runtime}

        <section className="studio-hub" aria-labelledby="studio-hub-title">
          <header className="studio-hub__header">
            <span className="panel__eyebrow">
              <Icon name="sparkles" size={15} />
              {t("overview.hub.eyebrow")}
            </span>
            <div>
              <h2 id="studio-hub-title">{t("overview.hub.title")}</h2>
              <p>{t("overview.hub.description")}</p>
            </div>
          </header>

          <div
            className="studio-room-grid studio-room-portals"
            aria-label={t("overview.hub.aria")}
          >
            {ROOM_DEFINITIONS.map((definition, index) => {
              const titleKey = `overview.room.${definition.id}.title` as const;
              const descriptionKey =
                `overview.room.${definition.id}.description` as const;
              const descriptionId = `studio-room-${definition.id}-description`;

              return (
                <button
                  className={`studio-room-portal studio-room-portal--${definition.tone}`}
                  type="button"
                  aria-describedby={descriptionId}
                  key={definition.id}
                  onClick={() => onRoomChange(definition.id)}
                  style={
                    {
                      "--portal-index": index,
                    } as CSSProperties
                  }
                >
                  <span className="studio-room-portal__track" aria-hidden="true">
                    <span />
                  </span>
                  <span className="studio-room-portal__number" aria-hidden="true">
                    0{index + 1}
                  </span>
                  <span className="studio-room-portal__icon" aria-hidden="true">
                    <Icon name={definition.icon} size={25} />
                  </span>
                  <span className="studio-room-portal__copy">
                    <strong>{t(titleKey)}</strong>
                    <span id={descriptionId}>{t(descriptionKey)}</span>
                  </span>
                  <span className="studio-room-portal__footer">
                    <small>{roomMeta(definition.id)}</small>
                    <span aria-hidden="true">
                      {t("overview.room.open")}
                      <Icon name="arrow-right" size={16} />
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        </section>
      </div>
    );
  }

  const activeDefinition = ROOM_DEFINITIONS.find(
    (definition) => definition.id === room,
  );
  const roomTitleKey = `overview.room.${room}.title` as const;
  const roomDescriptionKey = `overview.room.${room}.description` as const;

  return (
    <div className={`overview-workspace overview-workspace--room overview-workspace--${room}`}>
      <header className="studio-room-header">
        <button
          className="studio-room-header__back"
          type="button"
          onClick={() => onRoomChange("home")}
        >
          <Icon name="arrow-right" size={17} />
          {t("overview.room.back")}
        </button>

        <nav
          className="studio-room-header__breadcrumb"
          aria-label={t("overview.room.breadcrumb")}
        >
          <button type="button" onClick={() => onRoomChange("home")}>
            {t("nav.overview")}
          </button>
          <span aria-hidden="true">/</span>
          <span aria-current="page">{t(roomTitleKey)}</span>
        </nav>

        <div
          className={`studio-room-header__icon studio-room-header__icon--${activeDefinition?.tone ?? "violet"}`}
          aria-hidden="true"
        >
          <Icon name={activeDefinition?.icon ?? "sparkles"} size={27} />
        </div>
        <div className="studio-room-header__copy">
          <span className="panel__eyebrow">{t("overview.room.current")}</span>
          <h1>{t(roomTitleKey)}</h1>
          <p>{t(roomDescriptionKey)}</p>
        </div>
        <span className="studio-room-header__meta">{roomMeta(room)}</span>
      </header>

      <div className="studio-room-content">
        {room === "speakers" ? speakerStudio : null}
        {room === "quality" ? qualityLab : null}
        {room === "pipeline" ? pipelineObservatory : null}
      </div>
    </div>
  );
}
