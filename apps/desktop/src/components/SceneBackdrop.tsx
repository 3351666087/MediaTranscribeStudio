import dayArtwork from "../assets/day.jpg";
import nightArtwork from "../assets/night.jpg";
import sceneArtwork from "../assets/scene.png";
import "./SceneBackdrop.css";

/**
 * Keeps the repository's original remote artwork as explicit, inspectable
 * layers instead of hiding it inside an opaque CSS background stack.
 */
export function SceneBackdrop() {
  return (
    <div
      className="scene-backdrop"
      aria-hidden="true"
      data-background-contract="remote-day-night-scene"
    >
      <img
        className="scene-backdrop__theme scene-backdrop__theme--light"
        src={dayArtwork}
        alt=""
        draggable={false}
      />
      <img
        className="scene-backdrop__theme scene-backdrop__theme--dark"
        src={nightArtwork}
        alt=""
        draggable={false}
      />
      <img
        className="scene-backdrop__illustration"
        src={sceneArtwork}
        alt=""
        draggable={false}
      />
      <span className="scene-backdrop__aurora" />
      <span className="scene-backdrop__veil" />
    </div>
  );
}
