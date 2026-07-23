import { render } from "@testing-library/react";
import { SceneBackdrop } from "./SceneBackdrop";

describe("SceneBackdrop", () => {
  it("renders the original day, night, and PNG scene artwork as separate layers", () => {
    const { container } = render(<SceneBackdrop />);
    const backdrop = container.querySelector(".scene-backdrop");
    const images = container.querySelectorAll("img");

    expect(backdrop).toHaveAttribute(
      "data-background-contract",
      "remote-day-night-scene",
    );
    expect(images).toHaveLength(3);
    expect(images[0]).toHaveAttribute("src", expect.stringMatching(/day.*\.jpg$/u));
    expect(images[1]).toHaveAttribute(
      "src",
      expect.stringMatching(/night.*\.jpg$/u),
    );
    expect(images[2]).toHaveAttribute(
      "src",
      expect.stringMatching(/scene.*\.png$/u),
    );
    expect(
      container.querySelector(".scene-backdrop__theme--light"),
    ).toBeInTheDocument();
    expect(
      container.querySelector(".scene-backdrop__theme--dark"),
    ).toBeInTheDocument();
  });
});
