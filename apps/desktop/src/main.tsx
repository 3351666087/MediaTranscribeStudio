import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles/tokens.css";
import "./styles/global.css";
import { initializeTheme } from "./theme";

const root = document.getElementById("root");

if (!root) {
  throw new Error("The #root mount element is missing.");
}

initializeTheme();

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
