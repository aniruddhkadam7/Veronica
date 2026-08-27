import React from "react";
import ReactDOM from "react-dom/client";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import App from "./App";
import { VeronicaOverlay } from "./VeronicaOverlay";
import { VeronicaWidget } from "./VeronicaWidget";
import "./tailwind.css";
import "./overlay.css";

// The overlay window loads the exact same index.html/bundle as the main
// window (Tauri's WebviewUrl::App points both at "index.html") — which root
// component renders is decided here by the window's label, set when the
// overlay is created in Rust.
const FIXED_OVERLAY_LABELS = new Set(["veronica-overlay", "veronica-widget"]);

const label = getCurrentWebviewWindow().label;
const isOverlay = FIXED_OVERLAY_LABELS.has(label);
if (isOverlay) {
  document.body.classList.add("overlay-window");
  // Also mark <html>. Vite bundles every stylesheet into one file regardless
  // of which component imported it, so App.css's `html, body { background }`
  // rule reaches this window too and would paint an opaque canvas behind the
  // overlay — defeating the window's transparency. Marking the root element
  // lets overlay.css override it by specificity, without depending on CSS
  // `:has()` support in whatever WebView2 version is installed.
  document.documentElement.classList.add("overlay-window");
}
if (label === "veronica-widget") {
  // Separate class from "overlay-window" so overlay.css's widget-only rules
  // (see .veronica-widget-root) don't leak into the full chat overlay.
  document.body.classList.add("veronica-widget-window");
  document.documentElement.classList.add("veronica-widget-window");
}

function Root() {
  if (label === "veronica-overlay") return <VeronicaOverlay />;
  if (label === "veronica-widget") return <VeronicaWidget />;
  return <App />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
