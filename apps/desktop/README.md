# MediaTranscribe Studio Desktop

MediaTranscribe Studio Desktop is an offline-first Tauri 2 workspace built with
React and TypeScript. The WebView communicates with Rust only through fixed
commands and versioned DTOs. Browser development mode uses a contract-compatible
local mock instead of a second UI-specific data model.

## Desktop-only UI architecture

- The production interface is the Tauri native desktop window with a
  React/TypeScript WebView application.
- The repository does not ship a browser launcher, Python UI, frontend web
  server, or browser-file-system workflow.
- Vite's browser surface and controlled adapters exist only for development and
  deterministic component tests. They are not a production runtime.
- Production media drag-and-drop is subscribed from the current Tauri window;
  it does not use HTML drag events, `File`, `FileList`, or the browser File
  System Access API.

## Local commands

```powershell
npm ci
npm run lint
npm run typecheck
npm test -- --run
npm run build
npm run tauri dev
```

## Runtime boundaries

- The application does not load CDNs, remote fonts, remote images, telemetry, or
  network APIs.
- `src/bridge/desktop-backend.ts` is the only frontend backend port. The local
  mock and Tauri IPC implement the same interface.
- Tauri exposes only the fixed `get_snapshot`, `create_job`, `cancel_job`,
  `update_speaker`, `apply_review_decision`, and `open_artifact` commands.
  Frontend code does not construct or execute shell commands.
- No Rust command was added for media drag-and-drop or output-path generation.
  The TypeScript bridge uses Tauri window events and Tauri path helpers.
- The TypeScript boundary validates bounded multi-file intake, absolute local
  path shape, traversal/control-character rejection, lexical containment, and
  source/output inequality. It deliberately does not treat filename extensions
  as a media-security boundary. Canonical existence, regular-file identity,
  permissions, final symlink resolution, content probing, decodability, and
  filesystem containment remain at the trusted native/backend boundary.
- `src/contracts/studio.ts` contains the versioned UI contracts.
- `src/assets/day.jpg` and `src/assets/night.jpg` are packaged light/dark
  full-viewport backgrounds. `src/assets/scene.webp` and `src/assets/scene.png`
  remain offline compatibility fallbacks. CSS never fetches a remote image.

## Interface language, theme, and accessibility

The first interface-language wave includes:

```text
English
简体中文
繁體中文
日本語
한국어
Español
Français
Deutsch
Português (Brasil)
```

- English is the deterministic default and mandatory fallback, independent of
  the operating-system or WebView language.
- Missing localized keys fail closed to English. Runtime evidence, model names,
  filenames, transcript content, and backend errors are not machine-translated.
- The explicit locale preference is stored under
  `media-transcribe-studio.ui-locale`.
- Theme choices are `system`, `light`, and `dark`. The explicit preference is
  stored under `media-transcribe-studio.ui-theme`; system mode responds live to
  `prefers-color-scheme`.
- Day and night scenes use full-viewport `cover` rendering around
  `center 48%`, with theme overlays and frosted surfaces preserving text
  contrast.
- Glass surfaces include both standard and WebKit backdrop filters plus an
  opaque fallback. The UI also defines strong `focus-visible` treatment,
  forced-colors behavior, reduced-motion behavior, and 320 CSS-pixel layout
  safeguards.

## Native media drag-and-drop

The production adapter subscribes to:

```ts
getCurrentWindow().onDragDropEvent(...)
```

It accepts a bounded batch of absolute local file paths. The native Windows
picker intentionally has no extension filter, and drag-and-drop also admits
uncommon, misleading, and extensionless filenames. Filename extensions are
only UI hints; trusted FFprobe/FFmpeg content evidence decides whether each
source is usable media before processing.

A valid intake opens the task creator and fills both paths for every queue row.
The editable default output directory is a safe sibling derived from each
source:

```text
D:\Media\meeting.mov
→ D:\Media\meeting-MediaTranscribeStudio
```

Relative, traversal-bearing, control-character, overlong, and
UNC/network/device paths fail closed. Empty and oversized batches also fail
closed. Browser tests use only a controlled adapter that emits the same typed
events; production behavior does not depend on browser UI or browser filesystem
APIs.

## Global UI and source-language policy

- Production-facing navigation, controls, status text, accessibility labels,
  document metadata, and README content are written in English and stored as
  UTF-8.
- Speaker labels default to neutral names such as `Speaker 1` and can be renamed
  without assuming a language, role, or fixed participant count.
- Transcript evidence remains in its source language. Human review may correct
  the source transcript, but it does not silently translate, summarize, or
  rewrite it.
- Locale-sensitive search uses the runtime locale rather than a hard-coded
  Chinese locale.

## Dynamic speaker-count strategies

- `auto` asks the local diarization pipeline to detect the count and return
  ranked candidates with confidence evidence.
- `manual` accepts a positive safe integer and materializes consecutive
  `speaker-1..speaker-N` profiles.
- `hybrid` supplies minimum, maximum, and prior counts. The detected result must
  remain inside the configured bounds.
- The UI does not assume five speakers or any other business maximum. Speaker
  profiles can be renamed, locked, and marked for review. Resource failures are
  reported explicitly rather than changing the semantic speaker count.

## Current desktop surfaces

- Create transcription jobs
- Configure automatic, manual, and hybrid speaker-count policies
- Inspect detected-count evidence and the Dynamic-N cascade
- Rename, lock, filter, and paginate speaker profiles
- Inspect model strategy, runtime safety boundaries, stages, and events
- Review low-confidence segments with immutable source evidence and audit fields
- Browse generated artifacts
- Inspect PDF hard gates, the 14-dimension visual review, and the repair queue

Repository mock data demonstrates variable participant counts only. Production
counts must come from automatic detection, an explicit manual count, or a bounded
hybrid policy.

## Translation, polishing, and summaries

The versioned `CreateJobRequest` contract exposes source language, local-LLM
mode/model/endpoint, translation targets, polishing, summary, output locale, and
business prompt version. Translation, polishing, and summaries are explicit
opt-in derived operations; none of them silently changes source transcript
evidence.

The desktop boundary must preserve source and derived artifacts separately,
record local-model and prompt-version provenance, validate every requested
locale without assuming Chinese, and report unsupported or failed operations
deterministically. The review editor remains a source-transcript correction
surface even when derived business outputs are enabled.

## Design validation

The UI follows the local `frontend-design-pack-global` design guidance and keeps
an offline fallback under `.codex/frontend-design-runtime/`.

The fallback can validate locally available rules when the upstream registry is
unavailable, but it is not evidence of release readiness. The latest inspected
Design Pack state reports:

```text
implementationReady: false
releaseEligible: false
offlineRuntimeReady: true
releaseStatus: package-not-release-ready
repository synchronized: false
```

Run the sibling Design Pack inspector from the repository root when that project
is available:

```powershell
node ..\frontend-design-pack-global\scripts\run-codex-design-workflow.mjs inspect
```

A production release must remain blocked until the Design Pack reports release
eligibility and the full desktop workflow passes visual, accessibility, offline,
and interaction validation.

## Current integration gaps

- The Tauri backend launches and supervises the Python ASR and voiceprint
  worker and can submit fixed protocol commands. Projection of validated worker
  events into `StudioStore`, live frontend synchronization, and terminal-state
  reconciliation are not yet complete.
- The OpenHTMLtoPDF and PDFBox rendering sidecar is not yet connected to the
  desktop workflow.
- The local Design Pack runtime is an offline fallback only while the upstream
  registry and release checks remain incomplete.
