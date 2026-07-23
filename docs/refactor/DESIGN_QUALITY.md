# Design Quality Gate

## Purpose

`tools.design_quality` turns the repository's executable visual requirements
into a deterministic, fail-closed quality gate. It audits the existing
TypeScript/Tauri desktop source without modifying it and validates two
machine-readable release-evidence contracts:

- real native-window screenshot evidence;
- real OCR and placement-analysis results for the day, night, and scene
  background assets.

This gate is repository-local. It does **not** claim that the external Design
Pack has passed.

## External Design Pack status

The canonical Design Pack V3 preparation path is currently blocked by an
upstream materializer/evaluator/schema contract mismatch. The first explicit
blocker is:

```text
repairCatalog.passPolicy is required
```

Every design-quality report therefore includes:

```json
{
  "passageClaimed": false,
  "status": "blocked-upstream-v3-contract",
  "blocker": "repairCatalog.passPolicy is required"
}
```

This state is informational and must not be rewritten as a Design Pack pass.
The local checks remain enforceable while the upstream V3 contract is blocked.

## Audit modes

### Source audit

```powershell
conda run -n media-asr python -m tools.design_quality --project-root .
```

The source audit checks the desktop source, the complete repository-local CSS
import graph, and the presence/use of committed visual assets. It enumerates
every TypeScript/JavaScript module below `apps/desktop/src`, resolves its
relative CSS imports, follows CSS `@import` edges, and scans the resulting
stylesheets in deterministic path order. The required product graph includes:

```text
apps/desktop/src/styles/tokens.css
apps/desktop/src/styles/global.css
apps/desktop/src/components/SceneBackdrop.css
apps/desktop/src/components/DesktopCompanion.css
```

Missing, unreadable, empty, unsupported, or source-tree-escaping imports fail
the audit. Component styles are gated even before a component is reachable
from `main.tsx`; App integration is checked separately.

Screenshot and OCR checks appear in the report but pass when no manifests are
supplied because evidence is not required in source-audit mode.

### Release audit

```powershell
conda run -n media-asr python -m tools.design_quality `
  --project-root . `
  --release `
  --screenshot-manifest path\to\native-screenshots.json `
  --ocr-manifest path\to\background-ocr.json
```

Release mode is fail-closed:

- omitting the screenshot manifest produces
  `DQ-SCREENSHOT-EVIDENCE-MISSING`;
- omitting the OCR manifest produces `DQ-OCR-EVIDENCE-MISSING`;
- unreadable, schema-invalid, stale, synthetic, browser-only, or inconsistent
  evidence fails the audit;
- an unexpected exception inside any check produces
  `DQ-INTERNAL-FAIL-CLOSED`.

The command exits `0` only for a passing report, `1` for policy findings, and
`2` for a fatal command-level error. `--output <path>` may be used to write the
same JSON report printed to stdout.

## Executable policy

| Area | Enforced rule | Primary failure codes |
| --- | --- | --- |
| Motion properties | Transitions may animate only `transform`, `opacity`, `color`, `background-color`, `border-color`, and `box-shadow`. Keyframes may animate only `transform` and `opacity`. `transition: all` and unresolved keyframes are rejected. | `DQ-MOTION-PROPERTY`, `DQ-MOTION-KEYFRAME-PROPERTY`, `DQ-MOTION-KEYFRAME-UNKNOWN`, `DQ-MOTION-UNPARSEABLE` |
| CSS source graph | Every repository-local CSS import below `apps/desktop/src` is resolved and scanned, including transitive CSS `@import` edges. The four required product stylesheets listed above must be imported. Imports may not be missing, empty, unreadable, non-relative, or escape the source root. | `DQ-CSS-SOURCE-GRAPH`, `DQ-CSS-IMPORT-MISSING`, `DQ-CSS-IMPORT-EMPTY`, `DQ-CSS-IMPORT-READ`, `DQ-CSS-IMPORT-BOUNDARY`, `DQ-CSS-IMPORT-UNSUPPORTED`, `DQ-CSS-REQUIRED-IMPORT` |
| Motion timing | Ordinary UI transitions and finite interaction animations must use literal durations no greater than `300ms`; `ease-in` is rejected. Long ambient/idle loops are permitted only when they are explicitly infinite and the same selector disables the animation under `prefers-reduced-motion: reduce`. Intrinsic GIF playback is not treated as a CSS interaction duration. | `DQ-MOTION-DURATION`, `DQ-MOTION-DURATION-UNKNOWN`, `DQ-MOTION-EASING`, `DQ-REDUCED-MOTION-MOVEMENT` |
| Canonical easing | Tokens must define the exact Design Pack curves `--ease-out: cubic-bezier(0.23, 1, 0.32, 1)`, `--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1)`, and `--ease-drawer: cubic-bezier(0.32, 0.72, 0, 1)`. | `DQ-EASING-TOKEN` |
| Frequent navigation | CSS smooth scrolling is forbidden. Frequent navigation reset code must use `behavior: "auto"`, not `"smooth"`. | `DQ-NAV-SMOOTH-SCROLL`, `DQ-NAV-AUTO-SCROLL`, `DQ-NAV-RESET-CONTRACT` |
| Hover movement | Hover transforms are allowed only inside `@media (hover: hover) and (pointer: fine)`. | `DQ-HOVER-MOTION-GATE` |
| Reduced motion | Any selector using transform movement or transform keyframes must explicitly remove the movement under `prefers-reduced-motion: reduce`. Merely shortening a movement animation is insufficient. | `DQ-REDUCED-MOTION-MOVEMENT` |
| Forced colors | The forced-colors branch must cover the root/canvas and core glass/control surfaces without disabling system color adjustment. | `DQ-FORCED-COLORS-MISSING`, `DQ-FORCED-COLORS-CANVAS`, `DQ-FORCED-COLORS-SURFACE`, `DQ-FORCED-COLORS-ADJUST` |
| Layered background | `SceneBackdrop` must explicitly import and render day, night, and scene image layers, declare the `remote-day-night-scene` contract, and be mounted by `App` in loading, failure, and ready states. The root is fixed/full-screen, all images use `cover`, inactive theme art is hidden, the selected light/dark layer and scene layer are visible, reduced motion removes positional transforms, and forced colors hides decorative artwork. | `DQ-BACKGROUND-ASSET`, `DQ-BACKGROUND-LAYER-CONTRACT`, `DQ-BACKGROUND-APP-INTEGRATION`, `DQ-BACKGROUND-FULLSCREEN`, `DQ-BACKGROUND-COVER`, `DQ-BACKGROUND-THEME-VISIBILITY`, `DQ-BACKGROUND-SCENE-VISIBILITY`, `DQ-BACKGROUND-REDUCED-MOTION`, `DQ-BACKGROUND-FORCED-COLORS` |
| Desktop companion | `companion.gif` must have SHA-256 `84555A0B2AD4B96C0282C50B5A3FD92D6AB3933E0FA72935E3FD59322389C4EB`. `DesktopCompanion` must retain its accessible region/status/button/image, keyboard and pointer contracts, reduced-motion and forced-colors branches, and a localized `App` mount. | `DQ-COMPANION-ASSET`, `DQ-COMPANION-ASSET-HASH`, `DQ-COMPANION-ACCESSIBILITY`, `DQ-COMPANION-REDUCED-MOTION`, `DQ-COMPANION-FORCED-COLORS`, `DQ-COMPANION-APP-INTEGRATION`, `DQ-COMPANION-I18N` |
| Glass and palette | Required glass/palette tokens must exist in light and dark scopes; core glass surfaces need a translucent background, backdrop blur, border, shadow, and a non-backdrop-filter fallback. | `DQ-TOKEN-MISSING`, `DQ-GLASS-SURFACE`, `DQ-GLASS-BACKGROUND`, `DQ-GLASS-FALLBACK` |
| Page hierarchy | The overview uses one `h1`, a lower-level home heading, typed room definitions for `home`, `speakers`, `quality`, and `pipeline`, focused-room breadcrumb/back navigation, and mounts only the selected room. | `DQ-HIERARCHY-H1`, `DQ-HIERARCHY-HUB-HEADING`, `DQ-HIERARCHY-ROOM-DEFINITIONS`, `DQ-HIERARCHY-ROOM-TYPE`, `DQ-HIERARCHY-MARKER`, `DQ-HIERARCHY-APP-WIRING` |
| Overview structure | The home overview must remain one clipped, frosted portal/workspace with explicit dividers and a non-interactive depth track. Portal segments must stay transparent, square, and shadowless; rendering `studio-room-card` surfaces in `OverviewWorkspace` is rejected as a card-wall regression. | `DQ-STRUCTURE-PORTAL-MARKUP`, `DQ-STRUCTURE-CARD-WALL`, `DQ-STRUCTURE-WORKSPACE`, `DQ-STRUCTURE-PORTAL-SURFACE`, `DQ-STRUCTURE-DIVIDER`, `DQ-STRUCTURE-PORTAL-TRACK` |
| Locales | Exactly nine locale catalogs must be mapped. English defines the complete key contract; every locale must have exact key parity, non-empty values, and identical placeholder names and multiplicity. `Partial` catalogs, English spreads/direct mappings for non-English locales, and runtime fallback to English are forbidden. | `DQ-I18N-LOCALES`, `DQ-I18N-CATALOG-MAP`, `DQ-I18N-KEY-MISSING`, `DQ-I18N-KEY-EXTRA`, `DQ-I18N-EMPTY`, `DQ-I18N-PLACEHOLDER`, `DQ-I18N-PARTIAL-CATALOG`, `DQ-I18N-ENGLISH-SPREAD-FALLBACK`, `DQ-I18N-ENGLISH-FALLBACK`, `DQ-I18N-PARSE` |
| Native screenshot evidence | The release manifest must cover the light/dark landing views, speaker studio, quality lab, pipeline observatory, preferences, and drag overlay with real PNG captures from the Tauri native webview. | `DQ-SCREENSHOT-EVIDENCE-MISSING`, `DQ-SCREENSHOT-EVIDENCE` |
| Background OCR evidence | The release manifest must cover the real day, night, and scene assets and include engine metadata, hashes, dimensions, text-region results, and placement/occlusion assessment. | `DQ-OCR-EVIDENCE-MISSING`, `DQ-OCR-EVIDENCE` |

The required locale set is:

```text
en, zh-Hans, zh-Hant, ja, ko, es, fr, de, pt-BR
```

The required glass tokens are:

```text
--surface-glass
--surface-strong
--surface-soft
--surface-control
--glass-border
--glass-highlight
--background-overlay
```

The required canonical easing tokens are:

```css
--ease-out: cubic-bezier(0.23, 1, 0.32, 1);
--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
--ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);
```

## Native screenshot evidence

The contract is
`contracts/design-quality-screenshot-evidence.schema.json`.

A release manifest must:

1. identify its build and source revision;
2. describe the capture platform and native window size;
3. map every required coverage slot to a unique evidence record;
4. use project-relative, traversal-safe PNG paths;
5. declare `captureSurface` as `tauri-native-webview`;
6. declare `realCapture: true` and `synthetic: false`;
7. provide the actual pixel dimensions and lowercase SHA-256 digest.

The validator reads each referenced file, checks the PNG signature, recomputes
its dimensions and digest, verifies coverage IDs, and rejects browser-only
captures. A JSON declaration alone is not sufficient.

## Background OCR evidence

The contract is `contracts/design-quality-ocr-result.schema.json`.

A release manifest must cover:

- `day-background`;
- `night-background`;
- `scene-background`.

Each result records the real source image path, digest, dimensions, OCR engine
and version, run time, detected text regions with confidence and bounds, and a
background placement assessment. The assessment includes a recommended
background position, optional primary-subject bounds, occlusion risk, safe
control regions, and notes.

The validator supports PNG, JPEG, and WebP dimension verification. It rejects
path traversal, stale hashes, false dimensions, duplicate IDs, role/coverage
mismatches, and contradictory `noTextDetected`/`textRegions` values.

## Evidence integrity

Do not add placeholder screenshots or invented OCR output to satisfy release
mode. Unit-test image fixtures exercise parsers and contracts only; they are
created in pytest temporary directories, explicitly identified as contract
fixtures, and are never release evidence.

No Design Pack passage, screenshot, OCR observation, capture metadata, or
placement conclusion may be fabricated. If a real native capture or OCR run is
unavailable, the correct release result is failure.

## Current repository expectation

The gate intentionally reports current desktop-source violations rather than
weakening policy, because design-quality work does not modify
`apps/desktop/**` or `apps/desktop/src/i18n/**`.

Do not copy a historical error count into release notes or treat this document
as a snapshot of the working tree. The current result is always the JSON
produced by:

```powershell
conda run -n media-asr python -m tools.design_quality --project-root .
```

A failing source audit is expected whenever imported CSS violates the motion
policy, a locale is incomplete or relies on English fallback, a required
background/companion contract is not wired, or the overview regresses to a
card wall. Policy findings are distinct from `DQ-INTERNAL-FAIL-CLOSED`; the
latter means a check itself could not complete deterministically and must be
treated as an infrastructure failure.

## Tests

Run only the design-quality tests with:

```powershell
conda run -n media-asr python -m pytest -q `
  tests\test_design_quality.py `
  tests\test_design_quality_contracts.py
```

The tests cover imported-CSS discovery and boundary failures, normal versus
ambient motion timing, pointer gating, reduced-motion behavior, all canonical
easing curves, smooth-scroll rejection, the explicit three-layer background
contract, companion hash/accessibility/App wiring, complete nine-locale
key/placeholder parity and no-fallback rules, portal/workspace structure and
card-wall rejection, fail-closed report metadata, release evidence
requirements, JSON Schema validity, image hash/dimension checks, native
capture enforcement, OCR role coverage, and OCR result consistency.
