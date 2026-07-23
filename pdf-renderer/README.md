# MediaTranscribeStudio Java PDF Renderer

The designated production PDF boundary is **Java 17 + OpenHTMLtoPDF + Apache
PDFBox**. Python callers may assemble a versioned report document and invoke
this sidecar, but production PDF generation, rendering, inspection, and repair
remain Java-only.

> **Validation status:** the renderer and its quality contracts are under
> active pre-release validation. Successful unit or renderer tests do not prove
> commercial readiness, Design Pack release eligibility, or completed
> real-media acceptance.

## Implemented invariants and intended production behavior

- Supports any positive speaker count `N`; there is no fixed five-speaker branch.
  Regression coverage includes `N=1/2/5/8/13`.
- Speaker presentation has one canonical styling entry point:

  ```java
  new SpeakerPalette().style(speaker.order, speaker.colorToken)
  ```

- Bundles the open-source static TrueType font `LXGW WenKai v1.522`. Runtime
  rendering does not scan system fonts, access the network, or load remote
  resources.
- Produces searchable text for the scripts covered by the bundled fonts and
  validated renderer path. PDFBox verifies the embedded font, transcript body,
  segment count, timestamps, and complete speaker set. This does not imply
  universal script or language coverage.
- Enforces 13 non-compensable hard gates and 14 `AESTHETIC-*` facets. Facet
  weights sum to exactly `1.0`, and the minimum passing score is `85`.
- Allows at most five deterministic layout-repair rounds. Repairs may adjust
  only templates, CSS, font sizes, margins, and pagination; the transcript
  content hash must remain identical before and after repair.
- Renders every page to PNG and also produces a contact sheet, PDF inspection,
  quality report, repair queue, and artifact manifest.
- Writes exactly one JSON object to CLI stdout. All logs go to stderr. Unknown
  fields, trailing JSON, parse failures, and quality failures produce a non-zero
  exit code.

## Multilingual and bidirectional coverage

- Treats transcript language and report-interface locale as independent
  metadata. `language` and each segment's language describe immutable source
  content; optional `reportLocale` selects interface copy without translating
  or rewriting transcript evidence.
- Chinese report locales preserve the established Chinese copy pack. Other
  requested locales currently fall back to professional English copy and
  truthfully emit `en-US` on the XHTML root until a native copy pack exists.
- Emits canonical BCP-47 metadata through both `lang` and `xml:lang`, with
  per-segment language and direction overrides.
- Supports left-to-right and right-to-left layout, including ICU-based
  bidirectional splitting and reordering plus direction-aware CSS mirroring.
- Regression coverage intentionally retains a Chinese source-transcript
  fixture under `zh-CN`, alongside right-to-left coverage for tags
  including `ar-SA`, `fa-IR`, `he-IL`, `ur-PK`, and `az-Arab`. This fixture is
  test evidence, not public-interface prose or a claim of complete language
  support.

## Build

```powershell
mvn "-Dmaven.repo.local=target/m2" clean test
mvn "-Dmaven.repo.local=target/m2" package
```

The standalone shaded JAR is written to:

```text
target/pdf-renderer.jar
```

## CLI

```powershell
java -jar target/pdf-renderer.jar --request D:\jobs\request.json
```

Requests conform to `schemas/pdf-render-request.schema.json` and reference a
versioned `ReportDocument` through `reportDocumentPath`.

`reportDocumentPath` must be located inside `outputDirectory`. This constraint
keeps manifest entries safely relative and prevents output-directory traversal.

### Successful output layout

```text
input/report-document.json
render/report.xhtml
render/report.pdf
artifacts/screens/page-001.png
artifacts/contact-sheet.png
artifacts/pdf-inspection.json
artifacts/pdf-extracted-text.txt
artifacts/quality-report.json
artifacts/repair-queue.json
artifacts/manifest.json
artifacts/render-result.json
qa/round-01/...
```

## Internalized Design Pack validation

This module internalizes the applicable offline quality contract from
`frontend-design-pack-global`, including hierarchy, typography, color,
density, restraint, real-content stress, font-failure, image-failure, and
script-failure checks.

The external template package currently declares `implementationReady=false`
and `releaseEligible=false`. Accordingly, this module claims only that its
project-level PDF QA contract has been internalized; it does not claim that the
external template package or this renderer is release-ready.
