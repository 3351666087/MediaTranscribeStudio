# Output Customization / DIY Domain

## Scope

`backend/output_customization.py` is a pure, offline configuration domain for
commercial output customization. It composes user choices that will later be
compiled into the existing:

- `report-style` contract for report appearance;
- `subtitle-output` contract for subtitle planning;
- `subtitle-delivery` boundary for sidecar, soft-mux, and burn-in execution;
- `subtitle-visual-qa` contract for representative-frame visual gates.

It does **not** render PDF/HTML, write subtitles, invoke FFmpeg, inspect fonts,
probe media, or mutate source artifacts. The canonical contract is
`contracts/output-customization.schema.json`.

## Supported DIY controls

### Reports

- layout and density;
- A4, A5, Letter, or Legal paper;
- portrait or landscape orientation and per-edge margins;
- font pack, writing-system fallbacks, and embedding policy;
- cover style and SHA-256-bound artwork identity;
- header and footer content and page/chapter/title fields;
- bounded watermark text, opacity, rotation, and repetition;
- semantic, fixed-interval, or manual chapters;
- speaker legend position and fields;
- elapsed or SMPTE timestamp format and placement;
- accent color and high-contrast mode.

### Subtitles

- SRT, WebVTT, or ASS;
- existing shipped themes, including YouTube-style themes;
- font pack, font size, weight, italic, and line height;
- alignment and percentage safe areas;
- foreground, outline, shadow, and background styling;
- cue line, duration, gap, and reading-speed policy;
- speaker label template and position;
- deterministic automatic, accessible, or monochrome speaker colors;
- unlimited speaker-specific color overrides;
- evidence-bound real word-progress karaoke only.

### Delivery and export

- sidecar, soft-mux, or burn-in intent;
- deferred reusable presets or bound source/output paths;
- soft-mux container and probed/explicit subtitle codec intent;
- explicit high-quality burn-in strategy;
- PDF, HTML, DOCX, ODT, JSON, text, Markdown, CSV, and TSV intents;
- directory or ZIP packaging;
- safe tokenized file names;
- mandatory checksum and provenance manifests.

The configuration expresses intent. Actual container, codec, renderer, font,
and packaged-runtime capabilities still have to pass their existing execution
and QA boundaries.

## Defaults

`default_output_customization()` resolves a complete `studio-balanced`
snapshot:

- modern editorial, comfortable A4 portrait report;
- Noto-oriented global font fallbacks without claiming installation or
  embedding;
- editorial cover, header, footer, chapters, speaker legend, and millisecond
  elapsed timestamps;
- ASS `youtube-clean` subtitles at 52 px with two-line, safe-area, outline,
  shadow, and background defaults;
- deterministic speaker colors with label-and-pattern collision fallback;
- karaoke off;
- sidecar delivery with paths deferred and execution disabled;
- PDF report, JSON/text transcript, SRT/WebVTT alternates, and JSON data;
- source preservation, no overwrite, mandatory provenance, and reversible
  derived-artifact publication.

`resolve_output_customization(overrides)` accepts strict partial overrides.
Unknown fields are rejected before merging.

## Canonicalization and verification

Every accepted snapshot:

1. contains all fields and safety declarations;
2. normalizes text to Unicode NFC;
3. normalizes colors to uppercase `#RRGGBB`;
4. sorts unordered export selections by stable domain order;
5. sorts speaker overrides by speaker ID;
6. sorts evidence hash sets;
7. rejects NaN and infinity;
8. serializes as UTF-8 JSON with sorted object keys and no insignificant
   whitespace;
9. exposes the lowercase SHA-256 of those exact canonical bytes;
10. validates against Draft 2020-12 JSON Schema.

Primary APIs:

```python
default_output_customization()
resolve_output_customization(overrides)
validate_output_customization(payload)
canonical_output_customization_json(payload)
deterministic_output_customization_hash(payload)
```

## Reversible editing

`apply_output_customization_patch(current, overrides)` returns a frozen
`OutputCustomizationChange` containing canonical before/after snapshots and
their SHA-256 digests.

`change.revert(current)` succeeds only when:

- both stored snapshots are still canonical and hash-valid; and
- `current` exactly matches the verified after hash.

This prevents an undo operation from silently overwriting unrelated later
edits.

Workflow reversibility is also explicit in the canonical configuration:

- source media is immutable;
- only derived artifacts may be created;
- reverting means deleting derived artifacts;
- sidecar and soft-mux bitstreams are marked reversible;
- burn-in is correctly marked non-reversible at the derived-media bitstream
  level, while the untouched source remains authoritative.

## Fail-closed safety

### Source and output paths

- source overwrite is a schema constant and runtime invariant;
- existing-output overwrite is forbidden;
- atomic publication remains required;
- deferred presets have two null paths and are not execution-ready;
- bound presets require both paths;
- Windows drive, UNC, and local normalized aliases are rejected at runtime;
- readiness is derived from path binding and cannot be asserted manually.

This domain does not check whether an output already exists. The delivery
executor retains that filesystem race and no-clobber responsibility.

### HDR burn-in

Burn-in is never inferred or defaulted. It requires:

- an explicit named high-quality strategy;
- `sourceDynamicRange: "sdr"`;
- verified, SHA-256-bound dynamic-range probe evidence;
- representative-frame visual QA.

HDR and unknown dynamic range fail closed. Sidecar and soft-mux remain
available.

### No fake karaoke

`word-progress` requires:

- ASS output;
- a forced aligner, native word timestamps, or human-authored timing;
- `verified: true`;
- exact subtitle and transcript SHA-256 bindings;
- a positive word count.

Segment interpolation, even splitting, language-model timing, missing
evidence, and a karaoke-named theme without real timing are rejected.

### Font claims

The default is `not-asserted`; successfully naming or rendering a font does
not prove installation, packaging, or embedding.

Any verified availability or embedding claim requires:

- an approved verification method;
- `verified: true`;
- a manifest SHA-256;
- one or more font-file SHA-256 digests.

Evidence without a claim is also rejected. Output-specific embedding claims
must ultimately come from PDFBox or media-attachment inspection rather than a
configuration label.

## Arbitrary speaker counts

There is no `speakerCount`, `maximumSpeakers`, or five-speaker ceiling.
Automatic color identity uses a deterministic speaker-ID hash algorithm.
Explicit overrides are an unbounded array with runtime-unique speaker IDs.
When color distance can no longer carry identity alone, the mandatory
collision fallback is `label-and-pattern`; downstream visual QA remains the
authority for actual distinguishability.

## Tests

Focused tests cover:

- safe defaults and Draft 2020-12 validation;
- canonical JSON and SHA-256 determinism;
- Unicode, color, list, hash, and speaker normalization;
- strict missing/unknown-field rejection;
- report, subtitle, timestamp, chapter, and export cross-field rules;
- arbitrary speaker counts;
- font-evidence gates;
- source/output alias protection;
- sidecar, soft-mux, and every named burn-in strategy;
- HDR, fake-karaoke, overwrite, and visual-QA fail-closed behavior;
- hash-bound apply/revert and tamper detection;
- runtime gates that are intentionally stronger than portable JSON Schema.

Run:

```powershell
conda run -n media-asr python -m pytest `
  tests/test_output_customization.py `
  -q `
  -p no:cacheprovider `
  --basetemp=C:\Users\33516\Documents\Playground\mts-pytest-temp\output-customization
```

No test requires FFmpeg, a GUI, network access, or installed fonts.
