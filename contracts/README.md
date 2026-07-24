# MediaTranscribeStudio Contracts

This directory defines the versioned JSON boundaries shared by the desktop
application, Python worker, local business-processing pipeline, reporting
layer, and Java PDF renderer.

The schemas are language-neutral. They represent language with canonical
metadata rather than assuming that transcript content is English, Chinese, or
any other single language.

## Versioning

- Contract versions use semantic versioning.
- Producers must write `schemaVersion`.
- Consumers must reject unsupported major versions.
- Minor versions may add optional fields without changing existing meaning.
- Patch versions may clarify validation or documentation without changing the
  portable data model.

The current contract family uses version `1.0.0`.

## Language identifiers

Request boundaries accept practical BCP-47 language tags and canonicalize them
before persistence:

- canonical tags such as `en`, `en-US`, and `fr-CA` identify known languages;
- `auto` is allowed only as a request-time instruction;
- `und` identifies an undetermined language;
- `mul` identifies intentionally multilingual content.

Schema-valid language metadata does not guarantee that an installed ASR or
local LLM model supports that language. Model capability is validated
separately against the installed model packs and runtime configuration. These
contracts do not claim universal language coverage.

## Speaker-count invariants

The contracts do not impose a fixed five-speaker ceiling. Speaker count is
resolved by one of three policies:

- `auto`: estimate from available evidence;
- `manual`: require a user-specified positive count;
- `hybrid`: estimate within explicit lower and upper bounds.

For a resolved count `N`, canonical IDs are contiguous from `speaker-1` through
`speaker-N`. Runtime validators enforce agreement between the declared count,
speaker set, profiles, score vectors, and segment assignments.

The committed five-speaker synthetic fixture is a regression case, not a
default or product limit.

## Deterministic runtime invariants

JSON Schema validates portable structure. Python and Java validators also
enforce cross-field rules that are unsafe or impractical to express in portable
JSON Schema alone:

1. Every transcript segment references a declared speaker ID.
2. Segment IDs are unique, and timestamps are bounded and monotonic.
3. Overlap is represented explicitly rather than inferred from conflicting
   ordinary segments.
4. Source recognition evidence is preserved; derived text requires explicit
   provenance and versioning.
5. Local LLM output cannot override locked acoustic or human speaker
   decisions.
6. Translation, polishing, and summary outputs remain separate from the source
   transcript.
7. Summary evidence references valid segment IDs and time ranges.
8. PDF quality processing does not change transcript text, speaker IDs, or
   timestamps.
9. Artifact paths are relative to the job output directory, and verified
   artifacts include real SHA-256 digests.

## Contract inventory

| File | Purpose |
|---|---|
| `artifact-manifest.schema.json` | Content-addressed artifact inventory and integrity metadata. |
| `business-processing-request.schema.json` | Opt-in local translation, source-language polishing, and summary request. |
| `job-event.schema.json` | Versioned worker event stream. |
| `output-customization.schema.json` | Evidence-bearing canonical report, subtitle, delivery, export, and reversibility snapshot. |
| `output-recipe.schema.json` | Compact native desktop presentation recipe compiled into canonical output customizations. |
| `pdf-quality-report.schema.json` | PDF hard gates, quality findings, and deterministic repair queue. |
| `pdf-render-request.schema.json` | Request sent to the Java PDF sidecar. |
| `pdf-render-result.schema.json` | Renderer outcome and generated artifact paths. |
| `polish-output.schema.json` | Versioned source-language semantic-polishing artifact. |
| `report-document.schema.json` | Canonical transcript and report document with a dynamic speaker set. |
| `semantic-arbitration.schema.json` | Constrained semantic proposal for human or deterministic arbitration. |
| `summary-output.schema.json` | Evidence-grounded summary artifact. |
| `translation-output.schema.json` | Versioned translation artifact with validated language metadata. |

`validate_contracts.py` provides repository-side schema validation support.

## Worker event stream

`job-event.schema.json` defines the ordered, language-neutral event envelope
published by the worker. Event payloads may reference transcript or report
artifacts in any supported language, including multilingual content; language
and model coverage are determined by the installed runtime packs.

The public event types are:

- `job.started`
- `stage.started`
- `stage.progress`
- `artifact.created`
- `review.required`
- `review.decision.persisted`
- `warning`
- `job.failed`
- `job.completed`
- `job.cancelled`

`review.required` announces review work that requires a human decision.
`review.decision.persisted` confirms that the worker durably recorded that
decision. Consumers must still use the referenced auditable artifacts as the
source of truth; the event does not authorize silent changes to transcript
text, speaker identity, timestamps, or language metadata.

## Business-processing boundary

Translation, semantic polishing, and summaries are explicit derived operations.
Their contracts preserve:

- source document identity and version;
- requested and produced language metadata;
- local model and provider provenance;
- deterministic validation status;
- source-segment evidence where required;
- separation from source transcript and speaker evidence.

Local small models operate only within these derived-artifact boundaries. They
cannot silently rewrite the source transcript, timestamps, speaker assignments,
voiceprint evidence, or human locks. Any source-language correction that is
accepted into a later source revision must remain explicit, attributable, and
auditable.

The default local provider implementation is loopback-only and rejects
redirects, but transport policy is enforced by the backend rather than by JSON
Schema.

## PDF boundary

`report-document.schema.json` is the canonical input to the reporting and PDF
path. The Java sidecar consumes the render request and uses OpenHTMLtoPDF plus
PDFBox. Renderer output and quality results are returned through their own
schemas so that PDF generation cannot silently alter transcript content.

`language` identifies the original transcript content. The optional
`reportLocale` field controls report-interface copy only. It never translates
or rewrites transcript text, speaker identities, timestamps, evidence, or
audit records. Chinese report locales use the established Chinese copy pack;
all other requested locales currently use a truthful `en-US` fallback until a
native copy pack is implemented. Segment-level `language` and text direction
remain independent from the report interface.

The existence of these contracts does not establish end-to-end desktop or
release readiness. Full integration and real-media acceptance remain separate
validation requirements.

## Privacy

Committed fixtures must be synthetic. Keep real meeting media, transcript text,
speaker names, source paths, acoustic evidence, embeddings, human ground truth,
and sensitive logs outside the repository. Private regression inputs should be
supplied through local configuration or environment variables.
