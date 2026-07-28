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
5. Local LLM output cannot override human locks or fabricate acoustic
   evidence; it may select any eligible bound candidate.
6. Semantic arbitration suggestions, translation, and summary outputs remain
   separate from the source transcript.
7. Summary evidence references valid segment IDs and time ranges.
8. PDF quality processing does not change transcript text, speaker IDs, or
   timestamps.
9. Artifact paths are relative to the job output directory, and verified
   artifacts include real SHA-256 digests.
10. Voice-activity classifications agree with their VAD windows, exact speech
    coverage, source hash, and transcribable-speech disposition.
11. Mandatory semantic arbitration receives a source-media-, transcript-,
    producer-revision-, payload-, candidate-, group-, and lattice-hash-bound
    candidate space. A missing or single-candidate domain is explicitly
    unavailable and cannot be reported as semantically repaired.
12. Job-level semantic arbitration must decide every group exactly once by
    ranking all eligible candidate IDs or requesting a bounded domain
    challenger. Composition remains isolated, recomputes the selected lattice,
    and rejects unresolved requests, cross-domain inconsistencies, human-lock
    changes, and hash rebinding.

## Contract inventory

| File | Purpose |
|---|---|
| `artifact-manifest.schema.json` | Content-addressed artifact inventory and integrity metadata. |
| `asr-evidence.schema.json` | Immutable model-, source-window-, token-, score-, candidate-ID-, and hash-bound ASR top-1/N-best evidence. |
| `business-processing-request.schema.json` | Opt-in local translation and summary request. |
| `final-adjudicated-transcript.schema.json` | Unified hash-bound final disposition: legacy reviewed suggestions (`1.1`), mandatory candidate composition (`1.2`), or verified absence of transcribable speech. |
| `job-event.schema.json` | Versioned worker event stream. |
| `output-customization.schema.json` | Evidence-bearing canonical report, subtitle, delivery, export, and reversibility snapshot. |
| `output-recipe.schema.json` | Compact native desktop presentation recipe compiled into canonical output customizations. |
| `pdf-quality-report.schema.json` | PDF hard gates, quality findings, and deterministic repair queue. |
| `pdf-render-request.schema.json` | Request sent to the Java PDF sidecar. |
| `pdf-render-result.schema.json` | Renderer outcome and generated artifact paths. |
| `report-document.schema.json` | Canonical transcript and report document with a dynamic speaker set. |
| `semantic-candidate-generation.schema.json` | Registered bounded challenger results and the deterministically extended lattice used for the next arbitration round. |
| `semantic-candidate-lattice.schema.json` | Hash-bound speech disposition, complete speaker timeline/cardinality, per-turn speaker, language-span, and ASR-text candidate groups with explicit domain availability. |
| `semantic-composition.schema.json` | Deterministically recomputed, isolated post-arbitration speech, complete timeline, speaker, language, and final-text state. |
| `semantic-job-arbitration.schema.json` | Complete-job candidate rankings and bounded candidate-generation requests from the mandatory local semantic model. |
| `semantic-arbitration.schema.json` | Constrained semantic proposal for human or deterministic arbitration. |
| `summary-output.schema.json` | Evidence-grounded summary artifact. |
| `translation-output.schema.json` | Versioned translation artifact with validated language metadata. |
| `voice-activity.schema.json` | Auditable speech-candidate, lexical-speech, and no-speech terminal evidence. |

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

Translation and summaries are explicit derived operations. Mandatory semantic
arbitration produces constrained review suggestions before those optional
operations. Their contracts preserve:

- source document identity and version;
- requested and produced language metadata;
- local model and provider provenance;
- deterministic validation status;
- source-segment evidence where required;
- separation from source transcript and speaker evidence.

Local models operate only within these candidate and derived-artifact
boundaries. They cannot rewrite source media, raw ASR evidence, candidate
identity, voiceprint evidence, or human locks. A mandatory arbitrator may
select any eligible candidate, and `semantic-composition.v1` may project those
selections into an isolated delivery document without mutating the persisted
source transcript. Independent polishing is not part of the product or public
contract; source-language repair belongs exclusively to mandatory semantic
arbitration.

Formal quality scoring uses only
`final-adjudicated-transcript.schema.json`. Legacy `1.1` speech artifacts bind
the reviewed suggestion-only transcript. Version `1.2` binds the input lattice,
job arbitration, deterministic composition, resolved review queue, complete
selected timeline, and selected `speakerId`, `language`, `startMs`, `endMs`,
and `finalText` tuple. Its
`no-transcribable-speech` disposition contains no transcript, speaker, or
semantic fields and instead binds the source media to validated
`voice-activity.v1.json` evidence. VAD, diarization, voiceprint, LID, ASR, and
semantic candidate metrics remain useful diagnostics, but no individual
front-model result can authorize release. Missing or incomplete semantic
processing, open review work, an invalid no-speech disposition, or a broken
source hash blocks the final artifact and therefore blocks scoring.

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
