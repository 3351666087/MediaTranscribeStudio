# MediaTranscribeStudio contracts

These JSON Schemas are the language-neutral boundary between the desktop
application, the Python media worker, and the Java PDF renderer.

## Versioning

- Contract versions use semantic versioning.
- Producers MUST write `schemaVersion`.
- Consumers MUST reject unsupported major versions.
- Minor versions may only add optional fields.
- Patch versions may only clarify validation or documentation.

The first production contract family is `1.0.0`.

## Privacy boundary

Fixtures committed to this repository are synthetic. Real meeting text, speaker
names, source paths, and acoustic evidence must stay outside the repository and
be supplied through environment variables during local regression tests.

## Deterministic invariants

JSON Schema validates the portable shape. Runtime validators in Python and Java
also enforce invariants that cannot be expressed safely with portable JSON
Schema alone:

1. Canonical speaker IDs are contiguous and dynamic: `speaker-1` through
   `speaker-N`, where `N` is resolved by `auto`, `manual`, or `hybrid`
   speaker-count policy.
2. Every transcript segment references one of those IDs.
3. Segment IDs are unique and timestamps are monotonic, bounded, and
   non-overlapping unless overlap evidence is explicit.
4. `rawText` is immutable; `normalizedText` and `displayText` changes always
   have an audit revision.
5. A local LLM cannot override a locked acoustic or manual speaker decision.
6. PDF QA never changes transcript text, speaker IDs, or timestamps.
7. Artifact paths are relative to the job output directory and every verified
   artifact has a real SHA-256 digest.

## Contract files

- `report-document.schema.json` — canonical dynamic-speaker transcript
  document. The speaker set is the contiguous `speaker-1..speaker-N` sequence;
  the schema does not impose a product-level speaker-count ceiling.
- `pdf-render-request.schema.json` — renderer invocation.
- `pdf-render-result.schema.json` — renderer outcome and artifact paths.
- `artifact-manifest.schema.json` — content-addressed generated artifacts.
- `pdf-quality-report.schema.json` — hard gates, 14 Design Pack facets, and
  repair queue.
- `job-event.schema.json` — versioned worker event stream.
- `semantic-arbitration.schema.json` — constrained local-LLM proposal.

The committed `synthetic-report-document.json` is deliberately an `N=5`
regression fixture. It verifies a historically difficult meeting shape; it is
not a system limit or default.
