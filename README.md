# MediaTranscribeStudio

MediaTranscribeStudio is an offline-first transcription and speaker-attribution
system for local audio and video. The repository is being refactored around
versioned contracts, dynamic speaker counts, multilingual metadata, auditable
human review, local business processing, and a Java PDF sidecar.

> **Project status:** active refactor and validation. This repository must not
> be described as commercially release-ready. Real-model acceptance, complete
> desktop integration, packaging migration, and end-to-end quality validation
> are still in progress.

Public product documentation is written in professional English for an
international audience. Source media, transcript content, and explicitly
identified regression fixtures may use other languages.

## What the repository supports

- **Dynamic speaker counts:** the domain model supports `auto`, `manual`, and
  `hybrid` speaker-count policies. Manual mode accepts any positive count, and
  hybrid mode accepts explicit lower and upper bounds. There is no fixed
  five-speaker product limit; practical limits depend on the selected models,
  media, memory, and compute resources.
- **Evidence-preserving transcription:** recognized text, timestamps, acoustic
  evidence, speaker decisions, and human overrides are represented separately
  so later operations do not silently rewrite source evidence.
- **Multilingual metadata:** request boundaries accept practical BCP-47
  language tags. Persisted artifacts use canonical tags such as `en-US`, with
  `und` for undetermined language and `mul` for multilingual content.
  Actual transcription and derived-text coverage depends on the installed ASR
  and local-LLM model packs; the repository does not claim universal language
  support.
- **Local business processing:** the backend and versioned contracts implement
  opt-in translation, source-language semantic polishing, and
  evidence-grounded summaries as separate derived artifacts produced through a
  local LLM provider.
- **Java PDF production path:** the implemented PDF sidecar uses
  OpenHTMLtoPDF `1.0.10` and Apache PDFBox `2.0.30`. Python prepares canonical
  report data and invokes the sidecar; it is not the production PDF renderer.
- **Fail-closed review:** unresolved speaker counts, invalid contracts,
  unavailable dependencies, and insufficient evidence are surfaced as
  failures or review requirements instead of being reported as successful
  output.

The desktop source contains work toward these capabilities, but the complete
desktop workflow has not been release-validated. Backend or contract support
does not by itself prove that every feature is exposed and production-ready in
the current UI.

## Architecture

```text
Local audio or video
  -> FFmpeg media preparation
  -> FunASR speech activity and timing boundaries
  -> Qwen3-ASR transcription
  -> CAM++ speaker embeddings
  -> dynamic speaker-count selection and clustering
  -> selective ERes2NetV2 or pyannote escalation
  -> human review and versioned transcript artifacts
  -> optional local translation, polishing, and summaries
  -> canonical ReportDocument
  -> Java OpenHTMLtoPDF + PDFBox sidecar
  -> PDF artifacts and quality reports
```

The repository contains a cost-aware speaker pipeline that uses CAM++ as the
primary embedding path and reserves additional local processing for difficult
segments. Model availability and compatibility must be verified in the target
`media-asr` environment before a real job is accepted.

## Speaker-count policy

Speaker count is a per-job policy:

| Mode | Behavior |
|---|---|
| `auto` | Estimate the count from available evidence. |
| `manual` | Require the user-specified positive count. |
| `hybrid` | Estimate within user-specified minimum and maximum bounds, with an optional prior. |

After a count `N` is resolved:

1. Canonical speaker IDs are `speaker-1` through `speaker-N`.
2. The declared count, speaker set, score vectors, profiles, and segment
   assignments must agree.
3. Manual mode must preserve the requested count.
4. Hybrid mode must remain within its configured bounds.
5. Resource exhaustion must fail or request review; it must not alter the
   semantic participant count.
6. Explicit human locks take precedence over automatic semantic or acoustic
   proposals.

The synthetic scaling benchmark exercises counts up to 129 by default, but
synthetic partition correctness is not evidence of real-media diarization
accuracy. See
[`benchmarks/speaker_scaling/README.md`](benchmarks/speaker_scaling/README.md).

## Multilingual policy

Language identifiers are canonicalized at system boundaries:

- `auto` is a request-time instruction and is not a persisted content
  language.
- Canonical BCP-47 tags, for example `en`, `en-US`, or `fr-CA`, identify known
  artifact languages.
- `und` identifies content whose language is not determined.
- `mul` identifies content that intentionally contains multiple languages.

This policy is language-neutral, but it does not imply universal model
coverage. The selected ASR and local LLM models must support the requested
languages, and all required model files must be available locally.

## Translation, polishing, and summaries

The backend supports three opt-in business operations:

- translation to one or more validated target languages;
- semantic polishing in the source language;
- summaries whose claims reference valid source segment IDs and time ranges.

These operations produce separate, versioned artifacts. They do not mutate the
source transcript, speaker assignments, speaker identity evidence, acoustic
evidence, or timestamps. No local model may silently promote a translation,
polish, summary, or semantic suggestion into source evidence. The default
Ollama-compatible provider is restricted to loopback endpoints, rejects
redirects, and requires schema-valid JSON output.

The local small-model benchmark currently rejects the tested models for
automatic transcript edits. Those results apply only to the documented models,
prompt, corpus, and benchmark date; they do not establish general model quality
or diarization accuracy. See
[`benchmarks/local_llm/README.md`](benchmarks/local_llm/README.md).

## PDF path

The production PDF boundary is the Java module in `pdf-renderer/`:

```text
ReportDocument JSON
  -> canonical XHTML and CSS
  -> OpenHTMLtoPDF 1.0.10
  -> PDFBox 2.0.30 validation and inspection
  -> PDF, page images, manifests, and quality reports
```

The renderer is designed to preserve transcript text, timestamps, speaker IDs,
and language metadata. A PDF quality result must not repair content by changing
those values. Full product-level visual and release acceptance is not yet
established.

## Repository layout

```text
apps/desktop/                  React, TypeScript, and Tauri desktop work
backend/                       Python orchestration, domain logic, models, and local business processing
benchmarks/                    Synthetic and de-identified evaluation harnesses
contracts/                     Versioned JSON Schemas for process and artifact boundaries
docs/refactor/                 Architecture, migration, and parity tracking
packaging/                     Legacy Python packaging infrastructure retained during migration
pdf-renderer/                  Java OpenHTMLtoPDF and PDFBox sidecar
reporting/                     ReportDocument assembly and sidecar invocation
tests/                         Python contract and integration tests
production.config.example.json Example production configuration shape
```

Legacy Python application and packaging paths remain during migration. Their
presence does not make them the intended final desktop or release architecture.

## Development requirements

The active refactor uses:

- Windows 11 for the primary desktop development path;
- Node.js 22 or later;
- Rust stable;
- Python in the `media-asr` Conda environment;
- FFmpeg;
- JDK 17 (the Maven build currently enforces the Java 17 release line);
- Maven 3.9 or later;
- locally installed model and font files.

### Python validation

```powershell
conda run -n media-asr python -m pytest -q
```

Strict worker preflight:

```powershell
conda run -n media-asr python -m backend.worker `
  --config C:\absolute\path\to\production.config.json `
  --preflight
```

`production.config.example.json` documents the configuration shape. Copy it to
an untracked file and replace placeholders with real absolute paths before
running local jobs.

### Desktop validation

```powershell
Set-Location apps/desktop
npm ci
npm run lint
npm run typecheck
npm test -- --run
npm run build

Set-Location src-tauri
cargo fmt --check
cargo check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

These commands describe the source validation surfaces; this README does not
claim that the complete desktop product currently passes release acceptance.

### Java PDF validation

```powershell
Set-Location pdf-renderer
mvn test
```

## Contract and process boundaries

Rust and Python communicate through versioned JSONL messages. Python owns
domain invariants, model orchestration, persistence, business-processing
artifacts, and Java sidecar invocation. Rust owns desktop child-process
supervision, cancellation, timeouts, and recovery.

The schemas in [`contracts/`](contracts/) define portable boundaries for:

- transcript and report documents;
- worker events;
- speaker and semantic proposals;
- business-processing requests and outputs;
- PDF render requests, results, manifests, and quality reports.

JSON Schema validates portable structure. Python and Java validators enforce
additional cross-field invariants that are unsafe or impractical to express in
portable schema alone.

## Validation limits

The following areas remain unresolved and must be validated before a release
claim:

- complete Tauri, Rust, Python, model, Java, and PDF quality-gate integration;
- real-media accuracy across languages, speaker counts, overlap patterns, and
  adverse acoustic conditions;
- completion of the required real-media automatic-speaker and independent
  manual-five-speaker acceptance packages;
- local model dependency compatibility and resource envelopes;
- full desktop exposure and behavior for business-processing controls;
- accessibility, interaction, and visual acceptance against the applicable
  design guidance;
- a passing Design Pack release evaluation; the current external pack reports
  `implementationReady=false` and `releaseEligible=false`;
- migration from the legacy Python packaging toolchain to the intended desktop
  distribution path.

Metrics must be reported by domain. Synthetic partition checks are not DER or
JER, schema-valid LLM output is not semantic safety, and successful PDF
generation is not proof of content or visual fidelity.

## Privacy and repository policy

- Do not commit meeting media, transcripts, names, speaker embeddings, acoustic
  evidence, ground truth, or sensitive logs.
- Do not commit model weights, model caches, Conda environments, generated
  desktop builds, Rust `target`, or Maven `target`.
- Keep credentials, access tokens, private download URLs, and personal
  absolute paths out of committed configuration.
- Keep production media processing and business processing local. Networked
  bootstrap or packaging utilities in the legacy `packaging/` directory are
  separate distribution tooling and must not be confused with the local
  meeting-content processing boundary.
- Keep artifacts versioned, traceable, and reproducible.
