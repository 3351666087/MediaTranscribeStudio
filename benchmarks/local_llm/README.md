# Local Small-LLM Benchmark

This directory evaluates whether a local Ollama-compatible model can perform
conservative transcript cleanup under a strict, non-diarization output
contract.

The checked-in benchmark profile uses a held-out Chinese-language corpus. That
is a property of this evaluation dataset, not a language restriction in the
MediaTranscribeStudio product contracts. Translation, polishing, and summary
artifacts use validated language metadata and can be evaluated with additional
language-specific corpora. Coverage depends on the installed local model pack
and a task-specific evaluation; no result in this directory establishes
universal language support.

## Product artifact boundary

In the product architecture, local small models may produce translation,
source-language semantic-polishing, and evidence-grounded summary artifacts.
Those outputs are separate, versioned derivatives. They do not replace or
silently rewrite the source transcript, timestamps, speaker assignments,
voiceprint evidence, or human locks.

This benchmark evaluates a narrower correction-proposal contract. It must not
be interpreted as authorization for automatic source-transcript mutation or
speaker reassignment.

## Scope

The model may propose limited text cleanup. It is not allowed to:

- assign, rename, merge, or split speakers;
- split or merge turns;
- alter overlap structure;
- translate or summarize the source;
- perform unrestricted stylistic rewriting;
- invent terminology that is not supported by the supplied glossary or
  evidence.

Because speaker mutation and turn splitting are absent from the writable output
contract, this benchmark does **not** measure speaker-count estimation,
diarization, or speaker-attribution accuracy.

For Ollama models that expose a thinking capability, the harness sends
`"think": false`. Only schema-constrained `message.content` is eligible for
evaluation. If a response contains private reasoning but no usable content, the
sample fails closed as a runtime incompatibility. Reasoning text is never
promoted to the answer or persisted in a report.

## Privacy contract

- The four external ground-truth files are supplied only through environment
  variables.
- No repository fixture contains real meeting text.
- Per-sample results contain opaque IDs, hashes, lengths, metrics, timings, and
  failure codes only.
- Prompts and model responses exist in process memory only.
- Reports run a source-text leak check before they are written.
- The harness writes only below this benchmark directory.

## Required environment variables

```powershell
$env:MTS_BENCH_FINAL_TRANSCRIPT = 'D:\private\final\transcript.json'
$env:MTS_BENCH_PRE_TRANSCRIPT = 'D:\private\pre-refinement\transcript.json'
$env:MTS_BENCH_TURN_CORRECTIONS = 'D:\private\manual-turn-corrections.json'
$env:MTS_BENCH_SENTENCE_DECISIONS = 'D:\private\manual-sentence-decisions.json'
```

The benchmark contains no hard-coded path to private meeting data.

## Reproducible run

Use the required Conda environment:

```powershell
Set-Location <repository-root>\benchmarks\local_llm
$env:PYTHONDONTWRITEBYTECODE = '1'
conda run -n media-asr python -m pytest -q -p no:cacheprovider
conda run -n media-asr python run_benchmark.py `
  --model 'qwen3.5:4b' `
  --max-dev 48 `
  --max-heldout 24 `
  --max-safety 16 `
  --seed 20260721
```

`run.ps1` provides the same flow with parameter overrides.

## Dataset protocol

1. Align the pre-refinement and final transcripts by original turn ID.
2. Exclude manual split turns and whole-turn speaker overrides from semantic
   cleanup scoring.
3. Convert manual split and speaker-override cases into safety challenges.
   Their only acceptable behavior is `review_required` with byte-identical
   source text.
4. Split by time rather than by random turn: the first 70 percent of the
   meeting timeline is development data, and the final 30 percent is held out.
5. When sample limits are supplied, select evenly spaced samples
   deterministically within each already-contiguous partition.

## Outcome classes

The harness distinguishes three deployment outcomes:

- `auto_apply`: eligible only when strict runtime guards and measured
  regression thresholds pass;
- `suggestion_only`: schema-valid suggestions that still require human review;
- `reject`: invalid JSON, contract violations, unsafe edits, or runtime
  failures.

`overlapEscalationF1` measures whether overlap candidates are correctly routed
to human review. `splitOperationF1` is intentionally not applicable because
split generation is forbidden rather than evaluated.

## Checked-in decisions

The formal reports were executed on July 21, 2026. They contain anonymized
metrics and fingerprints only:

- `qwen2.5:1.5b`: `reject_for_production`
- `qwen3.5:4b` Q4_K_M: `reject_for_production`

These are benchmark decision labels for the documented automatic-edit policy;
they are not release-readiness verdicts for the repository as a whole.

For the documented 88-sample `qwen3.5:4b` run:

| Metric | Result |
|---|---:|
| Contract-validity rate | `0.784` |
| Unsafe text-modification rate | `0.205` |
| Auto-apply candidate regression rate | `0.135` |

These results do not justify automatic transcript edits. They apply only to the
named model build, prompt, corpus, harness revision, and execution date. A
parseable response is not evidence of semantic safety, multilingual quality,
or speaker-attribution quality.

## Product relationship

MediaTranscribeStudio's backend supports separate, versioned local artifacts
for translation, source-language semantic polishing, and evidence-grounded
summaries. This cleanup benchmark is narrower: it evaluates whether a small
model can safely propose constrained corrections to transcript text. Speaker
evidence and the immutable source transcript remain outside the model's
writable output contract.

Before enabling any model for another language or business operation, create an
appropriate held-out corpus, preserve the same privacy boundary, define
operation-specific safety failures, and record a new formal decision. Results
must not be generalized across models, quantizations, prompts, languages, or
tasks.
