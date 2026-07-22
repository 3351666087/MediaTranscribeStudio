# Local small-LLM benchmark

This benchmark measures whether a local Ollama model can perform **conservative
Chinese transcript cleanup** without being allowed to change speakers, split
overlapping turns, translate, summarize, polish, or invent terminology.

The harness is deliberately self-contained. It writes only below this directory
and never persists source, target, context, model response text, speaker names,
or sentence-level ground truth.

For Ollama models with the `thinking` capability, the harness always sends
`"think": false`. Only schema-constrained `message.content` is eligible for
evaluation. A response that contains private reasoning but has empty content
fails closed as a runtime incompatibility; reasoning text is never promoted to
the answer or persisted in the report.

## Privacy contract

- The four external ground-truth files are supplied only through environment
  variables.
- No repository fixture contains real meeting text.
- Per-sample results contain opaque IDs, hashes, lengths, metrics, timings, and
  failure codes only.
- The prompt and Ollama response exist in process memory only.
- Reports run a source-text leak check before they are written.
- The model cannot emit a speaker assignment or turn split in the output schema.

## Required environment variables

```powershell
$env:MTS_BENCH_FINAL_TRANSCRIPT = 'D:\...\output\transcript.json'
$env:MTS_BENCH_PRE_TRANSCRIPT = 'D:\...\output\pre-speaker-refinement\transcript.json'
$env:MTS_BENCH_TURN_CORRECTIONS = 'D:\...\manual-turn-corrections.json'
$env:MTS_BENCH_SENTENCE_DECISIONS = 'D:\...\manual-sentence-decisions.json'
```

There are no hard-coded paths to the private meeting data.

## Reproducible run

Use the required Conda interpreter:

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

`run.ps1` provides the same command with parameter overrides.

## Dataset protocol

1. Align the pre-refinement and final transcripts by original turn ID.
2. Exclude manual split turns and whole-turn speaker overrides from semantic
   cleanup scoring.
3. Turn manual split and speaker-override cases into safety challenges. Their
   only acceptable behavior is `review_required` with byte-identical source
   text.
4. Split by time, not by random turn: the first 70% of the meeting timeline is
   development data and the final 30% is held out.
5. If limits are supplied, deterministically choose evenly spaced samples
   inside each already-contiguous partition.

## Interpretation

The benchmark distinguishes three deployment outcomes:

- `auto_apply`: eligible only if strict runtime guards and measured regression
  thresholds pass.
- `suggestion_only`: valid structured suggestions that still require a person.
- `reject`: invalid JSON, contract violations, unsafe edits, or runtime failure.

The 1.5B baseline is never allowed to:

- change or infer a speaker;
- split or merge turns;
- silently resolve overlap;
- correct a domain term without a supplied glossary;
- use its self-reported confidence as an application decision.

`overlapEscalationF1` measures whether overlap candidates are correctly sent to
human review. `splitOperationF1` is intentionally reported as not applicable:
split generation is a forbidden capability, not a benchmark target.

## Formal decisions

The checked-in formal reports use the execution date `2026-07-21` and contain
only anonymized metrics and fingerprints:

- `qwen2.5:1.5b`: `reject_for_production`
- `qwen3.5:4b` Q4_K_M: `reject_for_production`

For the 88-sample `qwen3.5:4b` run, the final contract-validity rate was `0.784`,
the unsafe text-modification rate was `0.205`, and the measured regression rate
among auto-apply candidates was `0.135`. The model is therefore disabled in the
production transcript path rather than downgraded to a silent automatic mode.
Its JSON compatibility does not establish speaker attribution quality because
speaker mutation and turn splitting are intentionally absent from the writable
output contract.
