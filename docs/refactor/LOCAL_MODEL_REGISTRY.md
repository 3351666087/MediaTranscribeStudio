# Local Model Registry

`local-model-registry.json` is the reviewed inventory of model artifacts that
may participate in the local production or challenger pipelines. It is an
audit index, not a downloader and not a model-promotion decision.

Each entry records the upstream provider and immutable revision or digest,
license evidence stored beside the model, compatible deployment slots, runtime
engine, declared quantization state, target hardware profile, local path, and
the raw SHA-256 identity of its local manifest. `not-declared` means that the
downloaded artifact metadata does not declare a quantization scheme; it does
not assert a floating-point dtype.

Distinct Ollama tags share one content-addressed model-store root by design.
Their registry identity is therefore the combination of that root and a unique
OCI manifest path; non-Ollama artifacts must continue to use distinct local
roots.

Registry status is inventory metadata, not a deployment pointer:

- `production` records the model's reviewed incumbent history.
- `challenger` records an installed candidate awaiting or carrying comparison
  evidence.
- The active model is selected only by `production.config.json`. A reviewed
  challenger may occupy an active slot without rewriting its registry status.

`deploymentSlots` declares compatibility only. For example, both ERes2NetV2
variants declare `secondary-speaker-verification` with the
`modelscope-eres2netv2` adapter. It does not declare which one is active.

The hardware profile describes the execution host on which the artifacts are
installed. It is not a benchmark result and does not claim that every
challenger has passed latency, memory, or quality gates.

## Validation

Run schema and cross-reference validation without touching model files:

```powershell
D:\MediaTranscribeStudio\runtime\media-asr\python.exe `
  D:\MediaTranscribeStudio\tools\model_registry.py
```

Verify every installed artifact locally, with no network access:

```powershell
D:\MediaTranscribeStudio\runtime\media-asr\python.exe `
  D:\MediaTranscribeStudio\tools\model_registry.py --verify-local
```

Limit expensive hashing to one or more entries by repeating `--model`:

```powershell
D:\MediaTranscribeStudio\runtime\media-asr\python.exe `
  D:\MediaTranscribeStudio\tools\model_registry.py --verify-local `
  --model ollama-qwen3.5-27b-q4-k-m
```

Validation fails closed on unknown schema fields, duplicate JSON keys or
identities, unsupported status/policy/provider values, missing concrete SPDX
license evidence, malformed SHA-256/digests, unsafe paths, source/manifest
mismatches, missing runtimes, and changed files. For MTS manifests,
`--verify-local` recomputes every listed model-file hash. For Ollama, it binds
the tag manifest digest and verifies the config and every referenced OCI layer,
including the license layer and quantization metadata.

The registry does not replace `production-models.lock.json`. The lock controls
installation from ModelScope; the registry unifies the identities of
ModelScope, Hugging Face, and Ollama artifacts after installation.

## Atomic Speaker-Verifier Promotion

`tools/promote_speaker_verifier.py` replaces the active secondary verifier only
when a human blind-review decision binds the incumbent and challenger, the
blind-review and comparison evidence hashes, and the exact current production
configuration SHA-256. The tool recomputes both referenced evidence-file
hashes, verifies the challenger's full local manifest, checks slot/adapter
compatibility, publishes an immutable byte-identical rollback snapshot, takes
a crash-released OS advisory lock, rechecks the configuration SHA-256, and
atomically replaces the configuration. The new configuration retains the
decision, evidence, and rollback snapshot identities.

```powershell
D:\MediaTranscribeStudio\runtime\media-asr\python.exe `
  D:\MediaTranscribeStudio\tools\promote_speaker_verifier.py `
  --config D:\MediaTranscribeStudio\production.config.json `
  --registry D:\MediaTranscribeStudio\local-model-registry.json `
  --decision D:\mts-eval\speaker-promotion-decision.v1.json `
  --blind-review-artifact D:\mts-eval\blind-review-result.v1.json `
  --comparison-artifact D:\mts-eval\speaker-comparison.v1.json `
  --rollback-root D:\mts-eval\production-config-rollbacks `
  --expected-config-sha256 <current-config-file-sha256>
```

Registry `usage.status` is intentionally not a promotion gate. Reusing a
decision or racing an older configuration fails the compare-and-swap because
the decision is valid for exactly one previous configuration hash. The lock
file is intentionally retained as a stable inode; only its OS lock is
authoritative, so stale PID metadata after a crash cannot block a later
promotion.

## Atomic Semantic-Model Promotion

`tools/promote_semantic_model.py` uses `speaker.localLlmModel` and
`speaker.localLlmModelDigest` as the only active semantic-model pointer.
`production` means the current incumbent, not a protected model: a compatible
challenger that wins a same-case-batch blind product review can replace it
immediately. There is no incumbent margin, cooling-off period, or registry
status gate. The displaced model remains eligible to win a later blind review
and replace the new incumbent immediately.

The promotion decision must bind the exact current production-config SHA-256,
the incumbent and challenger registry identities, one truth-redacted case-set
SHA-256, a blind batch ID, distinct blind-review and comparison artifact
hashes, and an explicit reviewer decision naming the challenger as the winner.
It must also attest that every candidate was reviewed on the same case batch
and model identities remained hidden during review. The reviewer may be a
human or an explicitly delegated Codex semantic reviewer; automatic scores
alone cannot authorize promotion.

The tool verifies the challenger's complete local Ollama manifest and blobs,
matches the declared incumbent to the active model name and digest, shares the
same production-config OS lock as speaker-model promotion, writes an immutable
byte-identical rollback snapshot, rechecks the compare-and-swap immediately
before replacement, and atomically replaces the config. The receipt is emitted
as JSON on stdout and retains all decision, evidence, active-model, and
rollback identities.

```powershell
D:\MediaTranscribeStudio\runtime\media-asr\python.exe `
  D:\MediaTranscribeStudio\tools\promote_semantic_model.py `
  --config D:\MediaTranscribeStudio\production.config.json `
  --registry D:\MediaTranscribeStudio\local-model-registry.json `
  --decision D:\mts-eval\semantic-promotion-decision.v1.json `
  --blind-review-artifact D:\mts-eval\blind-product-review.v1.json `
  --comparison-artifact D:\mts-eval\semantic-model-comparison.v1.json `
  --rollback-root D:\mts-eval\production-config-rollbacks `
  --expected-config-sha256 <current-config-file-sha256>
```
