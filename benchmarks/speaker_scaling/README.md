# Synthetic Speaker-Scaling Benchmark

This package benchmarks the repository's existing speaker-count selection and
clustering entry point:

```python
backend.speaker_pipeline._cluster
```

It does not copy or reimplement the clustering algorithm. The harness creates
deterministic synthetic `EmbeddingRecord` and `SpeechWindow` objects, then calls
the same backend entry point used by the application.

## Purpose

The benchmark checks algorithmic behavior and runtime scaling across increasing
speaker counts. Speaker count is language-neutral, and a five-person meeting is
only one test case. The domain model supports:

- `manual`, with a user-specified positive count;
- `auto`, with count estimation from available evidence;
- `hybrid`, with estimation constrained by configured bounds.

There is no fixed five-speaker product ceiling. Practical limits depend on the
selected models, media characteristics, memory, and compute resources.

Speaker cardinality is language-neutral, but this synthetic benchmark does not
establish transcription or derived-text language coverage. Those capabilities
depend on the installed ASR and local-LLM model packs and require separate
language-specific evaluation. No universal language support is claimed.

## Synthetic scenarios

The benchmark covers four deterministic scenarios:

1. `orthogonal`: mutually orthogonal persistent speaker embeddings;
2. `near-voices`: two speaker centers have cosine similarity `0.94`, while the
   remaining centers are orthogonal;
3. `singleton-outlier`: persistent orthogonal speakers plus one orthogonal
   singleton excluded from persistent-speaker truth;
4. `shuffled-orthogonal`: the orthogonal case supplied in reverse temporal
   order.

## Metric limits

`partitionCorrect` is an exact synthetic partition check that is invariant to
cluster-label permutation. It is **not DER**, must not be presented as DER, and
does not estimate performance on real speech.

For `singleton-outlier`, the singleton's assignment is ignored. Persistent
speakers must remain one-to-one with predicted clusters, and the predicted
count must equal the persistent-speaker count.

The benchmark does not open media or load CAM++, ASR, VAD, diarization, or PDF
components. Reported wall time covers only the Python clustering entry point;
it excludes decoding, model inference, media I/O, and report rendering.

## Run

From the repository root:

```powershell
conda run -n media-asr python -m benchmarks.speaker_scaling
```

The default matrix uses speaker counts `1,2,3,5,8,13,32,64,129`, three samples
per persistent speaker, all three count modes, all four scenarios, and one
repeat.

```powershell
conda run -n media-asr python -m benchmarks.speaker_scaling `
  --speaker-counts 1,2,5,13,129 `
  --samples-per-speaker 4 `
  --modes manual,auto,hybrid `
  --repeat 3 `
  --output-json benchmarks/speaker_scaling/reports/local.benchmark.json
```

Successful standard output is exactly one strict JSON document. When
`--output-json` is used, the same bytes are written to the requested file. Use
the ignored `reports/` directory or the ignored `*.benchmark.json` suffix for
local artifacts.

The harness intentionally configures two spherical k-means refinement
iterations so high-cardinality runs remain practical. Each report records the
complete configuration, backend module SHA-256, resolved algorithm method,
Python runtime, platform, and CPU metadata. Results are specific to the
machine, runtime, and backend source revision that produced them.

## Self-tests

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
conda run -n media-asr python -m pytest -p no:cacheprovider -q `
  benchmarks/speaker_scaling/tests
```

## Interpretation

Use this benchmark to detect regressions in deterministic clustering behavior
and high-cardinality runtime. Do not use it to claim:

- real-media speaker-count accuracy;
- DER, JER, overlap accuracy, or speaker-attribution quality;
- robustness to noise, reverberation, code-switching, or similar voices;
- a validated production resource ceiling.

Those claims require representative multilingual media, human ground truth,
model inference, and separately reported diarization metrics.
