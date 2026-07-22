# Speaker-scaling synthetic benchmark

This package benchmarks the repository's existing speaker-count selection and
clustering entry point:

```python
backend.speaker_pipeline._cluster
```

It does not copy or reimplement the clustering algorithm. It creates only
deterministic synthetic `EmbeddingRecord` and `SpeechWindow` objects and calls
the same backend entry point used by the application.

## Scope and metric

The benchmark covers four deterministic synthetic scenarios:

1. `orthogonal`: mutually orthogonal persistent speaker embeddings.
2. `near-voices`: two speaker centers have cosine similarity `0.94`; remaining
   centers are orthogonal.
3. `singleton-outlier`: persistent orthogonal speakers plus one orthogonal
   singleton that is excluded from persistent-speaker truth.
4. `shuffled-orthogonal`: the orthogonal case supplied in reverse temporal
   order.

`partitionCorrect` is an exact synthetic partition check that is invariant to
cluster-label permutation. It is **not DER**, must not be presented as DER, and
does not estimate performance on real speech. In the singleton case, the
singleton's assignment is ignored, while the persistent speakers must remain
one-to-one with the predicted clusters and the predicted count must equal the
persistent-speaker count.

No media is opened. No CAM++, ASR, VAD, diarization, or other model is loaded.
The wall time covers only the current Python clustering entry point; it excludes
audio decoding, model inference, media I/O, and PDF/report rendering.

## Run

From the repository root:

```powershell
conda run -n media-asr python -m benchmarks.speaker_scaling
```

The defaults exercise speaker counts `1,2,3,5,8,13,32,64,129`, three samples per
persistent speaker, all three count modes, all four scenarios, and one repeat.

```powershell
conda run -n media-asr python -m benchmarks.speaker_scaling `
  --speaker-counts 1,2,5,13,129 `
  --samples-per-speaker 4 `
  --modes manual,auto,hybrid `
  --repeat 3 `
  --output-json benchmarks/speaker_scaling/reports/local.benchmark.json
```

Successful stdout is exactly one strict JSON document. When `--output-json` is
used, the same bytes are written to the requested file. Use the ignored
`reports/` directory or the ignored `*.benchmark.json` suffix for local
artifacts.

The benchmark intentionally configures two spherical k-means refinement
iterations so high-cardinality scaling runs remain practical. The complete
configuration, backend module SHA-256, resolved algorithm method, Python
runtime, platform, and CPU metadata are included in every report. Results are
specific to the current machine, Python runtime, and backend source revision.

## Self-tests

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
conda run -n media-asr python -m pytest -p no:cacheprovider -q `
  benchmarks/speaker_scaling/tests
```
