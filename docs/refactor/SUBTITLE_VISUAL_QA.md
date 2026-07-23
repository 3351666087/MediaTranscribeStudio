# Subtitle Representative-Frame Visual QA

## Status

This document defines the independent, offline subtitle visual-QA domain in:

```text
backend/subtitle_visual_qa.py
contracts/subtitle-visual-qa.schema.json
tests/test_subtitle_visual_qa.py
```

The contract version is `1.0.0`.

This layer is a **hard publication gate**. It consumes evidence created after
subtitle rendering and returns a deterministic pass/fail result. It does not
render media, modify subtitle text, inspect a GUI, download a font, query a
font service, invoke FFmpeg, open a browser, or use the network.

## Responsibility boundary

An upstream renderer or evidence collector is responsible for producing:

- SHA-256-bound render and render-configuration identity;
- deterministic representative-frame selection identity;
- rendered frame dimensions, timestamp, and image SHA-256;
- per-cue subtitle bounds and visible-ink bounds;
- renderer clipping, edge-touch, and overflow counters;
- worst-case rendered foreground/background pixel samples;
- renderer/font resolution and Unicode glyph-coverage evidence;
- cue text, rendered line breaks, duration, speaker, and style identity;
- speaker palette colors;
- authentic word-level timing evidence when word-progress karaoke is used.

The visual-QA domain treats filenames, UI state, CSS declarations, ASS style
names, and requested font family names as intent only. They are not evidence
that the rendered result is correct.

## Fail-closed principles

1. Missing expected representative frames fail.
2. Every cue in the supplied representative cue set must occur in at least one
   supplied frame.
3. Every visual cue instance requires geometry, clipping counters, font/glyph
   evidence, and sampled rendered pixels.
4. Missing evidence does not become a warning or an assumed pass.
5. All gate failures are stable machine-readable codes.
6. `passed` is true only when every named gate passes.
7. The result always declares `failClosed: true`.
8. The result binds to the canonical request using `inputSha256`.

Malformed or structurally ambiguous requests raise
`SubtitleVisualQAInputError`. Evidence that is structurally valid but
insufficient produces a normal failed result.

## Contract shape

One Draft 2020-12 JSON Schema validates both envelope kinds:

```text
subtitle-visual-qa-request
subtitle-visual-qa-result
```

They are unambiguous because `kind` is a constant in each branch.

The request contains:

```text
renderArtifact
policy
sampling
speakers
cues
frames
```

The result contains:

```text
passed
failClosed
inputSha256
gateOrder
gates
metrics
fontTruth
failureCodes
```

Unknown properties are rejected in both request and result contracts.

## Determinism

Canonical JSON uses:

- UTF-8;
- Unicode preserved rather than ASCII-escaped;
- lexicographically sorted object keys;
- compact separators;
- rejection of `NaN` and Infinity.

The result:

- processes gate names in a fixed order;
- sorts violations by code and stable evidence identity;
- sorts global failure codes;
- sorts background coverage;
- sorts speaker pairs by speaker ID;
- calculates metrics from request evidence rather than wall-clock state.

Mapping insertion order therefore does not change `inputSha256` or canonical
result JSON.

The deterministic boundary does not claim that two different renderers will
produce identical frame pixels. It guarantees that the same accepted evidence
produces the same QA result.

## Hard gates

### 1. Sampling evidence

The request supplies:

- a deterministic sampling strategy identifier;
- a selection-artifact SHA-256;
- the complete `expectedFrameIds` set.

The gate rejects:

- missing expected frames;
- unbound extra frames;
- duplicate cue instances in one frame;
- frame timestamps outside the parent cue interval;
- representative cues that never appear in a frame.

This layer verifies the supplied selection envelope. It does not pretend to
re-run or prove the upstream selection algorithm.

### 2. Safe area

Safe-area margins are explicit ratios of the rendered frame:

```text
leftRatio
rightRatio
topRatio
bottomRatio
```

The visible-ink rectangle, rather than only a layout container, must remain
inside the calculated safe rectangle. Pixel margins use deterministic ceiling
rounding.

The shipped default is five percent on every side.

### 3. Clipping and overflow

The gate combines geometry and renderer evidence. It rejects:

- subtitle bounds outside the frame;
- visible ink outside the frame;
- visible ink outside the declared subtitle bounds;
- visible ink geometrically touching a frame edge;
- non-zero clipped-pixel evidence;
- non-zero frame-edge-touching pixel evidence;
- renderer-reported overflow.

Geometry alone cannot prove that no glyph was clipped, so zero-valued renderer
counters remain required.

### 4. Contrast on dark and light backgrounds

Contrast is calculated locally from supplied `#RRGGBB` pixel samples using
WCAG relative luminance:

```text
(lighter + 0.05) / (darker + 0.05)
```

The request does not supply a trusted contrast score. It supplies the rendered
foreground/background colors and bounded pixel counts; the domain calculates
the score.

Each sample must:

- be marked as sampled from a rendered frame;
- bind its extraction evidence with SHA-256;
- contain enough foreground and background pixels;
- meet the configured minimum contrast ratio;
- have a `dark` or `light` classification consistent with measured background
  luminance.

Every subtitle `styleId` must have valid evidence for every configured required
background class. The default requires both `dark` and `light`. A style cannot
pass because another style was tested on the missing background class.

The sampled foreground should represent the worst-case effective glyph edge,
outline, or opaque subtitle-box text pixel selected by the upstream evidence
collector. A style with white fill over a light scene cannot report only an
unrelated black outline sample unless the collection method defines and binds
that sample as the critical rendered readability edge.

### 5. Font resolution and missing glyphs

Every rendered cue instance requires evidence for:

- requested family/fallback intent;
- actual resolved family;
- verified renderer font resolution;
- verification method;
- SHA-256-bound evidence artifact;
- expected renderable Unicode code-point count;
- covered renderable Unicode code-point count;
- missing code points;
- tofu/replacement glyph count.

The gate rejects:

- absent font evidence;
- unresolved or unverified font selection;
- missing evidence artifacts;
- unverified glyph coverage;
- expected counts that do not match the exact cue text;
- incomplete coverage;
- any listed missing code point;
- any tofu/replacement glyph.

Whitespace and Unicode control characters do not count as renderable code
points. Other Unicode code points, including punctuation and combining marks,
do count. The renderer evidence remains responsible for shaping and
font-fallback correctness.

## Font truthfulness

Requested fonts and successfully rendered glyphs do **not** prove installation
or embedding.

Installation status is one of:

```text
not-asserted
verified-installed
```

Embedding status is one of:

```text
not-asserted
verified-embedded
verified-not-embedded
```

Every positive `verified-*` claim requires:

- a non-placeholder verification method;
- an evidence-artifact SHA-256;
- a font-artifact SHA-256.

When status is `not-asserted`, all claim evidence fields must be empty and the
method must be `not-provided`. This prevents a requested family or renderer
fallback from silently becoming an installation/embedding claim.

The result's `fontTruth` section reports only:

- evidence instance count;
- positive claims that were actually evidence-backed;
- instances where no positive claim was made.

It does not infer installation or embedding from glyph coverage.

### 6. Line count and reading speed

The gate calculates:

- rendered line count;
- renderable code points per line;
- cue reading speed as non-whitespace, non-control Unicode code points per
  second;
- reconstruction of the cue's non-whitespace text from rendered lines.

It rejects excessive lines, excessive line length, excessive reading speed,
and rendered-line text that does not reconstruct the cue.

Line breaks may consume or replace whitespace, which is why reconstruction
ignores whitespace only. It does not ignore punctuation or other text.

### 7. Speaker color distinguishability

Speaker colors are evaluated pairwise with deterministic CIEDE2000 distance in
CIELAB space. Exact duplicate colors are valid input evidence but fail the
gate with a distance of zero.

The default minimum CIEDE2000 distance is `18.0`.

This gate measures palette separation. Per-frame subtitle/background
legibility remains the responsibility of the contrast gate. A palette must
pass both.

### 8. Visual overlap

For every pair of subtitle instances in the same representative frame, the
domain calculates visible-ink rectangle intersection area.

The default permitted overlap is zero pixels. Legitimate simultaneous speech
can still pass when its subtitles are stacked or otherwise spatially
separated. Source-audio overlap is not itself treated as a rendering defect.

### 9. Authentic karaoke timing

`word-progress` karaoke never passes without verified word-level timing.

Accepted true timing sources are:

```text
forced-aligner
native-word-timestamps
human-authored
```

The following are representable for forensic input but always fail:

```text
segment-interpolation
synthetic-even-split
```

Authentic word timing must:

- be marked verified;
- carry a SHA-256-bound evidence artifact;
- bind to the exact cue text SHA-256;
- contain at least one timed word;
- use positive durations;
- remain inside the parent cue;
- remain ordered and non-overlapping;
- reconstruct the cue's non-whitespace text exactly.

This prevents a segment-duration animation or evenly divided highlight from
being described as word-level karaoke.

## Example

```python
from backend.subtitle_visual_qa import (
    default_subtitle_visual_qa_policy,
    evaluate_subtitle_visual_qa,
)

request = {
    "kind": "subtitle-visual-qa-request",
    "schemaVersion": "1.0.0",
    "analysisId": "movie-42-visual-qa",
    "renderArtifact": {
        "artifactSha256": "a" * 64,
        "renderer": "ffmpeg-libass",
        "rendererVersion": "pinned-local-version",
        "renderConfigurationSha256": "b" * 64,
    },
    "policy": default_subtitle_visual_qa_policy(),
    "sampling": {
        "strategy": "cue-midpoint-plus-scene-extremes-v1",
        "selectionArtifactSha256": "c" * 64,
        "expectedFrameIds": ["frame-0001"],
    },
    "speakers": [
        {"speakerId": "speaker-1", "color": "#00A8E8"},
    ],
    "cues": [
        {
            "cueId": "cue-1",
            "startMs": 1000,
            "endMs": 4000,
            "text": "Hello",
            "speakerId": "speaker-1",
            "styleId": "youtube-clean",
            "renderedLines": ["Hello"],
            "karaokeMode": "none",
            "wordTimingEvidence": None,
        }
    ],
    "frames": [
        # Frame evidence omitted here for brevity. The JSON Schema contains
        # the complete geometry, font, and contrast sample contract.
    ],
}

result = evaluate_subtitle_visual_qa(request)
payload = result.to_dict()

if not result.passed:
    raise RuntimeError(payload["failureCodes"])
```

The abbreviated example intentionally does not pass because its expected frame
evidence is absent. A consumer cannot pass the gate by copying only the
configuration envelope.

## Integration sequence

Recommended local pipeline:

```text
subtitle composition
→ subtitle render or burn-in candidate
→ deterministic representative-frame selection
→ frame rendering
→ geometry/pixel/font/glyph evidence extraction
→ subtitle visual QA
→ publication allowed only when passed == true
```

Do not run this gate before rendering. Style declarations are insufficient.

Do not convert failures to warnings for final publication. A UI may explain a
failure and let the user choose a different style, font, placement, palette, or
line policy, but the corrected candidate must be rendered and evaluated again.

## Testing

Run the isolated contract/domain suite:

```powershell
python -m pytest `
  tests\test_subtitle_visual_qa.py `
  -q `
  --basetemp=C:\Users\33516\Documents\Playground\mts-pytest-temp\subtitle-visual-qa `
  -p no:cacheprovider
```

Coverage includes:

- request and result JSON Schema validation;
- canonical hash and output determinism;
- representative-frame completeness;
- safe area;
- every clipping/overflow signal;
- rendered pixel provenance, sample size, luminance class, and contrast;
- dark/light coverage per style;
- missing, unverified, incomplete, tofu, and mismatched glyph evidence;
- evidence-backed versus unasserted font installation/embedding truth;
- line count, line length, text reconstruction, and reading speed;
- pairwise speaker color separation;
- visual overlap;
- valid, absent, synthetic, unverified, out-of-range, overlapping, hash-mismatched,
  and text-mismatched karaoke timing;
- malformed and additional-property rejection;
- stable failure ordering.

## Non-claims

Passing this domain means that the supplied, SHA-256-bound representative-frame
evidence passed the declared gates. It does not claim:

- that every video frame was inspected;
- that the upstream frame selector is unbiased;
- that a font is installed or embedded without explicit claim evidence;
- that OCR proved transcript correctness;
- that a font license permits redistribution;
- that the media source was unchanged;
- that subtitle delivery or muxing succeeded;
- that the full video passed audio, encoding, or container QA.

Those claims belong to their own evidence domains and publication gates.
