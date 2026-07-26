# Report Appearance Customization Domain

## Status

`backend/report_styles.py` is the offline, presentation-only domain layer for
report appearance. Its public JSON contract is
`contracts/report-style.schema.json`.

The current contract version is `1.0.0`.

This layer is deliberately independent from transcription, semantic
arbitration, translation, summarization, HTML construction, PDF rendering, and
filesystem access. It can therefore be used by the native UI, job contract assembly,
OpenHTMLtoPDF renderer adapter, and artifact provenance code without granting
any of those callers authority to alter source transcript content.

## Non-negotiable invariants

Every valid effective configuration contains:

```json
{
  "sourceProtection": {
    "preserveOriginalTranscript": true,
    "mutatesOriginalTranscript": false,
    "presentationOnly": true
  }
}
```

Those values are constants in both Python validation and JSON Schema. The
domain accepts no transcript text and exposes no transcript mutation method.
Appearance changes are applied only to a derived report.

The implementation is:

- offline: no network access, remote asset resolution, subprocess execution,
  or model invocation;
- immutable: validated configurations are frozen dataclasses;
- reversible: presets remain unchanged, `effective_defaults()` returns a
  detached editable copy, and disabled safe logo choices may be retained;
- deterministic: canonical JSON and its SHA-256 digest are stable for the same
  effective configuration;
- fail-closed: unknown keys, missing keys, incorrect JSON types, false
  provenance claims, and unsafe paths are rejected.

## Presets

Six conservative production presets and one editable baseline are provided:

| Preset | Primary use |
| --- | --- |
| `modern-editorial` | Balanced general-purpose transcript report |
| `conversation-focus` | Speaker-forward interview and meeting reports |
| `executive-brief` | Restrained summary-led stakeholder delivery |
| `compact-review` | Dense review, QA, and timestamp inspection |
| `accessible-high-contrast` | Large-print, high-contrast reading |
| `archive-monochrome` | Stable monochrome archival presentation |
| `custom` | Safe editable baseline for user-authored combinations |

Presets are fully materialized effective configurations rather than opaque
renderer switches. A renderer can therefore reproduce the same result without
depending on hidden preset logic.

`preset` records the selected baseline. Partial overrides may change the
effective fields while retaining that baseline identifier for UI reset and
provenance. A caller cannot spoof a different preset inside the override
mapping.

## Configurable surface

The versioned configuration covers:

- layout template;
- primary font family;
- ordered Latin, CJK, and RTL font fallback declarations;
- density;
- cover style;
- speaker color mode;
- timestamp detail;
- section inclusion;
- page size and orientation;
- four physical page margins in millimeters;
- independent header and footer behavior;
- optional local logo intent;
- brand accent color;
- high-contrast mode.

Section inclusion supports cover, table of contents, metadata, speaker
directory, source transcript, translation, semantic review notes, summary,
quality appendix, and provenance. At least one substantive content section must
remain enabled.

Configuration does not contain raw CSS, raw HTML, JavaScript, renderer command
arguments, or arbitrary template paths. This keeps the native UI controls
bounded and prevents report customization from becoming an execution surface.

## Font fallback declarations

Font policy separates three ordered fallback groups:

- `latin`;
- `cjk`, covering Chinese, Japanese, and Korean renderer preferences;
- `rtl`, covering right-to-left script renderer preferences.

The configuration always includes:

```json
{
  "availability": "declared-not-verified",
  "embedding": "not-embedded"
}
```

These are truthfulness declarations, not optional status fields. A font name in
the configuration means only “try this family in this order.” It does **not**
mean that the family is installed, licensed for embedding, resolved by the
renderer, or embedded in the PDF.

The future OpenHTMLtoPDF adapter must separately:

1. resolve fonts from an approved local font registry;
2. record actual file identity and licensing policy;
3. embed only approved local font files;
4. use PDFBox to verify the fonts actually embedded in the resulting PDF;
5. report the verified result as renderer evidence, never by rewriting this
   appearance declaration.

## Logo security boundary

`logo.path` is an intent relative to a renderer-controlled local asset root.
The domain never opens the path.

Accepted file types are raster-only:

- PNG;
- JPEG/JPG;
- WebP.

The validator rejects:

- HTTP, HTTPS, `file:`, `data:`, and every other URI scheme;
- Windows drive-absolute paths;
- POSIX root-absolute paths;
- UNC and network paths;
- path traversal through `.` or `..` segments;
- doubled separators and empty path segments;
- percent escapes;
- query strings and fragments;
- control characters;
- Windows-dangerous segment endings;
- unsupported or active image formats such as SVG.

Both slash styles are accepted for a safe relative input. Canonical output uses
forward slashes so equivalent Windows and portable inputs hash identically.

Enabling a logo requires a safe path and non-empty alternative text. A disabled
logo may retain a safe path so the user can turn it back on without losing
their choice. Unsafe paths are rejected even while disabled.

Path validation does not replace renderer containment checks. Before reading an
asset, the renderer must resolve the canonical relative path under its approved
asset root, reject symlink or junction escapes, enforce a byte-size limit,
decode the image defensively, and record the resulting content hash.

## Python API

### List and inspect presets

```python
from backend.report_styles import (
    effective_defaults,
    get_report_style_preset,
    list_report_style_presets,
)

names = list_report_style_presets()
immutable = get_report_style_preset("modern-editorial")
editable = effective_defaults("modern-editorial")
```

### Resolve partial DIY choices

```python
from backend.report_styles import resolve_report_style

style = resolve_report_style(
    "custom",
    {
        "density": "relaxed",
        "brandAccent": "#2e90fa",
        "margins": {"leftMm": 22, "rightMm": 22},
        "logo": {
            "enabled": True,
            "path": "branding/customer-logo.png",
            "altText": "Customer logo"
        }
    },
)
```

Nested overrides are merged onto detached preset defaults. The caller's mapping
is not mutated. The resolved value is validated as a complete configuration.

### Validate a complete payload

```python
from backend.report_styles import validate_report_style

style = validate_report_style(payload)
```

Validation checks exact keys recursively. It rejects additional properties
rather than silently ignoring them.

### Canonicalization and provenance hash

```python
from backend.report_styles import (
    canonical_report_style_dict,
    deterministic_report_style_hash,
)

canonical = canonical_report_style_dict(style)
digest = deterministic_report_style_hash(style)
```

Canonicalization:

- emits all effective defaults;
- emits one stable camel-case JSON shape;
- normalizes Unicode strings to NFC;
- normalizes accepted logo separators to `/`;
- normalizes colors to uppercase `#RRGGBB`;
- normalizes numeric measurements to finite floating-point values;
- serializes UTF-8 JSON with sorted keys and no insignificant whitespace.

The hash is a lowercase 64-character SHA-256 hexadecimal digest of that
canonical JSON. It can be recorded in a job manifest, renderer request, PDF
metadata evidence, cache key, or QA report. It is not a digital signature.

## JSON Schema parity

`contracts/report-style.schema.json` uses JSON Schema Draft 2020-12 and models
the same complete effective configuration as `ReportStyleConfig.from_dict()`.
Both layers enforce:

- exact version;
- required fields and no additional properties;
- the same enum values;
- the same numeric ranges;
- script-complete non-empty font fallback arrays;
- truthful font availability and embedding declarations;
- at least one content section;
- safe local-relative logo paths and enabled-logo dependencies;
- immutable source transcript declarations.

The Python layer additionally performs canonical normalization after
validation. Consumers at process boundaries should validate the JSON Schema
before transport and validate again with the Python domain when constructing
the internal immutable value.

Any future field or enum change requires:

1. a new schema version when compatibility is not additive;
2. synchronized Python and JSON Schema changes;
3. migration code outside this pure domain;
4. preset regeneration;
5. parity tests for accepted and rejected payloads;
6. renderer support before the native UI exposes the option.

## Renderer integration contract

This module does not render a PDF. The intended downstream flow is:

```text
Native TypeScript DIY controls
  -> complete report-style JSON
  -> JSON Schema validation
  -> immutable Python ReportStyleConfig
  -> report document assembly without source-text edits
  -> escaped canonical XHTML and bounded CSS tokens
  -> OpenHTMLtoPDF rendering
  -> PDFBox inspection
  -> Design Pack and accessibility gates
  -> immutable artifact/provenance record
```

The renderer adapter should map enums to an allowlisted set of CSS classes and
page rules. It must never interpolate user strings into raw CSS, HTML,
JavaScript, URLs, or command arguments. Header/footer text and logo alternative
text must be escaped as document text.

The renderer must continue to treat the source transcript artifact as
immutable. Section inclusion selects which existing semantic suggestions and
derived business artifacts are presented; it does not create, rewrite,
translate, or summarize them.

## Testing

`tests/test_report_styles.py` validates:

- Draft 2020-12 schema correctness;
- all presets and unique deterministic hashes;
- immutability and detached defaults;
- nested override behavior and caller-input preservation;
- canonical Unicode, color, numeric, and path behavior;
- Python/Schema agreement across invalid enums, types, ranges, and claims;
- broad URL, absolute path, UNC, traversal, encoding, separator, and extension
  attacks;
- safe Chinese/Unicode relative logo paths;
- enabled and disabled logo behavior;
- script-complete font fallback declarations;
- section-content requirements;
- hard source-protection invariants;
- strict unknown/missing-field rejection.

The focused test command is:

```powershell
python -m pytest tests/test_report_styles.py -q `
  --basetemp=C:\Users\33516\Documents\Playground\mts-pytest-temp\report-styles `
  -p no:cacheprovider
```
