# Ultimate Pre-release Parity Gate

## Purpose

This document defines the release, legacy-removal, and `main` replacement gate for the Ultimate version of MediaTranscribeStudio. The gate is implemented by:

- `docs/refactor/ultimate-parity.json`: the machine-readable policy and gate manifest.
- `tools/check_ultimate_parity.py`: the fail-closed evaluator.
- `tests/test_ultimate_parity.py`: focused policy, evidence, and cutover tests.

The gate exists to prevent a feature-complete-looking build from replacing the existing product before business parity, real-media quality, offline behavior, packaging, rollback, and explicit change authorization have all been demonstrated.

## Fail-closed principles

1. Missing, stale, malformed, unbound, or unverifiable evidence fails the owning gate.
2. A gate passes only when its manifest state is `passed`, every repository check passes, every command check has a valid attestation, and every required evidence attestation passes.
3. Evidence is bound to the current Git `HEAD`. Evidence from another commit is rejected.
4. Evidence must be no more than 336 hours old.
5. Evidence must be stored outside the repository.
6. Artifact paths must be safe, relative paths below the external evidence root.
7. Artifact byte counts and lowercase SHA-256 digests must match the files on disk.
8. Real-media artifacts must use unique canonical roles, declared media types and
   formats, exact file extensions, strict file signatures, and role-specific
   schemas. Arbitrary `.bin` files cannot satisfy a canonical role.
9. Commands are executed as argument arrays with `shell=False`; command attestations are bound to a digest of the declared command, working directory, and timeout.
10. Passing tests never imply approval to delete legacy code or replace `main`.
11. Any malformed policy, missing Git state, unsafe evidence root, dirty working tree, failed gate, or missing approval blocks release.

## Scope and capability matrix

Every capability is required and maps bidirectionally to exactly one release gate.

| Capability | Gate | Required result |
| --- | --- | --- |
| Production transcription | `ULT-TRANSCRIPTION-001` | Stable transcription, timing, language, persistence, and failure behavior |
| Speaker diarization | `ULT-SPEAKER-001` | Accurate boundaries, identities, cardinality, sequence decoding, and semantic attribution |
| Local translation | `ULT-TRANSLATION-001` | Requested-language output without speaker or timing mutation |
| Local polishing | `ULT-POLISH-001` | Schema-bound semantic correction with invariant preservation |
| Local summary | `ULT-SUMMARY-001` | Evidence-bound summary with valid segment references |
| Export | `ULT-EXPORT-001` | Contract-valid report document, artifact manifest, and exports |
| Dynamic-N | `ULT-DYNAMIC-N-001` | Automatic, manual, and hybrid speaker cardinality without a fixed product maximum |
| Java PDF | `ULT-JAVA-PDF-001` | OpenHTMLtoPDF rendering plus PDFBox integrity and visual validation |
| React, TypeScript, and Tauri desktop | `ULT-DESKTOP-001` | Production desktop checks for web and Rust layers |
| Design Pack | `ULT-DESIGN-PACK-001` | Verified wide and narrow UI evidence plus all visual facets |
| Globalization | `ULT-GLOBALIZATION-001` | Normalized language identifiers, professional English public prose, and explicit model-pack coverage limits |
| Real MOV, automatic speakers | `ULT-REAL-MOV-AUTO-001` | Fresh automatic-cardinality evidence on the approved real media |
| Real MOV, manual five speakers | `ULT-REAL-MOV-MANUAL-5-001` | Independent five-speaker regression on the same source |
| Offline and security | `ULT-OFFLINE-SECURITY-001` | Offline operation and loopback-only local model providers |
| Packaging and rollback | `ULT-PACKAGING-ROLLBACK-001` | Install, upgrade, uninstall, recovery, and rollback evidence |
| Legacy cutover | `ULT-LEGACY-CUTOVER-001` | Complete gates plus explicit, separate approvals |

## Gate-state semantics

The manifest permits four states:

- `not_started`: required work or evidence has not begun.
- `in_progress`: implementation or validation exists but release evidence is incomplete.
- `blocked`: a known prerequisite, failure, or missing evidence prevents completion.
- `passed`: the declared implementation is ready to be evaluated against all repository, command, and evidence checks.

The state is not proof. A `passed` state with missing evidence still fails. A non-`passed` state always fails release even if unrelated checks happen to succeed.

No gate may be changed to `passed` merely because a unit test ran once. The owner must produce fresh, current-commit evidence for the complete gate contract.

## External evidence root

Set the evidence root with either:

```text
MTS_ULTIMATE_EVIDENCE_ROOT
```

or:

```text
--evidence-root <external-directory>
```

The resolved directory must be outside the repository. The checker rejects an evidence root equal to the repository or nested below it. The repository stores only policy and evidence references, not private source-media locations, transcript contents, generated customer documents, or release artifacts.

The approved real-media manifest records only this basename:

```text
2026-07-21 12-58-40.mov
```

It must never record the private absolute source path or transcript text.

## Evidence contracts

### Generic capability attestation

```json
{
  "schemaVersion": "1.0.0",
  "kind": "capability-attestation",
  "gateId": "ULT-TRANSCRIPTION-001",
  "status": "passed",
  "commitSha": "0123456789abcdef0123456789abcdef01234567",
  "generatedAt": "2026-07-22T08:00:00Z"
}
```

The `kind` and `gateId` must match the manifest entry. `commitSha` must equal the repository's current `HEAD`.

### Command attestation

When a declared command passes, `--execute --write-attestations` may write:

```json
{
  "schemaVersion": "1.0.0",
  "kind": "command-attestation",
  "gateId": "ULT-TRANSCRIPTION-001",
  "status": "passed",
  "commitSha": "0123456789abcdef0123456789abcdef01234567",
  "generatedAt": "2026-07-22T08:00:00Z",
  "checkId": "transcription-focused-pytest",
  "commandSha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "exitCode": 0
}
```

The checker calculates `commandSha256`. A changed argument, working directory, or timeout invalidates the old command attestation.

### Artifact evidence

An attestation that declares artifacts must use entries shaped as follows:

```json
{
  "role": "pdf",
  "path": "runs/example/report.pdf",
  "mediaType": "application/pdf",
  "format": "pdf",
  "bytes": 12345,
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

Requirements:

- `path` is relative to the external evidence root.
- The path cannot be absolute and cannot contain traversal.
- The file must exist.
- `bytes` must be a positive integer and must equal the file size.
- `sha256` must be lowercase and must equal the file digest.
- Duplicate artifact paths and duplicate canonical roles are rejected.
- A real-media artifact must use the exact extension, `mediaType`, and `format`
  declared for its canonical role.
- JSON artifacts must be strict UTF-8 objects. Duplicate keys, non-finite
  numbers, wrong schema versions, and missing required fields are rejected.
- PDF artifacts must begin with a PDF header and contain a terminal EOF marker.
- Contact sheets must have a PNG signature and a positive-width,
  positive-height IHDR.

### Real MOV evidence

The two real-media attestations are:

```text
real-media/auto.json
real-media/manual-5.json
```

Both must:

- use `kind` equal to `real-media-attestation`;
- refer to the approved basename;
- provide a valid lowercase `sourceSha256`;
- have `terminalState` equal to `completed`;
- persist a concrete language instead of `auto`;
- report positive and equal detected and resolved speaker counts;
- list contiguous IDs from `speaker-1` through `speaker-N`;
- provide all seven canonical artifacts with verified hashes, byte counts,
  formats, schemas, and file signatures;
- bind the transcript, report document, PDF inspection, quality report, and
  artifact manifest to the same job, document, and source identity;
- preserve identical segment IDs, timestamps, and speaker assignments between
  the transcript and report document;
- bind the PDFBox inspection to the exact SHA-256 digest of the attested PDF;
- bind the artifact manifest to the exact bytes and SHA-256 digest of every
  declared renderer output;
- pass speaker-count, diarization, boundary, ASR, semantic, efficiency, and PDF metric domains;
- provide the complete raw fields for every metric domain so the checker can
  recompute acceptance instead of trusting a declared status;
- pass every Java PDF hard gate in canonical order;
- pass every Design Pack facet in canonical order;
- meet the PDF quality score threshold;
- report no missing segments, timestamps, speaker IDs, font failures, blank pages, overflow findings, or remote assets.

Required real-media artifact roles are:

```text
transcript
report-document
pdf
pdf-inspection
quality-report
artifact-manifest
contact-sheet
```

Each metric domain must contain `status: "passed"` and the following concrete
measurements:

| Domain | Required raw fields |
| --- | --- |
| `speakerCount` | `expectedCount`, `detectedCount`, `resolvedCount`, `absoluteError`, `reviewRequired` |
| `diarization` | `segmentCount`, `auditedSegmentCount`, `speakerAssignmentErrors`, `unresolvedSpeakerSegments`, `humanAuditCompleted` |
| `boundary` | `segmentCount`, `invalidIntervals`, `nonMonotonicIntervals`, `outOfBoundsIntervals` |
| `asr` | `segmentCount`, `auditedSegmentCount`, `emptySegments`, `unresolvedTextSegments`, `sourceLanguagePreserved` |
| `semantic` | `reviewedRevisionCount`, `unresolvedRevisionCount`, `rawTranscriptImmutable`, `speakerLocksPreserved`, `humanApproved` |
| `efficiency` | `mediaDurationSeconds`, `wallClockSeconds`, `realTimeFactor`, `peakRamMb`, `peakVramMb` |
| `pdf` | `pageCount`, `qualityScore`, `hardGateFailureCount`, `facetFailureCount`, `pdfBoxValidated` |

The checker recomputes these values from the attestation and typed artifacts
where possible. It requires complete human audit, zero unresolved speaker,
timing, text, or semantic errors, immutable source evidence, preserved speaker
locks, successful PDFBox validation, and a real-time factor no greater than
`4.0`. A status-only object such as `{ "status": "passed" }` is invalid.

The automatic run must use `speakerCountMode` equal to `auto` and must not contain `manualSpeakerCount` or `requestedSpeakerCount` overrides.

The manual run must use `speakerCountMode` equal to `manual`; `requestedSpeakerCount`, `manualSpeakerCount`, `detectedSpeakerCount`, and `resolvedSpeakerCount` must all equal five.

The two runs must have the same `sourceSha256` and different `runId` values. Five speakers is a regression fixture, not a product maximum.

## Dynamic-N policy

The supported modes are exactly:

```text
auto
manual
hybrid
```

The mandatory cardinality regression matrix is:

```text
1, 2, 5, 8, 13
```

`fixedProductMaximum` is `null`. Product behavior must not assume five speakers or impose a fixed global speaker limit. Manual mode requires an explicit count. Automatic mode cannot carry a manual override. Hybrid mode may use bounded prior information but must preserve the final Dynamic-N invariants.

## Java PDF quality policy

The only approved rendering stack for this gate is:

- OpenHTMLtoPDF `1.0.10`
- PDFBox `2.0.30`
- Minimum quality score: `85`

All hard gates must pass in this exact order:

1. `PDF-OPENABLE`
2. `PDF-PAGE-COUNT`
3. `PDF-PAGE-SIZE`
4. `PDF-TRANSCRIPT-TEXT-INTEGRITY`
5. `PDF-SEGMENT-COUNT`
6. `PDF-TIMESTAMP-INTEGRITY`
7. `PDF-SPEAKER-SET-INTEGRITY`
8. `PDF-FONT-EMBEDDED`
9. `PDF-NO-BLANK-PAGES`
10. `PDF-NO-CONTENT-OVERFLOW`
11. `PDF-OFFLINE-ASSETS`
12. `PDF-PAGE-EVIDENCE`
13. `PDF-IMMUTABLE-CONTENT-HASH`

The following Design Pack facets are mandatory and ordered:

1. `AESTHETIC-COHERENCE`
2. `AESTHETIC-DISTINCTION`
3. `AESTHETIC-REFINEMENT`
4. `AESTHETIC-PROPORTION`
5. `AESTHETIC-HIERARCHY`
6. `AESTHETIC-TYPOGRAPHY`
7. `AESTHETIC-COLOR-RELATIONSHIPS`
8. `AESTHETIC-RHYTHM`
9. `AESTHETIC-DENSITY`
10. `AESTHETIC-RESTRAINT`
11. `AESTHETIC-REAL-CONTENT-STRESS`
12. `AESTHETIC-FONT-FAILURE`
13. `AESTHETIC-IMAGE-FAILURE`
14. `AESTHETIC-SCRIPT-FAILURE`

The PDF gate validates content integrity, provenance, and visual fitness.
`pdf-inspection` must name `PDFBox` version `2.0.30`, identify the same
document, and contain the exact SHA-256 digest of the inspected PDF. A visually
attractive PDF with missing transcript content or a mismatched inspection
fails. A content-complete PDF with overflow, missing fonts, broken assets,
unverified pages, or an unbound quality/manifest document also fails.

## Globalization policy

- A persisted language value cannot remain `auto`.
- Unknown language uses `und`.
- Multilingual content uses `mul`.
- Public README explanatory prose must use professional English for an
  international audience.
- Clearly identified non-English strings may be retained only when they are
  necessary regression fixtures, rendering examples, or source-language test
  evidence.
- Transcription, translation, polishing, summary, export, UI, and PDF behavior must not assume Chinese-only input.
- The real MOV regression may use a concrete Chinese language tag, but no product API may treat that fixture as the global default.
- Runtime language coverage depends on the installed and validated ASR,
  alignment, font, and local-LLM model packs. The project does not claim
  universal language support.

The current machine-readable manifest and `readmes_english` checker still
enforce a literal no-CJK rule. Therefore, an intentional non-English regression
fixture in a public README remains a known policy/checker mismatch and keeps
the globalization gate blocked until that machine policy is revised in a
separately authorized change. This documentation audit does not alter JSON
parity policy or checker code.

## Local small-model artifact policy

- Local small models may produce translation, source-language semantic-polish,
  and evidence-grounded summary artifacts only through versioned derived
  contracts.
- Derived artifacts must retain source-document identity, language metadata,
  model provenance, validation status, and evidence references where required.
- No model output may silently rewrite source transcript text, timestamps,
  speaker assignments, voiceprint evidence, or human locks.
- A human-approved source-language correction must be stored as an explicit,
  attributable revision rather than retroactively changing immutable source
  evidence.

## Offline and security policy

- Local LLM translation, polishing, and summary providers must be restricted to loopback endpoints.
- Malformed provider responses, remote endpoints, schema violations, non-finite values, and invalid evidence references must fail closed.
- Release evidence must demonstrate that required production workflows do not depend on remote assets or unapproved network services.
- Evidence and customer-derived artifacts remain outside the repository.

## Decision logic

### Parity eligibility

`parityEligible` is true only when all required gates except `ULT-LEGACY-CUTOVER-001` pass.

### Legacy-removal authorization

`legacyRemovalAllowed` requires:

1. `parityEligible` is true.
2. `policy.authorization.legacyRemoval.approved` is true.
3. `approvedBy`, `approvedAt`, and `changeTicket` are present and valid.

Until then, the following paths are protected:

```text
main.py
ui_app.py
mts_ui
pipeline.py
output_formatter.py
report_generator.py
packaging
```

The manifest records an approved baseline source commit and the exact Git
mode, type, and object ID for every protected root. The checker verifies all
of the following before legacy removal is authorized:

- the recorded baseline object identities are the objects stored at the
  baseline source commit;
- the baseline source commit is an ancestor of the actual current `HEAD`;
- every protected root at the actual current `HEAD` still has the approved
  blob or tree identity;
- every protected worktree file is read as raw bytes and compared with the
  approved baseline blob bytes without invoking `git hash-object --path` or
  any repository-configured clean filter;
- the guard's only content equivalence rule is LF/CRLF normalization for
  NUL-free, valid UTF-8 text; binary files require exact byte equality;
- `filter`, `working-tree-encoding`, `ident`, and other declared
  content-transforming Git attributes on protected files fail closed rather
  than participating in comparison;
- the protected filesystem tree has no missing or extra entries, empty-shell
  replacement, file/directory type replacement, or executable-mode change;
- no protected root or descendant is a symbolic link, junction, or other
  reparse point.

The guard obtains the actual current `HEAD` from Git and requires it to match
the `head_commit` snapshot supplied by the audit. A caller cannot bypass the
guard by passing a stale or invented commit. Renaming, deleting, modifying,
emptying, or replacing a protected path therefore fails closed even when the
path name itself still exists.

The audit does not treat `git status --porcelain` as a sufficient clean-tree
proof. Any tracked path carrying `assume-unchanged` or `skip-worktree` fails
the clean-tree gate. The release manifest must also be a regular tracked file
inside the repository, have no content-transforming Git attributes, and match
the raw bytes of its blob at the actual current `HEAD` (with only the same
explicit UTF-8 LF/CRLF equivalence rule). This prevents index flags from hiding
an uncommitted approval or gate-policy rewrite.

The manifest bytes are read once for parsing and passed unchanged into the Git
binding check. The checker compares both those exact decision bytes and the
current filesystem bytes with the committed manifest blob. Restoring or
swapping the file between parsing and the later repository inspection cannot
make a different in-memory approval payload pass the integrity gate.

### Baseline trust boundary

The baseline checks establish **internal repository consistency**, not an
external trust anchor. The manifest's `sourceCommit` and protected object IDs
are stored in the same repository as the checker. An actor who can rewrite the
manifest, its source history, and the checker coherently can create a different
internally consistent baseline.

The external evidence directory provides storage separation but is not, by
itself, a signed or independently trusted authorization source. This repository
currently contains no verified signature, protected CI attestation, transparency
log entry, or separately administered baseline digest that would make the
manifest tamper-evident against a repository administrator. Release governance
must supply and verify such an external anchor before treating the baseline as
independently authorized. Until then, the checker claims detection of internal
inconsistency and unauthorized local mutation only.

### Main replacement authorization

`mainReplacementAllowed` requires:

1. Every release gate, including `ULT-LEGACY-CUTOVER-001`, passes.
2. Legacy removal has explicit approval.
3. Main replacement has a separate explicit approval.

`approvedAt` values must be RFC3339 timestamps with an explicit `Z` or numeric
timezone. Invalid, timezone-free, and future timestamps fail closed. Main
replacement approval cannot precede legacy-removal approval:
`mainReplacement.approvedAt` must be greater than or equal to
`legacyRemoval.approvedAt`; equal timestamps are allowed. Until the
requirements pass, the protected local and remote `main` refs must remain at
the baselines recorded in the manifest.

The checker resolves every protected ref directly from Git and verifies that
the supplied ref snapshot matches the live value before comparing it with the
approved ref baseline. Legacy-removal authorization and main-replacement
authorization remain separate decisions; neither approval is inferred from
the other.

The public `legacyRemovalAllowed` and `mainReplacementAllowed` results are
**pre-execution permissions**. They are computed only after the relevant HEAD,
approved-baseline, worktree, protected-ref baseline, ref snapshot, and
authorization-order checks complete. Approvals participate in the final
conjunction but never disable any protection check. A valid-looking approval
can never leave either result `true` when a required integrity check fails or
after a protected path/ref has already changed.
The complete evaluator additionally gates both results on repository
integrity, including ordinary dirty-tree entries, hidden index flags, and the
decision-byte-bound manifest check.

### Release eligibility

`releaseEligible` requires all of the following:

- every release gate passes;
- both explicit approvals pass;
- the current `HEAD` is bound to the guard snapshot and descends from the
  approved legacy baseline;
- protected legacy Git/worktree identities and protected refs have no
  premature mutation;
- approval ordering is valid;
- the evidence root is external and safe;
- the working tree is clean.

Approvals are never inferred from tests, evidence, branch names, issue text, or a successful build.

## CLI

Run from the repository root:

```powershell
python tools/check_ultimate_parity.py --validate-manifest
```

Run the complete read-only audit:

```powershell
python tools/check_ultimate_parity.py --format json
```

Use an explicit external evidence root:

```powershell
python tools/check_ultimate_parity.py --evidence-root D:\secure-evidence\mts-ultimate --format json
```

Execute declared command checks without writing attestations:

```powershell
python tools/check_ultimate_parity.py --evidence-root D:\secure-evidence\mts-ultimate --execute --format json
```

Execute commands and write attestations for successful commands:

```powershell
python tools/check_ultimate_parity.py --evidence-root D:\secure-evidence\mts-ultimate --execute --write-attestations --format json
```

Filter displayed gate detail without changing full eligibility evaluation:

```powershell
python tools/check_ultimate_parity.py --gate ULT-JAVA-PDF-001 --format json
```

Exit codes:

- `0`: the manifest-only validation succeeded, or the complete release audit is eligible.
- `1`: the manifest is valid but the release is blocked.
- `2`: configuration, manifest, Git, or evidence-root safety is malformed.

## Release procedure

1. Freeze the candidate commit and record its Git SHA.
2. Confirm every required implementation and contract is present.
3. Run focused Python, Java, Node, browser, Rust, packaging, and security checks.
4. Run the Design Pack evaluation and repository-organization validation.
5. Run the approved real MOV twice: once in automatic mode and once with manual count five.
6. Store all attestations and artifacts under an external evidence root.
7. Verify artifact sizes and SHA-256 digests.
8. Set a gate state to `passed` only after its full evidence package is complete.
9. Run the checker without `--execute` to verify the saved evidence independently.
10. Require a clean working tree.
11. Record explicit legacy-removal approval with approver, timestamp, and change ticket.
12. Complete and attest the cutover gate.
13. Record a separate explicit `main` replacement approval.
14. Run the complete audit again.
15. Replace `main` only when `releaseEligible`, `legacyRemovalAllowed`, and `mainReplacementAllowed` are all true.

## Rollback procedure

1. Preserve the pre-cutover `main` SHA, release artifacts, configuration, data-migration plan, and installer versions.
2. Validate downgrade compatibility and data backup restoration before release.
3. Define objective rollback triggers for startup failure, worker failure, data corruption, severe diarization regression, PDF integrity failure, or packaging failure.
4. Stop distribution and restore the protected release ref when a rollback trigger fires.
5. Restore the last supported installer and worker bundle.
6. Restore or migrate persisted state according to the tested recovery plan.
7. Re-run smoke tests and artifact-integrity checks on the restored version.
8. Record the incident and invalidate evidence from the failed candidate.
9. Require a new commit-bound evidence set before attempting release again.

## Current verdict on July 22, 2026

The Ultimate release is blocked.

Current manifest states are:

- `in_progress`: transcription, speaker diarization, translation, polishing, summary, Dynamic-N, Java PDF, desktop, and offline/security.
- `blocked`: export, Design Pack, globalization, packaging/rollback, and legacy cutover.
- `not_started`: real MOV automatic mode and real MOV manual five-speaker mode.

Known blockers include:

- No external evidence root is configured for the default audit.
- No fresh current-commit attestation set exists for every required gate.
- The real MOV automatic and manual five-speaker evidence packages are not complete.
- The Design Pack runtime is prepared but its release evaluation is not verified as passing.
- The Design Pack source currently reports `implementationReady=false` and
  `releaseEligible=false`; no documentation statement overrides those values.
- The parity checker still treats any CJK character in a public README as a
  globalization failure. The intentionally retained Chinese renderer fixture
  therefore remains a documented checker-policy mismatch.
- Packaging, upgrade, uninstall, recovery, and rollback evidence is incomplete.
- The working tree is not clean.
- Legacy-removal approval is false.
- Main-replacement approval is false.

Therefore:

```text
parityEligible = false
legacyRemovalAllowed = false
mainReplacementAllowed = false
releaseEligible = false
```

Legacy deletion and `main` replacement remain prohibited. The checker must continue to fail closed until every required gate, evidence contract, guardrail, clean-tree requirement, and explicit approval passes.
