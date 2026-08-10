# Native macOS Packaging

The active macOS release path is
`packaging/tauri/build_tauri_macos_release.py`. It packages the React/Tauri
desktop application and deliberately bypasses the legacy PyInstaller flow in
`packaging/build_macos.py`, which remains available only for historical
maintenance.

## What the release contains

The Tauri build receives a temporary configuration overlay. The overlay maps a
weight-free runtime payload to:

```text
MediaTranscribe Studio.app/
└── Contents/Resources/mts-runtime/
    ├── backend/
    ├── contracts/
    ├── reporting/
    ├── configs/                 # provider and portable model catalogs
    ├── bootstrap/               # Initialize-MtsRuntime.sh, Manage-MtsModels.sh
    ├── tools/pyannote_runtime.py
    ├── production.config.*.json
    └── pdf-renderer/target/pdf-renderer.jar
```

The payload is an explicit source allowlist. Python environments, model
weights, media, local registries, production configs, API keys, and cache
directories are excluded. `runtime-bootstrap.v1.json` records
`bundledModelArtifacts: false`; operators choose a model root and download
models after installation.

This is currently a **native shell candidate**, not a self-contained ML
runtime. It does not bundle Python, Python wheels, FFmpeg/FFprobe, a Java
runtime, Ollama, or model weights. In particular,
`requirements-media-asr.txt` is retained only as build-environment evidence;
it is a Windows conda export and is not a macOS installer specification.
Shipping a usable offline transcription product still requires separate
`osx-arm64` and `osx-64` runtime payloads (or equivalent per-architecture
installers). A universal2 GUI cannot make architecture-specific PyTorch and
native Python wheels universal by itself.

## Architecture-specific runtime path

The shortest supported path is to keep the GUI universal2 and select a runtime
pack from the native `uname -m` value. The current arm64 pack can be built from
these upstream artifacts:

- [python-build-standalone 20260807](https://github.com/astral-sh/python-build-standalone/releases/tag/20260807), using the stripped CPython 3.12.13 `aarch64-apple-darwin` install-only archive;
- [PyTorch 2.10.0](https://pypi.org/pypi/torch/2.10.0/json) and [torchaudio 2.10.0](https://pypi.org/pypi/torchaudio/2.10.0/json), whose CPython 3.12 macOS wheels are published for arm64;
- [Temurin 17.0.20+8](https://github.com/adoptium/temurin17-binaries/releases/tag/jdk-17.0.20%2B8), reduced with `jlink` when the renderer module set is fixed;
- [FFmpeg 9.0 source](https://ffmpeg.org/releases/ffmpeg-9.0.tar.xz), compiled on the Apple runner with a recorded minimal LGPL configuration. GPL components and `--enable-nonfree` must stay disabled unless the distribution license is deliberately changed.

Intel cannot currently claim equivalent support from the same official wheel
set. PyPI publishes no CPython 3.12 macOS x86_64 wheel for Torch 2.10.0; the
latest x86_64 entry in the official Torch release metadata is 2.2.2, while
[pyannote.audio 4.0.4](https://pypi.org/pypi/pyannote.audio/4.0.4/json)
requires Torch and torchaudio 2.8 or newer. Therefore an Intel build must use a
separately frozen legacy/degraded feature set, omit pyannote coherently, or wait
for a maintained native Torch build. It must not silently reuse the arm64 pack
or advertise feature parity.

Runtime packs must publish their own file ledger, upstream URL/revision,
license notices, architecture, minimum macOS version, and SHA-256. Models remain
outside the application and runtime archives and are selected or downloaded to
the user-owned model root after integrity verification.

The release directory contains a machine-readable manifest and checksum plus
the requested artifacts:

```text
macos-release-manifest.json
macos-release-manifest.json.sha256
artifacts/app/*.app/       # optional direct bundle
artifacts/zip/*.app.zip    # optional deterministic development archive
artifacts/dmg/*.dmg        # optional native installer image
```

The default development selection is `app,dmg`. A ZIP can still be requested
for cross-host and unsigned fixture testing, but it is not accepted on the
`stable` channel yet. Python's standard ZIP writer does not preserve all
macOS extended attributes and resource forks. A stable portable archive must
first move to Apple's `ditto` archive/extract path and pass native stapler,
codesign, and Gatekeeper checks after extraction.

Every regular file and symlink target in an app bundle is hashed. The manifest
contract is `mts-tauri-macos-release/v1`; its schema is
`packaging/tauri/macos-release-manifest.schema.json`.

## Plan without a Mac

Dry-run validates the three application versions, `package-lock.json`,
`Cargo.lock`, target architecture, source allowlist, and release commands. It
does not create a target, staging, output, or lock file:

```bash
python packaging/tauri/build_tauri_macos_release.py \
  --target-triple universal-apple-darwin \
  --bundles app,dmg,zip \
  --allow-unsigned-development \
  --source-date-epoch 1784678400 \
  --dry-run
```

To retain the plan on a non-Apple host, opt in to an explicit output path;
the command writes only the JSON plan and its checksum, never a fake native
binary:

```bash
python packaging/tauri/build_tauri_macos_release.py \
  --target-triple universal-apple-darwin \
  --bundles app,dmg \
  --channel development \
  --allow-unsigned-development \
  --dry-run \
  --plan-output /mnt/d/mts-release/macos/macos-release-plan-universal2.json
```

Valid targets are `x86_64-apple-darwin`, `aarch64-apple-darwin`, and
`universal-apple-darwin` (the latter requires both Rust targets and Apple
universal linking support). Compilation is rejected on non-macOS hosts rather
than pretending that a Linux build is a macOS binary.

Cross-host plans are commit-specific evidence. Regenerate the plan immediately
before dispatch and retain it with its checksum; a saved plan must not be called
current after any allowlisted runtime input or lock file changes. The audited
2026-08-10 development plan is
`D:\mts-release\macos\macos-release-plan-universal2-r7.json`; it records a
100-file, 37,371,072-byte, weight-free runtime payload and has SHA-256
`1b67594982971582f8f1bc57ab468b0eb1d253c23a1d18633fe7c87d6154eb34`.
It is still only a build plan, not a Mach-O, app bundle, DMG, signing,
notarization, transport, or launch result.

## Build and collect

On a macOS builder, a stable distributable release requires a Developer ID
Application identity plus App Store Connect API credentials:

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Example Corp (TEAMID)"
export APPLE_API_KEY="KEYID"
export APPLE_API_ISSUER="00000000-0000-0000-0000-000000000000"
export APPLE_API_KEY_PATH="/secure/path/AuthKey_KEYID.p8"
python packaging/tauri/build_tauri_macos_release.py \
  --target-triple universal-apple-darwin \
  --bundles app,dmg \
  --channel stable \
  --require-notarization \
  --output-directory "$HOME/Downloads/mts-release/0.1.0-macos-universal2" \
  --source-date-epoch "$(git show -s --format=%ct HEAD)"
```

The script runs `npm ci` and a locked Tauri build, injects the runtime overlay,
checks that lock inputs did not change, parses `Info.plist`, and verifies the
main Mach-O architecture. Universal2 is accepted only when both `arm64` and
`x86_64` slices are present. The embedded runtime must match the source
allowlist byte-for-byte; unexpected, changed, secret-like, model, or media
files fail collection.

`signed` is a candidate-only mode: it verifies the exact signing authority but
records notarization as `not-claimed`. `signed-notarized` lets Tauri
notarize/staple the app, explicitly submits the DMG with `notarytool`, staples
it, then runs stapler and Gatekeeper assessments on both the app and DMG before
the manifest may record `stapled`. The `stable` channel rejects unsigned,
signed-but-not-notarized, non-Developer-ID, and portable ZIP combinations.

For an already-built app (for example, an artifact produced on a dedicated
Mac builder), use `--skip-compile`. A prebuilt app without the runtime overlay
can only be repaired in an explicitly unsigned development collection:

```bash
python packaging/tauri/build_tauri_macos_release.py \
  --target-triple arm64-apple-darwin \
  --skip-compile \
  --allow-unsigned-development \
  --bundles app,zip
```

The repair is made in a staging copy; the source `.app` is never modified.
Published releases must not use `--allow-unsigned-development`.

Verify a collected candidate independently:

```bash
python packaging/tauri/verify_tauri_macos_release.py \
  --release-directory /path/to/release \
  --expected-target universal-apple-darwin \
  --require-native \
  --run-bootstrap \
  --launch-smoke-seconds 8 \
  --launch-via-open
```

The verifier rehashes the exact manifest ledger, safely extracts ZIP, mounts
DMG read-only, validates each embedded app and runtime, runs the initializer in
a temporary user directory, parses the resulting production configuration,
and requires the GUI process to stay alive for the requested smoke interval.
The Apple workflow performs that smoke through `/usr/bin/open`, then locates
the exact bundle executable process and terminates only the PID it launched;
this exercises LaunchServices/Finder semantics instead of merely invoking the
Mach-O directly.

For a stable release, the manifest's self-reported certificate name is not a
trust anchor. Pass the independently obtained Developer ID identity and perform
the checks on macOS; the verifier rejects stable artifacts without both:

```bash
python packaging/tauri/verify_tauri_macos_release.py \
  --release-directory /path/to/stable-release \
  --expected-target universal-apple-darwin \
  --expected-signing-identity "Developer ID Application: Example Corp (TEAMID)" \
  --require-native
```

The Apple workflow supplies this external identity to the original candidate,
the pre-upload tar roundtrip, and the independently downloaded transport
verification. Unsigned development jobs receive an empty identity value.

## First run and models

After copying the app to `/Applications`, initialize user-owned paths from the
embedded runtime:

```bash
Contents/Resources/mts-runtime/bootstrap/Initialize-MtsRuntime.sh
```

The script resolves its runtime root from its own staged location, defaults
data to `~/Library/Application Support/MediaTranscribeStudio`, binds the
Windows-oriented example paths to macOS data/model directories, and never
overwrites an existing operator config. Set `MTS_MODEL_ROOT` to place models
on another volume. It records the resolved interpreter in the user-owned
`config/worker-python.path` hint so Finder launches do not depend on a terminal
`PATH`; `MTS_WORKER_PYTHON` still overrides that hint. Model downloads use the
catalog/model manager and read credentials only from environment variables.

When no isolated pyannote interpreter is supplied, initialization removes the
Windows `.exe` hint, disables the pyannote fallback coherently, and clears its
model path. On macOS it also replaces the Windows CUDA defaults with the
portable CPU/float32 profile. This makes the generated config structurally
valid and fail closed on genuinely missing runtime/model dependencies; it does
not claim that those dependencies are bundled.

The Tauri worker also discovers `Contents/Resources/mts-runtime` directly; set
`MTS_RUNTIME_ROOT` only when running a custom external payload.

## Release gates

This packaging result proves artifact integrity and runtime placement. It does
not, by itself, prove Apple notarization, model availability, diarization DER,
semantic quality, or end-to-end PDF acceptance. Those remain separate evidence
gates in `docs/refactor/TASKS.md`.

## Apple runner

This Linux/Windows checkout cannot emit a valid Mach-O executable. The manual
workflow `.github/workflows/tauri-macos-release.yml` is the reproducible native
entrypoint: official Actions are pinned to immutable commit SHAs, Node is fixed
at 22.22.2, Rust is fixed at 1.97.1, and Python is constrained to 3.12. It
installs both Apple Rust targets, builds the locked Java sidecar, runs the
packaging tests and cross-host plan, then runs the independent native verifier.
It maps `x86_64-apple-darwin` to the
native `macos-15-intel` runner and maps arm64 and universal2 to the Apple
Silicon `macos-15` runner; the retiring `macos-14` label is not used.
Choose `unsigned-development` only for local QA. A signed candidate requires the
`MACOS_CODESIGN_IDENTITY`, `MACOS_CERTIFICATE_P12_BASE64`,
`MACOS_CERTIFICATE_PASSWORD`, and `MACOS_KEYCHAIN_PASSWORD` repository secrets;
the identity is deliberately absent from unsigned jobs even when the secret is
configured. `signed-notarized` additionally requires `MACOS_NOTARY_KEY_ID`,
`MACOS_NOTARY_ISSUER_ID`, and `MACOS_NOTARY_PRIVATE_KEY_BASE64`. The private
`.p8`, imported `.p12`, and temporary signing keychain are removed in an
`always()` cleanup step. Before upload, the verified release directory is
packed as tar+gzip to preserve Unix modes and app-bundle symlinks, hashed,
safely extracted, and passed through the native verifier again. Only that tar
and its SHA-256 companion enter the Actions artifact transport. A dependent job
on the same target-native runner downloads those two files through
`actions/download-artifact`, checks the external checksum, rejects absolute,
traversing, duplicate, escaping-link, special-file, and symlink-parent archive
entries before extraction, and reruns native app/DMG trust checks, bootstrap,
and the LaunchServices smoke. The workflow does not create or publish a GitHub
Release.
