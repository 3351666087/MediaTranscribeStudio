# Native Linux Tauri Packaging

`packaging/tauri/build_tauri_linux_release.py` is the locked Linux release
entry point for the Tauri desktop application. It builds and collects AppImage,
deb, and rpm artifacts without changing the separate Windows install, upgrade,
rollback, and Authenticode contract.

The entry point requires Python 3.11 or newer (it uses the standard-library
`tomllib` parser) and is intended to run on a Linux build host for compilation.

## Build contract

Before compiling, the script requires and hashes both dependency lock files:

```text
apps/desktop/package-lock.json
apps/desktop/src-tauri/Cargo.lock
```

It also requires the application version to match in all three sources:

```text
apps/desktop/package.json
apps/desktop/src-tauri/tauri.conf.json
apps/desktop/src-tauri/Cargo.toml
```

Supported target triples are:

| Architecture | Rust target triple |
| --- | --- |
| x64 | `x86_64-unknown-linux-gnu` |
| arm64 | `aarch64-unknown-linux-gnu` |

`--target-triple` is canonical; `--architecture x64|arm64` is a convenience
alias for callers shared with the Windows packaging entry point. Supplying both
options requires them to agree.

The default bundle set is `appimage,deb,rpm`. Cross-compilation still requires
the corresponding Rust target, linker, and system packaging dependencies on the
build host; selecting a target triple does not install those prerequisites.

## Inspect without writing

Dry-run validates versions, lock files, channel, target triple, and resolved
paths. It does not run npm, create the Cargo target directory, acquire the build
lock, or create the release output:

```bash
python3 packaging/tauri/build_tauri_linux_release.py \
  --target-triple x86_64-unknown-linux-gnu \
  --source-date-epoch 1784678400 \
  --output-directory /mnt/d/releases/MediaTranscribeStudio/0.1.0-linux-x64 \
  --dry-run
```

The JSON plan includes the exact locked commands. The compile step uses
`npm ci`, Tauri's non-interactive `--ci` mode, an explicit Rust target, the
selected Tauri bundles, and Cargo's `--locked` mode. Both dependency lock
hashes are checked again after compilation and before artifacts are collected.

## Create a release

Install the Linux WebKit/GTK/AppImage/rpm/deb prerequisites required by Tauri,
then run:

```bash
python3 packaging/tauri/build_tauri_linux_release.py \
  --target-triple x86_64-unknown-linux-gnu \
  --source-date-epoch 1784678400 \
  --output-directory /mnt/d/releases/MediaTranscribeStudio/0.1.0-linux-x64
```

Only one normal build may use a Cargo target directory at a time. A non-blocking
advisory lock is held at `target/.mts-tauri-linux-build.lock` for compilation
and collection. For a CI job that already built the requested bundles, pass
`--skip-compile` and, when necessary, `--target-directory`; this mode assumes
the prebuilt target tree is no longer being written by another process.

The output is immutable for a single invocation: the command refuses to write
over an existing output directory. It stages and verifies every copied artifact
before atomically publishing the directory.

```text
release/
├── linux-release-manifest.json
├── linux-release-manifest.json.sha256
└── artifacts/
    ├── appimage/*.AppImage
    ├── deb/*.deb
    └── rpm/*.rpm
```

The `mts-tauri-linux-release/v1` manifest records the version, channel, source
date epoch, Git commit, target triple, exact build commands, dependency-lock
hashes, and a byte size/SHA-256 ledger for every artifact. Validate it against
`packaging/tauri/linux-release-manifest.schema.json` before publishing. The
plain SHA-256 ledger detects corruption but is not a substitute for signing the
release in the distribution channel.

## Embedded runtime and startup

The build stages a weight-free runtime under the Tauri Linux resource directory:

```text
usr/lib/MediaTranscribe Studio/mts-runtime/
├── backend/
├── contracts/
├── reporting/
├── configs/
├── bootstrap/
└── pdf-renderer/target/pdf-renderer.jar
```

Model weights, Python environments, media, local production configuration, and
credentials are excluded. `Initialize-MtsRuntime.sh` creates the operator-owned
configuration and model directories; set `MTS_MODEL_ROOT` when models belong on
another volume. The Rust worker receives Tauri's `resource_dir()` so AppImage,
deb, and rpm layouts resolve the same `mts-runtime` root. It also removes
AppImage's injected `PYTHONHOME`/`PYTHONPATH` before launching an external
Python interpreter; the AppImage wrapper's minimal `/usr` tree must not be used
as that interpreter's standard library.

The collector extracts each requested package and verifies one runtime marker,
the exact source allowlist, forbidden model/media extensions, and the standard
`usr/lib/<package>/mts-runtime` placement before writing the manifest. The
current x86_64 candidate verified AppImage, deb, and rpm independently; all
three contain the same 100-file, 37,371,547-byte runtime payload.

## Current x86_64 evidence

The current locked candidate is under
`D:\mts-release\linux-audit-r4\release`. Its manifest SHA-256 is
`c48666f3054255205212d6d961a308e856d2f4e8f7027abcf9109371b72042a1`.
The artifact ledger is:

| Bundle | Bytes | SHA-256 |
| --- | ---: | --- |
| AppImage | 127,404,536 | `2539c27f487235260f893edcc8020a3c8911a2c5a6065dcd6df655e5f32dffc5` |
| deb | 53,028,866 | `a252bf906ba2cb01cfb2ebd3ed0d4257a85ea65c48eb292460da5cf7c408fd83` |
| rpm | 53,036,915 | `c840fdd35099dfe33f97a114be58ac8268d154be710f77fa25a1f1d5f060c7eb` |

Schema validation, manifest checksum, artifact size/hash, target architecture,
package metadata, and byte-for-byte runtime verification all passed. AppImage,
deb, and rpm extraction layouts each launched a real 1440x920 Tauri window
under WSLg. The embedded initializer also created a clean Linux user config
that `ProductionConfig.load()` accepted, with pyannote and overlap recovery
disabled when their isolated runtime is absent. Packaging tests report
`9 passed`; the locked npm tree reports zero vulnerabilities.

This is package and bootstrap evidence, not an end-to-end transcription claim.
The candidate deliberately does not bundle Python/PyTorch, FFmpeg, Java, model
weights, or a completed first-run dependency installer. Native package-manager
install/upgrade/uninstall/rollback and a native arm64 build remain release
gates.

The manual CI entry point is `.github/workflows/tauri-linux-release.yml`. It
maps x64 to `ubuntu-24.04` and arm64 to the native `ubuntu-24.04-arm` runner,
installs the native Linux GUI/package prerequisites, and builds the locked PDF
sidecar. It does not pretend that an x64 host plus a cross GCC provides arm64
WebKitGTK/GTK libraries.

Before Actions upload, the workflow archives the complete verified release as
tar+gzip, writes its SHA-256 companion, extracts it into a clean directory, and
rechecks the manifest/artifact ledger plus AppImage executable mode. Only the
tar and checksum are passed to `actions/upload-artifact`, whose own transport
does not preserve Unix executable bits. The workflow uploads a candidate only;
it does not claim a signed distribution release or a full desktop acceptance
run.
