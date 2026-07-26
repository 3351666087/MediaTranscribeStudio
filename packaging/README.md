# Legacy Packaging Toolchain

This directory contains the Python, PyInstaller, Inno Setup, and macOS
packaging infrastructure from the legacy MediaTranscribeStudio application.

> **Migration status:** these scripts are retained for compatibility and
> investigation during the desktop refactor. They are not the verified final
> React, TypeScript, and Tauri release pipeline, and their presence must not be
> interpreted as release approval.

## Contents

| Path | Purpose |
|---|---|
| `one_click_build.py` | Legacy Windows-oriented build orchestration. |
| `build_macos.py` | Legacy macOS application packaging flow. |
| `dist_config.py` | Distribution metadata, branding defaults, and optional hosted-payload configuration. |
| `bootstrap_installer.py` | Python bootstrap installer flow. |
| `bootstrap_installer_qt.py` | Qt bootstrap installer frontend. |
| `bootstrap_installer_macos.py` | macOS bootstrap installer flow. |
| `bootstrap_macos.py` | macOS bootstrap support. |
| `build_*.spec` | PyInstaller specifications for legacy application, bootstrapper, and uninstaller targets. |
| `inno/*.iss` | Inno Setup definitions for Windows installers. |
| `upload_hf_dmg.py` | Utility for uploading a macOS distribution artifact to a configured Hugging Face repository. |

Additional helper, icon-generation, and uninstaller scripts support those
legacy flows.

## Important boundaries

- These tools package the legacy Python application; they do not validate the
  intended Tauri desktop architecture.
- Some flows support hosted bootstrap payloads or uploads and therefore may use
  network access. Do not describe this directory as an offline-only packaging
  system.
- Build scripts may inspect, stage, or bundle large local model caches and
  runtime payloads. Review resolved paths and artifact contents before running
  a build.
- Language coverage is determined by the model packs intentionally included in
  or installed alongside a build. Packaging a runtime does not establish
  universal language support or validate every language-model combination.
- Empty default hosted URLs in `dist_config.py` are placeholders, not release
  endpoints.
- A successful installer build does not prove application parity, model
  availability, PDF correctness, signing, update behavior, rollback behavior,
  or release readiness.

Local meeting-content processing and local LLM business processing are separate
runtime privacy boundaries. Legacy distribution utilities must never upload
meeting media, source transcripts, speaker evidence, semantic-arbitration
artifacts, derived translation or summary artifacts, or generated reports.

## Historical commands

Run these commands only when maintaining or evaluating the legacy packaging
path:

```powershell
python packaging/one_click_build.py
python packaging/build_macos.py
python packaging/upload_hf_dmg.py `
  --repo-id your-organization/your-repository `
  --file dist/MediaTranscribeStudio-Full-macOS.zip
```

Exact dependencies vary by target and may include PyInstaller, Inno Setup,
`dmgbuild`, platform SDKs, and signing tools. Inspect the scripts and resolved
configuration before execution.

## Safety and release hygiene

1. Keep signing certificates, private keys, access tokens, and hosted URLs out
   of source control.
2. Supply credentials through an approved secret-management mechanism.
3. Use organization-controlled distribution locations rather than personal
   endpoints.
4. Keep generated installers, archives, build directories, model caches, and
   bundled runtimes out of Git.
5. Verify that staged payloads contain no private media, transcript data,
   speaker embeddings, acoustic evidence, or local configuration secrets.
6. Produce checksums and retain reproducible build metadata for every artifact
   under evaluation.
7. Treat code signing, notarization, installer testing, rollback, and update
   validation as separate release gates.

## Migration guidance

New release work should target the active desktop architecture under
`apps/desktop/` rather than extending this legacy toolchain by default. Remove
or archive legacy packaging only after the replacement path has passed its
documented parity, installation, upgrade, rollback, offline-runtime, and
artifact-integrity gates.
