# Packaging Guide

This directory contains the Windows and macOS packaging toolchain for MediaTranscribeStudio.

## Contents

- `one_click_build.py`: main Windows-oriented release pipeline
- `build_macos.py`: macOS application packaging flow
- `dist_config.py`: release metadata, hosted artifact URLs, and branding defaults
- `bootstrap_installer*.py`: bootstrap installers / setup frontends
- `inno/*.iss`: Inno Setup definitions for Windows installers

## Before Building

1. Install Python packaging dependencies from the project root.
2. Install platform tooling such as `PyInstaller`, `Inno Setup`, `dmgbuild`, and code-signing tools if required.
3. Set real distribution values in environment variables or update `dist_config.py`.
4. Keep signing certificates, upload tokens, and hosted artifact URLs out of source control.

## Common Commands

```bash
python packaging/one_click_build.py
python packaging/build_macos.py
python packaging/upload_hf_dmg.py --repo-id your-org/your-repo --file dist/MediaTranscribeStudio-Full-macOS.zip
```

## Recommended Release Hygiene

- Use hosted URLs that you control instead of hardcoded personal endpoints.
- Leave `DEFAULT_PAYLOAD_URL` and `DEFAULT_MACOS_INSTALLER_URL` empty until a release is ready.
- Provide signing credentials via environment variables, never in committed files.
- Commit source, icons, scripts, and installer definitions only; do not commit generated installers.
