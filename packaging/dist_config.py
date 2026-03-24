from __future__ import annotations

import os
from pathlib import Path


# User-facing product name (folder name and default executable name).
APP_NAME = "MediaTranscribeStudio"

# Main GUI exe generated from the application PyInstaller build.
MAIN_EXE_NAME = f"{APP_NAME}.exe"

# Small installer/downloader stub exe.
BOOTSTRAPPER_EXE_NAME = "MediaTranscribeStudio-Setup.exe"

# Uninstaller shipped inside the payload archive.
UNINSTALL_EXE_NAME = "uninstall.exe"

# Default install directory (per-user, no admin required).
DEFAULT_INSTALL_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME

# Default online installer source.
# New default architecture uses a lightweight Qt frontend that downloads the
# Inno Core installer (and optional split .bin files) online.
DEFAULT_PAYLOAD_URL = ""

# Optional explicit split .bin URLs for the Inno Core bundle (exe + bins = 4 links total).
# Leave empty to let the Qt frontend infer and probe `-1.bin`, `-2.bin`, ... automatically.
DEFAULT_INNO_CORE_BIN_URLS = []

# Optional integrity check. Leave empty to skip.
DEFAULT_PAYLOAD_SHA256 = ""

# Default hosted full payload for macOS. The lightweight first-run launcher
# downloads this archive from Hugging Face on first launch, then replaces
# itself with the real app bundle. Unsigned builds can safely use `.zip`.
DEFAULT_MACOS_INSTALLER_URL = ""

# Optional author / publisher line shown on branded installer surfaces.
APP_AUTHOR = os.environ.get("APP_AUTHOR") or os.environ.get("APP_PUBLISHER") or "MediaTranscribeStudio Team"

# Optional integrity check for the hosted macOS full-app payload archive.
DEFAULT_MACOS_INSTALLER_SHA256 = ""

# Internal macOS bootstrapper bundle name. The final DMG renames it back to
# `APP_NAME.app` so the user only sees the real product name.
MACOS_BOOTSTRAPPER_APP_NAME = f"{APP_NAME}-Setup.app"

# Payload archive file name downloaded by the bootstrapper.
PAYLOAD_ARCHIVE_NAME = f"{APP_NAME}-payload.tar.zst"

# Optional macOS icon lookup candidates (first existing file wins).
MACOS_ICON_CANDIDATES = [
    Path("assets") / "app.icns",
    Path("app.icns"),
]

# Optional icon lookup candidates (first existing file wins).
ICON_CANDIDATES = [
    Path("assets") / "app.ico",
    Path("assets") / "installer.ico",
    Path("app.ico"),
]

# Whether the main app spec should include local checkpoints/ by default.
INCLUDE_CHECKPOINTS_DEFAULT = True
