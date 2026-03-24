# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from pathlib import Path


_specpath = Path(SPECPATH).resolve()
SPEC_DIR = _specpath if _specpath.is_dir() else _specpath.parent
SRC_ROOT = SPEC_DIR.parent
if str(SPEC_DIR) not in sys.path:
    sys.path.insert(0, str(SPEC_DIR))

from dist_config import ICON_CANDIDATES, MACOS_BOOTSTRAPPER_APP_NAME, MACOS_ICON_CANDIDATES


def _resolve_icon():
    explicit = str(os.environ.get("MACOS_BOOTSTRAPPER_ICON_PATH", "") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists() and path.suffix.lower() == ".icns":
            return str(path)
    candidates = [Path("assets") / "installer.icns", *MACOS_ICON_CANDIDATES, *ICON_CANDIDATES]
    for rel in candidates:
        path = SRC_ROOT / rel
        if path.exists() and path.suffix.lower() == ".icns":
            return str(path)
    return None


app_name = str(MACOS_BOOTSTRAPPER_APP_NAME)
base_name = Path(app_name).stem
icon_path = _resolve_icon()

datas = []
if icon_path:
    datas.append((icon_path, "."))
for rel in (Path("pictures") / "scene.png", Path("pictures") / "scene.jpg"):
    asset_path = SRC_ROOT / rel
    if asset_path.exists():
        datas.append((str(asset_path), str(rel.parent)))

a = Analysis(
    [str(SPEC_DIR / "bootstrap_installer_macos.py")],
    pathex=[str(SPEC_DIR), str(SRC_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=["PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "tkinter.ttk", "tkinter.font", "tkinter.filedialog", "tkinter.messagebox"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=base_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=base_name,
)

app = BUNDLE(
    coll,
    name=app_name,
    icon=icon_path,
    bundle_identifier=os.environ.get(
        "MACOS_BOOTSTRAPPER_BUNDLE_ID",
        f"cn.wisemodel.{base_name.lower().replace('-', '')}",
    ),
)
