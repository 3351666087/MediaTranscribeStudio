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

from dist_config import BOOTSTRAPPER_EXE_NAME, ICON_CANDIDATES
from spec_helpers import collect_windows_conda_runtime_binaries, dedupe_toc_entries


def _resolve_icon():
    # Prefer a dedicated installer icon if available.
    candidates = [Path("assets") / "installer.ico", *ICON_CANDIDATES]
    for rel in candidates:
        p = SRC_ROOT / rel
        if p.exists():
            return str(p)
    return None


def _resolve_embedded_inno_core():
    raw = os.environ.get("INNO_CORE_SETUP_PATH", "").strip().strip('"')
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = SRC_ROOT / p
    try:
        if p.exists() and p.is_file():
            return p.resolve()
    except Exception:
        return None
    return None


icon_path = _resolve_icon()
inno_core_setup_path = _resolve_embedded_inno_core()
binaries = collect_windows_conda_runtime_binaries(include_tk=False)

datas = []
if icon_path:
    datas.append((icon_path, "."))
if inno_core_setup_path is not None:
    datas.append((str(inno_core_setup_path), "."))

a = Analysis(
    [str(SPEC_DIR / "bootstrap_installer_qt.py")],
    pathex=[str(SPEC_DIR), str(SRC_ROOT)],
    binaries=dedupe_toc_entries(binaries),
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
    a.binaries,
    a.datas,
    [],
    name=Path(BOOTSTRAPPER_EXE_NAME).stem,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=icon_path,
)
