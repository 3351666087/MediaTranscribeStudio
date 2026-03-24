# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path


_specpath = Path(SPECPATH).resolve()
SPEC_DIR = _specpath if _specpath.is_dir() else _specpath.parent
SRC_ROOT = SPEC_DIR.parent
if str(SPEC_DIR) not in sys.path:
    sys.path.insert(0, str(SPEC_DIR))

from dist_config import ICON_CANDIDATES, UNINSTALL_EXE_NAME
from spec_helpers import collect_windows_conda_runtime_binaries, dedupe_toc_entries


def _resolve_icon():
    for rel in ICON_CANDIDATES:
        p = SRC_ROOT / rel
        if p.exists():
            return str(p)
    return None


icon_path = _resolve_icon()
binaries = collect_windows_conda_runtime_binaries(include_tk=False)

a = Analysis(
    [str(SPEC_DIR / "uninstall_app_qt.py")],
    pathex=[str(SPEC_DIR), str(SRC_ROOT)],
    binaries=dedupe_toc_entries(binaries),
    datas=[(icon_path, ".")] if icon_path else [],
    hiddenimports=["PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "tkinter.ttk", "tkinter.font", "tkinter.messagebox"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=Path(UNINSTALL_EXE_NAME).stem,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=icon_path,
)
