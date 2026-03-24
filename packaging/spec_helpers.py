from __future__ import annotations

import os
import sys
from pathlib import Path


def dedupe_toc_entries(entries):
    """Deduplicate PyInstaller TOC-style entries while preserving order."""
    seen = set()
    out = []
    for item in entries or []:
        try:
            src, dest = item[0], item[1]
            key = (str(src).lower(), str(dest))
        except Exception:
            key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _candidate_env_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for raw in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if not raw:
            continue
        try:
            p = Path(raw).resolve()
        except Exception:
            p = Path(raw)
        k = str(p).lower()
        if k in seen or not p.exists():
            continue
        seen.add(k)
        roots.append(p)
    return roots


def _dll_search_dirs(root: Path) -> list[Path]:
    return [
        root / "Library" / "bin",
        root / "DLLs",
        root / "bin",
    ]


def collect_windows_conda_runtime_binaries(*, include_tk: bool = False):
    """
    Collect conda runtime DLLs that PyInstaller can miss when building simple Qt apps.

    This specifically covers stdlib extension dependencies in conda envs, e.g.
    _ctypes -> ffi.dll, _ssl/_hashlib -> libssl/libcrypto, and compression libs.
    """

    if os.name != "nt":
        return []

    patterns = [
        "ffi*.dll",
        "libssl*.dll",
        "libcrypto*.dll",
        "liblzma*.dll",
        "libbz2*.dll",
        "LIBBZ2*.dll",
    ]
    if include_tk:
        patterns.extend(["tcl*.dll", "tk*.dll"])

    out = []
    seen_names: set[str] = set()
    for root in _candidate_env_roots():
        for dll_dir in _dll_search_dirs(root):
            if not dll_dir.exists():
                continue
            for pattern in patterns:
                try:
                    matches = sorted(dll_dir.glob(pattern))
                except Exception:
                    matches = []
                for p in matches:
                    if not p.is_file():
                        continue
                    name_key = p.name.lower()
                    if name_key in seen_names:
                        continue
                    seen_names.add(name_key)
                    out.append((str(p), "."))
    return dedupe_toc_entries(out)
