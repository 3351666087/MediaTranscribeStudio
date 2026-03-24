# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import sysconfig
from fnmatch import fnmatch
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)
from PyInstaller.building.datastruct import TOC, Tree

_specpath = Path(SPECPATH).resolve()
SPEC_DIR = _specpath if _specpath.is_dir() else _specpath.parent
if str(SPEC_DIR) not in sys.path:
    sys.path.insert(0, str(SPEC_DIR))

from dist_config import APP_NAME, ICON_CANDIDATES, INCLUDE_CHECKPOINTS_DEFAULT, MACOS_ICON_CANDIDATES

SRC_ROOT = SPEC_DIR.parent
ENTRY_SCRIPT = SRC_ROOT / "main.py"


def _safe_collect_submodules(pkg_name: str) -> list[str]:
    try:
        return collect_submodules(pkg_name)
    except Exception:
        return []


def _safe_collect_all(pkg_name: str, **kwargs):
    try:
        return collect_all(pkg_name, **kwargs)
    except Exception:
        return ([], [], [])


def _safe_collect_data(pkg_name: str, **kwargs):
    try:
        return collect_data_files(pkg_name, **kwargs)
    except Exception:
        return []


def _safe_collect_bins(pkg_name: str):
    try:
        return collect_dynamic_libs(pkg_name)
    except Exception:
        return []


def _safe_copy_metadata(dist_name: str):
    try:
        return copy_metadata(dist_name)
    except Exception:
        return []


def _package_dir(pkg_name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(pkg_name)
    except Exception:
        return None
    if not spec or not spec.origin:
        return None
    try:
        return Path(spec.origin).resolve().parent
    except Exception:
        return None


def _safe_collect_package_files(pkg_name: str, patterns: list[str]):
    """Explicitly collect package data files when generic hooks miss them."""
    pkg_dir = _package_dir(pkg_name)
    if pkg_dir is None or not pkg_dir.exists():
        return []

    out = []
    pkg_dest_root = Path(*pkg_name.split("."))
    for pattern in patterns:
        try:
            matched = sorted(pkg_dir.glob(pattern))
        except Exception:
            matched = []
        for src in matched:
            if not src.is_file():
                continue
            try:
                rel_parent = src.parent.relative_to(pkg_dir)
            except Exception:
                rel_parent = Path()
            dest_dir = str(pkg_dest_root / rel_parent)
            out.append((str(src), dest_dir))
    return out


def _safe_collect_package_tree(
    pkg_name: str,
    *,
    binary_suffixes: set[str] | None = None,
    extra_binary_patterns: tuple[str, ...] = (),
):
    """
    Collect a package by walking its files directly, without importing submodules.

    This is used for packages like MLX whose extension modules can crash when
    imported in restricted/headless build environments.
    """
    pkg_dir = _package_dir(pkg_name)
    if pkg_dir is None or not pkg_dir.exists():
        return [], []

    if binary_suffixes is None:
        binary_suffixes = {".pyd", ".so", ".dylib", ".dll", ".bundle"}

    datas_out: list[tuple[str, str]] = []
    bins_out: list[tuple[str, str]] = []
    pkg_dest_root = Path(*pkg_name.split("."))
    seen_extra_binary: set[Path] = set()

    for src in pkg_dir.rglob("*"):
        try:
            if not src.is_file():
                continue
        except Exception:
            continue

        try:
            rel_parent = src.parent.relative_to(pkg_dir)
        except Exception:
            rel_parent = Path()
        dest_dir = str(pkg_dest_root / rel_parent)
        if src.suffix.lower() in binary_suffixes:
            bins_out.append((str(src), dest_dir))
        else:
            datas_out.append((str(src), dest_dir))

    for pattern in extra_binary_patterns:
        try:
            matched = sorted(pkg_dir.glob(pattern))
        except Exception:
            matched = []
        for src in matched:
            if not src.is_file() or src in seen_extra_binary:
                continue
            seen_extra_binary.add(src)
            try:
                rel_parent = src.parent.relative_to(pkg_dir)
            except Exception:
                rel_parent = Path()
            dest_dir = str(pkg_dest_root / rel_parent)
            bins_out.append((str(src), dest_dir))

    return datas_out, bins_out


def _is_playwright_local_browsers_path(value) -> bool:
    text = str(value or "").replace("\\", "/")
    return (
        "/playwright/driver/package/.local-browsers/" in f"/{text}"
        or text.endswith("/playwright/driver/package/.local-browsers")
    )


def _is_cv2_private_openssl_path(value) -> bool:
    text = str(value or "").replace("\\", "/")
    return any(
        marker in f"/{text}"
        for marker in (
            "/cv2/.dylibs/libcrypto.3.dylib",
            "/cv2/.dylibs/libssl.3.dylib",
        )
    )


def _filter_playwright_local_browsers_entries(entries):
    out = []
    for entry in entries:
        if isinstance(entry, (tuple, list)):
            probe_parts = entry[:2]
        else:
            probe_parts = (entry,)
        if any(_is_playwright_local_browsers_path(part) for part in probe_parts):
            continue
        out.append(entry)
    return out


def _filter_cv2_private_openssl_entries(entries):
    out = []
    for entry in entries:
        if isinstance(entry, (tuple, list)):
            probe_parts = entry[:2]
        else:
            probe_parts = (entry,)
        if any(_is_cv2_private_openssl_path(part) for part in probe_parts):
            continue
        out.append(entry)
    return out


def _filter_playwright_local_browsers_toc(entries):
    out = []
    removed = 0
    for entry in entries:
        probe_parts = entry[:2] if isinstance(entry, tuple) else (entry,)
        if any(_is_playwright_local_browsers_path(part) for part in probe_parts):
            removed += 1
            continue
        out.append(entry)
    return TOC(out), removed


def _filter_cv2_private_openssl_toc(entries):
    out = []
    removed = 0
    for entry in entries:
        probe_parts = entry[:2] if isinstance(entry, tuple) else (entry,)
        if any(_is_cv2_private_openssl_path(part) for part in probe_parts):
            removed += 1
            continue
        out.append(entry)
    return TOC(out), removed


def _dedupe_item_key(item):
    if isinstance(item, Tree):
        return (
            "Tree",
            str(item.root or ""),
            str(item.prefix or ""),
            tuple(str(x) for x in (item.excludes or [])),
            str(item.typecode or ""),
        )

    if isinstance(item, tuple):
        return ("tuple", tuple(_dedupe_item_key(part) for part in item))

    if isinstance(item, list):
        return ("list", tuple(_dedupe_item_key(part) for part in item))

    if isinstance(item, dict):
        return (
            "dict",
            tuple(sorted((str(key), _dedupe_item_key(value)) for key, value in item.items())),
        )

    if isinstance(item, Path):
        return ("Path", str(item))

    try:
        hash(item)
    except Exception:
        return (type(item).__name__, repr(item))
    return ("hashable", item)


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        key = _dedupe_item_key(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _tree_entry_excluded(name: str, rel_path: str, excludes: list[str] | None) -> bool:
    rel_text = str(rel_path or "").replace("\\", "/")
    for pattern in excludes or []:
        text = str(pattern or "").strip()
        if not text:
            continue
        if any(ch in text for ch in "*?[]"):
            if fnmatch(name, text) or (rel_text and fnmatch(rel_text, text)):
                return True
        elif name == text or rel_text == text:
            return True
    return False


def _collect_directory_tree_data(root: Path, dest_root: str, excludes: list[str] | None = None):
    out: list[tuple[str, str]] = []
    if not root.exists() or not root.is_dir():
        return out

    dest_base = Path(dest_root)
    for current_root, dirnames, filenames in os.walk(root):
        current_root_path = Path(current_root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not _tree_entry_excluded(
                name,
                (current_root_path / name).relative_to(root).as_posix(),
                excludes,
            )
        )
        for filename in sorted(filenames):
            rel_file = (current_root_path / filename).relative_to(root).as_posix()
            if _tree_entry_excluded(filename, rel_file, excludes):
                continue

            src = current_root_path / filename
            try:
                rel_parent = src.parent.relative_to(root)
            except Exception:
                rel_parent = Path()

            dest_dir = dest_base / rel_parent if str(rel_parent) not in {"", "."} else dest_base
            out.append((str(src), str(dest_dir)))
    return out


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _skip_playwright_local_browsers() -> bool:
    # On macOS, bundling Chromium inside the raw PyInstaller output tends to
    # break COLLECT/BUNDLE codesign. The dedicated macOS packager copies the
    # prepared browser payload back into the finished .app afterwards.
    return _env_flag("PYI_SKIP_PLAYWRIGHT_LOCAL_BROWSERS", sys.platform == "darwin")


def _safe_add_tree_data(items: list[tuple[str, str]], src: Path, dest: str) -> None:
    try:
        if not src.exists():
            return
        if src.is_dir():
            items.extend(_collect_directory_tree_data(src, str(dest)))
        else:
            items.append((str(src), str(dest)))
    except Exception:
        pass


def _safe_add_relative_tree_data(
    items: list[tuple[str, str]],
    root: Path | None,
    rel_path: Path,
    dest_root: str,
) -> None:
    if root is None:
        return
    try:
        src = root / rel_path
        if not src.exists():
            return
        dest_path = Path(dest_root) / rel_path
        if src.is_dir():
            items.extend(_collect_directory_tree_data(src, str(dest_path)))
        else:
            dest_dir = dest_path.parent if str(dest_path.parent) not in {"", "."} else Path(dest_root)
            items.append((str(src), str(dest_dir)))
    except Exception:
        pass


def _split_env_paths(raw: str) -> list[Path]:
    parts: list[Path] = []
    for item in str(raw or "").split(os.pathsep):
        text = str(item or "").strip().strip('"')
        if text:
            parts.append(Path(text).expanduser())
    return parts


def _unique_existing_dirs(candidates: list[Path | None]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate.expanduser()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if resolved.exists() and resolved.is_dir():
                out.append(resolved)
        except Exception:
            continue
    return out


def _hf_repo_cache_dir(repo: str) -> Path | None:
    normalized = [part.strip() for part in str(repo or "").strip().strip("/").split("/") if part.strip()]
    if not normalized:
        return None
    return Path("models--" + "--".join(normalized))


def _discover_native_bundle_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("BUNDLE_NATIVE_DIRS", "BUNDLE_NATIVE_DIR"):
        candidates.extend(_split_env_paths(os.environ.get(env_name, "")))

    for rel in (
        "native/install",
        "native/dist",
        "build/native/install",
        "build/install",
        "build/cmake-install",
        "dist/native",
    ):
        candidates.append(SRC_ROOT / rel)

    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if resolved.exists() and resolved.is_dir():
                out.append(resolved)
        except Exception:
            continue
    return out


def _collect_native_bundle_artifacts(root: Path):
    datas_out: list[tuple[str, str]] = []
    bins_out: list[tuple[str, str]] = []
    binary_suffixes = {".dll", ".exe", ".pyd"} if os.name == "nt" else {".dylib", ".so", ".bundle"}
    for src in root.rglob("*"):
        try:
            if not src.is_file():
                continue
        except Exception:
            continue

        try:
            rel_parent = src.parent.relative_to(root)
        except Exception:
            rel_parent = Path()
        dest_dir = str(Path("native") / rel_parent)
        is_binary = src.suffix.lower() in binary_suffixes
        if not is_binary and os.name != "nt":
            try:
                is_binary = src.suffix == "" and os.access(src, os.X_OK)
            except Exception:
                is_binary = False
        if is_binary:
            bins_out.append((str(src), dest_dir))
        else:
            datas_out.append((str(src), dest_dir))
    return datas_out, bins_out


def _site_packages_dir() -> Path | None:
    candidates: list[Path] = []
    try:
        purelib = sysconfig.get_paths().get("purelib")
        if purelib:
            candidates.append(Path(purelib))
    except Exception:
        pass
    try:
        candidates.append(Path(sys.prefix) / "Lib" / "site-packages")
    except Exception:
        pass
    try:
        base_prefix = getattr(sys, "base_prefix", None)
        if base_prefix:
            candidates.append(Path(base_prefix) / "Lib" / "site-packages")
    except Exception:
        pass

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
        except Exception:
            continue
    return None


def _site_packages_pack_all_excludes() -> list[str]:
    excludes = [
        "av",
        "av/*",
        "av/**",
        "av-*.dist-info",
        "av.libs",
        "av.libs/*",
    ]
    if _skip_playwright_local_browsers():
        excludes.extend(
            [
                "playwright/driver/package/.local-browsers",
                "playwright/driver/package/.local-browsers/*",
                "playwright/driver/package/.local-browsers/**",
            ]
        )
    return excludes


def _resolve_icon():
    env_icon = os.environ.get("APP_ICON")
    if env_icon:
        p = Path(env_icon)
        if not p.is_absolute():
            p = SRC_ROOT / p
        if p.exists() and (sys.platform != "darwin" or p.suffix.lower() == ".icns"):
            return str(p)

    candidates = MACOS_ICON_CANDIDATES if sys.platform == "darwin" else ICON_CANDIDATES
    for rel in candidates:
        p = SRC_ROOT / rel
        if p.exists() and (sys.platform != "darwin" or p.suffix.lower() == ".icns"):
            return str(p)
    return None


def _resolve_optional_tool_executable(name: str) -> Path | None:
    exe_names = [f"{name}.exe", name] if os.name == "nt" else [name]

    specific_env = os.environ.get(f"{name.upper()}_BINARY", "").strip()
    if specific_env:
        p = Path(specific_env).expanduser()
        try:
            if p.exists() and p.is_file():
                return p.resolve()
            if p.exists() and p.is_dir():
                for exe in exe_names:
                    candidate = p / exe
                    if candidate.exists() and candidate.is_file():
                        return candidate.resolve()
        except Exception:
            pass
        resolved_env_name = shutil.which(specific_env)
        if resolved_env_name:
            try:
                return Path(resolved_env_name).resolve()
            except Exception:
                return Path(resolved_env_name)

    # When only FFMPEG_BINARY is configured, infer ffprobe from sibling files.
    if name == "ffprobe":
        ffmpeg_env = os.environ.get("FFMPEG_BINARY", "").strip().strip('"')
        if ffmpeg_env:
            ffmpeg_path = Path(ffmpeg_env).expanduser()
            sibling_candidates = [
                ffmpeg_path.parent,
                ffmpeg_path.parent / "bin",
            ]
            for base in sibling_candidates:
                for exe in exe_names:
                    p = base / exe
                    try:
                        if p.exists() and p.is_file():
                            return p.resolve()
                    except Exception:
                        continue

    for env_name in ("BUNDLE_FFMPEG_DIR", "FFMPEG_DIR", "FFMPEG_HOME", "FFMPEG_ROOT"):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        base = Path(raw).expanduser()
        for candidate in [base, base / "bin"]:
            for exe in exe_names:
                p = candidate / exe
                try:
                    if p.exists() and p.is_file():
                        return p.resolve()
                except Exception:
                    continue

    # Conda environments usually place ffmpeg executables under Library/bin.
    conda_roots = []
    conda_prefix = str(os.environ.get("CONDA_PREFIX", "") or "").strip()
    if conda_prefix:
        conda_roots.append(Path(conda_prefix))
    conda_roots.append(Path(sys.prefix))
    base_prefix = getattr(sys, "base_prefix", None)
    if base_prefix:
        conda_roots.append(Path(base_prefix))

    for root in conda_roots:
        for candidate in (root / "Library" / "bin", root / "Scripts", root / "bin"):
            for exe in exe_names:
                p = candidate / exe
                try:
                    if p.exists() and p.is_file():
                        return p.resolve()
                except Exception:
                    continue

    # imageio-ffmpeg ships a standalone ffmpeg binary that often works in offline bundles.
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg  # type: ignore

            imageio_exe = Path(str(imageio_ffmpeg.get_ffmpeg_exe()))
            if imageio_exe.exists() and imageio_exe.is_file():
                return imageio_exe.resolve()
        except Exception:
            pass

    resolved = shutil.which(name)
    if resolved:
        try:
            return Path(resolved).resolve()
        except Exception:
            return Path(resolved)
    return None


def _add_binary_once(binaries: list[tuple[str, str]], src: Path, dest: str) -> bool:
    entry = (str(src), str(dest))
    if entry in binaries:
        return False
    binaries.append(entry)
    return True


def _add_ffmpeg_sidecar_binaries(
    binaries: list[tuple[str, str]],
    ffmpeg_path: Path | None,
) -> int:
    if ffmpeg_path is None:
        return 0

    sidecar_patterns = (
        "avcodec*.dll",
        "avdevice*.dll",
        "avfilter*.dll",
        "avformat*.dll",
        "avutil*.dll",
        "swresample*.dll",
        "swscale*.dll",
        "postproc*.dll",
        "libass*.dll",
        "ass*.dll",
        "freetype*.dll",
        "libfreetype*.dll",
        "fribidi*.dll",
        "harfbuzz*.dll",
        "graphite2*.dll",
        "fontconfig*.dll",
        "iconv*.dll",
        "libcharset*.dll",
        "expat*.dll",
        "libexpat*.dll",
        "xml2*.dll",
        "libxml2*.dll",
        "libpng*.dll",
        "png*.dll",
        "brotli*.dll",
        "brotlicommon*.dll",
        "brotlidec*.dll",
        "bz2*.dll",
        "libbz2*.dll",
        "zlib*.dll",
        "libz*.dll",
        "intl*.dll",
        "glib-2*.dll",
        "pcre2-*.dll",
        "winpthread*.dll",
        "gcc_s*.dll",
        "stdc++*.dll",
    )
    added = 0
    ffmpeg_dir = ffmpeg_path.parent
    for pattern in sidecar_patterns:
        try:
            matched = sorted(ffmpeg_dir.glob(pattern))
        except Exception:
            matched = []
        for item in matched:
            if item.is_file() and _add_binary_once(binaries, item, "tools/ffmpeg"):
                added += 1
    return added


def _add_required_ffmpeg_binaries(binaries: list[tuple[str, str]]) -> None:
    ffmpeg = _resolve_optional_tool_executable("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "PyInstaller spec: required ffmpeg executable was not found. "
            "Install ffmpeg, or set FFMPEG_BINARY / BUNDLE_FFMPEG_DIR / FFMPEG_DIR."
        )

    ffprobe = _resolve_optional_tool_executable("ffprobe")

    if ffprobe is None and ffmpeg is not None:
        for candidate in (
            ffmpeg.parent / "ffprobe.exe",
            ffmpeg.parent / "ffprobe",
            ffmpeg.parent / "bin" / "ffprobe.exe",
            ffmpeg.parent / "bin" / "ffprobe",
        ):
            try:
                if candidate.exists() and candidate.is_file():
                    ffprobe = candidate.resolve()
                    break
            except Exception:
                continue

    for tool_path in (ffmpeg, ffprobe):
        if tool_path is not None:
            _add_binary_once(binaries, tool_path, "tools/ffmpeg")

    sidecar_added = _add_ffmpeg_sidecar_binaries(binaries, ffmpeg)

    extra = f" (+{sidecar_added} dll sidecars)" if sidecar_added else ""
    if ffprobe is not None:
        print(f"PyInstaller spec: bundling ffmpeg + ffprobe from {ffmpeg.parent}{extra}")
    else:
        print(
            "PyInstaller spec: bundling required ffmpeg from "
            f"{ffmpeg.parent}{extra}; ffprobe not found and will remain optional"
        )


def _add_required_python_ssl_binaries(binaries: list[tuple[str, str]]) -> None:
    if sys.platform != "darwin":
        return

    lib_dirs = []
    for candidate in (
        Path(sys.prefix) / "lib",
        Path(getattr(sys, "base_prefix", "")) / "lib" if getattr(sys, "base_prefix", None) else None,
    ):
        if candidate is not None:
            lib_dirs.append(candidate)

    required = ("libssl.3.dylib", "libcrypto.3.dylib")
    added = 0
    for lib_dir in lib_dirs:
        try:
            exists = lib_dir.exists() and lib_dir.is_dir()
        except Exception:
            exists = False
        if not exists:
            continue
        for name in required:
            src = lib_dir / name
            try:
                if src.exists() and src.is_file() and _add_binary_once(binaries, src.resolve(), "."):
                    added += 1
            except Exception:
                continue

    if added:
        print(f"PyInstaller spec: bundling Python OpenSSL runtime dylibs ({added} files)")


icon_path = _resolve_icon()

datas = []
binaries = []
hiddenimports = []

if icon_path:
    datas.append((str(icon_path), "."))

_add_required_ffmpeg_binaries(binaries)
_add_required_python_ssl_binaries(binaries)

native_bundle_dirs = _discover_native_bundle_dirs()
for native_dir in native_bundle_dirs:
    native_datas, native_bins = _collect_native_bundle_artifacts(native_dir)
    datas.extend(native_datas)
    binaries.extend(native_bins)
    print(
        "PyInstaller spec: bundling native artifacts from "
        f"{native_dir} ({len(native_bins)} bin, {len(native_datas)} data files)"
    )


# Keep config beside the built exe so the UI can persist edits cleanly.
cfg = SRC_ROOT / "config.yaml"
if cfg.exists():
    datas.append((str(cfg), "."))

pictures_dir = SRC_ROOT / "pictures"
if pictures_dir.exists() and pictures_dir.is_dir():
    datas.append((str(pictures_dir), "pictures"))

# Optional conda/site-packages "pack_all" mode. This mirrors the entire current
# environment site-packages into the frozen app to minimize missing optional
# package/data edge cases (especially for NeMo/MSDD stacks).
pack_all = _env_flag("PACK_ALL", True)
if pack_all:
    sp_dir = _site_packages_dir()
    if sp_dir is not None:
        datas.extend(
            _collect_directory_tree_data(
                sp_dir,
                ".",
                excludes=_site_packages_pack_all_excludes(),
            )
        )

# Large local checkpoints are optional, but usually desired for offline use.
include_checkpoints = (
    os.environ.get("INCLUDE_CHECKPOINTS", str(INCLUDE_CHECKPOINTS_DEFAULT)).strip().lower()
    in {"1", "true", "yes", "on"}
)
checkpoints_dir = SRC_ROOT / "checkpoints"
if include_checkpoints and checkpoints_dir.exists() and checkpoints_dir.is_dir():
    datas.append((str(checkpoints_dir), "checkpoints"))

# Optional model/cache bundling for offline and no-redownload packaged runs.
# Default ON (can disable with INCLUDE_RUNTIME_MODEL_CACHES=0).
include_runtime_model_caches = _env_flag("INCLUDE_RUNTIME_MODEL_CACHES", True)
if include_runtime_model_caches:
    localappdata = os.environ.get("LOCALAPPDATA", "")
    home = Path.home()
    model_cache_bundle_mode = str(
        os.environ.get("MODEL_CACHE_BUNDLE_MODE", "full") or "full"
    ).strip().lower()
    # Project-local NeMo artifacts are always worth packing for fully offline runs.
    _safe_add_tree_data(datas, SRC_ROOT / "output_files" / ".nemo_msdd", "model_caches/nemo")

    if model_cache_bundle_mode in {"curated", "minimal", "selected"}:
        hf_hub_roots = _unique_existing_dirs(
            [
                Path(os.environ["HUGGINGFACE_HUB_CACHE"])
                if os.environ.get("HUGGINGFACE_HUB_CACHE")
                else None,
                (Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
                home / ".cache" / "huggingface" / "hub",
                (Path(localappdata) / "huggingface" / "hub") if localappdata else None,
                Path(localappdata) / "huggingface" if localappdata else None,
            ]
        )
        modelscope_roots = _unique_existing_dirs(
            [
                Path(os.environ["MODELSCOPE_CACHE"]) if os.environ.get("MODELSCOPE_CACHE") else None,
                home / ".cache" / "modelscope",
                Path(localappdata) / "modelscope" if localappdata else None,
            ]
        )
        torch_roots = _unique_existing_dirs(
            [
                Path(os.environ["TORCH_HOME"]) if os.environ.get("TORCH_HOME") else None,
                home / ".cache" / "torch",
            ]
        )
        nemo_roots = _unique_existing_dirs(
            [
                Path(os.environ["NEMO_CACHE_DIR"]) if os.environ.get("NEMO_CACHE_DIR") else None,
                home / ".cache" / "torch" / "NeMo",
            ]
        )

        for rel in _split_env_paths(os.environ.get("BUNDLE_HF_CACHE_DIRS", "")):
            for root in hf_hub_roots:
                _safe_add_relative_tree_data(datas, root, rel, "model_caches/huggingface/hub")
        for repo in str(os.environ.get("BUNDLE_HF_MODEL_REPOS", "") or "").split(os.pathsep):
            rel = _hf_repo_cache_dir(repo)
            if rel is None:
                continue
            for root in hf_hub_roots:
                _safe_add_relative_tree_data(datas, root, rel, "model_caches/huggingface/hub")

        for rel in _split_env_paths(os.environ.get("BUNDLE_MODELSCOPE_CACHE_SUBDIRS", "")):
            for root in modelscope_roots:
                _safe_add_relative_tree_data(datas, root, rel, "model_caches/modelscope")

        for rel in _split_env_paths(os.environ.get("BUNDLE_TORCH_CACHE_SUBDIRS", "")):
            for root in torch_roots:
                _safe_add_relative_tree_data(datas, root, rel, "model_caches/torch")

        for rel in _split_env_paths(os.environ.get("BUNDLE_NEMO_CACHE_SUBDIRS", "")):
            for root in nemo_roots:
                _safe_add_relative_tree_data(datas, root, rel, "model_caches/nemo")
    else:
        cache_mappings = [
            # Environment/user caches
            (Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else None, "model_caches/huggingface"),
            (
                Path(os.environ["HUGGINGFACE_HUB_CACHE"])
                if os.environ.get("HUGGINGFACE_HUB_CACHE")
                else None,
                "model_caches/huggingface/hub",
            ),
            (Path(os.environ["MODELSCOPE_CACHE"]) if os.environ.get("MODELSCOPE_CACHE") else None, "model_caches/modelscope"),
            (Path(os.environ["TORCH_HOME"]) if os.environ.get("TORCH_HOME") else None, "model_caches/torch"),
            (Path(os.environ["NEMO_CACHE_DIR"]) if os.environ.get("NEMO_CACHE_DIR") else None, "model_caches/nemo"),
            # Common Windows/macOS defaults when env vars are not set
            (home / ".cache" / "huggingface", "model_caches/huggingface"),
            (home / ".cache" / "huggingface" / "hub", "model_caches/huggingface/hub"),
            (home / ".cache" / "modelscope", "model_caches/modelscope"),
            (home / ".cache" / "modelscope" / "hub", "model_caches/modelscope/hub"),
            (home / ".cache" / "torch", "model_caches/torch"),
            (home / ".cache" / "torch" / "hub", "model_caches/torch/hub"),
            (home / ".cache" / "torch" / "hub" / "checkpoints", "model_caches/torch/hub/checkpoints"),
            (home / ".cache" / "torch" / "whisper", "model_caches/torch/whisper"),
            (home / ".cache" / "torch" / "NeMo", "model_caches/nemo"),
            (Path(localappdata) / "huggingface" if localappdata else None, "model_caches/huggingface"),
            (Path(localappdata) / "modelscope" if localappdata else None, "model_caches/modelscope"),
        ]
        for src, dest in cache_mappings:
            if src is None:
                continue
            _safe_add_tree_data(datas, src, dest)

    extra_cache_dirs = os.environ.get("BUNDLE_MODEL_CACHE_DIRS", "").strip()
    if extra_cache_dirs:
        for idx, raw_dir in enumerate(extra_cache_dirs.split(os.pathsep)):
            path_str = str(raw_dir or "").strip().strip('"')
            if not path_str:
                continue
            _safe_add_tree_data(datas, Path(path_str), f"model_caches/extra_{idx}")


# Maximal package collection for optional backends.
# For fragile ML stacks, prefer direct filesystem walking over import-driven
# helper collection so build-time version mismatches do not flood logs or drop
# package contents.
stable_collect_all_pkgs = (
    "playwright",
    "IPython",
    "traitlets",
    "jedi",
    "parso",
    "pygments",
    "matplotlib_inline",
    "stack_data",
    "executing",
    "asttokens",
    "pure_eval",
    "omegaconf",
    "hydra",
    "pyannote.pipeline",
    "transformers",
    "datasets",
)
for pkg in stable_collect_all_pkgs:
    all_datas, all_bins, all_hidden = _safe_collect_all(pkg)
    if pkg == "playwright" and _skip_playwright_local_browsers():
        all_datas = _filter_playwright_local_browsers_entries(all_datas)
        all_bins = _filter_playwright_local_browsers_entries(all_bins)
    datas.extend(all_datas)
    binaries.extend(all_bins)
    hiddenimports.extend(all_hidden)

fragile_package_tree_pkgs = (
    "prompt_toolkit",
    "modelscope",
    "pyannote",
    "pyannote.audio",
    "pyannote.core",
    "pyannote.database",
    "pyannote.metrics",
    "lightning",
    "lightning_fabric",
    "pytorch_lightning",
    "torchmetrics",
    "nemo",
    "nemo_text_processing",
    "clearvoice",
    "speechbrain",
    "hyperpyyaml",
    "asteroid",
    "asteroid_filterbanks",
)
for pkg in fragile_package_tree_pkgs:
    tree_datas, tree_bins = _safe_collect_package_tree(pkg)
    datas.extend(tree_datas)
    binaries.extend(tree_bins)
    hiddenimports.append(pkg)


# PyInstaller already has hooks for many of these packages; these are safety nets
# for runtime imports used by optional backends.
optional_hidden_pkgs = [
    "mts_ui",
    "openai",
    "playwright",
    "playwright.sync_api",
    "pdfkit",
    "IPython",
    "imageio_ffmpeg",
    "faster_whisper",
    "funasr",
    "huggingface_hub",
    "omegaconf",
    "pyannote.pipeline",
    "pytorch_metric_learning",
    "torch_audiomentations",
    "lhotse",
    "webdataset",
    "megatron",
    "megatron.core",
    "transformers",
    "datasets",
]
for pkg in optional_hidden_pkgs:
    hiddenimports.extend(_safe_collect_submodules(pkg))

# These packages are mirrored as source files via direct tree collection above;
# avoid import-driven recursive discovery because many optional extras fail to
# import in a build environment yet still work from packaged source on disk.
hiddenimports.extend(
    [
        "lightning",
        "lightning_fabric",
        "pytorch_lightning",
        "torchmetrics",
        "clearvoice",
        "pyannote",
        "pyannote.audio",
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "speechbrain",
        "hyperpyyaml",
        "yamlargparse",
        "asteroid",
        "asteroid_filterbanks",
    ]
)

# A few direct modules can be enough when full submodule collection is unavailable.
hiddenimports.extend(
    [
        "sitecustomize",
        "yaml",
        "yaml.loader",
        "yaml.dumper",
        "tkinter",
        "tkinter.ttk",
    ]
)

# Dynamic libraries for heavy numerical/runtime packages.
for pkg in ("torch", "torchaudio", "numpy", "ctranslate2", "PySide6", "taichi"):
    binaries.extend(_safe_collect_bins(pkg))

# Package data and metadata often needed by runtime version checks/plugins.
for pkg in (
    "PySide6",
    "tqdm",
    "funasr",
    "modelscope",
    "taichi",
    "omegaconf",
    "lightning",
    "lightning_fabric",
    "pytorch_lightning",
    "torchmetrics",
    "faster_whisper",
    "playwright",
    "IPython",
    "speechbrain",
    "hyperpyyaml",
    "pyannote.pipeline",
    "clearvoice",
):
    pkg_datas = _safe_collect_data(pkg)
    if pkg == "playwright" and _skip_playwright_local_browsers():
        pkg_datas = _filter_playwright_local_browsers_entries(pkg_datas)
    datas.extend(pkg_datas)

# Some packages rely on non-Python assets at runtime; keep explicit fallbacks for
# files that commonly go missing in onedir builds (e.g. version checks / Taichi bitcode).
datas.extend(_safe_collect_package_files("funasr", ["version.txt"]))
datas.extend(_safe_collect_package_files("taichi", ["_lib/runtime/*.bc"]))
datas.extend(_safe_collect_package_files("lightning", ["version.info"]))
datas.extend(_safe_collect_package_files("lightning_fabric", ["version.info", "py.typed"]))
# faster-whisper VAD relies on a packaged ONNX asset that PyInstaller hooks may miss.
datas.extend(_safe_collect_package_files("faster_whisper", ["assets/*.onnx"]))
# pyannote/speechbrain runtime configs and templates.
datas.extend(_safe_collect_package_files("speechbrain", ["**/*.yaml", "**/*.json", "**/*.txt"]))
datas.extend(_safe_collect_package_files("pyannote.audio", ["**/*.yaml", "**/*.json"]))
# Playwright browser binaries installed with PLAYWRIGHT_BROWSERS_PATH=0 live here.
if not _skip_playwright_local_browsers():
    datas.extend(_safe_collect_package_files("playwright", ["driver/package/.local-browsers/**/*"]))

# MLX packages are collected via direct file walk only. Avoid PyInstaller helper
# imports here, because importing mlx.core can crash in restricted/headless build
# environments that do not expose a Metal device.
hiddenimports.extend(["mlx", "mlx_whisper"])
mlx_datas, mlx_bins = _safe_collect_package_tree(
    "mlx",
    extra_binary_patterns=("lib/*.metallib",),
)
datas.extend(mlx_datas)
binaries.extend(mlx_bins)
mlx_whisper_datas, mlx_whisper_bins = _safe_collect_package_tree("mlx_whisper")
datas.extend(mlx_whisper_datas)
binaries.extend(mlx_whisper_bins)

for dist_name in (
    "PyYAML",
    "numpy",
    "torch",
    "torchaudio",
    "PySide6",
    "playwright",
    "ipython",
    "openai",
    "faster-whisper",
    "ctranslate2",
    "tqdm",
    "funasr",
    "modelscope",
    "taichi",
    "omegaconf",
    "lightning",
    "lightning-fabric",
    "pytorch-lightning",
    "torchmetrics",
    "lightning-utilities",
    "traitlets",
    "prompt-toolkit",
    "pygments",
    "jedi",
    "parso",
    "hydra-core",
    "nemo-toolkit",
    "speechbrain",
    "hyperpyyaml",
    "yamlargparse",
    "asteroid",
    "asteroid-filterbanks",
    "clearvoice",
    "pyannote.pipeline",
    "pyannote.audio",
    "pyannote.core",
    "pyannote.database",
    "pyannote.metrics",
):
    datas.extend(_safe_copy_metadata(dist_name))

for dist_name in ("mlx", "mlx-whisper"):
    datas.extend(_safe_copy_metadata(dist_name))

# Deduplicate while preserving order.
hiddenimports = _dedupe_preserve_order(hiddenimports)
datas = _dedupe_preserve_order(datas)
binaries = _dedupe_preserve_order(binaries)

if sys.platform == "darwin":
    datas = _filter_cv2_private_openssl_entries(datas)
    binaries = _filter_cv2_private_openssl_entries(binaries)

module_collection_mode = {
    # NeMo diarization/torchscript paths use inspect.getsource() and require .py source
    # files to be available at runtime in frozen builds.
    "nemo": "pyz+py",
    "pyannote": "pyz+py",
    "pyannote.audio": "pyz+py",
    "pyannote.core": "pyz+py",
    "omegaconf": "pyz+py",
    "lightning": "pyz+py",
    "lightning_fabric": "pyz+py",
    "pytorch_lightning": "pyz+py",
    "hydra": "pyz+py",
    "IPython": "pyz+py",
    "speechbrain": "pyz+py",
    "clearvoice": "pyz+py",
    "pyannote.pipeline": "pyz+py",
    "mts_ui": "pyz+py",
}

a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(SRC_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "av",
        "av.__main__",
        "av.about",
        "av.audio",
        "av.audio.codeccontext",
        "av.audio.fifo",
        "av.audio.format",
        "av.audio.frame",
        "av.audio.layout",
        "av.audio.plane",
        "av.audio.resampler",
        "av.audio.stream",
        "av.buffer",
        "av.bytesource",
        "av.codec",
        "av.codec.codec",
        "av.codec.context",
        "av.container",
        "av.container.core",
        "av.container.input",
        "av.container.output",
        "av.datasets",
        "av.error",
        "av.filter",
        "av.filter.context",
        "av.filter.filter",
        "av.filter.graph",
        "av.filter.loudnorm",
        "av.format",
        "av.frame",
        "av.logging",
        "av.option",
        "av.packet",
        "av.plane",
        "av.sidedata",
        "av.stream",
        "av.subtitles",
        "av.subtitles.codeccontext",
        "av.subtitles.stream",
        "av.subtitles.subtitle",
        "av.utils",
        "av.video",
        "av.video.codeccontext",
        "av.video.format",
        "av.video.frame",
        "av.video.plane",
        "av.video.reformatter",
        "av.video.stream",
        "torchcodec",
        "torchcodec._core",
        "torchcodec.decoders",
        "torchcodec.encoders",
        "torchcodec.samplers",
        "torchcodec.transforms",
    ],
    noarchive=False,
    module_collection_mode=module_collection_mode,
)

if sys.platform == "darwin" and _skip_playwright_local_browsers():
    a.binaries, removed_bins = _filter_playwright_local_browsers_toc(a.binaries)
    a.datas, removed_datas = _filter_playwright_local_browsers_toc(a.datas)
    if removed_bins or removed_datas:
        print(
            "PyInstaller spec: removed Playwright .local-browsers TOC entries "
            f"(binaries={removed_bins}, datas={removed_datas})"
        )

if sys.platform == "darwin":
    a.binaries, removed_cv2_bins = _filter_cv2_private_openssl_toc(a.binaries)
    a.datas, removed_cv2_datas = _filter_cv2_private_openssl_toc(a.datas)
    if removed_cv2_bins or removed_cv2_datas:
        print(
            "PyInstaller spec: removed cv2 private OpenSSL copies "
            f"(binaries={removed_cv2_bins}, datas={removed_cv2_datas})"
        )

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
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
    name=APP_NAME,
)

if sys.platform == "darwin" and not _env_flag("PYI_SKIP_MACOS_BUNDLE", False):
    mac_icon_path = icon_path if icon_path and str(icon_path).lower().endswith(".icns") else None
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=mac_icon_path,
        bundle_identifier=os.environ.get(
            "MACOS_BUNDLE_ID",
            f"cn.wisemodel.{APP_NAME.lower()}",
        ),
    )
elif sys.platform == "darwin":
    print("PyInstaller spec: skipping macOS BUNDLE; external wrapper will assemble .app from onedir output")
