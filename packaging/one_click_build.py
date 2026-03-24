from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import urllib.parse
import zipfile
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
PACKAGING_DIR = ROOT / "packaging"
DIST_DIR = ROOT / "dist"
DIST_CONFIG_PATH = PACKAGING_DIR / "dist_config.py"
DEFAULT_INNO_CORE_URL_HINT = str(os.environ.get("MTS_DEFAULT_INNO_CORE_URL", "") or "").strip()
_INVALID_BUNDLE_NEMO_ARTIFACTS: set[str] = set()


def load_dist_config():
    spec = importlib.util.spec_from_file_location("dist_config_runtime", DIST_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {DIST_CONFIG_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pyinstaller_cmd() -> list[str]:
    exe = shutil.which("pyinstaller")
    if exe:
        return [exe]
    return [sys.executable, "-m", "PyInstaller"]


def find_iscc_exe() -> Path | None:
    for name in ("iscc.exe", "iscc", "ISCC.exe", "ISCC"):
        exe = shutil.which(name)
        if exe:
            return Path(exe)
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_signtool_exe() -> Path | None:
    explicit = str(os.environ.get("SIGNTOOL_EXE", "")).strip().strip('"')
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    for name in ("signtool.exe", "signtool"):
        exe = shutil.which(name)
        if exe:
            return Path(exe)

    windows_kits_roots = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10" / "bin",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Windows Kits" / "10" / "bin",
    ]
    candidates: list[Path] = []
    for root in windows_kits_roots:
        if not root.exists():
            continue
        for arch in ("x64", "x86"):
            candidates.extend(Path(p) for p in glob(str(root / "*" / arch / "signtool.exe")))
    if not candidates:
        return None
    # Prefer the highest version path lexicographically (works for SDK version folder names).
    return sorted(candidates, key=lambda p: str(p).lower())[-1]


def release_version_string() -> str:
    raw = str(os.environ.get("RELEASE_VERSION", "")).strip()
    if raw:
        return raw
    # Numeric dotted default is friendly for file metadata and Inno VersionInfo.
    return datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M")


def _resolve_icon_path(cfg) -> Path | None:
    def _is_valid_ico_file(path: Path) -> bool:
        try:
            data = path.read_bytes()[:8]
        except Exception:
            return False
        if len(data) < 6:
            return False
        # ICO header: Reserved(2)=0, Type(2)=1, Count(2)>=1
        reserved = int.from_bytes(data[0:2], "little", signed=False)
        icon_type = int.from_bytes(data[2:4], "little", signed=False)
        count = int.from_bytes(data[4:6], "little", signed=False)
        return reserved == 0 and icon_type == 1 and count >= 1

    def _is_png_file(path: Path) -> bool:
        try:
            return path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        except Exception:
            return False

    def _png_dimensions(path: Path) -> tuple[int, int] | None:
        try:
            data = path.read_bytes()
        except Exception:
            return None
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        if data[12:16] != b"IHDR":
            return None
        try:
            width = int.from_bytes(data[16:20], "big", signed=False)
            height = int.from_bytes(data[20:24], "big", signed=False)
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        return width, height

    def _write_ico_from_png_bytes(
        png_bytes: bytes,
        *,
        width: int,
        height: int,
        out_path: Path,
    ) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w_byte = 0 if width >= 256 else max(1, min(255, int(width)))
        h_byte = 0 if height >= 256 else max(1, min(255, int(height)))
        # ICO file header + one directory entry that points to a PNG frame.
        header = struct.pack("<HHH", 0, 1, 1)
        entry = struct.pack(
            "<BBBBHHII",
            w_byte,
            h_byte,
            0,      # color count (0 for >= 8bpp / PNG)
            0,      # reserved
            1,      # planes
            32,     # bit count (hint only for PNG-compressed frame)
            len(png_bytes),
            6 + 16, # offset to image data
        )
        out_path.write_bytes(header + entry + png_bytes)
        return out_path

    def _repair_png_to_ico_if_needed(path: Path) -> Path | None:
        if not _is_png_file(path):
            return None
        dims = _png_dimensions(path)
        if not dims:
            return None
        width, height = dims
        png_bytes = path.read_bytes()
        fixed_dir = ROOT / "build" / "generated_icons"
        suffix = f"{width}x{height}"
        out_path = fixed_dir / f"{path.stem}-{suffix}.fixed.ico"
        try:
            repaired = _write_ico_from_png_bytes(png_bytes, width=width, height=height, out_path=out_path)
        except Exception:
            return None
        if _is_valid_ico_file(repaired):
            print(f"[info] Repaired icon container for Inno: {path} -> {repaired}")
            return repaired.resolve()
        return None

    candidates: list[Path] = []
    try:
        for rel in getattr(cfg, "ICON_CANDIDATES", []) or []:
            candidates.append(ROOT / Path(rel))
    except Exception:
        pass
    # Prefer a dedicated installer icon if present.
    candidates = [ROOT / "assets" / "installer.ico", *candidates, ROOT / "app.ico"]
    for p in candidates:
        try:
            if p.exists() and _is_valid_ico_file(p):
                return p.resolve()
            if p.exists():
                repaired = _repair_png_to_ico_if_needed(p)
                if repaired is not None:
                    return repaired
        except Exception:
            continue
    return None


def _try_gui_yes_no(title: str, message: str, *, default: bool = False) -> Optional[bool]:
    if _env_flag("NO_BUILD_DIALOGS", False):
        return None
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return None
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
            root.update_idletasks()
        except Exception:
            pass
        result = messagebox.askyesno(
            title,
            message,
            default=("yes" if default else "no"),
            parent=root,
        )
        return bool(result)
    except Exception:
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _ask_yes_no(title: str, message: str, *, default: bool = False) -> bool:
    gui_result = _try_gui_yes_no(title, message, default=default)
    if gui_result is not None:
        print(f"\n[{title}] {'Yes' if gui_result else 'No'}")
        print(message)
        return bool(gui_result)

    print(f"\n== {title} ==")
    print(message)
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        ans = input(f"Confirm {suffix}: ").strip().lower()
        if not ans:
            return bool(default)
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please type y or n.")


def _ask_reuse_existing_payload(payload_archive: Path) -> bool:
    if not payload_archive.exists():
        return False
    return _ask_yes_no(
        "Reuse Existing Payload",
        (
            "Detected an existing payload archive.\n\n"
            f"{payload_archive}\n\n"
            "Do you want to skip app/uninstaller rebuild + payload repacking,\n"
            "and directly provide/upload-confirm the payload URL to build Setup.exe?"
        ),
        default=False,
    )


def _ask_reuse_existing_inno_core_bundle(
    core_exe: Path,
    core_bins: list[Path],
    *,
    default_core_url: str,
) -> bool:
    if not core_exe.exists():
        return False
    required = [core_exe, *core_bins]
    lines = "\n".join(str(p) for p in required)
    use_url = str(default_core_url or "").strip() or DEFAULT_INNO_CORE_URL_HINT
    return _ask_yes_no(
        "Reuse Existing Inno Core Bundle",
        (
            "Detected an existing Inno Core installer bundle in dist/:\n\n"
            f"{lines}\n\n"
            "Do you want to use the online Inno Core URL below,\n"
            "skip app/uninstaller rebuild + Inno core repack,\n"
            "and only build the lightweight Qt frontend installer?\n\n"
            f"{use_url}"
        ),
        default=True,
    )


def run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _signing_requested() -> bool:
    if _env_flag("ENABLE_CODE_SIGNING", False):
        return True
    return bool(str(os.environ.get("SIGN_CERT_FILE", "")).strip() or str(os.environ.get("SIGN_CERT_SHA1", "")).strip())


def _build_signtool_command(signtool_exe: Path, target: Path) -> list[str]:
    cert_file = str(os.environ.get("SIGN_CERT_FILE", "")).strip().strip('"')
    cert_password = str(os.environ.get("SIGN_CERT_PASSWORD", "")).strip()
    cert_sha1 = str(os.environ.get("SIGN_CERT_SHA1", "")).strip().replace(" ", "")
    subject_name = str(os.environ.get("SIGN_SUBJECT_NAME", "")).strip()
    timestamp_url = str(os.environ.get("SIGN_TIMESTAMP_URL", "")).strip() or "http://timestamp.digicert.com"

    cmd = [str(signtool_exe), "sign", "/fd", "SHA256", "/td", "SHA256", "/tr", timestamp_url]

    if cert_file:
        cmd.extend(["/f", cert_file])
        if cert_password:
            cmd.extend(["/p", cert_password])
    elif cert_sha1:
        cmd.extend(["/sha1", cert_sha1])
        if _env_flag("SIGN_USE_MACHINE_STORE", False):
            cmd.append("/sm")
        if subject_name:
            cmd.extend(["/n", subject_name])
    else:
        raise RuntimeError(
            "Code signing requested but no certificate was configured. "
            "Set SIGN_CERT_FILE (+ SIGN_CERT_PASSWORD) or SIGN_CERT_SHA1."
        )

    if _env_flag("SIGN_APPEND_TIMESTAMP_FALLBACK", False):
        # Placeholder for future customization; no-op to keep cmd shape stable.
        pass

    cmd.append(str(target))
    return cmd


def sign_file_if_enabled(path: Path) -> None:
    path = Path(path)
    if not _signing_requested():
        return
    if not path.exists():
        raise RuntimeError(f"Cannot sign missing file: {path}")
    signtool_exe = find_signtool_exe()
    if signtool_exe is None:
        raise RuntimeError(
            "Code signing requested but signtool.exe was not found. "
            "Install Windows SDK and/or set SIGNTOOL_EXE."
        )
    run(_build_signtool_command(signtool_exe, path))


def compile_inno_wrapper(
    *,
    cfg,
    bootstrapper_inner_exe: Path,
    output_dir: Path,
    output_base_filename: str,
) -> Path:
    iscc_exe = find_iscc_exe()
    if iscc_exe is None:
        raise RuntimeError("ISCC.exe not found. Install Inno Setup or add it to PATH.")

    iss_path = PACKAGING_DIR / "inno" / "bootstrapper_wrapper.iss"
    if not iss_path.exists():
        raise RuntimeError(f"Inno wrapper script not found: {iss_path}")
    if not bootstrapper_inner_exe.exists():
        raise RuntimeError(f"Inner bootstrapper exe not found: {bootstrapper_inner_exe}")

    icon_path = _resolve_icon_path(cfg)
    app_name = str(getattr(cfg, "APP_NAME", "MediaTranscribeStudio"))
    app_publisher = str(os.environ.get("APP_PUBLISHER", "")).strip() or app_name
    app_version = release_version_string()
    inner_name = bootstrapper_inner_exe.name
    if icon_path is None:
        print("[warn] No valid ICO file found for Inno SetupIconFile; wrapper will use default icon.")

    cmd = [
        str(iscc_exe),
        f"/DAppName={app_name}",
        f"/DAppVersion={app_version}",
        f"/DAppPublisher={app_publisher}",
        f"/DBootstrapperPath={str(bootstrapper_inner_exe.resolve())}",
        f"/DInnerBootstrapperName={inner_name}",
        f"/DOutputDir={str(output_dir.resolve())}",
        f"/DOutputBaseFilename={output_base_filename}",
    ]
    if icon_path is not None:
        cmd.append(f"/DSetupIconFile={str(icon_path)}")
    cmd.append(str(iss_path.resolve()))

    run(cmd)
    return output_dir / f"{output_base_filename}.exe"


def compile_inno_main_installer(
    *,
    cfg,
    app_source_dir: Path,
    output_dir: Path,
    output_base_filename: str,
) -> Path:
    iscc_exe = find_iscc_exe()
    if iscc_exe is None:
        raise RuntimeError("ISCC.exe not found. Install Inno Setup or add it to PATH.")

    iss_path = PACKAGING_DIR / "inno" / "main_app_installer.iss"
    if not iss_path.exists():
        raise RuntimeError(f"Inno main installer script not found: {iss_path}")
    if not app_source_dir.exists() or not app_source_dir.is_dir():
        raise RuntimeError(f"App source directory not found for Inno main installer: {app_source_dir}")

    icon_path = _resolve_icon_path(cfg)
    app_name = str(getattr(cfg, "APP_NAME", "MediaTranscribeStudio"))
    app_publisher = str(os.environ.get("APP_PUBLISHER", "")).strip() or app_name
    app_version = release_version_string()
    main_exe_name = str(getattr(cfg, "MAIN_EXE_NAME", f"{app_name}.exe"))
    uninstall_frontend_name = str(getattr(cfg, "UNINSTALL_EXE_NAME", "uninstall.exe"))

    if icon_path is None:
        print("[warn] No valid ICO file found for Inno SetupIconFile; core installer will use default icon.")

    cmd = [
        str(iscc_exe),
        f"/DAppName={app_name}",
        f"/DAppVersion={app_version}",
        f"/DAppPublisher={app_publisher}",
        f"/DAppSourceDir={str(app_source_dir.resolve())}",
        f"/DMainExeName={main_exe_name}",
        f"/DUninstallFrontendExeName={uninstall_frontend_name}",
        f"/DOutputDir={str(output_dir.resolve())}",
        f"/DOutputBaseFilename={output_base_filename}",
    ]
    if icon_path is not None:
        cmd.append(f"/DSetupIconFile={str(icon_path)}")
    cmd.append(str(iss_path.resolve()))

    run(cmd)
    return output_dir / f"{output_base_filename}.exe"


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def human_bytes(num: float) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num:.1f} B"


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            continue
    return total


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _is_conda_env() -> bool:
    if str(os.environ.get("CONDA_PREFIX", "")).strip():
        return True
    try:
        return (Path(sys.prefix) / "conda-meta").exists()
    except Exception:
        return False


def _dedupe_existing_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower()
        if key in seen:
            continue
        seen.add(key)
        if rp.exists():
            out.append(rp)
    return out


def create_tar_zst_payload(src_dir: Path, archive_path: Path) -> None:
    if not src_dir.exists():
        raise RuntimeError(f"Source directory not found: {src_dir}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    tar_exe = shutil.which("tar")
    if not tar_exe:
        raise RuntimeError("tar.exe not found. Windows built-in tar is required to create .tar.zst payloads.")

    # Try the common tar variants in order (Windows bsdtar first, then GNU tar forms).
    candidates = [
        [tar_exe, "-a", "-cf", str(archive_path), "-C", str(src_dir), "."],
        [tar_exe, "--zstd", "-cf", str(archive_path), "-C", str(src_dir), "."],
        [tar_exe, "-I", "zstd", "-cf", str(archive_path), "-C", str(src_dir), "."],
    ]

    last_error = None
    for cmd in candidates:
        print(f"\n>>> {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode == 0 and archive_path.exists() and archive_path.stat().st_size > 0:
            return
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
        last_error = tail[0]
        try:
            if archive_path.exists():
                archive_path.unlink()
        except Exception:
            pass

    raise RuntimeError(
        "Failed to create .tar.zst payload with system tar. "
        "Ensure tar supports zstd (Windows bsdtar `-a`) or install GNU tar + zstd. "
        f"Last error: {last_error}"
    )


def playwright_package_local_browsers_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("playwright")
    except Exception:
        spec = None
    if spec is None or not spec.origin:
        return None
    try:
        pkg_dir = Path(spec.origin).resolve().parent
    except Exception:
        return None
    return pkg_dir / "driver" / "package" / ".local-browsers"


def prepare_bundled_playwright_chromium() -> None:
    if not _env_flag("BUNDLE_PLAYWRIGHT_CHROMIUM", True):
        print("\n== Playwright Chromium bundling disabled by env (BUNDLE_PLAYWRIGHT_CHROMIUM=0) ==")
        return

    browsers_dir = playwright_package_local_browsers_dir()
    if browsers_dir is None:
        print("\n[warn] Playwright package not found; skipping bundled Chromium preparation.")
        return

    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    env.setdefault("PLAYWRIGHT_SKIP_BROWSER_GC", "1")

    print("\n== Preparing bundled Playwright Chromium (package-local .local-browsers) ==")
    run([sys.executable, "-m", "playwright", "install", "chromium"], env=env)

    if not browsers_dir.exists():
        raise RuntimeError(
            "Playwright Chromium install did not create package-local .local-browsers directory. "
            f"Expected: {browsers_dir}"
        )

    browser_subdirs = [p for p in browsers_dir.iterdir() if p.is_dir()]
    if not browser_subdirs:
        raise RuntimeError(f"Playwright .local-browsers exists but is empty: {browsers_dir}")

    print(f"Playwright browsers dir: {browsers_dir}")
    print(f"Playwright browsers size: {human_bytes(dir_size_bytes(browsers_dir))}")


def _bundle_nemo_artifact_names() -> list[str]:
    names = ["diar_msdd_telephonic.nemo"]
    raw = str(os.environ.get("BUNDLE_NEMO_ARTIFACTS", "") or "").strip()
    if raw:
        for part in raw.split(os.pathsep):
            text = str(part or "").strip().strip('"')
            if text and text not in names:
                names.append(text)
    return names


def _inspect_nemo_artifact(path: Path) -> tuple[bool, str]:
    try:
        if not path.exists():
            return False, "missing"
        if not path.is_file():
            return False, "not a file"
        size = int(path.stat().st_size)
    except Exception as e:
        return False, str(e or "stat failed")

    if size <= 0:
        return False, "empty file"

    try:
        if tarfile.is_tarfile(path) or zipfile.is_zipfile(path):
            return True, ""
    except Exception:
        pass

    preview = b""
    try:
        with path.open("rb") as fh:
            preview = fh.read(160)
    except Exception:
        pass

    preview_text = preview.lstrip()[:80].decode("utf-8", errors="ignore").strip()
    if preview.lstrip().startswith((b"{", b"[")):
        return False, f"looks like JSON payload instead of a NeMo archive ({size} bytes)"
    if size < 4096:
        return False, f"file too small to be a NeMo archive ({size} bytes)"
    if preview_text:
        return False, f"unrecognized NeMo archive format ({size} bytes, head={preview_text[:48]!r})"
    return False, f"unrecognized NeMo archive format ({size} bytes)"


def _warn_invalid_nemo_artifact(path: Path, reason: str) -> None:
    key = str(path)
    if key in _INVALID_BUNDLE_NEMO_ARTIFACTS:
        return
    _INVALID_BUNDLE_NEMO_ARTIFACTS.add(key)
    print(f"[warn] Ignoring invalid NeMo artifact {path}: {reason}")


def _discover_nemo_artifact_source(filename: str) -> Path | None:
    name = str(filename or "").strip()
    if not name:
        return None

    home = Path.home()
    raw_roots: list[Path] = [
        ROOT / "output_files" / ".nemo_msdd" / "model_artifacts",
        ROOT / "output_files" / ".nemo_msdd",
        home / ".cache" / "torch" / "NeMo",
    ]
    if os.environ.get("NEMO_CACHE_DIR"):
        raw_roots.append(Path(os.environ["NEMO_CACHE_DIR"]))
    if os.environ.get("NEMO_HOME"):
        raw_roots.append(Path(os.environ["NEMO_HOME"]))
    search_roots = _dedupe_existing_paths(raw_roots)

    extra = str(os.environ.get("BUNDLE_MODEL_CACHE_DIRS", "") or "").strip()
    if extra:
        for part in extra.split(os.pathsep):
            text = str(part or "").strip().strip('"')
            if text:
                search_roots.extend(_dedupe_existing_paths([Path(text)]))

    deduped_roots = _dedupe_existing_paths(search_roots)
    for root in deduped_roots:
        for candidate in (
            root / name,
            root / "model_artifacts" / name,
            root / ".nemo_msdd" / "model_artifacts" / name,
        ):
            try:
                if candidate.exists() and candidate.is_file():
                    valid, reason = _inspect_nemo_artifact(candidate)
                    if valid:
                        return candidate
                    _warn_invalid_nemo_artifact(candidate, reason)
            except Exception:
                continue

        try:
            for candidate in root.rglob(name):
                if candidate.is_file():
                    valid, reason = _inspect_nemo_artifact(candidate)
                    if valid:
                        return candidate
                    _warn_invalid_nemo_artifact(candidate, reason)
        except Exception:
            continue
    return None


def prepare_bundled_nemo_artifacts() -> list[Path]:
    if not _env_flag("BUNDLE_NEMO_ARTIFACTS_ENABLED", True):
        return []

    staged: list[Path] = []
    target_root = ROOT / "output_files" / ".nemo_msdd" / "model_artifacts"
    target_root.mkdir(parents=True, exist_ok=True)

    for filename in _bundle_nemo_artifact_names():
        target = target_root / filename
        try:
            if target.exists() and target.is_file():
                valid, reason = _inspect_nemo_artifact(target)
                if valid:
                    staged.append(target)
                    continue
                print(f"[warn] Replacing invalid staged NeMo artifact {target}: {reason}")
                try:
                    target.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        source = _discover_nemo_artifact_source(filename)
        if source is None:
            continue
        try:
            if target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass
            shutil.copy2(source, target)
            staged.append(target)
            print(f"Staged NeMo artifact for bundle: {source} -> {target}")
        except Exception as e:
            print(f"[warn] Failed to stage NeMo artifact {filename}: {e}")
    return staged


def discover_model_cache_dirs() -> list[Path]:
    home = Path.home()
    localappdata = Path(os.environ.get("LOCALAPPDATA", "")) if os.environ.get("LOCALAPPDATA") else None

    candidates: list[Path] = [
        ROOT / "checkpoints",
        ROOT / "output_files" / ".nemo_msdd",
        home / ".cache" / "huggingface",
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "modelscope",
        home / ".cache" / "modelscope" / "hub",
        home / ".cache" / "torch",
        home / ".cache" / "torch" / "hub",
        home / ".cache" / "torch" / "hub" / "checkpoints",
        home / ".cache" / "torch" / "whisper",
        home / ".cache" / "torch" / "NeMo",
    ]
    if localappdata is not None:
        candidates.extend(
            [
                localappdata / "huggingface",
                localappdata / "modelscope",
                localappdata / "MediaTranscribeStudio",
            ]
        )

    for env_name in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "MODELSCOPE_CACHE",
        "TORCH_HOME",
        "NEMO_CACHE_DIR",
    ):
        raw = str(os.environ.get(env_name, "")).strip()
        if raw:
            candidates.append(Path(raw))

    extra = str(os.environ.get("BUNDLE_MODEL_CACHE_DIRS", "")).strip()
    if extra:
        for part in extra.split(os.pathsep):
            part = str(part or "").strip().strip('"')
            if part:
                candidates.append(Path(part))

    return _dedupe_existing_paths(candidates)


def discover_native_bundle_dirs() -> list[Path]:
    candidates: list[Path] = [
        ROOT / "native" / "install",
        ROOT / "native" / "dist",
        ROOT / "build" / "native" / "install",
        ROOT / "build" / "install",
        ROOT / "build" / "cmake-install",
        ROOT / "dist" / "native",
    ]

    extra = str(os.environ.get("BUNDLE_NATIVE_DIRS", "")).strip()
    if extra:
        for part in extra.split(os.pathsep):
            part = str(part or "").strip().strip('"')
            if part:
                candidates.append(Path(part))

    single = str(os.environ.get("BUNDLE_NATIVE_DIR", "")).strip().strip('"')
    if single:
        candidates.append(Path(single))

    return _dedupe_existing_paths(candidates)


def build_app_pyinstaller_env() -> dict[str, str]:
    env = os.environ.copy()
    runtime_bin = str(Path(sys.executable).resolve().parent)
    path_parts = [part for part in str(env.get("PATH", "")).split(os.pathsep) if part]
    if runtime_bin not in path_parts:
        env["PATH"] = os.pathsep.join([runtime_bin, *path_parts]) if path_parts else runtime_bin
    pythonpath_parts = [part for part in str(env.get("PYTHONPATH", "")).split(os.pathsep) if part]
    root_str = str(ROOT)
    if root_str not in pythonpath_parts:
        env["PYTHONPATH"] = os.pathsep.join([root_str, *pythonpath_parts]) if pythonpath_parts else root_str

    # Always install/playwright cache into package-local path so build_app.spec can collect it.
    env["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    env.setdefault("PLAYWRIGHT_SKIP_BROWSER_GC", "1")

    cache_root = ROOT / ".build-cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    lhotse_home = cache_root / "lhotse"
    for cache_dir in (cache_root, mpl_cache, xdg_cache, lhotse_home):
        cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))
    env.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    env.setdefault("LHOTSE_HOME", str(lhotse_home))
    if sys.platform == "darwin":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        env.setdefault("MTS_TORCH_MPS_BRIDGE", "1")
        env.setdefault("MTS_TORCH_MPS_BRIDGE_TARGET", "15.0.0")
        env.setdefault("MTS_TORCH_MPS_BRIDGE_FUTURE_ONLY", "1")

    ffmpeg_in_env = Path(runtime_bin) / "ffmpeg"
    if ffmpeg_in_env.exists():
        env.setdefault("FFMPEG_BINARY", str(ffmpeg_in_env))
        env.setdefault("IMAGEIO_FFMPEG_EXE", str(ffmpeg_in_env))

    # Default to aggressive bundling; ffmpeg is forced on for packaged releases.
    if "INCLUDE_CHECKPOINTS" not in env:
        env["INCLUDE_CHECKPOINTS"] = "1"
    env["BUNDLE_FFMPEG"] = "1"
    if "INCLUDE_RUNTIME_MODEL_CACHES" not in env:
        # Offline-first default for packaged releases: prefer a larger app bundle
        # over downloading models on the first real user run.
        env["INCLUDE_RUNTIME_MODEL_CACHES"] = "1"
    if "PACK_ALL" not in env:
        # Default ON to reduce missing optional dependency/data issues for
        # NeMo/pyannote/media-asr stacks.
        env["PACK_ALL"] = "1"

    staged_nemo_artifacts = prepare_bundled_nemo_artifacts()
    if staged_nemo_artifacts:
        env["BUNDLED_NEMO_ARTIFACTS"] = os.pathsep.join(str(path) for path in staged_nemo_artifacts)

    native_dirs = discover_native_bundle_dirs()
    if native_dirs:
        existing_native = str(env.get("BUNDLE_NATIVE_DIRS", "")).strip()
        merged_native = [str(p) for p in native_dirs]
        if existing_native:
            merged_native.append(existing_native)
        env["BUNDLE_NATIVE_DIRS"] = os.pathsep.join(merged_native)

    return env


def build_native_helpers() -> list[Path]:
    if not _env_flag("BUILD_NATIVE_HELPERS", True):
        print("\n== Native Helper Build ==")
        print("Disabled by env (BUILD_NATIVE_HELPERS=0).")
        return []

    native_root = ROOT / "native"
    cmake_lists = native_root / "CMakeLists.txt"
    if not cmake_lists.exists():
        return []

    cmake_exe = shutil.which("cmake")
    if not cmake_exe:
        raise RuntimeError("cmake was not found on PATH, but native/CMakeLists.txt exists.")

    build_dir = ROOT / "build" / "native"
    install_dir = build_dir / "install"

    print("\n== Native Helper Build ==")
    print(f"Source : {native_root}")
    print(f"Build  : {build_dir}")
    print(f"Install: {install_dir}")

    run(
        [
            cmake_exe,
            "-S",
            str(native_root),
            "-B",
            str(build_dir),
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
    )
    run([cmake_exe, "--build", str(build_dir), "--config", "Release"])
    run([cmake_exe, "--install", str(build_dir), "--config", "Release"])

    patterns = ["mts_media_helper*"]
    if sys.platform == "darwin":
        patterns.append("libmts_apple_runtime_shim*.dylib")
    elif os.name == "nt":
        patterns = ["mts_media_helper*.exe", "mts_apple_runtime_shim*.dll"]

    outputs: list[Path] = []
    for pattern in patterns:
        outputs.extend(install_dir.rglob(pattern))
    outputs = _dedupe_existing_paths([path for path in outputs if path.is_file()])

    helper_present = any("mts_media_helper" in path.name for path in outputs)
    if not helper_present:
        raise RuntimeError(
            "Native helper build finished, but no installed mts_media_helper artifact was found."
        )
    for path in outputs:
        print(f"OK native helper: {path}")
    return outputs


def print_model_bundle_summary(env: dict[str, str]) -> None:
    enabled = str(env.get("INCLUDE_RUNTIME_MODEL_CACHES", "")).strip().lower() in {"1", "true", "yes", "on"}
    print("\n== Runtime Model Bundling ==")
    print(f"INCLUDE_CHECKPOINTS       : {env.get('INCLUDE_CHECKPOINTS', '')}")
    print(f"INCLUDE_RUNTIME_MODEL_CACHES: {env.get('INCLUDE_RUNTIME_MODEL_CACHES', '')}")
    print(f"PACK_ALL (site-packages mirror): {env.get('PACK_ALL', '')}")
    bundled_nemo = str(env.get("BUNDLED_NEMO_ARTIFACTS", "") or "").strip()
    if bundled_nemo:
        print("Bundled NeMo artifacts:")
        for part in bundled_nemo.split(os.pathsep):
            text = str(part or "").strip()
            if text:
                print(f"  {text}")
    if not enabled:
        print("Model cache bundling disabled (models will download online at runtime).")
        return

    cache_dirs = discover_model_cache_dirs()
    if not cache_dirs:
        print("No existing model cache directories detected.")
        return

    total = 0
    for p in cache_dirs:
        size = dir_size_bytes(p)
        total += size
        print(f"  {human_bytes(size):>10}  {p}")
    print(f"  {'-' * 10}")
    print(f"  {human_bytes(total):>10}  total candidate model/cache data")


def print_native_bundle_summary(env: dict[str, str]) -> None:
    print("\n== Native Bundle Inputs ==")
    raw = str(env.get("BUNDLE_NATIVE_DIRS", "")).strip()
    if not raw:
        print("No CMake/native artifact directories detected.")
        return

    seen: set[str] = set()
    dirs: list[Path] = []
    for part in raw.split(os.pathsep):
        text = str(part or "").strip().strip('"')
        if not text:
            continue
        path = Path(text)
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved).lower()
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        dirs.append(resolved)

    if not dirs:
        print("No existing native bundle directories matched the configured env.")
        return

    total = 0
    for path in dirs:
        size = dir_size_bytes(path)
        total += size
        print(f"  {human_bytes(size):>10}  {path}")
    print(f"  {'-' * 10}")
    print(f"  {human_bytes(total):>10}  total native/CMake bundle input")


def validate_bundled_app_contents(app_dir: Path, *, require_model_caches: bool) -> None:
    internal = app_dir / "_internal"
    missing: list[str] = []
    required_paths = [
        internal / "playwright" / "driver" / "package",
        internal / "playwright" / "driver" / "package" / ".local-browsers",
        internal / "nemo",
        internal / "lightning_fabric" / "version.info",
        internal / "pyannote",
        internal / "omegaconf",
        internal / "IPython",
    ]
    if require_model_caches:
        required_paths.append(internal / "model_caches")

    for p in required_paths:
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise RuntimeError(
            "Bundled app validation failed; required packaged components are missing:\n- "
            + "\n- ".join(missing)
        )

    print("\n== Bundle Validation ==")
    for p in required_paths:
        if p.is_dir():
            size = dir_size_bytes(p)
            print(f"OK {p} ({human_bytes(size)})")
        else:
            print(f"OK {p}")

    if require_model_caches:
        expected_model_entries: list[Path] = []
        for repo in str(os.environ.get("BUNDLE_HF_MODEL_REPOS", "") or "").split(os.pathsep):
            rel = _hf_repo_cache_dir(repo)
            if rel is not None:
                expected_model_entries.append(
                    internal / "model_caches" / "huggingface" / "hub" / rel
                )
        for rel in _split_env_paths(os.environ.get("BUNDLE_MODELSCOPE_CACHE_SUBDIRS", "")):
            expected_model_entries.append(
                internal / "model_caches" / "modelscope" / rel
            )
        for rel in _split_env_paths(os.environ.get("BUNDLE_TORCH_CACHE_SUBDIRS", "")):
            expected_model_entries.append(
                internal / "model_caches" / "torch" / rel
            )
        for rel in _split_env_paths(os.environ.get("BUNDLE_NEMO_CACHE_SUBDIRS", "")):
            expected_model_entries.append(
                internal / "model_caches" / "nemo" / rel
            )

        missing_models = [str(path) for path in expected_model_entries if not path.exists()]
        if missing_models:
            raise RuntimeError(
                "Bundled app validation failed; required packaged model caches are missing:\n- "
                + "\n- ".join(missing_models[:40])
            )
        if expected_model_entries:
            print("\n== Model Cache Validation ==")
            for path in expected_model_entries:
                if path.is_dir():
                    print(f"OK {path} ({human_bytes(dir_size_bytes(path))})")
                else:
                    print(f"OK {path}")

    ffmpeg_candidates = [
        internal / "tools" / "ffmpeg" / "ffmpeg.exe",
        internal / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        internal / "ffmpeg" / "ffmpeg.exe",
        internal / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    ffprobe_candidates = [
        internal / "tools" / "ffmpeg" / "ffprobe.exe",
        internal / "tools" / "ffmpeg" / "bin" / "ffprobe.exe",
        internal / "ffmpeg" / "ffprobe.exe",
        internal / "ffmpeg" / "bin" / "ffprobe.exe",
    ]
    ffmpeg_path = next((p for p in ffmpeg_candidates if p.exists()), None)
    ffprobe_path = next((p for p in ffprobe_candidates if p.exists()), None)

    print("\n== External Tools (ffmpeg) ==")
    if ffmpeg_path:
        print(f"OK ffmpeg  : {ffmpeg_path}")
    else:
        raise RuntimeError(
            "Bundled app validation failed; required ffmpeg binary is missing from the packaged app."
        )
    if ffprobe_path:
        print(f"OK ffprobe : {ffprobe_path}")
    else:
        print("WARN ffprobe: not bundled (media duration/probe helpers will rely on system install)")

    native_candidates = [
        internal / "native",
        internal / "tools" / "native",
    ]
    native_root = next((p for p in native_candidates if p.exists()), None)
    print("\n== Native Runtime ==")
    if native_root is not None:
        print(f"OK native   : {native_root} ({human_bytes(dir_size_bytes(native_root))})")
    else:
        print("INFO native : no packaged native/CMake helper directory detected")


def _derive_inno_core_split_urls(core_url: str, *, count: int = 3) -> list[str]:
    raw = str(core_url or "").strip()
    if not raw:
        return []
    try:
        parsed = urllib.parse.urlparse(raw)
        path = parsed.path or ""
        slash = path.rfind("/")
        filename = path[slash + 1 :] if slash >= 0 else path
        stem = Path(filename).stem
        if not stem:
            return []
        prefix = path[: slash + 1] if slash >= 0 else ""
        return [
            urllib.parse.urlunparse(parsed._replace(path=f"{prefix}{stem}-{idx}.bin"))
            for idx in range(1, max(0, int(count)) + 1)
        ]
    except Exception:
        return []


def _list_inno_core_bin_files(core_exe: Path) -> list[Path]:
    stem = core_exe.stem
    out: list[tuple[int, Path]] = []
    for p in core_exe.parent.glob(f"{stem}-*.bin"):
        if not p.is_file():
            continue
        m = re.match(rf"^{re.escape(stem)}-(\d+)\.bin$", p.name, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        out.append((idx, p))
    out.sort(key=lambda item: item[0])
    return [p for _idx, p in out]


def _remove_inno_core_bin_files(core_exe: Path) -> int:
    removed = 0
    for p in _list_inno_core_bin_files(core_exe):
        try:
            p.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def update_dist_config_defaults(*, url: str, sha256: str, inno_core_bin_urls: Optional[list[str]] = None) -> bool:
    text = DIST_CONFIG_PATH.read_text(encoding="utf-8")
    new_text, url_count = re.subn(
        r'^DEFAULT_PAYLOAD_URL = .*?$',
        f'DEFAULT_PAYLOAD_URL = {json.dumps(url, ensure_ascii=False)}',
        text,
        flags=re.M,
    )
    new_text, sha_count = re.subn(
        r'^DEFAULT_PAYLOAD_SHA256 = .*?$',
        f'DEFAULT_PAYLOAD_SHA256 = {json.dumps(sha256.upper(), ensure_ascii=False)}',
        new_text,
        flags=re.M,
    )
    bin_count = 0
    if inno_core_bin_urls is not None:
        clean_bin_urls = [str(u).strip() for u in (inno_core_bin_urls or []) if str(u).strip()]
        new_text, bin_count = re.subn(
            r'^DEFAULT_INNO_CORE_BIN_URLS = .*?$',
            f'DEFAULT_INNO_CORE_BIN_URLS = {json.dumps(clean_bin_urls, ensure_ascii=False)}',
            new_text,
            flags=re.M,
        )
    if url_count < 1 or sha_count < 1 or (inno_core_bin_urls is not None and bin_count < 1):
        raise RuntimeError(
            "Failed to update dist_config.py (patterns not found): "
            f"url_matches={url_count}, sha_matches={sha_count}, bin_matches={bin_count}"
        )
    if new_text == text:
        # Values are already current; treat as success.
        return False
    DIST_CONFIG_PATH.write_text(new_text, encoding="utf-8")
    return True


def update_dist_config_macos_defaults(*, installer_url: str, installer_sha256: str) -> bool:
    text = DIST_CONFIG_PATH.read_text(encoding="utf-8")
    new_text, url_count = re.subn(
        r'^DEFAULT_MACOS_INSTALLER_URL = [^\n]*(?:\n[ \t]+".*")*(?:\n\))?',
        f'DEFAULT_MACOS_INSTALLER_URL = {json.dumps(installer_url, ensure_ascii=False)}',
        text,
        flags=re.M,
    )
    new_text, sha_count = re.subn(
        r'^DEFAULT_MACOS_INSTALLER_SHA256 = .*?$',
        f'DEFAULT_MACOS_INSTALLER_SHA256 = {json.dumps(installer_sha256.upper(), ensure_ascii=False)}',
        new_text,
        flags=re.M,
    )
    if url_count < 1 or sha_count < 1:
        raise RuntimeError(
            "Failed to update dist_config.py macOS defaults "
            f"(patterns not found): url_matches={url_count}, sha_matches={sha_count}"
        )
    if new_text == text:
        return False
    DIST_CONFIG_PATH.write_text(new_text, encoding="utf-8")
    return True


def prompt_url() -> str:
    env_override = str(os.environ.get("PAYLOAD_URL_OVERRIDE", "")).strip()
    if env_override:
        print("\n== Using PAYLOAD_URL_OVERRIDE from environment ==")
        print(env_override)
        return env_override
    print("\n==== 需要你上传 payload tar.zst ====")
    print("上传完成后，把下载链接粘贴到下面，然后回车继续构建安装器。")
    print("直接回车会取消后续步骤（不会构建安装器）。")
    return input("Payload URL: ").strip()


def prompt_url_with_default(
    *,
    default_url: str = "",
    accept_empty_default: bool = False,
    env_override_name: str = "PAYLOAD_URL_OVERRIDE",
    label: str = "Payload URL",
) -> str:
    env_override = str(os.environ.get(env_override_name, "")).strip()
    if env_override:
        print(f"\n== Using {env_override_name} from environment ==")
        print(env_override)
        return env_override

    default_url = str(default_url or "").strip()
    print(f"\n==== Upload / Confirm {label} ====")
    print(f"Paste {label} and press Enter.")
    if default_url:
        if accept_empty_default:
            print("Directly press Enter to use the configured default URL.")
        else:
            print("Directly press Enter to trigger a confirmation dialog for the configured default URL.")
    print("Type q and press Enter to cancel.")

    while True:
        raw = input("Payload URL: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return ""
        if raw:
            return raw
        if not default_url:
            return ""
        if accept_empty_default:
            return default_url
        if _ask_yes_no(
            f"Use Default {label}",
            f"将使用以下链接：\n\n{default_url}\n\n是否同意？",
            default=True,
        ):
            return default_url


def _main_qt_frontend_inno_core() -> int:
    cfg = load_dist_config()
    app_name = str(cfg.APP_NAME)
    main_exe_name = str(cfg.MAIN_EXE_NAME)
    uninstaller_name = str(cfg.UNINSTALL_EXE_NAME)
    bootstrapper_name = str(cfg.BOOTSTRAPPER_EXE_NAME)

    app_dir = DIST_DIR / app_name
    app_main_exe = app_dir / main_exe_name
    uninstaller_exe = DIST_DIR / uninstaller_name
    frontend_setup_exe = DIST_DIR / bootstrapper_name
    inno_core_setup_exe = DIST_DIR / f"{Path(bootstrapper_name).stem}-Core.exe"
    inno_core_bin_files = _list_inno_core_bin_files(inno_core_setup_exe)

    pyi = pyinstaller_cmd()
    build_native_helpers()
    app_build_env = build_app_pyinstaller_env()
    discovered_cache_dirs = discover_model_cache_dirs()
    signing_requested = _signing_requested()
    build_inno_core = _env_flag("BUILD_INNO_CORE_INSTALLER", True)
    embed_inno_core = _env_flag("EMBED_INNO_CORE_IN_FRONTEND", False)
    default_installer_core_url = str(getattr(cfg, "DEFAULT_PAYLOAD_URL", "") or "").strip()
    confirmed_reuse_core_url = ""
    if inno_core_setup_exe.exists():
        default_installer_core_url = DEFAULT_INNO_CORE_URL_HINT
        reuse_existing_inno_core_bundle = _ask_reuse_existing_inno_core_bundle(
            inno_core_setup_exe,
            inno_core_bin_files,
            default_core_url=default_installer_core_url,
        )
        if reuse_existing_inno_core_bundle:
            confirmed_reuse_core_url = default_installer_core_url
    else:
        reuse_existing_inno_core_bundle = False

    print("== One-Click Build Start (Qt Frontend + Inno Core) ==")
    print(f"Repo root: {ROOT}")
    print(f"Inno core build  : {'enabled' if build_inno_core else 'disabled'}")
    print(f"Embed core in Qt : {'yes' if embed_inno_core else 'no (online download expected)'}")
    print(f"Reuse core bundle : {'yes' if reuse_existing_inno_core_bundle else 'no'}")
    print(f"Code signing     : {'enabled' if signing_requested else 'disabled'}")
    print_model_bundle_summary(app_build_env)
    print_native_bundle_summary(app_build_env)

    if not reuse_existing_inno_core_bundle:
        prepare_bundled_playwright_chromium()

        run([*pyi, "packaging/build_uninstaller.spec", "--noconfirm"])
        run([*pyi, "packaging/build_app.spec", "--noconfirm"], env=app_build_env)

        if not uninstaller_exe.exists():
            raise RuntimeError(f"Uninstaller build output not found: {uninstaller_exe}")
        if not app_dir.exists():
            raise RuntimeError(f"Main app build output not found: {app_dir}")
        if not app_main_exe.exists():
            raise RuntimeError(f"Main app executable not found: {app_main_exe}")

        sign_file_if_enabled(uninstaller_exe)
        sign_file_if_enabled(app_main_exe)

        validate_bundled_app_contents(
            app_dir,
            require_model_caches=(
                str(app_build_env.get("INCLUDE_RUNTIME_MODEL_CACHES", "")).strip().lower()
                in {"1", "true", "yes", "on"}
                and bool(discovered_cache_dirs)
            ),
        )

        target_uninstaller = app_dir / uninstaller_name
        print(f"\n>>> Copy {uninstaller_exe} -> {target_uninstaller}")
        shutil.copy2(uninstaller_exe, target_uninstaller)

        if build_inno_core:
            if inno_core_setup_exe.exists():
                try:
                    inno_core_setup_exe.unlink()
                except Exception:
                    pass
            removed_old_bins = _remove_inno_core_bin_files(inno_core_setup_exe)
            if removed_old_bins > 0:
                print(f">>> Removed {removed_old_bins} stale split .bin file(s) before rebuild.")
            built_core = compile_inno_main_installer(
                cfg=cfg,
                app_source_dir=app_dir,
                output_dir=DIST_DIR,
                output_base_filename=Path(inno_core_setup_exe).stem,
            )
            if not built_core.exists():
                raise RuntimeError(f"Inno core installer build output not found: {built_core}")
            inno_core_setup_exe = built_core
            inno_core_bin_files = _list_inno_core_bin_files(inno_core_setup_exe)
            sign_file_if_enabled(inno_core_setup_exe)
            print(f"\n>>> Inno core installer built: {inno_core_setup_exe}")
            if inno_core_bin_files:
                print(f">>> Split bin files detected: {len(inno_core_bin_files)}")
                for p in inno_core_bin_files:
                    print(f"    split: {p}")
            else:
                print(">>> No split .bin files detected for this core installer.")
        else:
            if not inno_core_setup_exe.exists():
                raise RuntimeError(
                    "BUILD_INNO_CORE_INSTALLER=0 but no existing Inno core installer was found at "
                    f"{inno_core_setup_exe}"
                )
            inno_core_bin_files = _list_inno_core_bin_files(inno_core_setup_exe)
            print(f"\n>>> Reusing existing Inno core installer: {inno_core_setup_exe}")
    else:
        required = [inno_core_setup_exe, *inno_core_bin_files]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError("Selected reuse mode for Inno core bundle, but files are missing:\n- " + "\n- ".join(missing))
        print(f"\n>>> Reusing existing Inno core bundle from dist/: {inno_core_setup_exe}")
        if inno_core_bin_files:
            for p in inno_core_bin_files:
                print(f"    split: {p}")
        else:
            print("    split: <none>")

    bootstrap_env = os.environ.copy()
    if embed_inno_core:
        bootstrap_env["INNO_CORE_SETUP_PATH"] = str(inno_core_setup_exe.resolve())
    else:
        # Keep frontend lightweight: download Inno core (and any split .bin parts) online at install time.
        if confirmed_reuse_core_url:
            url = confirmed_reuse_core_url
            # Use only the fixed core URL; let installer auto-probe online split .bin files.
            inno_core_bin_urls: list[str] = []
            print("\n>>> Reuse confirmed: using fixed online Inno core URL (no local app/core rebuild).")
            print(f"    core: {url}")
            print("    split: <auto-probe online at install time>")
        else:
            url = prompt_url_with_default(
                default_url=default_installer_core_url,
                accept_empty_default=True,
            )
            if not url:
                print("\nCancelled: no online Inno core URL provided, frontend installer was not built.")
                return 0
            split_count = len(inno_core_bin_files)
            inno_core_bin_urls = _derive_inno_core_split_urls(url, count=split_count)
            if inno_core_bin_urls:
                print(f"\n>>> Using {1 + len(inno_core_bin_urls)} installer source links "
                      f"(core exe + {len(inno_core_bin_urls)} split .bin files):")
                print(f"    core: {url}")
                for idx, bin_url in enumerate(inno_core_bin_urls, start=1):
                    print(f"    bin{idx}: {bin_url}")
            else:
                print("\n>>> Using 1 installer source link (core exe only, no split .bin files).")
        updated_defaults = update_dist_config_defaults(url=url, sha256="", inno_core_bin_urls=inno_core_bin_urls)
        if updated_defaults:
            print(f"\n>>> Updated default installer URL in {DIST_CONFIG_PATH}")
        else:
            print(f"\n>>> Default installer URL already up to date in {DIST_CONFIG_PATH}")

    run([*pyi, "packaging/build_bootstrapper.spec", "--noconfirm"], env=bootstrap_env)

    if not frontend_setup_exe.exists():
        raise RuntimeError(f"Qt frontend installer build output not found: {frontend_setup_exe}")

    sign_file_if_enabled(frontend_setup_exe)

    print("\n== Build Complete (Qt Frontend + Inno Core) ==")
    print(f"App dir (source)     : {app_dir.resolve()}")
    print(f"Inno Core Installer  : {inno_core_setup_exe.resolve()}")
    if not embed_inno_core:
        print("Frontend Mode        : online download Inno core from configured URL")
    print(f"Setup.exe (Qt UI)    : {frontend_setup_exe.resolve()}")
    return 0


def main() -> int:
    if _env_flag("USE_QT_FRONTEND_INNO_CORE", True):
        return _main_qt_frontend_inno_core()

    cfg = load_dist_config()
    app_name = str(cfg.APP_NAME)
    main_exe_name = str(cfg.MAIN_EXE_NAME)
    uninstaller_name = str(cfg.UNINSTALL_EXE_NAME)
    bootstrapper_name = str(cfg.BOOTSTRAPPER_EXE_NAME)
    payload_archive_name = str(cfg.PAYLOAD_ARCHIVE_NAME)

    app_dir = DIST_DIR / app_name
    app_main_exe = app_dir / main_exe_name
    uninstaller_exe = DIST_DIR / uninstaller_name
    payload_archive = DIST_DIR / payload_archive_name
    bootstrapper_exe = DIST_DIR / bootstrapper_name
    bootstrapper_inner_exe = DIST_DIR / f"{Path(bootstrapper_name).stem}-Inner.exe"
    final_setup_exe = bootstrapper_exe
    default_payload_url = str(getattr(cfg, "DEFAULT_PAYLOAD_URL", "") or "").strip()

    pyi = pyinstaller_cmd()
    build_native_helpers()
    app_build_env = build_app_pyinstaller_env()
    discovered_cache_dirs = discover_model_cache_dirs()
    build_inno_wrapper = _env_flag("BUILD_INNO_WRAPPER", True)
    signing_requested = _signing_requested()
    reuse_existing_payload = _ask_reuse_existing_payload(payload_archive)

    print("== One-Click Build Start ==")
    print(f"Repo root: {ROOT}")
    print(f"Inno wrapper build: {'enabled' if build_inno_wrapper else 'disabled'}")
    print(f"Code signing     : {'enabled' if signing_requested else 'disabled'}")
    print(f"Reuse payload    : {'yes' if reuse_existing_payload else 'no'}")
    print_model_bundle_summary(app_build_env)
    print_native_bundle_summary(app_build_env)

    if not reuse_existing_payload:
        prepare_bundled_playwright_chromium()

        run([*pyi, "packaging/build_uninstaller.spec", "--noconfirm"])
        run([*pyi, "packaging/build_app.spec", "--noconfirm"], env=app_build_env)

        if not uninstaller_exe.exists():
            raise RuntimeError(f"Uninstaller build output not found: {uninstaller_exe}")
        if not app_dir.exists():
            raise RuntimeError(f"Main app build output not found: {app_dir}")
        if not app_main_exe.exists():
            raise RuntimeError(f"Main app executable not found: {app_main_exe}")

        # Sign user-facing executables before they are copied/archived.
        sign_file_if_enabled(uninstaller_exe)
        sign_file_if_enabled(app_main_exe)

        validate_bundled_app_contents(
            app_dir,
            require_model_caches=(
                str(app_build_env.get("INCLUDE_RUNTIME_MODEL_CACHES", "")).strip().lower()
                in {"1", "true", "yes", "on"}
                and bool(discovered_cache_dirs)
            ),
        )

        target_uninstaller = app_dir / uninstaller_name
        print(f"\n>>> Copy {uninstaller_exe} -> {target_uninstaller}")
        shutil.copy2(uninstaller_exe, target_uninstaller)

        print(f"\n>>> Create payload archive: {app_dir} -> {payload_archive}")
        create_tar_zst_payload(app_dir, payload_archive)
    else:
        print(f"\n>>> Reusing existing payload archive: {payload_archive}")
        if not payload_archive.exists():
            raise RuntimeError(f"Selected reuse mode, but payload archive not found: {payload_archive}")

    digest = sha256sum(payload_archive)
    archive_size = payload_archive.stat().st_size

    print("\n== Payload Ready ==")
    print(f"Archive path : {payload_archive.resolve()}")
    print(f"Archive size : {archive_size / (1024 * 1024):.2f} MB")
    print(f"SHA256   : {digest}")

    url = prompt_url_with_default(default_url=default_payload_url)
    if not url:
        print("\n已取消：未提供链接，安装器未构建。")
        print(f"你可以稍后手动更新 {DIST_CONFIG_PATH} 后再运行 PyInstaller 构建安装器。")
        return 0

    updated_defaults = update_dist_config_defaults(url=url, sha256=digest)
    if updated_defaults:
        print(f"\n>>> Updated defaults in {DIST_CONFIG_PATH}")
    else:
        print(f"\n>>> Defaults already up to date in {DIST_CONFIG_PATH}")

    run([*pyi, "packaging/build_bootstrapper.spec", "--noconfirm"])
    if not bootstrapper_exe.exists():
        raise RuntimeError(f"Bootstrapper build output not found: {bootstrapper_exe}")

    # Sign the UI bootstrapper first; if we wrap with Inno, we keep a signed inner copy.
    sign_file_if_enabled(bootstrapper_exe)

    if build_inno_wrapper:
        print(f"\n>>> Preserve inner bootstrapper UI exe: {bootstrapper_exe} -> {bootstrapper_inner_exe}")
        shutil.copy2(bootstrapper_exe, bootstrapper_inner_exe)
        wrapped = compile_inno_wrapper(
            cfg=cfg,
            bootstrapper_inner_exe=bootstrapper_inner_exe,
            output_dir=DIST_DIR,
            output_base_filename=Path(bootstrapper_name).stem,
        )
        sign_file_if_enabled(wrapped)
        final_setup_exe = wrapped
        print(f"\n>>> Inno wrapper built: {wrapped}")
    else:
        final_setup_exe = bootstrapper_exe

    print("\n== Build Complete ==")
    print(f"Payload     : {payload_archive.resolve()}")
    print(f"SHA256      : {digest}")
    if build_inno_wrapper:
        print(f"Inner UI installer : {bootstrapper_inner_exe.resolve()}")
    print(f"Setup.exe          : {final_setup_exe.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
