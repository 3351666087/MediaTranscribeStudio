from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import queue
import random
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import tkinter as tk_types

try:
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import filedialog, messagebox, ttk
    _TK_IMPORT_ERROR = None
except Exception as _e:
    # Qt bootstrapper reuses the backend in this module; keep tkinter optional so
    # the Qt build can exclude Tcl/Tk runtime files.
    tk = None  # type: ignore[assignment]
    tkfont = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR = _e


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dist_config import (  # noqa: E402
    APP_NAME,
    DEFAULT_INSTALL_DIR,
    DEFAULT_PAYLOAD_SHA256,
    DEFAULT_PAYLOAD_URL,
    MAIN_EXE_NAME,
    PAYLOAD_ARCHIVE_NAME,
    UNINSTALL_EXE_NAME,
)


MANIFEST_NAME = "install_manifest.json"
CHUNK_SIZE = 8 * 1024 * 1024
SHORTCUT_TIMEOUT_SEC = 20
RANGE_IO_CHUNK_SIZE = 4 * 1024 * 1024
DOWNLOAD_RETRIES = 3
DOWNLOAD_PART_RETRIES = 4
DOWNLOAD_PROGRESS_EMIT_INTERVAL_SEC = 0.15
DOWNLOAD_PARALLEL_THRESHOLD = 64 * 1024 * 1024
DOWNLOAD_MIN_PIECE_SIZE = 32 * 1024 * 1024
DOWNLOAD_TARGET_PIECE_SIZE = 192 * 1024 * 1024
DOWNLOAD_MAX_CONNECTIONS = 20
DOWNLOAD_URL_TIMEOUT_SEC = 60.0
DOWNLOAD_RESUME_FLUSH_INTERVAL_SEC = 2.0
DOWNLOAD_RESUME_FLUSH_SEGMENTS = 8
DYNAMIC_SPLIT_MIN_REMAINING = 24 * 1024 * 1024
DYNAMIC_SPLIT_MIN_TAIL = 8 * 1024 * 1024
DOWNLOAD_RETRY_BACKOFF_CAP_SEC = 180.0
HF_OFFICIAL_HOST = "huggingface.co"
HF_MIRROR_HOST = "hf-mirror.com"
HF_GOOGLE_PROBE_HOSTS = ("www.google.com", "www.gstatic.com", "google.com")
HF_ROUTE_PROBE_TIMEOUT_SEC = 1.4
HF_REGION_CN = "cn"
HF_REGION_GLOBAL = "global"
HF_REGION_UNKNOWN = "unknown"
_HF_ROUTE_DECISION_LOCK = threading.Lock()
_HF_ROUTE_DECISION_CACHE: Optional[tuple[tuple[str, str], str]] = None


class InstallerCancelled(RuntimeError):
    pass


def _windows_hidden_subprocess_kwargs() -> dict:
    if os.name != "nt":
        return {}
    kwargs = {
        "creationflags": 0x08000000,  # CREATE_NO_WINDOW
    }
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = startupinfo
    except Exception:
        pass
    return kwargs


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_bytes(num: float) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num:.1f} B"


_GENERIC_INSTALL_PARENT_NAMES = {
    "app",
    "apps",
    "application",
    "applications",
    "program",
    "programs",
    "program files",
    "program files (x86)",
    "software",
    "tools",
    "utilities",
    "utils",
    "desktop",
    "documents",
    "downloads",
}


def normalize_user_path_text(value: object) -> str:
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    try:
        raw = str(Path(raw).expanduser())
    except Exception:
        pass
    try:
        raw = os.path.normpath(raw)
    except Exception:
        pass
    return raw


def normalized_path_key(value: object) -> str:
    raw = normalize_user_path_text(value)
    if os.name == "nt":
        raw = os.path.normcase(raw)
    return raw


def _paths_equal(a: object, b: object) -> bool:
    return normalized_path_key(a) == normalized_path_key(b)


def _is_drive_or_share_root(path: Path) -> bool:
    try:
        return path.parent == path
    except Exception:
        return False


def _known_install_container_keys() -> set[str]:
    out: set[str] = set()
    for raw in [
        os.environ.get("LOCALAPPDATA"),
        os.environ.get("APPDATA"),
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramData"),
        os.environ.get("PUBLIC"),
        os.environ.get("USERPROFILE"),
        os.environ.get("WINDIR"),
        tempfile.gettempdir(),
        str(DEFAULT_INSTALL_DIR.parent),
        str(Path.home()),
        str(Path.home() / "Desktop"),
        str(Path.home() / "Documents"),
        str(Path.home() / "Downloads"),
    ]:
        key = normalized_path_key(raw)
        if key:
            out.add(key)
    return out


def looks_like_existing_install_dir(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
    except Exception:
        return False
    for name in (MANIFEST_NAME, MAIN_EXE_NAME, UNINSTALL_EXE_NAME):
        try:
            if (path / name).exists():
                return True
        except Exception:
            continue
    return False


def _is_generic_install_parent_dir(path: Path) -> bool:
    if _is_drive_or_share_root(path):
        return True
    name = (path.name or "").strip().strip("\\/").lower()
    if name in _GENERIC_INSTALL_PARENT_NAMES:
        return True
    if normalized_path_key(path) in _known_install_container_keys():
        return True
    return False


def suggest_install_dir_from_folder_pick(
    selected_dir: Path,
    *,
    current_install_dir: Optional[Path] = None,
) -> Path:
    selected_text = normalize_user_path_text(selected_dir)
    selected = Path(selected_text or str(selected_dir)).expanduser()
    if not str(selected).strip():
        return selected

    if _paths_equal(selected.name, APP_NAME):
        return selected
    if looks_like_existing_install_dir(selected):
        return selected
    if current_install_dir is not None:
        try:
            if _paths_equal(selected, current_install_dir.parent):
                return selected / APP_NAME
        except Exception:
            pass
    if _is_generic_install_parent_dir(selected):
        return selected / APP_NAME
    return selected


def validate_install_dir_choice(install_dir: Path) -> Optional[str]:
    candidate_text = normalize_user_path_text(install_dir)
    if not candidate_text:
        return "Install directory cannot be empty."
    candidate = Path(candidate_text).expanduser()
    if _is_drive_or_share_root(candidate):
        return (
            f"Refusing to install into a drive root:\n{candidate}\n\n"
            f"Please choose a dedicated folder such as:\n{candidate / APP_NAME}"
        )
    if normalized_path_key(candidate) in _known_install_container_keys():
        return (
            f"Refusing to install into a system/container folder:\n{candidate}\n\n"
            f"Please choose a dedicated app folder (for example: {candidate / APP_NAME})."
        )
    return None


def _env_int(name: str, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except Exception:
        return int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return int(value)


def _env_float(
    name: str,
    default: float,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return float(value)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _prefer_hf_mirror_enabled() -> bool:
    return _env_flag("PREFER_HF_MIRROR", True)


def _forced_hf_region() -> str:
    raw = str(os.environ.get("BOOTSTRAP_HF_REGION", "") or "").strip().lower()
    if raw in {"cn", "china", "domestic", "mainland"}:
        return HF_REGION_CN
    if raw in {"global", "intl", "international", "overseas", "foreign", "world"}:
        return HF_REGION_GLOBAL
    return HF_REGION_UNKNOWN


def _url_host(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(str(url or ""))
        host = str(parsed.netloc or "").rsplit("@", 1)[-1]
        if ":" in host:
            host = host.split(":", 1)[0]
        return host.strip().lower()
    except Exception:
        return ""


def _swap_hf_host_url(url: str) -> Optional[str]:
    raw = str(url or "").strip()
    if not raw:
        return None
    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return None
    host = _url_host(raw)
    if host not in {HF_OFFICIAL_HOST, HF_MIRROR_HOST}:
        return None
    replacement = HF_MIRROR_HOST if host == HF_OFFICIAL_HOST else HF_OFFICIAL_HOST
    netloc = str(parsed.netloc or "")
    if not netloc:
        return None

    prefix = ""
    host_port = netloc
    if "@" in host_port:
        prefix, host_port = host_port.rsplit("@", 1)
    port = ""
    if ":" in host_port:
        _host_only, port = host_port.split(":", 1)
    new_host_port = replacement if not port else f"{replacement}:{port}"
    new_netloc = f"{prefix}@{new_host_port}" if prefix else new_host_port
    return urllib.parse.urlunparse(parsed._replace(netloc=new_netloc))


def _probe_tcp_latency(host: str, *, timeout_sec: float) -> Optional[float]:
    target = str(host or "").strip()
    if not target:
        return None
    start = time.perf_counter()
    sock = None
    try:
        sock = socket.create_connection((target, 443), timeout=max(0.2, float(timeout_sec)))
        latency = max(0.001, time.perf_counter() - start)
        return float(latency)
    except Exception:
        return None
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


def _probe_tcp_latency_best(host: str, *, timeout_sec: float, attempts: int) -> Optional[float]:
    tries = max(1, min(5, int(attempts)))
    best: Optional[float] = None
    for _ in range(tries):
        latency = _probe_tcp_latency(host, timeout_sec=timeout_sec)
        if latency is None:
            continue
        if best is None or latency < best:
            best = latency
    return best


def _probe_google_connectivity(timeout_sec: float, *, attempts: int = 1) -> tuple[bool, Optional[float], str]:
    for host in HF_GOOGLE_PROBE_HOSTS:
        latency = _probe_tcp_latency_best(host, timeout_sec=timeout_sec, attempts=attempts)
        if latency is not None:
            return True, float(latency), host
    return False, None, ""


def _decide_hf_host_priority() -> tuple[tuple[str, str], str]:
    global _HF_ROUTE_DECISION_CACHE
    with _HF_ROUTE_DECISION_LOCK:
        if _HF_ROUTE_DECISION_CACHE is not None:
            return _HF_ROUTE_DECISION_CACHE

        default_order: tuple[str, str]
        if _prefer_hf_mirror_enabled():
            default_order = (HF_MIRROR_HOST, HF_OFFICIAL_HOST)
        else:
            default_order = (HF_OFFICIAL_HOST, HF_MIRROR_HOST)

        forced_region = _forced_hf_region()
        if forced_region in {HF_REGION_CN, HF_REGION_GLOBAL}:
            chosen = (
                [HF_MIRROR_HOST, HF_OFFICIAL_HOST]
                if forced_region == HF_REGION_CN
                else [HF_OFFICIAL_HOST, HF_MIRROR_HOST]
            )
            note = (
                f"HF route priority: {chosen[0]} -> {chosen[1]} "
                f"(environment={forced_region}, source=BOOTSTRAP_HF_REGION)"
            )
            _HF_ROUTE_DECISION_CACHE = ((chosen[0], chosen[1]), note)
            return _HF_ROUTE_DECISION_CACHE

        if not _env_flag("BOOTSTRAP_HF_AUTO_ROUTE", True):
            note = (
                "HF route auto-probe disabled; "
                f"priority={default_order[0]} -> {default_order[1]} "
                f"(environment={HF_REGION_UNKNOWN}, source=default-order)"
            )
            _HF_ROUTE_DECISION_CACHE = (default_order, note)
            return _HF_ROUTE_DECISION_CACHE

        timeout_sec = _env_float(
            "BOOTSTRAP_HF_ROUTE_PROBE_TIMEOUT_SEC",
            HF_ROUTE_PROBE_TIMEOUT_SEC,
            min_value=0.4,
            max_value=5.0,
        )
        probe_attempts = _env_int(
            "BOOTSTRAP_HF_ROUTE_PROBE_ATTEMPTS",
            2,
            min_value=1,
            max_value=5,
        )
        google_ok, google_latency, google_host = _probe_google_connectivity(
            timeout_sec,
            attempts=probe_attempts,
        )
        official_latency = _probe_tcp_latency_best(
            HF_OFFICIAL_HOST,
            timeout_sec=timeout_sec,
            attempts=probe_attempts,
        )
        mirror_latency = _probe_tcp_latency_best(
            HF_MIRROR_HOST,
            timeout_sec=timeout_sec,
            attempts=probe_attempts,
        )

        detected_region = HF_REGION_UNKNOWN
        if google_ok and official_latency is not None:
            # Reachable Google + official HF implies global-friendly network.
            detected_region = HF_REGION_GLOBAL
        elif (not google_ok) and mirror_latency is not None:
            # Google blocked/unreachable but mirror is reachable implies CN-like network.
            detected_region = HF_REGION_CN
        elif official_latency is None and mirror_latency is not None:
            detected_region = HF_REGION_CN
        elif mirror_latency is None and official_latency is not None:
            detected_region = HF_REGION_GLOBAL
        elif official_latency is not None and mirror_latency is not None:
            # Tight threshold keeps domestic/global classification strict and stable.
            if official_latency <= mirror_latency * 0.90:
                detected_region = HF_REGION_GLOBAL
            elif mirror_latency <= official_latency * 0.90:
                detected_region = HF_REGION_CN

        if detected_region == HF_REGION_CN:
            chosen = [HF_MIRROR_HOST, HF_OFFICIAL_HOST]
        elif detected_region == HF_REGION_GLOBAL:
            chosen = [HF_OFFICIAL_HOST, HF_MIRROR_HOST]
        else:
            chosen = list(default_order)

        def _fmt_latency(label: str, value: Optional[float]) -> str:
            if value is None:
                return f"{label}=unreachable"
            return f"{label}={int(round(value * 1000.0))}ms"

        if google_ok and google_latency is not None:
            google_part = f"google={google_host}:{int(round(google_latency * 1000.0))}ms"
        else:
            google_part = "google=unreachable"

        source = "auto-probe"
        if detected_region == HF_REGION_UNKNOWN:
            source = "auto-probe-fallback-default-order"
        note = (
            f"HF route priority: {chosen[0]} -> {chosen[1]} "
            f"(environment={detected_region}, source={source}, attempts={probe_attempts}, "
            f"{google_part}, {_fmt_latency('official', official_latency)}, {_fmt_latency('mirror', mirror_latency)})"
        )
        _HF_ROUTE_DECISION_CACHE = ((chosen[0], chosen[1]), note)
        return _HF_ROUTE_DECISION_CACHE


def hf_route_probe_note() -> str:
    _hosts, note = _decide_hf_host_priority()
    return note


def _hf_url_candidates_base(url: str) -> list[str]:
    raw = str(url or "").strip()
    if not raw:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        key = str(candidate or "").strip()
        if not key:
            return
        lk = key.lower()
        if lk in seen:
            return
        seen.add(lk)
        candidates.append(key)

    _add(raw)
    swapped = _swap_hf_host_url(raw)
    if swapped:
        _add(swapped)
    return candidates


def hf_url_candidates(url: str) -> list[str]:
    candidates = _hf_url_candidates_base(url)
    if len(candidates) <= 1:
        return candidates

    host_order, _note = _decide_hf_host_priority()
    ordered: list[str] = []
    for host in host_order:
        for candidate in candidates:
            if _url_host(candidate) == host and candidate not in ordered:
                ordered.append(candidate)
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def sha256sum(path: Path, progress_cb: Optional[Callable[[int, int], None]] = None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress_cb:
                progress_cb(done, total)
    return h.hexdigest().lower()


def is_safe_extract_path(base_dir: Path, target_path: Path) -> bool:
    try:
        base = base_dir.resolve()
        target = target_path.resolve()
        return str(target).startswith(str(base))
    except Exception:
        return False


def safe_extract_zip_python(
    zip_path: Path,
    dest_dir: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        total_uncompressed = sum((i.file_size or 0) for i in infos if not i.is_dir())
        done = 0
        for info in infos:
            out_path = dest_dir / info.filename
            if not is_safe_extract_path(dest_dir, out_path):
                raise RuntimeError(f"Unsafe archive path: {info.filename}")
            zf.extract(info, path=dest_dir)
            if not info.is_dir():
                done += int(info.file_size or 0)
                if progress_cb:
                    progress_cb(done, total_uncompressed)


def try_extract_with_tar(zip_path: Path, dest_dir: Path) -> tuple[bool, str]:
    tar_path = shutil.which("tar")
    if not tar_path:
        return False, "tar.exe not found"
    candidates = [
        [tar_path, "-xf", str(zip_path), "-C", str(dest_dir)],
    ]
    if "".join(zip_path.suffixes).lower().endswith(".tar.zst"):
        candidates.extend(
            [
                [tar_path, "--zstd", "-xf", str(zip_path), "-C", str(dest_dir)],
                [tar_path, "-I", "zstd", "-xf", str(zip_path), "-C", str(dest_dir)],
            ]
        )
    last_err = "unknown tar error"
    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60 * 60,
                **_windows_hidden_subprocess_kwargs(),
            )
        except Exception as e:
            last_err = f"tar failed: {e}"
            continue
        if result.returncode == 0:
            return True, "ok"
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:] or ["unknown tar error"]
        last_err = tail[0]
    return False, last_err


def flatten_payload_root(extract_dir: Path) -> Path:
    items = [p for p in extract_dir.iterdir()]
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    return extract_dir


def deploy_tree(src_root: Path, install_dir: Path) -> None:
    validation_error = validate_install_dir_choice(install_dir)
    if validation_error:
        raise RuntimeError(validation_error)
    install_dir.parent.mkdir(parents=True, exist_ok=True)

    if install_dir.exists():
        shutil.rmtree(install_dir)

    if src_root.name == install_dir.name and src_root.parent.exists():
        try:
            shutil.move(str(src_root), str(install_dir))
            return
        except Exception:
            pass

    install_dir.mkdir(parents=True, exist_ok=True)
    for child in src_root.iterdir():
        shutil.move(str(child), str(install_dir / child.name))


def windows_desktop_dir() -> Path:
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"


def windows_start_menu_programs_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    return (
        Path.home()
        / "AppData"
        / "Roaming"
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
    )


def create_windows_shortcut(
    *,
    link_path: Path,
    target_path: Path,
    working_dir: Path,
    icon_path: Optional[Path] = None,
    description: str = "",
    arguments: str = "",
) -> None:
    if os.name != "nt":
        raise RuntimeError("Shortcut creation is currently implemented for Windows only.")
    import ctypes.wintypes as wintypes

    link_path.parent.mkdir(parents=True, exist_ok=True)

    CLSCTX_INPROC_SERVER = 0x1
    COINIT_APARTMENTTHREADED = 0x2
    S_OK = 0x00000000
    S_FALSE = 0x00000001
    RPC_E_CHANGED_MODE = 0x80010106

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    def _guid(text: str) -> GUID:
        g = GUID()
        hr = ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(g))
        if int(hr) != 0:
            raise RuntimeError(f"CLSIDFromString failed for {text} ({int(hr) & 0xFFFFFFFF:#010x})")
        return g

    def _hr_hex(hr: int) -> str:
        return f"0x{(int(hr) & 0xFFFFFFFF):08X}"

    def _check_hr(hr: int, op: str) -> None:
        if int(hr) < 0:
            raise RuntimeError(f"{op} failed ({_hr_hex(hr)})")

    def _vtbl(ptr: ctypes.c_void_p):
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents

    def _com_method(ptr: ctypes.c_void_p, index: int, restype, argtypes):
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(_vtbl(ptr)[index])

    def _release(ptr: ctypes.c_void_p) -> None:
        if not ptr:
            return
        try:
            _com_method(ptr, 2, wintypes.ULONG, [])(ptr)  # IUnknown::Release
        except Exception:
            pass

    def _query_interface(ptr: ctypes.c_void_p, iid: GUID) -> ctypes.c_void_p:
        out = ctypes.c_void_p()
        qi = _com_method(ptr, 0, wintypes.LONG, [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)])
        hr = qi(ptr, ctypes.byref(iid), ctypes.byref(out))  # IUnknown::QueryInterface
        _check_hr(hr, "QueryInterface")
        return out

    shell_link_ptr = ctypes.c_void_p()
    persist_ptr = ctypes.c_void_p()
    need_uninit = False
    try:
        hr_init = ctypes.oledll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        hr_init_u32 = int(hr_init) & 0xFFFFFFFF
        if hr_init_u32 in (S_OK, S_FALSE):
            need_uninit = True
        elif hr_init_u32 != RPC_E_CHANGED_MODE:
            raise RuntimeError(f"CoInitializeEx failed ({_hr_hex(hr_init)})")

        clsid_shell_link = _guid("{00021401-0000-0000-C000-000000000046}")  # CLSID_ShellLink
        iid_ishelllinkw = _guid("{000214F9-0000-0000-C000-000000000046}")   # IID_IShellLinkW
        iid_ipersistfile = _guid("{0000010B-0000-0000-C000-000000000046}")  # IID_IPersistFile

        hr_create = ctypes.oledll.ole32.CoCreateInstance(
            ctypes.byref(clsid_shell_link),
            None,
            CLSCTX_INPROC_SERVER,
            ctypes.byref(iid_ishelllinkw),
            ctypes.byref(shell_link_ptr),
        )
        _check_hr(hr_create, "CoCreateInstance(ShellLink)")

        target_str = str(target_path)
        working_dir_str = str(working_dir)
        icon_file_str = str(icon_path or target_path)
        desc_str = str(description or "")
        args_str = str(arguments or "")

        set_path = _com_method(shell_link_ptr, 20, wintypes.LONG, [wintypes.LPCWSTR])            # IShellLinkW::SetPath
        set_desc = _com_method(shell_link_ptr, 7, wintypes.LONG, [wintypes.LPCWSTR])              # IShellLinkW::SetDescription
        set_workdir = _com_method(shell_link_ptr, 9, wintypes.LONG, [wintypes.LPCWSTR])           # IShellLinkW::SetWorkingDirectory
        set_args = _com_method(shell_link_ptr, 11, wintypes.LONG, [wintypes.LPCWSTR])             # IShellLinkW::SetArguments
        set_icon = _com_method(shell_link_ptr, 17, wintypes.LONG, [wintypes.LPCWSTR, ctypes.c_int])  # IShellLinkW::SetIconLocation

        _check_hr(set_path(shell_link_ptr, target_str), "IShellLinkW.SetPath")
        _check_hr(set_workdir(shell_link_ptr, working_dir_str), "IShellLinkW.SetWorkingDirectory")
        _check_hr(set_icon(shell_link_ptr, icon_file_str, 0), "IShellLinkW.SetIconLocation")
        _check_hr(set_desc(shell_link_ptr, desc_str), "IShellLinkW.SetDescription")
        _check_hr(set_args(shell_link_ptr, args_str), "IShellLinkW.SetArguments")

        persist_ptr = _query_interface(shell_link_ptr, iid_ipersistfile)
        save = _com_method(persist_ptr, 6, wintypes.LONG, [wintypes.LPCWSTR, wintypes.BOOL])  # IPersistFile::Save
        _check_hr(save(persist_ptr, str(link_path), True), "IPersistFile.Save")
    finally:
        _release(persist_ptr)
        _release(shell_link_ptr)
        if need_uninit:
            try:
                ctypes.oledll.ole32.CoUninitialize()
            except Exception:
                pass


def create_install_shortcuts(
    install_dir: Path,
    *,
    create_desktop: bool,
    create_start_menu: bool,
) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    warnings: list[str] = []
    main_exe = install_dir / MAIN_EXE_NAME
    uninstall_exe = install_dir / UNINSTALL_EXE_NAME

    if not main_exe.exists():
        warnings.append(f"Main executable not found for shortcut creation: {main_exe}")
        return records, warnings

    def _try_create(link_path: Path, target: Path, kind: str, description: str) -> None:
        try:
            create_windows_shortcut(
                link_path=link_path,
                target_path=target,
                working_dir=install_dir,
                icon_path=main_exe,
                description=description,
            )
            records.append({"path": str(link_path), "target": str(target), "kind": kind})
        except Exception as e:
            warnings.append(str(e))

    if create_desktop:
        _try_create(
            windows_desktop_dir() / f"{APP_NAME}.lnk",
            main_exe,
            "desktop",
            f"Launch {APP_NAME}",
        )

    if create_start_menu:
        start_menu_dir = windows_start_menu_programs_dir() / APP_NAME
        _try_create(
            start_menu_dir / f"{APP_NAME}.lnk",
            main_exe,
            "start_menu",
            f"Launch {APP_NAME}",
        )
        if uninstall_exe.exists():
            _try_create(
                start_menu_dir / f"Uninstall {APP_NAME}.lnk",
                uninstall_exe,
                "start_menu_uninstall",
                f"Uninstall {APP_NAME}",
            )
        else:
            warnings.append(f"Uninstaller not found for Start Menu shortcut: {uninstall_exe}")

    return records, warnings


class InstallerWorker:
    @staticmethod
    def _parse_payload_url_candidates(*values: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            text = str(raw or "").replace("\r", "\n")
            # QLineEdit is single-line, so support `|` as an explicit mirror separator.
            for part in text.replace("|", "\n").split("\n"):
                url = str(part or "").strip().strip('"').strip("'")
                if not url:
                    continue
                # Keep UI thread responsive: avoid network route probing here.
                expanded = _hf_url_candidates_base(url) or [url]
                for one in expanded:
                    key = one.strip().lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(one)
        return out

    def __init__(
        self,
        payload_url: str,
        install_dir: Path,
        payload_sha256: str = "",
        *,
        create_desktop_shortcut: bool = True,
        create_start_menu_shortcut: bool = True,
    ):
        raw_payload_url = payload_url.strip()
        extra_mirrors = str(os.environ.get("BOOTSTRAP_DOWNLOAD_MIRRORS", "") or "")
        self.payload_urls = self._parse_payload_url_candidates(raw_payload_url, extra_mirrors)
        self.payload_url = self.payload_urls[0] if self.payload_urls else raw_payload_url
        self.install_dir = install_dir
        self.payload_sha256 = payload_sha256.strip().lower()
        self.create_desktop_shortcut = bool(create_desktop_shortcut)
        self.create_start_menu_shortcut = bool(create_start_menu_shortcut)
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.clear_workspace_on_cancel = False
        self.download_max_connections = _env_int(
            "BOOTSTRAP_DOWNLOAD_MAX_CONNECTIONS",
            DOWNLOAD_MAX_CONNECTIONS,
            min_value=1,
            max_value=64,
        )
        self.download_range_io_chunk_size = _env_int(
            "BOOTSTRAP_DOWNLOAD_IO_CHUNK_MB",
            RANGE_IO_CHUNK_SIZE // (1024 * 1024),
            min_value=1,
            max_value=32,
        ) * 1024 * 1024
        self.download_min_piece_size = _env_int(
            "BOOTSTRAP_DOWNLOAD_MIN_PIECE_MB",
            DOWNLOAD_MIN_PIECE_SIZE // (1024 * 1024),
            min_value=8,
            max_value=1024,
        ) * 1024 * 1024
        self.download_target_piece_size = _env_int(
            "BOOTSTRAP_DOWNLOAD_PIECE_MB",
            DOWNLOAD_TARGET_PIECE_SIZE // (1024 * 1024),
            min_value=16,
            max_value=2048,
        ) * 1024 * 1024
        if self.download_target_piece_size < self.download_min_piece_size:
            self.download_target_piece_size = self.download_min_piece_size
        self.download_segments_per_connection = _env_int(
            "BOOTSTRAP_DOWNLOAD_SEGMENTS_PER_CONNECTION",
            3,
            min_value=1,
            max_value=16,
        )
        self.download_timeout_sec = _env_float(
            "BOOTSTRAP_DOWNLOAD_TIMEOUT_SEC",
            DOWNLOAD_URL_TIMEOUT_SEC,
            min_value=5.0,
            max_value=600.0,
        )
        self.download_resume_flush_interval_sec = _env_float(
            "BOOTSTRAP_DOWNLOAD_RESUME_FLUSH_SEC",
            DOWNLOAD_RESUME_FLUSH_INTERVAL_SEC,
            min_value=0.2,
            max_value=30.0,
        )
        self.download_resume_flush_segments = _env_int(
            "BOOTSTRAP_DOWNLOAD_RESUME_FLUSH_SEGMENTS",
            DOWNLOAD_RESUME_FLUSH_SEGMENTS,
            min_value=1,
            max_value=256,
        )
        self.download_progress_emit_interval_sec = _env_float(
            "BOOTSTRAP_DOWNLOAD_PROGRESS_INTERVAL_SEC",
            DOWNLOAD_PROGRESS_EMIT_INTERVAL_SEC,
            min_value=0.05,
            max_value=2.0,
        )
        hf_payload_mode = any(
            _url_host(one) in {HF_OFFICIAL_HOST, HF_MIRROR_HOST}
            for one in (self.payload_urls or [self.payload_url])
        )
        self.download_request_gap_min_sec = _env_float(
            "BOOTSTRAP_DOWNLOAD_REQUEST_GAP_SEC",
            (0.02 if hf_payload_mode else 0.0),
            min_value=0.0,
            max_value=5.0,
        )
        self.download_request_gap_max_sec = _env_float(
            "BOOTSTRAP_DOWNLOAD_REQUEST_GAP_MAX_SEC",
            (1.5 if hf_payload_mode else 3.0),
            min_value=0.2,
            max_value=120.0,
        )
        self.download_rate_limit_status_emit_interval_sec = _env_float(
            "BOOTSTRAP_DOWNLOAD_RL_STATUS_INTERVAL_SEC",
            2.0,
            min_value=0.2,
            max_value=30.0,
        )
        self.download_parallel_round_cap = _env_int(
            "BOOTSTRAP_DOWNLOAD_PARALLEL_ROUND_CAP",
            18,
            min_value=2,
            max_value=200,
        )

    def emit(self, event_type: str, **payload) -> None:
        self.events.put({"type": event_type, **payload})

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="InstallerWorker", daemon=True)
        self.thread.start()

    def cancel(self, *, clear_workspace: bool = False) -> None:
        if clear_workspace:
            self.clear_workspace_on_cancel = True
        self.cancel_event.set()

    def _installer_workspace_root(self) -> Path:
        return Path(tempfile.gettempdir()) / f"{APP_NAME}-setup"

    def _cancel_message(self) -> str:
        return "Installation cancelled by user."

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise InstallerCancelled(self._cancel_message())

    def _sleep_with_cancel(self, seconds: float) -> None:
        deadline = time.time() + max(0.0, float(seconds))
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def _run(self) -> None:
        temp_root = self._installer_workspace_root()
        archive_path = temp_root / PAYLOAD_ARCHIVE_NAME
        extract_dir = temp_root / "extracted"
        run_ok = False
        try:
            self._raise_if_cancelled()
            self.emit("status", text="Preparing installer workspace...")
            temp_root.mkdir(parents=True, exist_ok=True)
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            extract_dir.mkdir(parents=True, exist_ok=True)

            self._raise_if_cancelled()
            self._download_archive(archive_path)

            self._raise_if_cancelled()
            if self.payload_sha256:
                self.emit("status", text="Verifying archive integrity (SHA256)...")
                digest = sha256sum(
                    archive_path,
                    progress_cb=lambda done, total: self.emit(
                        "verify_progress", done=done, total=total
                    ),
                )
                if digest != self.payload_sha256:
                    raise RuntimeError(
                        f"SHA256 mismatch.\nExpected: {self.payload_sha256}\nActual:   {digest}"
                    )

            self._raise_if_cancelled()
            self._extract_archive(archive_path, extract_dir)

            self._raise_if_cancelled()
            payload_root = flatten_payload_root(extract_dir)
            self.emit("status", text=f"Installing to {self.install_dir} ...")
            deploy_tree(payload_root, self.install_dir)

            self._raise_if_cancelled()
            self.emit("status", text="Creating shortcuts...")
            shortcut_records, shortcut_warnings = create_install_shortcuts(
                self.install_dir,
                create_desktop=self.create_desktop_shortcut,
                create_start_menu=self.create_start_menu_shortcut,
            )

            self._raise_if_cancelled()
            self._write_manifest(shortcut_records=shortcut_records, shortcut_warnings=shortcut_warnings)
            self.emit(
                "done",
                install_dir=str(self.install_dir),
                shortcuts=shortcut_records,
                shortcut_warnings=shortcut_warnings,
            )
            run_ok = True
        except InstallerCancelled as e:
            self.emit("cancelled", message=str(e))
        except Exception as e:
            self.emit("error", message=str(e))
        finally:
            try:
                keep_resume_workspace = False
                if not (self.cancel_event.is_set() and self.clear_workspace_on_cancel):
                    try:
                        part_path = self._archive_part_path(archive_path)
                        state_path = self._archive_resume_state_path(archive_path)
                        keep_resume_workspace = (not run_ok) and (part_path.exists() or state_path.exists())
                    except Exception:
                        keep_resume_workspace = False
                if temp_root.exists() and not keep_resume_workspace:
                    shutil.rmtree(temp_root, ignore_errors=True)
                elif keep_resume_workspace:
                    try:
                        if extract_dir.exists():
                            shutil.rmtree(extract_dir, ignore_errors=True)
                    except Exception:
                        pass
            except Exception:
                pass

    def _download_headers(self, extra_headers: Optional[dict[str, str]] = None) -> dict[str, str]:
        headers = {
            "User-Agent": f"{APP_NAME}-Bootstrapper/1.0",
            # Avoid transparent content encoding so Range math stays byte-accurate.
            "Accept-Encoding": "identity",
        }
        if extra_headers:
            headers.update({str(k): str(v) for k, v in extra_headers.items()})
        return headers

    def _parse_total_from_content_range(self, content_range: str) -> int:
        # Expected: "bytes 0-0/12345"
        raw = (content_range or "").strip()
        if "/" not in raw:
            return 0
        try:
            total = raw.rsplit("/", 1)[1].strip()
            if not total or total == "*":
                return 0
            return int(total)
        except Exception:
            return 0

    def _error_chain(self, err: BaseException) -> list[BaseException]:
        chain: list[BaseException] = []
        seen: set[int] = set()
        cur: Optional[BaseException] = err
        while cur is not None:
            obj_id = id(cur)
            if obj_id in seen:
                break
            seen.add(obj_id)
            chain.append(cur)
            cur = (
                cur.__cause__
                if isinstance(cur.__cause__, BaseException)
                else (cur.__context__ if isinstance(cur.__context__, BaseException) else None)
            )
        return chain

    def _http_status_from_error(self, err: BaseException | None) -> int:
        if err is None:
            return 0
        for item in self._error_chain(err):
            code = getattr(item, "code", None)
            try:
                if code is not None:
                    return int(code)
            except Exception:
                continue
        text = str(err or "")
        for token in ("HTTP Error 429", "HTTP 429"):
            if token in text:
                return 429
        return 0

    def _retry_after_delay_sec(self, err: BaseException | None) -> float:
        if err is None:
            return 0.0
        for item in self._error_chain(err):
            headers = getattr(item, "headers", None)
            if headers is None:
                continue
            try:
                raw = str(headers.get("Retry-After") or "").strip()
            except Exception:
                raw = ""
            if not raw:
                continue
            if raw.isdigit():
                return max(0.0, min(float(raw), DOWNLOAD_RETRY_BACKOFF_CAP_SEC))
            try:
                dt = parsedate_to_datetime(raw)
                if dt is None:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                delay = (dt - now).total_seconds()
                if delay > 0:
                    return max(0.0, min(delay, DOWNLOAD_RETRY_BACKOFF_CAP_SEC))
            except Exception:
                continue
        return 0.0

    def _is_rate_limited_error(self, err: BaseException | None) -> bool:
        status = self._http_status_from_error(err)
        if status == 429:
            return True
        text = str(err or "").lower()
        return ("too many requests" in text) or ("rate limit" in text and "download" in text)

    def _is_unexpected_eof_error(self, err: BaseException | None) -> bool:
        if err is None:
            return False
        text = str(err or "").lower()
        return (
            "unexpected eof" in text
            or "incompleteread" in text
            or "remote end closed connection" in text
        )

    def _is_connection_reset_error(self, err: BaseException | None) -> bool:
        if err is None:
            return False
        text = str(err or "").lower()
        return (
            "winerror 10054" in text
            or "[errno 104]" in text
            or "connection reset" in text
            or "connection aborted" in text
            or "forcibly closed by the remote host" in text
            or "existing connection was forcibly closed" in text
        )

    def _is_transient_download_error(self, err: BaseException | None) -> bool:
        if err is None:
            return False
        if (
            self._is_rate_limited_error(err)
            or self._is_unexpected_eof_error(err)
            or self._is_connection_reset_error(err)
        ):
            return True
        status = self._http_status_from_error(err)
        if status in {408, 425, 500, 502, 503, 504}:
            return True
        text = str(err or "").lower()
        transient_tokens = (
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "broken pipe",
        )
        return any(tok in text for tok in transient_tokens)

    def _short_error_text(self, err: BaseException | None) -> str:
        if err is None:
            return "unknown error"
        status = self._http_status_from_error(err)
        raw = str(err).strip() or err.__class__.__name__
        if status and f"{status}" not in raw:
            return f"HTTP {status}: {raw}"
        return raw

    def _retry_delay_sec(self, err: BaseException | None, *, attempt: int, mode: str = "single") -> float:
        attempt = max(1, int(attempt))
        status = self._http_status_from_error(err)
        retry_after = self._retry_after_delay_sec(err)

        if status == 429 or self._is_rate_limited_error(err):
            # Use gentler backoff for segmented mode; we prefer reducing request burst and
            # connection count before entering long hard waits. Honor Retry-After when present.
            if mode.startswith("parallel"):
                base = min(10.0, 0.9 * (2 ** (attempt - 1)))
                jitter = random.uniform(0.0, min(2.0, 0.4 * attempt))
                return max(retry_after, min(45.0, base + jitter))
            else:
                base = min(DOWNLOAD_RETRY_BACKOFF_CAP_SEC, 4.0 * (2 ** (attempt - 1)))
            jitter = random.uniform(0.0, min(3.0, 0.6 * attempt))
            return max(retry_after, min(DOWNLOAD_RETRY_BACKOFF_CAP_SEC, base + jitter))

        if status in {500, 502, 503, 504}:
            base = min(45.0, 1.5 * (2 ** (attempt - 1)))
            return min(DOWNLOAD_RETRY_BACKOFF_CAP_SEC, base + random.uniform(0.0, 1.0))

        return min(12.0, 0.8 * attempt + random.uniform(0.0, 0.6))

    def _probe_parallel_download(self) -> tuple[int, bool, str]:
        """
        Returns (total_bytes, range_supported, probe_note).
        """
        candidate_urls = list(getattr(self, "payload_urls", None) or [self.payload_url])
        first_note = "content length unavailable"
        first_result: Optional[tuple[int, bool, str]] = None

        for idx, url in enumerate(candidate_urls, start=1):
            self._raise_if_cancelled()
            total_bytes = 0
            accept_ranges = ""
            note = ""
            try:
                head_req = urllib.request.Request(
                    url,
                    headers=self._download_headers(),
                    method="HEAD",
                )
                with urllib.request.urlopen(head_req, timeout=30) as resp:
                    total = resp.headers.get("Content-Length")
                    if total and str(total).isdigit():
                        total_bytes = int(total)
                    accept_ranges = str(resp.headers.get("Accept-Ranges") or "").strip().lower()
            except Exception as e:
                note = f"HEAD failed: {self._short_error_text(e)}"

            # Fast path when server explicitly advertises byte ranges and we know size.
            if total_bytes > 0 and accept_ranges == "bytes":
                suffix = f" (mirror {idx}/{len(candidate_urls)})" if len(candidate_urls) > 1 else ""
                return total_bytes, True, f"accept-ranges=bytes{suffix}"

            # Some CDNs omit Accept-Ranges but still honor Range requests.
            try:
                self._raise_if_cancelled()
                test_req = urllib.request.Request(
                    url,
                    headers=self._download_headers({"Range": "bytes=0-0"}),
                )
                with urllib.request.urlopen(test_req, timeout=30) as resp:
                    status = int(getattr(resp, "status", resp.getcode()))
                    content_range = str(resp.headers.get("Content-Range") or "")
                    if status == 206:
                        parsed_total = self._parse_total_from_content_range(content_range)
                        if parsed_total > 0:
                            total_bytes = parsed_total
                        suffix = f" (mirror {idx}/{len(candidate_urls)})" if len(candidate_urls) > 1 else ""
                        return total_bytes, True, f"range test succeeded{suffix}"
            except Exception as e:
                if not note:
                    note = f"range test failed: {self._short_error_text(e)}"

            if total_bytes > 0:
                suffix = f" (mirror {idx}/{len(candidate_urls)})" if len(candidate_urls) > 1 else ""
                result = (total_bytes, False, f"range unsupported{suffix}")
                if first_result is None:
                    first_result = result
                continue

            if not first_note or first_note == "content length unavailable":
                first_note = note or first_note

        if first_result is not None:
            return first_result
        return 0, False, first_note or "content length unavailable"

    def _is_hf_payload_source(self) -> bool:
        for one in (getattr(self, "payload_urls", None) or [self.payload_url]):
            if _url_host(str(one or "")) in {HF_OFFICIAL_HOST, HF_MIRROR_HOST}:
                return True
        return False

    def _reorder_payload_urls_by_route(self, urls: list[str]) -> list[str]:
        ordered_in = [str(u).strip() for u in (urls or []) if str(u).strip()]
        if len(ordered_in) <= 1:
            return ordered_in
        preserve_first = bool(getattr(self, "_preserve_first_payload_url", False))
        pinned_first = ordered_in[0] if preserve_first else ""
        reorder_input = ordered_in[1:] if preserve_first else ordered_in
        if preserve_first and not reorder_input:
            return ordered_in
        host_order, _note = _decide_hf_host_priority()
        out: list[str] = []
        seen: set[str] = set()
        for host in host_order:
            for u in reorder_input:
                key = u.lower()
                if key in seen:
                    continue
                if _url_host(u) != host:
                    continue
                seen.add(key)
                out.append(u)
        for u in reorder_input:
            key = u.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(u)
        if preserve_first and pinned_first:
            key = pinned_first.lower()
            out = [pinned_first, *[u for u in out if u.lower() != key]]
        return out

    def _hf_parallel_connection_cap(self) -> int:
        forced_region = _forced_hf_region()
        if forced_region == HF_REGION_GLOBAL:
            default_cap = 8
        elif forced_region == HF_REGION_CN:
            default_cap = 6
        else:
            host_order, _note = _decide_hf_host_priority()
            default_cap = 8 if host_order[0] == HF_OFFICIAL_HOST else 6
        return _env_int(
            "BOOTSTRAP_HF_MAX_CONNECTIONS",
            default_cap,
            min_value=1,
            max_value=16,
        )

    def _choose_parallel_connections(self, total_bytes: int) -> int:
        env_override = (os.environ.get("BOOTSTRAP_DOWNLOAD_CONNECTIONS") or "").strip()
        if env_override:
            try:
                n = int(env_override)
                if n >= 1:
                    return max(1, min(self.download_max_connections, n))
            except Exception:
                pass

        if total_bytes < DOWNLOAD_PARALLEL_THRESHOLD:
            return 1

        hf_payload = self._is_hf_payload_source()
        # Size-driven heuristic: use a more conservative scale for HF/CDN sources
        # to reduce TCP reset bursts in constrained networks.
        if hf_payload:
            by_size = 1 + (total_bytes // (256 * 1024 * 1024))
        else:
            by_size = 1 + (total_bytes // (128 * 1024 * 1024))
        cpu_hint = max(2, min(16, (os.cpu_count() or 8)))
        cap = min(self.download_max_connections, max(6, cpu_hint * 2 + 2))
        if hf_payload:
            cap = min(cap, self._hf_parallel_connection_cap())
        return max(2, min(cap, int(by_size)))

    def _build_download_ranges(self, total_bytes: int, connections: int) -> list[tuple[int, int]]:
        if total_bytes <= 0:
            return []
        if connections <= 1:
            return [(0, total_bytes - 1)]

        # Prefer fewer/larger segments to reduce request churn on rate-limited CDNs.
        # IDM-style tail splitting still improves utilization near the end.
        segs_per_conn = max(1, int(getattr(self, "download_segments_per_connection", 2)))
        ideal_piece = max(self.download_min_piece_size, total_bytes // max(connections * segs_per_conn, 1))
        piece_size = max(self.download_min_piece_size, min(self.download_target_piece_size, ideal_piece))

        ranges: list[tuple[int, int]] = []
        start = 0
        while start < total_bytes:
            end = min(total_bytes - 1, start + piece_size - 1)
            ranges.append((start, end))
            start = end + 1
        return ranges

    def _archive_part_path(self, archive_path: Path) -> Path:
        return archive_path.with_suffix(archive_path.suffix + ".part")

    def _archive_resume_state_path(self, archive_path: Path) -> Path:
        return archive_path.with_suffix(archive_path.suffix + ".resume.json")

    def _cleanup_download_temp_files(self, archive_path: Path) -> None:
        for p in (self._archive_part_path(archive_path), self._archive_resume_state_path(archive_path)):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    def _normalize_completed_ranges(self, ranges: list[tuple[int, int]], *, total_bytes: int) -> list[tuple[int, int]]:
        cleaned: list[tuple[int, int]] = []
        for start, end in ranges:
            try:
                s = max(0, int(start))
                e = min(total_bytes - 1, int(end))
            except Exception:
                continue
            if total_bytes <= 0 or s > e:
                continue
            cleaned.append((s, e))
        if not cleaned:
            return []
        cleaned.sort(key=lambda x: (x[0], x[1]))
        merged: list[tuple[int, int]] = [cleaned[0]]
        for s, e in cleaned[1:]:
            ps, pe = merged[-1]
            if s <= (pe + 1):
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    def _sum_ranges(self, ranges: list[tuple[int, int]]) -> int:
        total = 0
        for s, e in ranges:
            total += (int(e) - int(s) + 1)
        return total

    def _split_ranges_to_piece_size(
        self,
        ranges: list[tuple[int, int]],
        *,
        max_piece_size: int,
    ) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        max_piece = max(1, int(max_piece_size))
        for start, end in ranges:
            s = int(start)
            e = int(end)
            while s <= e:
                x = min(e, s + max_piece - 1)
                out.append((s, x))
                s = x + 1
        return out

    def _load_resume_state(
        self,
        archive_path: Path,
        *,
        total_bytes: int,
    ) -> list[tuple[int, int]]:
        part_path = self._archive_part_path(archive_path)
        state_path = self._archive_resume_state_path(archive_path)
        if not (part_path.exists() and state_path.exists()):
            return []
        try:
            if part_path.stat().st_size != total_bytes:
                return []
        except Exception:
            return []

        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        if not isinstance(payload, dict):
            return []
        if int(payload.get("version") or 0) != 1:
            return []
        saved_url = str(payload.get("url") or "")
        saved_urls_raw = payload.get("urls") or []
        saved_urls: list[str] = []
        if isinstance(saved_urls_raw, list):
            for item in saved_urls_raw:
                if isinstance(item, str) and item.strip():
                    saved_urls.append(item.strip())
        saved_sha256 = str(payload.get("payload_sha256") or "").strip().lower()
        current_sha256 = str(self.payload_sha256 or "").strip().lower()
        current_urls = list(getattr(self, "payload_urls", None) or [self.payload_url])
        url_match = (saved_url == self.payload_url) or any(u in current_urls for u in saved_urls)
        sha_match = bool(current_sha256 and saved_sha256 and saved_sha256 == current_sha256)
        if not (url_match or sha_match):
            return []
        if int(payload.get("total_bytes") or 0) != int(total_bytes):
            return []

        raw_ranges = payload.get("completed_ranges") or []
        parsed: list[tuple[int, int]] = []
        if isinstance(raw_ranges, list):
            for item in raw_ranges:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                ):
                    parsed.append((int(item[0]), int(item[1])))
        return self._normalize_completed_ranges(parsed, total_bytes=total_bytes)

    def _write_resume_state(
        self,
        archive_path: Path,
        *,
        total_bytes: int,
        completed_ranges: list[tuple[int, int]],
    ) -> None:
        state_path = self._archive_resume_state_path(archive_path)
        tmp_state = state_path.with_suffix(state_path.suffix + ".tmp")
        payload = {
            "version": 1,
            "url": self.payload_url,
            "urls": list(getattr(self, "payload_urls", None) or [self.payload_url]),
            "payload_sha256": str(self.payload_sha256 or "").strip().lower(),
            "total_bytes": int(total_bytes),
            "completed_ranges": [[int(s), int(e)] for s, e in completed_ranges],
            "updated_at_utc": utc_now_iso(),
        }
        tmp_state.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_state.replace(state_path)

    def _compact_segmented_workspace_for_single_resume(
        self,
        archive_path: Path,
        *,
        total_bytes_hint: int = 0,
    ) -> int:
        """
        Convert segmented download workspace into a single-stream resumable .part file.

        Segmented mode preallocates the .part file to full size and tracks downloaded
        chunks in a sidecar state file. Single-stream resume expects .part size to be
        the contiguous downloaded prefix length, so we compact only the confirmed prefix.
        """
        part_path = self._archive_part_path(archive_path)
        state_path = self._archive_resume_state_path(archive_path)
        if not part_path.exists():
            return 0
        if not state_path.exists():
            try:
                return max(0, int(part_path.stat().st_size))
            except Exception:
                return 0

        try:
            part_size = max(0, int(part_path.stat().st_size))
        except Exception:
            return 0

        total_bytes = int(max(0, total_bytes_hint))
        if total_bytes <= 0:
            total_bytes = part_size
        if total_bytes <= 0:
            return 0

        completed_ranges = self._load_resume_state(archive_path, total_bytes=total_bytes)
        if not completed_ranges:
            # Metadata is stale/corrupt; avoid false single-stream resume from a preallocated file.
            try:
                if state_path.exists():
                    state_path.unlink()
            except Exception:
                pass
            if part_size >= total_bytes and total_bytes > 0:
                try:
                    part_path.unlink()
                except Exception:
                    pass
            return 0

        prefix_end = -1
        for seg_start, seg_end in completed_ranges:
            if seg_start > (prefix_end + 1):
                break
            prefix_end = max(prefix_end, seg_end)
        contiguous_bytes = max(0, min(total_bytes, prefix_end + 1))

        if contiguous_bytes <= 0:
            try:
                part_path.unlink()
            except Exception:
                pass
            try:
                if state_path.exists():
                    state_path.unlink()
            except Exception:
                pass
            return 0

        if part_size != contiguous_bytes:
            compact_tmp = part_path.with_suffix(part_path.suffix + ".single.tmp")
            try:
                with part_path.open("rb") as src, compact_tmp.open("wb") as dst:
                    remaining = contiguous_bytes
                    while remaining > 0:
                        chunk = src.read(min(CHUNK_SIZE, remaining))
                        if not chunk:
                            raise RuntimeError(
                                "Segmented workspace compaction failed: truncated .part content."
                            )
                        dst.write(chunk)
                        remaining -= len(chunk)
                compact_tmp.replace(part_path)
            finally:
                try:
                    if compact_tmp.exists():
                        compact_tmp.unlink()
                except Exception:
                    pass

        try:
            if state_path.exists():
                state_path.unlink()
        except Exception:
            pass
        return contiguous_bytes

    def _download_archive_single(
        self,
        archive_path: Path,
        *,
        retries: int = DOWNLOAD_RETRIES,
        total_bytes_hint: int = 0,
        range_supported_hint: Optional[bool] = None,
    ) -> None:
        last_error = None
        tmp_path = self._archive_part_path(archive_path)
        candidate_urls = list(getattr(self, "payload_urls", None) or [self.payload_url])
        if not candidate_urls:
            candidate_urls = [self.payload_url]

        for attempt in range(1, retries + 1):
            self._raise_if_cancelled()
            start_time = time.time()
            session_bytes = 0
            total_bytes = int(max(0, total_bytes_hint))
            url = candidate_urls[(attempt - 1) % len(candidate_urls)]
            range_supported = bool(range_supported_hint) if range_supported_hint is not None else False
            resume_from = 0
            try:
                if tmp_path.exists():
                    try:
                        resume_from = max(0, int(tmp_path.stat().st_size))
                    except Exception:
                        resume_from = 0
                if resume_from > 0 and total_bytes > 0:
                    if resume_from > total_bytes:
                        self.emit(
                            "status",
                            text=(
                                "Local resume file is larger than remote payload; "
                                "restarting single-connection download from scratch..."
                            ),
                        )
                        try:
                            tmp_path.unlink()
                        except Exception:
                            pass
                        try:
                            state_path = self._archive_resume_state_path(archive_path)
                            if state_path.exists():
                                state_path.unlink()
                        except Exception:
                            pass
                        resume_from = 0
                    elif resume_from == total_bytes:
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path.replace(archive_path)
                        try:
                            state_path = self._archive_resume_state_path(archive_path)
                            if state_path.exists():
                                state_path.unlink()
                        except Exception:
                            pass
                        self.emit(
                            "download_progress",
                            done=total_bytes,
                            total=total_bytes,
                            speed_bps=0.0,
                            mode="single",
                            connections=1,
                        )
                        return

                headers = self._download_headers()
                if resume_from > 0:
                    # Only attempt resume when we already know/suspect Range support.
                    if range_supported_hint is None:
                        range_supported = True
                    if range_supported:
                        headers = self._download_headers({"Range": f"bytes={resume_from}-"})

                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.download_timeout_sec) as resp:
                    status = int(getattr(resp, "status", resp.getcode()))
                    total = resp.headers.get("Content-Length")
                    content_range = str(resp.headers.get("Content-Range") or "")
                    total_from_cr = self._parse_total_from_content_range(content_range)
                    if total_from_cr > 0:
                        total_bytes = total_from_cr
                    elif total and str(total).isdigit():
                        if status == 206 and resume_from > 0:
                            total_bytes = max(total_bytes, resume_from + int(total))
                        else:
                            total_bytes = int(total)

                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    append_mode = False
                    if resume_from > 0 and status == 206:
                        range_supported = True
                        lower = content_range.lower()
                        if lower.startswith("bytes "):
                            try:
                                span = content_range.split(" ", 1)[1].split("/", 1)[0]
                                got_start_s, _got_end_s = span.split("-", 1)
                                got_start = int(got_start_s.strip())
                                if got_start != resume_from:
                                    raise RuntimeError(
                                        f"Resume range mismatch (expected start {resume_from}, got {got_start})"
                                    )
                            except Exception as e:
                                raise RuntimeError(f"Invalid resume Content-Range: {content_range}") from e
                        append_mode = True
                    elif resume_from > 0 and status != 206:
                        # Server ignored Range; restart from scratch on this attempt.
                        range_supported = False
                        resume_from = 0
                        try:
                            if tmp_path.exists():
                                tmp_path.unlink()
                        except Exception:
                            pass

                    mode = "ab" if append_mode else "wb"
                    done = int(resume_from)
                    with tmp_path.open(mode) as f:
                        while True:
                            self._raise_if_cancelled()
                            chunk = resp.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            self._raise_if_cancelled()
                            f.write(chunk)
                            session_bytes += len(chunk)
                            done += len(chunk)
                            elapsed = max(0.001, time.time() - start_time)
                            self.emit(
                                "download_progress",
                                done=done,
                                total=total_bytes,
                                speed_bps=(session_bytes / elapsed),
                                mode="single",
                                connections=1,
                            )
                    if total_bytes > 0 and done < total_bytes:
                        raise RuntimeError(
                            f"Unexpected EOF in single download ({done}/{total_bytes} bytes)"
                        )
                tmp_path.replace(archive_path)
                try:
                    state_path = self._archive_resume_state_path(archive_path)
                    if state_path.exists():
                        state_path.unlink()
                except Exception:
                    pass
                return
            except InstallerCancelled:
                raise
            except urllib.error.HTTPError as e:
                if self._http_status_from_error(e) == 416 and resume_from > 0:
                    cleared_part = False
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                            cleared_part = True
                    except Exception:
                        pass
                    try:
                        state_path = self._archive_resume_state_path(archive_path)
                        if state_path.exists():
                            state_path.unlink()
                    except Exception:
                        pass
                    if cleared_part:
                        self.emit(
                            "status",
                            text=(
                                "Server rejected resume range (HTTP 416); "
                                "clearing stale partial download and retrying..."
                            ),
                        )
                        last_error = e
                        continue
                last_error = e
            except urllib.error.URLError as e:
                last_error = e
            except Exception as e:
                last_error = e

            # Preserve tmp_path so the next attempt (or next launch) can resume via HTTP Range.
            if not bool(range_supported_hint) and resume_from <= 0 and self._is_rate_limited_error(last_error):
                # Some servers only reveal range support after success; keep retry logic mirror-first.
                pass

            if attempt >= retries:
                self.emit(
                    "status",
                    text=(
                        f"Download failed (attempt {attempt}/{retries}): "
                        f"{self._short_error_text(last_error)}"
                    ),
                )
                break

            delay_sec = self._retry_delay_sec(last_error, attempt=attempt, mode="single")
            if self._is_rate_limited_error(last_error):
                self.emit(
                    "status",
                    text=(
                        f"Download rate-limited (attempt {attempt}/{retries}; "
                        f"waiting {int(round(delay_sec))}s before retry)..."
                    ),
                )
            else:
                self.emit(
                    "status",
                    text=(
                        f"Download failed (attempt {attempt}/{retries}: "
                        f"{self._short_error_text(last_error)}), retrying in {int(round(delay_sec))}s..."
                    ),
                )
            self._sleep_with_cancel(max(0.2, delay_sec))

        raise RuntimeError(f"Download failed: {last_error}")

    def _download_archive_parallel(
        self,
        archive_path: Path,
        *,
        total_bytes: int,
        connections: int,
    ) -> None:
        if total_bytes <= 0:
            raise RuntimeError("Parallel download requires known content length.")
        if connections <= 0:
            connections = 1

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._archive_part_path(archive_path)

        piece_template = self._build_download_ranges(total_bytes, max(1, connections))
        piece_size = max((e - s + 1) for s, e in piece_template) if piece_template else self.download_target_piece_size

        completed_ranges: list[tuple[int, int]] = []
        if tmp_path.exists():
            completed_ranges = self._load_resume_state(archive_path, total_bytes=total_bytes)
            if not completed_ranges:
                # Stale partial file without matching state; restart cleanly.
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        if not tmp_path.exists():
            with tmp_path.open("wb") as f:
                f.truncate(total_bytes)
            completed_ranges = []
            self._write_resume_state(
                archive_path,
                total_bytes=total_bytes,
                completed_ranges=completed_ranges,
            )
        else:
            # If state file disappeared but part remains, recreate an empty state file to keep future resume logic consistent.
            try:
                self._write_resume_state(
                    archive_path,
                    total_bytes=total_bytes,
                    completed_ranges=completed_ranges,
                )
            except Exception:
                pass

        resumed_bytes = self._sum_ranges(completed_ranges)
        if resumed_bytes > 0:
            self.emit(
                "status",
                text=(
                    f"Resuming segmented download: {human_bytes(resumed_bytes)} / "
                    f"{human_bytes(total_bytes)} already present..."
                ),
            )

        if resumed_bytes >= total_bytes:
            tmp_path.replace(archive_path)
            try:
                self._archive_resume_state_path(archive_path).unlink()
            except Exception:
                pass
            self.emit(
                "download_progress",
                done=total_bytes,
                total=total_bytes,
                speed_bps=0.0,
                mode="parallel",
                connections=max(1, connections),
            )
            return

        missing_ranges: list[tuple[int, int]] = []
        cursor = 0
        for s, e in completed_ranges:
            if cursor < s:
                missing_ranges.append((cursor, s - 1))
            cursor = max(cursor, e + 1)
        if cursor < total_bytes:
            missing_ranges.append((cursor, total_bytes - 1))

        work_ranges = self._split_ranges_to_piece_size(missing_ranges, max_piece_size=piece_size)
        ranges_queue: "queue.Queue[tuple[int, int]]" = queue.Queue()
        for byte_range in work_ranges:
            ranges_queue.put(byte_range)

        start_time = time.time()
        progress_lock = threading.Lock()
        emit_lock = threading.Lock()
        resume_lock = threading.Lock()
        split_lock = threading.Lock()
        cancel_event = threading.Event()
        error_queue: "queue.Queue[Exception]" = queue.Queue()
        done_bytes = resumed_bytes
        session_bytes = 0
        last_emit = 0.0
        dynamic_splits = 0
        last_resume_flush = time.time()
        pending_resume_segments = 0
        gate_lock = threading.Lock()
        initial_request_gap = float(max(0.0, self.download_request_gap_min_sec))
        if connections >= 8:
            # Stagger initial burst of range requests to avoid instant CDN rate-limit spikes.
            initial_request_gap = max(initial_request_gap, min(0.08, 0.8 / float(max(1, connections))))
        adaptive_state: dict[str, float] = {
            "request_gap_sec": initial_request_gap,
            "next_request_time": 0.0,
            "cooldown_until": 0.0,
            "rate_limit_hits": 0.0,
            "stable_requests": 0.0,
            "last_rl_status_emit": 0.0,
            "last_rl_bump": 0.0,
        }
        mirror_lock = threading.Lock()
        mirror_rr_index = 0
        mirror_urls = list(getattr(self, "payload_urls", None) or [self.payload_url])
        mirror_states: list[dict[str, Any]] = [
            {
                "url": u,
                "inflight": 0,
                "cooldown_until": 0.0,
                "penalty": 0.0,
                "rl_hits": 0,
            }
            for u in mirror_urls
        ]

        def is_cancelled() -> bool:
            return cancel_event.is_set() or self.cancel_event.is_set()

        def sleep_with_cancel(seconds: float) -> None:
            deadline = time.time() + max(0.0, float(seconds))
            while not is_cancelled():
                remaining = deadline - time.time()
                if remaining <= 0:
                    return
                time.sleep(min(0.1, remaining))

        def acquire_mirror() -> Optional[tuple[int, str]]:
            nonlocal mirror_rr_index
            while not is_cancelled():
                sleep_for = 0.05
                with mirror_lock:
                    if not mirror_states:
                        return None
                    now = time.time()
                    ready: list[int] = []
                    earliest = float("inf")
                    for idx, st in enumerate(mirror_states):
                        cd = float(st.get("cooldown_until", 0.0))
                        earliest = min(earliest, cd)
                        if cd <= now:
                            ready.append(idx)
                    if ready:
                        n = len(mirror_states)
                        rr_start = int(mirror_rr_index) % max(1, n)
                        best_idx = min(
                            ready,
                            key=lambda i: (
                                float(mirror_states[i].get("inflight", 0))
                                + float(mirror_states[i].get("penalty", 0.0)),
                                (int(i) - rr_start) % max(1, n),
                            ),
                        )
                        mirror_states[best_idx]["inflight"] = int(mirror_states[best_idx].get("inflight", 0)) + 1
                        mirror_rr_index = (best_idx + 1) % max(1, n)
                        return best_idx, str(mirror_states[best_idx].get("url", self.payload_url))
                    if earliest != float("inf"):
                        sleep_for = max(0.03, min(0.5, earliest - now))
                sleep_with_cancel(sleep_for)
            return None

        def release_mirror(mirror_idx: Optional[int], err: Optional[Exception] = None) -> None:
            if mirror_idx is None:
                return
            with mirror_lock:
                if mirror_idx < 0 or mirror_idx >= len(mirror_states):
                    return
                st = mirror_states[mirror_idx]
                st["inflight"] = max(0, int(st.get("inflight", 0)) - 1)
                now = time.time()
                if err is None:
                    st["penalty"] = max(0.0, float(st.get("penalty", 0.0)) * 0.7 - 0.05)
                    return

                if self._is_rate_limited_error(err):
                    rl_hits = int(st.get("rl_hits", 0)) + 1
                    st["rl_hits"] = rl_hits
                    st["penalty"] = min(10.0, float(st.get("penalty", 0.0)) + 1.5)
                    cd = min(30.0, self._retry_delay_sec(err, attempt=rl_hits, mode="parallel"))
                    st["cooldown_until"] = max(float(st.get("cooldown_until", 0.0)), now + max(0.3, cd))
                    return

                err_text = str(err or "").lower()
                if (
                    "content-length changed" in err_text
                    or "unexpected byte range" in err_text
                    or "resume range mismatch" in err_text
                ):
                    st["penalty"] = min(50.0, float(st.get("penalty", 0.0)) + 10.0)
                    st["cooldown_until"] = max(float(st.get("cooldown_until", 0.0)), now + 300.0)
                    return

                if self._is_unexpected_eof_error(err) or self._is_transient_download_error(err):
                    st["penalty"] = min(6.0, float(st.get("penalty", 0.0)) + 0.5)
                    st["cooldown_until"] = max(float(st.get("cooldown_until", 0.0)), now + 0.5)
                    return

                st["penalty"] = min(8.0, float(st.get("penalty", 0.0)) + 1.0)

        def wait_for_request_slot() -> bool:
            while not is_cancelled():
                sleep_for = 0.0
                with gate_lock:
                    now = time.time()
                    target = max(
                        float(adaptive_state.get("next_request_time", 0.0)),
                        float(adaptive_state.get("cooldown_until", 0.0)),
                    )
                    if now >= target:
                        gap = max(0.0, float(adaptive_state.get("request_gap_sec", 0.0)))
                        adaptive_state["next_request_time"] = now + gap
                        return True
                    sleep_for = max(0.02, min(0.35, target - now))
                sleep_with_cancel(sleep_for)
            return False

        def note_request_success(*, request_bytes: int) -> None:
            if request_bytes <= 0:
                return
            with gate_lock:
                stable = float(adaptive_state.get("stable_requests", 0.0)) + 1.0
                adaptive_state["stable_requests"] = stable
                gap = max(0.0, float(adaptive_state.get("request_gap_sec", 0.0)))
                floor = max(0.0, float(self.download_request_gap_min_sec))
                if gap > floor and stable >= max(3.0, float(max(1, connections // 2))):
                    # Additive increase / multiplicative decrease (toward faster) after stable requests.
                    adaptive_state["request_gap_sec"] = max(floor, gap * 0.82 - 0.015)
                    adaptive_state["stable_requests"] = 0.0

        def note_request_failure(err: Exception) -> None:
            now = time.time()
            with gate_lock:
                adaptive_state["stable_requests"] = 0.0
                gap = max(0.0, float(adaptive_state.get("request_gap_sec", 0.0)))
                gap_max = max(0.2, float(self.download_request_gap_max_sec))
                if self._is_rate_limited_error(err):
                    burst_merge_window = 1.2
                    last_bump = float(adaptive_state.get("last_rl_bump", 0.0))
                    in_burst = (now - last_bump) < burst_merge_window
                    if not in_burst:
                        rl_attempt = int(adaptive_state.get("rate_limit_hits", 0.0)) + 1
                        adaptive_state["rate_limit_hits"] = float(rl_attempt)
                        adaptive_state["last_rl_bump"] = now
                        adaptive_state["request_gap_sec"] = min(
                            gap_max,
                            max(0.08, (gap * 1.32) if gap > 0 else 0.10),
                        )
                    else:
                        rl_attempt = max(1, int(adaptive_state.get("rate_limit_hits", 0.0)))
                        # Collapse a burst of simultaneous 429s into one penalty event.
                        adaptive_state["request_gap_sec"] = min(
                            gap_max,
                            max(gap, 0.08),
                        )
                    cooldown = self._retry_delay_sec(err, attempt=max(1, rl_attempt), mode="parallel")
                    adaptive_state["cooldown_until"] = max(
                        float(adaptive_state.get("cooldown_until", 0.0)),
                        now + max(0.3, cooldown),
                    )
                    last_emit = float(adaptive_state.get("last_rl_status_emit", 0.0))
                    if (now - last_emit) >= self.download_rate_limit_status_emit_interval_sec:
                        adaptive_state["last_rl_status_emit"] = now
                        self.emit(
                            "status",
                            text=(
                                "Segment scheduler entered rate-limit safe mode: "
                                f"request-gap~{adaptive_state['request_gap_sec']:.2f}s"
                            ),
                        )
                    return
                if self._is_unexpected_eof_error(err):
                    adaptive_state["request_gap_sec"] = min(
                        gap_max,
                        max(0.03, (gap * 1.15) if gap > 0 else 0.03),
                    )
                    return
                status = self._http_status_from_error(err)
                if status in {500, 502, 503, 504, 408, 425}:
                    adaptive_state["request_gap_sec"] = min(
                        gap_max,
                        max(0.05, (gap * 1.3) if gap > 0 else 0.05),
                    )

        def add_progress(delta: int, *, force_emit: bool = False) -> None:
            nonlocal done_bytes, session_bytes, last_emit
            if delta < 0:
                return
            if delta:
                with progress_lock:
                    done_bytes += delta
                    session_bytes += delta
                    done_snapshot = done_bytes
                    session_snapshot = session_bytes
            else:
                with progress_lock:
                    done_snapshot = done_bytes
                    session_snapshot = session_bytes
            now = time.time()
            with emit_lock:
                if (
                    not force_emit
                    and (now - last_emit) < self.download_progress_emit_interval_sec
                    and done_snapshot < total_bytes
                ):
                    return
                last_emit = now
            elapsed = max(0.001, now - start_time)
            self.emit(
                "download_progress",
                done=done_snapshot,
                total=total_bytes,
                speed_bps=(session_snapshot / elapsed),
                mode="parallel",
                connections=connections,
            )

        def flush_resume_state(*, force: bool = False) -> None:
            nonlocal completed_ranges, last_resume_flush, pending_resume_segments
            snapshot: Optional[list[tuple[int, int]]] = None
            with resume_lock:
                if not force and pending_resume_segments <= 0:
                    return
                now = time.time()
                if (
                    not force
                    and pending_resume_segments < self.download_resume_flush_segments
                    and (now - last_resume_flush) < self.download_resume_flush_interval_sec
                ):
                    return
                completed_ranges = self._normalize_completed_ranges(completed_ranges, total_bytes=total_bytes)
                snapshot = list(completed_ranges)
                pending_resume_segments = 0
                last_resume_flush = now
            if snapshot is None:
                return
            try:
                self._write_resume_state(
                    archive_path,
                    total_bytes=total_bytes,
                    completed_ranges=snapshot,
                )
            except Exception:
                # Resume metadata persistence failure should not abort a successful download.
                pass

        def mark_segment_completed(seg_start: int, seg_end: int) -> None:
            nonlocal completed_ranges
            nonlocal pending_resume_segments
            with resume_lock:
                completed_ranges.append((int(seg_start), int(seg_end)))
                pending_resume_segments += 1
            flush_resume_state(force=False)

        if resumed_bytes > 0:
            add_progress(0, force_emit=True)

        def queue_size_hint() -> int:
            try:
                return max(0, int(ranges_queue.qsize()))
            except Exception:
                return 0

        def maybe_dynamic_split(current_offset: int, current_end: int) -> int:
            """
            Adaptively split a long in-flight segment when the queue runs dry.

            This reduces tail latency by returning a tail subrange for another worker,
            similar to IDM-style dynamic segment splitting.
            """
            nonlocal dynamic_splits
            if connections <= 1 or is_cancelled():
                return current_end
            with gate_lock:
                cooldown_until = float(adaptive_state.get("cooldown_until", 0.0))
                req_gap = float(adaptive_state.get("request_gap_sec", 0.0))
            if cooldown_until > time.time() or req_gap >= 0.20:
                # In rate-limit safe mode, reduce request fan-out and avoid extra range requests.
                return current_end
            remaining = int(current_end) - int(current_offset) + 1
            if remaining < DYNAMIC_SPLIT_MIN_REMAINING:
                return current_end
            if queue_size_hint() > 0:
                return current_end

            # Use observed throughput to pick a tail chunk target (~2s of data), clamped.
            with progress_lock:
                session_snapshot = session_bytes
            elapsed = max(0.001, time.time() - start_time)
            avg_bps = session_snapshot / elapsed if session_snapshot > 0 else 0.0
            throughput_tail = int(avg_bps * 2.0) if avg_bps > 0 else 0
            tail_target = max(
                DYNAMIC_SPLIT_MIN_TAIL,
                min(
                    max(DYNAMIC_SPLIT_MIN_REMAINING, piece_size),
                    throughput_tail or piece_size,
                    remaining // 2,
                ),
            )
            if tail_target < DYNAMIC_SPLIT_MIN_TAIL:
                return current_end

            new_end = int(current_end) - int(tail_target)
            if (new_end - int(current_offset) + 1) < DYNAMIC_SPLIT_MIN_TAIL:
                new_end = int(current_offset) + (remaining // 2) - 1
            if new_end <= int(current_offset) or new_end >= int(current_end):
                return current_end

            tail_start = new_end + 1
            tail_len = int(current_end) - tail_start + 1
            if tail_len < DYNAMIC_SPLIT_MIN_TAIL:
                return current_end

            with split_lock:
                if queue_size_hint() > 0 or is_cancelled():
                    return current_end
                ranges_queue.put((tail_start, int(current_end)))
                dynamic_splits += 1

            return new_end

        def validate_range_response(resp, expected_start: int, expected_end: int) -> None:
            status = int(getattr(resp, "status", resp.getcode()))
            if status != 206:
                raise RuntimeError(f"Server did not honor Range request (HTTP {status}).")
            content_range = str(resp.headers.get("Content-Range") or "")
            total_from_cr = self._parse_total_from_content_range(content_range)
            if total_from_cr and total_from_cr != total_bytes:
                raise RuntimeError("Content-Length changed during download.")
            lower = content_range.lower()
            if not lower.startswith("bytes "):
                return
            try:
                span = content_range.split(" ", 1)[1].split("/", 1)[0]
                got_start_s, got_end_s = span.split("-", 1)
                got_start = int(got_start_s.strip())
                got_end = int(got_end_s.strip())
            except Exception:
                return
            if got_start != expected_start or got_end > expected_end:
                raise RuntimeError("Server returned unexpected byte range.")

        def worker_thread(worker_index: int) -> None:
            try:
                with tmp_path.open("r+b", buffering=0) as out_f:
                    while not is_cancelled():
                        try:
                            start, end = ranges_queue.get_nowait()
                        except queue.Empty:
                            return

                        offset = int(start)
                        for attempt in range(1, DOWNLOAD_PART_RETRIES + 1):
                            if is_cancelled():
                                return
                            mirror_idx: Optional[int] = None
                            req_start_offset = int(offset)
                            try:
                                if not wait_for_request_slot():
                                    return
                                mirror_pick = acquire_mirror()
                                if mirror_pick is None:
                                    return
                                mirror_idx, req_url = mirror_pick
                                req = urllib.request.Request(
                                    req_url,
                                    headers=self._download_headers({"Range": f"bytes={offset}-{end}"}),
                                )
                                with urllib.request.urlopen(req, timeout=self.download_timeout_sec) as resp:
                                    validate_range_response(resp, offset, end)
                                    out_f.seek(offset)
                                    while offset <= end:
                                        if is_cancelled():
                                            return
                                        split_end = maybe_dynamic_split(offset, end)
                                        if split_end != end:
                                            end = split_end
                                        to_read = min(self.download_range_io_chunk_size, end - offset + 1)
                                        chunk = resp.read(to_read)
                                        if not chunk:
                                            break
                                        out_f.write(chunk)
                                        offset += len(chunk)
                                        add_progress(len(chunk))

                                if offset > end:
                                    mark_segment_completed(start, end)
                                    release_mirror(mirror_idx, None)
                                    mirror_idx = None
                                    note_request_success(request_bytes=(offset - req_start_offset))
                                    break
                                raise RuntimeError(
                                    f"Unexpected EOF in segment {start}-{end} (worker {worker_index})"
                                )
                            except Exception as e:
                                release_mirror(mirror_idx, e)
                                mirror_idx = None
                                note_request_failure(e)
                                if offset > start:
                                    # Commit already-written prefix immediately so resume metadata survives
                                    # repeated EOF/rate-limit interruptions (IDM-style segmented resume).
                                    committed_end = int(offset) - 1
                                    if committed_end >= int(start):
                                        mark_segment_completed(start, committed_end)
                                        start = int(offset)
                                    if start > end:
                                        note_request_success(request_bytes=max(0, int(offset) - int(req_start_offset)))
                                        break
                                if attempt >= DOWNLOAD_PART_RETRIES:
                                    raise
                                if self._is_rate_limited_error(e):
                                    # Shared request gate/cooldown manages most of the wait; keep local sleep tiny.
                                    sleep_with_cancel(0.05)
                                else:
                                    sleep_with_cancel(
                                        max(0.05, self._retry_delay_sec(e, attempt=attempt, mode="parallel-part"))
                                    )
                                continue
            except Exception as e:
                cancel_event.set()
                error_queue.put(e)

        workers = [
            threading.Thread(target=worker_thread, args=(idx,), name=f"DLPart-{idx}", daemon=True)
            for idx in range(connections)
        ]
        for t in workers:
            t.start()
        for t in workers:
            t.join()

        if self.cancel_event.is_set():
            raise InstallerCancelled(self._cancel_message())
        flush_resume_state(force=True)

        if not error_queue.empty():
            first_error = error_queue.get()
            # Keep partial file + resume metadata for true resume on next run.
            raise RuntimeError(f"Parallel download failed: {first_error}") from first_error

        with progress_lock:
            done_snapshot = done_bytes
        if done_snapshot != total_bytes:
            raise RuntimeError(
                f"Downloaded bytes mismatch (expected {total_bytes}, got {done_snapshot})."
            )

        add_progress(0, force_emit=True)  # final progress event
        if dynamic_splits > 0:
            self.emit(
                "status",
                text=f"Segmented download complete (dynamic splits: {dynamic_splits}).",
            )
        tmp_path.replace(archive_path)
        try:
            state_path = self._archive_resume_state_path(archive_path)
            if state_path.exists():
                state_path.unlink()
        except Exception:
            pass

    def _download_archive(self, archive_path: Path) -> None:
        self._raise_if_cancelled()
        candidate_urls = list(getattr(self, "payload_urls", None) or [self.payload_url])
        reordered_urls = self._reorder_payload_urls_by_route(candidate_urls)
        if reordered_urls:
            self.payload_urls = reordered_urls
            self.payload_url = reordered_urls[0]
            candidate_urls = reordered_urls
        mirror_count = len(candidate_urls)
        if mirror_count > 1:
            route_note = hf_route_probe_note()
            if route_note:
                self.emit(
                    "status",
                    text=f"Multi-source mirror mode enabled ({mirror_count} URLs). {route_note}",
                )
            else:
                self.emit("status", text=f"Multi-source mirror mode enabled ({mirror_count} URLs).")
        self.emit("status", text="Probing server capabilities...")
        total_bytes, range_supported, probe_note = self._probe_parallel_download()
        connections = self._choose_parallel_connections(total_bytes)

        if range_supported and total_bytes > 0 and connections > 1:
            attempted_conns: list[int] = []
            parallel_error: Optional[Exception] = None
            current_connections = max(1, int(connections))
            max_rounds = max(2, int(self.download_parallel_round_cap))

            while True:
                attempted_conns.append(current_connections)
                self.emit(
                    "status",
                    text=(
                        f"Downloading payload via segmented mode "
                        f"({max(1, current_connections)} connections, {human_bytes(total_bytes)})..."
                    ),
                )
                try:
                    self._download_archive_parallel(
                        archive_path,
                        total_bytes=total_bytes,
                        connections=current_connections,
                    )
                    return
                except InstallerCancelled:
                    raise
                except Exception as e:
                    parallel_error = e
                    rate_limited = self._is_rate_limited_error(e)
                    transient = self._is_transient_download_error(e)
                    round_idx = len(attempted_conns)
                    next_connections = current_connections
                    if current_connections > 1:
                        next_connections = max(1, current_connections // 2)
                        if current_connections > 2 and next_connections in attempted_conns:
                            next_connections = max(1, current_connections - 1)

                    if transient and round_idx < max_rounds:
                        if rate_limited:
                            delay_sec = self._retry_delay_sec(e, attempt=round_idx, mode="parallel")
                            if next_connections < current_connections:
                                self.emit(
                                    "status",
                                    text=(
                                        "Segmented download rate-limited "
                                        f"({self._short_error_text(e)}). "
                                        f"Waiting {int(round(delay_sec))}s, then retrying with {next_connections} connection(s)..."
                                    ),
                                )
                            else:
                                self.emit(
                                    "status",
                                    text=(
                                        "Segmented download still rate-limited at safe mode "
                                        f"({self._short_error_text(e)}). "
                                        f"Waiting {int(round(delay_sec))}s, then resuming segmented mode..."
                                    ),
                                )
                            self._sleep_with_cancel(max(0.5, delay_sec))
                            current_connections = next_connections
                            continue

                        if self._is_unexpected_eof_error(e):
                            if next_connections < current_connections:
                                self.emit(
                                    "status",
                                    text=(
                                        "Segmented transfer interrupted by EOF; "
                                        f"continuing segmented resume with {next_connections} connection(s)..."
                                    ),
                                )
                                current_connections = next_connections
                            else:
                                self.emit(
                                    "status",
                                    text="Segmented transfer interrupted by EOF; continuing segmented resume...",
                                )
                            continue

                        # Generic transient error: keep segmented resume and retry once more.
                        delay_sec = self._retry_delay_sec(e, attempt=round_idx, mode="parallel")
                        if next_connections < current_connections:
                            self.emit(
                                "status",
                                text=(
                                    "Segmented download transient failure "
                                    f"({self._short_error_text(e)}). "
                                    f"Waiting {int(round(delay_sec))}s, then retrying with "
                                    f"{next_connections} connection(s)..."
                                ),
                            )
                        else:
                            self.emit(
                                "status",
                                text=(
                                    "Segmented download transient failure "
                                    f"({self._short_error_text(e)}). "
                                    f"Waiting {int(round(delay_sec))}s, then retrying segmented resume..."
                                    ),
                                )
                        self._sleep_with_cancel(max(0.3, min(8.0, delay_sec)))
                        current_connections = next_connections
                        continue

                    break

            self.emit(
                "status",
                text=(
                    f"Segmented download failed ({parallel_error}). "
                    "Falling back to single connection..."
                ),
            )
            recovered_bytes = 0
            try:
                recovered_bytes = self._compact_segmented_workspace_for_single_resume(
                    archive_path,
                    total_bytes_hint=total_bytes,
                )
            except Exception as e:
                self.emit(
                    "status",
                    text=(
                        "Failed to convert segmented resume workspace "
                        f"({self._short_error_text(e)}); restarting single-connection download."
                    ),
                )
                self._cleanup_download_temp_files(archive_path)
            else:
                if recovered_bytes > 0 and total_bytes > 0:
                    self.emit(
                        "status",
                        text=(
                            "Switching to single-connection resume "
                            f"({human_bytes(recovered_bytes)} / {human_bytes(total_bytes)})..."
                        ),
                    )
                elif recovered_bytes > 0:
                    self.emit(
                        "status",
                        text=(
                            "Switching to single-connection resume "
                            f"({human_bytes(recovered_bytes)} already downloaded)..."
                        ),
                    )
            self._download_archive_single(
                archive_path,
                total_bytes_hint=total_bytes,
                range_supported_hint=range_supported,
            )
            return
        if range_supported and total_bytes > 0 and connections <= 1:
            if probe_note:
                probe_note = f"{probe_note}; parallel disabled (connections=1)"
            else:
                probe_note = "parallel disabled (connections=1)"

        note = probe_note or "server limitation"
        if total_bytes > 0:
            self.emit(
                "status",
                text=f"Downloading payload archive (single connection; {note}, {human_bytes(total_bytes)})...",
            )
        else:
            self.emit("status", text=f"Downloading payload archive (single connection; {note})...")
        self._download_archive_single(
            archive_path,
            total_bytes_hint=total_bytes,
            range_supported_hint=range_supported,
        )

    def _extract_archive(self, archive_path: Path, extract_dir: Path) -> None:
        self.emit("status", text="Extracting payload archive (trying system tar for speed)...")
        ok, msg = try_extract_with_tar(archive_path, extract_dir)
        if ok:
            self.emit("extract_done_tar")
            return

        if archive_path.suffix.lower() == ".zip":
            self.emit("status", text=f"System tar unavailable ({msg}). Falling back to Python zip extractor...")
            safe_extract_zip_python(
                archive_path,
                extract_dir,
                progress_cb=lambda done, total: self.emit("extract_progress", done=done, total=total),
            )
            return

        raise RuntimeError(
            f"System tar extraction failed ({msg}). "
            "This installer payload is .tar.zst and requires Windows tar.exe with zstd support."
        )

    def _write_manifest(
        self,
        *,
        shortcut_records: Optional[list[dict]] = None,
        shortcut_warnings: Optional[list[str]] = None,
    ) -> None:
        manifest_path = self.install_dir / MANIFEST_NAME
        payload = {
            "app_name": APP_NAME,
            "main_exe_name": MAIN_EXE_NAME,
            "uninstall_exe_name": UNINSTALL_EXE_NAME,
            "payload_url": self.payload_url,
            "payload_urls": list(getattr(self, "payload_urls", None) or [self.payload_url]),
            "payload_sha256": self.payload_sha256,
            "installed_at_utc": utc_now_iso(),
            "install_dir": normalize_user_path_text(self.install_dir) or str(self.install_dir),
            "shortcuts": shortcut_records or [],
            "shortcut_warnings": shortcut_warnings or [],
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class InstallerApp:
    def __init__(self, root: "tk_types.Tk", payload_url: str, install_dir: Path, payload_sha256: str):
        self.root = root
        self.root.title(f"{APP_NAME} Setup")
        self.root.geometry("820x520")
        self.root.minsize(760, 500)
        self.root.configure(bg="#eef2f6")

        self.worker: Optional[InstallerWorker] = None
        self.installing = False
        self.completed = False
        self.payload_sha256 = payload_sha256
        self._closing_after_cancel = False
        self._cancelling = False
        self._cancel_clears_workspace = False

        self.url_var = tk.StringVar(value=payload_url)
        self.dir_var = tk.StringVar(value=normalize_user_path_text(install_dir) or str(install_dir))
        self.status_var = tk.StringVar(value="Ready")
        self.detail_var = tk.StringVar(value="Set the payload URL and install directory, then click Install.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.desktop_shortcut_var = tk.BooleanVar(value=True)
        self.start_menu_shortcut_var = tk.BooleanVar(value=True)
        self.auto_launch_var = tk.BooleanVar(value=True)

        self._configure_style()
        self._build_ui()
        self.root.after(100, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("App.TFrame", background="#eef2f6")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("HeaderTitle.TLabel", background="#17324d", foreground="#ffffff", font=("Segoe UI Semibold", 16))
        style.configure("HeaderSub.TLabel", background="#17324d", foreground="#d7e6f5", font=("Segoe UI", 9))
        style.configure("Section.TLabelframe", background="#ffffff")
        style.configure("Section.TLabelframe.Label", background="#ffffff", foreground="#17324d", font=("Segoe UI Semibold", 10))
        style.configure("Body.TLabel", background="#ffffff", foreground="#202a33")
        style.configure("Muted.TLabel", background="#ffffff", foreground="#5f6e7d")
        style.configure("Status.TLabel", background="#ffffff", foreground="#0f2f4a", font=("Segoe UI Semibold", 10))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 9))
        style.configure("TCheckbutton", background="#ffffff")
        style.configure("Horizontal.TProgressbar", troughcolor="#dbe4ec", background="#2f7fd1", bordercolor="#dbe4ec")

        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(size=10)
        except Exception:
            pass

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=14, style="App.TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        header = ttk.Frame(container, style="Card.TFrame", padding=(18, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.configure(style="Card.TFrame")
        header["padding"] = (18, 14)
        header.columnconfigure(0, weight=1)
        header_bg = tk.Frame(header, bg="#17324d", bd=0, highlightthickness=0)
        header_bg.grid(row=0, column=0, sticky="ew")
        ttk.Label(header_bg, text=f"{APP_NAME} Setup", style="HeaderTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header_bg,
            text="Download payload, verify integrity, install locally, and create shortcuts.",
            style="HeaderSub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        header_bg.grid_columnconfigure(0, weight=1)
        # Add internal padding on the colored header surface.
        for child in header_bg.winfo_children():
            child.grid_configure(padx=14)
        ttk.Separator(container, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=(12, 10))

        content = ttk.Frame(container, style="App.TFrame")
        content.grid(row=3, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(3, weight=1)

        source_box = ttk.LabelFrame(content, text="Source", padding=12, style="Section.TLabelframe")
        source_box.grid(row=0, column=0, sticky="ew")
        source_box.columnconfigure(0, weight=1)
        source_box.columnconfigure(1, weight=0)
        ttk.Label(source_box, text="Payload URL", style="Body.TLabel").grid(row=0, column=0, sticky="w")
        url_entry = ttk.Entry(source_box, textvariable=self.url_var)
        url_entry.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        dest_box = ttk.LabelFrame(content, text="Destination", padding=12, style="Section.TLabelframe")
        dest_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        dest_box.columnconfigure(0, weight=1)
        ttk.Label(dest_box, text="Install Directory", style="Body.TLabel").grid(row=0, column=0, sticky="w")
        self.dir_entry = ttk.Entry(dest_box, textvariable=self.dir_var)
        self.dir_entry.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.dir_entry.bind("<FocusOut>", self._on_dir_focus_out)
        ttk.Button(dest_box, text="Browse...", command=self._choose_dir).grid(
            row=1, column=1, sticky="e", padx=(8, 0), pady=(6, 0)
        )

        options_box = ttk.LabelFrame(content, text="Options", padding=12, style="Section.TLabelframe")
        options_box.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(options_box, text="Create desktop shortcut", variable=self.desktop_shortcut_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Checkbutton(options_box, text="Create Start Menu shortcuts", variable=self.start_menu_shortcut_var).grid(
            row=0, column=1, sticky="w", padx=(16, 0)
        )
        ttk.Checkbutton(options_box, text="Launch app after install", variable=self.auto_launch_var).grid(
            row=0, column=2, sticky="w", padx=(16, 0)
        )

        progress_box = ttk.LabelFrame(content, text="Progress", padding=12, style="Section.TLabelframe")
        progress_box.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        progress_box.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(progress_box, maximum=100.0, variable=self.progress_var, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")

        ttk.Label(progress_box, textvariable=self.status_var, style="Status.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Label(progress_box, textvariable=self.detail_var, style="Muted.TLabel", wraplength=760, justify="left").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )

        button_bar = ttk.Frame(container, style="App.TFrame")
        button_bar.grid(row=4, column=0, sticky="e", pady=(12, 0))
        self.install_btn = ttk.Button(button_bar, text="Install", command=self._start_install, style="Accent.TButton")
        self.install_btn.pack(side="left")
        self.launch_btn = ttk.Button(button_bar, text="Launch App", command=self._launch_app, state="disabled")
        self.launch_btn.pack(side="left", padx=(8, 0))
        self.open_btn = ttk.Button(button_bar, text="Open Folder", command=self._open_install_dir, state="disabled")
        self.open_btn.pack(side="left", padx=(8, 0))
        self.cancel_btn = ttk.Button(button_bar, text="Cancel Download", command=self._cancel_download, state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.close_btn = ttk.Button(button_bar, text="Close", command=self._on_close)
        self.close_btn.pack(side="left", padx=(8, 0))

    def _begin_cancel(self, *, clear_workspace: bool, close_after_cancel: bool) -> None:
        if self._cancelling:
            return
        self._cancelling = True
        self._closing_after_cancel = bool(close_after_cancel)
        self._cancel_clears_workspace = bool(clear_workspace)
        self.install_btn.configure(state="disabled")
        self.launch_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.cancel_btn.configure(state="disabled")
        self.close_btn.configure(state="disabled" if close_after_cancel else "normal")
        self.status_var.set("Cancelling...")
        if clear_workspace:
            self.detail_var.set("Stopping the current task and clearing installer cache. Please wait...")
        else:
            self.detail_var.set("Stopping the current task and keeping installer cache for resume. Please wait...")
        self.progress.stop()
        self.progress.configure(mode="indeterminate")
        self.progress.start(20)
        if self.worker is not None:
            self.worker.cancel(clear_workspace=clear_workspace)
            return
        self.installing = False
        self._cancelling = False
        if close_after_cancel:
            self.root.destroy()

    def _cancel_download(self) -> None:
        if not self.installing or self._cancelling:
            return
        if not messagebox.askyesno(
            "Cancel Download",
            "Cancelling now will discard the current resume cache and require a fresh download next time.\n\nContinue?",
        ):
            return
        self._begin_cancel(clear_workspace=True, close_after_cancel=False)

    def _on_dir_focus_out(self, _event=None) -> None:
        self._normalize_dir_field()

    def _normalize_dir_field(self) -> None:
        current = self.dir_var.get()
        normalized = normalize_user_path_text(current)
        if normalized and normalized != current:
            self.dir_var.set(normalized)

    def _current_install_dir(self) -> Path:
        self._normalize_dir_field()
        raw = normalize_user_path_text(self.dir_var.get()) or str(DEFAULT_INSTALL_DIR)
        return Path(raw).expanduser()

    def _choose_dir(self) -> None:
        current_dir = self._current_install_dir()
        initial_dir = current_dir.parent if _paths_equal(current_dir.name, APP_NAME) else current_dir
        selected = filedialog.askdirectory(
            initialdir=normalize_user_path_text(initial_dir) or str(DEFAULT_INSTALL_DIR.parent)
        )
        if selected:
            resolved = suggest_install_dir_from_folder_pick(Path(selected), current_install_dir=current_dir)
            self.dir_var.set(normalize_user_path_text(resolved))
            if not _paths_equal(resolved, selected):
                self.detail_var.set(f"Using app subfolder: {resolved}")

    def _open_install_dir(self) -> None:
        install_dir = self._current_install_dir()
        if not install_dir.exists():
            messagebox.showwarning("Missing Folder", f"Install directory not found:\n{install_dir}")
            return
        try:
            if os.name == "nt":
                os.startfile(str(install_dir))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(install_dir)])
        except Exception as e:
            messagebox.showerror("Open Folder Failed", str(e))

    def _start_install(self) -> None:
        if self.installing:
            return

        payload_url = self.url_var.get().strip()
        if not payload_url:
            messagebox.showerror("Missing URL", "Please provide the payload archive URL first.")
            return

        install_dir = self._current_install_dir()
        self.dir_var.set(normalize_user_path_text(install_dir))
        validation_error = validate_install_dir_choice(install_dir)
        if validation_error:
            messagebox.showerror("Invalid Install Directory", validation_error)
            return
        if install_dir.exists():
            try:
                has_contents = any(install_dir.iterdir())
            except Exception:
                has_contents = True
            if has_contents:
                overwrite = messagebox.askyesno(
                    "Replace Existing Installation",
                    f"{install_dir}\n\nThis folder already exists and will be removed before install.\nContinue?",
                )
                if not overwrite:
                    return

        self.worker = InstallerWorker(
            payload_url=payload_url,
            install_dir=install_dir,
            payload_sha256=self.payload_sha256,
            create_desktop_shortcut=bool(self.desktop_shortcut_var.get()),
            create_start_menu_shortcut=bool(self.start_menu_shortcut_var.get()),
        )
        self.worker.start()
        self.installing = True
        self.completed = False
        self._closing_after_cancel = False
        self._cancelling = False
        self._cancel_clears_workspace = False
        self.install_btn.configure(state="disabled")
        self.launch_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.close_btn.configure(state="normal")
        self.status_var.set("Installing...")
        self.detail_var.set("Downloading and extracting payload.")
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress_var.set(0.0)

    def _poll_events(self) -> None:
        if self.worker is not None:
            while True:
                try:
                    event = self.worker.events.get_nowait()
                except queue.Empty:
                    break
                self._handle_event(event)
        self.root.after(100, self._poll_events)

    def _handle_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "status":
            self.status_var.set(str(event.get("text") or ""))
            return

        if event_type == "download_progress":
            done = int(event.get("done") or 0)
            total = int(event.get("total") or 0)
            speed = float(event.get("speed_bps") or 0.0)
            mode = str(event.get("mode") or "")
            connections = int(event.get("connections") or 0)
            mode_suffix = ""
            if mode == "parallel":
                mode_suffix = f" [segmented x{max(1, connections)}]"
            elif mode == "single":
                mode_suffix = " [single]"
            if total > 0:
                pct = min(70.0, (done / total) * 70.0)
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.progress_var.set(pct)
                self.detail_var.set(
                    f"Download{mode_suffix}: {human_bytes(done)} / {human_bytes(total)} ({human_bytes(speed)}/s)"
                )
            else:
                self.progress.configure(mode="indeterminate")
                if str(self.progress.cget("mode")) == "indeterminate":
                    self.progress.start(20)
                self.detail_var.set(f"Download{mode_suffix}: {human_bytes(done)} ({human_bytes(speed)}/s)")
            return

        if event_type == "verify_progress":
            done = int(event.get("done") or 0)
            total = int(event.get("total") or 0)
            self.progress.stop()
            self.progress.configure(mode="determinate")
            if total > 0:
                self.progress_var.set(70.0 + (done / total) * 10.0)
            self.detail_var.set(f"Verifying SHA256: {human_bytes(done)} / {human_bytes(total)}")
            return

        if event_type == "extract_done_tar":
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress_var.set(95.0)
            self.detail_var.set("Extraction complete via system tar.")
            return

        if event_type == "extract_progress":
            done = int(event.get("done") or 0)
            total = int(event.get("total") or 0)
            self.progress.stop()
            self.progress.configure(mode="determinate")
            if total > 0:
                self.progress_var.set(80.0 + (done / total) * 15.0)
            self.detail_var.set(f"Extracting (Python): {human_bytes(done)} / {human_bytes(total)}")
            return

        if event_type == "done":
            self.installing = False
            self.completed = True
            self._cancelling = False
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress_var.set(100.0)
            install_dir = normalize_user_path_text(event.get("install_dir") or self.dir_var.get()) or self.dir_var.get()
            self.dir_var.set(normalize_user_path_text(install_dir) or str(install_dir))
            self.status_var.set("Installation completed")

            shortcuts = event.get("shortcuts") or []
            shortcut_warnings = [str(x) for x in (event.get("shortcut_warnings") or []) if str(x).strip()]
            uninstall_path = Path(install_dir) / UNINSTALL_EXE_NAME
            lines = [f"Installed to: {install_dir}"]
            if uninstall_path.exists():
                lines.append(f"Uninstaller: {uninstall_path.name}")
            else:
                lines.append(f"Warning: {UNINSTALL_EXE_NAME} not found in payload.")
            lines.append(f"Shortcuts created: {len(shortcuts)}")
            if shortcut_warnings:
                lines.append("Shortcut warnings: " + " | ".join(shortcut_warnings[:2]))
            self.detail_var.set("\n".join(lines))

            main_exe_path = Path(install_dir) / MAIN_EXE_NAME
            self.install_btn.configure(state="normal")
            self.launch_btn.configure(state="normal" if main_exe_path.exists() else "disabled")
            self.open_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self.close_btn.configure(state="normal")

            info_lines = [f"{APP_NAME} installed successfully.", "", install_dir]
            if shortcut_warnings:
                info_lines.extend(["", "Shortcut warnings:"] + shortcut_warnings[:3])
            self.worker = None
            if self._closing_after_cancel:
                self.root.after(0, self.root.destroy)
                return
            messagebox.showinfo("Install Complete", "\n".join(info_lines))
            if bool(self.auto_launch_var.get()):
                self._launch_app()
            return

        if event_type == "cancelled":
            self.installing = False
            self.completed = False
            self._cancelling = False
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress_var.set(0.0)
            self.status_var.set("Cancelled")
            if self._cancel_clears_workspace:
                self.detail_var.set("Stopped the current task and cleared the installer cache.")
            else:
                self.detail_var.set("Stopped the current task and kept the installer cache for resume.")
            self.install_btn.configure(state="normal")
            self.launch_btn.configure(state="disabled")
            self.open_btn.configure(state="disabled")
            self.cancel_btn.configure(state="disabled")
            self.close_btn.configure(state="normal")
            self.worker = None
            if self._closing_after_cancel:
                self.root.after(0, self.root.destroy)
            return

        if event_type == "error":
            self.installing = False
            self.completed = False
            self._cancelling = False
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.install_btn.configure(state="normal")
            self.launch_btn.configure(state="disabled")
            self.open_btn.configure(state="disabled")
            self.cancel_btn.configure(state="disabled")
            self.close_btn.configure(state="normal")
            msg = str(event.get("message") or "Unknown error")
            self.status_var.set("Installation failed")
            self.detail_var.set(msg)
            self.worker = None
            if self._closing_after_cancel:
                self.root.after(0, self.root.destroy)
                return
            messagebox.showerror("Install Failed", msg)
            return

    def _launch_app(self) -> None:
        install_dir = self._current_install_dir()
        exe_path = install_dir / MAIN_EXE_NAME
        if not exe_path.exists():
            messagebox.showwarning("Missing App", f"Main executable not found:\n{exe_path}")
            return
        try:
            kwargs = {"cwd": str(install_dir)}
            if os.name == "nt":
                kwargs["creationflags"] = 0x00000008  # DETACHED_PROCESS
            subprocess.Popen([str(exe_path)], **kwargs)
        except Exception as e:
            messagebox.showerror("Launch Failed", str(e))

    def _on_close(self) -> None:
        if self.installing:
            if self._cancelling:
                return
            choice = messagebox.askyesnocancel(
                "Close Installer",
                "Installation is still running.\n\nYes: close and keep cache for resume.\nNo: close and clear the current cache.\nCancel: continue downloading.",
            )
            if choice is None:
                return
            self._begin_cancel(clear_workspace=(choice is False), close_after_cancel=True)
            return
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} bootstrap installer")
    parser.add_argument("--url", default=DEFAULT_PAYLOAD_URL, help="Payload archive URL (.tar.zst)")
    parser.add_argument("--sha256", default=DEFAULT_PAYLOAD_SHA256, help="Optional payload SHA256")
    parser.add_argument("--dir", default=str(DEFAULT_INSTALL_DIR), help="Install directory")
    return parser.parse_args()


def main() -> int:
    if tk is None:
        raise RuntimeError("tkinter is unavailable in this build/environment.") from _TK_IMPORT_ERROR
    args = parse_args()
    root = tk.Tk()
    app = InstallerApp(
        root=root,
        payload_url=str(args.url or ""),
        install_dir=Path(normalize_user_path_text(args.dir) or str(DEFAULT_INSTALL_DIR)).expanduser(),
        payload_sha256=str(args.sha256 or ""),
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
