from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bootstrap_installer as legacy  # noqa: E402
from dist_config import (  # noqa: E402
    APP_NAME,
    DEFAULT_INSTALL_DIR,
    DEFAULT_INNO_CORE_BIN_URLS,
    DEFAULT_PAYLOAD_SHA256,
    DEFAULT_PAYLOAD_URL,
    ICON_CANDIDATES,
    MAIN_EXE_NAME,
    UNINSTALL_EXE_NAME,
)


INSTALL_SCOPE_PER_USER = "per-user"
INSTALL_SCOPE_ALL_USERS = "all-users"
INNO_CORE_SETUP_NAME = "MediaTranscribeStudio-Setup-Core.exe"


def _hide_console_window_if_present() -> None:
    if os.name != "nt":
        return
    try:
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


def _set_windows_app_user_model_id(app_id: str) -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        pass


def _resolve_qt_icon() -> Optional[QIcon]:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        try:
            candidates.append(Path(sys.executable))
        except Exception:
            pass
    for rel in ICON_CANDIDATES:
        candidates.append(THIS_DIR.parent / rel)
        candidates.append(THIS_DIR / rel)
    candidates.append(THIS_DIR.parent / "app.ico")
    for path in candidates:
        try:
            if not path.exists():
                continue
            icon = QIcon(str(path))
            if not icon.isNull():
                return icon
        except Exception:
            continue
    return None


def _runtime_path_candidates(name: str) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    env_override = str(os.environ.get("INNO_CORE_SETUP_PATH", "")).strip().strip('"')
    if env_override:
        candidates.append(Path(env_override))

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / name)

    if getattr(sys, "frozen", False):
        try:
            candidates.append(Path(sys.executable).resolve().parent / name)
        except Exception:
            pass

    candidates.extend(
        [
            THIS_DIR / name,
            THIS_DIR.parent / "dist" / name,
            THIS_DIR.parent / name,
        ]
    )

    out: list[Path] = []
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(rp)
    return out


def find_inno_core_setup_binary() -> Optional[Path]:
    for p in _runtime_path_candidates(INNO_CORE_SETUP_NAME):
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def _url_path_lower(url: str) -> str:
    try:
        return urllib.parse.urlparse(str(url or "")).path.lower()
    except Exception:
        return str(url or "").lower()


def _looks_like_exe_url(url: str) -> bool:
    return _url_path_lower(url).endswith(".exe")


def _hf_url_candidates(url: str) -> list[str]:
    try:
        return legacy.hf_url_candidates(url)
    except Exception:
        raw = str(url or "").strip()
        return [raw] if raw else []


def normalize_scope(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"all", "all-users", "machine", "admin"}:
        return INSTALL_SCOPE_ALL_USERS
    return INSTALL_SCOPE_PER_USER


def _normalized_path_key(value: object) -> str:
    return legacy.normalized_path_key(value)


def default_install_dir_for_scope(scope: str) -> Path:
    if normalize_scope(scope) == INSTALL_SCOPE_ALL_USERS:
        return Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / APP_NAME
    return DEFAULT_INSTALL_DIR


def windows_desktop_dir(scope: str = INSTALL_SCOPE_PER_USER) -> Path:
    if normalize_scope(scope) == INSTALL_SCOPE_ALL_USERS:
        return Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "Desktop"
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"


def windows_start_menu_programs_dir(scope: str = INSTALL_SCOPE_PER_USER) -> Path:
    if normalize_scope(scope) == INSTALL_SCOPE_ALL_USERS:
        return (
            Path(os.environ.get("ProgramData", r"C:\ProgramData"))
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
        )
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


def is_windows_admin() -> bool:
    if os.name != "nt":
        return True
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def relaunch_self_elevated(extra_args: list[str]) -> bool:
    if os.name != "nt":
        raise RuntimeError("Elevation is only supported on Windows.")
    if getattr(sys, "frozen", False):
        exe = sys.executable
        params = subprocess.list2cmdline(extra_args)
    else:
        exe = sys.executable
        params = subprocess.list2cmdline([str(Path(__file__).resolve()), *extra_args])
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", exe, params, None, 1)
    return int(rc) > 32


def create_install_shortcuts(
    install_dir: Path,
    *,
    create_desktop: bool,
    create_start_menu: bool,
    scope: str,
) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    warnings: list[str] = []
    main_exe = install_dir / MAIN_EXE_NAME
    uninstall_exe = install_dir / UNINSTALL_EXE_NAME
    scope = normalize_scope(scope)

    if not main_exe.exists():
        warnings.append(f"Main executable not found for shortcut creation: {main_exe}")
        return records, warnings

    def _try_create(link_path: Path, target: Path, kind: str, description: str) -> None:
        try:
            legacy.create_windows_shortcut(
                link_path=link_path,
                target_path=target,
                working_dir=install_dir,
                icon_path=main_exe,
                description=description,
            )
            records.append({"path": str(link_path), "target": str(target), "kind": kind, "scope": scope})
        except Exception as e:
            warnings.append(str(e))

    if create_desktop:
        _try_create(windows_desktop_dir(scope) / f"{APP_NAME}.lnk", main_exe, "desktop", f"Launch {APP_NAME}")
    if create_start_menu:
        start_dir = windows_start_menu_programs_dir(scope) / APP_NAME
        _try_create(start_dir / f"{APP_NAME}.lnk", main_exe, "start_menu", f"Launch {APP_NAME}")
        if uninstall_exe.exists():
            _try_create(
                start_dir / f"Uninstall {APP_NAME}.lnk",
                uninstall_exe,
                "start_menu_uninstall",
                f"Uninstall {APP_NAME}",
            )
        else:
            warnings.append(f"Uninstaller not found for Start Menu shortcut: {uninstall_exe}")
    return records, warnings


class InstallerWorker(legacy.InstallerWorker):
    @staticmethod
    def _parse_explicit_inno_bin_urls(values: Optional[list[str]]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in (values or []):
            text = str(raw or "").replace("\r", "\n")
            for part in text.replace("|", "\n").split("\n"):
                url = str(part or "").strip().strip('"').strip("'")
                if not url:
                    continue
                key = url.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(url)
        return out

    def __init__(
        self,
        *args,
        install_scope: str = INSTALL_SCOPE_PER_USER,
        inno_core_bin_urls: Optional[list[str]] = None,
        auto_launch_after_install: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.install_scope = normalize_scope(install_scope)
        # Explicit split-bin URLs should remain 1:1 with user input.
        # Do not expand mirror/official candidates here, otherwise 3 URLs can become 6.
        self.inno_core_bin_urls = self._parse_explicit_inno_bin_urls(inno_core_bin_urls)
        self.auto_launch_after_install = bool(auto_launch_after_install)

    def _inno_task_names(self) -> list[str]:
        tasks: list[str] = []
        if self.create_desktop_shortcut:
            tasks.append("desktopicon")
        if self.create_start_menu_shortcut:
            tasks.append("startmenuicon")
        if self.auto_launch_after_install:
            tasks.append("autolaunch")
        return tasks

    def _inno_task_cli_args(self) -> list[str]:
        known_tasks = ("desktopicon", "startmenuicon", "autolaunch")
        selected = self._inno_task_names()
        selected_set = {x.strip().lower() for x in selected if str(x).strip()}
        deselected = [name for name in known_tasks if name not in selected_set]

        args: list[str] = []
        if selected:
            args.append(f"/TASKS={','.join(selected)}")
        if deselected:
            args.append("/MERGETASKS=" + ",".join(f"!{name}" for name in deselected))
        return args

    def _expected_shortcut_records(self) -> list[dict]:
        scope = normalize_scope(self.install_scope)
        main_exe = self.install_dir / MAIN_EXE_NAME
        uninstall_target = self.install_dir / UNINSTALL_EXE_NAME
        if not uninstall_target.exists():
            # Inno's generated uninstaller.
            uninstall_target = self.install_dir / "unins000.exe"

        records: list[dict] = []
        if self.create_desktop_shortcut:
            records.append(
                {
                    "path": str(windows_desktop_dir(scope) / f"{APP_NAME}.lnk"),
                    "target": str(main_exe),
                    "kind": "desktop",
                    "scope": scope,
                }
            )
        if self.create_start_menu_shortcut:
            start_dir = windows_start_menu_programs_dir(scope) / APP_NAME
            records.append(
                {
                    "path": str(start_dir / f"{APP_NAME}.lnk"),
                    "target": str(main_exe),
                    "kind": "start_menu",
                    "scope": scope,
                }
            )
            records.append(
                {
                    "path": str(start_dir / f"Uninstall {APP_NAME}.lnk"),
                    "target": str(uninstall_target),
                    "kind": "start_menu_uninstall",
                    "scope": scope,
                }
            )
        return records

    def _popen_hidden(self, cmd: list[str], *, cwd: Path) -> subprocess.Popen:
        kwargs = {
            "cwd": str(cwd),
        }
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            kwargs["startupinfo"] = startupinfo
        return subprocess.Popen(cmd, **kwargs)  # type: ignore[arg-type]

    def _emit_inno_log_delta(self, log_path: Path, offset: int) -> int:
        try:
            if not log_path.exists():
                return offset
            size = log_path.stat().st_size
            if size < offset:
                offset = 0
            if size == offset:
                return offset
            with log_path.open("rb") as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
            text = chunk.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                self.emit("log", text=f"[inno] {line}")
            return offset
        except Exception:
            return offset

    def _http_exists_single(self, url: str) -> bool:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=15):
                return True
        except urllib.error.HTTPError as e:
            if int(getattr(e, "code", 0) or 0) in {403, 405}:
                # Some CDNs block HEAD; try a small ranged GET.
                try:
                    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
                    with urllib.request.urlopen(req, timeout=15):
                        return True
                except urllib.error.HTTPError as e2:
                    return int(getattr(e2, "code", 0) or 0) not in {404}
                except Exception:
                    return False
            return int(getattr(e, "code", 0) or 0) not in {404}
        except Exception:
            return False

    def _first_existing_http_url(self, url: str) -> str | None:
        for candidate in _hf_url_candidates(url):
            if self._http_exists_single(candidate):
                return candidate
        return None

    def _replace_url_basename(self, base_url: str, new_name: str) -> str:
        parsed = urllib.parse.urlparse(base_url)
        path = parsed.path or ""
        slash = path.rfind("/")
        new_path = (path[: slash + 1] if slash >= 0 else "") + new_name
        return urllib.parse.urlunparse(parsed._replace(path=new_path))

    def _download_url_to_path(self, url: str, dest: Path) -> None:
        old_urls = list(getattr(self, "payload_urls", []) or [])
        old_url = str(getattr(self, "payload_url", "") or "")
        old_preserve_first = bool(getattr(self, "_preserve_first_payload_url", False))
        try:
            raw = str(url or "").strip()
            urls = [raw] if raw else []
            seen = {u.lower() for u in urls}
            for one in _hf_url_candidates(raw):
                key = str(one or "").strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                urls.append(one)
            self.payload_urls = urls or [url]
            self.payload_url = self.payload_urls[0]
            self._preserve_first_payload_url = True
            self._download_archive(dest)
        finally:
            self.payload_urls = old_urls
            self.payload_url = old_url
            self._preserve_first_payload_url = old_preserve_first

    def _bin_fingerprint(self, path: Path) -> tuple[int, str]:
        size = int(path.stat().st_size) if path.exists() else 0
        h = hashlib.sha1()
        if size <= 0:
            return (0, "0")
        with path.open("rb") as f:
            head = f.read(1024 * 1024)
            h.update(head)
            if size > 2 * 1024 * 1024:
                f.seek(max(0, size - 1024 * 1024))
                tail = f.read(1024 * 1024)
                h.update(tail)
        return (size, h.hexdigest())

    def _download_inno_core_bundle(self, core_url: str, *, temp_root: Path) -> Path:
        bundle_dir = temp_root / "inno-core-download"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        parsed_path = urllib.parse.urlparse(core_url).path or ""
        core_name = Path(parsed_path).name or INNO_CORE_SETUP_NAME
        if not core_name.lower().endswith(".exe"):
            core_name = INNO_CORE_SETUP_NAME
        core_path = bundle_dir / core_name

        self.emit("status", text="Downloading Inno core installer...")
        core_candidates = _hf_url_candidates(core_url)
        if len(core_candidates) > 1:
            self.emit("log", text=f"[download] core exe candidates (priority order): {' | '.join(core_candidates)}")
        else:
            self.emit("log", text=f"[download] core exe: {core_url}")
        self._download_url_to_path(core_url, core_path)

        stem = core_path.stem
        downloaded_bins = 0
        seen_bin_fingerprints: dict[tuple[int, str], Path] = {}
        explicit_bin_urls = list(getattr(self, "inno_core_bin_urls", None) or [])
        if explicit_bin_urls:
            self.emit("log", text=f"[download] using explicit split .bin URLs ({len(explicit_bin_urls)})")
            for idx, bin_url in enumerate(explicit_bin_urls, start=1):
                bin_name = f"{stem}-{idx}.bin"
                bin_path = bundle_dir / bin_name
                self.emit("status", text=f"Downloading Inno core split file {idx}...")
                self.emit("log", text=f"[download] split bin (explicit): {bin_url}")
                self._download_url_to_path(bin_url, bin_path)
                fp = self._bin_fingerprint(bin_path)
                same_as = seen_bin_fingerprints.get(fp)
                if same_as is not None:
                    raise RuntimeError(
                        "Duplicate split .bin detected from explicit URLs: "
                        f"{same_as.name} and {bin_path.name} have identical fingerprint. "
                        "Please verify uploaded split files and --bin-url list."
                    )
                seen_bin_fingerprints[fp] = bin_path
                downloaded_bins += 1
            if downloaded_bins > 0:
                self.emit("log", text=f"[download] downloaded {downloaded_bins} explicit split .bin file(s)")
            return core_path

        max_auto_bins = 64
        try:
            max_auto_bins = max(1, min(256, int(os.environ.get("INNO_CORE_AUTO_BIN_MAX", "64"))))
        except Exception:
            max_auto_bins = 64

        for idx in range(1, max_auto_bins + 1):
            bin_name = f"{stem}-{idx}.bin"
            bin_url = self._replace_url_basename(core_url, bin_name)
            existing_bin_url = self._first_existing_http_url(bin_url)
            if not existing_bin_url:
                if idx == 1:
                    self.emit("log", text="[download] no split .bin files detected for Inno core")
                break
            bin_path = bundle_dir / bin_name
            self.emit("status", text=f"Downloading Inno core split file {idx}...")
            self.emit("log", text=f"[download] split bin: {existing_bin_url}")
            self._download_url_to_path(existing_bin_url, bin_path)
            fp = self._bin_fingerprint(bin_path)
            same_as = seen_bin_fingerprints.get(fp)
            if same_as is not None:
                raise RuntimeError(
                    "Detected duplicate split .bin while auto-probing online files: "
                    f"{same_as.name} and {bin_path.name} are identical. "
                    "This usually means stale/invalid remote split files; "
                    "please upload only the current bundle or use explicit --bin-url entries."
                )
            seen_bin_fingerprints[fp] = bin_path
            downloaded_bins += 1

        if downloaded_bins >= max_auto_bins:
            self.emit(
                "log",
                text=(
                    f"[download] reached auto split-file probe limit ({max_auto_bins}); "
                    "stopping further .bin probing."
                ),
            )
        if downloaded_bins > 0:
            self.emit("log", text=f"[download] downloaded {downloaded_bins} split .bin file(s)")

        return core_path

    def _copy_inno_bundle_files(self, source_exe: Path, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        runtime_core_exe = dest_dir / source_exe.name
        shutil.copy2(source_exe, runtime_core_exe)
        try:
            for bin_file in sorted(source_exe.parent.glob(f"{source_exe.stem}-*.bin")):
                if bin_file.is_file():
                    shutil.copy2(bin_file, dest_dir / bin_file.name)
        except Exception:
            pass
        return runtime_core_exe

    def _installer_workspace_root(self) -> Path:
        # Keep temp data on the selected destination drive/path instead of the
        # system temp directory (often C:).
        return self.install_dir / ".installer-work"

    def _run_via_inno_core(self, inno_core_source: Path) -> None:
        temp_root = self._installer_workspace_root()
        core_dir = temp_root / "inno-core"
        runtime_core_exe = core_dir / INNO_CORE_SETUP_NAME
        log_path = temp_root / "inno-core-install.log"
        try:
            self._raise_if_cancelled()
            self.emit("status", text="Preparing embedded Inno installer...")
            core_dir.mkdir(parents=True, exist_ok=True)
            try:
                for stale in core_dir.iterdir():
                    if stale.is_file():
                        stale.unlink()
            except Exception:
                pass
            runtime_core_exe = self._copy_inno_bundle_files(inno_core_source, core_dir)
            try:
                if log_path.exists():
                    log_path.unlink()
            except Exception:
                pass

            scope_switch = "/ALLUSERS" if self.install_scope == INSTALL_SCOPE_ALL_USERS else "/CURRENTUSER"
            task_args = self._inno_task_cli_args()
            cmd = [
                str(runtime_core_exe),
                "/SP-",
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                scope_switch,
                f"/DIR={str(self.install_dir)}",
                f"/LOG={str(log_path)}",
            ]
            cmd.extend(task_args)

            self.emit("log", text=f"[core] launching embedded Inno core: {runtime_core_exe}")
            if task_args:
                self.emit("log", text=f"[core] task args: {' '.join(task_args)}")
            self.emit("status", text="Installing files with Inno core...")
            proc = self._popen_hidden(cmd, cwd=core_dir)

            log_offset = 0
            last_status_ts = 0.0
            while proc.poll() is None:
                if self.cancel_event.is_set():
                    self.emit("status", text="Cancelling Inno core installer...")
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    raise legacy.InstallerCancelled(self._cancel_message())
                log_offset = self._emit_inno_log_delta(log_path, log_offset)
                now = time.monotonic()
                if now - last_status_ts >= 0.8:
                    self.emit("status", text="Installing files with Inno core...")
                    last_status_ts = now
                time.sleep(0.12)

            rc = int(proc.wait())
            log_offset = self._emit_inno_log_delta(log_path, log_offset)
            if rc != 0:
                tail = ""
                try:
                    if log_path.exists():
                        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        tail = "\n".join(lines[-20:]).strip()
                except Exception:
                    tail = ""
                msg = f"Inno core installer failed (exit code {rc})."
                if tail:
                    msg += "\n\nRecent Inno log:\n" + tail
                raise RuntimeError(msg)

            self.emit("status", text="Finalizing installation...")
            shortcut_records = self._expected_shortcut_records()
            shortcut_warnings: list[str] = []
            try:
                self._write_manifest(shortcut_records=shortcut_records, shortcut_warnings=shortcut_warnings)
            except Exception as e:
                shortcut_warnings.append(f"Failed to write install manifest: {e}")
                self.emit("log", text=f"[warn] {shortcut_warnings[-1]}")

            self.emit(
                "done",
                install_dir=str(self.install_dir),
                install_scope=self.install_scope,
                backend="inno-core",
                shortcuts=shortcut_records,
                shortcut_warnings=shortcut_warnings,
            )
        except legacy.InstallerCancelled as e:
            self.emit("cancelled", message=str(e))
        except Exception as e:
            self.emit("error", message=str(e))
        finally:
            try:
                if temp_root.exists():
                    shutil.rmtree(temp_root, ignore_errors=True)
            except Exception:
                pass

    def _run(self) -> None:
        self._raise_if_cancelled()
        inno_core = find_inno_core_setup_binary()
        if inno_core is not None:
            self._run_via_inno_core(inno_core)
            return

        if _looks_like_exe_url(self.payload_url):
            temp_root = self._installer_workspace_root()
            downloaded_core: Optional[Path] = None
            try:
                temp_root.mkdir(parents=True, exist_ok=True)
                downloaded_core = self._download_inno_core_bundle(self.payload_url, temp_root=temp_root)
            except legacy.InstallerCancelled as e:
                self.emit("cancelled", message=str(e))
                return
            except Exception as e:
                self.emit("error", message=str(e))
                return
            finally:
                if downloaded_core is None and self.cancel_event.is_set() and self.clear_workspace_on_cancel:
                    try:
                        if temp_root.exists():
                            shutil.rmtree(temp_root, ignore_errors=True)
                    except Exception:
                        pass
            self._run_via_inno_core(downloaded_core)
            return

        temp_root = self._installer_workspace_root()
        archive_path = temp_root / legacy.PAYLOAD_ARCHIVE_NAME
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
                digest = legacy.sha256sum(
                    archive_path,
                    progress_cb=lambda done, total: self.emit("verify_progress", done=done, total=total),
                )
                if digest != self.payload_sha256:
                    raise RuntimeError(
                        f"SHA256 mismatch.\nExpected: {self.payload_sha256}\nActual:   {digest}"
                    )

            self._raise_if_cancelled()
            self._extract_archive(archive_path, extract_dir)

            self._raise_if_cancelled()
            payload_root = legacy.flatten_payload_root(extract_dir)
            self.emit("status", text=f"Installing to {self.install_dir} ...")
            legacy.deploy_tree(payload_root, self.install_dir)

            self._raise_if_cancelled()
            self.emit("status", text="Creating shortcuts...")
            shortcut_records, shortcut_warnings = create_install_shortcuts(
                self.install_dir,
                create_desktop=self.create_desktop_shortcut,
                create_start_menu=self.create_start_menu_shortcut,
                scope=self.install_scope,
            )
            self._raise_if_cancelled()
            self._write_manifest(shortcut_records=shortcut_records, shortcut_warnings=shortcut_warnings)
            self.emit(
                "done",
                install_dir=str(self.install_dir),
                install_scope=self.install_scope,
                backend="payload-archive",
                shortcuts=shortcut_records,
                shortcut_warnings=shortcut_warnings,
            )
            run_ok = True
        except legacy.InstallerCancelled as e:
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
                elif keep_resume_workspace and extract_dir.exists():
                    shutil.rmtree(extract_dir, ignore_errors=True)
            except Exception:
                pass

    def _write_manifest(
        self,
        *,
        shortcut_records: Optional[list[dict]] = None,
        shortcut_warnings: Optional[list[str]] = None,
    ) -> None:
        manifest_path = self.install_dir / legacy.MANIFEST_NAME
        payload = {
            "app_name": APP_NAME,
            "main_exe_name": MAIN_EXE_NAME,
            "uninstall_exe_name": UNINSTALL_EXE_NAME,
            "payload_url": self.payload_url,
            "payload_urls": list(getattr(self, "payload_urls", None) or [self.payload_url]),
            "payload_sha256": self.payload_sha256,
            "installed_at_utc": legacy.utc_now_iso(),
            "install_dir": str(self.install_dir),
            "install_scope": self.install_scope,
            "shortcuts": shortcut_records or [],
            "shortcut_warnings": shortcut_warnings or [],
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class InstallerWindow(QMainWindow):
    def __init__(
        self,
        *,
        payload_url: str,
        inno_core_bin_urls: Optional[list[str]],
        payload_sha256: str,
        install_dir: Path,
        scope: str,
        show_url: bool,
        auto_start: bool,
        create_desktop_shortcut: bool,
        create_start_menu_shortcut: bool,
        auto_launch: bool,
        elevated: bool,
    ) -> None:
        super().__init__()
        scope = normalize_scope(scope)
        recommended_dir = default_install_dir_for_scope(scope)
        requested_dir = Path(legacy.normalize_user_path_text(install_dir) or str(recommended_dir)).expanduser()
        custom_dir_requested = (
            _normalized_path_key(requested_dir) != _normalized_path_key(recommended_dir)
        )
        self.setWindowTitle(f"{APP_NAME} Setup")
        self.resize(980, 720)
        self.setMinimumSize(860, 640)

        self.worker: Optional[InstallerWorker] = None
        self.installing = False
        self.completed = False
        self.payload_sha256 = (payload_sha256 or "").strip()
        self._inno_core_bin_urls = InstallerWorker._parse_explicit_inno_bin_urls(inno_core_bin_urls)
        self._auto_start_requested = bool(auto_start)
        self._closing_after_cancel = False
        self._cancelling = False
        self._cancel_clears_workspace = False
        # Track the last auto-filled (recommended) path separately so a custom
        # path from CLI/elevated relaunch is not treated as an auto path.
        self._last_auto_dir = legacy.normalize_user_path_text(recommended_dir) or str(recommended_dir)
        self._dir_dirty = bool(custom_dir_requested)
        self._process_elevated = bool(elevated or is_windows_admin())
        self._embedded_inno_core = find_inno_core_setup_binary()

        self._build_ui()
        self._apply_styles()
        self._bind()

        self.url_edit.setText(payload_url or "")
        self.url_edit.setEchoMode(QLineEdit.Normal if (show_url or not payload_url) else QLineEdit.Password)
        self.reveal_btn.setChecked(bool(show_url))
        self.reveal_btn.setText("隐藏" if show_url else "显示")
        self.source_adv_box.setVisible(bool(show_url or (not payload_url and self._embedded_inno_core is None)))
        if self._embedded_inno_core is not None:
            self.url_edit.setEnabled(False)
            self.reveal_btn.setEnabled(False)
        self.scope_user_radio.setChecked(scope == INSTALL_SCOPE_PER_USER)
        self.scope_all_radio.setChecked(scope == INSTALL_SCOPE_ALL_USERS)
        self.dir_edit.setText(legacy.normalize_user_path_text(requested_dir) or str(requested_dir))
        self.desktop_cb.setChecked(create_desktop_shortcut)
        self.startmenu_cb.setChecked(create_start_menu_shortcut)
        self.autolaunch_cb.setChecked(auto_launch)
        self._update_source_status()
        self._update_scope_state()

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._poll_events)
        self.timer.start()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        header = QFrame(root)
        header.setObjectName("Header")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(18, 16, 18, 16)
        hl.setSpacing(4)
        t = QLabel(f"{APP_NAME} Setup", header)
        t.setObjectName("HeaderTitle")
        s = QLabel("PySide6 科幻风安装器 | 支持仅为我安装 / 为所有人安装（管理员）", header)
        s.setObjectName("HeaderSub")
        s.setText("官方安装器 | 支持仅为我安装 / 为所有人安装（管理员）")
        hl.addWidget(t)
        hl.addWidget(s)
        outer.addWidget(header)

        inf = QHBoxLayout()
        self.badge = QLabel("", root)
        self.badge.setObjectName("Badge")
        self.scope_hint = QLabel("", root)
        self.scope_hint.setObjectName("Hint")
        inf.addWidget(self.badge)
        inf.addWidget(self.scope_hint, 1)
        outer.addLayout(inf)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid, 1)

        self.source_box = self._card("下载源")
        grid.addWidget(self.source_box, 0, 0)
        sb = self.source_box.layout()
        self.source_status = QLabel("", self.source_box)
        self.source_status.setObjectName("Muted")
        self.source_status.setWordWrap(True)
        sb.addWidget(self.source_status)
        self.source_adv_box = QFrame(self.source_box)
        self.source_adv_box.setObjectName("Inset")
        adv = QVBoxLayout(self.source_adv_box)
        adv.setContentsMargins(10, 10, 10, 10)
        adv.setSpacing(6)
        ul = QLabel("Payload URL（默认隐藏）", self.source_adv_box)
        ul.setObjectName("Field")
        adv.addWidget(ul)
        ur = QHBoxLayout()
        self.url_edit = QLineEdit(self.source_adv_box)
        self.url_edit.setPlaceholderText("https://.../MediaTranscribeStudio-payload.tar.zst")
        self.reveal_btn = QToolButton(self.source_adv_box)
        self.reveal_btn.setObjectName("GhostTool")
        self.reveal_btn.setCheckable(True)
        self.reveal_btn.setText("显示")
        ur.addWidget(self.url_edit, 1)
        ur.addWidget(self.reveal_btn)
        adv.addLayout(ur)
        tip = QLabel("默认用内置链接，界面中隐藏显示；需要调试时再展开修改。", self.source_adv_box)
        tip.setObjectName("Mono")
        tip.setWordWrap(True)
        adv.addWidget(tip)
        sb.addWidget(self.source_adv_box)

        self.dest_box = self._card("安装位置与范围")
        grid.addWidget(self.dest_box, 0, 1)
        db = self.dest_box.layout()
        scope_row = QHBoxLayout()
        self.scope_group = QButtonGroup(self)
        self.scope_user_radio = QRadioButton("仅为我安装", self.dest_box)
        self.scope_all_radio = QRadioButton("为所有人安装（管理员）", self.dest_box)
        self.scope_group.addButton(self.scope_user_radio)
        self.scope_group.addButton(self.scope_all_radio)
        scope_row.addWidget(self.scope_user_radio)
        scope_row.addWidget(self.scope_all_radio)
        scope_row.addStretch(1)
        db.addLayout(scope_row)
        dl = QLabel("安装目录", self.dest_box)
        dl.setObjectName("Field")
        db.addWidget(dl)
        dr = QHBoxLayout()
        self.dir_edit = QLineEdit(self.dest_box)
        self.browse_btn = QPushButton("浏览...", self.dest_box)
        self.browse_btn.setObjectName("Ghost")
        self.reset_dir_btn = QPushButton("推荐路径", self.dest_box)
        self.reset_dir_btn.setObjectName("Ghost")
        dr.addWidget(self.dir_edit, 1)
        dr.addWidget(self.browse_btn)
        dr.addWidget(self.reset_dir_btn)
        db.addLayout(dr)
        self.scope_note = QLabel("", self.dest_box)
        self.scope_note.setObjectName("Muted")
        self.scope_note.setWordWrap(True)
        db.addWidget(self.scope_note)

        self.options_box = self._card("安装选项")
        grid.addWidget(self.options_box, 1, 0)
        ob = self.options_box.layout()
        self.desktop_cb = QCheckBox("创建桌面快捷方式", self.options_box)
        self.startmenu_cb = QCheckBox("创建开始菜单快捷方式", self.options_box)
        self.autolaunch_cb = QCheckBox("安装完成后自动启动", self.options_box)
        ob.addWidget(self.desktop_cb)
        ob.addWidget(self.startmenu_cb)
        ob.addWidget(self.autolaunch_cb)
        ob.addStretch(1)

        self.progress_box = self._card("进度与日志")
        grid.addWidget(self.progress_box, 1, 1)
        pb = self.progress_box.layout()
        self.progress = QProgressBar(self.progress_box)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        pb.addWidget(self.progress)
        self.status_label = QLabel("待命", self.progress_box)
        self.status_label.setObjectName("Status")
        self.detail_label = QLabel("配置完成后点击开始安装。", self.progress_box)
        self.detail_label.setObjectName("Muted")
        self.detail_label.setWordWrap(True)
        pb.addWidget(self.status_label)
        pb.addWidget(self.detail_label)
        self.log_view = QPlainTextEdit(self.progress_box)
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("Log")
        pb.addWidget(self.log_view, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.install_btn = QPushButton("开始安装", root)
        self.install_btn.setObjectName("Primary")
        self.launch_btn = QPushButton("启动程序", root)
        self.launch_btn.setObjectName("Ghost")
        self.open_btn = QPushButton("打开目录", root)
        self.open_btn.setObjectName("Ghost")
        self.cancel_btn = QPushButton("取消下载", root)
        self.cancel_btn.setObjectName("Ghost")
        self.close_btn = QPushButton("关闭", root)
        self.close_btn.setObjectName("Ghost")
        self.launch_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        for w in (self.install_btn, self.launch_btn, self.open_btn, self.cancel_btn, self.close_btn):
            w.setMinimumHeight(38)
            btns.addWidget(w)
        outer.addLayout(btns)

    def _card(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setObjectName("Card")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        return box

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget#Root { background: #050c12; color: #e8fbff; font-family: "Microsoft YaHei UI", "Segoe UI"; }
            QFrame#Header {
                border-radius: 14px;
                border: 1px solid rgba(88, 230, 255, 90);
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #071d2e, stop:0.6 #0a2f44, stop:1 #12192f);
            }
            QLabel#HeaderTitle { font-size: 22pt; font-weight: 800; letter-spacing: 1px; color: #f5feff; }
            QLabel#HeaderSub { color: rgba(215, 246, 255, 220); }
            QLabel#Badge {
                border-radius: 10px; padding: 5px 10px; font-weight: 700;
                background: rgba(0, 195, 255, 25); border: 1px solid rgba(0, 225, 255, 85); color: #aef7ff;
            }
            QLabel#Hint, QLabel#Muted { color: rgba(193, 230, 238, 210); }
            QLabel#Field { color: #9cefff; font-weight: 600; }
            QLabel#Mono { color: rgba(167, 208, 219, 180); font-family: Consolas, "Cascadia Mono"; }
            QLabel#Status { color: #d3fbff; font-size: 11pt; font-weight: 700; }
            QGroupBox#Card {
                border: 1px solid rgba(88, 230, 255, 70); border-radius: 14px;
                margin-top: 12px; background: rgba(10, 21, 30, 220); font-weight: 700;
            }
            QGroupBox#Card::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #b5f8ff; }
            QFrame#Inset { border: 1px solid rgba(88, 230, 255, 45); border-radius: 10px; background: rgba(3, 9, 14, 200); }
            QLineEdit {
                border: 1px solid rgba(92, 219, 255, 70); border-radius: 10px; padding: 8px 10px;
                background: rgba(3, 10, 15, 240); color: #ecfeff;
                selection-background-color: rgba(0, 225, 255, 110); selection-color: #00151b;
            }
            QLineEdit:focus { border-color: rgba(111, 242, 255, 180); }
            QLineEdit:disabled {
                color: rgba(236, 254, 255, 150);
                background: rgba(3, 10, 15, 170);
                border-color: rgba(92, 219, 255, 35);
            }
            QProgressBar {
                border: 1px solid rgba(92, 219, 255, 75); border-radius: 10px; text-align: center;
                background: rgba(2, 7, 11, 220); color: #d9fdff; font-weight: 700;
            }
            QProgressBar::chunk {
                border-radius: 9px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #39d8ff, stop:0.55 #32fff1, stop:1 #78ffb1);
            }
            QPlainTextEdit#Log {
                border: 1px solid rgba(88, 230, 255, 38); border-radius: 10px;
                background: rgba(2, 7, 11, 220); color: #9fefff; font-family: Consolas, "Cascadia Mono";
            }
            QPushButton#Primary {
                border-radius: 10px; padding: 0 14px; font-weight: 800; color: #001b21;
                border: 1px solid rgba(170, 255, 238, 170);
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #4ef7ff, stop:0.6 #46d2ff, stop:1 #75ffae);
            }
            QPushButton#Ghost, QToolButton#GhostTool {
                border-radius: 10px; padding: 0 12px; color: #defcff;
                border: 1px solid rgba(92, 219, 255, 70); background: rgba(9, 18, 27, 210);
            }
            QPushButton#Ghost:disabled { color: rgba(222, 252, 255, 90); border-color: rgba(92,219,255,20); }
            QCheckBox, QRadioButton { color: #e4fbff; }
            QCheckBox::indicator, QRadioButton::indicator { width: 18px; height: 18px; }
            QCheckBox::indicator {
                border-radius: 5px; border: 1px solid rgba(125, 236, 255, 110);
                background: rgba(5, 12, 18, 235);
            }
            QCheckBox::indicator:hover { border-color: rgba(125, 236, 255, 180); }
            QCheckBox::indicator:checked {
                border: 1px solid rgba(170, 255, 228, 220);
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #43e7ff, stop:1 #6dffb5);
            }
            QCheckBox::indicator:disabled {
                border-color: rgba(125, 236, 255, 45);
                background: rgba(5, 12, 18, 140);
            }
            QCheckBox::indicator:checked:disabled {
                border-color: rgba(170, 255, 228, 80);
                background: rgba(67, 231, 255, 90);
            }
            QRadioButton::indicator {
                border-radius: 9px; border: 1px solid rgba(125, 236, 255, 110);
                background: rgba(5, 12, 18, 235);
            }
            QRadioButton::indicator:hover { border-color: rgba(125, 236, 255, 180); }
            QRadioButton::indicator:checked {
                border: 1px solid rgba(170, 255, 228, 220);
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.65, fx:0.5, fy:0.5,
                    stop:0 #e9ffff, stop:0.33 #9affeb, stop:0.34 rgba(7, 13, 19, 255), stop:1 rgba(7, 13, 19, 255));
            }
            """
        )

    def _bind(self) -> None:
        self.close_btn.clicked.connect(self._request_close)
        self.cancel_btn.clicked.connect(self._request_cancel_download)
        self.install_btn.clicked.connect(self._start_install)
        self.launch_btn.clicked.connect(self._launch_app)
        self.open_btn.clicked.connect(self._open_dir)
        self.browse_btn.clicked.connect(self._choose_dir)
        self.reset_dir_btn.clicked.connect(self._reset_dir)
        self.url_edit.textChanged.connect(self._update_source_status)
        self.reveal_btn.toggled.connect(self._toggle_reveal)
        self.dir_edit.textEdited.connect(self._mark_dir_dirty)
        self.scope_user_radio.toggled.connect(self._on_scope_changed)
        self.scope_all_radio.toggled.connect(self._on_scope_changed)

    def _set_busy_controls(self, busy: bool) -> None:
        locked = bool(busy)
        for widget in (
            self.install_btn,
            self.reveal_btn,
            self.browse_btn,
            self.reset_dir_btn,
            self.scope_user_radio,
            self.scope_all_radio,
            self.desktop_cb,
            self.startmenu_cb,
            self.autolaunch_cb,
        ):
            widget.setEnabled(not locked)
        if self._embedded_inno_core is not None and not locked:
            self.reveal_btn.setEnabled(False)
            self.url_edit.setEnabled(False)
        else:
            self.url_edit.setEnabled(not locked)
        self.dir_edit.setEnabled(not locked)
        if locked:
            self.launch_btn.setEnabled(False)
            self.open_btn.setEnabled(False)
        self.cancel_btn.setEnabled(locked and not self._cancelling)
        self.close_btn.setEnabled(not self._cancelling)

    def _ask_running_close_mode(self) -> Optional[str]:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("关闭安装器")
        box.setText("当前下载/安装仍在进行。关闭前要不要保留缓存？")
        box.setInformativeText("保留缓存后，下次可以继续；清理缓存则会删除当前已下载内容。")
        keep_btn = box.addButton("保留缓存并关闭", QMessageBox.AcceptRole)
        clear_btn = box.addButton("清理缓存并关闭", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("继续下载", QMessageBox.RejectRole)
        box.setDefaultButton(keep_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is keep_btn:
            return "keep"
        if clicked is clear_btn:
            return "clear"
        if clicked is cancel_btn:
            return None
        return None

    def _begin_cancel(self, *, clear_workspace: bool, close_after_cancel: bool) -> None:
        if self._cancelling:
            return
        self._cancelling = True
        self._closing_after_cancel = bool(close_after_cancel)
        self._cancel_clears_workspace = bool(clear_workspace)
        self._set_busy_controls(True)
        self.install_btn.setEnabled(False)
        self.launch_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(not close_after_cancel)
        self.progress.setRange(0, 0)
        self.progress.setFormat("")
        self.status_label.setText("正在取消...")
        if clear_workspace:
            self.detail_label.setText("正在停止当前任务，并清理已下载缓存，请稍候。")
        else:
            self.detail_label.setText("正在停止当前任务，并保留缓存供下次继续，请稍候。")
        action = "clear-cache" if clear_workspace else "keep-cache"
        self._log(f"[cancel] user requested cancel ({action})")
        if self.worker is not None:
            self.worker.cancel(clear_workspace=clear_workspace)
            return
        self.installing = False
        self._cancelling = False
        if close_after_cancel:
            QTimer.singleShot(0, self.close)

    def _request_cancel_download(self) -> None:
        if not self.installing or self._cancelling:
            return
        if (
            QMessageBox.question(
                self,
                "取消下载",
                "取消当前下载/安装后，会放弃断点续传并清理当前缓存。\n\n确定要取消吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self._begin_cancel(clear_workspace=True, close_after_cancel=False)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._auto_start_requested:
            self._auto_start_requested = False
            QTimer.singleShot(250, self._start_install)

    def _log(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.log_view.appendPlainText(text)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _toggle_reveal(self, checked: bool) -> None:
        self.url_edit.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        self.reveal_btn.setText("隐藏" if checked else "显示")

    def _update_source_status(self) -> None:
        if self._embedded_inno_core is not None:
            self.source_status.setText(f"Using embedded Inno core installer: {self._embedded_inno_core.name}")
            return
        has_url = bool(self.url_edit.text().strip())
        explicit_bin_count = len(self._inno_core_bin_urls)
        if not has_url:
            self.source_status.setText("No online installer source URL configured.")
            return
        suffix = (
            f" + {explicit_bin_count} explicit .bin URL(s)"
            if explicit_bin_count > 0
            else " + auto-detect split .bin files"
        )
        if self.source_adv_box.isVisible():
            self.source_status.setText(f"Online source editable (debug mode){suffix}.")
        else:
            self.source_status.setText(f"Using built-in online source{suffix}.")

    def _selected_scope(self) -> str:
        return INSTALL_SCOPE_ALL_USERS if self.scope_all_radio.isChecked() else INSTALL_SCOPE_PER_USER

    def _update_scope_state(self) -> None:
        scope = self._selected_scope()
        admin = is_windows_admin()
        if scope == INSTALL_SCOPE_ALL_USERS:
            self.scope_note.setText(
                f"推荐目录：{default_install_dir_for_scope(scope)}。需要管理员权限（UAC），快捷方式会写入公共桌面/开始菜单。"
            )
            self.scope_hint.setText("安装范围：为所有人安装")
            if admin:
                self.badge.setText("管理员会话 / All Users 已就绪")
                self.badge.setStyleSheet("")
            else:
                self.badge.setText("普通权限 / 安装时将请求 UAC")
                self.badge.setStyleSheet(
                    "QLabel#Badge { border-radius:10px; padding:5px 10px; font-weight:700; "
                    "background: rgba(255,179,0,25); border:1px solid rgba(255,193,7,120); color:#ffe2a1; }"
                )
        else:
            self.scope_note.setText(f"推荐目录：{default_install_dir_for_scope(scope)}。无需管理员权限。")
            self.scope_hint.setText("安装范围：仅为当前用户")
            self.badge.setText("用户级安装 / 无需管理员")
            self.badge.setStyleSheet(
                "QLabel#Badge { border-radius:10px; padding:5px 10px; font-weight:700; "
                "background: rgba(0,195,255,25); border:1px solid rgba(0,225,255,85); color:#aef7ff; }"
            )

    def _mark_dir_dirty(self, _text: str) -> None:
        self._dir_dirty = True

    def _recommended_dir(self) -> Path:
        return default_install_dir_for_scope(self._selected_scope())

    def _on_scope_changed(self, _checked: bool) -> None:
        new_default = legacy.normalize_user_path_text(self._recommended_dir()) or str(self._recommended_dir())
        current = self.dir_edit.text().strip()
        if (not current) or (not self._dir_dirty) or (_normalized_path_key(current) == _normalized_path_key(self._last_auto_dir)):
            self.dir_edit.setText(new_default)
            self._last_auto_dir = new_default
            self._dir_dirty = False
        self._update_scope_state()

    def _choose_dir(self) -> None:
        current_dir = Path(legacy.normalize_user_path_text(self.dir_edit.text().strip()) or str(self._recommended_dir()))
        base_dir = current_dir.parent if _normalized_path_key(current_dir.name) == _normalized_path_key(APP_NAME) else current_dir
        base = legacy.normalize_user_path_text(base_dir) or str(base_dir)
        selected = QFileDialog.getExistingDirectory(self, "选择安装目录", base)
        if selected:
            resolved = legacy.suggest_install_dir_from_folder_pick(Path(selected), current_install_dir=current_dir)
            self.dir_edit.setText(legacy.normalize_user_path_text(resolved) or str(resolved))
            self._dir_dirty = True

    def _reset_dir(self) -> None:
        self.dir_edit.setText(legacy.normalize_user_path_text(self._recommended_dir()) or str(self._recommended_dir()))
        self._last_auto_dir = self.dir_edit.text().strip()
        self._dir_dirty = False

    def _open_dir(self) -> None:
        target = Path(legacy.normalize_user_path_text(self.dir_edit.text().strip()) or str(self._recommended_dir()))
        if not target.exists():
            QMessageBox.warning(self, "目录不存在", f"找不到目录：\n{target}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
            QMessageBox.warning(self, "打开失败", f"无法打开目录：\n{target}")

    def _confirm_cancel_and_close(self) -> None:
        if self._cancelling:
            return
        mode = self._ask_running_close_mode()
        if mode is None:
            return
        self._begin_cancel(clear_workspace=(mode == "clear"), close_after_cancel=True)

    def _request_close(self) -> None:
        if self.installing:
            self._confirm_cancel_and_close()
            return
        self.close()

    def _elevation_args(self) -> list[str]:
        args = [
            "--url",
            self.url_edit.text().strip(),
            "--sha256",
            self.payload_sha256,
            "--dir",
            legacy.normalize_user_path_text(self.dir_edit.text().strip()) or str(self._recommended_dir()),
            "--scope",
            self._selected_scope(),
            "--auto-start",
            "--elevated",
        ]
        for bin_url in self._inno_core_bin_urls:
            args.extend(["--bin-url", bin_url])
        if self.reveal_btn.isChecked():
            args.append("--show-url")
        if not self.desktop_cb.isChecked():
            args.append("--no-desktop-shortcut")
        if not self.startmenu_cb.isChecked():
            args.append("--no-start-menu-shortcut")
        if not self.autolaunch_cb.isChecked():
            args.append("--no-auto-launch")
        return args

    def _start_install(self) -> None:
        if self.installing:
            return
        payload_url = self.url_edit.text().strip()
        if not payload_url and self._embedded_inno_core is None:
            QMessageBox.critical(
                self,
                "Missing Payload URL",
                "No online installer source URL is configured. Please launch with --url.",
            )
            return
        if not payload_url and self._embedded_inno_core is not None:
            payload_url = "embedded://inno-core"
        scope = self._selected_scope()
        install_dir = Path(legacy.normalize_user_path_text(self.dir_edit.text().strip()) or str(self._recommended_dir())).expanduser()
        self.dir_edit.setText(legacy.normalize_user_path_text(install_dir) or str(install_dir))
        validation_error = legacy.validate_install_dir_choice(install_dir)
        if validation_error:
            QMessageBox.warning(self, "Invalid Install Directory", validation_error)
            return

        if scope == INSTALL_SCOPE_ALL_USERS and os.name == "nt" and not is_windows_admin():
            if (
                QMessageBox.question(
                    self,
                    "Administrator Permission Required",
                    "All-users install requires administrator permission. Relaunch elevated and continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                != QMessageBox.Yes
            ):
                return
            try:
                ok = relaunch_self_elevated(self._elevation_args())
            except Exception as e:
                QMessageBox.critical(self, "Elevation Failed", str(e))
                return
            if ok:
                self.close()
            else:
                QMessageBox.warning(self, "Elevation Cancelled", "Administrator permission was not granted. Install cancelled.")
            return

        if install_dir.exists():
            try:
                has_contents = any(install_dir.iterdir())
            except Exception:
                has_contents = True
            if has_contents and (
                QMessageBox.warning(
                    self,
                    "Overwrite Existing Directory",
                    "The install directory exists and may be overwritten:\n\n"
                    f"{install_dir}\n\n"
                    "Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                return

        self.worker = InstallerWorker(
            payload_url=payload_url,
            install_dir=install_dir,
            payload_sha256=self.payload_sha256,
            create_desktop_shortcut=self.desktop_cb.isChecked(),
            create_start_menu_shortcut=self.startmenu_cb.isChecked(),
            install_scope=scope,
            inno_core_bin_urls=list(self._inno_core_bin_urls),
            auto_launch_after_install=self.autolaunch_cb.isChecked(),
        )
        self.worker.start()
        self.installing = True
        self.completed = False
        self._closing_after_cancel = False
        self._cancelling = False
        self._cancel_clears_workspace = False
        self._set_busy_controls(True)
        self.status_label.setText("Installing...")
        if self._embedded_inno_core is not None:
            self.detail_label.setText("Using embedded Inno core installer (Inno handles file install/uninstall logic)...")
            self.progress.setRange(0, 0)
            self.progress.setFormat("")
        else:
            self.detail_label.setText("Downloading and preparing payload archive...")
            self.progress.setRange(0, 1000)
            self.progress.setValue(0)
            self.progress.setFormat("%p%")
        self._log(f"[start] scope={scope} admin={'yes' if is_windows_admin() else 'no'} dir={install_dir}")

    def _poll_events(self) -> None:
        if self.worker is None:
            return
        while True:
            try:
                event = self.worker.events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)

    def _set_pct(self, pct: float) -> None:
        self.progress.setValue(max(0, min(1000, int(round(pct * 10)))))

    def _handle_event(self, event: dict) -> None:
        t = str(event.get("type") or "")
        if t == "log":
            self._log(str(event.get("text") or ""))
            return
        if t == "status":
            text_value = str(event.get("text") or "")
            self.status_label.setText(text_value or "Processing...")
            self._log(f"[status] {text_value}")
            return
        if t == "download_progress":
            done = int(event.get("done") or 0)
            total = int(event.get("total") or 0)
            speed = float(event.get("speed_bps") or 0.0)
            if total > 0:
                self._set_pct((done / total) * 70.0)
                self.detail_label.setText(
                    f"Downloading: {legacy.human_bytes(done)} / {legacy.human_bytes(total)} ({legacy.human_bytes(speed)}/s)"
                )
            else:
                self.detail_label.setText(f"Downloading: {legacy.human_bytes(done)} ({legacy.human_bytes(speed)}/s)")
            return
        if t == "verify_progress":
            done = int(event.get("done") or 0)
            total = int(event.get("total") or 0)
            if total > 0:
                self._set_pct(70.0 + (done / total) * 10.0)
            self.detail_label.setText(f"Verifying SHA256: {legacy.human_bytes(done)} / {legacy.human_bytes(total)}")
            return
        if t == "extract_done_tar":
            self._set_pct(95.0)
            self.detail_label.setText("Extract complete (system tar)")
            self._log("[extract] completed via tar")
            return
        if t == "extract_progress":
            done = int(event.get("done") or 0)
            total = int(event.get("total") or 0)
            if total > 0:
                self._set_pct(80.0 + (done / total) * 15.0)
            self.detail_label.setText(f"Extracting: {legacy.human_bytes(done)} / {legacy.human_bytes(total)}")
            return
        if t == "done":
            self.installing = False
            self.completed = True
            self._cancelling = False
            self._set_busy_controls(False)
            self.progress.setRange(0, 1000)
            self.progress.setFormat("%p%")
            self._set_pct(100.0)
            install_dir = legacy.normalize_user_path_text(event.get("install_dir") or self.dir_edit.text()) or str(
                event.get("install_dir") or self.dir_edit.text()
            )
            scope = str(event.get("install_scope") or self._selected_scope())
            backend = str(event.get("backend") or "")
            self.dir_edit.setText(legacy.normalize_user_path_text(install_dir) or install_dir)
            self.status_label.setText("Installation Complete")
            shortcuts = event.get("shortcuts") or []
            warns = [str(x) for x in (event.get("shortcut_warnings") or []) if str(x).strip()]
            self.detail_label.setText(
                f"Location: {install_dir}\n"
                f"Scope: {scope}\n"
                f"Shortcuts: {len(shortcuts)}"
            )
            self.install_btn.setEnabled(True)
            self.open_btn.setEnabled(True)
            self.launch_btn.setEnabled((Path(legacy.normalize_user_path_text(install_dir) or install_dir) / MAIN_EXE_NAME).exists())
            self._log(f"[done] {install_dir} shortcuts={len(shortcuts)}")
            for w in warns[:5]:
                self._log(f"[warn] {w}")
            self.worker = None
            if self._closing_after_cancel:
                QTimer.singleShot(0, self.close)
                return
            msg = [f"{APP_NAME} installed successfully.", "", install_dir]
            if warns:
                msg.extend(["", "Shortcut warnings:"] + warns[:3])
            QMessageBox.information(self, "Installation Complete", "\n".join(msg))
            if self.autolaunch_cb.isChecked() and backend != "inno-core":
                self._launch_app()
            return
        if t == "cancelled":
            self.installing = False
            self.completed = False
            self._cancelling = False
            self._set_busy_controls(False)
            self.progress.setRange(0, 1000)
            self.progress.setFormat("%p%")
            self.progress.setValue(0)
            msg = str(event.get("message") or "Installation cancelled.")
            self.status_label.setText("已取消")
            if self._cancel_clears_workspace:
                self.detail_label.setText("已停止当前任务，并清理安装缓存。")
            else:
                self.detail_label.setText("已停止当前任务，缓存已保留，可下次继续。")
            self._log(f"[cancelled] {msg}")
            self.worker = None
            if self._closing_after_cancel:
                QTimer.singleShot(0, self.close)
            return
        if t == "error":
            self.installing = False
            self.completed = False
            self._cancelling = False
            self._set_busy_controls(False)
            if self.progress.minimum() == 0 and self.progress.maximum() == 0:
                self.progress.setRange(0, 1000)
                self.progress.setFormat("%p%")
                self.progress.setValue(0)
            self.install_btn.setEnabled(True)
            self.launch_btn.setEnabled(False)
            self.open_btn.setEnabled(False)
            msg = str(event.get("message") or "Unknown error")
            self.status_label.setText("Installation Failed")
            self.detail_label.setText(msg)
            self._log(f"[error] {msg}")
            self.worker = None
            if self._closing_after_cancel:
                QTimer.singleShot(0, self.close)
                return
            QMessageBox.critical(self, "Installation Failed", msg)

    def _launch_app(self) -> None:
        install_dir = Path(legacy.normalize_user_path_text(self.dir_edit.text().strip()) or str(self._recommended_dir()))
        exe_path = install_dir / MAIN_EXE_NAME
        if not exe_path.exists():
            QMessageBox.warning(self, "无法启动", f"未找到主程序：\n{exe_path}")
            return
        try:
            kwargs = {"cwd": str(install_dir)}
            if os.name == "nt":
                kwargs["creationflags"] = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE
                kwargs["startupinfo"] = startupinfo
            subprocess.Popen([str(exe_path)], **kwargs)
            self._log(f"[launch] {exe_path}")
        except Exception as e:
            QMessageBox.critical(self, "启动失败", str(e))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._closing_after_cancel:
            if self.installing:
                event.ignore()
                return
            event.accept()
            return
        if self.installing:
            event.ignore()
            self._confirm_cancel_and_close()
            return
        event.accept()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{APP_NAME} PySide6 bootstrap installer")
    p.add_argument("--url", default=DEFAULT_PAYLOAD_URL, help="Payload archive URL (.tar.zst)")
    p.add_argument(
        "--bin-url",
        dest="bin_urls",
        action="append",
        default=None,
        help="Explicit Inno core split .bin URL (repeat for multiple files)",
    )
    p.add_argument("--sha256", default=DEFAULT_PAYLOAD_SHA256, help="Optional payload SHA256")
    p.add_argument("--dir", default="", help="Install directory (optional)")
    p.add_argument("--scope", default=INSTALL_SCOPE_PER_USER, choices=[INSTALL_SCOPE_PER_USER, INSTALL_SCOPE_ALL_USERS])
    p.add_argument("--show-url", action="store_true", help="Reveal URL field")
    p.add_argument("--auto-start", action="store_true", help="Auto start install on launch")
    p.add_argument("--elevated", action="store_true", help="Marker flag for elevated relaunch")
    p.add_argument("--no-desktop-shortcut", action="store_true")
    p.add_argument("--no-start-menu-shortcut", action="store_true")
    p.add_argument("--no-auto-launch", action="store_true")
    args = p.parse_args()
    if args.bin_urls is None:
        try:
            args.bin_urls = [str(u).strip() for u in (DEFAULT_INNO_CORE_BIN_URLS or []) if str(u).strip()]
        except Exception:
            args.bin_urls = []
    return args


def main() -> int:
    _hide_console_window_if_present()
    _set_windows_app_user_model_id("MediaTranscribeStudio.Setup")
    args = parse_args()
    scope = normalize_scope(str(args.scope))
    install_dir = (
        Path(legacy.normalize_user_path_text(args.dir) or str(default_install_dir_for_scope(scope))).expanduser()
        if str(args.dir or "").strip()
        else default_install_dir_for_scope(scope)
    )
    app = QApplication(sys.argv)
    icon = _resolve_qt_icon()
    if icon is not None:
        try:
            app.setWindowIcon(icon)
        except Exception:
            pass
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    win = InstallerWindow(
        payload_url=str(args.url or ""),
        inno_core_bin_urls=list(getattr(args, "bin_urls", None) or []),
        payload_sha256=str(args.sha256 or ""),
        install_dir=install_dir,
        scope=scope,
        show_url=bool(args.show_url),
        auto_start=bool(args.auto_start),
        create_desktop_shortcut=not bool(args.no_desktop_shortcut),
        create_start_menu_shortcut=not bool(args.no_start_menu_shortcut),
        auto_launch=not bool(args.no_auto_launch),
        elevated=bool(args.elevated),
    )
    if icon is not None:
        try:
            win.setWindowIcon(icon)
        except Exception:
            pass
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
