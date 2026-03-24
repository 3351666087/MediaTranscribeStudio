from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import messagebox, ttk
    _TK_IMPORT_ERROR = None
except Exception as _e:
    # Qt uninstaller reuses backend helpers from this module; tkinter is optional there.
    tk = None  # type: ignore[assignment]
    tkfont = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR = _e


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dist_config import APP_NAME, MAIN_EXE_NAME, UNINSTALL_EXE_NAME  # noqa: E402


MANIFEST_NAME = "install_manifest.json"


def default_target_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def load_manifest(install_dir: Path) -> dict:
    manifest_path = install_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def default_shortcut_records() -> list[dict]:
    start_dir = windows_start_menu_programs_dir() / APP_NAME
    return [
        {"path": str(windows_desktop_dir() / f"{APP_NAME}.lnk"), "kind": "desktop"},
        {"path": str(start_dir / f"{APP_NAME}.lnk"), "kind": "start_menu"},
        {"path": str(start_dir / f"Uninstall {APP_NAME}.lnk"), "kind": "start_menu_uninstall"},
    ]


def manifest_shortcut_records(manifest: dict) -> list[dict]:
    raw = manifest.get("shortcuts")
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        out.append({"path": path, "kind": str(item.get("kind") or "shortcut")})
    return out


def remove_shortcuts_best_effort(manifest: dict) -> list[str]:
    warnings: list[str] = []
    records = manifest_shortcut_records(manifest)
    if not records and "shortcuts" not in manifest:
        records = default_shortcut_records()
    touched_dirs: set[Path] = set()

    for rec in records:
        path_str = str(rec.get("path") or "").strip()
        if not path_str:
            continue
        link_path = Path(path_str)
        try:
            if link_path.exists():
                link_path.unlink()
                touched_dirs.add(link_path.parent)
        except Exception as e:
            warnings.append(f"Failed to remove shortcut {link_path.name}: {e}")

    # Clean up the app-specific Start Menu folder if it is empty (or mostly just leftover links).
    start_menu_dir = windows_start_menu_programs_dir() / APP_NAME
    candidate_dirs = sorted({p for p in touched_dirs if p != windows_desktop_dir()} | {start_menu_dir}, key=str, reverse=True)
    for folder in candidate_dirs:
        try:
            if folder.exists() and folder.is_dir():
                if not any(folder.iterdir()):
                    folder.rmdir()
        except Exception:
            # Non-empty or locked folder is not fatal.
            pass
    return warnings


def _normalized_path_key(value: object) -> str:
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    try:
        raw = str(Path(raw))
    except Exception:
        pass
    try:
        raw = os.path.normpath(raw)
    except Exception:
        pass
    if os.name == "nt":
        raw = os.path.normcase(raw)
    return raw


def _is_path_within_root(candidate_path: object, root_dir: object) -> bool:
    candidate = _normalized_path_key(candidate_path)
    root = _normalized_path_key(root_dir)
    if not candidate or not root:
        return False
    if candidate == root:
        return True
    sep = os.sep if os.sep else "\\"
    if not root.endswith(sep):
        root = root + sep
    return candidate.startswith(root)


def _iter_process_entries_windows() -> list[tuple[int, int, str]]:
    if os.name != "nt":
        return []

    import ctypes.wintypes as wintypes

    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == INVALID_HANDLE_VALUE:
        return []
    entries_out: list[tuple[int, int, str]] = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            return []
        while True:
            try:
                entries_out.append(
                    (
                        int(entry.th32ProcessID),
                        int(entry.th32ParentProcessID),
                        str(entry.szExeFile or ""),
                    )
                )
            except Exception:
                pass
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
        return entries_out
    finally:
        try:
            kernel32.CloseHandle(snapshot)
        except Exception:
            pass


def _iter_process_ids_windows() -> list[int]:
    return [pid for pid, _ppid, _name in _iter_process_entries_windows()]


def _query_process_image_path_windows(pid: int) -> str:
    if os.name != "nt" or int(pid) <= 0:
        return ""

    import ctypes.wintypes as wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return ""
    try:
        # Long-path aware enough for common installed app paths.
        size = wintypes.DWORD(32768)
        buf = ctypes.create_unicode_buffer(int(size.value))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return ""
        return str(buf.value or "")
    finally:
        try:
            kernel32.CloseHandle(handle)
        except Exception:
            pass


def _process_children_index(entries: list[tuple[int, int, str]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for pid, ppid, _name in entries:
        if pid <= 0:
            continue
        children.setdefault(int(ppid), []).append(int(pid))
    return children


def _expand_pids_with_descendants_postorder(
    roots: list[int] | set[int],
    entries: list[tuple[int, int, str]],
    *,
    exclude_pid: int | None = None,
) -> list[int]:
    children = _process_children_index(entries)
    ordered: list[int] = []
    seen: set[int] = set()
    exclude = int(exclude_pid) if exclude_pid is not None else -1

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 3000))

    def _visit(pid: int) -> None:
        if pid <= 0 or pid == exclude or pid in seen:
            return
        seen.add(pid)
        for child_pid in children.get(pid, []):
            _visit(int(child_pid))
        ordered.append(pid)

    for root_pid in roots:
        _visit(int(root_pid))
    return ordered


def _terminate_process_windows(pid: int, *, wait_ms: int = 1200) -> None:
    if os.name != "nt":
        return
    pid = int(pid)
    if pid <= 0:
        return

    PROCESS_TERMINATE = 0x0001
    SYNCHRONIZE = 0x00100000
    desired_access = PROCESS_TERMINATE | SYNCHRONIZE
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(desired_access, False, pid)
    if not handle:
        return
    try:
        try:
            kernel32.TerminateProcess(handle, 1)
        except Exception:
            return
        try:
            kernel32.WaitForSingleObject(handle, int(max(0, wait_ms)))
        except Exception:
            pass
    finally:
        try:
            kernel32.CloseHandle(handle)
        except Exception:
            pass


def _kill_pid_tree_best_effort(
    pid: int,
    *,
    entries: list[tuple[int, int, str]] | None = None,
    exclude_pid: int | None = None,
) -> None:
    pid = int(pid)
    if pid <= 0:
        return
    snapshot = list(entries) if entries is not None else _iter_process_entries_windows()
    ordered = _expand_pids_with_descendants_postorder([pid], snapshot, exclude_pid=exclude_pid)
    for current_pid in ordered:
        _terminate_process_windows(current_pid)


def _kill_processes_by_image_name_best_effort(exe_name: str, *, exclude_pid: int | None = None) -> None:
    target_name = str(exe_name or "").strip().strip('"')
    if not target_name:
        return
    target_name = os.path.basename(target_name).lower()
    if not target_name:
        return

    snapshot = _iter_process_entries_windows()
    root_pids = [
        int(pid)
        for pid, _ppid, image_name in snapshot
        if str(image_name or "").strip().lower() == target_name
    ]
    ordered = _expand_pids_with_descendants_postorder(root_pids, snapshot, exclude_pid=exclude_pid)
    for current_pid in ordered:
        _terminate_process_windows(current_pid)


def _best_effort_kill_processes_in_target_dir(target_dir: Path, *, exclude_pid: int | None = None) -> None:
    if os.name != "nt":
        return
    try:
        target_root = str(target_dir.resolve())
    except Exception:
        target_root = str(target_dir)
    target_root = target_root.strip()
    if not target_root:
        return
    exclude = int(exclude_pid) if exclude_pid is not None else -1
    snapshot = _iter_process_entries_windows()
    matched_pids: list[int] = []
    for pid, _ppid, _image_name in snapshot:
        if pid <= 0 or pid == exclude:
            continue
        image_path = _query_process_image_path_windows(pid)
        if not image_path:
            continue
        if _is_path_within_root(image_path, target_root):
            matched_pids.append(pid)

    # Expand descendants to preserve previous /T behavior, then terminate in child-first order.
    ordered = _expand_pids_with_descendants_postorder(set(matched_pids), snapshot, exclude_pid=exclude)
    for pid in ordered:
        _terminate_process_windows(pid)


def best_effort_kill_app(
    exe_name: str,
    *,
    target_dir: Path | None = None,
) -> None:
    if os.name != "nt":
        return
    exe_name = (exe_name or "").strip()
    if target_dir is not None:
        _best_effort_kill_processes_in_target_dir(target_dir, exclude_pid=os.getpid())
        time.sleep(0.25)

    if exe_name:
        _kill_processes_by_image_name_best_effort(exe_name, exclude_pid=os.getpid())

    if target_dir is not None:
        time.sleep(0.35)
        _best_effort_kill_processes_in_target_dir(target_dir, exclude_pid=os.getpid())


_MOVEFILE_DELAY_UNTIL_REBOOT = 0x00000004
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
_FILE_ATTRIBUTE_READONLY = 0x00000001
_FILE_ATTRIBUTE_HIDDEN = 0x00000002
_FILE_ATTRIBUTE_SYSTEM = 0x00000004
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_DELETE_HELPER_TIMEOUT_SEC = 90.0


def _path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _detached_hidden_popen_kwargs() -> dict:
    kwargs = {
        "close_fds": True,
        "cwd": tempfile.gettempdir(),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            kwargs["startupinfo"] = startupinfo
        except Exception:
            pass
    return kwargs


def _clear_windows_file_attributes(path: Path) -> None:
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        raw_attrs = int(kernel32.GetFileAttributesW(str(path)))
    except Exception:
        return
    if raw_attrs == _INVALID_FILE_ATTRIBUTES:
        return
    new_attrs = raw_attrs & ~(_FILE_ATTRIBUTE_READONLY | _FILE_ATTRIBUTE_HIDDEN | _FILE_ATTRIBUTE_SYSTEM)
    if new_attrs == 0:
        new_attrs = _FILE_ATTRIBUTE_NORMAL
    try:
        ctypes.windll.kernel32.SetFileAttributesW(str(path), int(new_attrs))
    except Exception:
        pass


def _make_path_writable(path: Path) -> None:
    _clear_windows_file_attributes(path)
    try:
        mode = os.stat(path).st_mode
        os.chmod(path, mode | stat.S_IWRITE)
    except Exception:
        pass


def _make_tree_writable(target: Path) -> None:
    try:
        if target.exists():
            _make_path_writable(target)
    except Exception:
        pass
    try:
        if not target.exists() or not target.is_dir():
            return
    except Exception:
        return
    try:
        items = sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    except Exception:
        items = []
    for item in items:
        try:
            _make_path_writable(item)
        except Exception:
            continue


def _rmtree_best_effort(target: Path) -> bool:
    if not target.exists():
        return True

    _make_tree_writable(target)

    def _onerror(func, path, _exc_info):
        p = Path(path)
        _make_path_writable(p)
        try:
            func(path)
        except Exception:
            pass

    try:
        if target.is_dir():
            shutil.rmtree(target, onerror=_onerror)
        else:
            _make_path_writable(target)
            target.unlink(missing_ok=True)  # type: ignore[call-arg]
    except TypeError:
        try:
            target.unlink()
        except Exception:
            pass
    except Exception:
        pass
    return not target.exists()


def _wait_for_parent_exit_best_effort(pid: int, timeout_sec: float) -> None:
    if pid <= 0:
        return
    if os.name != "nt":
        time.sleep(min(max(timeout_sec, 0.0), 2.0))
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
        if not handle:
            return
        try:
            kernel32.WaitForSingleObject(handle, int(max(0.0, timeout_sec) * 1000))
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        pass


def _schedule_delete_on_reboot_path(path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        _make_path_writable(path)
    except Exception:
        pass
    try:
        return bool(ctypes.windll.kernel32.MoveFileExW(str(path), None, _MOVEFILE_DELAY_UNTIL_REBOOT))
    except Exception:
        return False


def _schedule_delete_tree_on_reboot(target: Path) -> None:
    if os.name != "nt":
        return
    paths: list[Path] = []
    try:
        if target.exists() and target.is_dir():
            paths.extend(sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True))
    except Exception:
        paths = []
    for item in paths:
        _schedule_delete_on_reboot_path(item)
    _schedule_delete_on_reboot_path(target)


def run_cleanup_helper(
    target_dir: Path,
    *,
    parent_pid: int = 0,
    helper_self: Path | None = None,
) -> int:
    target = Path(str(target_dir)).expanduser()
    _wait_for_parent_exit_best_effort(int(parent_pid or 0), timeout_sec=20.0)
    # One more best-effort kill in case the app restarted between UI confirmation and helper launch.
    best_effort_kill_app("", target_dir=target)

    deadline = time.time() + _DELETE_HELPER_TIMEOUT_SEC
    while time.time() < deadline:
        if _rmtree_best_effort(target):
            break
        time.sleep(0.75)

    if target.exists():
        _schedule_delete_tree_on_reboot(target)

    if helper_self is not None:
        _schedule_delete_on_reboot_path(Path(helper_self))
    return 0


def schedule_delete(target_dir: Path) -> None:
    target = Path(str(target_dir)).expanduser()
    helper_self: Path | None = None
    if getattr(sys, "frozen", False):
        current_exe = Path(sys.executable).resolve()
        launch_cmd = [str(current_exe)]
        if _path_is_within(current_exe, target):
            suffix = current_exe.suffix or ".exe"
            helper_self = Path(tempfile.gettempdir()) / f"{APP_NAME}-cleanup-{os.getpid()}-{int(time.time())}{suffix}"
            shutil.copy2(current_exe, helper_self)
            launch_cmd = [str(helper_self)]
    else:
        launch_cmd = [sys.executable, str(Path(__file__).resolve())]

    launch_cmd.extend(
        [
            "--cleanup-target",
            str(target),
            "--cleanup-parent-pid",
            str(os.getpid()),
        ]
    )
    if helper_self is not None:
        launch_cmd.extend(["--cleanup-self", str(helper_self)])

    subprocess.Popen(launch_cmd, **_detached_hidden_popen_kwargs())


class UninstallApp:
    def __init__(self, root: tk.Tk, target_dir: Path):
        self.root = root
        self.target_dir = target_dir
        self.root.title(f"Uninstall {APP_NAME}")
        self.root.geometry("760x390")
        self.root.minsize(700, 360)
        self.root.configure(bg="#f2f4f7")

        self.status_var = tk.StringVar(value="Ready to uninstall.")
        self.detail_var = tk.StringVar(value="The uninstaller will remove the app folder and can also remove shortcuts.")
        self.path_var = tk.StringVar(value=str(target_dir))
        self.remove_shortcuts_var = tk.BooleanVar(value=True)

        self.manifest = load_manifest(target_dir)
        self.main_exe_name = str(self.manifest.get("main_exe_name") or MAIN_EXE_NAME)
        self.uninstall_exe_name = str(self.manifest.get("uninstall_exe_name") or UNINSTALL_EXE_NAME)
        known_shortcuts = manifest_shortcut_records(self.manifest)
        if not known_shortcuts and "shortcuts" not in self.manifest:
            known_shortcuts = default_shortcut_records()
        self.known_shortcuts = known_shortcuts

        self._configure_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background="#f2f4f7")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Header.TFrame", background="#4a1f1f")
        style.configure("HeaderTitle.TLabel", background="#4a1f1f", foreground="#ffffff", font=("Segoe UI Semibold", 15))
        style.configure("HeaderSub.TLabel", background="#4a1f1f", foreground="#f0d6d6", font=("Segoe UI", 9))
        style.configure("Section.TLabelframe", background="#ffffff")
        style.configure("Section.TLabelframe.Label", background="#ffffff", foreground="#4a1f1f", font=("Segoe UI Semibold", 10))
        style.configure("Body.TLabel", background="#ffffff", foreground="#1f2933")
        style.configure("Muted.TLabel", background="#ffffff", foreground="#637282")
        style.configure("Warn.TLabel", background="#ffffff", foreground="#8a5a00")
        style.configure("Status.TLabel", background="#ffffff", foreground="#4a1f1f", font=("Segoe UI Semibold", 10))
        style.configure("TCheckbutton", background="#ffffff")
        try:
            tkfont.nametofont("TkDefaultFont").configure(size=10)
        except Exception:
            pass

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=14, style="App.TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        header = ttk.Frame(container, style="Header.TFrame", padding=(16, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=f"Uninstall {APP_NAME}", style="HeaderTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Removes the installed application folder. Shortcuts can be removed too.",
            style="HeaderSub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        ttk.Separator(container, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=(10, 10))

        frame = ttk.Frame(container, style="App.TFrame")
        frame.grid(row=2, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        target_box = ttk.LabelFrame(frame, text="Target", padding=12, style="Section.TLabelframe")
        target_box.grid(row=0, column=0, sticky="ew")
        target_box.columnconfigure(0, weight=1)
        ttk.Label(target_box, text=f"{APP_NAME} will be removed from this folder:", style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(target_box, textvariable=self.path_var, style="Muted.TLabel", wraplength=700, justify="left").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )

        warn_box = ttk.LabelFrame(frame, text="Warning", padding=12, style="Section.TLabelframe")
        warn_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        warn_text = (
            "This removes the entire install directory created by the bootstrapper.\n"
            "User-generated files stored inside that directory will also be deleted."
        )
        ttk.Label(warn_box, text=warn_text, style="Warn.TLabel", justify="left").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            warn_box,
            text=f"Also remove desktop / Start Menu shortcuts ({len(self.known_shortcuts)} known)",
            variable=self.remove_shortcuts_var,
        ).grid(row=1, column=0, sticky="w", pady=(10, 0))

        status_box = ttk.LabelFrame(frame, text="Status", padding=12, style="Section.TLabelframe")
        status_box.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        status_box.columnconfigure(0, weight=1)
        ttk.Label(status_box, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status_box, textvariable=self.detail_var, style="Muted.TLabel", wraplength=700, justify="left").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )

        btn_row = ttk.Frame(container, style="App.TFrame")
        btn_row.grid(row=3, column=0, sticky="e", pady=(12, 0))
        ttk.Button(btn_row, text="Cancel", command=self.root.destroy).pack(side="left")
        ttk.Button(btn_row, text="Uninstall", command=self._uninstall).pack(side="left", padx=(8, 0))

    def _uninstall(self) -> None:
        target = self.target_dir
        if not target.exists():
            messagebox.showinfo("Nothing to Remove", f"Folder not found:\n{target}")
            self.root.destroy()
            return

        confirm = messagebox.askyesno(
            "Confirm Uninstall",
            f"Remove {APP_NAME} from:\n\n{target}\n\nThis cannot be undone.",
        )
        if not confirm:
            return

        self.status_var.set("Closing running app (best effort)...")
        self.detail_var.set("Stopping the app process before deleting files.")
        self.root.update_idletasks()
        best_effort_kill_app(self.main_exe_name, target_dir=target)

        shortcut_warnings: list[str] = []
        if bool(self.remove_shortcuts_var.get()):
            self.status_var.set("Removing shortcuts (best effort)...")
            self.detail_var.set("Deleting desktop and Start Menu shortcuts if they exist.")
            self.root.update_idletasks()
            shortcut_warnings = remove_shortcuts_best_effort(self.manifest)

        self.status_var.set("Scheduling folder removal...")
        self.detail_var.set("A detached cleanup helper will delete the install folder after this uninstaller exits.")
        self.root.update_idletasks()
        try:
            schedule_delete(target)
        except Exception as e:
            messagebox.showerror("Uninstall Failed", str(e))
            self.status_var.set("Failed.")
            return

        info_lines = [
            "Uninstall has been scheduled.",
            "",
            "This window will close now.",
            "A background cleanup helper will remove the folder after the uninstaller exits.",
        ]
        if shortcut_warnings:
            info_lines.extend(["", "Shortcut warnings:"] + shortcut_warnings[:3])
        messagebox.showinfo("Uninstall Started", "\n".join(info_lines))
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} uninstaller")
    parser.add_argument("--target", default=str(default_target_dir()), help="Install directory to remove")
    parser.add_argument("--cleanup-target", default="", help=argparse.SUPPRESS)
    parser.add_argument("--cleanup-parent-pid", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--cleanup-self", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cleanup_target = str(args.cleanup_target or "").strip()
    if cleanup_target:
        helper_self_raw = str(args.cleanup_self or "").strip()
        helper_self = Path(helper_self_raw).expanduser() if helper_self_raw else None
        return run_cleanup_helper(
            Path(cleanup_target).expanduser(),
            parent_pid=int(args.cleanup_parent_pid or 0),
            helper_self=helper_self,
        )
    if tk is None:
        raise RuntimeError("tkinter is unavailable in this build/environment.") from _TK_IMPORT_ERROR
    root = tk.Tk()
    UninstallApp(root, Path(str(args.target)).expanduser())
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
