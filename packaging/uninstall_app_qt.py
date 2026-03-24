from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import uninstall_app as legacy  # noqa: E402
from dist_config import APP_NAME, ICON_CANDIDATES, MAIN_EXE_NAME, UNINSTALL_EXE_NAME  # noqa: E402


INSTALL_SCOPE_PER_USER = "per-user"
INSTALL_SCOPE_ALL_USERS = "all-users"


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


def _resolve_qt_icon() -> QIcon | None:
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


def normalize_scope(value: str) -> str:
    return INSTALL_SCOPE_ALL_USERS if str(value or "").strip().lower() in {"all", "all-users", "admin"} else INSTALL_SCOPE_PER_USER


def windows_desktop_dir(scope: str) -> Path:
    if normalize_scope(scope) == INSTALL_SCOPE_ALL_USERS:
        return Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "Desktop"
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"


def windows_start_menu_programs_dir(scope: str) -> Path:
    if normalize_scope(scope) == INSTALL_SCOPE_ALL_USERS:
        return Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    return Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"


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


def load_manifest(target_dir: Path) -> dict:
    return legacy.load_manifest(target_dir)


def detect_scope(target_dir: Path, manifest: dict) -> str:
    m_scope = normalize_scope(str(manifest.get("install_scope") or ""))
    if m_scope == INSTALL_SCOPE_ALL_USERS:
        return m_scope
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    try:
        if str(target_dir.resolve()).lower().startswith(str(program_files.resolve()).lower()):
            return INSTALL_SCOPE_ALL_USERS
    except Exception:
        pass
    return INSTALL_SCOPE_PER_USER


def default_shortcut_records(scope: str) -> list[dict]:
    start_dir = windows_start_menu_programs_dir(scope) / APP_NAME
    return [
        {"path": str(windows_desktop_dir(scope) / f"{APP_NAME}.lnk"), "kind": "desktop"},
        {"path": str(start_dir / f"{APP_NAME}.lnk"), "kind": "start_menu"},
        {"path": str(start_dir / f"Uninstall {APP_NAME}.lnk"), "kind": "start_menu_uninstall"},
    ]


def remove_shortcuts_best_effort(manifest: dict, scope: str) -> list[str]:
    records = legacy.manifest_shortcut_records(manifest)
    if records or "shortcuts" in manifest:
        return legacy.remove_shortcuts_best_effort(manifest)

    warnings: list[str] = []
    touched_dirs: set[Path] = set()
    scopes_to_try = [normalize_scope(scope)]
    other_scope = INSTALL_SCOPE_ALL_USERS if scopes_to_try[0] == INSTALL_SCOPE_PER_USER else INSTALL_SCOPE_PER_USER
    scopes_to_try.append(other_scope)
    seen_paths: set[str] = set()
    all_records: list[dict] = []
    for one_scope in scopes_to_try:
        for rec in default_shortcut_records(one_scope):
            path_str = str(rec.get("path") or "").strip()
            if not path_str:
                continue
            key = path_str.lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            all_records.append(rec)

    for rec in all_records:
        p = Path(str(rec.get("path") or ""))
        try:
            if p.exists():
                p.unlink()
                touched_dirs.add(p.parent)
        except Exception as e:
            warnings.append(f"Failed to remove shortcut {p.name}: {e}")
    candidate_dirs = set(touched_dirs)
    for one_scope in scopes_to_try:
        candidate_dirs.add(windows_start_menu_programs_dir(one_scope) / APP_NAME)
    for folder in sorted(candidate_dirs, key=str, reverse=True):
        try:
            if folder.exists() and folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        except Exception:
            pass
    return warnings


def requires_admin(target_dir: Path, scope: str) -> bool:
    if normalize_scope(scope) == INSTALL_SCOPE_ALL_USERS:
        return True
    if os.name != "nt":
        return False
    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramData", r"C:\ProgramData")),
    ]
    try:
        rp = str(target_dir.resolve()).lower()
        return any(rp.startswith(str(r.resolve()).lower()) for r in roots if r)
    except Exception:
        return False


def find_inno_uninstaller(target_dir: Path) -> Path | None:
    try:
        candidates = sorted(target_dir.glob("unins*.exe"))
    except Exception:
        return None
    for p in candidates:
        try:
            name = p.name.lower()
        except Exception:
            continue
        if name == str(UNINSTALL_EXE_NAME).lower():
            continue
        if name.startswith("unins") and p.is_file():
            return p
    return None


class UninstallWindow(QMainWindow):
    def __init__(
        self,
        *,
        target_dir: Path,
        auto_start: bool,
        remove_shortcuts_default: bool,
        elevated: bool,
    ) -> None:
        super().__init__()
        self.target_dir = target_dir
        self.manifest = load_manifest(target_dir)
        self.install_scope = detect_scope(target_dir, self.manifest)
        self.main_exe_name = str(self.manifest.get("main_exe_name") or MAIN_EXE_NAME)
        self.uninstall_exe_name = str(self.manifest.get("uninstall_exe_name") or UNINSTALL_EXE_NAME)
        self.known_shortcuts = legacy.manifest_shortcut_records(self.manifest) or default_shortcut_records(self.install_scope)
        self._auto_start_requested = bool(auto_start)
        self._process_elevated = bool(elevated or is_windows_admin())
        self._busy = False

        self.setWindowTitle(f"Uninstall {APP_NAME}")
        self.resize(860, 620)
        self.setMinimumSize(760, 560)
        self._build_ui()
        self._apply_styles()
        self._bind()
        self.path_label.setText(str(target_dir))
        self.scope_label.setText("为所有人安装" if self.install_scope == INSTALL_SCOPE_ALL_USERS else "仅当前用户")
        self.remove_shortcuts_cb.setChecked(remove_shortcuts_default)
        self._update_header_state()

    def _card(self, title: str) -> QGroupBox:
        b = QGroupBox(title)
        b.setObjectName("Card")
        l = QVBoxLayout(b)
        l.setContentsMargins(12, 12, 12, 12)
        l.setSpacing(8)
        return b

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
        t = QLabel(f"Uninstall {APP_NAME}", header)
        t.setObjectName("HeaderTitle")
        s = QLabel("PySide6 科幻风卸载器 | 支持管理员提权卸载“为所有人安装”的版本", header)
        s.setObjectName("HeaderSub")
        hl.addWidget(t)
        hl.addWidget(s)
        outer.addWidget(header)

        info = QHBoxLayout()
        self.badge = QLabel("", root)
        self.badge.setObjectName("Badge")
        self.scope_info = QLabel("", root)
        self.scope_info.setObjectName("Hint")
        info.addWidget(self.badge)
        info.addWidget(self.scope_info, 1)
        outer.addLayout(info)

        target_box = self._card("卸载目标")
        outer.addWidget(target_box)
        tl = target_box.layout()
        line1 = QLabel("安装目录", target_box)
        line1.setObjectName("Field")
        self.path_label = QLabel("", target_box)
        self.path_label.setObjectName("Muted")
        self.path_label.setWordWrap(True)
        line2 = QLabel("安装范围", target_box)
        line2.setObjectName("Field")
        self.scope_label = QLabel("", target_box)
        self.scope_label.setObjectName("Muted")
        tl.addWidget(line1)
        tl.addWidget(self.path_label)
        tl.addWidget(line2)
        tl.addWidget(self.scope_label)

        warn_box = self._card("操作说明")
        outer.addWidget(warn_box)
        wl = warn_box.layout()
        warn = QLabel(
            "将删除整个安装目录（包含其中的用户文件）。如果这是公共安装，卸载器会请求管理员权限。", warn_box
        )
        warn.setObjectName("Warn")
        warn.setWordWrap(True)
        wl.addWidget(warn)
        self.remove_shortcuts_cb = QCheckBox(
            f"同时删除桌面 / 开始菜单快捷方式（已知 {len(self.known_shortcuts)} 个）", warn_box
        )
        wl.addWidget(self.remove_shortcuts_cb)

        status_box = self._card("状态与日志")
        outer.addWidget(status_box, 1)
        sl = status_box.layout()
        self.status_label = QLabel("待命", status_box)
        self.status_label.setObjectName("Status")
        self.detail_label = QLabel("点击“开始卸载”后会先尝试关闭主程序，再删除快捷方式并安排目录删除。", status_box)
        self.detail_label.setObjectName("Muted")
        self.detail_label.setWordWrap(True)
        self.log_view = QPlainTextEdit(status_box)
        self.log_view.setObjectName("Log")
        self.log_view.setReadOnly(True)
        sl.addWidget(self.status_label)
        sl.addWidget(self.detail_label)
        sl.addWidget(self.log_view, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.open_btn = QPushButton("打开目录", root)
        self.open_btn.setObjectName("Ghost")
        self.uninstall_btn = QPushButton("开始卸载", root)
        self.uninstall_btn.setObjectName("Primary")
        self.close_btn = QPushButton("关闭", root)
        self.close_btn.setObjectName("Ghost")
        for w in (self.open_btn, self.uninstall_btn, self.close_btn):
            w.setMinimumHeight(38)
            btns.addWidget(w)
        outer.addLayout(btns)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget#Root { background: #070b12; color: #eafcff; font-family: "Microsoft YaHei UI", "Segoe UI"; }
            QFrame#Header {
                border-radius: 14px; border: 1px solid rgba(255,117,117,95);
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #260a12, stop:0.55 #431321, stop:1 #161a2f);
            }
            QLabel#HeaderTitle { font-size: 21pt; font-weight: 800; color: #fff5f7; }
            QLabel#HeaderSub { color: rgba(255,223,229,220); }
            QLabel#Badge {
                border-radius: 10px; padding: 5px 10px; font-weight: 700;
                background: rgba(255,117,117,22); border: 1px solid rgba(255,137,137,95); color: #ffd4d4;
            }
            QLabel#Hint, QLabel#Muted { color: rgba(224, 235, 244, 210); }
            QLabel#Field { color: #ffd4de; font-weight: 700; }
            QLabel#Warn { color: #ffd8a8; }
            QLabel#Status { color: #fff2f6; font-size: 11pt; font-weight: 700; }
            QGroupBox#Card {
                border: 1px solid rgba(255,117,117,70); border-radius: 14px; margin-top: 12px;
                background: rgba(18, 19, 28, 220); font-weight: 700;
            }
            QGroupBox#Card::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #ffc2ce; }
            QPlainTextEdit#Log {
                border: 1px solid rgba(255,117,117,35); border-radius: 10px;
                background: rgba(7, 8, 12, 220); color: #ffd7df; font-family: Consolas, "Cascadia Mono";
            }
            QPushButton#Primary {
                border-radius: 10px; padding: 0 14px; font-weight: 800; color: #2c040d;
                border: 1px solid rgba(255,214,221,170);
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ff8ea8, stop:0.6 #ff6f8f, stop:1 #ffa767);
            }
            QPushButton#Ghost {
                border-radius: 10px; padding: 0 12px; color: #ffeef3;
                border: 1px solid rgba(255,117,117,70); background: rgba(18, 19, 28, 210);
            }
            QPushButton#Ghost:disabled { color: rgba(255, 238, 243, 90); border-color: rgba(255,117,117,20); }
            QCheckBox { color: #fff0f5; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 5px; }
            QCheckBox::indicator {
                border: 1px solid rgba(255, 177, 190, 120);
                background: rgba(12, 8, 12, 235);
            }
            QCheckBox::indicator:hover { border-color: rgba(255, 177, 190, 190); }
            QCheckBox::indicator:checked {
                border: 1px solid rgba(255, 232, 207, 220);
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ff86a4, stop:1 #ffb66c);
            }
            QCheckBox::indicator:disabled {
                border-color: rgba(255, 177, 190, 45);
                background: rgba(12, 8, 12, 140);
            }
            QCheckBox::indicator:checked:disabled {
                border-color: rgba(255, 232, 207, 80);
                background: rgba(255, 134, 164, 90);
            }
            """
        )

    def _bind(self) -> None:
        self.close_btn.clicked.connect(self.close)
        self.uninstall_btn.clicked.connect(self._uninstall)
        self.open_btn.clicked.connect(self._open_dir)

    def _set_busy_controls(self, busy: bool) -> None:
        locked = bool(busy)
        self.uninstall_btn.setEnabled(not locked)
        self.open_btn.setEnabled(not locked)
        self.close_btn.setEnabled(not locked)
        self.remove_shortcuts_cb.setEnabled(not locked)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._auto_start_requested:
            self._auto_start_requested = False
            QTimer.singleShot(250, self._uninstall)

    def _log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_header_state(self) -> None:
        need_admin = requires_admin(self.target_dir, self.install_scope)
        self.scope_info.setText(
            f"检测到安装范围：{'为所有人安装' if self.install_scope == INSTALL_SCOPE_ALL_USERS else '仅当前用户'}"
        )
        if need_admin and not is_windows_admin():
            self.badge.setText("普通权限 / 卸载时将请求 UAC")
            self.badge.setStyleSheet(
                "QLabel#Badge { border-radius:10px; padding:5px 10px; font-weight:700; "
                "background: rgba(255,183,0,22); border:1px solid rgba(255,193,7,120); color:#ffe3a6; }"
            )
        elif is_windows_admin():
            self.badge.setText("管理员会话 / 可卸载公共安装")
            self.badge.setStyleSheet("")
        else:
            self.badge.setText("用户级卸载 / 无需管理员")
            self.badge.setStyleSheet(
                "QLabel#Badge { border-radius:10px; padding:5px 10px; font-weight:700; "
                "background: rgba(0,195,255,18); border:1px solid rgba(0,225,255,70); color:#b7f7ff; }"
            )

    def _open_dir(self) -> None:
        if not self.target_dir.exists():
            QMessageBox.information(self, "目录不存在", f"目录不存在：\n{self.target_dir}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.target_dir))):
            QMessageBox.warning(self, "打开失败", f"无法打开目录：\n{self.target_dir}")

    def _elevation_args(self) -> list[str]:
        args = ["--target", str(self.target_dir), "--auto-start", "--elevated"]
        if not self.remove_shortcuts_cb.isChecked():
            args.append("--no-remove-shortcuts")
        return args

    def _uninstall(self) -> None:
        if self._busy:
            return
        target = self.target_dir
        if not target.exists():
            QMessageBox.information(self, "Nothing to Uninstall", f"Directory does not exist:\n{target}")
            self.close()
            return

        if requires_admin(target, self.install_scope) and os.name == "nt" and not is_windows_admin():
            if (
                QMessageBox.question(
                    self,
                    "Administrator Permission Required",
                    "This uninstall appears to require administrator permission. Relaunch elevated and continue?",
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
                QMessageBox.warning(self, "Elevation Cancelled", "Administrator permission was not granted. Uninstall cancelled.")
            return

        if (
            QMessageBox.warning(
                self,
                "Confirm Uninstall",
                "All files under the following directory will be removed:\n\n"
                f"{target}\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        self._busy = True
        self._set_busy_controls(True)
        shortcut_warnings: list[str] = []
        try:
            self.status_label.setText("Stopping running app (best effort)...")
            self.detail_label.setText("Trying to close running processes before uninstall.")
            QApplication.processEvents()
            self._log(f"[kill] {self.main_exe_name}")
            legacy.best_effort_kill_app(self.main_exe_name, target_dir=target)

            inno_unins = find_inno_uninstaller(target)
            if inno_unins is not None:
                if not self.remove_shortcuts_cb.isChecked():
                    self._log("[info] Inno uninstaller will remove installer-managed shortcuts (checkbox ignored in Inno mode).")
                self.status_label.setText("Running Inno uninstaller...")
                self.detail_label.setText("Inno Setup is removing files, shortcuts, and registry entries.")
                QApplication.processEvents()
                cmd = [
                    str(inno_unins),
                    "/VERYSILENT",
                    "/SUPPRESSMSGBOXES",
                    "/NORESTART",
                    "/CLOSEAPPLICATIONS",
                    "/FORCECLOSEAPPLICATIONS",
                ]
                timeout_raw = str(os.getenv("UNINSTALL_INNO_TIMEOUT_SEC", "1800") or "").strip()
                try:
                    timeout_sec = int(float(timeout_raw))
                except Exception:
                    timeout_sec = 1800
                timeout_sec = max(120, min(timeout_sec, 86400))

                kwargs = {
                    "cwd": str(target),
                    "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                }
                if os.name == "nt":
                    kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = 0
                    kwargs["startupinfo"] = startupinfo
                self._log(f"[inno] {' '.join(cmd)}")
                proc = subprocess.Popen(cmd, **kwargs)
                started_at = time.monotonic()
                last_ui_tick = 0.0
                while proc.poll() is None:
                    QApplication.processEvents()
                    elapsed = time.monotonic() - started_at
                    if elapsed - last_ui_tick >= 1.0:
                        self.detail_label.setText(
                            f"Inno Setup is removing files, shortcuts, and registry entries... ({int(elapsed)}s)"
                        )
                        last_ui_tick = elapsed
                    if elapsed >= timeout_sec:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        time.sleep(1.0)
                        if proc.poll() is None:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                        raise RuntimeError(
                            f"Inno uninstaller timed out after {timeout_sec} seconds."
                        )
                    time.sleep(0.12)
                rc = int(proc.wait())
                if int(rc) != 0:
                    raise RuntimeError(f"Inno uninstaller failed (exit code {rc}).")
                self.status_label.setText("Uninstall Complete")
                self.detail_label.setText("Inno uninstall finished.")
                QMessageBox.information(
                    self,
                    "Uninstall Complete",
                    "The application was removed by the Inno uninstaller."
                )
                self.close()
                return

            if self.remove_shortcuts_cb.isChecked():
                self.status_label.setText("Removing shortcuts (best effort)...")
                self.detail_label.setText("Deleting desktop and Start Menu shortcuts if they exist.")
                QApplication.processEvents()
                shortcut_warnings = remove_shortcuts_best_effort(self.manifest, self.install_scope)
                for w in shortcut_warnings:
                    self._log(f"[warn] {w}")

            self.status_label.setText("Scheduling folder removal")
            self.detail_label.setText("A helper process will delete the install folder after this window closes.")
            QApplication.processEvents()
            legacy.schedule_delete(target)
            self._log(f"[schedule_delete] {target}")
        except Exception as e:
            self._busy = False
            self._set_busy_controls(False)
            self.status_label.setText("Uninstall Failed")
            self.detail_label.setText(str(e))
            self._log(f"[error] {e}")
            QMessageBox.critical(self, "Uninstall Failed", str(e))
            return

        info = [
            "Uninstall scheduled.",
            "",
            "After this window closes, the helper process will continue deleting the install directory.",
        ]
        if shortcut_warnings:
            info.extend(["", "Shortcut warnings:"] + shortcut_warnings[:3])
        QMessageBox.information(self, "Uninstall Started", "\n".join(info))
        self.close()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._busy:
            event.accept()
            return
        event.accept()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{APP_NAME} PySide6 uninstaller")
    p.add_argument("--target", default=str(legacy.default_target_dir()), help="Install directory to remove")
    p.add_argument("--auto-start", action="store_true", help="Auto start uninstall on launch")
    p.add_argument("--elevated", action="store_true", help="Marker flag for elevated relaunch")
    p.add_argument("--no-remove-shortcuts", action="store_true", help="Do not remove shortcuts")
    p.add_argument("--cleanup-target", default="", help=argparse.SUPPRESS)
    p.add_argument("--cleanup-parent-pid", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--cleanup-self", default="", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    _hide_console_window_if_present()
    _set_windows_app_user_model_id("MediaTranscribeStudio.Uninstall")
    args = parse_args()
    cleanup_target = str(args.cleanup_target or "").strip()
    if cleanup_target:
        helper_self_raw = str(args.cleanup_self or "").strip()
        helper_self = Path(helper_self_raw).expanduser() if helper_self_raw else None
        return legacy.run_cleanup_helper(
            Path(cleanup_target).expanduser(),
            parent_pid=int(args.cleanup_parent_pid or 0),
            helper_self=helper_self,
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
    win = UninstallWindow(
        target_dir=Path(str(args.target)).expanduser(),
        auto_start=bool(args.auto_start),
        remove_shortcuts_default=not bool(args.no_remove_shortcuts),
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
