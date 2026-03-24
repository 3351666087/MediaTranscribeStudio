from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

try:
    import certifi
except Exception:
    certifi = None  # type: ignore[assignment]

from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import bootstrap_installer as legacy  # noqa: E402
except Exception:
    legacy = None

from dist_config import (  # noqa: E402
    APP_NAME,
    DEFAULT_MACOS_INSTALLER_SHA256,
    DEFAULT_MACOS_INSTALLER_URL,
    ICON_CANDIDATES,
    MACOS_ICON_CANDIDATES,
)


DEFAULT_CACHE_DIR = Path.home() / "Library" / "Caches" / APP_NAME / "bootstrap"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DMG_MOUNT_TOOL = shutil.which("hdiutil") or "/usr/bin/hdiutil"
DITTO_TOOL = shutil.which("ditto") or "/usr/bin/ditto"
OPEN_TOOL = shutil.which("open") or "/usr/bin/open"
OSASCRIPT_TOOL = shutil.which("osascript") or "/usr/bin/osascript"
XATTR_TOOL = shutil.which("xattr") or "/usr/bin/xattr"
CODESIGN_TOOL = shutil.which("codesign") or "/usr/bin/codesign"
APP_BUNDLE_NAME = f"{APP_NAME}.app"
SUPPORTED_PAYLOAD_EXTENSIONS = (".tar.gz", ".tgz", ".dmg", ".zip", ".tar")


class BootstrapCancelled(RuntimeError):
    pass


def _resolve_qt_icon() -> Optional[QIcon]:
    candidates = [
        Path("installer.icns"),
        Path("setup.icns"),
        Path("assets") / "installer.icns",
        *MACOS_ICON_CANDIDATES,
        Path("assets") / "app.png",
        Path("app.png"),
        *ICON_CANDIDATES,
    ]
    for rel in candidates:
        path = PROJECT_ROOT / rel
        if path.exists():
            try:
                return QIcon(str(path))
            except Exception:
                return None
    return None


def _resolve_scene_path() -> Optional[Path]:
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(str(meipass)))
    roots.append(PROJECT_ROOT)

    for root in roots:
        for rel in (Path("pictures") / "scene.png", Path("pictures") / "scene.jpg"):
            path = root / rel
            if path.exists():
                return path
    return None


def _hf_url_candidates(url: str) -> list[str]:
    raw = str(url or "").strip()
    if not raw:
        return []
    if legacy is not None:
        try:
            return legacy.hf_url_candidates(raw)
        except Exception:
            pass
    return [raw]


def _hf_route_probe_note() -> str:
    if legacy is not None:
        try:
            return str(legacy.hf_route_probe_note() or "").strip()
        except Exception:
            pass
    return ""


def _human_bytes(num: float) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num:.1f} B"


def _filename_from_url(url: str) -> str:
    path = urllib.parse.urlparse(str(url or "").strip()).path
    name = Path(path).name.strip()
    if name:
        return name
    fallback_path = urllib.parse.urlparse(str(DEFAULT_MACOS_INSTALLER_URL or "").strip()).path
    fallback_name = Path(fallback_path).name.strip()
    if fallback_name:
        return fallback_name
    return f"{APP_NAME}-full-macOS.zip"


def _payload_kind(value: Path | str) -> str:
    name = str(value or "").strip().lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar"):
        return "tar"
    if name.endswith(".dmg"):
        return "dmg"
    if name.endswith(".zip"):
        return "zip"
    return "zip"


def _payload_label(kind: str) -> str:
    return {
        "dmg": "DMG",
        "zip": "ZIP",
        "tar": "TAR",
    }.get(str(kind or "").strip().lower(), "archive")


def _resolved_cert_bundle() -> Optional[Path]:
    for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        candidate = str(os.environ.get(env_name, "") or "").strip()
        if candidate and Path(candidate).exists():
            return Path(candidate)
    if certifi is not None:
        try:
            candidate = Path(certifi.where()).resolve()
            if candidate.exists():
                return candidate
        except Exception:
            pass
    return None


def _configure_tls_cert_env() -> Optional[Path]:
    cert_path = _resolved_cert_bundle()
    if cert_path is None:
        return None
    cert_text = str(cert_path)
    for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if os.environ.get(env_name) != cert_text:
            os.environ[env_name] = cert_text
    return cert_path


TLS_CERT_PATH = _configure_tls_cert_env()


def _download_ssl_context() -> Optional[ssl.SSLContext]:
    cert_path = _configure_tls_cert_env()
    if cert_path is None:
        return None
    try:
        return ssl.create_default_context(cafile=str(cert_path))
    except Exception:
        return None


def _urlopen_with_tls(request: urllib.request.Request, *, timeout: float):
    context = _download_ssl_context()
    if context is not None:
        return urllib.request.urlopen(request, timeout=timeout, context=context)
    return urllib.request.urlopen(request, timeout=timeout)


def _default_auto_start() -> bool:
    return bool(str(DEFAULT_MACOS_INSTALLER_URL or "").strip())


def _current_app_bundle() -> Optional[Path]:
    try:
        executable = Path(sys.executable).resolve()
    except Exception:
        executable = Path(sys.executable)
    for parent in executable.parents:
        if parent.suffix.lower() == ".app":
            return parent
    return None


def _is_applications_bundle(path: Optional[Path]) -> bool:
    if path is None:
        return False
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        resolved = path.expanduser()

    for base in (Path("/Applications"), Path.home() / "Applications"):
        try:
            if resolved == base or resolved.is_relative_to(base):
                return True
        except Exception:
            continue
    return False


def _looks_like_volume_path(path: Optional[Path]) -> bool:
    if path is None:
        return False
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        resolved = path.expanduser()
    return str(resolved).startswith("/Volumes/")


def _open_path(path: Path) -> None:
    subprocess.Popen([OPEN_TOOL, str(path)])


def _shell_quote(value: Path | str) -> str:
    return shlex.quote(str(value))


def _applescript_quote(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _safe_path_for_log(path: Optional[Path]) -> str:
    if path is None:
        return "(unknown)"
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _parse_total_from_content_range(content_range: str) -> int:
    raw = str(content_range or "").strip()
    if "/" not in raw:
        return 0
    try:
        total = raw.rsplit("/", 1)[1].strip()
        if not total or total == "*":
            return 0
        return int(total)
    except Exception:
        return 0


def _clear_quarantine(path: Path, *, log_cb=None) -> None:
    try:
        if not path.exists():
            return
    except Exception:
        return
    if not XATTR_TOOL or not Path(XATTR_TOOL).exists():
        return

    proc = subprocess.run(
        [XATTR_TOOL, "-dr", "com.apple.quarantine", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and log_cb is not None:
        log_cb(f"[xattr] cleared com.apple.quarantine on {path}")


def _main_executable_for_bundle(app_bundle: Path) -> Path:
    executable = app_bundle / "Contents" / "MacOS" / app_bundle.stem
    if executable.exists():
        return executable
    candidates = sorted(path for path in (app_bundle / "Contents" / "MacOS").glob("*") if path.is_file())
    if candidates:
        return candidates[0]
    return executable


def _verify_app_signature(app_bundle: Path, *, log_cb=None) -> None:
    if not CODESIGN_TOOL or not Path(CODESIGN_TOOL).exists():
        if log_cb is not None:
            log_cb("[codesign-warn] codesign not found; skip signature verification")
        return

    verify_targets = [
        ("executable", _main_executable_for_bundle(app_bundle)),
        ("app", app_bundle),
    ]
    for label, target in verify_targets:
        proc = subprocess.run(
            [CODESIGN_TOOL, "--verify", "--verbose=2", str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"codesign verification failed for {target}"
            raise RuntimeError(f"Downloaded app signature check failed for {label}:\n{detail}")
        if log_cb is not None:
            log_cb(f"[codesign-ok] {label}: {target}")


def _find_app_bundle(root: Path) -> Optional[Path]:
    preferred = root / APP_BUNDLE_NAME
    if preferred.exists() and preferred.is_dir():
        return preferred

    candidates = [path for path in root.glob("*.app") if path.is_dir()]
    if not candidates:
        candidates = [path for path in root.rglob("*.app") if path.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (0 if path.name == APP_BUNDLE_NAME else 1, len(path.parts), str(path)))
    return candidates[0]


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_root = dest_dir.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            target_path = (dest_root / member.name).resolve()
            if target_path != dest_root and dest_root not in target_path.parents:
                raise RuntimeError(f"Blocked unsafe tar member outside extraction root: {member.name}")
        archive.extractall(dest_dir)


class SceneWidget(QWidget):
    def __init__(self, scene_path: Optional[Path], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QPixmap(str(scene_path)) if scene_path and scene_path.exists() else QPixmap()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()

        if not self._scene.isNull():
            scaled = self._scene.scaled(
                rect.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            x = (rect.width() - scaled.width()) // 2
            y = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            gradient = QLinearGradient(0, 0, 0, rect.height())
            gradient.setColorAt(0.0, QColor(246, 236, 233))
            gradient.setColorAt(0.6, QColor(236, 221, 224))
            gradient.setColorAt(1.0, QColor(214, 203, 216))
            painter.fillRect(rect, gradient)

        painter.fillRect(rect, QColor(18, 12, 28, 118))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 28))
        painter.drawEllipse(-80, -80, 360, 220)
        painter.drawEllipse(rect.width() - 280, rect.height() - 240, 320, 220)

        path = QPainterPath()
        path.moveTo(120, rect.height() - 130)
        path.cubicTo(260, rect.height() - 210, 460, rect.height() - 30, 640, rect.height() - 120)
        path.cubicTo(720, rect.height() - 162, 780, rect.height() - 170, 830, rect.height() - 132)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 226, 232, 170), 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(path)
        painter.drawLine(792, rect.height() - 160, 842, rect.height() - 136)
        painter.drawLine(806, rect.height() - 102, 842, rect.height() - 136)
        painter.setPen(QPen(QColor(255, 252, 255, 96), 3))
        painter.drawEllipse(708, rect.height() - 202, 20, 20)
        painter.drawEllipse(736, rect.height() - 222, 10, 10)
        painter.drawEllipse(766, rect.height() - 210, 14, 14)

        super().paintEvent(event)


class BootstrapWorker(QObject):
    progress = Signal(int, int)
    status = Signal(str)
    log = Signal(str)
    finished = Signal(bool, object)

    def __init__(
        self,
        *,
        url: str,
        sha256: str,
        cache_dir: Path,
        target_bundle: Path,
    ) -> None:
        super().__init__()
        self.url = str(url or "").strip()
        self.sha256 = str(sha256 or "").strip().lower()
        self.cache_dir = cache_dir
        self.target_bundle = target_bundle
        self.cancel_event = threading.Event()
        self.clear_cache_on_cancel = False

    def cancel(self, *, clear_cache: bool = False) -> None:
        if clear_cache:
            self.clear_cache_on_cancel = True
        self.cancel_event.set()

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise BootstrapCancelled("下载已取消。")

    def run(self) -> None:
        try:
            self._raise_if_cancelled()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.status.emit("正在准备完整版安装包...")
            downloaded_payload = self._download_payload()
            self._raise_if_cancelled()
            payload_kind = _payload_kind(downloaded_payload)
            if payload_kind == "dmg":
                self.status.emit("正在挂载下载好的完整版 DMG...")
            else:
                self.status.emit(f"正在解包下载好的完整版 {_payload_label(payload_kind)}...")
            staged_app = self._stage_app_bundle(downloaded_payload)
            self._raise_if_cancelled()
            self.status.emit("正在校验完整版应用签名...")
            _verify_app_signature(staged_app, log_cb=self.log.emit)
            self._raise_if_cancelled()
            helper_path, log_path = self._write_replace_helper(staged_app)
            result = {
                "downloaded_path": str(downloaded_payload),
                "payload_kind": payload_kind,
                "staged_app_path": str(staged_app),
                "helper_path": str(helper_path),
                "log_path": str(log_path),
            }
            self.finished.emit(True, result)
        except BootstrapCancelled as exc:
            self.finished.emit(
                False,
                {
                    "cancelled": True,
                    "message": str(exc),
                    "cache_cleared": bool(self.clear_cache_on_cancel),
                },
            )
        except Exception as exc:
            self.finished.emit(False, str(exc))
        finally:
            if self.cancel_event.is_set() and self.clear_cache_on_cancel:
                shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _download_payload(self) -> Path:
        target_name = _filename_from_url(self.url)
        if not any(target_name.lower().endswith(ext) for ext in SUPPORTED_PAYLOAD_EXTENSIONS):
            target_name = f"{Path(target_name).stem or APP_NAME}-full-macOS.zip"
        final_path = self.cache_dir / target_name
        tmp_path = self.cache_dir / f"{target_name}.part"

        if final_path.exists():
            if self.sha256:
                digest = self._sha256_of_file(final_path)
                if digest == self.sha256:
                    self.log.emit(f"[cache-hit] {final_path}")
                    self.progress.emit(final_path.stat().st_size, final_path.stat().st_size)
                    return final_path
            else:
                self.log.emit(f"[cache-hit] {final_path}")
                self.progress.emit(final_path.stat().st_size, final_path.stat().st_size)
                return final_path

        candidates = _hf_url_candidates(self.url) or [self.url]
        last_error = "No download URL candidates were available."

        for candidate in candidates:
            self._raise_if_cancelled()
            hasher = hashlib.sha256() if self.sha256 else None
            downloaded = 0
            total = 0
            self.status.emit(f"正在下载完整版安装包：{candidate}")
            self.log.emit(f"[download] {candidate}")
            try:
                resume_from = 0
                headers = {"User-Agent": f"{APP_NAME}-Bootstrapper/1.0"}
                if tmp_path.exists():
                    try:
                        resume_from = max(0, int(tmp_path.stat().st_size))
                    except Exception:
                        resume_from = 0
                if resume_from > 0:
                    headers["Range"] = f"bytes={resume_from}-"
                    self.log.emit(f"[resume] {candidate} from {_human_bytes(resume_from)}")
                    if hasher is not None:
                        with tmp_path.open("rb") as existing:
                            for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                                hasher.update(chunk)
                    downloaded = resume_from

                request = urllib.request.Request(candidate, headers=headers)
                with _urlopen_with_tls(request, timeout=60.0) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    if resume_from > 0 and status != 206:
                        try:
                            tmp_path.unlink()
                        except Exception:
                            pass
                        resume_from = 0
                        downloaded = 0
                        hasher = hashlib.sha256() if self.sha256 else None
                        request = urllib.request.Request(
                            candidate,
                            headers={"User-Agent": f"{APP_NAME}-Bootstrapper/1.0"},
                        )
                        response.close()
                        with _urlopen_with_tls(request, timeout=60.0) as restarted:
                            status = int(getattr(restarted, "status", restarted.getcode()))
                            content_length = str(restarted.headers.get("Content-Length") or "0")
                            total = int(content_length) if content_length.isdigit() else 0
                            with tmp_path.open("wb") as handle:
                                while True:
                                    self._raise_if_cancelled()
                                    chunk = restarted.read(DOWNLOAD_CHUNK_SIZE)
                                    if not chunk:
                                        break
                                    handle.write(chunk)
                                    downloaded += len(chunk)
                                    if hasher is not None:
                                        hasher.update(chunk)
                                    self.progress.emit(downloaded, total)
                    else:
                        content_length = str(response.headers.get("Content-Length") or "0")
                        total_from_range = _parse_total_from_content_range(str(response.headers.get("Content-Range") or ""))
                        if total_from_range > 0:
                            total = total_from_range
                        elif content_length.isdigit():
                            total = resume_from + int(content_length) if status == 206 else int(content_length)
                        mode = "ab" if resume_from > 0 and status == 206 else "wb"
                        with tmp_path.open(mode) as handle:
                            while True:
                                self._raise_if_cancelled()
                                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                                if not chunk:
                                    break
                                handle.write(chunk)
                                downloaded += len(chunk)
                                if hasher is not None:
                                    hasher.update(chunk)
                                self.progress.emit(downloaded, total)

                self._raise_if_cancelled()
                if total > 0 and downloaded < total:
                    raise RuntimeError(f"下载意外中断：{downloaded}/{total}")

                if hasher is not None:
                    actual = hasher.hexdigest().lower()
                    if actual != self.sha256:
                        try:
                            if tmp_path.exists():
                                tmp_path.unlink()
                        except Exception:
                            pass
                        raise RuntimeError(
                            "SHA256 mismatch:\n"
                            f"expected {self.sha256}\n"
                            f"actual   {actual}"
                        )

                tmp_path.replace(final_path)
                _clear_quarantine(final_path, log_cb=self.log.emit)
                self.log.emit(f"[saved] {final_path} ({_human_bytes(downloaded or final_path.stat().st_size)})")
                return final_path
            except BootstrapCancelled:
                raise
            except Exception as exc:
                last_error = str(exc)
                self.log.emit(f"[failed] {candidate}: {exc}")

        raise RuntimeError(last_error)

    def _copy_into_stage(self, source_app: Path, staged_app: Path, *, source_label: str) -> Path:
        self._raise_if_cancelled()
        self.status.emit("正在复制正式版应用...")
        if source_app.resolve() != staged_app.resolve():
            self.log.emit(f"[extract:{source_label.lower()}] {source_app} -> {staged_app}")
            copy_proc = subprocess.run(
                [DITTO_TOOL, str(source_app), str(staged_app)],
                capture_output=True,
                text=True,
                check=False,
            )
            if copy_proc.returncode != 0:
                raise RuntimeError(
                    copy_proc.stderr.strip()
                    or copy_proc.stdout.strip()
                    or f"Failed to copy app bundle from downloaded {source_label}."
                )
        _clear_quarantine(staged_app, log_cb=self.log.emit)
        return staged_app

    def _stage_app_bundle_from_dmg(self, dmg_path: Path, *, stage_root: Path, staged_app: Path) -> Path:
        mount_root = Path(tempfile.mkdtemp(prefix=f"{APP_NAME}-mount-"))
        mounted = False
        try:
            self._raise_if_cancelled()
            attach = subprocess.run(
                [
                    DMG_MOUNT_TOOL,
                    "attach",
                    str(dmg_path),
                    "-nobrowse",
                    "-readonly",
                    "-mountpoint",
                    str(mount_root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if attach.returncode != 0:
                raise RuntimeError(attach.stderr.strip() or attach.stdout.strip() or "Failed to mount downloaded DMG.")
            mounted = True

            source_app = _find_app_bundle(mount_root)
            if source_app is None:
                raise RuntimeError(f"No .app bundle was found inside the downloaded DMG: {dmg_path}")

            return self._copy_into_stage(source_app, staged_app, source_label="DMG")
        finally:
            if mounted:
                subprocess.run(
                    [DMG_MOUNT_TOOL, "detach", str(mount_root), "-force"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            shutil.rmtree(mount_root, ignore_errors=True)

    def _stage_app_bundle_from_zip(self, archive_path: Path, *, stage_root: Path, staged_app: Path) -> Path:
        if not DITTO_TOOL or not Path(DITTO_TOOL).exists():
            raise RuntimeError("ditto is required to extract the downloaded ZIP payload.")
        extract_proc = subprocess.run(
            [DITTO_TOOL, "-x", "-k", str(archive_path), str(stage_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        if extract_proc.returncode != 0:
            raise RuntimeError(
                extract_proc.stderr.strip() or extract_proc.stdout.strip() or "Failed to extract downloaded ZIP payload."
            )

        source_app = _find_app_bundle(stage_root)
        if source_app is None:
            raise RuntimeError(f"No .app bundle was found inside the downloaded ZIP archive: {archive_path}")
        return self._copy_into_stage(source_app, staged_app, source_label="ZIP")

    def _stage_app_bundle_from_tar(self, archive_path: Path, *, stage_root: Path, staged_app: Path) -> Path:
        try:
            _safe_extract_tar(archive_path, stage_root)
        except Exception as exc:
            raise RuntimeError(f"Failed to extract downloaded TAR payload: {exc}") from exc

        source_app = _find_app_bundle(stage_root)
        if source_app is None:
            raise RuntimeError(f"No .app bundle was found inside the downloaded TAR archive: {archive_path}")
        return self._copy_into_stage(source_app, staged_app, source_label="TAR")

    def _stage_app_bundle(self, payload_path: Path) -> Path:
        stage_root = Path(tempfile.mkdtemp(prefix=f"{APP_NAME}-stage-", dir=str(self.cache_dir)))
        staged_app = stage_root / APP_BUNDLE_NAME
        kind = _payload_kind(payload_path)
        self.log.emit(f"[payload] {payload_path} ({_payload_label(kind)})")
        if kind == "dmg":
            return self._stage_app_bundle_from_dmg(payload_path, stage_root=stage_root, staged_app=staged_app)
        if kind == "zip":
            return self._stage_app_bundle_from_zip(payload_path, stage_root=stage_root, staged_app=staged_app)
        if kind == "tar":
            return self._stage_app_bundle_from_tar(payload_path, stage_root=stage_root, staged_app=staged_app)
        raise RuntimeError(f"Unsupported hosted macOS payload format: {payload_path}")

    def _write_replace_helper(self, staged_app: Path) -> tuple[Path, Path]:
        helper_dir = self.cache_dir / "helpers"
        helper_dir.mkdir(parents=True, exist_ok=True)
        log_path = helper_dir / "replace.log"
        helper_path = helper_dir / f"replace-{os.getpid()}.sh"

        replace_cmd = " ".join(
            [
                "/bin/rm",
                "-rf",
                _shell_quote(self.target_bundle),
                "&&",
                _shell_quote(DITTO_TOOL),
                _shell_quote(staged_app),
                _shell_quote(self.target_bundle),
                "&&",
                "(",
                _shell_quote(XATTR_TOOL),
                "-dr",
                "com.apple.quarantine",
                _shell_quote(self.target_bundle),
                ">/dev/null",
                "2>&1",
                "||",
                "true",
                ")",
            ]
        )
        display_msg = (
            "完整版替换失败，请查看日志："
            f" {_safe_path_for_log(log_path)}"
        )

        script = textwrap.dedent(
            f"""\
            #!/bin/bash
            set -u

            CURRENT_PID={os.getpid()}
            TARGET_APP={_shell_quote(self.target_bundle)}
            TARGET_PARENT={_shell_quote(self.target_bundle.parent)}
            SOURCE_APP={_shell_quote(staged_app)}
            SOURCE_ROOT={_shell_quote(staged_app.parent)}
            LOG_FILE={_shell_quote(log_path)}
            REPLACE_CMD={_shell_quote(replace_cmd)}
            OPEN_BIN={_shell_quote(OPEN_TOOL)}
            OSASCRIPT_BIN={_shell_quote(OSASCRIPT_TOOL)}

            wait_for_pid() {{
              for _ in $(seq 1 120); do
                if ! kill -0 "$CURRENT_PID" 2>/dev/null; then
                  return 0
                fi
                sleep 1
              done
              return 0
            }}

            show_error() {{
              "$OSASCRIPT_BIN" -e 'display dialog {_applescript_quote(display_msg)} buttons {{"OK"}} default button "OK" with title {_applescript_quote(APP_NAME)}' >/dev/null 2>&1 || true
            }}

            run_replace() {{
              /bin/sh -lc "$REPLACE_CMD" >>"$LOG_FILE" 2>&1
            }}

            wait_for_pid

            if [ -w "$TARGET_PARENT" ] && {{ [ ! -e "$TARGET_APP" ] || [ -w "$TARGET_APP" ]; }}; then
              if ! run_replace; then
                show_error
                exit 1
              fi
            else
              if ! "$OSASCRIPT_BIN" -e 'do shell script {_applescript_quote(replace_cmd)} with administrator privileges' >>"$LOG_FILE" 2>&1; then
                show_error
                exit 1
              fi
            fi

            "$OPEN_BIN" "$TARGET_APP" >>"$LOG_FILE" 2>&1 || true
            /bin/rm -rf "$SOURCE_ROOT" >/dev/null 2>&1 || true
            exit 0
            """
        )
        helper_path.write_text(script, encoding="utf-8")
        helper_path.chmod(0o755)
        self.log.emit(f"[helper] {helper_path}")
        self.log.emit(f"[replace-log] {log_path}")
        return helper_path, log_path

    @staticmethod
    def _sha256_of_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest().lower()


class InstallerWindow(QMainWindow):
    def __init__(
        self,
        *,
        installer_url: str,
        installer_sha256: str,
        cache_dir: Path,
        auto_start: bool,
    ) -> None:
        super().__init__()
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[BootstrapWorker] = None
        self.downloading = False
        self.installer_url = str(installer_url or "").strip()
        self.installer_sha256 = str(installer_sha256 or "").strip()
        self.cache_dir = cache_dir
        self.target_bundle = _current_app_bundle()
        self.route_note = _hf_route_probe_note()
        self.scene_path = _resolve_scene_path()
        self.last_prepared: Optional[dict[str, str]] = None
        self.ready_to_install = False
        self._closing_after_cancel = False
        self._cancelling = False
        self._cancel_clears_cache = False

        self.setWindowTitle(APP_NAME)
        self.resize(920, 640)
        self.setMinimumSize(860, 600)

        root = SceneWidget(self.scene_path, self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(28, 28, 28, 28)
        root_layout.setSpacing(0)

        card = QFrame(root)
        card.setObjectName("Card")
        card.setStyleSheet(
            """
            QFrame#Card {
                background: rgba(7, 28, 52, 188);
                border: 1px solid rgba(156, 214, 255, 62);
                border-radius: 28px;
            }
            QLabel {
                color: #eef8ff;
            }
            QPlainTextEdit {
                background: rgba(5, 18, 34, 168);
                border: 1px solid rgba(162, 220, 255, 40);
                border-radius: 18px;
                color: #d9f5ff;
                padding: 10px;
            }
            QProgressBar {
                background: rgba(181, 226, 255, 24);
                border: 1px solid rgba(156, 214, 255, 42);
                border-radius: 11px;
                color: #eff9ff;
                text-align: center;
                min-height: 22px;
            }
            QProgressBar::chunk {
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2f8dff, stop:0.5 #39c4ff, stop:1 #75f0ff);
            }
            QPushButton {
                background: rgba(212, 241, 255, 216);
                border: 1px solid rgba(224, 247, 255, 96);
                border-radius: 16px;
                color: #06243f;
                font-size: 15px;
                font-weight: 700;
                min-height: 44px;
                padding: 0 18px;
            }
            QPushButton:hover {
                background: rgba(229, 248, 255, 236);
            }
            QPushButton:disabled {
                background: rgba(178, 216, 240, 38);
                color: rgba(232, 248, 255, 118);
            }
            """
        )
        root_layout.addWidget(card, 1)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 26, 28, 24)
        layout.setSpacing(16)

        eyebrow = QLabel("Lightweight First-Run Launcher")
        eyebrow.setStyleSheet("color: rgba(204, 234, 255, 188); font-size: 13px; letter-spacing: 0.5px;")
        layout.addWidget(eyebrow)

        title = QLabel(APP_NAME)
        title_font = QFont()
        title_font.setPointSize(26)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel(
            "拖入 Applications 后首次打开，这个轻量版会自动从 Hugging Face 下载完整版安装包，"
            "退出并把自己替换成正式 app。"
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: rgba(227, 244, 255, 212); font-size: 15px;")
        layout.addWidget(subtitle)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: rgba(192, 225, 245, 182); font-size: 13px;")
        layout.addWidget(self.path_label)

        if self.route_note:
            route_label = QLabel(self.route_note)
            route_label.setWordWrap(True)
            route_label.setStyleSheet("color: rgba(198, 231, 247, 160); font-size: 12px;")
            layout.addWidget(route_label)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "background: rgba(13, 42, 78, 136); border: 1px solid rgba(156, 214, 255, 62); "
            "border-radius: 16px; padding: 12px; color: #eaf7ff;"
        )
        layout.addWidget(self.warning_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("准备就绪。")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        self.start_btn = QPushButton("下载并替换为完整版")
        self.start_btn.clicked.connect(self.start_install)
        buttons.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("取消下载")
        self.cancel_btn.clicked.connect(self.cancel_download)
        self.cancel_btn.setEnabled(False)
        buttons.addWidget(self.cancel_btn)

        self.apps_btn = QPushButton("打开 Applications")
        self.apps_btn.clicked.connect(lambda: _open_path(Path("/Applications")))
        buttons.addWidget(self.apps_btn)

        self.log_btn = QPushButton("打开缓存目录")
        self.log_btn.clicked.connect(lambda: _open_path(self.cache_dir))
        buttons.addWidget(self.log_btn)

        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("下载与替换日志会显示在这里。")
        layout.addWidget(self.log_edit, 1)

        self.setCentralWidget(root)
        self._refresh_ready_state()

        if auto_start and self.start_btn.isEnabled():
            QTimer.singleShot(350, self.start_install)

    def _append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(str(text or ""))

    def _refresh_ready_state(self) -> None:
        bundle = self.target_bundle
        bundle_text = _safe_path_for_log(bundle)
        self.path_label.setText(f"当前运行位置：{bundle_text}")
        self.ready_to_install = False

        if not self.installer_url:
            self.warning_label.setText("当前轻量版没有嵌入 Hugging Face 完整版地址，无法继续。")
            self.start_btn.setEnabled(False)
            return

        if bundle is None:
            self.warning_label.setText("没有检测到当前 .app 包路径，无法执行自替换。")
            self.start_btn.setEnabled(False)
            return

        if _looks_like_volume_path(bundle) or not _is_applications_bundle(bundle):
            self.warning_label.setText(
                "请先把这个 app 拖到 Applications 目录，再从 Applications 里打开。"
            )
            self.start_btn.setEnabled(False)
            self.status_label.setText("等待被拖入 Applications。")
            return

        self.warning_label.setText(
            "检测到你已经从 Applications 启动。下载完成后程序会退出，必要时系统会请求授权来替换当前 app。"
        )
        self.ready_to_install = True
        self.start_btn.setEnabled(not self.downloading)

    def _set_running(self, running: bool) -> None:
        self.downloading = running
        self.start_btn.setEnabled(self.ready_to_install and not running)
        self.cancel_btn.setEnabled(running and not self._cancelling)
        self.apps_btn.setEnabled(not running)
        self.log_btn.setEnabled(not self._closing_after_cancel)

    def _cache_has_content(self) -> bool:
        try:
            return self.cache_dir.exists() and any(self.cache_dir.iterdir())
        except Exception:
            return False

    def _ask_close_cache_mode(self, *, downloading: bool) -> Optional[str]:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("关闭安装器")
        if downloading:
            box.setText("当前正在下载/准备完整版。关闭前要不要保留缓存？")
            box.setInformativeText("保留缓存后，下次可以继续；清理缓存则会删除当前已下载内容。")
        else:
            box.setText("本地已有缓存。关闭前要不要保留它们？")
            box.setInformativeText("保留缓存可减少下次下载时间；清理缓存会删除已下载的安装包与暂存文件。")
        keep_btn = box.addButton("保留缓存并关闭", QMessageBox.AcceptRole)
        clear_btn = box.addButton("清理缓存并关闭", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("取消", QMessageBox.RejectRole)
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

    def _begin_cancel(self, *, clear_cache: bool, close_after_cancel: bool) -> None:
        if self._cancelling:
            return
        self._cancelling = True
        self._closing_after_cancel = bool(close_after_cancel)
        self._cancel_clears_cache = bool(clear_cache)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.apps_btn.setEnabled(False)
        self.log_btn.setEnabled(not close_after_cancel)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("正在取消当前任务...")
        if clear_cache:
            self._append_log("[cancel] stop requested; clearing cache")
        else:
            self._append_log("[cancel] stop requested; preserving cache")
        if self.worker is not None:
            self.worker.cancel(clear_cache=clear_cache)
            return
        self.downloading = False
        self._cancelling = False
        if clear_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        if close_after_cancel:
            QTimer.singleShot(0, self.close)

    def cancel_download(self) -> None:
        if not self.downloading or self._cancelling:
            return
        if (
            QMessageBox.question(
                self,
                "取消下载",
                "取消当前下载后，会放弃断点续传并清理当前缓存。\n\n确定要取消吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self._begin_cancel(clear_cache=True, close_after_cancel=False)

    def start_install(self) -> None:
        if self.downloading:
            return

        if not self.installer_url:
            QMessageBox.warning(self, "Missing URL", "This lightweight app does not have an embedded Hugging Face URL yet.")
            return

        if self.target_bundle is None or not _is_applications_bundle(self.target_bundle):
            QMessageBox.information(
                self,
                "Move To Applications First",
                "请先把 app 拖到 Applications，然后从 Applications 里再打开一次。",
            )
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在准备完整版安装包...")
        self._append_log(f"[target] {self.target_bundle}")
        self._append_log(f"[cache] {self.cache_dir}")
        self._append_log(f"[url] {self.installer_url}")
        if TLS_CERT_PATH is not None:
            self._append_log(f"[tls-ca] {TLS_CERT_PATH}")
        self._closing_after_cancel = False
        self._cancelling = False
        self._cancel_clears_cache = False
        self._set_running(True)
        self.last_prepared = None

        self.worker_thread = QThread(self)
        self.worker = BootstrapWorker(
            url=self.installer_url,
            sha256=self.installer_sha256,
            cache_dir=self.cache_dir,
            target_bundle=self.target_bundle,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self.status_label.setText)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _on_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(downloaded)
            self.status_label.setText(f"正在下载：{_human_bytes(downloaded)} / {_human_bytes(total)}")
        else:
            self.progress_bar.setRange(0, 0)
            self.status_label.setText(f"正在下载：{_human_bytes(downloaded)}")

    def _on_finished(self, success: bool, payload: object) -> None:
        self._set_running(False)
        self.worker = None
        self.worker_thread = None

        if isinstance(payload, dict) and payload.get("cancelled"):
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.downloading = False
            self._cancelling = False
            cache_cleared = bool(payload.get("cache_cleared"))
            if cache_cleared:
                self.status_label.setText("已取消下载，缓存已清理。")
            else:
                self.status_label.setText("已取消下载，缓存已保留，可下次继续。")
            if self._closing_after_cancel:
                QTimer.singleShot(0, self.close)
                return
            self.start_btn.setEnabled(self.ready_to_install)
            self.cancel_btn.setEnabled(False)
            self.apps_btn.setEnabled(True)
            self.log_btn.setEnabled(True)
            return

        if not success:
            self._cancelling = False
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.status_label.setText("完整版准备失败。")
            self.cancel_btn.setEnabled(False)
            self.apps_btn.setEnabled(True)
            self.log_btn.setEnabled(True)
            QMessageBox.critical(self, "Preparation Failed", str(payload))
            return

        self._cancelling = False
        result = payload if isinstance(payload, dict) else {}
        self.last_prepared = {str(k): str(v) for k, v in result.items()}
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_label.setText("完整版已就绪，准备替换当前轻量版。")
        self._append_log(f"[prepared] {json.dumps(self.last_prepared, ensure_ascii=False)}")

        confirm = QMessageBox.question(
            self,
            "Replace Lightweight App",
            "完整版已经下载并解包完成。\n\n接下来会退出当前轻量 app，并把它替换为正式版；如果 /Applications 需要权限，系统会弹出授权提示。\n\n现在继续吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if confirm != QMessageBox.Yes:
            self.status_label.setText("已取消自动替换，缓存文件保留在本地。")
            return

        helper_path = Path(self.last_prepared.get("helper_path", ""))
        if not helper_path.exists():
            QMessageBox.critical(self, "Helper Missing", f"Replacement helper not found:\n{helper_path}")
            return

        subprocess.Popen(
            ["/bin/bash", str(helper_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.status_label.setText("正在退出轻量版并切换到正式版...")
        QTimer.singleShot(180, QApplication.instance().quit)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.downloading:
            if self._cancelling:
                event.ignore()
                return
            mode = self._ask_close_cache_mode(downloading=True)
            if mode is None:
                event.ignore()
                return
            event.ignore()
            self._begin_cancel(clear_cache=(mode == "clear"), close_after_cancel=True)
            return
        if self._cache_has_content():
            mode = self._ask_close_cache_mode(downloading=False)
            if mode is None:
                event.ignore()
                return
            if mode == "clear":
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                self._append_log(f"[cache-cleared] {self.cache_dir}")
            event.accept()
            return
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} macOS bootstrap launcher")
    parser.add_argument("--url", default=DEFAULT_MACOS_INSTALLER_URL, help="Hosted full macOS payload URL")
    parser.add_argument("--sha256", default=DEFAULT_MACOS_INSTALLER_SHA256, help="Optional hosted payload SHA256")
    parser.add_argument("--dir", default=str(DEFAULT_CACHE_DIR), help="Cache directory used for first-run download")
    parser.add_argument("--auto-start", dest="auto_start", action="store_true", help="Start downloading immediately on launch")
    parser.add_argument("--no-auto-start", dest="auto_start", action="store_false", help="Do not auto-start download on launch")
    parser.set_defaults(auto_start=_default_auto_start())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    window = InstallerWindow(
        installer_url=str(args.url or ""),
        installer_sha256=str(args.sha256 or ""),
        cache_dir=Path(str(args.dir or DEFAULT_CACHE_DIR)).expanduser(),
        auto_start=bool(args.auto_start),
    )
    if icon is not None:
        try:
            window.setWindowIcon(icon)
        except Exception:
            pass
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
