from __future__ import annotations

import logging
import os
import shutil
import socket
import ssl
import sys
import time
from pathlib import Path
from typing import Any

try:
    import sitecustomize  # noqa: F401
except Exception:
    sitecustomize = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)
_DOWNLOAD_ROUTE_ENV_CONFIGURED = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def app_root() -> Path:
    """Return the user-facing application root in dev and PyInstaller modes."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()


def internal_root() -> Path:
    """Return the PyInstaller onedir internal root, or app root in dev mode."""
    if getattr(sys, "frozen", False):
        candidates: list[Path] = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(str(meipass)))
        candidates.extend(
            [
                APP_ROOT / "_internal",
                APP_ROOT.parent / "Resources" / "_internal",
                APP_ROOT.parent / "Frameworks" / "_internal",
            ]
        )
        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate.resolve()
            except Exception:
                continue
        return APP_ROOT / "_internal"
    return APP_ROOT


INTERNAL_ROOT = internal_root()


def user_app_support_root() -> Path:
    """Return a writable per-user application data root."""
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        elif os.name == "nt":
            base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        else:
            base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
        root = base / "MediaTranscribeStudio"
    else:
        root = APP_ROOT

    root.mkdir(parents=True, exist_ok=True)
    return root


USER_APP_SUPPORT_ROOT = user_app_support_root()


def user_log_root() -> Path:
    """Return a writable per-user log directory."""
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            base = Path.home() / "Library" / "Logs"
        elif os.name == "nt":
            base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        else:
            base = Path(os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state"))
        root = base / "MediaTranscribeStudio"
    else:
        root = APP_ROOT

    root.mkdir(parents=True, exist_ok=True)
    return root


USER_LOG_ROOT = user_log_root()


def resolve_app_writable_path(path_like: str | Path, *, kind: str = "data") -> Path:
    """
    Resolve app-managed writable paths safely in dev and packaged runs.

    Relative paths stay project-relative in development, but move to per-user
    writable directories in frozen builds to avoid writing inside the app bundle.
    """
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    base = USER_LOG_ROOT if kind == "log" else USER_APP_SUPPORT_ROOT
    return base / path


def runtime_cache_root() -> Path:
    """Return a writable runtime cache root for dev and packaged runs."""
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            base = Path.home() / "Library" / "Caches"
        elif os.name == "nt":
            base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
        root = base / "MediaTranscribeStudio"
    else:
        root = APP_ROOT / ".runtime_cache"

    root.mkdir(parents=True, exist_ok=True)
    return root


RUNTIME_CACHE_ROOT = runtime_cache_root()


def _resolve_host_addresses(
    host: str,
    *,
    port: int = 443,
) -> tuple[list[tuple[int, tuple[Any, ...]]], float | None]:
    target = str(host or "").strip()
    if not target:
        return [], None

    start = time.perf_counter()
    try:
        infos = socket.getaddrinfo(
            target,
            int(port),
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except Exception:
        return [], None

    resolved_ms = max(0.001, float(time.perf_counter() - start))
    addresses: list[tuple[int, tuple[Any, ...]]] = []
    seen: set[tuple[int, str, int]] = set()
    for family, _socktype, _proto, _canonname, sockaddr in infos:
        try:
            host_part = str(sockaddr[0])
            port_part = int(sockaddr[1])
        except Exception:
            continue
        key = (int(family), host_part, port_part)
        if key in seen:
            continue
        seen.add(key)
        addresses.append((int(family), tuple(sockaddr)))
    return addresses, resolved_ms


def _probe_sockaddr_latency(
    family: int,
    sockaddr: tuple[Any, ...],
    timeout_sec: float,
) -> float | None:
    sock: socket.socket | None = None
    start = time.perf_counter()
    try:
        sock = socket.socket(int(family), socket.SOCK_STREAM)
        sock.settimeout(max(0.15, float(timeout_sec)))
        sock.connect(sockaddr)
        return max(0.001, float(time.perf_counter() - start))
    except Exception:
        return None
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


def _probe_tls_latency(
    host: str,
    family: int,
    sockaddr: tuple[Any, ...],
    timeout_sec: float,
) -> float | None:
    raw_sock: socket.socket | None = None
    tls_sock: ssl.SSLSocket | None = None
    start = time.perf_counter()
    try:
        raw_sock = socket.socket(int(family), socket.SOCK_STREAM)
        raw_sock.settimeout(max(0.15, float(timeout_sec)))
        raw_sock.connect(sockaddr)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        tls_sock = context.wrap_socket(raw_sock, server_hostname=str(host or "").strip())
        return max(0.001, float(time.perf_counter() - start))
    except Exception:
        return None
    finally:
        if tls_sock is not None:
            try:
                tls_sock.close()
            except Exception:
                pass
        elif raw_sock is not None:
            try:
                raw_sock.close()
            except Exception:
                pass


def _probe_endpoint_metrics(host: str, timeout_sec: float) -> dict[str, Any]:
    target = str(host or "").strip()
    result: dict[str, Any] = {
        "host": target,
        "dns_ms": None,
        "tcp_ms": None,
        "tls_ms": None,
        "addr_count": 0,
    }
    if not target:
        return result

    addresses, dns_ms = _resolve_host_addresses(target)
    result["dns_ms"] = dns_ms
    result["addr_count"] = len(addresses)
    if not addresses:
        return result

    best_family = 0
    best_sockaddr: tuple[Any, ...] | None = None
    best_tcp: float | None = None
    for family, sockaddr in addresses[:3]:
        tcp_ms = _probe_sockaddr_latency(family, sockaddr, timeout_sec)
        if tcp_ms is None:
            continue
        if best_tcp is None or tcp_ms < best_tcp:
            best_tcp = tcp_ms
            best_family = family
            best_sockaddr = sockaddr
    result["tcp_ms"] = best_tcp
    if best_sockaddr is None:
        return result

    best_tls = _probe_tls_latency(target, best_family, best_sockaddr, timeout_sec)
    if best_tls is None:
        for family, sockaddr in addresses[:3]:
            if family == best_family and sockaddr == best_sockaddr:
                continue
            best_tls = _probe_tls_latency(target, family, sockaddr, timeout_sec)
            if best_tls is not None:
                break
    result["tls_ms"] = best_tls
    return result


def _probe_score(metrics: dict[str, Any]) -> float | None:
    for key in ("tls_ms", "tcp_ms", "dns_ms"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return float(value)
    return None


def _format_probe_metrics(label: str, metrics: dict[str, Any]) -> str:
    score = _probe_score(metrics)
    if score is None:
        return f"{label}=unreachable"

    parts: list[str] = []
    for key, prefix in (("dns_ms", "dns"), ("tcp_ms", "tcp"), ("tls_ms", "tls")):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and float(value) > 0.0:
            parts.append(f"{prefix}{int(round(float(value) * 1000.0))}ms")
    return f"{label}=" + "/".join(parts or [f"{int(round(score * 1000.0))}ms"])


def detect_download_route(timeout_sec: float | None = None) -> dict[str, str]:
    try:
        probe_timeout = max(
            0.20,
            min(
                1.20,
                float(
                    timeout_sec
                    if timeout_sec is not None
                    else (os.environ.get("MTS_ROUTE_PROBE_TIMEOUT_SEC", "0.55") or "0.55")
                ),
            ),
        )
    except Exception:
        probe_timeout = 0.55

    probe_table = {
        "official": _probe_endpoint_metrics("huggingface.co", probe_timeout),
        "mirror": _probe_endpoint_metrics("hf-mirror.com", probe_timeout),
        "google": _probe_endpoint_metrics("www.google.com", probe_timeout),
        "gstatic": _probe_endpoint_metrics("www.gstatic.com", probe_timeout),
        "baidu": _probe_endpoint_metrics("www.baidu.com", probe_timeout),
        "qq": _probe_endpoint_metrics("www.qq.com", probe_timeout),
    }
    probe_scores = {
        key: _probe_score(value)
        for key, value in probe_table.items()
    }

    official_score = probe_scores.get("official")
    mirror_score = probe_scores.get("mirror")
    global_scores = [
        score for key, score in probe_scores.items() if key in {"google", "gstatic"} and score is not None
    ]
    cn_scores = [
        score for key, score in probe_scores.items() if key in {"baidu", "qq"} and score is not None
    ]
    best_global = min(global_scores) if global_scores else None
    best_cn = min(cn_scores) if cn_scores else None

    region = "global"
    if best_global is not None and official_score is not None:
        if mirror_score is None or official_score <= mirror_score * 1.15:
            region = "global"
        elif best_cn is not None and mirror_score is not None and mirror_score <= official_score * 1.10:
            region = "cn"
        else:
            region = "global"
    elif best_cn is not None:
        if best_global is None:
            region = "cn"
        elif mirror_score is not None and (official_score is None or mirror_score <= official_score * 1.10):
            region = "cn"
        elif best_cn < best_global * 0.90:
            region = "cn"
        else:
            region = "global"
    elif mirror_score is not None and official_score is None:
        region = "cn"
    elif official_score is not None and mirror_score is None:
        region = "global"
    elif mirror_score is not None and official_score is not None:
        region = "cn" if mirror_score < official_score else "global"

    note = (
        f"startup-probe[dns/tcp/tls] region={region} ("
        + ", ".join(
            _format_probe_metrics(label, probe_table[label])
            for label in ("official", "mirror", "google", "gstatic", "baidu", "qq")
        )
        + ")"
    )
    primary_endpoint = "https://hf-mirror.com" if region == "cn" else "https://huggingface.co"
    return {
        "region": region,
        "note": note,
        "primary_endpoint": primary_endpoint,
    }


def configure_startup_download_environment() -> dict[str, str]:
    global _DOWNLOAD_ROUTE_ENV_CONFIGURED
    applied: dict[str, str] = {}
    if _DOWNLOAD_ROUTE_ENV_CONFIGURED:
        return applied

    region = str(os.environ.get("MTS_DOWNLOAD_REGION", "") or "").strip().lower()
    note = str(os.environ.get("MTS_DOWNLOAD_REGION_NOTE", "") or "").strip()

    if region not in {"cn", "global"}:
        if _env_flag("MTS_DISABLE_STARTUP_ROUTE_DETECT", False):
            region = "global"
            note = "startup route detect disabled"
        else:
            route = detect_download_route()
            region = str(route.get("region", "global") or "global").strip().lower()
            note = str(route.get("note", "") or "").strip()

    primary_endpoint = (
        "https://hf-mirror.com" if region == "cn" else "https://huggingface.co"
    )
    for env_name, value in (
        ("MTS_DOWNLOAD_REGION", region or "global"),
        ("MTS_DOWNLOAD_REGION_NOTE", note),
        ("MTS_HF_PRIMARY_ENDPOINT", primary_endpoint),
    ):
        if value and os.environ.get(env_name) != value:
            os.environ[env_name] = value
            applied[env_name] = value

    current_hf_endpoint = str(os.environ.get("HF_ENDPOINT", "") or "").strip()
    if current_hf_endpoint in {"", "https://huggingface.co", "https://hf-mirror.com"}:
        if current_hf_endpoint != primary_endpoint:
            os.environ["HF_ENDPOINT"] = primary_endpoint
            applied["HF_ENDPOINT"] = primary_endpoint

    _DOWNLOAD_ROUTE_ENV_CONFIGURED = True
    return applied


def configure_runtime_cache_environment() -> dict[str, str]:
    """Point library caches to a writable runtime cache root."""
    applied: dict[str, str] = {}
    cache_map = {
        "MPLCONFIGDIR": RUNTIME_CACHE_ROOT / "matplotlib",
    }
    for env_name, path in cache_map.items():
        path.mkdir(parents=True, exist_ok=True)
        value = str(path)
        if os.environ.get(env_name) != value:
            os.environ[env_name] = value
            applied[env_name] = value

    if sys.platform == "darwin":
        darwin_defaults = {
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "MTS_TORCH_MPS_BRIDGE": "1",
            "MTS_TORCH_MPS_BRIDGE_TARGET": "15.0.0",
            "MTS_TORCH_MPS_BRIDGE_FUTURE_ONLY": "1",
        }
        for env_name, value in darwin_defaults.items():
            if os.environ.get(env_name) != value:
                os.environ[env_name] = value
                applied[env_name] = value
    applied.update(configure_startup_download_environment())
    return applied


DEFAULT_RUNTIME_CACHE_ENV = configure_runtime_cache_environment()


def bundled_internal_path(*parts: str) -> Path:
    return INTERNAL_ROOT.joinpath(*parts)


def _path_key(value: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(str(value)))
    except Exception:
        return str(value).lower()


def _prepend_path_entries(entries: list[Path]) -> bool:
    valid: list[str] = []
    for p in entries:
        try:
            if p.exists() and p.is_dir():
                valid.append(str(p))
        except Exception:
            continue
    if not valid:
        return False

    current_parts = [x for x in str(os.environ.get("PATH", "")).split(os.pathsep) if x]
    seen = {_path_key(x) for x in current_parts}
    prepend: list[str] = []
    for item in valid:
        key = _path_key(item)
        if key in seen:
            continue
        seen.add(key)
        prepend.append(item)
    if not prepend:
        return False

    os.environ["PATH"] = os.pathsep.join(prepend + current_parts)
    return True


def _first_existing_file(candidates: list[Path]) -> Path | None:
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def find_tool_executable(
    name: str,
    configured: str = "",
    *,
    extra_env_vars: tuple[str, ...] = (),
) -> str | None:
    """
    Resolve a tool executable path in both dev and PyInstaller frozen runtime.

    Resolution order:
    1) Explicit configured path/name
    2) Tool-specific environment variables
    3) Bundled onedir tool folders (_internal/tools/ffmpeg/*)
    4) Common ffmpeg root env vars (FFMPEG_DIR / FFMPEG_HOME / ...)
    5) Common conda locations
    6) PATH lookup via shutil.which
    """
    tool = str(name or "").strip()
    if not tool:
        return None

    exe_names = [f"{tool}.exe", tool] if os.name == "nt" else [tool]

    def _resolve_candidate(value: str) -> str | None:
        candidate = str(value or "").strip().strip('"')
        if not candidate:
            return None

        p = Path(candidate).expanduser()
        try:
            if p.exists() and p.is_file():
                return str(p.resolve())
            if p.exists() and p.is_dir():
                for exe_name in exe_names:
                    sub = p / exe_name
                    if sub.exists() and sub.is_file():
                        return str(sub.resolve())
        except Exception:
            pass

        resolved = shutil.which(candidate)
        if resolved:
            try:
                return str(Path(resolved).resolve())
            except Exception:
                return str(resolved)
        return None

    # 1) explicit configured
    resolved = _resolve_candidate(configured)
    if resolved:
        return resolved

    # 2) environment variables
    env_names = [f"{tool.upper()}_BINARY", *list(extra_env_vars)]
    if tool == "ffprobe":
        env_names.extend(["FFPROBE_BINARY"])
    elif tool == "ffmpeg":
        env_names.extend(["FFMPEG_BINARY"])
    for env_name in env_names:
        raw = str(os.environ.get(env_name, "") or "").strip()
        if not raw:
            continue
        resolved = _resolve_candidate(raw)
        if resolved:
            return resolved

    if tool == "ffprobe":
        ffmpeg_env = str(os.environ.get("FFMPEG_BINARY", "") or "").strip().strip('"')
        if ffmpeg_env:
            ffmpeg_path = Path(ffmpeg_env).expanduser()
            sibling_candidates = [
                ffmpeg_path.parent / "ffprobe.exe",
                ffmpeg_path.parent / "ffprobe",
                ffmpeg_path.parent / "bin" / "ffprobe.exe",
                ffmpeg_path.parent / "bin" / "ffprobe",
            ]
            sibling = _first_existing_file(sibling_candidates)
            if sibling is not None:
                try:
                    return str(sibling.resolve())
                except Exception:
                    return str(sibling)

    # 3) bundled/project-local directories
    bundled_dirs = [
        INTERNAL_ROOT / "native" / "bin",
        INTERNAL_ROOT / "native",
        INTERNAL_ROOT / "tools" / "native" / "bin",
        INTERNAL_ROOT / "tools" / "native",
        APP_ROOT / "native" / "install" / "bin",
        APP_ROOT / "native" / "install",
        APP_ROOT / "build" / "native" / "install" / "bin",
        APP_ROOT / "build" / "native" / "install",
        APP_ROOT / "native" / "build" / "Release",
        APP_ROOT / "native" / "build" / "Debug",
        APP_ROOT / "build" / "native" / "Release",
        APP_ROOT / "build" / "native" / "Debug",
        INTERNAL_ROOT / "tools" / "ffmpeg" / "bin",
        INTERNAL_ROOT / "tools" / "ffmpeg",
        INTERNAL_ROOT / "ffmpeg" / "bin",
        INTERNAL_ROOT / "ffmpeg",
    ]
    for folder in bundled_dirs:
        for exe_name in exe_names:
            try:
                candidate = folder / exe_name
                if candidate.exists() and candidate.is_file():
                    return str(candidate.resolve())
            except Exception:
                continue

    # 4) explicit native/ffmpeg roots
    for env_name in ("MTS_NATIVE_ROOT", "BUNDLE_NATIVE_DIR", "BUNDLE_FFMPEG_DIR", "FFMPEG_DIR", "FFMPEG_HOME", "FFMPEG_ROOT"):
        raw = str(os.environ.get(env_name, "") or "").strip().strip('"')
        if not raw:
            continue
        base = Path(raw).expanduser()
        for folder in (base, base / "bin"):
            for exe_name in exe_names:
                try:
                    candidate = folder / exe_name
                    if candidate.exists() and candidate.is_file():
                        return str(candidate.resolve())
                except Exception:
                    continue

    # 5) common conda locations
    conda_roots = []
    conda_prefix = str(os.environ.get("CONDA_PREFIX", "") or "").strip()
    if conda_prefix:
        conda_roots.append(Path(conda_prefix))
    try:
        conda_roots.append(Path(sys.prefix))
    except Exception:
        pass
    for root in conda_roots:
        for folder in (root / "Library" / "bin", root / "Scripts", root / "bin"):
            for exe_name in exe_names:
                try:
                    candidate = folder / exe_name
                    if candidate.exists() and candidate.is_file():
                        return str(candidate.resolve())
                except Exception:
                    continue

    # 6) PATH lookup
    resolved = shutil.which(tool)
    if resolved:
        try:
            return str(Path(resolved).resolve())
        except Exception:
            return str(resolved)
    return None


def configure_bundled_runtime_environment() -> dict[str, str]:
    """
    Point runtime caches/tools to bundled resources when running a frozen build.

    Returns a mapping of environment variables that were applied/updated.
    """
    applied: dict[str, str] = {}
    internal = INTERNAL_ROOT

    applied.update(configure_runtime_cache_environment())

    if not internal.exists():
        return applied

    native_dirs = [
        internal / "native" / "bin",
        internal / "native",
        internal / "tools" / "native" / "bin",
        internal / "tools" / "native",
    ]
    if _prepend_path_entries(native_dirs):
        applied["PATH"] = os.environ.get("PATH", "")

    native_root = next(
        (
            folder
            for folder in (internal / "native", internal / "tools" / "native")
            if folder.exists() and folder.is_dir()
        ),
        None,
    )
    if native_root is not None:
        native_root_str = str(native_root)
        if os.environ.get("MTS_NATIVE_ROOT") != native_root_str:
            os.environ["MTS_NATIVE_ROOT"] = native_root_str
            applied["MTS_NATIVE_ROOT"] = native_root_str

    # Prefer bundled ffmpeg/ffprobe (when packaged under _internal/tools/ffmpeg).
    ffmpeg_dirs = [
        internal / "tools" / "ffmpeg" / "bin",
        internal / "tools" / "ffmpeg",
        internal / "ffmpeg" / "bin",
        internal / "ffmpeg",
    ]
    if _prepend_path_entries(ffmpeg_dirs):
        applied["PATH"] = os.environ.get("PATH", "")

    ffmpeg_exe = _first_existing_file(
        [p / "ffmpeg.exe" for p in ffmpeg_dirs] + [p / "ffmpeg" for p in ffmpeg_dirs]
    )
    if ffmpeg_exe is None:
        ffmpeg_resolved = find_tool_executable("ffmpeg")
        if ffmpeg_resolved:
            ffmpeg_exe = Path(ffmpeg_resolved)
    if ffmpeg_exe is not None:
        ffmpeg_str = str(ffmpeg_exe)
        if os.environ.get("FFMPEG_BINARY") != ffmpeg_str:
            os.environ["FFMPEG_BINARY"] = ffmpeg_str
            applied["FFMPEG_BINARY"] = ffmpeg_str
        if os.environ.get("IMAGEIO_FFMPEG_EXE") != ffmpeg_str:
            os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_str
            applied["IMAGEIO_FFMPEG_EXE"] = ffmpeg_str
        _prepend_path_entries([ffmpeg_exe.parent])
        applied["PATH"] = os.environ.get("PATH", "")

    ffprobe_exe = _first_existing_file(
        [p / "ffprobe.exe" for p in ffmpeg_dirs] + [p / "ffprobe" for p in ffmpeg_dirs]
    )
    if ffprobe_exe is None and ffmpeg_exe is not None:
        # Many FFmpeg distributions place ffprobe beside ffmpeg.
        ffprobe_exe = _first_existing_file(
            [
                ffmpeg_exe.parent / "ffprobe.exe",
                ffmpeg_exe.parent / "ffprobe",
                ffmpeg_exe.parent / "bin" / "ffprobe.exe",
                ffmpeg_exe.parent / "bin" / "ffprobe",
            ]
        )
    if ffprobe_exe is None:
        ffprobe_resolved = find_tool_executable("ffprobe")
        if ffprobe_resolved:
            ffprobe_exe = Path(ffprobe_resolved)
    if ffprobe_exe is not None:
        ffprobe_str = str(ffprobe_exe)
        if os.environ.get("FFPROBE_BINARY") != ffprobe_str:
            os.environ["FFPROBE_BINARY"] = ffprobe_str
            applied["FFPROBE_BINARY"] = ffprobe_str
        _prepend_path_entries([ffprobe_exe.parent])
        applied["PATH"] = os.environ.get("PATH", "")

    # Prefer bundled Playwright browsers if they were packed into the onedir build.
    pw_browsers = internal / "playwright" / "driver" / "package" / ".local-browsers"
    if pw_browsers.exists():
        target = str(pw_browsers)
        if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") != target:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = target
            applied["PLAYWRIGHT_BROWSERS_PATH"] = target
        os.environ.setdefault("PLAYWRIGHT_SKIP_BROWSER_GC", "1")

    # Optional pre-bundled model caches (if the build packed them under _internal/model_caches).
    cache_root = internal / "model_caches"
    if cache_root.exists():
        cache_map = {
            "HF_HOME": cache_root / "huggingface",
            "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
            "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
            "TRANSFORMERS_CACHE": cache_root / "huggingface" / "hub",
            "HF_DATASETS_CACHE": cache_root / "huggingface" / "datasets",
            "MODELSCOPE_CACHE": cache_root / "modelscope",
            "TORCH_HOME": cache_root / "torch",
            "TORCH_HUB": cache_root / "torch" / "hub",
            "NEMO_CACHE_DIR": cache_root / "nemo",
            "NEMO_HOME": cache_root / "nemo",
        }
        for env_name, path in cache_map.items():
            if not path.exists():
                continue
            value = str(path)
            if os.environ.get(env_name) != value:
                os.environ[env_name] = value
                applied[env_name] = value

    return applied


DEFAULT_BUNDLED_RUNTIME_ENV = configure_bundled_runtime_environment()
