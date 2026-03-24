from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import getpass
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx
from huggingface_hub import HfApi, get_token as hf_hub_get_token
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.lfs import LFS_HEADERS, UploadInfo, post_lfs_batch_info
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "dist" / "MediaTranscribeStudio-Full-macOS.zip"
if not DEFAULT_FILE.exists():
    DEFAULT_FILE = ROOT / "dist" / "MediaTranscribeStudio-Full-macOS.dmg"
DEFAULT_REPO_ID = str(os.environ.get("MACOS_HF_REPO_ID", "") or "").strip() or "your-org/your-repo"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_REVISION = "main"
DEFAULT_WORKERS = 6
DEFAULT_REQUEST_RETRIES = -1
DEFAULT_STATE_ROOT = ROOT / "build" / "hf_upload_state"
DEFAULT_ENDPOINT = (
    str(os.environ.get("MTS_HF_UPLOAD_ENDPOINT", "") or "").strip()
    or str(os.environ.get("HF_ENDPOINT", "") or "").strip()
    or str(os.environ.get("MTS_HF_PRIMARY_ENDPOINT", "") or "").strip()
    or "https://huggingface.co"
)
HASH_READ_SIZE = 64 * 1024 * 1024
STREAM_READ_SIZE = 4 * 1024 * 1024
STATE_VERSION = 1
RETRYABLE_HTTPX_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.ProtocolError,
    httpx.RemoteProtocolError,
)
API_REQUEST_TIMEOUT = None
PART_UPLOAD_TIMEOUT = None
COMPLETION_REQUEST_TIMEOUT = None
UPLOAD_SOURCE_DIR_NAME = "upload_source"
UPLOAD_TOKEN_ENV_NAMES = ("MACOS_HF_UPLOAD_TOKEN", "MTS_HF_UPLOAD_TOKEN", "HF_WRITE_TOKEN")
HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
DEFAULT_MACOS_HF_UPLOAD_TOKEN = ""
_THREAD_HTTP_CLIENT = threading.local()


class UploadSessionExpiredError(RuntimeError):
    pass


class NetworkRetryError(RuntimeError):
    pass


class PartFileObj:
    """Seekable file-like view over a byte range of a local file."""

    def __init__(self, path: Path, offset: int, length: int) -> None:
        self._path = path
        self._offset = offset
        self._length = length
        self._file = path.open("rb")
        self._file.seek(offset)
        self._pos = 0

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        self._file.close()

    def read(self, size: int = -1) -> bytes:
        remaining = self._length - self._pos
        if remaining <= 0:
            return b""
        if size is None or size < 0 or size > remaining:
            size = remaining
        data = self._file.read(size)
        self._pos += len(data)
        return data

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self._pos + offset
        elif whence == os.SEEK_END:
            target = self._length + offset
        else:
            raise ValueError(f"Unsupported whence value: {whence}")
        target = max(0, min(target, self._length))
        self._file.seek(self._offset + target, os.SEEK_SET)
        self._pos = target
        return self._pos

    def __enter__(self) -> "PartFileObj":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self):
        while True:
            chunk = self.read(STREAM_READ_SIZE)
            if not chunk:
                break
            yield chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the built macOS full payload archive (zip/dmg) to a Hugging Face repo "
            "with multipart upload state saved locally for resume."
        )
    )
    parser.add_argument(
        "--file",
        default=str(DEFAULT_FILE),
        help=f"Local artifact to upload. Default: {DEFAULT_FILE}",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Target Hugging Face repo. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default=DEFAULT_REPO_TYPE,
        help=f"Target repo type. Default: {DEFAULT_REPO_TYPE}",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Target branch or revision. Default: {DEFAULT_REVISION}",
    )
    parser.add_argument(
        "--path-in-repo",
        default=DEFAULT_FILE.name,
        help=f"Remote file path inside the repo. Default: {DEFAULT_FILE.name}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel multipart upload workers. Default: {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--state-dir",
        default="",
        help="Optional local directory used to persist upload state for resume.",
    )
    parser.add_argument(
        "--commit-message",
        default=f"Upload {DEFAULT_FILE.name}",
        help="Commit summary to use after the blob upload is complete.",
    )
    parser.add_argument(
        "--commit-description",
        default="",
        help="Optional commit description.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Hub API endpoint. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=DEFAULT_REQUEST_RETRIES,
        help=f"Retries for Hub API calls on network failure. Use -1 for infinite retry. Default: {DEFAULT_REQUEST_RETRIES}",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--ensure-repo",
        action="store_true",
        help="Create the target repo if it does not already exist.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-upload LFS verify request.",
    )
    parser.add_argument(
        "--fresh-session",
        action="store_true",
        help="Ignore any cached multipart session and negotiate a new one.",
    )
    parser.add_argument(
        "--no-auto-restart-expired",
        action="store_true",
        help="Fail instead of clearing the local multipart session when the server says it expired.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def normalize_repo_path(path_in_repo: str) -> str:
    text = str(path_in_repo or "").strip().replace("\\", "/").lstrip("/")
    if not text:
        raise ValueError("`path_in_repo` cannot be empty.")
    normalized = PurePosixPath(text).as_posix()
    if normalized in {".", ""}:
        raise ValueError("`path_in_repo` resolves to an empty path.")
    if normalized.startswith("../") or "/../" in f"/{normalized}/" or normalized == "..":
        raise ValueError("`path_in_repo` cannot contain parent directory traversal.")
    return normalized


def sanitize_state_label(repo_id: str, repo_type: str, path_in_repo: str) -> str:
    raw = f"{repo_type}__{repo_id}__{path_in_repo}"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def resolve_state_dir(args: argparse.Namespace, path_in_repo: str) -> Path:
    if args.state_dir:
        return Path(args.state_dir).expanduser().resolve()
    label = sanitize_state_label(args.repo_id, args.repo_type, path_in_repo)
    return (DEFAULT_STATE_ROOT / label).resolve()


def default_state_dir(*, repo_id: str, repo_type: str, path_in_repo: str) -> Path:
    label = sanitize_state_label(repo_id, repo_type, path_in_repo)
    return (DEFAULT_STATE_ROOT / label).resolve()


def build_public_file_url(
    *,
    endpoint: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
) -> str:
    base = str(endpoint or "").strip().rstrip("/")
    if not base:
        raise ValueError("endpoint cannot be empty")

    normalized_repo_type = str(repo_type or "").strip().lower()
    normalized_repo_id = str(repo_id or "").strip()
    normalized_revision = str(revision or "").strip()
    normalized_path = normalize_repo_path(path_in_repo)
    if normalized_repo_type not in {"model", "dataset", "space"}:
        raise ValueError(f"Unsupported repo_type: {repo_type}")
    if not normalized_repo_id:
        raise ValueError("repo_id cannot be empty")
    if not normalized_revision:
        raise ValueError("revision cannot be empty")

    repo_segment = quote(normalized_repo_id, safe="/")
    revision_segment = quote(normalized_revision, safe="")
    path_segment = quote(normalized_path, safe="/")
    if normalized_repo_type == "dataset":
        return f"{base}/datasets/{repo_segment}/resolve/{revision_segment}/{path_segment}"
    if normalized_repo_type == "space":
        return f"{base}/spaces/{repo_segment}/resolve/{revision_segment}/{path_segment}"
    return f"{base}/{repo_segment}/resolve/{revision_segment}/{path_segment}"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def retries_are_infinite(retries: int) -> bool:
    return int(retries) < 0


def retry_limit_reached(attempt: int, retries: int) -> bool:
    if retries_are_infinite(retries):
        return False
    return attempt >= max(1, int(retries) + 1)


def describe_retry_attempt(attempt: int, retries: int) -> str:
    if retries_are_infinite(retries):
        return f"{attempt}/inf"
    return f"{attempt}/{max(1, int(retries) + 1)}"


def should_retry_closed_client(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    return "client has been closed" in text or "cannot send a request" in text


def reset_thread_http_client() -> None:
    client = getattr(_THREAD_HTTP_CLIENT, "client", None)
    setattr(_THREAD_HTTP_CLIENT, "client", None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def get_thread_http_client(timeout: httpx.Timeout | None) -> httpx.Client:
    client = getattr(_THREAD_HTTP_CLIENT, "client", None)
    if client is None or getattr(client, "is_closed", False):
        client = httpx.Client(timeout=timeout, follow_redirects=True)
        setattr(_THREAD_HTTP_CLIENT, "client", client)
    return client


def request_with_retry(
    method: str,
    url: str,
    *,
    description: str,
    request_retries: int,
    timeout: httpx.Timeout | None,
    retry_on_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
    **kwargs,
) -> httpx.Response:
    attempt = 0
    wait_seconds = 1.0
    io_obj_initial_pos = None
    data_obj = kwargs.get("data")
    if data_obj is not None:
        tell = getattr(data_obj, "tell", None)
        if callable(tell):
            try:
                io_obj_initial_pos = int(tell())
            except Exception:
                io_obj_initial_pos = None

    while True:
        attempt += 1
        if io_obj_initial_pos is not None:
            seek = getattr(data_obj, "seek", None)
            if callable(seek):
                try:
                    seek(io_obj_initial_pos)
                except Exception:
                    pass
        client = get_thread_http_client(timeout)
        try:
            response = client.request(method=method, url=url, timeout=timeout, **kwargs)
            if response.status_code in retry_on_status_codes:
                if retry_limit_reached(attempt, request_retries):
                    hf_raise_for_status(response)
                    return response
                print(
                    f"[network] {description} got HTTP {response.status_code}. "
                    f"Retrying in {int(wait_seconds)}s ({describe_retry_attempt(attempt, request_retries)}) ..."
                )
                response.close()
                time.sleep(wait_seconds)
                wait_seconds = min(wait_seconds * 2.0, 30.0)
                continue
            return response
        except RETRYABLE_HTTPX_EXCEPTIONS as exc:
            reset_thread_http_client()
            if retry_limit_reached(attempt, request_retries):
                raise NetworkRetryError(
                    f"{description} failed after {describe_retry_attempt(attempt, request_retries)} attempts: {exc}"
                ) from exc
            print(
                f"[network] {description} failed ({exc}). "
                f"Retrying in {int(wait_seconds)}s ({describe_retry_attempt(attempt, request_retries)}) ..."
            )
            time.sleep(wait_seconds)
            wait_seconds = min(wait_seconds * 2.0, 30.0)
        except RuntimeError as exc:
            if not should_retry_closed_client(exc):
                raise
            reset_thread_http_client()
            if retry_limit_reached(attempt, request_retries):
                raise NetworkRetryError(
                    f"{description} failed after {describe_retry_attempt(attempt, request_retries)} attempts: {exc}"
                ) from exc
            print(
                f"[network] {description} lost its HTTP client ({exc}). "
                f"Retrying in {int(wait_seconds)}s ({describe_retry_attempt(attempt, request_retries)}) ..."
            )
            time.sleep(wait_seconds)
            wait_seconds = min(wait_seconds * 2.0, 30.0)


def load_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read upload state from {state_file}: {exc}") from exc


def save_state(state_file: Path, state: dict[str, Any]) -> None:
    ensure_parent_dir(state_file)
    tmp_path = state_file.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    tmp_path.replace(state_file)


def prepare_upload_source_file(file_path: Path, state_dir: Path) -> Path:
    source_dir = state_dir / UPLOAD_SOURCE_DIR_NAME
    source_dir.mkdir(parents=True, exist_ok=True)
    staged_path = source_dir / file_path.name
    meta_path = source_dir / "source.json"

    def _write_meta(mode: str) -> None:
        meta = {
            "original_path": str(file_path),
            "staged_path": str(staged_path),
            "mode": mode,
            "updated_at": utc_now(),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        os.chmod(meta_path, 0o600)

    if staged_path.exists():
        try:
            if os.path.samefile(file_path, staged_path):
                _write_meta("existing")
                return staged_path
        except Exception:
            pass
        staged_stat = staged_path.stat()
        source_stat = file_path.stat()
        if staged_stat.st_size == source_stat.st_size and staged_stat.st_mtime_ns == source_stat.st_mtime_ns:
            _write_meta("existing")
            return staged_path
        staged_path.unlink()

    mode = "hardlink"
    try:
        os.link(file_path, staged_path)
    except Exception:
        mode = "copy"
        try:
            import shutil

            shutil.copy2(file_path, staged_path)
        except Exception:
            mode = "symlink"
            if staged_path.exists() or staged_path.is_symlink():
                staged_path.unlink()
            os.symlink(file_path, staged_path)

    _write_meta(mode)
    print(f"Upload source: {staged_path} ({mode})")
    return staged_path


def build_empty_state(
    *,
    file_path: Path,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "file": {
            "path": str(file_path),
        },
        "target": {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": revision,
            "path_in_repo": path_in_repo,
        },
        "upload": {
            "transfer": None,
            "chunk_size": None,
            "upload_url": None,
            "verify_url": None,
            "part_urls": {},
            "completed_parts": {},
            "upload_completed": False,
            "verified": False,
            "committed": False,
        },
    }


def empty_upload_state() -> dict[str, Any]:
    return {
        "transfer": None,
        "chunk_size": None,
        "upload_url": None,
        "verify_url": None,
        "part_urls": {},
        "completed_parts": {},
        "upload_completed": False,
        "verified": False,
        "committed": False,
    }


def reset_upload_tracking_state(state: dict[str, Any]) -> None:
    state["upload"] = empty_upload_state()
    state.pop("commit", None)
    update_state_timestamp(state)


def ensure_state_matches(
    state: dict[str, Any],
    *,
    file_path: Path,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
) -> dict[str, Any]:
    if not state:
        return build_empty_state(
            file_path=file_path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            path_in_repo=path_in_repo,
        )

    target = state.get("target") or {}
    state_file_path = str((state.get("file") or {}).get("path") or "")
    if (
        target.get("repo_id") != repo_id
        or target.get("repo_type") != repo_type
        or target.get("revision") != revision
        or target.get("path_in_repo") != path_in_repo
        or state_file_path != str(file_path)
        or int(state.get("version") or 0) != STATE_VERSION
    ):
        return build_empty_state(
            file_path=file_path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            path_in_repo=path_in_repo,
        )

    state.setdefault("upload", {})
    state["upload"].setdefault("part_urls", {})
    state["upload"].setdefault("completed_parts", {})
    state["upload"].setdefault("upload_completed", False)
    state["upload"].setdefault("verified", False)
    state["upload"].setdefault("committed", False)
    return state


def update_state_timestamp(state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()


@contextmanager
def acquire_process_lock(lock_path: Path):
    ensure_parent_dir(lock_path)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            holder = handle.read().strip()
            details = holder or "Another upload process is already active for this state directory."
            raise RuntimeError(
                f"Another upload_hf_dmg.py process is already using this upload state.\n{details}\n"
                "Stop the other process before starting a new one."
            ) from exc

        payload = {
            "pid": os.getpid(),
            "started_at": utc_now(),
            "command": " ".join(sys.argv),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()


def format_network_hint(endpoint: str) -> str:
    return (
        f"Cannot reach Hugging Face endpoint {endpoint}. "
        "Check network/proxy/VPN access to the Hub, or override the endpoint with "
        "`--endpoint ...` / `HF_ENDPOINT=...` if your environment requires a different route."
    )


def run_with_network_retry(
    description: str,
    func,
    *,
    retries: int,
    endpoint: str,
):
    last_exc: Exception | None = None
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except RETRYABLE_HTTPX_EXCEPTIONS as exc:
            last_exc = exc
            if retry_limit_reached(attempt, retries):
                break
            wait_seconds = min(2 ** (attempt - 1), 8)
            print(
                f"[network] {description} failed ({exc}). "
                f"Retrying in {wait_seconds}s ({describe_retry_attempt(attempt, retries)}) ..."
            )
            time.sleep(wait_seconds)
        except RuntimeError as exc:
            if not should_retry_closed_client(exc):
                raise
            last_exc = exc
            reset_thread_http_client()
            if retry_limit_reached(attempt, retries):
                break
            wait_seconds = min(2 ** (attempt - 1), 8)
            print(
                f"[network] {description} lost its HTTP client ({exc}). "
                f"Retrying in {wait_seconds}s ({describe_retry_attempt(attempt, retries)}) ..."
            )
            time.sleep(wait_seconds)

    assert last_exc is not None
    raise NetworkRetryError(
        f"{description} failed after {describe_retry_attempt(attempt, retries)} attempts: {last_exc}\n"
        f"{format_network_hint(endpoint)}"
    )


def get_token() -> str:
    token, source = resolve_upload_token(interactive=sys.stdin.isatty())
    if token:
        print(f"Using access token from {source}.")
        return token

    raise RuntimeError(
        "Missing Hugging Face access token. Set MACOS_HF_UPLOAD_TOKEN/HF_TOKEN, run `hf auth login`, or configure the token before running this script."
    )


def _store_upload_token(token: str) -> None:
    clean = str(token or "").strip()
    if not clean:
        return
    for env_name in (*UPLOAD_TOKEN_ENV_NAMES, *HF_TOKEN_ENV_NAMES):
        os.environ[env_name] = clean


def _read_env_token(env_names: tuple[str, ...]) -> tuple[str, str]:
    for env_name in env_names:
        value = str(os.environ.get(env_name, "") or "").strip()
        if value:
            _store_upload_token(value)
            return value, f"${env_name}"
    return "", ""


def resolve_upload_token(*, interactive: bool) -> tuple[str, str]:
    token, source = _read_env_token(UPLOAD_TOKEN_ENV_NAMES)
    if token:
        return token, source

    built_in_token = str(DEFAULT_MACOS_HF_UPLOAD_TOKEN or "").strip()
    if built_in_token:
        _store_upload_token(built_in_token)
        return built_in_token, "built-in macOS upload token"

    token, source = _read_env_token(HF_TOKEN_ENV_NAMES)
    if token:
        return token, source

    cached_token = str(hf_hub_get_token() or "").strip()
    if cached_token:
        _store_upload_token(cached_token)
        return cached_token, "local Hugging Face login"

    config_path = ROOT / "config.py"
    if config_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("upload_hf_config_runtime", config_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                config_cls = getattr(module, "Config", None)
                if config_cls is not None:
                    config_cls(config_path=None)
        except Exception as exc:
            print(f"[warn] Failed to load project config token fallback: {exc}")
        else:
            token, source = _read_env_token(HF_TOKEN_ENV_NAMES)
            if token:
                return token, "project config"

    if interactive and sys.stdin.isatty():
        token = getpass.getpass("Hugging Face access token: ").strip()
        if token:
            _store_upload_token(token)
            return token, "interactive prompt"

    return "", ""


def get_cached_identity(state: dict[str, Any], file_path: Path) -> tuple[dict[str, Any], bool]:
    file_meta = state.get("file") or {}
    stat = file_path.stat()
    cached = (
        file_meta.get("path") == str(file_path)
        and int(file_meta.get("size") or -1) == stat.st_size
        and int(file_meta.get("mtime_ns") or -1) == stat.st_mtime_ns
        and bool(file_meta.get("sha256"))
    )
    identity = {
        "path": str(file_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": str(file_meta.get("sha256") or ""),
    }
    return identity, cached


def compute_sha256(file_path: Path) -> str:
    total_size = file_path.stat().st_size
    hasher = hashlib.sha256()
    processed = 0
    last_report_at = 0.0
    started_at = time.time()

    print(f"Hashing {file_path} ({human_bytes(total_size)}) ...")
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_READ_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            processed += len(chunk)

            now = time.time()
            if now - last_report_at >= 3.0:
                elapsed = max(now - started_at, 0.001)
                rate = processed / elapsed
                percent = (processed / total_size * 100.0) if total_size else 100.0
                print(
                    f"[hash] {percent:6.2f}%  {human_bytes(processed)}/{human_bytes(total_size)}"
                    f"  {human_bytes(int(rate))}/s"
                )
                last_report_at = now

    elapsed = max(time.time() - started_at, 0.001)
    print(f"Hash complete in {elapsed:.1f}s: {hasher.hexdigest()}")
    return hasher.hexdigest()


def read_sample(file_path: Path, size: int = 512) -> bytes:
    with file_path.open("rb") as handle:
        return handle.read(size)


def build_upload_info(file_path: Path, state: dict[str, Any]) -> UploadInfo:
    previous_identity = dict(state.get("file") or {})
    identity, can_reuse_hash = get_cached_identity(state, file_path)
    if can_reuse_hash:
        print(f"Reusing cached SHA256 for {file_path.name}.")
    else:
        identity["sha256"] = compute_sha256(file_path)

    previous_sha256 = str(previous_identity.get("sha256") or "")
    current_sha256 = str(identity.get("sha256") or "")
    same_blob = bool(previous_sha256 and current_sha256 and previous_sha256 == current_sha256)
    same_target = str(previous_identity.get("path") or "") == str(identity.get("path") or "")
    if previous_identity and (not same_blob or not same_target):
        print("Detected a different upload source blob. Clearing cached multipart session and commit state.")
        reset_upload_tracking_state(state)

    state["file"] = identity
    update_state_timestamp(state)
    sample = read_sample(file_path)
    return UploadInfo(
        sha256=bytes.fromhex(identity["sha256"]),
        size=identity["size"],
        sample=sample,
    )


def get_sorted_part_urls(upload_state: dict[str, Any], upload_size: int, chunk_size: int) -> list[tuple[int, str]]:
    part_urls = upload_state.get("part_urls") or {}
    sorted_pairs = sorted((int(part_no), str(url)) for part_no, url in part_urls.items())
    expected_parts = math.ceil(upload_size / chunk_size)
    if len(sorted_pairs) != expected_parts:
        raise RuntimeError(
            f"Multipart session is incomplete: expected {expected_parts} part URLs but found {len(sorted_pairs)}."
        )
    return sorted_pairs


def request_new_upload_session(
    *,
    api: HfApi,
    token: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    upload_info: UploadInfo,
    request_retries: int,
) -> dict[str, Any]:
    instructions, errors, chosen_transfer = run_with_network_retry(
        "negotiating LFS upload session",
        lambda: post_lfs_batch_info(
            upload_infos=[upload_info],
            token=token,
            repo_type=repo_type,
            repo_id=repo_id,
            revision=revision,
            endpoint=api.endpoint,
        ),
        retries=request_retries,
        endpoint=api.endpoint,
    )
    if errors:
        raise RuntimeError(f"Hub LFS batch request returned errors: {errors}")
    if not instructions:
        raise RuntimeError("Hub LFS batch request returned no upload instructions.")

    instruction = instructions[0]
    actions = instruction.get("actions")
    if actions is None:
        return {
            "transfer": chosen_transfer or "already_present",
            "chunk_size": None,
            "upload_url": None,
            "verify_url": None,
            "part_urls": {},
            "completed_parts": {},
            "upload_completed": True,
            "verified": False,
            "committed": False,
        }

    upload_action = actions.get("upload") or {}
    verify_action = actions.get("verify") or {}
    header = upload_action.get("header") or {}
    chunk_size = header.get("chunk_size")

    if chunk_size is None:
        return {
            "transfer": "basic",
            "chunk_size": None,
            "upload_url": str(upload_action["href"]),
            "verify_url": str(verify_action.get("href") or ""),
            "part_urls": {},
            "completed_parts": {},
            "upload_completed": False,
            "verified": False,
            "committed": False,
        }

    part_urls = {str(int(part_no)): str(url) for part_no, url in header.items() if str(part_no).isdigit()}
    if not part_urls:
        raise RuntimeError("Multipart upload was selected but no part URLs were returned by the Hub.")

    return {
        "transfer": "multipart",
        "chunk_size": int(chunk_size),
        "upload_url": str(upload_action["href"]),
        "verify_url": str(verify_action.get("href") or ""),
        "part_urls": part_urls,
        "completed_parts": {},
        "upload_completed": False,
        "verified": False,
        "committed": False,
    }


def is_session_expired_exception(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        response_text = str(getattr(response, "text", "") or "")
        request = getattr(response, "request", None)
        request_url = str(getattr(request, "url", "") or "")
        if status_code in {401, 403} and any(
            marker in response_text.lower()
            for marker in ("expired", "expire", "signaturedoesnotmatch", "request has expired", "expiredtoken")
        ):
            return True
        if status_code == 400 and (
            "uploadid=" in request_url.lower()
            or "uploadpart" in request_url.lower()
            or any(
                marker in response_text.lower()
                for marker in (
                    "invalidpart",
                    "invalidpartorder",
                    "nosuchupload",
                    "signaturedoesnotmatch",
                    "request has expired",
                    "expiredtoken",
                    "bad request",
                )
            )
        ):
            return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "request has expired",
            "expiredtoken",
            "signaturedoesnotmatch",
            "invalidpart",
            "invalidpartorder",
            "nosuchupload",
            "uploadpart",
        )
    )


def print_upload_progress(
    *,
    completed_parts: int,
    total_parts: int,
    uploaded_bytes: int,
    total_bytes: int,
) -> None:
    percent = (uploaded_bytes / total_bytes * 100.0) if total_bytes else 100.0
    print(
        f"[upload] parts {completed_parts}/{total_parts}  "
        f"{human_bytes(uploaded_bytes)}/{human_bytes(total_bytes)}  {percent:6.2f}%"
    )


def upload_basic_file(
    *,
    file_path: Path,
    upload_state: dict[str, Any],
    state: dict[str, Any],
    state_file: Path,
    request_retries: int,
    endpoint: str,
) -> None:
    upload_url = str(upload_state.get("upload_url") or "").strip()
    if not upload_url:
        raise RuntimeError("Basic upload session is missing the upload URL.")

    total_size = int((state.get("file") or {}).get("size") or 0)
    print(f"Uploading single-part blob ({human_bytes(total_size)}) ...")
    with file_path.open("rb") as handle:
        response = request_with_retry(
            "PUT",
            upload_url,
            description="uploading single-part blob",
            request_retries=request_retries,
            timeout=PART_UPLOAD_TIMEOUT,
            data=handle,
        )
    response.raise_for_status()

    upload_state["upload_completed"] = True
    update_state_timestamp(state)
    save_state(state_file, state)
    print("[upload] single-part upload complete")


def complete_multipart_upload(
    *,
    upload_state: dict[str, Any],
    sha256_hex: str,
    request_retries: int,
) -> None:
    upload_url = str(upload_state.get("upload_url") or "").strip()
    if not upload_url:
        raise RuntimeError("Multipart upload session is missing the completion URL.")

    chunk_size = int(upload_state.get("chunk_size") or 0)
    if chunk_size <= 0:
        raise RuntimeError("Multipart upload session is missing chunk_size.")

    completed_parts = upload_state.get("completed_parts") or {}
    payload = {
        "oid": sha256_hex,
        "parts": [
            {
                "partNumber": int(part_no),
                "etag": str((completed_parts.get(str(part_no)) or {}).get("etag") or ""),
            }
            for part_no, _ in sorted((int(k), v) for k, v in completed_parts.items())
        ],
    }
    if not payload["parts"]:
        raise RuntimeError("Cannot finalize multipart upload without completed parts.")

    response = request_with_retry(
        "POST",
        upload_url,
        description="finalizing multipart upload",
        request_retries=request_retries,
        timeout=COMPLETION_REQUEST_TIMEOUT,
        json=payload,
        headers=LFS_HEADERS,
    )
    hf_raise_for_status(response)


def upload_multipart_file(
    *,
    file_path: Path,
    upload_state: dict[str, Any],
    state: dict[str, Any],
    state_file: Path,
    workers: int,
    request_retries: int,
    endpoint: str,
) -> None:
    total_size = int((state.get("file") or {}).get("size") or 0)
    chunk_size = int(upload_state.get("chunk_size") or 0)
    if chunk_size <= 0:
        raise RuntimeError("Multipart upload session is missing chunk_size.")

    sorted_part_urls = get_sorted_part_urls(upload_state, total_size, chunk_size)
    completed_parts = upload_state.setdefault("completed_parts", {})
    completed_bytes = sum(int((info or {}).get("size") or 0) for info in completed_parts.values())
    total_parts = len(sorted_part_urls)

    if completed_parts:
        print_upload_progress(
            completed_parts=len(completed_parts),
            total_parts=total_parts,
            uploaded_bytes=completed_bytes,
            total_bytes=total_size,
        )

    missing_parts = [part_no for part_no, _ in sorted_part_urls if str(part_no) not in completed_parts]
    if not missing_parts:
        print("All multipart chunks are already uploaded in local state.")
    else:
        print(
            f"Uploading {len(missing_parts)} missing parts with {workers} workers "
            f"(chunk size {human_bytes(chunk_size)}) ..."
        )

    lock = threading.Lock()

    def upload_one_part(part_no: int, part_url: str) -> None:
        nonlocal completed_bytes

        offset = (part_no - 1) * chunk_size
        part_size = min(chunk_size, total_size - offset)
        with PartFileObj(file_path, offset, part_size) as handle:
            response = request_with_retry(
                "PUT",
                part_url,
                description=f"uploading part {part_no}",
                request_retries=request_retries,
                timeout=PART_UPLOAD_TIMEOUT,
                data=handle,
            )
            try:
                response.raise_for_status()
            except Exception as exc:
                if is_session_expired_exception(exc):
                    raise UploadSessionExpiredError(str(exc)) from exc
                raise

        etag = str(response.headers.get("etag") or response.headers.get("ETag") or "").strip()
        if not etag:
            raise RuntimeError(f"Missing ETag after uploading part {part_no}.")

        with lock:
            completed_parts[str(part_no)] = {
                "etag": etag,
                "size": part_size,
                "completed_at": utc_now(),
            }
            completed_bytes += part_size
            update_state_timestamp(state)
            save_state(state_file, state)
            print_upload_progress(
                completed_parts=len(completed_parts),
                total_parts=total_parts,
                uploaded_bytes=completed_bytes,
                total_bytes=total_size,
            )

    if missing_parts:
        future_to_part: dict[Any, int] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for part_no, part_url in sorted_part_urls:
                if part_no in missing_parts:
                    future = executor.submit(upload_one_part, part_no, part_url)
                    future_to_part[future] = part_no

            for future in as_completed(future_to_part):
                try:
                    future.result()
                except UploadSessionExpiredError:
                    raise
                except Exception as exc:
                    part_no = future_to_part.get(future)
                    if part_no is not None:
                        raise RuntimeError(f"Failed while uploading part {part_no}: {exc}") from exc
                    raise

    try:
        complete_multipart_upload(
            upload_state=upload_state,
            sha256_hex=str((state.get("file") or {}).get("sha256") or ""),
            request_retries=request_retries,
        )
    except Exception as exc:
        if is_session_expired_exception(exc):
            raise UploadSessionExpiredError(str(exc)) from exc
        raise
    upload_state["upload_completed"] = True
    update_state_timestamp(state)
    save_state(state_file, state)
    print("[upload] multipart completion request accepted")


def verify_uploaded_blob(
    *,
    token: str,
    state: dict[str, Any],
    upload_state: dict[str, Any],
    request_retries: int,
) -> None:
    verify_url = str(upload_state.get("verify_url") or "").strip()
    if not verify_url or upload_state.get("verified"):
        return

    file_meta = state.get("file") or {}
    payload = {
        "oid": str(file_meta.get("sha256") or ""),
        "size": int(file_meta.get("size") or 0),
    }
    response = request_with_retry(
        "POST",
        verify_url,
        description="verifying uploaded blob",
        request_retries=request_retries,
        timeout=COMPLETION_REQUEST_TIMEOUT,
        headers=build_hf_headers(token=token),
        json=payload,
    )
    hf_raise_for_status(response)
    upload_state["verified"] = True
    print("[upload] LFS verify request succeeded")


def remote_file_matches(
    *,
    api: HfApi,
    token: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
    sha256_hex: str,
    size: int,
    request_retries: int,
) -> bool:
    try:
        infos = run_with_network_retry(
            "checking remote file state",
            lambda: api.get_paths_info(
                repo_id=repo_id,
                paths=[path_in_repo],
                revision=revision,
                repo_type=repo_type,
                token=token,
            ),
            retries=request_retries,
            endpoint=api.endpoint,
        )
    except Exception:
        return False

    if not infos:
        return False

    info = infos[0]
    remote_size = int(getattr(info, "size", -1) or -1)
    lfs = getattr(info, "lfs", None)
    remote_oid = None
    if isinstance(lfs, dict):
        remote_oid = lfs.get("oid")
    else:
        remote_oid = getattr(lfs, "oid", None)
    return str(remote_oid or "") == sha256_hex and remote_size == size


def commit_uploaded_file(
    *,
    api: HfApi,
    token: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
    sha256_hex: str,
    size: int,
    commit_message: str,
    commit_description: str,
    request_retries: int,
) -> dict[str, Any]:
    revision_quoted = quote(revision, safe="")
    commit_url = f"{api.endpoint}/api/{repo_type}s/{repo_id}/commit/{revision_quoted}"

    payload_items = [
        {
            "key": "header",
            "value": {
                "summary": commit_message,
                "description": commit_description or "",
            },
        },
        {
            "key": "lfsFile",
            "value": {
                "path": path_in_repo,
                "algo": "sha256",
                "oid": sha256_hex,
                "size": size,
            },
        },
    ]
    payload = b"".join(json.dumps(item).encode("utf-8") + b"\n" for item in payload_items)
    headers = {
        "Content-Type": "application/x-ndjson",
        **build_hf_headers(token=token),
    }
    response = request_with_retry(
        "POST",
        commit_url,
        description="creating Hub commit",
        request_retries=request_retries,
        timeout=COMPLETION_REQUEST_TIMEOUT,
        headers=headers,
        content=payload,
    )
    hf_raise_for_status(response, endpoint_name="commit")
    return response.json()


def ensure_repo_exists(
    *,
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool,
    request_retries: int,
) -> None:
    run_with_network_retry(
        "creating or checking target repo",
        lambda: api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private or None, exist_ok=True),
        retries=request_retries,
        endpoint=api.endpoint,
    )


def run_upload(
    *,
    api: HfApi,
    token: str,
    file_path: Path,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
    workers: int,
    private: bool,
    verify: bool,
    state_file: Path,
    commit_message: str,
    commit_description: str,
    fresh_session: bool,
    auto_restart_expired: bool,
    ensure_repo: bool,
    request_retries: int,
) -> int:
    state = ensure_state_matches(
        load_state(state_file),
        file_path=file_path,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
    )

    upload_info = build_upload_info(file_path, state)
    save_state(state_file, state)

    identity = state.get("file") or {}
    sha256_hex = str(identity.get("sha256") or "")
    size = int(identity.get("size") or 0)

    if ensure_repo:
        ensure_repo_exists(
            api=api,
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            request_retries=request_retries,
        )

    upload_state = state.setdefault("upload", {})
    if upload_state.get("committed") and remote_file_matches(
        api=api,
        token=token,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
        sha256_hex=sha256_hex,
        size=size,
        request_retries=request_retries,
    ):
        print("Remote file already matches local state. Nothing to do.")
        return 0

    if upload_state.get("upload_completed"):
        print("Local state indicates the blob upload already completed. Skipping straight to verify/commit.")
    else:
        has_cached_session = (
            not fresh_session and str(upload_state.get("transfer") or "") in {"multipart", "basic"}
        )
        if not has_cached_session:
            try:
                state["upload"] = request_new_upload_session(
                    api=api,
                    token=token,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    upload_info=upload_info,
                    request_retries=request_retries,
                )
            except RepositoryNotFoundError as exc:
                if ensure_repo:
                    raise RuntimeError(
                        f"Target repo {repo_type}/{repo_id} was not found even after `--ensure-repo`: {exc}"
                    ) from exc
                raise RuntimeError(
                    f"Target repo {repo_type}/{repo_id} was not found. "
                    "If it should be auto-created, rerun with `--ensure-repo`."
                ) from exc
            upload_state = state["upload"]
            update_state_timestamp(state)
            save_state(state_file, state)

    attempt = 0
    while not upload_state.get("upload_completed"):
        attempt += 1
        transfer = str(upload_state.get("transfer") or "")
        try:
            if transfer == "multipart":
                upload_multipart_file(
                    file_path=file_path,
                    upload_state=upload_state,
                    state=state,
                    state_file=state_file,
                    workers=workers,
                    request_retries=request_retries,
                    endpoint=api.endpoint,
                )
            elif transfer == "basic":
                upload_basic_file(
                    file_path=file_path,
                    upload_state=upload_state,
                    state=state,
                    state_file=state_file,
                    request_retries=request_retries,
                    endpoint=api.endpoint,
                )
            elif transfer in {"already_present", "xet"}:
                upload_state["upload_completed"] = True
                update_state_timestamp(state)
                save_state(state_file, state)
            else:
                raise RuntimeError(f"Unsupported upload transfer mode: {transfer or '<empty>'}")
        except RepositoryNotFoundError as exc:
            if ensure_repo:
                raise RuntimeError(
                    f"Target repo {repo_type}/{repo_id} was not found even after `--ensure-repo`: {exc}"
                ) from exc
            raise RuntimeError(
                f"Target repo {repo_type}/{repo_id} was not found. "
                "If it should be auto-created, rerun with `--ensure-repo`."
            ) from exc
        except UploadSessionExpiredError:
            if auto_restart_expired:
                print("Multipart session expired on the server. Clearing cached part state and requesting a fresh session.")
                try:
                    state["upload"] = request_new_upload_session(
                        api=api,
                        token=token,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        revision=revision,
                        upload_info=upload_info,
                        request_retries=request_retries,
                    )
                except RepositoryNotFoundError as exc:
                    if ensure_repo:
                        raise RuntimeError(
                            f"Target repo {repo_type}/{repo_id} was not found even after `--ensure-repo`: {exc}"
                        ) from exc
                    raise RuntimeError(
                        f"Target repo {repo_type}/{repo_id} was not found. "
                        "If it should be auto-created, rerun with `--ensure-repo`."
                    ) from exc
                upload_state = state["upload"]
                update_state_timestamp(state)
                save_state(state_file, state)
                continue
            raise

    if verify:
        verify_uploaded_blob(
            token=token,
            state=state,
            upload_state=upload_state,
            request_retries=request_retries,
        )
        update_state_timestamp(state)
        save_state(state_file, state)

    if remote_file_matches(
        api=api,
        token=token,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
        sha256_hex=sha256_hex,
        size=size,
        request_retries=request_retries,
    ):
        print("Remote file already points at the same blob. Skipping commit.")
        upload_state["committed"] = True
        update_state_timestamp(state)
        save_state(state_file, state)
        return 0

    commit_info = commit_uploaded_file(
        api=api,
        token=token,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
        sha256_hex=sha256_hex,
        size=size,
        commit_message=commit_message,
        commit_description=commit_description,
        request_retries=request_retries,
    )
    upload_state["committed"] = True
    state["commit"] = commit_info
    update_state_timestamp(state)
    save_state(state_file, state)

    commit_url = str(commit_info.get("commitUrl") or commit_info.get("commit_url") or "").strip()
    if commit_url:
        print(f"Commit created: {commit_url}")
    else:
        print("Commit created.")
    return 0


def upload_file_to_hf(
    *,
    file_path: Path,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    revision: str = DEFAULT_REVISION,
    path_in_repo: str = "",
    endpoint: str = DEFAULT_ENDPOINT,
    workers: int = DEFAULT_WORKERS,
    private: bool = False,
    verify: bool = True,
    state_dir: Path | None = None,
    commit_message: str = "",
    commit_description: str = "",
    fresh_session: bool = False,
    auto_restart_expired: bool = True,
    ensure_repo: bool = False,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
) -> dict[str, str]:
    original_file = Path(file_path).expanduser().resolve()
    if not original_file.is_file():
        raise RuntimeError(f"File not found: {original_file}")

    normalized_path = normalize_repo_path(path_in_repo or original_file.name)
    resolved_state_dir = (
        Path(state_dir).expanduser().resolve()
        if state_dir is not None
        else default_state_dir(
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=normalized_path,
        )
    )
    state_file = resolved_state_dir / "state.json"
    lock_file = resolved_state_dir / "upload.lock"
    resolved_file = prepare_upload_source_file(original_file, resolved_state_dir)

    token = get_token()
    api = HfApi(
        endpoint=str(endpoint).strip(),
        token=token,
        library_name="mts-hf-upload",
        library_version="1.1",
    )
    get_session().timeout = API_REQUEST_TIMEOUT

    summary = str(commit_message or "").strip() or f"Upload {original_file.name}"
    description = str(commit_description or "").strip()

    print(f"Local file   : {original_file}")
    print(f"Upload file  : {resolved_file}")
    print(f"Repo target  : {repo_type}/{repo_id}@{revision}:{normalized_path}")
    print(f"Hub endpoint : {api.endpoint}")
    print(f"State file   : {state_file}")
    print(f"Lock file    : {lock_file}")

    with acquire_process_lock(lock_file):
        run_upload(
            api=api,
            token=token,
            file_path=resolved_file,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            path_in_repo=normalized_path,
            workers=workers,
            private=private,
            verify=verify,
            state_file=state_file,
            commit_message=summary,
            commit_description=description,
            fresh_session=fresh_session,
            auto_restart_expired=auto_restart_expired,
            ensure_repo=ensure_repo,
            request_retries=request_retries,
        )

    public_url = build_public_file_url(
        endpoint=api.endpoint,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=normalized_path,
    )
    return {
        "url": public_url,
        "endpoint": str(api.endpoint),
        "repo_id": str(repo_id),
        "repo_type": str(repo_type),
        "revision": str(revision),
        "path_in_repo": normalized_path,
        "state_file": str(state_file),
        "lock_file": str(lock_file),
    }


def main() -> int:
    args = parse_args()
    file_path = Path(args.file).expanduser().resolve()
    if not file_path.is_file():
        raise SystemExit(f"File not found: {file_path}")

    if args.workers <= 0:
        raise SystemExit("--workers must be >= 1")
    if args.request_retries < -1:
        raise SystemExit("--request-retries must be >= -1")

    path_in_repo = normalize_repo_path(args.path_in_repo)
    state_dir = resolve_state_dir(args, path_in_repo)
    state_file = state_dir / "state.json"
    lock_file = state_dir / "upload.lock"

    token = get_token()
    api = HfApi(
        endpoint=str(args.endpoint).strip(),
        token=token,
        library_name="mts-hf-upload",
        library_version="1.1",
    )
    get_session().timeout = API_REQUEST_TIMEOUT
    staged_file = prepare_upload_source_file(file_path, state_dir)

    print(f"Local file   : {file_path}")
    print(f"Upload file  : {staged_file}")
    print(f"Repo target  : {args.repo_type}/{args.repo_id}@{args.revision}:{path_in_repo}")
    print(f"Hub endpoint : {api.endpoint}")
    print(f"State file   : {state_file}")
    print(f"Lock file    : {lock_file}")

    with acquire_process_lock(lock_file):
        return run_upload(
            api=api,
            token=token,
            file_path=staged_file,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            path_in_repo=path_in_repo,
            workers=args.workers,
            private=args.private,
            verify=not args.no_verify,
            state_file=state_file,
            commit_message=args.commit_message,
            commit_description=args.commit_description,
            fresh_session=args.fresh_session,
            auto_restart_expired=not args.no_auto_restart_expired,
            ensure_repo=args.ensure_repo,
            request_retries=args.request_retries,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NetworkRetryError as exc:
        raise SystemExit(str(exc))
    except KeyboardInterrupt:
        raise SystemExit(130)
