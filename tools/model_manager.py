"""Cross-platform model download manager for MediaTranscribeStudio.

The manager deliberately delegates model transfer to the vendors' supported
CLIs.  It never accepts access tokens on the command line and records only the
environment-variable name used for authentication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "local-model-registry.json"
)
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,239}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,159}$")
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")


class ModelManagerError(ValueError):
    """Raised when a model download request is unsafe or cannot run."""


@dataclass(frozen=True)
class DownloadRequest:
    provider: str
    model: str
    destination: Path
    revision: str | None = None
    endpoint: str | None = None
    token_env: str | None = None
    executable: str | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip().casefold()
        model = self.model.strip()
        destination = self.destination.expanduser().resolve(strict=False)
        if provider not in {"ollama", "huggingface"}:
            raise ModelManagerError("provider must be ollama or huggingface")
        if not _MODEL_NAME_RE.fullmatch(model) or ".." in model.split("/"):
            raise ModelManagerError("model contains unsupported characters")
        revision = self.revision
        if revision is not None:
            revision = revision.strip()
            if not _REVISION_RE.fullmatch(revision) or ".." in revision.split("/"):
                raise ModelManagerError("revision contains unsupported characters")
        if provider == "huggingface" and revision is None:
            raise ModelManagerError(
                "Hugging Face downloads require an explicit pinned revision"
            )
        token_env = self.token_env
        if token_env is not None:
            token_env = token_env.strip()
            if not _ENV_NAME_RE.fullmatch(token_env):
                raise ModelManagerError("token_env must be an environment variable name")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "token_env", token_env)


def default_model_root() -> Path:
    """Prefer the data drive requested for this product, then user storage."""

    override = os.environ.get("MTS_MODEL_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve(strict=False)
    if os.name == "nt" and Path("D:/").exists():
        return Path("D:/models")
    wsl_drive = Path("/mnt/d")
    if wsl_drive.is_dir():
        return wsl_drive / "models"
    return Path.home() / ".local" / "share" / "MediaTranscribeStudio" / "models"


def _safe_destination(root: Path, provider: str, model: str) -> Path:
    root = root.expanduser().resolve(strict=False)
    parts = [part for part in model.replace(":", "--").split("/") if part]
    candidate = root.joinpath(provider, *parts).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ModelManagerError("model destination escapes the configured root") from exc
    return candidate


def build_download_command(request: DownloadRequest) -> tuple[list[str], dict[str, str]]:
    """Return a shell-free command and non-secret environment overlay."""

    if request.provider == "ollama":
        executable = request.executable or "ollama"
        command = [executable, "pull", request.model]
        environment = {"OLLAMA_MODELS": str(request.destination)}
        if request.endpoint:
            environment["OLLAMA_HOST"] = request.endpoint
        return command, environment

    executable = request.executable or "hf"
    command = [
        executable,
        "download",
        request.model,
        "--revision",
        str(request.revision),
        "--local-dir",
        str(request.destination),
    ]
    # hf reads HF_TOKEN itself. If a deployment uses a differently named
    # secret, copy it only into the child process environment.
    environment: dict[str, str] = {}
    if request.token_env and request.token_env != "HF_TOKEN":
        secret = os.environ.get(request.token_env)
        if secret:
            environment["HF_TOKEN"] = secret
    return command, environment


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _command_identity(command: Sequence[str]) -> str:
    payload = json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_download(
    request: DownloadRequest,
    *,
    dry_run: bool = False,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    command, overlay = build_download_command(request)
    executable = command[0]
    resolved = shutil.which(executable) if not Path(executable).is_file() else executable
    if not dry_run and resolved is None:
        raise ModelManagerError(f"required executable is unavailable: {executable}")

    receipt: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "model-download-receipt",
        "provider": request.provider,
        "model": request.model,
        "revision": request.revision,
        "destination": str(request.destination),
        "tokenEnvironmentVariable": request.token_env,
        "commandSha256": _command_identity(command),
        "status": "planned" if dry_run else "running",
        "generatedAt": _utc_now(),
    }
    if dry_run:
        return receipt

    request.destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(overlay)
    completed = runner(
        command,
        check=False,
        env=environment,
        stdin=subprocess.DEVNULL,
    )
    return_code = int(getattr(completed, "returncode", 1))
    receipt["status"] = "completed" if return_code == 0 else "failed"
    receipt["returnCode"] = return_code
    if return_code != 0:
        raise ModelManagerError(
            f"{request.provider} download failed with exit code {return_code}"
        )
    return receipt


def registry_models(path: Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelManagerError(f"cannot read model registry: {path}") from exc
    raw_models = document.get("models") if isinstance(document, Mapping) else None
    if not isinstance(raw_models, list):
        raise ModelManagerError("model registry has no models array")
    output: list[dict[str, Any]] = []
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            continue
        source = raw.get("source")
        usage = raw.get("usage")
        if not isinstance(source, Mapping) or not isinstance(usage, Mapping):
            continue
        output.append(
            {
                "id": raw.get("id"),
                "displayName": raw.get("displayName"),
                "provider": source.get("provider"),
                "repository": source.get("repository"),
                "revision": source.get("revision") or source.get("tag"),
                "status": usage.get("status"),
                "roles": usage.get("roles"),
                "localPath": (
                    raw.get("local", {}).get("path")
                    if isinstance(raw.get("local"), Mapping)
                    else None
                ),
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    list_parser = subcommands.add_parser("list", help="List registered models")
    list_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)

    pull = subcommands.add_parser("pull", help="Download a model")
    pull.add_argument("--provider", choices=("ollama", "huggingface"), required=True)
    pull.add_argument("--model", required=True)
    pull.add_argument("--revision")
    pull.add_argument("--model-root", type=Path, default=default_model_root())
    pull.add_argument("--destination", type=Path)
    pull.add_argument("--endpoint")
    pull.add_argument("--token-env", default="HF_TOKEN")
    pull.add_argument("--executable")
    pull.add_argument("--receipt", type=Path)
    pull.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            payload: Any = {
                "schemaVersion": SCHEMA_VERSION,
                "models": registry_models(args.registry),
            }
        else:
            destination = args.destination or _safe_destination(
                args.model_root,
                args.provider,
                args.model,
            )
            payload = run_download(
                DownloadRequest(
                    provider=args.provider,
                    model=args.model,
                    destination=destination,
                    revision=args.revision,
                    endpoint=args.endpoint,
                    token_env=args.token_env,
                    executable=args.executable,
                ),
                dry_run=args.dry_run,
            )
            if args.receipt:
                args.receipt.parent.mkdir(parents=True, exist_ok=True)
                args.receipt.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except ModelManagerError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
