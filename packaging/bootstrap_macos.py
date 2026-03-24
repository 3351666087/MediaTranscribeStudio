from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = ROOT / ".venv-macos"
DEFAULT_MAMBA_ROOT = ROOT / ".micromamba-root"
BREW_PACKAGES = [
    "python@3.11",
    "cmake",
    "ffmpeg",
    "zstd",
]
OPTIONAL_BREW_PACKAGES = {"create-dmg"}
MICROMAMBA_PACKAGES = [
    "python=3.11",
    "cmake",
    "ffmpeg",
    "zstd",
]
SKIP_PIP_PACKAGES = {
    "cuda-bindings",
    "cuda-pathfinder",
    "numba-cuda",
    "nv-one-logger-core",
    "nv-one-logger-pytorch-lightning-integration",
    "nv-one-logger-training-telemetry",
    "pefile",
    "pywin32-ctypes",
    "wkhtmltopdf",
}
REWRITE_PIP_PACKAGES = {
    "torch": "torch==2.10.0",
    "torchaudio": "torchaudio==2.10.0",
    "torchvision": "torchvision==0.25.0",
}
EXTRA_PIP_PACKAGES = [
    "pip>=25",
    "setuptools>=75,<81",
    "wheel>=0.45",
    "tiktoken>=0.12",
    "addict>=2.4",
    "dmgbuild>=1.6",
]


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path = ROOT) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def brew_executable() -> str | None:
    for candidate in (
        shutil.which("brew"),
        "/opt/homebrew/bin/brew",
        "/usr/local/bin/brew",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def ensure_homebrew() -> str:
    brew = brew_executable()
    if brew:
        return brew

    install_script = subprocess.run(
        [
            "curl",
            "-fsSL",
            "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    ).stdout
    subprocess.run(
        ["/bin/bash", "-s"],
        input=install_script,
        text=True,
        check=True,
        cwd=str(ROOT),
    )
    brew = brew_executable()
    if not brew:
        raise RuntimeError("Homebrew installation finished but `brew` was still not found.")
    return brew


def micromamba_subdir() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "osx-arm64"
    return "osx-64"


def ensure_micromamba(root_prefix: Path) -> Path:
    bin_dir = root_prefix / "bin"
    micromamba = bin_dir / "micromamba"
    if micromamba.exists():
        return micromamba

    bin_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://micro.mamba.pm/api/micromamba/{micromamba_subdir()}/latest"
    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tmp_file:
        archive_path = Path(tmp_file.name)
    try:
        with urllib.request.urlopen(url) as response, archive_path.open("wb") as handle:
            handle.write(response.read())
        with tarfile.open(archive_path, mode="r:bz2") as archive:
            member = next(
                item
                for item in archive.getmembers()
                if item.name.endswith("/bin/micromamba") or item.name == "bin/micromamba"
            )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError("Failed to extract micromamba from archive.")
            micromamba.write_bytes(extracted.read())
        micromamba.chmod(0o755)
    finally:
        try:
            archive_path.unlink()
        except Exception:
            pass

    return micromamba


def ensure_micromamba_env(micromamba: Path, *, root_prefix: Path, env_prefix: Path) -> Path:
    python_path = env_prefix / "bin" / "python"
    if python_path.exists():
        return python_path

    run(
        [
            str(micromamba),
            "create",
            "-y",
            "-r",
            str(root_prefix),
            "-p",
            str(env_prefix),
            *MICROMAMBA_PACKAGES,
        ]
    )
    if not python_path.exists():
        raise RuntimeError(f"micromamba created the environment but python was not found: {python_path}")
    return python_path


def ensure_brew_packages(brew: str) -> None:
    missing: list[str] = []
    for package in BREW_PACKAGES:
        proc = subprocess.run(
            [brew, "list", package],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            missing.append(package)

    failures: list[str] = []
    for package in missing:
        try:
            run([brew, "install", package])
        except subprocess.CalledProcessError:
            if package in OPTIONAL_BREW_PACKAGES:
                print(f"\n[warn] Optional Homebrew package failed to install: {package}")
                failures.append(package)
                continue
            raise

    if failures:
        print(f"\nOptional Homebrew packages unavailable: {', '.join(failures)}")


def requirement_name(raw: str) -> str:
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)", raw)
    return (match.group(1) if match else raw).strip().lower().replace("_", "-")


def generate_requirements(output_path: Path) -> list[str]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    result: list[str] = []
    seen: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = requirement_name(stripped)
        if name in SKIP_PIP_PACKAGES:
            continue
        requirement = REWRITE_PIP_PACKAGES.get(name, stripped)
        normalized_name = requirement_name(requirement)
        if normalized_name in seen:
            continue
        seen.add(normalized_name)
        result.append(requirement)

    for requirement in EXTRA_PIP_PACKAGES:
        name = requirement_name(requirement)
        if name in seen:
            continue
        seen.add(name)
        result.insert(0, requirement)

    output_path.write_text("\n".join(result) + "\n", encoding="utf-8")
    return result


def preferred_python() -> str:
    for candidate in (
        shutil.which("python3.11"),
        "/opt/homebrew/bin/python3.11",
        "/usr/local/bin/python3.11",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("Python 3.11 was not found after Homebrew installation.")


def ensure_venv(venv_dir: Path, python_exe: str) -> Path:
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        run([python_exe, "-m", "venv", str(venv_dir)])
    return venv_python


def install_pip_packages(venv_python: Path, requirements_path: Path) -> None:
    run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip>=25",
            "setuptools>=75,<81",
            "wheel>=0.45",
        ]
    )
    batch_cmd = [str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)]
    batch = subprocess.run(batch_cmd, cwd=str(ROOT))
    if batch.returncode == 0:
        return

    print("\n[warn] Batch pip install failed; retrying package-by-package to maximize macOS compatibility.")
    failures: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        proc = subprocess.run([str(venv_python), "-m", "pip", "install", requirement], cwd=str(ROOT))
        if proc.returncode != 0:
            failures.append(requirement)

    if failures:
        print("\n[warn] The following pip packages could not be installed on this Mac:")
        for requirement in failures:
            print(f"  - {requirement}")


def install_playwright_chromium(venv_python: Path) -> None:
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    env.setdefault("PLAYWRIGHT_SKIP_BROWSER_GC", "1")
    run([str(venv_python), "-m", "playwright", "install", "chromium"], env=env)


def verify_mps_runtime(venv_python: Path) -> None:
    verify_script = ROOT / "tools" / "verify_mps_runtime.py"
    if not verify_script.exists():
        raise RuntimeError(f"MPS verification script is missing: {verify_script}")
    run([str(venv_python), str(verify_script)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap MediaTranscribeStudio for macOS.")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--mamba-root", type=Path, default=DEFAULT_MAMBA_ROOT)
    parser.add_argument(
        "--requirements-out",
        type=Path,
        default=ROOT / "requirements-macos.txt",
    )
    parser.add_argument("--skip-brew", action="store_true")
    parser.add_argument("--skip-pip", action="store_true")
    parser.add_argument("--skip-playwright", action="store_true")
    parser.add_argument("--skip-mps-verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    if sys.platform != "darwin":
        raise SystemExit("bootstrap_macos.py must be run on macOS.")

    args = parse_args()

    python_exe: str | None = None
    if not args.skip_brew:
        brew = brew_executable()
        if brew:
            ensure_brew_packages(brew)
            python_exe = preferred_python()
        else:
            print("\n[info] Homebrew is not installed or requires admin privileges; using micromamba fallback.")

    if python_exe is None:
        micromamba = ensure_micromamba(args.mamba_root)
        python_exe = str(
            ensure_micromamba_env(
                micromamba,
                root_prefix=args.mamba_root,
                env_prefix=args.venv,
            )
        )

    generated = generate_requirements(args.requirements_out)
    print(f"\nGenerated {args.requirements_out} with {len(generated)} pip requirement(s).")
    if (args.venv / "conda-meta").exists():
        venv_python = args.venv / "bin" / "python"
    else:
        venv_python = ensure_venv(args.venv, python_exe)

    if not args.skip_pip:
        install_pip_packages(venv_python, args.requirements_out)

    if not args.skip_playwright:
        install_playwright_chromium(venv_python)

    if not args.skip_mps_verify:
        verify_mps_runtime(venv_python)

    print("\n== macOS bootstrap complete ==")
    print(f"Virtualenv : {args.venv}")
    print(f"Python     : {venv_python}")
    print(f"Requirements: {args.requirements_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
