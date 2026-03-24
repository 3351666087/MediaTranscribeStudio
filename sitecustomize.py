from __future__ import annotations

import ctypes
import importlib
import importlib.abc
import importlib.machinery
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import warnings
from dataclasses import dataclass
from pathlib import Path


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_version_text(raw: str) -> tuple[int, int, int]:
    parts = [part for part in str(raw or "").strip().split(".") if part.strip()]
    values: list[int] = []
    for part in parts[:3]:
        try:
            values.append(int(part))
        except Exception:
            values.append(0)
    while len(values) < 3:
        values.append(0)
    return tuple(values[:3])  # type: ignore[return-value]


def _actual_macos_version() -> tuple[int, int, int]:
    if sys.platform != "darwin":
        return (0, 0, 0)

    try:
        import platform

        raw = str(platform.mac_ver()[0] or "").strip()
    except Exception:
        raw = ""
    if raw:
        return _parse_version_text(raw)

    try:
        proc = subprocess.run(
            ["sw_vers", "-productVersion"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return _parse_version_text(proc.stdout)
    except Exception:
        return (0, 0, 0)


def _version_at_least(
    current: tuple[int, int, int],
    required: tuple[int, int, int],
) -> bool:
    return tuple(int(x) for x in current) >= tuple(int(x) for x in required)


def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _runtime_shim_candidates() -> list[Path]:
    root = _project_root()
    exe_dir = Path(sys.executable).resolve().parent
    candidates = [
        root / "build" / "native" / "install" / "lib" / "libmts_apple_runtime_shim.dylib",
        root / "build" / "native" / "install" / "bin" / "libmts_apple_runtime_shim.dylib",
        root / "native" / "install" / "lib" / "libmts_apple_runtime_shim.dylib",
        root / "native" / "install" / "bin" / "libmts_apple_runtime_shim.dylib",
        exe_dir / "_internal" / "native" / "lib" / "libmts_apple_runtime_shim.dylib",
        exe_dir / "_internal" / "native" / "bin" / "libmts_apple_runtime_shim.dylib",
        exe_dir / "_internal" / "native" / "libmts_apple_runtime_shim.dylib",
        exe_dir / "native" / "lib" / "libmts_apple_runtime_shim.dylib",
        exe_dir / "native" / "bin" / "libmts_apple_runtime_shim.dylib",
        exe_dir / "native" / "libmts_apple_runtime_shim.dylib",
    ]
    seen: set[str] = set()
    out: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


_APPLE_RUNTIME_SHIM: ctypes.CDLL | None = None
_APPLE_RUNTIME_BRIDGE_ACTIVE = False


def _load_apple_runtime_shim() -> ctypes.CDLL | None:
    global _APPLE_RUNTIME_SHIM
    if _APPLE_RUNTIME_SHIM is not None:
        return _APPLE_RUNTIME_SHIM

    for candidate in _runtime_shim_candidates():
        try:
            if not candidate.exists():
                continue
            lib = ctypes.CDLL(str(candidate))
            _APPLE_RUNTIME_SHIM = lib
            os.environ.setdefault("MTS_APPLE_RUNTIME_SHIM_PATH", str(candidate))
            return lib
        except Exception:
            continue
    return None


def _install_future_macos_torch_bridge() -> None:
    global _APPLE_RUNTIME_BRIDGE_ACTIVE
    if sys.platform != "darwin":
        return
    if not _env_flag("MTS_TORCH_MPS_BRIDGE", True):
        return
    if _APPLE_RUNTIME_BRIDGE_ACTIVE:
        return

    actual = _actual_macos_version()
    target = _parse_version_text(
        os.environ.get("MTS_TORCH_MPS_BRIDGE_TARGET", "15.0.0")
    )
    future_only = _env_flag("MTS_TORCH_MPS_BRIDGE_FUTURE_ONLY", True)

    if future_only and not (actual > target):
        return

    lib = _load_apple_runtime_shim()
    if lib is None:
        return

    try:
        lib.mts_apple_compat_bridge_install.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.mts_apple_compat_bridge_install.restype = ctypes.c_int
    except Exception:
        return

    try:
        result = int(
            lib.mts_apple_compat_bridge_install(
                int(target[0]),
                int(target[1]),
                int(target[2]),
                int(1 if future_only else 0),
                int(1 if _env_flag("MTS_TORCH_MPS_BRIDGE_DEBUG", False) else 0),
            )
        )
    except Exception:
        return

    if result <= 0:
        return

    _APPLE_RUNTIME_BRIDGE_ACTIVE = True
    os.environ["MTS_TORCH_MPS_BRIDGE_ACTIVE"] = "1"
    os.environ["MTS_TORCH_MPS_BRIDGE_ACTUAL_VERSION"] = ".".join(str(x) for x in actual)
    os.environ["MTS_TORCH_MPS_BRIDGE_TARGET_EFFECTIVE"] = ".".join(
        str(x) for x in target
    )

    try:
        lib.mts_metal_device_available.argtypes = []
        lib.mts_metal_device_available.restype = ctypes.c_int
        os.environ["MTS_TORCH_MPS_BRIDGE_METAL_DEVICE"] = str(
            int(lib.mts_metal_device_available())
        )
    except Exception:
        pass
    try:
        lib.mts_mps_device_available.argtypes = []
        lib.mts_mps_device_available.restype = ctypes.c_int
        os.environ["MTS_TORCH_MPS_BRIDGE_MPS_DEVICE"] = str(
            int(lib.mts_mps_device_available())
        )
    except Exception:
        pass


def _configure_warning_filters() -> None:
    # torchmetrics and some PyInstaller runtime hooks still import pkg_resources
    # for legacy version checks. Keep startup clean until upstream removes it.
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The pkg_resources package is slated for removal as early as 2025-11-30\..*",
        category=UserWarning,
    )


class _PyAVStubOperationError(RuntimeError):
    pass


def _install_pyav_stub() -> None:
    if not _env_flag("MTS_DISABLE_REAL_PYAV", sys.platform == "darwin"):
        return
    if "av" in sys.modules:
        return

    av_mod = types.ModuleType("av")
    av_mod.__file__ = "<mts-av-stub>"
    av_mod.__package__ = "av"
    av_mod.__path__ = []  # type: ignore[attr-defined]
    av_mod.__version__ = "0+stub"
    av_mod.__mts_stub__ = True

    class _FFmpegError(_PyAVStubOperationError):
        pass

    class _AVError(_FFmpegError):
        pass

    class _InvalidDataError(_AVError):
        pass

    def _unsupported(*_args, **_kwargs):
        raise _PyAVStubOperationError(
            "Real PyAV is disabled in this app runtime to avoid loading a second "
            "FFmpeg stack. Audio decoding is patched to use ffmpeg directly."
        )

    class _StubLogging:
        ERROR = 0

        @staticmethod
        def set_level(_value):
            return None

    error_mod = types.ModuleType("av.error")
    error_mod.InvalidDataError = _InvalidDataError

    video_mod = types.ModuleType("av.video")
    video_frame_mod = types.ModuleType("av.video.frame")

    class VideoFrame:
        pict_type = "NONE"

    class PictureType:
        NONE = "NONE"

    video_frame_mod.VideoFrame = VideoFrame
    video_frame_mod.PictureType = PictureType
    video_mod.frame = video_frame_mod

    audio_mod = types.ModuleType("av.audio")
    audio_resampler_mod = types.ModuleType("av.audio.resampler")
    audio_fifo_mod = types.ModuleType("av.audio.fifo")

    class AudioResampler:
        def __init__(self, *_args, **_kwargs):
            _unsupported()

    class AudioFifo:
        def __init__(self, *_args, **_kwargs):
            self.samples = 0
            _unsupported()

    audio_resampler_mod.AudioResampler = AudioResampler
    audio_fifo_mod.AudioFifo = AudioFifo
    audio_mod.resampler = audio_resampler_mod
    audio_mod.fifo = audio_fifo_mod

    av_mod.logging = _StubLogging
    av_mod.open = _unsupported
    av_mod.FFmpegError = _FFmpegError
    av_mod.AVError = _AVError
    av_mod.error = error_mod
    av_mod.video = video_mod
    av_mod.audio = audio_mod

    sys.modules["av"] = av_mod
    sys.modules["av.error"] = error_mod
    sys.modules["av.video"] = video_mod
    sys.modules["av.video.frame"] = video_frame_mod
    sys.modules["av.audio"] = audio_mod
    sys.modules["av.audio.resampler"] = audio_resampler_mod
    sys.modules["av.audio.fifo"] = audio_fifo_mod
    os.environ.setdefault("MTS_PYAV_STUB_ACTIVE", "1")


def _find_ffmpeg_binary() -> str | None:
    env_candidates = [
        os.environ.get("FFMPEG_BINARY"),
        os.environ.get("IMAGEIO_FFMPEG_EXE"),
    ]
    for raw in env_candidates:
        text = str(raw or "").strip().strip('"')
        if not text:
            continue
        candidate = Path(text).expanduser()
        try:
            if candidate.exists() and candidate.is_file():
                return str(candidate.resolve())
        except Exception:
            pass

    root = _project_root()
    exe_dir = Path(sys.executable).resolve().parent
    local_candidates = [
        Path(sys.executable).resolve().parent / "ffmpeg",
        root / "build" / "dist" / "ffmpeg" / "ffmpeg",
        root / "build" / "dist" / "tools" / "ffmpeg" / "ffmpeg",
        root / "build" / "dist" / "_internal" / "tools" / "ffmpeg" / "ffmpeg",
        root / ".venv-macos" / "bin" / "ffmpeg",
        exe_dir / "_internal" / "tools" / "ffmpeg" / "ffmpeg",
        exe_dir / "_internal" / "ffmpeg" / "ffmpeg",
        exe_dir / "ffmpeg" / "ffmpeg",
    ]
    for candidate in local_candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return str(candidate.resolve())
        except Exception:
            continue

    resolved = shutil.which("ffmpeg")
    return str(resolved) if resolved else None


def _decode_audio_with_ffmpeg_binary(
    input_file,
    sampling_rate: int,
    split_stereo: bool,
):
    import numpy as np

    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found for patched faster-whisper audio decode")

    temp_path: Path | None = None
    cleanup_path: Path | None = None
    if isinstance(input_file, (str, bytes, os.PathLike)):
        temp_path = Path(input_file).expanduser()
    else:
        fd, raw_path = tempfile.mkstemp(prefix="mts-fw-audio-", suffix=".bin")
        os.close(fd)
        cleanup_path = Path(raw_path)
        reader = getattr(input_file, "read", None)
        if not callable(reader):
            raise TypeError(f"Unsupported audio input type for ffmpeg decode: {type(input_file)!r}")
        data = reader()
        if isinstance(data, str):
            data = data.encode("utf-8", "ignore")
        cleanup_path.write_bytes(bytes(data))
        temp_path = cleanup_path

    channels = 2 if split_stereo else 1
    cmd = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(temp_path),
        "-vn",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        str(channels),
        "-ar",
        str(int(sampling_rate)),
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
        )
    finally:
        if cleanup_path is not None:
            try:
                cleanup_path.unlink(missing_ok=True)
            except Exception:
                pass

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace")[:400]
        raise RuntimeError(f"ffmpeg decode failed: {stderr}")

    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    if audio.size <= 0:
        if split_stereo:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.zeros((0,), dtype=np.float32)

    audio /= 32768.0
    if split_stereo:
        return audio[0::2], audio[1::2]
    return audio


def _patch_faster_whisper_audio(module) -> None:
    if getattr(module, "__mts_decode_patched__", False):
        return

    original = getattr(module, "decode_audio", None)
    if not callable(original):
        return

    def _patched_decode_audio(input_file, sampling_rate=16000, split_stereo=False):
        return _decode_audio_with_ffmpeg_binary(
            input_file,
            sampling_rate=int(sampling_rate),
            split_stereo=bool(split_stereo),
        )

    _patched_decode_audio.__name__ = "decode_audio"
    module._mts_original_decode_audio = original
    module.decode_audio = _patched_decode_audio
    module.__mts_decode_patched__ = True


def _patch_faster_whisper_transcribe(module) -> None:
    try:
        audio_mod = importlib.import_module("faster_whisper.audio")
    except Exception:
        return
    patched = getattr(audio_mod, "decode_audio", None)
    if callable(patched):
        module.decode_audio = patched
        module.__mts_decode_patched__ = True


_MODULE_PATCHERS = {
    "faster_whisper.audio": _patch_faster_whisper_audio,
    "faster_whisper.transcribe": _patch_faster_whisper_transcribe,
}


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped_loader):
        self.fullname = fullname
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create = getattr(self.wrapped_loader, "create_module", None)
        if callable(create):
            return create(spec)
        return None

    def exec_module(self, module):
        exec_module = getattr(self.wrapped_loader, "exec_module", None)
        if callable(exec_module):
            exec_module(module)
        else:
            load_module = getattr(self.wrapped_loader, "load_module", None)
            if callable(load_module):
                load_module(self.fullname)
        patcher = _MODULE_PATCHERS.get(self.fullname)
        if callable(patcher):
            patcher(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        patcher = _MODULE_PATCHERS.get(fullname)
        if not callable(patcher):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if isinstance(spec.loader, _PatchLoader):
            return spec
        spec.loader = _PatchLoader(fullname, spec.loader)
        return spec


def _install_module_patch_finder() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _PatchFinder):
            break
    else:
        sys.meta_path.insert(0, _PatchFinder())

    for fullname, patcher in _MODULE_PATCHERS.items():
        module = sys.modules.get(fullname)
        if module is not None and callable(patcher):
            patcher(module)


def _patch_torchaudio_legacy_api() -> None:
    try:
        import torchaudio
    except Exception:
        return

    missing = [
        name
        for name in (
            "AudioMetaData",
            "list_audio_backends",
            "set_audio_backend",
            "get_audio_backend",
            "info",
        )
        if not hasattr(torchaudio, name)
    ]
    if not missing:
        return

    try:
        import soundfile as sf
    except Exception:
        sf = None

    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    current_backend = "soundfile" if sf is not None else None

    def list_audio_backends() -> list[str]:
        backends: list[str] = []
        if sf is not None:
            backends.append("soundfile")
        return backends

    def set_audio_backend(name: str | None) -> None:
        nonlocal current_backend
        if name is None:
            current_backend = None
            return
        if name not in list_audio_backends():
            raise ValueError(f"Unsupported torchaudio backend on this build: {name}")
        current_backend = name

    def get_audio_backend() -> str | None:
        return current_backend

    def _bits_per_sample(info_obj: object) -> int:
        subtype = str(getattr(info_obj, "subtype", "") or "")
        match = re.search(r"(\d+)", subtype)
        return int(match.group(1)) if match else 0

    def info(uri, format=None, backend=None) -> AudioMetaData:
        if sf is not None:
            try:
                meta = sf.info(uri)
                return AudioMetaData(
                    sample_rate=int(meta.samplerate),
                    num_frames=int(meta.frames),
                    num_channels=int(meta.channels),
                    bits_per_sample=_bits_per_sample(meta),
                    encoding=str(getattr(meta, "subtype", "") or getattr(meta, "format", "") or "UNKNOWN"),
                )
            except Exception:
                pass

        waveform, sample_rate = torchaudio.load(uri, format=format, backend=backend)
        num_channels = int(waveform.shape[0]) if getattr(waveform, "ndim", 0) >= 2 else 1
        num_frames = int(waveform.shape[-1]) if getattr(waveform, "ndim", 0) >= 1 else 0
        return AudioMetaData(
            sample_rate=int(sample_rate),
            num_frames=num_frames,
            num_channels=num_channels,
            bits_per_sample=0,
            encoding="UNKNOWN",
        )

    torchaudio.AudioMetaData = AudioMetaData
    torchaudio.list_audio_backends = list_audio_backends
    torchaudio.set_audio_backend = set_audio_backend
    torchaudio.get_audio_backend = get_audio_backend
    torchaudio.info = info

    backend_ns = getattr(torchaudio, "backend", None)
    if backend_ns is None:
        backend_ns = types.SimpleNamespace()
        torchaudio.backend = backend_ns

    common_ns = getattr(backend_ns, "common", None)
    if common_ns is None:
        common_ns = types.SimpleNamespace()
        backend_ns.common = common_ns
    common_ns.AudioMetaData = AudioMetaData


def _patch_torchmetrics_get_num_classes_compat() -> None:
    try:
        import torch
        from torchmetrics.utilities import data as tm_data
    except Exception:
        return

    if hasattr(tm_data, "get_num_classes"):
        return

    def _get_num_classes(preds=None, target=None, num_classes=None):
        if num_classes is not None:
            try:
                n = int(num_classes)
                if n > 0:
                    return n
            except Exception:
                pass

        max_idx = -1
        for item in (preds, target):
            if item is None:
                continue
            try:
                tensor = torch.as_tensor(item)
            except Exception:
                continue
            if tensor.numel() <= 0:
                continue
            try:
                candidate = int(torch.max(tensor).item())
            except Exception:
                continue
            if candidate > max_idx:
                max_idx = candidate
        return int(max_idx + 1) if max_idx >= 0 else 0

    tm_data.get_num_classes = _get_num_classes


def _patch_pytorch_lightning_model_summary_alias() -> None:
    legacy_name = "pytorch_lightning.utilities.model_summary"
    if legacy_name in sys.modules:
        return

    target_module = None
    for candidate in (
        "lightning.pytorch.utilities.model_summary",
        "lightning.pytorch.utilities.model_summary.model_summary",
    ):
        try:
            target_module = importlib.import_module(candidate)
            break
        except Exception:
            continue

    if target_module is None:
        return

    sys.modules[legacy_name] = target_module
    try:
        import pytorch_lightning.utilities as plu

        if not hasattr(plu, "model_summary"):
            plu.model_summary = target_module
    except Exception:
        pass


_configure_warning_filters()
_install_future_macos_torch_bridge()
_install_pyav_stub()
_install_module_patch_finder()
_patch_torchaudio_legacy_api()
_patch_torchmetrics_get_num_classes_compat()
_patch_pytorch_lightning_model_summary_alias()
