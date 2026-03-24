"""
audio_extractor.py - Audio extraction and preprocessing from media files.

Priority chain:
  1. FFmpeg extract → soundfile/torchaudio CPU load (large files)
  2. torchaudio direct load (small files)
  3. FFmpeg subprocess fallback

设计原则：
  - 音频解码优先使用系统硬件媒体栈（macOS: VideoToolbox）
  - 大文件分块读取，绝不一次性分配超过500MB
  - 预处理尽量留在本地加速后端（CUDA / MPS / CPU），避免不必要的数据搬运
  - 临时文件用完即删
"""

import os
import importlib
import logging
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import sitecustomize  # noqa: F401
except Exception:
    sitecustomize = None  # type: ignore[assignment]

import numpy as np
import torch
import torchaudio

from runtime_paths import find_tool_executable
from utils import (
    accelerator_backend,
    is_video_file,
    mps_is_available,
    preferred_torch_device,
    windows_hidden_subprocess_kwargs,
)

logger = logging.getLogger(__name__)

# 单次内存分配上限（samples数），避免numpy OOM
# 500MB / 4 bytes = 125M samples ≈ 2.17小时@16kHz
MAX_LOAD_SAMPLES = 125_000_000

# 小文件阈值：低于此大小直接整体加载（50MB）
SMALL_FILE_BYTES = 50 * 1024 * 1024

# torchaudio backend 尝试顺序（按成功率优先）
TORCHAUDIO_BACKEND_CANDIDATES = (
    None,
    "ffmpeg",
    "sox_io",
    "soundfile",
    "sox",
)


class AudioExtractor:
    """Extract and preprocess audio from media files with platform-aware acceleration."""

    def __init__(self, config):
        self.config = config
        self.audio_cfg = config["audio"]
        self.target_sr = self.audio_cfg["target_sample_rate"]
        self.target_channels = self.audio_cfg["target_channels"]
        self._torch_accel_device = preferred_torch_device(indexed_cuda=True)
        self._ffmpeg_hwaccel = self._select_ffmpeg_hwaccel()
        self.use_hwaccel_ffmpeg = bool(
            self.audio_cfg.get("use_gpu_decode", True) and self._ffmpeg_hwaccel
            and self._ffmpeg_hwaccel_can_help_audio_extract()
        )
        # On macOS this targets MPS; on CUDA machines it targets CUDA.
        self.use_accelerated_audio_ops = bool(
            self.audio_cfg.get("use_gpu_audio_ops", False)
            and self._torch_accel_device != "cpu"
        )
        self._ffmpeg_binary = self._resolve_ffmpeg_binary()

        # Resampler 缓存（CPU上）
        self._resamplers = {}

        decode_accel = self._ffmpeg_hwaccel if self.use_hwaccel_ffmpeg else "cpu"
        logger.info(
            f"AudioExtractor initialized: target_sr={self.target_sr}, "
            f"ffmpeg_accel={decode_accel}"
        )
        if (
            self.audio_cfg.get("use_gpu_decode", True)
            and self._ffmpeg_hwaccel
            and not self.use_hwaccel_ffmpeg
        ):
            postprocess_accel = (
                accelerator_backend() if self.use_accelerated_audio_ops else "cpu"
            )
            logger.info(
                "FFmpeg audio extraction stays on CPU; %s only helps video "
                "decode/encode, while this PCM export path remains CPU-bound. "
                "Post-extract waveform ops still run on %s when available.",
                self._ffmpeg_hwaccel,
                postprocess_accel,
            )
        if self._ffmpeg_binary:
            logger.info(f"FFmpeg decoder ready: {self._ffmpeg_binary}")
        else:
            logger.warning(
                "FFmpeg decoder unavailable. "
                "Will try direct torchaudio decode as fallback."
            )
        waveform_accel = accelerator_backend() if self.use_accelerated_audio_ops else "cpu"
        logger.info(f"torchaudio decode optimization: audio_ops_accelerator={waveform_accel}")

    def _select_ffmpeg_hwaccel(self) -> Optional[str]:
        if sys.platform == "darwin":
            return "videotoolbox"
        if torch.cuda.is_available():
            return "cuda"
        return None

    @staticmethod
    def _ffmpeg_hwaccel_can_help_audio_extract() -> bool:
        # This code path only demuxes/decodes audio to PCM (`-vn`), so FFmpeg
        # video hwaccel backends like VideoToolbox/CUDA do not materially help.
        return False

    def _resolve_ffmpeg_binary(self) -> Optional[str]:
        """Find or auto-provision FFmpeg decoder binary."""
        configured = str(self.audio_cfg.get("ffmpeg_path", "") or "").strip()
        resolved = find_tool_executable("ffmpeg", configured=configured)
        if resolved:
            return resolved

        return self._ensure_imageio_ffmpeg()

    @staticmethod
    def _install_python_package(requirement: str) -> bool:
        """Install package at runtime as last-resort bootstrap."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", requirement],
                capture_output=True,
                text=True,
                timeout=900,
                **windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                return True

            stderr = (result.stderr or result.stdout or "").strip()
            logger.warning(
                f"Failed to install {requirement} automatically: {stderr[-300:]}"
            )
            return False
        except Exception as e:
            logger.warning(f"Auto-install failed for {requirement}: {e}")
            return False

    def _ensure_imageio_ffmpeg(self) -> Optional[str]:
        """
        Auto-install imageio-ffmpeg and use its bundled FFmpeg binary.
        This avoids hard dependency on system ffmpeg in Windows.
        """
        imageio_ffmpeg = None
        try:
            import imageio_ffmpeg as _imageio_ffmpeg
            imageio_ffmpeg = _imageio_ffmpeg
        except ImportError:
            logger.info("imageio-ffmpeg not installed, installing automatically...")
            if not self._install_python_package("imageio-ffmpeg>=0.4.9"):
                return None
            try:
                import imageio_ffmpeg as _imageio_ffmpeg
                imageio_ffmpeg = _imageio_ffmpeg
            except ImportError:
                logger.warning("imageio-ffmpeg import failed after installation")
                return None

        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_exe and Path(ffmpeg_exe).exists():
                logger.info(f"Using bundled FFmpeg decoder: {ffmpeg_exe}")
                return ffmpeg_exe
        except Exception as e:
            logger.warning(f"Failed to provision bundled FFmpeg decoder: {e}")

        return None

    def _get_resampler(
        self,
        orig_sr: int,
        device: str = "cpu",
    ) -> torchaudio.transforms.Resample:
        """Get or create cached resampler for sample rate and device."""
        key = (orig_sr, device)
        if key not in self._resamplers:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.target_sr,
            )
            if device != "cpu":
                resampler = resampler.to(device)
            self._resamplers[key] = resampler
        return self._resamplers[key]

    @staticmethod
    def _numel_safe(tensor: torch.Tensor) -> int:
        try:
            return int(tensor.numel())
        except Exception:
            return 0

    def _torchaudio_load_robust(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """Try torchaudio with multiple backends for max decode success."""
        errors = []

        for backend in TORCHAUDIO_BACKEND_CANDIDATES:
            kwargs = {"normalize": True}
            if backend is not None:
                kwargs["backend"] = backend

            try:
                waveform, sr = torchaudio.load(str(audio_path), **kwargs)
                if self._numel_safe(waveform) == 0:
                    raise RuntimeError("decoded empty waveform")
                logger.debug(
                    f"torchaudio decode success: backend={backend or 'auto'}, "
                    f"shape={tuple(waveform.shape)}, sr={sr}"
                )
                return waveform.cpu(), sr
            except TypeError as e:
                if backend is None:
                    errors.append(f"auto:{e}")
                # backend 参数不支持时跳过
                continue
            except Exception as e:
                errors.append(f"{backend or 'auto'}:{e}")
                continue

        detail = "; ".join(errors[-4:]) if errors else "unknown error"
        raise RuntimeError(f"torchaudio decode failed: {detail}")

    @staticmethod
    def _resolve_stream_reader():
        """
        Resolve torchaudio StreamReader dynamically to avoid hard import errors.
        Returns StreamReader class or None when unavailable.
        """
        try:
            io_mod = getattr(torchaudio, "io", None)
            stream_reader = getattr(io_mod, "StreamReader", None)
            if stream_reader is not None:
                return stream_reader
        except Exception:
            pass

        try:
            io_mod = importlib.import_module("torchaudio.io")
            stream_reader = getattr(io_mod, "StreamReader", None)
            if stream_reader is not None:
                return stream_reader
        except Exception:
            pass

        return None

    def _decode_video_with_torchaudio(self, video_path: Path) -> Optional[Tuple[torch.Tensor, int]]:
        """Fallback video decode with torchaudio StreamReader."""
        stream_reader_cls = self._resolve_stream_reader()
        if stream_reader_cls is None:
            logger.debug("torchaudio StreamReader unavailable in current environment")
            return None

        frames_per_chunk = max(self.target_sr * 5, 16000)
        chunks = []

        try:
            streamer = stream_reader_cls(str(video_path))
            streamer.add_basic_audio_stream(
                frames_per_chunk=frames_per_chunk,
                sample_rate=self.target_sr,
                num_channels=self.target_channels,
            )

            for item in streamer.stream():
                if not item:
                    continue
                chunk = item[0]
                if chunk is None or self._numel_safe(chunk) == 0:
                    continue
                # StreamReader 输出通常是 (frames, channels)，统一转成 (channels, frames)
                if chunk.dim() == 2:
                    chunk = chunk.transpose(0, 1)
                elif chunk.dim() == 1:
                    chunk = chunk.unsqueeze(0)
                else:
                    chunk = chunk.reshape(1, -1)
                chunks.append(chunk.contiguous().cpu())

            if not chunks:
                return None

            waveform = torch.cat(chunks, dim=1).float()
            return waveform, self.target_sr

        except Exception as e:
            logger.warning(f"torchaudio StreamReader video decode failed: {e}")
            return None

    def _process_waveform(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Convert mono + resample + normalize.
        Prefer accelerated ops when available, fallback to CPU automatically.
        """
        target_device = self._torch_accel_device if self.use_accelerated_audio_ops else "cpu"
        wf = waveform

        if target_device != "cpu":
            try:
                move_kwargs = {"non_blocking": True} if target_device.startswith("cuda") else {}
                wf = wf.to(target_device, **move_kwargs)
            except Exception as e:
                logger.warning(f"Move waveform to accelerator failed, fallback CPU: {e}")
                target_device = "cpu"
                wf = waveform.cpu()
        else:
            wf = waveform.cpu()

        wf = self._to_mono(wf)
        wf = self._resample(wf, orig_sr, device=target_device)

        try:
            peak = float(torch.max(torch.abs(wf)).item())
        except Exception:
            peak = 0.0
        if peak > 1e-8:
            wf = wf / peak

        if target_device != "cpu":
            try:
                move_kwargs = {"non_blocking": True} if target_device.startswith("cuda") else {}
                wf = wf.to("cpu", **move_kwargs)
            except Exception as e:
                logger.warning(f"Accelerated postprocess finalize failed, forcing CPU copy: {e}")
                wf = wf.cpu()

        return wf.contiguous()

    def extract(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Extract audio from media file and return (audio_numpy, sample_rate).
        Audio is mono, target_sr Hz, float32 normalized.

        解码阶段优先使用平台硬件能力，后处理按可用后端自动选择。
        """
        logger.info(f"Extracting audio from: {file_path.name}")

        audio_path = file_path
        temp_created = False

        try:
            # For video files, extract audio track first via FFmpeg
            if is_video_file(file_path):
                extracted = self._extract_audio_track(file_path)
                if extracted is not None:
                    audio_path = extracted
                    temp_created = True
                else:
                    logger.warning(
                        "FFmpeg extraction failed, trying direct torchaudio decode."
                    )
                    decoded = self._decode_video_with_torchaudio(file_path)
                    if decoded is None:
                        raise RuntimeError(
                            f"Failed to decode video audio with both FFmpeg and torchaudio: {file_path}"
                        )
                    waveform, sr = decoded
                    processed = self._process_waveform(waveform, sr)
                    audio_np = processed.numpy().flatten()
                    if audio_np.size == 0:
                        raise RuntimeError(f"Extracted empty audio from: {file_path}")
                    duration = len(audio_np) / self.target_sr
                    logger.info(
                        f"Audio extracted via torchaudio StreamReader: {duration:.1f}s, "
                        f"{len(audio_np)} samples @ {self.target_sr}Hz"
                    )
                    return audio_np, self.target_sr

            # Load audio (CPU only, chunked for large files)
            waveform, sr = self._load_audio_safe(audio_path)

            # Mono + resample + normalize (prefer accelerated ops)
            waveform = self._process_waveform(waveform, sr)

            # To numpy
            audio_np = waveform.numpy().flatten()
            if audio_np.size == 0:
                raise RuntimeError(f"Extracted empty audio from: {file_path}")

            duration = len(audio_np) / self.target_sr
            logger.info(
                f"Audio extracted: {duration:.1f}s, "
                f"{len(audio_np)} samples @ {self.target_sr}Hz"
            )

            return audio_np, self.target_sr

        finally:
            # Cleanup temp file
            if temp_created and audio_path != file_path:
                try:
                    os.unlink(str(audio_path))
                except OSError:
                    pass

    def _extract_audio_track(self, video_path: Path) -> Optional[Path]:
        """
        Extract audio track from video using FFmpeg.
        直接输出目标格式（16kHz, mono, PCM s16le），避免后续重采样。
        """
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="audio_")
        os.close(fd)
        temp_wav = Path(temp_path)
        ffmpeg_bin = self._ffmpeg_binary
        if not ffmpeg_bin or not Path(ffmpeg_bin).exists():
            ffmpeg_bin = self._resolve_ffmpeg_binary()
            self._ffmpeg_binary = ffmpeg_bin

        if not ffmpeg_bin:
            logger.error(
                "No FFmpeg decoder available. "
                "Install ffmpeg or imageio-ffmpeg."
            )
            if temp_wav.exists():
                try:
                    os.unlink(str(temp_wav))
                except OSError:
                    pass
            return None

        # 策略1: FFmpeg hwaccel（主要加速视频容器/视频流解析路径）
        if self.use_hwaccel_ffmpeg:
            cmd = [
                ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                "-hwaccel", str(self._ffmpeg_hwaccel),
                "-i", str(video_path),
                "-vn",                           # no video
                "-acodec", "pcm_s16le",          # PCM 16-bit
                "-ar", str(self.target_sr),      # target sample rate
                "-ac", str(self.target_channels), # mono
                str(temp_wav),
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600,
                    **windows_hidden_subprocess_kwargs(),
                )
                if result.returncode == 0 and temp_wav.exists() and temp_wav.stat().st_size > 44:
                    logger.debug(
                        "Audio extracted via FFmpeg (%s acceleration)",
                        self._ffmpeg_hwaccel,
                    )
                    return temp_wav
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg hwaccel timed out, retrying without hwaccel")
            except Exception as e:
                logger.debug(f"FFmpeg hwaccel failed: {e}")

        # 策略2: 纯CPU FFmpeg
        cmd_fallback = [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(self.target_sr),
            "-ac", str(self.target_channels),
            str(temp_wav),
        ]
        try:
            result = subprocess.run(
                cmd_fallback, capture_output=True, text=True, timeout=600,
                **windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0 and temp_wav.exists() and temp_wav.stat().st_size > 44:
                logger.debug("Audio extracted via FFmpeg (CPU)")
                return temp_wav
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                # 清理失败的文件
                if temp_wav.exists():
                    try:
                        os.unlink(str(temp_wav))
                    except OSError:
                        pass
                return None
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            if temp_wav.exists():
                try:
                    os.unlink(str(temp_wav))
                except OSError:
                    pass
            return None

    def _load_audio_safe(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        安全加载音频文件，全部在CPU上完成。
        大文件使用 soundfile 分块读取以避免内存爆炸。
        小文件使用 torchaudio 一次性加载。
        """
        file_size = audio_path.stat().st_size

        # ── 小文件：torchaudio 直接加载 ──────────────────────
        if file_size < SMALL_FILE_BYTES:
            try:
                waveform, sr = self._torchaudio_load_robust(audio_path)
                logger.debug(
                    f"Loaded via torchaudio (small file): "
                    f"shape={waveform.shape}, sr={sr}, "
                    f"file_size={file_size / (1024*1024):.1f}MB"
                )
                return waveform, sr

            except Exception as e:
                logger.debug(f"torchaudio failed for small file ({e}), trying soundfile")

        # ── 大文件：soundfile 分块读取 ───────────────────────
        try:
            import soundfile as sf

            # 先获取文件信息（不加载数据）
            info = sf.info(str(audio_path))
            total_frames = info.frames
            sr = info.samplerate
            channels = info.channels

            logger.debug(
                f"Large audio: {total_frames} frames, {sr}Hz, "
                f"{channels}ch, {total_frames/sr:.1f}s, "
                f"file_size={file_size / (1024*1024):.1f}MB"
            )

            # 计算每块读取的帧数
            # 目标：每块 < 200MB（float32），留足余量
            max_frames_per_chunk = min(
                MAX_LOAD_SAMPLES // max(channels, 1),
                total_frames
            )

            if total_frames <= max_frames_per_chunk:
                # 可以一次读完
                data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
                # data shape: (frames, channels)
                waveform = torch.from_numpy(data.T).float()  # (channels, frames)
                del data
                logger.debug(f"Loaded via soundfile (single read): shape={waveform.shape}")
                return waveform, sr

            # 分块读取
            logger.info(
                f"  Large file ({total_frames/sr:.0f}s), "
                f"reading in chunks of {max_frames_per_chunk/sr:.0f}s..."
            )
            chunks = []
            frames_read = 0

            with sf.SoundFile(str(audio_path), 'r') as f:
                while frames_read < total_frames:
                    read_size = min(max_frames_per_chunk, total_frames - frames_read)
                    chunk = f.read(read_size, dtype="float32", always_2d=True)

                    if chunk.shape[0] == 0:
                        break

                    # 立即转为mono减少内存
                    if chunk.shape[1] > 1:
                        chunk = chunk.mean(axis=1, keepdims=True)

                    chunks.append(chunk)
                    frames_read += chunk.shape[0]

                    logger.debug(
                        f"  Read chunk: {chunk.shape[0]} frames "
                        f"({frames_read}/{total_frames})"
                    )

            # 拼接所有块
            if not chunks:
                raise RuntimeError(f"No audio data read from {audio_path}")

            all_data = np.concatenate(chunks, axis=0)  # (total_frames, 1)
            del chunks  # 释放分块列表

            waveform = torch.from_numpy(all_data.T).float()  # (1, total_frames)
            del all_data

            logger.debug(f"Loaded via soundfile (chunked): shape={waveform.shape}")
            return waveform, sr

        except ImportError:
            logger.warning("soundfile not installed, trying torchaudio for large file")

        # ── 最终回退：torchaudio（可能慢但通常可行）──────────
        try:
            waveform, sr = self._torchaudio_load_robust(audio_path)
            logger.debug(f"Loaded via torchaudio (fallback): shape={waveform.shape}")
            return waveform, sr

        except Exception as e:
            logger.error(f"All audio loading methods failed for {audio_path}: {e}")
            raise RuntimeError(
                f"Cannot load audio: {audio_path}\n"
                f"  File size: {file_size / (1024*1024):.1f}MB\n"
                f"  Error: {e}\n"
                f"  Fix:\n"
                f"    pip install soundfile imageio-ffmpeg\n"
                f"    or install system ffmpeg and set FFMPEG_BINARY"
            )

    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert to mono by averaging channels."""
        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _resample(
        self,
        waveform: torch.Tensor,
        orig_sr: int,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Resample to target sample rate on CPU / accelerated runtime."""
        if orig_sr == self.target_sr:
            return waveform

        resolved_device = device or str(waveform.device)
        resampler = self._get_resampler(orig_sr, device=resolved_device)
        waveform = resampler(waveform)
        logger.debug(f"Resampled {orig_sr}Hz → {self.target_sr}Hz")
        return waveform

    def extract_to_device_tensor(self, file_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Extract audio and move it to the best available torch device.
        Returns (tensor, sample_rate).
        """
        audio_np, sr = self.extract(file_path)
        tensor = torch.from_numpy(audio_np).float()

        device = preferred_torch_device(indexed_cuda=True)
        if device.startswith("cuda"):
            tensor = tensor.pin_memory().to(device, non_blocking=True)
        elif mps_is_available():
            tensor = tensor.to("mps")

        return tensor, sr
