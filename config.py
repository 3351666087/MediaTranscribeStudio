"""
config.py - Configuration management with validation and defaults.
"""

import os
import copy
import sys
import yaml
import logging
from typing import Dict, Optional, Any
from pathlib import Path

from runtime_paths import APP_ROOT, resolve_app_writable_path

logger = logging.getLogger(__name__)

DEFAULT_NGC_API_USERNAME = "$oauthtoken"
DEFAULT_NGC_API_KEY = ""


def _default_overlay_font_name() -> str:
    if sys.platform == "darwin":
        return "PingFang SC"
    if os.name == "nt":
        return "Microsoft YaHei"
    return "Noto Sans CJK SC"


def _default_overlay_video_codec() -> str:
    if sys.platform == "darwin":
        return "h264_videotoolbox"
    if os.name == "nt":
        return "h264_nvenc"
    return "libx264"


def _default_audio_decode_acceleration() -> bool:
    return os.name == "nt" or sys.platform == "darwin"


def _default_audio_ops_acceleration() -> bool:
    return sys.platform == "darwin"


def _default_pin_memory() -> bool:
    return sys.platform != "darwin"


def _default_non_blocking() -> bool:
    return sys.platform != "darwin"


def _default_funasr_device() -> str:
    if sys.platform == "darwin":
        return "auto"
    return "cuda:0" if os.name == "nt" else "cpu"


def _default_faster_whisper_model_size() -> str:
    return "medium" if sys.platform == "darwin" else "large-v3"


def _default_mlx_whisper_model_repo() -> str:
    if sys.platform == "darwin":
        return "mlx-community/whisper-large-v3-turbo"
    return ""


def _default_faster_whisper_device() -> str:
    if sys.platform == "darwin":
        return "auto"
    return "cuda" if os.name == "nt" else "cpu"


def _default_faster_whisper_compute_type() -> str:
    return "float16" if os.name == "nt" else "int8"


def _default_faster_whisper_cpu_threads() -> int:
    cpu_count = max(1, int(os.cpu_count() or 4))
    if sys.platform == "darwin":
        return max(4, min(8, cpu_count))
    return max(4, min(8, cpu_count))


def _default_taichi_arch() -> str:
    if sys.platform == "darwin":
        return "metal"
    return "cuda" if os.name == "nt" else "cpu"


def _default_performance_dtype() -> str:
    return "float32" if sys.platform == "darwin" else "float16"


def _default_pyannote_diar_model_candidates() -> list[str]:
    return [
        "pyannote/speaker-diarization-community-1",
        "pyannote/speaker-diarization-3.1",
    ]


def _default_pyannote_diar_model_name() -> str:
    return _default_pyannote_diar_model_candidates()[0]


def _default_overlap_prefer_pyannote() -> bool:
    return True


def _default_pyannote_prefer_local_snapshot() -> bool:
    return True


def _read_ngc_api_key_from_well_known_files() -> str:
    for candidate in (
        Path.home() / ".ngc" / "config",
        Path.home() / ".config" / "ngc" / "config",
    ):
        try:
            if not candidate.exists() or not candidate.is_file():
                continue
            for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
                text = str(line or "").strip()
                if not text or text.startswith("#"):
                    continue
                if "=" not in text:
                    continue
                key, value = text.split("=", 1)
                if key.strip().lower() in {"apikey", "api_key"}:
                    token = value.strip()
                    if token:
                        return token
        except Exception:
            continue
    return ""

# ── Default configuration (used when config.yaml is missing) ──────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "input_dir": "input_files",
        "output_dir": "output_files",
        "summary_file": "all_results_summary.txt",
    },
    "audio": {
        "target_sample_rate": 16000,
        "target_channels": 1,
        "chunk_mode": "auto",
        "auto_chunk_target_sec": 240,
        "auto_chunk_min_sec": 90,
        "auto_chunk_max_sec": 420,
        "overlap_sec": 1.0,
        "vad_aggressiveness": 3,
        "use_gpu_decode": _default_audio_decode_acceleration(),
        "use_gpu_audio_ops": _default_audio_ops_acceleration(),
        "pin_memory": _default_pin_memory(),
        "non_blocking": _default_non_blocking(),
    },
    "report": {
        "generate_html": True,
        "generate_pdf": True,
        "pdf_options": {
            "engine": "auto",
            "page_size": "A4",
            "margin_top": "15mm",
            "margin_bottom": "15mm",
            "margin_left": "15mm",
            "margin_right": "15mm",
            "display_header_footer": False,
            "print_background": True,
            "footer_template": (
                '<div style="font-size:9px;color:#aaa;'
                'text-align:center;width:100%;">'
                '<span class="pageNumber"></span> / '
                '<span class="totalPages"></span></div>'
            ),
            "header_template": "",
            "wkhtmltopdf_path": "",
        },
    },
    "video_text_overlay": {
        "enabled": False,
        "renderer": "ffmpeg",
        "embed_subtitle_stream": True,
        "style_preset": "modern_box",
        "output_suffix": ".captioned",
        "copy_audio": True,
        "font_name": _default_overlay_font_name(),
        "font_size": 28,
        "font_color": "#FFFFFF",
        "style_effect": "auto",
        "highlight_color": "#FFD54A",
        "outline_color": "#000000",
        "outline_px": 1,
        "box_color": "#000000",
        "box_opacity": 70,
        "font_bold": True,
        "bottom_margin_px": 56,
        "side_margin_px": 72,
        "max_line_chars": 18,
        "max_lines_per_caption": 2,
        "ffmpeg_path": "",
        "ffprobe_path": "",
        "fonts_dir": "",
        "ffmpeg_video_codec": _default_overlay_video_codec(),
        "ffmpeg_preset": "medium",
        "ffmpeg_crf": 20,
        "webm_video_codec": "libvpx-vp9",
        "webm_audio_codec": "libopus",
        "webm_crf": 32,
        "webm_cpu_used": 2,
        "webm_audio_bitrate": "160k",
    },
    "asr": {
        "engine": "auto",
        "funasr": {
            "model": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "vad_model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "spk_model": "iic/speech_campplus_sv_zh-cn_16k-common",
            "punc_model": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            "device": _default_funasr_device(),
            "batch_size": 16,
            "use_fp16": True,
            "hotword": "",
            "disable_update": True,
            "trust_remote_code": True,
        },
        "faster_whisper": {
            "model_size": _default_faster_whisper_model_size(),
            "device": _default_faster_whisper_device(),
            "mlx_model_repo": _default_mlx_whisper_model_repo(),
            "prefer_mlx": False,
            "allow_full_large_v3_mlx": False,
            "compute_type": _default_faster_whisper_compute_type(),
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            "word_timestamps": False,
            "language": None,
            "num_workers": 1,
            "cpu_threads": _default_faster_whisper_cpu_threads(),
            "hf_token": "",
            "mirror_endpoint": "https://hf-mirror.com",
            "official_endpoint": "https://huggingface.co",
            "mirror_retries": 3,
            "official_retries": 2,
            "download_retry_wait_sec": 1.5,
        },
        "nemo_msdd": {
            "enabled": True,
            "strict": False,
            "device": "auto",
            "final_strategy": "msdd_primary",
            "retry_cpu_on_mps_failure": sys.platform != "darwin",
            "model_path": "diar_msdd_telephonic",
            "ngc_api_key": DEFAULT_NGC_API_KEY,
            "vad_model": "vad_multilingual_marblenet",
            "speaker_model": "titanet_large",
            "model_sources": {},
            "preload_models": True,
            "download_device": "cpu",
            "download_retries": 4,
            "download_retry_wait_sec": 2.0,
            "num_speakers": 0,
            "min_speakers": 1,
            "max_speakers": 8,
            "cache_dir": "output_files/.nemo_msdd",
            "keep_temp_files": False,
            "prefer_asr_vad": True,
            "asr_vad_pad_sec": 0.05,
            "asr_vad_merge_gap_sec": 0.15,
            "asr_vad_min_duration_sec": 0.10,
            "sortformer": {
                "enabled": True,
                "model_name": "nvidia/diar_streaming_sortformer_4spk-v2.1",
                "batch_size": 1,
                "device": "auto",
                "hf_token": "",
                "auto_endpoint_probe": False,
                "mirror_endpoint": "https://hf-mirror.com",
                "official_endpoint": "https://huggingface.co",
                "mirror_retries": 3,
                "official_retries": 2,
                "download_retry_wait_sec": 1.5,
                "chunk_len": 340,
                "chunk_right_context": 40,
                "fifo_len": 40,
                "spkcache_update_period": 300,
                "spkcache_len": 188,
                "merge_gap_sec": 0.08,
                "min_turn_sec": 0.05,
            },
            "hybrid_fusion": {
                "enabled": True,
                "boundary_snap_sec": 0.18,
                "min_piece_sec": 0.20,
                "pyannote_weight": 1.25,
                "msdd_weight": 1.0,
                "sortformer_weight": 0.55,
                "map_min_overlap_ratio": 0.62,
                "map_min_overlap_sec": 1.2,
                "map_min_margin_ratio": 1.25,
                "count_longform_preserve_msdd_sec": 480.0,
                "count_msdd_pyannote_delta": 1,
                "merge_weak_speakers": True,
                "overlap_union": {
                    "enabled": True,
                    "merge_gap_sec": 0.08,
                    "min_region_sec": 0.20,
                },
                "posterior_decoder": {
                    "enabled": True,
                    "frame_hop_sec": 0.08,
                    "min_turn_sec": 0.20,
                    "switch_penalty": 0.42,
                    "stay_bonus": 0.16,
                    "boundary_relief": 0.28,
                    "use_native_viterbi": True,
                    "sortformer_saturation_scale": 0.65,
                    "global_prior": {
                        "enabled": True,
                        "emission_scale": 0.18,
                        "agreement_bonus": 0.14,
                        "exclusive_bonus": 0.22,
                        "anchor_bonus": 0.16,
                        "unanchored_penalty": 0.12,
                        "sortformer_cap_penalty": 0.28,
                    },
                    "overlap_decode": {
                        "enabled": True,
                        "primary_min_prob": 0.48,
                        "secondary_min_prob": 0.38,
                        "frame_overlap_gate": 0.34,
                        "agreement_gate": 0.34,
                        "seed_bonus": 0.08,
                        "max_active_speakers": 2,
                        "min_overlap_sec": 0.16,
                        "merge_gap_sec": 0.08,
                    },
                    "dump_examples": {
                        "enabled": False,
                        "output_dir": "output_files/posterior_fusion_examples",
                    },
                    "trainer": {
                        "examples_dir": "output_files/posterior_fusion_examples",
                        "rttm_dir": "output_files/posterior_fusion_rttm",
                        "output_path": "output_files/.posterior_fusion_calibrator.json",
                        "epochs": 6,
                        "threshold_min": 0.20,
                        "threshold_max": 0.75,
                        "threshold_num": 12,
                        "primary_loss_candidates": [0.85, 1.0, 1.20],
                        "activity_loss_candidates": [0.90, 1.0, 1.15],
                        "overlap_loss_candidates": [1.0, 1.35, 1.70],
                        "objective_weights": {
                            "primary": 1.0,
                            "activity": 1.0,
                            "overlap": 1.35,
                        },
                    },
                    "calibrator": {
                        "enabled": True,
                        "persist_path": "output_files/.posterior_fusion_calibrator.json",
                        "online_learning_rate": 0.03,
                        "l2_reg": 0.002,
                        "min_pseudo_margin": 0.12,
                        "min_pseudo_models": 2,
                        "activity_threshold": 0.46,
                        "activity_bias": -1.05,
                        "feature_weights": {
                            "msdd": 1.20,
                            "pyannote": 1.35,
                            "sortformer": 0.72,
                            "msdd_pyannote_agree": 0.70,
                            "msdd_sortformer_agree": 0.42,
                            "pyannote_sortformer_agree": 0.50,
                            "agreement_ratio": 0.35,
                            "boundary_support": 0.22,
                            "exclusive_support": 0.32,
                            "anchor_support": 0.18,
                        },
                        "activity_feature_weights": {
                            "msdd": 1.10,
                            "pyannote": 1.18,
                            "sortformer": 0.58,
                            "msdd_pyannote_agree": 0.82,
                            "msdd_sortformer_agree": 0.48,
                            "pyannote_sortformer_agree": 0.58,
                            "agreement_ratio": 0.52,
                            "boundary_support": -0.08,
                            "exclusive_support": 0.18,
                            "anchor_support": 0.10,
                        },
                    },
                },
            },
            "vad_onset": 0.8,
            "vad_offset": 0.6,
            "vad_pad_onset": 0.05,
            "vad_pad_offset": -0.05,
            "vad_min_duration_on": 0.1,
            "vad_min_duration_off": 0.2,
            "sigmoid_threshold": 0.7,
            "infer_batch_size": 25,
            "seq_eval_mode": False,
            "split_infer": True,
            "diar_window_length": 50,
            "overlap_infer_spk_limit": 5,
            "use_speaker_model_from_ckpt": True,
            "num_workers": 0 if os.name == "nt" else 1,
            "torchaudio_fallback": {
                "enabled": True,
                "sample_rate": 16000,
                "n_mfcc": 20,
                "min_segment_sec": 0.6,
                "max_segment_sec": 12.0,
                "max_probe_segments": 120,
                "auto_silhouette_min": 0.08,
                "num_speakers": 0,
                "min_speakers": 1,
                "max_speakers": 8,
            },
            "pyannote_fallback": {
                "enabled": True,
                "provider": "pyannote.audio",
                "model_name": _default_pyannote_diar_model_name(),
                "model_candidates": _default_pyannote_diar_model_candidates(),
                "prefer_local_snapshot": _default_pyannote_prefer_local_snapshot(),
                "trust_remote_code": True,
                "hf_token": "",
                "device": "mps" if sys.platform == "darwin" else "auto",
                "require_gpu": bool(sys.platform == "darwin"),
                "auto_endpoint_probe": False,
                "mirror_endpoint": "https://hf-mirror.com",
                "official_endpoint": "https://huggingface.co",
                "mirror_retries": 3,
                "official_retries": 2,
                "download_retry_wait_sec": 1.5,
                "merge_gap_sec": 0.08,
                "min_turn_sec": 0.05,
                "num_speakers": 0,
                "min_speakers": 1,
                "max_speakers": 8,
                "specialize_longform": True,
                "specialize_min_view_sec": 0.8,
                "specialize_max_view_sec": 6.0,
                "specialize_view_sec": 1.8,
                "specialize_max_views": 3,
                "specialize_batch_size": 32 if sys.platform == "darwin" else 24,
                "specialize_short_turn_sec": 1.8,
                "specialize_merge_similarity": 0.91,
                "specialize_short_merge_similarity": 0.86,
                "specialize_attach_similarity": 0.84,
                "specialize_dense_turn_threshold": 240,
                "specialize_dense_max_views": 2,
                "specialize_dense_view_sec": 1.25,
                "specialize_ultra_dense_turn_threshold": 420,
                "specialize_ultra_dense_max_views": 1,
                "specialize_ultra_dense_view_sec": 0.95,
            },
            "overlap_handling": {
                "enabled": False,
                "min_region_sec": 0.25,
                "window_pad_sec": 0.12,
                "max_region_sec": 20.0,
                "max_regions": 120,
                "min_segment_overlap_ratio": 0.15,
                "osd": {
                    "enabled": True,
                    "provider": "pyannote",
                    "model_name": "pyannote/overlapped-speech-detection",
                    "trust_remote_code": True,
                    "hf_token": "",
                    "device": "auto",
                    "auto_endpoint_probe": False,
                    "mirror_endpoint": "https://hf-mirror.com",
                    "official_endpoint": "https://huggingface.co",
                    "mirror_retries": 3,
                    "official_retries": 2,
                    "download_retry_wait_sec": 1.5,
                    "pad_sec": 0.10,
                    "merge_gap_sec": 0.08,
                },
                "separation": {
                    "enabled": True,
                    "provider": "auto",
                    "prefer_pyannote": _default_overlap_prefer_pyannote(),
                    "prefer_local_snapshot": _default_pyannote_prefer_local_snapshot(),
                    "task": "speech_separation",
                    "model_name": "MossFormer2_SS_16K",
                    "pyannote_model_name": "pyannote/speech-separation-ami-1.0",
                    "trust_remote_code": True,
                    "hf_token": "",
                    "device": "auto",
                    "auto_endpoint_probe": False,
                    "mirror_endpoint": "https://hf-mirror.com",
                    "official_endpoint": "https://huggingface.co",
                    "mirror_retries": 3,
                    "official_retries": 2,
                    "download_retry_wait_sec": 1.5,
                    "num_speakers": 2,
                    "max_streams": 2,
                    "min_stream_peak": 1e-4,
                    "allow_mixed_audio_fallback": False,
                },
                "merge": {
                    "drop_original_overlap": False,
                    "drop_if_overlap_ratio_ge": 0.75,
                },
            },
        },
    },
    "language": {
        "auto_detect": True,
        "primary_languages": ["zh", "en", "yue"],
        "fallback_language": "zh",
        "chunk_level_engine_switch": True,
        "chunk_level_engine_switch_min_duration_sec": 6.0,
        "chunk_engine_switch_min_probability": 0.60,
    },
    "translation": {
        "enabled": False,
        "target_language": "zh",
        "max_concurrent": 8,
        "skip_same_language": True,
    },
    "llm": {
        "enabled": False,
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.5-plus",
        "model_candidates": ["qwen3.5-plus", "qwen-plus", "qwen-max"],
        "max_concurrent": 8,
        "provider_max_concurrent": 2,
        "timeout": 60,
        "max_retries": 3,
        "retry_delay": 1.0,
        "server_error_cooldown_sec": 2.0,
        "temperature": 0.3,
        "max_tokens": 4096,
        "enable_thinking": False,
        "summary_prompt": "",
        "lang_detect_prompt": "",
        "hierarchical_summary": {
            "enabled": True,
            "chunk_target_chars": 8000,
            "single_pass_chars": 6000,
            "chunk_overlap_chars": 400,
            "reduce_target_chars": 10000,
            "max_chunks": 32,
        },
        "request_limits": {
            "segment_text_chars": 4000,
            "payload_warn_chars": 12000,
            "payload_hard_chars": 18000,
        },
        "speaker_arbitration": {
            "enabled": False,
            "window_chars": 9000,
            "max_segments_per_window": 36,
            "window_overlap_segments": 4,
            "min_confidence": 0.62,
            "max_changed_ratio": 0.40,
            "max_highlight_terms": 3,
            "enable_segment_merge": True,
            "merge_min_confidence": 0.68,
            "max_merge_group_size": 4,
            "max_merge_gap_sec": 0.90,
            "max_merged_duration_sec": 10.0,
        },
        "optimize_language": {
            "enabled": False,
            "max_concurrent": 8,
            "strict_keep_text": True,
            "infer_missing_words": True,
        },
    },
    "performance": {
        "dtype": _default_performance_dtype(),
        "use_torch_compile": True,
        "use_cuda_graph": False,
        "use_cudnn_benchmark": True,
        "cuda_streams": 2,
        "max_concurrent_files": 2,
        "gpu_memory_fraction": 0.98,
        "empty_cache_interval": 5,
        "cache_cleanup_threshold_pct": 96.0,
        "batch_accumulation": 4,
        "aggressive_cuda_cleanup": False,
        "force_cuda_cleanup_every": 8,
    },
    "taichi": {
        "enabled": True,
        "arch": _default_taichi_arch(),
        "default_fp": 32,
    },
    "output": {
        "txt_format": "[{start} --> {end}] {speaker}: {text}",
        "json_indent": 2,
        "timestamp_format": "HH:MM:SS.mmm",
        "speaker_labels": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    },
    "logging": {
        "level": "INFO",
        "log_file": "pipeline.log",
        "show_progress": True,
    },
    "resume": {
        "enabled": True,
        "persist_segments": True,
    },
    "ui": {
        "posterior_fusion": {
            "auto_run_twice": False,
        },
    },
}

# ── Supported media extensions ────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
ALL_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def _resolve_hf_token(cfg: Dict[str, Any]) -> str:
    """Resolve Hugging Face token from env first, then config."""
    env_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or ""
    ).strip()
    if env_token:
        return env_token

    asr_cfg = cfg.get("asr", {})
    fw_cfg = asr_cfg.get("faster_whisper", {}) or {}
    wx_cfg = asr_cfg.get("whisperx", {}) or {}
    cfg_token = (fw_cfg.get("hf_token") or wx_cfg.get("hf_token") or "").strip()
    if cfg_token:
        return cfg_token

    # Last fallback: bundled default token from DEFAULT_CONFIG, if provided.
    default_fw_cfg = DEFAULT_CONFIG.get("asr", {}).get("faster_whisper", {}) or {}
    return str(default_fw_cfg.get("hf_token") or "").strip()


def _apply_hf_token_env(cfg: Dict[str, Any]) -> None:
    """Export token to common env vars so HF Hub requests are authenticated."""
    token = _resolve_hf_token(cfg)
    if not token:
        return

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    os.environ["HUGGINGFACE_TOKEN"] = token


def _resolve_ngc_api_key(cfg: Dict[str, Any]) -> str:
    """Resolve NGC API key from env first, then config."""
    env_key = (
        os.getenv("NGC_API_KEY")
        or os.getenv("NVIDIA_NGC_API_KEY")
        or os.getenv("NGC_CLI_API_KEY")
        or ""
    ).strip()
    if env_key:
        return env_key

    asr_cfg = cfg.get("asr", {})
    nemo_cfg = asr_cfg.get("nemo_msdd", {}) or {}
    cfg_key = str(nemo_cfg.get("ngc_api_key") or "").strip()
    if cfg_key:
        return cfg_key
    file_key = _read_ngc_api_key_from_well_known_files()
    if file_key:
        return file_key
    return DEFAULT_NGC_API_KEY


def _apply_ngc_api_key_env(cfg: Dict[str, Any]) -> None:
    """Export NGC API key to common env vars for NeMo model downloads."""
    key = _resolve_ngc_api_key(cfg)
    if not key:
        return

    os.environ["NGC_API_KEY"] = key
    os.environ["NVIDIA_NGC_API_KEY"] = key
    os.environ["NGC_CLI_API_KEY"] = key


def _migrate_legacy_clearvoice_to_nemo_msdd(cfg: Dict[str, Any]) -> None:
    """Best-effort migration from deprecated asr.clearvoice settings."""
    asr_cfg = cfg.get("asr", {})
    if not isinstance(asr_cfg, dict):
        return

    clearvoice_cfg = asr_cfg.get("clearvoice")
    nemo_cfg = asr_cfg.get("nemo_msdd")
    if not isinstance(clearvoice_cfg, dict):
        return
    if isinstance(nemo_cfg, dict) and nemo_cfg:
        # User already provided new settings; only drop legacy key.
        asr_cfg.pop("clearvoice", None)
        return

    auto_ref_cfg = clearvoice_cfg.get("auto_reference", {})
    if not isinstance(auto_ref_cfg, dict):
        auto_ref_cfg = {}

    asr_cfg["nemo_msdd"] = {
        "enabled": bool(clearvoice_cfg.get("enabled", True)),
        "strict": False,
        "device": "auto",
        "model_path": "diar_msdd_telephonic",
        "ngc_api_key": DEFAULT_NGC_API_KEY,
        "vad_model": "vad_multilingual_marblenet",
        "speaker_model": "titanet_large",
        "model_sources": {},
        "preload_models": True,
        "download_device": "cpu",
        "download_retries": 4,
        "download_retry_wait_sec": 2.0,
        "num_speakers": int(auto_ref_cfg.get("num_speakers", 0) or 0),
        "min_speakers": int(auto_ref_cfg.get("min_speakers", 1) or 1),
        "max_speakers": int(auto_ref_cfg.get("max_speakers", 8) or 8),
        "cache_dir": "output_files/.nemo_msdd",
        "keep_temp_files": False,
        "sortformer": {
            "enabled": True,
            "model_name": "nvidia/diar_streaming_sortformer_4spk-v2.1",
            "batch_size": 1,
            "device": "auto",
            "hf_token": "",
            "auto_endpoint_probe": False,
            "mirror_endpoint": "https://hf-mirror.com",
            "official_endpoint": "https://huggingface.co",
            "mirror_retries": 3,
            "official_retries": 2,
            "download_retry_wait_sec": 1.5,
            "chunk_len": 340,
            "chunk_right_context": 40,
            "fifo_len": 40,
            "spkcache_update_period": 300,
            "spkcache_len": 188,
            "merge_gap_sec": 0.08,
            "min_turn_sec": 0.05,
        },
        "hybrid_fusion": {
            "enabled": True,
            "boundary_snap_sec": 0.18,
            "min_piece_sec": 0.20,
            "pyannote_weight": 1.25,
            "msdd_weight": 1.0,
            "sortformer_weight": 0.55,
            "map_min_overlap_ratio": 0.62,
            "map_min_overlap_sec": 1.2,
            "map_min_margin_ratio": 1.25,
            "count_longform_preserve_msdd_sec": 480.0,
            "count_msdd_pyannote_delta": 1,
            "merge_weak_speakers": True,
            "overlap_union": {
                "enabled": True,
                "merge_gap_sec": 0.08,
                "min_region_sec": 0.20,
            },
            "posterior_decoder": {
                "enabled": True,
                "frame_hop_sec": 0.08,
                "min_turn_sec": 0.20,
                "switch_penalty": 0.42,
                "stay_bonus": 0.16,
                "boundary_relief": 0.28,
                "use_native_viterbi": True,
                "sortformer_saturation_scale": 0.65,
                "global_prior": {
                    "enabled": True,
                    "emission_scale": 0.18,
                    "agreement_bonus": 0.14,
                    "exclusive_bonus": 0.22,
                    "anchor_bonus": 0.16,
                    "unanchored_penalty": 0.12,
                    "sortformer_cap_penalty": 0.28,
                },
                "overlap_decode": {
                    "enabled": True,
                    "primary_min_prob": 0.48,
                    "secondary_min_prob": 0.38,
                    "frame_overlap_gate": 0.34,
                    "agreement_gate": 0.34,
                    "seed_bonus": 0.08,
                    "max_active_speakers": 2,
                    "min_overlap_sec": 0.16,
                    "merge_gap_sec": 0.08,
                },
                "dump_examples": {
                    "enabled": False,
                    "output_dir": "output_files/posterior_fusion_examples",
                },
                "trainer": {
                    "examples_dir": "output_files/posterior_fusion_examples",
                    "rttm_dir": "output_files/posterior_fusion_rttm",
                    "output_path": "output_files/.posterior_fusion_calibrator.json",
                    "epochs": 6,
                    "threshold_min": 0.20,
                    "threshold_max": 0.75,
                    "threshold_num": 12,
                    "primary_loss_candidates": [0.85, 1.0, 1.20],
                    "activity_loss_candidates": [0.90, 1.0, 1.15],
                    "overlap_loss_candidates": [1.0, 1.35, 1.70],
                    "objective_weights": {
                        "primary": 1.0,
                        "activity": 1.0,
                        "overlap": 1.35,
                    },
                },
                "calibrator": {
                    "enabled": True,
                    "persist_path": "output_files/.posterior_fusion_calibrator.json",
                    "online_learning_rate": 0.03,
                    "l2_reg": 0.002,
                    "min_pseudo_margin": 0.12,
                    "min_pseudo_models": 2,
                    "activity_threshold": 0.46,
                    "activity_bias": -1.05,
                    "feature_weights": {
                        "msdd": 1.20,
                        "pyannote": 1.35,
                        "sortformer": 0.72,
                        "msdd_pyannote_agree": 0.70,
                        "msdd_sortformer_agree": 0.42,
                        "pyannote_sortformer_agree": 0.50,
                        "agreement_ratio": 0.35,
                        "boundary_support": 0.22,
                        "exclusive_support": 0.32,
                        "anchor_support": 0.18,
                    },
                    "activity_feature_weights": {
                        "msdd": 1.10,
                        "pyannote": 1.18,
                        "sortformer": 0.58,
                        "msdd_pyannote_agree": 0.82,
                        "msdd_sortformer_agree": 0.48,
                        "pyannote_sortformer_agree": 0.58,
                        "agreement_ratio": 0.52,
                        "boundary_support": -0.08,
                        "exclusive_support": 0.18,
                        "anchor_support": 0.10,
                    },
                },
            },
        },
        "vad_onset": 0.8,
        "vad_offset": 0.6,
        "vad_pad_onset": 0.05,
        "vad_pad_offset": -0.05,
        "vad_min_duration_on": 0.1,
        "vad_min_duration_off": 0.2,
        "sigmoid_threshold": 0.7,
        "infer_batch_size": 25,
        "seq_eval_mode": False,
        "split_infer": True,
        "diar_window_length": 50,
        "overlap_infer_spk_limit": 5,
        "use_speaker_model_from_ckpt": True,
        "num_workers": 0 if os.name == "nt" else 1,
        "pyannote_fallback": {
            "enabled": True,
            "provider": "pyannote.audio",
            "model_name": _default_pyannote_diar_model_name(),
            "model_candidates": _default_pyannote_diar_model_candidates(),
            "prefer_local_snapshot": _default_pyannote_prefer_local_snapshot(),
            "trust_remote_code": True,
            "hf_token": "",
            "device": "auto",
            "auto_endpoint_probe": False,
            "mirror_endpoint": "https://hf-mirror.com",
            "official_endpoint": "https://huggingface.co",
            "mirror_retries": 3,
            "official_retries": 2,
            "download_retry_wait_sec": 1.5,
            "merge_gap_sec": 0.08,
            "min_turn_sec": 0.05,
            "num_speakers": 0,
            "min_speakers": 1,
            "max_speakers": 8,
        },
        "overlap_handling": {
            "enabled": True,
            "min_region_sec": 0.25,
            "window_pad_sec": 0.12,
            "max_region_sec": 20.0,
            "max_regions": 120,
            "min_segment_overlap_ratio": 0.15,
            "osd": {
                "enabled": True,
                "provider": "pyannote",
                "model_name": "pyannote/overlapped-speech-detection",
                "trust_remote_code": True,
                "hf_token": "",
                "device": "auto",
                "auto_endpoint_probe": False,
                "mirror_endpoint": "https://hf-mirror.com",
                "official_endpoint": "https://huggingface.co",
                "mirror_retries": 3,
                "official_retries": 2,
                "download_retry_wait_sec": 1.5,
                "pad_sec": 0.10,
                "merge_gap_sec": 0.08,
            },
            "separation": {
                "enabled": True,
                "provider": "auto",
                "prefer_pyannote": _default_overlap_prefer_pyannote(),
                "prefer_local_snapshot": _default_pyannote_prefer_local_snapshot(),
                "task": "speech_separation",
                "model_name": "MossFormer2_SS_16K",
                "pyannote_model_name": "pyannote/speech-separation-ami-1.0",
                "trust_remote_code": True,
                "hf_token": "",
                "device": "auto",
                "auto_endpoint_probe": False,
                "mirror_endpoint": "https://hf-mirror.com",
                "official_endpoint": "https://huggingface.co",
                "mirror_retries": 3,
                "official_retries": 2,
                "download_retry_wait_sec": 1.5,
                "num_speakers": 2,
                "max_streams": 2,
                "min_stream_peak": 1e-4,
                "allow_mixed_audio_fallback": False,
            },
            "merge": {
                "drop_original_overlap": False,
                "drop_if_overlap_ratio_ge": 0.75,
            },
        },
    }
    asr_cfg.pop("clearvoice", None)
    logger.info("Migrated config: converted asr.clearvoice to asr.nemo_msdd defaults")


def _migrate_legacy_nemo_nested_sections(cfg: Dict[str, Any]) -> None:
    """
    Fold misplaced diarization sections back under asr.nemo_msdd.

    Some configs incorrectly place pyannote/overlap settings at asr.* level
    or under asr.pyannote_fallback.*. Runtime code only reads
    asr.nemo_msdd.pyannote_fallback / overlap_handling, so normalize here.
    """
    asr_cfg = cfg.get("asr", {})
    if not isinstance(asr_cfg, dict):
        return

    nemo_cfg = asr_cfg.get("nemo_msdd", {}) or {}
    if not isinstance(nemo_cfg, dict):
        nemo_cfg = {}

    legacy_pyannote = asr_cfg.get("pyannote_fallback", {}) or {}
    legacy_root_overlap = asr_cfg.get("overlap_handling", {}) or {}
    legacy_root_separation = asr_cfg.get("separation", {}) or {}
    legacy_root_merge = asr_cfg.get("merge", {}) or {}

    migrated_parts: list[str] = []

    if isinstance(legacy_pyannote, dict) and legacy_pyannote:
        pyannote_payload = {
            key: copy.deepcopy(value)
            for key, value in legacy_pyannote.items()
            if key not in {"overlap_handling", "separation", "merge"}
        }
        if pyannote_payload:
            current_pyannote = nemo_cfg.get("pyannote_fallback", {}) or {}
            if not isinstance(current_pyannote, dict):
                current_pyannote = {}
            nemo_cfg["pyannote_fallback"] = _deep_merge(
                current_pyannote,
                pyannote_payload,
            )
            migrated_parts.append("pyannote_fallback")

    overlap_payload: Dict[str, Any] = {}
    if isinstance(legacy_root_overlap, dict) and legacy_root_overlap:
        overlap_payload = _deep_merge(overlap_payload, legacy_root_overlap)

    if isinstance(legacy_pyannote, dict) and legacy_pyannote:
        nested_overlap = legacy_pyannote.get("overlap_handling", {}) or {}
        if isinstance(nested_overlap, dict) and nested_overlap:
            overlap_payload = _deep_merge(overlap_payload, nested_overlap)

        nested_separation = legacy_pyannote.get("separation", {}) or {}
        if isinstance(nested_separation, dict) and nested_separation:
            current_sep = overlap_payload.get("separation", {}) or {}
            if not isinstance(current_sep, dict):
                current_sep = {}
            overlap_payload["separation"] = _deep_merge(current_sep, nested_separation)

        nested_merge = legacy_pyannote.get("merge", {}) or {}
        if isinstance(nested_merge, dict) and nested_merge:
            current_merge = overlap_payload.get("merge", {}) or {}
            if not isinstance(current_merge, dict):
                current_merge = {}
            overlap_payload["merge"] = _deep_merge(current_merge, nested_merge)

    if isinstance(legacy_root_separation, dict) and legacy_root_separation:
        current_sep = overlap_payload.get("separation", {}) or {}
        if not isinstance(current_sep, dict):
            current_sep = {}
        overlap_payload["separation"] = _deep_merge(current_sep, legacy_root_separation)

    if isinstance(legacy_root_merge, dict) and legacy_root_merge:
        current_merge = overlap_payload.get("merge", {}) or {}
        if not isinstance(current_merge, dict):
            current_merge = {}
        overlap_payload["merge"] = _deep_merge(current_merge, legacy_root_merge)

    if overlap_payload:
        current_overlap = nemo_cfg.get("overlap_handling", {}) or {}
        if not isinstance(current_overlap, dict):
            current_overlap = {}
        nemo_cfg["overlap_handling"] = _deep_merge(current_overlap, overlap_payload)
        migrated_parts.append("overlap_handling")

    if not migrated_parts:
        return

    asr_cfg["nemo_msdd"] = nemo_cfg
    asr_cfg.pop("pyannote_fallback", None)
    asr_cfg.pop("overlap_handling", None)
    asr_cfg.pop("separation", None)
    asr_cfg.pop("merge", None)
    logger.info(
        "Migrated config: folded legacy ASR diarization keys into asr.nemo_msdd (%s)",
        ", ".join(migrated_parts),
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config:
    """Unified configuration object with dot-notation access."""

    def __init__(self, config_path: Optional[str] = None):
        self._data: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)

        # Try to load from yaml
        if config_path is None:
            candidates = [
                resolve_app_writable_path("config.yaml"),
                APP_ROOT / "config.yaml",
                Path.cwd() / "config.yaml",
            ]
            for c in candidates:
                if c.exists():
                    config_path = str(c)
                    break

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_cfg = yaml.safe_load(f) or {}
                self._data = _deep_merge(DEFAULT_CONFIG, user_cfg)
                logger.info(f"Loaded configuration from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        else:
            logger.info("No config.yaml found. Using default configuration.")

        # Handle backward compatibility: whisperx → faster_whisper
        if "whisperx" in self._data.get("asr", {}):
            legacy_cfg = self._data["asr"].pop("whisperx") or {}
            current_cfg = self._data["asr"].get("faster_whisper", {})
            # legacy whisperx fields should override current faster_whisper fields
            self._data["asr"]["faster_whisper"] = _deep_merge(current_cfg, legacy_cfg)
            logger.info("Migrated config: merged asr.whisperx into asr.faster_whisper")

        _migrate_legacy_clearvoice_to_nemo_msdd(self._data)
        _migrate_legacy_nemo_nested_sections(self._data)

        _apply_hf_token_env(self._data)
        _apply_ngc_api_key_env(self._data)

        # Resolve paths relative to project root
        project_root = resolve_app_writable_path(".")
        input_dir = Path(self._data["paths"]["input_dir"])
        output_dir = Path(self._data["paths"]["output_dir"])
        if not input_dir.is_absolute():
            input_dir = project_root / input_dir
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        self._data["paths"]["input_dir"] = str(input_dir)
        self._data["paths"]["output_dir"] = str(output_dir)

        self._validate()

    def _validate(self):
        """Validate critical configuration values."""
        input_dir = Path(self._data["paths"]["input_dir"])
        if not input_dir.exists():
            input_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created input directory: {input_dir}")

        output_dir = Path(self._data["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_dtypes = {"float16", "bfloat16", "float32", "int8"}
        if self._data["performance"]["dtype"] not in valid_dtypes:
            logger.warning("Invalid dtype, falling back to float16")
            self._data["performance"]["dtype"] = "float16"
        try:
            cache_interval = int(
                (self._data.get("performance", {}) or {}).get("empty_cache_interval", 5)
            )
        except (TypeError, ValueError):
            cache_interval = 5
        self._data.setdefault("performance", {})["empty_cache_interval"] = max(0, cache_interval)

        try:
            cleanup_threshold = float(
                (self._data.get("performance", {}) or {}).get(
                    "cache_cleanup_threshold_pct",
                    96.0,
                )
            )
        except (TypeError, ValueError):
            cleanup_threshold = 96.0
        self._data.setdefault("performance", {})["cache_cleanup_threshold_pct"] = max(
            50.0,
            min(99.5, cleanup_threshold),
        )

        # Normalize engine name
        engine = self._data["asr"]["engine"]
        if engine in ("whisperx", "faster-whisper"):
            self._data["asr"]["engine"] = "faster_whisper"
        elif engine == "mlx-whisper":
            self._data["asr"]["engine"] = "mlx_whisper"

        audio_cfg = self._data.get("audio", {}) or {}
        chunk_mode = str(audio_cfg.get("chunk_mode", "auto") or "auto").strip().lower()
        if chunk_mode not in {"auto", "fixed"}:
            chunk_mode = "auto"
        audio_cfg["chunk_mode"] = chunk_mode

        if "chunk_duration_sec" in audio_cfg and audio_cfg.get("chunk_duration_sec") is not None:
            # Backward compatibility: keep legacy fixed chunk field if user still sets it.
            try:
                fixed_chunk = float(audio_cfg.get("chunk_duration_sec"))
            except (TypeError, ValueError):
                fixed_chunk = 240.0
            audio_cfg["chunk_duration_sec"] = max(15.0, fixed_chunk)

        try:
            auto_target = float(audio_cfg.get("auto_chunk_target_sec", 240) or 240)
        except (TypeError, ValueError):
            auto_target = 240.0
        try:
            auto_min = float(audio_cfg.get("auto_chunk_min_sec", 90) or 90)
        except (TypeError, ValueError):
            auto_min = 90.0
        try:
            auto_max = float(audio_cfg.get("auto_chunk_max_sec", 420) or 420)
        except (TypeError, ValueError):
            auto_max = 420.0

        auto_min = max(15.0, auto_min)
        auto_max = max(auto_min, auto_max)
        auto_target = max(auto_min, min(auto_max, auto_target))

        audio_cfg["auto_chunk_min_sec"] = auto_min
        audio_cfg["auto_chunk_max_sec"] = auto_max
        audio_cfg["auto_chunk_target_sec"] = auto_target
        if sys.platform == "darwin":
            audio_cfg["use_gpu_decode"] = bool(
                audio_cfg.get("use_gpu_decode", _default_audio_decode_acceleration())
            )
            audio_cfg["use_gpu_audio_ops"] = bool(
                audio_cfg.get("use_gpu_audio_ops", _default_audio_ops_acceleration())
            )
            audio_cfg["pin_memory"] = False
            audio_cfg["non_blocking"] = False
        self._data["audio"] = audio_cfg

        asr_cfg = self._data.get("asr", {}) or {}
        if not isinstance(asr_cfg, dict):
            asr_cfg = {}
        funasr_cfg = asr_cfg.get("funasr", {}) if isinstance(asr_cfg.get("funasr", {}), dict) else {}
        if not isinstance(funasr_cfg, dict):
            funasr_cfg = {}
        if "disable_update" not in funasr_cfg:
            funasr_cfg["disable_update"] = True
        if "trust_remote_code" not in funasr_cfg:
            funasr_cfg["trust_remote_code"] = True
        asr_cfg["funasr"] = funasr_cfg
        self._data["asr"] = asr_cfg

        if sys.platform == "darwin":
            funasr_cfg["device"] = str(funasr_cfg.get("device", "") or "").strip() or "auto"
            funasr_cfg["use_fp16"] = False
            try:
                funasr_cfg["batch_size"] = max(
                    1,
                    min(int(funasr_cfg.get("batch_size", 8) or 8), 12),
                )
            except Exception:
                funasr_cfg["batch_size"] = 8

            fw_cfg = self._data.setdefault("asr", {}).setdefault("faster_whisper", {})
            fw_cfg["device"] = str(fw_cfg.get("device", "") or "").strip() or "auto"
            fw_cfg["mlx_model_repo"] = str(
                fw_cfg.get("mlx_model_repo", "") or ""
            ).strip() or _default_mlx_whisper_model_repo()
            fw_cfg["prefer_mlx"] = bool(fw_cfg.get("prefer_mlx", False))
            fw_cfg["allow_full_large_v3_mlx"] = bool(
                fw_cfg.get("allow_full_large_v3_mlx", False)
            )
            fw_compute_type = str(fw_cfg.get("compute_type", "") or "").strip().lower()
            if fw_compute_type in {"float16", ""}:
                fw_cfg["compute_type"] = "int8"
            fw_model_size = str(fw_cfg.get("model_size", "") or "").strip().lower()
            if fw_model_size in {"", "small", "large", "large-v2", "large-v3"}:
                fw_cfg["model_size"] = _default_faster_whisper_model_size()
            try:
                fw_cfg["cpu_threads"] = max(
                    1,
                    int(
                        fw_cfg.get(
                            "cpu_threads",
                            _default_faster_whisper_cpu_threads(),
                        )
                    ),
                )
            except Exception:
                fw_cfg["cpu_threads"] = _default_faster_whisper_cpu_threads()

            overlay_cfg = self._data.setdefault("video_text_overlay", {})
            overlay_font = str(overlay_cfg.get("font_name", "") or "").strip().lower()
            if overlay_font in {"microsoft yahei", "microsoft yahei ui", ""}:
                overlay_cfg["font_name"] = _default_overlay_font_name()
            overlay_codec = str(overlay_cfg.get("ffmpeg_video_codec", "") or "").strip().lower()
            if overlay_codec in {"h264_nvenc", "hevc_nvenc", ""}:
                overlay_cfg["ffmpeg_video_codec"] = _default_overlay_video_codec()

            perf_cfg = self._data.setdefault("performance", {})
            perf_cfg["dtype"] = "float32"
            perf_cfg["use_cuda_graph"] = False
            perf_cfg["use_cudnn_benchmark"] = False

            taichi_cfg = self._data.setdefault("taichi", {})
            taichi_cfg["arch"] = _default_taichi_arch()

        nemo_cfg = ((self._data.get("asr", {}) or {}).get("nemo_msdd", {}) or {})
        if not isinstance(nemo_cfg, dict):
            nemo_cfg = {}
        if str(nemo_cfg.get("device", "") or "").strip() == "":
            nemo_cfg["device"] = "auto"

        nemo_defaults = {
            "infer_batch_size": 25,
            "seq_eval_mode": False,
            "split_infer": True,
            "diar_window_length": 50,
            "overlap_infer_spk_limit": 5,
            "use_speaker_model_from_ckpt": True,
            "preload_models": True,
            "download_device": "cpu",
            "download_retries": 4,
            "download_retry_wait_sec": 2.0,
            "num_workers": 0 if os.name == "nt" else 1,
            "model_sources": {},
            "sortformer": {
                "enabled": True,
                "model_name": "nvidia/diar_streaming_sortformer_4spk-v2.1",
                "batch_size": 1,
                "device": "auto",
                "hf_token": "",
                "auto_endpoint_probe": False,
                "mirror_endpoint": "https://hf-mirror.com",
                "official_endpoint": "https://huggingface.co",
                "mirror_retries": 3,
                "official_retries": 2,
                "download_retry_wait_sec": 1.5,
                "chunk_len": 340,
                "chunk_right_context": 40,
                "fifo_len": 40,
                "spkcache_update_period": 300,
                "spkcache_len": 188,
                "merge_gap_sec": 0.08,
                "min_turn_sec": 0.05,
            },
            "hybrid_fusion": {
                "enabled": True,
                "boundary_snap_sec": 0.18,
                "min_piece_sec": 0.20,
                "pyannote_weight": 1.25,
                "msdd_weight": 1.0,
                "sortformer_weight": 0.55,
                "map_min_overlap_ratio": 0.62,
                "map_min_overlap_sec": 1.2,
                "map_min_margin_ratio": 1.25,
                "count_longform_preserve_msdd_sec": 480.0,
                "count_msdd_pyannote_delta": 1,
                "merge_weak_speakers": True,
                "overlap_union": {
                    "enabled": True,
                    "merge_gap_sec": 0.08,
                    "min_region_sec": 0.20,
                },
                "posterior_decoder": {
                    "enabled": True,
                    "frame_hop_sec": 0.08,
                    "min_turn_sec": 0.20,
                    "switch_penalty": 0.42,
                    "stay_bonus": 0.16,
                    "boundary_relief": 0.28,
                    "use_native_viterbi": True,
                    "sortformer_saturation_scale": 0.65,
                    "global_prior": {
                        "enabled": True,
                        "emission_scale": 0.18,
                        "agreement_bonus": 0.14,
                        "exclusive_bonus": 0.22,
                        "anchor_bonus": 0.16,
                        "unanchored_penalty": 0.12,
                        "sortformer_cap_penalty": 0.28,
                    },
                    "overlap_decode": {
                        "enabled": True,
                        "primary_min_prob": 0.48,
                        "secondary_min_prob": 0.38,
                        "frame_overlap_gate": 0.34,
                        "agreement_gate": 0.34,
                        "seed_bonus": 0.08,
                        "max_active_speakers": 2,
                        "min_overlap_sec": 0.16,
                        "merge_gap_sec": 0.08,
                    },
                    "dump_examples": {
                        "enabled": False,
                        "output_dir": "output_files/posterior_fusion_examples",
                    },
                    "trainer": {
                        "examples_dir": "output_files/posterior_fusion_examples",
                        "rttm_dir": "output_files/posterior_fusion_rttm",
                        "output_path": "output_files/.posterior_fusion_calibrator.json",
                        "epochs": 6,
                        "threshold_min": 0.20,
                        "threshold_max": 0.75,
                        "threshold_num": 12,
                        "primary_loss_candidates": [0.85, 1.0, 1.20],
                        "activity_loss_candidates": [0.90, 1.0, 1.15],
                        "overlap_loss_candidates": [1.0, 1.35, 1.70],
                        "objective_weights": {
                            "primary": 1.0,
                            "activity": 1.0,
                            "overlap": 1.35,
                        },
                    },
                    "calibrator": {
                        "enabled": True,
                        "persist_path": "output_files/.posterior_fusion_calibrator.json",
                        "online_learning_rate": 0.03,
                        "l2_reg": 0.002,
                        "min_pseudo_margin": 0.12,
                        "min_pseudo_models": 2,
                        "activity_threshold": 0.46,
                        "activity_bias": -1.05,
                        "feature_weights": {
                            "msdd": 1.20,
                            "pyannote": 1.35,
                            "sortformer": 0.72,
                            "msdd_pyannote_agree": 0.70,
                            "msdd_sortformer_agree": 0.42,
                            "pyannote_sortformer_agree": 0.50,
                            "agreement_ratio": 0.35,
                            "boundary_support": 0.22,
                            "exclusive_support": 0.32,
                            "anchor_support": 0.18,
                        },
                        "activity_feature_weights": {
                            "msdd": 1.10,
                            "pyannote": 1.18,
                            "sortformer": 0.58,
                            "msdd_pyannote_agree": 0.82,
                            "msdd_sortformer_agree": 0.48,
                            "pyannote_sortformer_agree": 0.58,
                            "agreement_ratio": 0.52,
                            "boundary_support": -0.08,
                            "exclusive_support": 0.18,
                            "anchor_support": 0.10,
                        },
                    },
                },
            },
            "pyannote_fallback": {
                "enabled": True,
                "provider": "pyannote.audio",
                "model_name": _default_pyannote_diar_model_name(),
                "model_candidates": _default_pyannote_diar_model_candidates(),
                "prefer_local_snapshot": _default_pyannote_prefer_local_snapshot(),
                "trust_remote_code": True,
                "hf_token": "",
                "device": "mps" if sys.platform == "darwin" else "auto",
                "require_gpu": bool(sys.platform == "darwin"),
                "auto_endpoint_probe": False,
                "mirror_endpoint": "https://hf-mirror.com",
                "official_endpoint": "https://huggingface.co",
                "mirror_retries": 3,
                "official_retries": 2,
                "download_retry_wait_sec": 1.5,
                "merge_gap_sec": 0.08,
                "min_turn_sec": 0.05,
                "num_speakers": 0,
                "min_speakers": 1,
                "max_speakers": 8,
                "specialize_longform": True,
                "specialize_min_view_sec": 0.8,
                "specialize_max_view_sec": 6.0,
                "specialize_view_sec": 1.8,
                "specialize_max_views": 3,
                "specialize_batch_size": 32 if sys.platform == "darwin" else 24,
                "specialize_short_turn_sec": 1.8,
                "specialize_merge_similarity": 0.91,
                "specialize_short_merge_similarity": 0.86,
                "specialize_attach_similarity": 0.84,
                "specialize_dense_turn_threshold": 240,
                "specialize_dense_max_views": 2,
                "specialize_dense_view_sec": 1.25,
                "specialize_ultra_dense_turn_threshold": 420,
                "specialize_ultra_dense_max_views": 1,
                "specialize_ultra_dense_view_sec": 0.95,
            },
            "overlap_handling": {
                "enabled": True,
                "min_region_sec": 0.25,
                "window_pad_sec": 0.12,
                "max_region_sec": 20.0,
                "max_regions": 120,
                "min_segment_overlap_ratio": 0.15,
                "osd": {
                    "enabled": True,
                    "provider": "pyannote",
                    "model_name": "pyannote/overlapped-speech-detection",
                    "trust_remote_code": True,
                    "hf_token": "",
                    "device": "auto",
                    "auto_endpoint_probe": False,
                    "mirror_endpoint": "https://hf-mirror.com",
                    "official_endpoint": "https://huggingface.co",
                    "mirror_retries": 3,
                    "official_retries": 2,
                    "download_retry_wait_sec": 1.5,
                    "pad_sec": 0.10,
                    "merge_gap_sec": 0.08,
                },
                "separation": {
                    "enabled": True,
                    "provider": "auto",
                    "prefer_pyannote": _default_overlap_prefer_pyannote(),
                    "prefer_local_snapshot": _default_pyannote_prefer_local_snapshot(),
                    "task": "speech_separation",
                    "model_name": "MossFormer2_SS_16K",
                    "pyannote_model_name": "pyannote/speech-separation-ami-1.0",
                    "trust_remote_code": True,
                    "hf_token": "",
                    "device": "auto",
                    "auto_endpoint_probe": False,
                    "mirror_endpoint": "https://hf-mirror.com",
                    "official_endpoint": "https://huggingface.co",
                    "mirror_retries": 3,
                    "official_retries": 2,
                    "download_retry_wait_sec": 1.5,
                    "num_speakers": 2,
                    "max_streams": 2,
                    "min_stream_peak": 1e-4,
                    "allow_mixed_audio_fallback": False,
                },
                "merge": {
                    "drop_original_overlap": False,
                    "drop_if_overlap_ratio_ge": 0.75,
                },
            },
        }
        for key, value in nemo_defaults.items():
            if key not in nemo_cfg or nemo_cfg[key] is None:
                nemo_cfg[key] = copy.deepcopy(value)
            elif isinstance(value, dict) and isinstance(nemo_cfg.get(key), dict):
                nemo_cfg[key] = _deep_merge(value, nemo_cfg[key])

        if sys.platform == "darwin":
            nemo_cfg["device"] = str(nemo_cfg.get("device", "") or "").strip() or "auto"
            nemo_cfg["download_device"] = "cpu"
            nemo_cfg["num_workers"] = 0
            sortformer_cfg = nemo_cfg.setdefault("sortformer", {})
            sortformer_cfg["device"] = str(sortformer_cfg.get("device", "") or "").strip() or "auto"
            pyannote_cfg = nemo_cfg.setdefault("pyannote_fallback", {})
            pyannote_cfg["device"] = str(pyannote_cfg.get("device", "") or "").strip() or "mps"
            pyannote_cfg.setdefault("require_gpu", True)
            overlap_cfg = nemo_cfg.setdefault("overlap_handling", {})
            overlap_cfg.setdefault("osd", {})["device"] = (
                str(overlap_cfg.get("osd", {}).get("device", "") or "").strip() or "auto"
            )
            overlap_cfg.setdefault("separation", {})["device"] = (
                str(overlap_cfg.get("separation", {}).get("device", "") or "").strip() or "auto"
            )

        pyannote_cfg = nemo_cfg.setdefault("pyannote_fallback", {})
        preferred_pyannote_model = _default_pyannote_diar_model_name()
        configured_pyannote_model = str(pyannote_cfg.get("model_name", "") or "").strip()
        configured_candidates = pyannote_cfg.get("model_candidates", [])
        candidate_values = []
        if isinstance(configured_candidates, list):
            candidate_values = [str(item or "").strip() for item in configured_candidates]
        elif isinstance(configured_candidates, tuple):
            candidate_values = [str(item or "").strip() for item in configured_candidates]
        legacy_candidate_orders = {
            (),
            (
                "pyannote/speaker-diarization-community-1",
                "pyannote/speaker-diarization-3.1",
            ),
            (
                "pyannote/speaker-diarization-3.1",
                "pyannote/speaker-diarization-community-1",
            ),
        }
        candidate_values_tuple = tuple(candidate_values)
        if configured_pyannote_model in {
            "",
            "pyannote/speaker-diarization-community-1",
        } or (
            configured_pyannote_model == "pyannote/speaker-diarization-3.1"
            and candidate_values_tuple in legacy_candidate_orders
        ):
            pyannote_cfg["model_name"] = preferred_pyannote_model
        if candidate_values_tuple in legacy_candidate_orders:
            pyannote_cfg["model_candidates"] = _default_pyannote_diar_model_candidates()
        overlap_sep_cfg = nemo_cfg.setdefault("overlap_handling", {}).setdefault("separation", {})
        prefer_pyannote_raw = overlap_sep_cfg.get(
            "prefer_pyannote",
            _default_overlap_prefer_pyannote(),
        )
        if isinstance(prefer_pyannote_raw, str):
            overlap_sep_cfg["prefer_pyannote"] = prefer_pyannote_raw.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            overlap_sep_cfg["prefer_pyannote"] = bool(prefer_pyannote_raw)

        self._data.setdefault("asr", {})["nemo_msdd"] = nemo_cfg

        translation_cfg = self._data.get("translation", {})
        if not isinstance(translation_cfg, dict):
            translation_cfg = {}
        translation_defaults = {
            "enabled": False,
            "target_language": "zh",
            "max_concurrent": 8,
            "skip_same_language": True,
        }
        for key, value in translation_defaults.items():
            if key not in translation_cfg:
                translation_cfg[key] = copy.deepcopy(value)

        target_language = str(
            translation_cfg.get("target_language", "zh") or "zh"
        ).strip().lower().replace("_", "-")
        if not target_language:
            target_language = "zh"
        translation_cfg["target_language"] = target_language

        try:
            max_concurrent = int(translation_cfg.get("max_concurrent", 8))
        except (TypeError, ValueError):
            max_concurrent = 8
        translation_cfg["max_concurrent"] = max(1, min(64, max_concurrent))
        translation_cfg["enabled"] = bool(translation_cfg.get("enabled", False))
        translation_cfg["skip_same_language"] = bool(
            translation_cfg.get("skip_same_language", True)
        )
        self._data["translation"] = translation_cfg

        llm_cfg = self._data.get("llm", {})
        if not isinstance(llm_cfg, dict):
            llm_cfg = {}
        llm_defaults = copy.deepcopy(DEFAULT_CONFIG.get("llm", {}) or {})
        llm_cfg = _deep_merge(llm_defaults, llm_cfg)
        llm_cfg["enabled"] = bool(llm_cfg.get("enabled", False))

        try:
            llm_max_concurrent = int(llm_cfg.get("max_concurrent", 8))
        except (TypeError, ValueError):
            llm_max_concurrent = 8
        llm_cfg["max_concurrent"] = max(1, min(64, llm_max_concurrent))

        optimize_cfg = llm_cfg.get("optimize_language", {})
        if not isinstance(optimize_cfg, dict):
            optimize_cfg = {}
        optimize_defaults = (llm_defaults.get("optimize_language", {}) or {})
        optimize_cfg = _deep_merge(optimize_defaults, optimize_cfg)
        optimize_cfg["enabled"] = bool(optimize_cfg.get("enabled", False))
        optimize_cfg["strict_keep_text"] = bool(
            optimize_cfg.get("strict_keep_text", True)
        )
        optimize_cfg["infer_missing_words"] = bool(
            optimize_cfg.get("infer_missing_words", True)
        )
        try:
            optimize_max_concurrent = int(
                optimize_cfg.get("max_concurrent", llm_cfg["max_concurrent"])
            )
        except (TypeError, ValueError):
            optimize_max_concurrent = llm_cfg["max_concurrent"]
        optimize_cfg["max_concurrent"] = max(1, min(64, optimize_max_concurrent))
        llm_cfg["optimize_language"] = optimize_cfg

        speaker_arb_cfg = llm_cfg.get("speaker_arbitration", {})
        if not isinstance(speaker_arb_cfg, dict):
            speaker_arb_cfg = {}
        speaker_arb_defaults = (llm_defaults.get("speaker_arbitration", {}) or {})
        speaker_arb_cfg = _deep_merge(speaker_arb_defaults, speaker_arb_cfg)
        speaker_arb_cfg["enabled"] = bool(speaker_arb_cfg.get("enabled", False))
        try:
            speaker_arb_cfg["window_chars"] = max(
                1500,
                min(32000, int(speaker_arb_cfg.get("window_chars", 9000))),
            )
        except (TypeError, ValueError):
            speaker_arb_cfg["window_chars"] = 9000
        try:
            speaker_arb_cfg["max_segments_per_window"] = max(
                6,
                min(120, int(speaker_arb_cfg.get("max_segments_per_window", 36))),
            )
        except (TypeError, ValueError):
            speaker_arb_cfg["max_segments_per_window"] = 36
        try:
            speaker_arb_cfg["window_overlap_segments"] = max(
                0,
                min(20, int(speaker_arb_cfg.get("window_overlap_segments", 4))),
            )
        except (TypeError, ValueError):
            speaker_arb_cfg["window_overlap_segments"] = 4
        try:
            speaker_arb_cfg["min_confidence"] = max(
                0.0,
                min(1.0, float(speaker_arb_cfg.get("min_confidence", 0.62))),
            )
        except (TypeError, ValueError):
            speaker_arb_cfg["min_confidence"] = 0.62
        try:
            speaker_arb_cfg["max_changed_ratio"] = max(
                0.0,
                min(1.0, float(speaker_arb_cfg.get("max_changed_ratio", 0.40))),
            )
        except (TypeError, ValueError):
            speaker_arb_cfg["max_changed_ratio"] = 0.40
        try:
            speaker_arb_cfg["max_highlight_terms"] = max(
                1,
                min(8, int(speaker_arb_cfg.get("max_highlight_terms", 3))),
            )
        except (TypeError, ValueError):
            speaker_arb_cfg["max_highlight_terms"] = 3
        llm_cfg["speaker_arbitration"] = speaker_arb_cfg

        self._data["llm"] = llm_cfg

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Access config via dotted key: config.get('asr.faster_whisper.model_size')"""
        keys = dotted_key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"Config({yaml.dump(self._data, default_flow_style=False)[:200]}...)"
