"""
llm_processor.py - LLM分析处理器（不修改原文）

功能：
  - 自动语言检测（轻量规则优先，LLM兜底）
  - 会议摘要 / 要点提取 / 待办事项
  - 高并发异步处理
  - 关闭思考模式

原则：绝不修改、润色、替换任何转写原文
"""

import asyncio
import json
import hashlib
import logging
import os
import re
import time
import threading
import concurrent.futures
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from speaker_semantic_arbiter import (
    SpeakerArbitrationResult,
    SpeakerProfile,
    SpeakerSemanticArbiter,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysis:
    """LLM分析结果（不含润色）"""
    language: str = "zh"
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProcessor:
    """
    LLM分析处理器

    重要：此模块绝不修改任何转写原文。
    仅用于：
      1. 语言检测
      2. 生成会议摘要
      3. 提取关键要点
      4. 提取待办事项
    """

    def __init__(self, config):
        self.config = config
        self.llm_cfg = config.get("llm", {})

        self.api_key = ""
        self.api_key_source = ""
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = "qwen3.5-plus"
        self.model_candidates: List[str] = []
        self.max_concurrent = 8
        self.provider_max_concurrent = 2
        self.timeout = 60
        self.max_retries = 3
        self.retry_delay = 1.0
        self.server_error_cooldown_sec = 2.0
        self.temperature = 0.3
        self.max_tokens = 4096
        self.enable_thinking = False
        self.summary_prompt = ""
        self.lang_detect_prompt = ""
        self.hierarchical_summary_cfg: Dict[str, Any] = {}
        self.hierarchical_summary_enabled = True
        self.analysis_chunk_target_chars = 8000
        self.analysis_single_pass_chars = 6000
        self.analysis_chunk_overlap_chars = 400
        self.analysis_reduce_target_chars = 10000
        self.analysis_max_chunks = 32
        self.segment_request_max_chars = 4000
        self.request_payload_warn_chars = 12000
        self.request_payload_hard_chars = 18000
        self.disabled_reason = ""
        self.analysis_enabled = False
        self.translation_cfg: Dict[str, Any] = {}
        self.translation_enabled = False
        self.translation_target_language = "zh"
        self.translation_skip_same_language = True
        self.translation_max_concurrent = 8
        self.optimize_cfg: Dict[str, Any] = {}
        self.optimize_enabled = False
        self.optimize_strict_keep_text = True
        self.optimize_infer_missing_words = True
        self.optimize_max_concurrent = 8
        self.optimize_batch_target_chars = 6000
        self.optimize_batch_max_segments = 12
        self.optimize_batch_min_segments = 3
        self.speaker_arbitration_cfg: Dict[str, Any] = {}
        self.speaker_arbitration_enabled = False
        self.speaker_arbitration_min_confidence = 0.62
        self.speaker_arbitration_max_changed_ratio = 0.40
        self.speaker_arbitration_enable_segment_merge = True
        self.speaker_arbitration_merge_min_confidence = 0.68
        self.speaker_arbitration_max_merge_group_size = 4
        self.speaker_arbitration_max_merge_gap_sec = 0.90
        self.speaker_arbitration_max_merged_duration_sec = 10.0
        self.speaker_arbiter: Optional[SpeakerSemanticArbiter] = None

        # 统计（_failed 表示连续失败次数）
        self._total_requests = 0
        self._successful = 0
        self._failed = 0
        self._total_tokens = 0

        # 后台事件循环
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._loop_ready = threading.Event()
        self._analysis_semaphore: Optional[asyncio.Semaphore] = None
        self._translation_semaphore: Optional[asyncio.Semaphore] = None
        self._optimize_semaphore: Optional[asyncio.Semaphore] = None
        self._provider_semaphore: Optional[asyncio.Semaphore] = None
        self._provider_cooldown_until = 0.0

        # 流水线任务存储
        self._stats_lock = threading.Lock()
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

        self.analysis_enabled = bool(self.llm_cfg.get("enabled", False))
        raw_translation_cfg = self.config.get("translation", {}) or {}
        self.translation_cfg = (
            dict(raw_translation_cfg) if isinstance(raw_translation_cfg, dict) else {}
        )
        self.translation_enabled = bool(self.translation_cfg.get("enabled", False))
        self.translation_target_language = self._normalize_language_code(
            str(self.translation_cfg.get("target_language", "zh") or "zh")
        )
        if not self.translation_target_language:
            self.translation_target_language = "zh"
        self.translation_skip_same_language = bool(
            self.translation_cfg.get("skip_same_language", True)
        )
        try:
            self.translation_max_concurrent = max(
                1,
                int(self.translation_cfg.get("max_concurrent", 8)),
            )
        except (TypeError, ValueError):
            self.translation_max_concurrent = 8

        raw_optimize_cfg = self.llm_cfg.get("optimize_language", {}) or {}
        self.optimize_cfg = (
            dict(raw_optimize_cfg) if isinstance(raw_optimize_cfg, dict) else {}
        )
        self.optimize_enabled = bool(self.optimize_cfg.get("enabled", False))
        self.optimize_strict_keep_text = bool(
            self.optimize_cfg.get("strict_keep_text", True)
        )
        self.optimize_infer_missing_words = bool(
            self.optimize_cfg.get("infer_missing_words", True)
        )
        try:
            self.optimize_max_concurrent = max(
                1,
                int(
                    self.optimize_cfg.get(
                        "max_concurrent",
                        self.llm_cfg.get("max_concurrent", 8),
                    )
                ),
            )
        except (TypeError, ValueError):
            self.optimize_max_concurrent = 8

        raw_speaker_arb_cfg = self.llm_cfg.get("speaker_arbitration", {}) or {}
        self.speaker_arbitration_cfg = (
            dict(raw_speaker_arb_cfg) if isinstance(raw_speaker_arb_cfg, dict) else {}
        )
        self.speaker_arbitration_enabled = bool(
            self.speaker_arbitration_cfg.get("enabled", False)
        )
        try:
            self.speaker_arbitration_min_confidence = max(
                0.0,
                min(
                    1.0,
                    float(
                        self.speaker_arbitration_cfg.get(
                            "min_confidence",
                            0.62,
                        )
                    ),
                ),
            )
        except (TypeError, ValueError):
            self.speaker_arbitration_min_confidence = 0.62
        try:
            self.speaker_arbitration_max_changed_ratio = max(
                0.0,
                min(
                    1.0,
                    float(
                        self.speaker_arbitration_cfg.get(
                            "max_changed_ratio",
                            0.40,
                        )
                    ),
                ),
            )
        except (TypeError, ValueError):
            self.speaker_arbitration_max_changed_ratio = 0.40
        self.speaker_arbitration_enable_segment_merge = bool(
            self.speaker_arbitration_cfg.get("enable_segment_merge", True)
        )
        try:
            self.speaker_arbitration_merge_min_confidence = max(
                0.0,
                min(
                    1.0,
                    float(
                        self.speaker_arbitration_cfg.get(
                            "merge_min_confidence",
                            0.68,
                        )
                    ),
                ),
            )
        except (TypeError, ValueError):
            self.speaker_arbitration_merge_min_confidence = 0.68
        try:
            self.speaker_arbitration_max_merge_group_size = max(
                2,
                int(
                    self.speaker_arbitration_cfg.get(
                        "max_merge_group_size",
                        4,
                    )
                ),
            )
        except (TypeError, ValueError):
            self.speaker_arbitration_max_merge_group_size = 4
        try:
            self.speaker_arbitration_max_merge_gap_sec = max(
                0.0,
                float(
                    self.speaker_arbitration_cfg.get(
                        "max_merge_gap_sec",
                        0.90,
                    )
                ),
            )
        except (TypeError, ValueError):
            self.speaker_arbitration_max_merge_gap_sec = 0.90
        try:
            self.speaker_arbitration_max_merged_duration_sec = max(
                0.5,
                float(
                    self.speaker_arbitration_cfg.get(
                        "max_merged_duration_sec",
                        10.0,
                    )
                ),
            )
        except (TypeError, ValueError):
            self.speaker_arbitration_max_merged_duration_sec = 10.0

        self.enabled = bool(
            self.analysis_enabled
            or self.translation_enabled
            or self.optimize_enabled
            or self.speaker_arbitration_enabled
        )
        if not self.enabled:
            self.disabled_reason = (
                "llm.enabled=false and translation.enabled=false "
                "and llm.optimize_language.enabled=false "
                "and llm.speaker_arbitration.enabled=false"
            )
            logger.info(
                "LLM processor disabled by config "
                "(llm.enabled=false and translation.enabled=false "
                "and llm.optimize_language.enabled=false "
                "and llm.speaker_arbitration.enabled=false)"
            )
            return

        # Robust key resolution: env first, then config hardcoded value.
        self.api_key, self.api_key_source = self._resolve_api_key()
        if self.api_key and self.api_key_source == "config":
            os.environ.setdefault("DASHSCOPE_API_KEY", self.api_key)
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)

        self.base_url = self.llm_cfg.get(
            "base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = self.llm_cfg.get("model", "qwen3.5-plus")
        try:
            self.max_concurrent = max(1, int(self.llm_cfg.get("max_concurrent", 8)))
        except (TypeError, ValueError):
            self.max_concurrent = 8
        self.model_candidates = self._build_model_candidates()
        try:
            self.provider_max_concurrent = max(
                1,
                int(
                    self.llm_cfg.get(
                        "provider_max_concurrent",
                        self._default_provider_max_concurrent(self.max_concurrent),
                    )
                ),
            )
        except (TypeError, ValueError):
            self.provider_max_concurrent = self._default_provider_max_concurrent(
                self.max_concurrent
            )
        if self.translation_max_concurrent <= 0:
            self.translation_max_concurrent = self.max_concurrent
        if self.optimize_max_concurrent <= 0:
            self.optimize_max_concurrent = self.max_concurrent
        self.timeout = self.llm_cfg.get("timeout", 60)
        self.max_retries = self.llm_cfg.get("max_retries", 3)
        self.retry_delay = self.llm_cfg.get("retry_delay", 1.0)
        self.server_error_cooldown_sec = float(
            self.llm_cfg.get("server_error_cooldown_sec", 2.0) or 2.0
        )
        self.temperature = self.llm_cfg.get("temperature", 0.3)
        self.max_tokens = self.llm_cfg.get("max_tokens", 4096)
        self.enable_thinking = self.llm_cfg.get("enable_thinking", False)

        self.summary_prompt = self.llm_cfg.get("summary_prompt", "")
        self.lang_detect_prompt = self.llm_cfg.get("lang_detect_prompt", "")
        raw_hier_cfg = self.llm_cfg.get("hierarchical_summary", {}) or {}
        self.hierarchical_summary_cfg = (
            dict(raw_hier_cfg) if isinstance(raw_hier_cfg, dict) else {}
        )
        raw_request_limits = self.llm_cfg.get("request_limits", {}) or {}
        self.request_limits_cfg = (
            dict(raw_request_limits) if isinstance(raw_request_limits, dict) else {}
        )
        self.hierarchical_summary_enabled = bool(
            self.hierarchical_summary_cfg.get("enabled", True)
        )
        try:
            self.analysis_chunk_target_chars = max(
                2000,
                int(self.hierarchical_summary_cfg.get("chunk_target_chars", 8000)),
            )
        except (TypeError, ValueError):
            self.analysis_chunk_target_chars = 8000
        try:
            self.analysis_single_pass_chars = max(
                2000,
                int(self.hierarchical_summary_cfg.get("single_pass_chars", 6000)),
            )
        except (TypeError, ValueError):
            self.analysis_single_pass_chars = 6000
        try:
            self.analysis_chunk_overlap_chars = max(
                0,
                int(self.hierarchical_summary_cfg.get("chunk_overlap_chars", 400)),
            )
        except (TypeError, ValueError):
            self.analysis_chunk_overlap_chars = 400
        try:
            self.analysis_reduce_target_chars = max(
                2000,
                int(self.hierarchical_summary_cfg.get("reduce_target_chars", 10000)),
            )
        except (TypeError, ValueError):
            self.analysis_reduce_target_chars = 10000
        try:
            self.analysis_max_chunks = max(
                1,
                int(self.hierarchical_summary_cfg.get("max_chunks", 32)),
            )
        except (TypeError, ValueError):
            self.analysis_max_chunks = 32
        try:
            self.segment_request_max_chars = max(
                1000,
                int(self.request_limits_cfg.get("segment_text_chars", 4000)),
            )
        except (TypeError, ValueError):
            self.segment_request_max_chars = 4000
        try:
            self.request_payload_warn_chars = max(
                2000,
                int(self.request_limits_cfg.get("payload_warn_chars", 12000)),
            )
        except (TypeError, ValueError):
            self.request_payload_warn_chars = 12000
        try:
            self.request_payload_hard_chars = max(
                self.request_payload_warn_chars,
                int(self.request_limits_cfg.get("payload_hard_chars", 18000)),
            )
        except (TypeError, ValueError):
            self.request_payload_hard_chars = max(
                self.request_payload_warn_chars,
                18000,
            )
        try:
            default_batch_target = min(
                8000,
                max(2400, int(self.request_payload_warn_chars) - 3000),
            )
            self.optimize_batch_target_chars = max(
                1600,
                int(self.optimize_cfg.get("batch_target_chars", default_batch_target)),
            )
        except (TypeError, ValueError):
            self.optimize_batch_target_chars = min(
                8000,
                max(2400, int(self.request_payload_warn_chars) - 3000),
            )
        try:
            self.optimize_batch_max_segments = max(
                1,
                int(self.optimize_cfg.get("batch_max_segments", 12)),
            )
        except (TypeError, ValueError):
            self.optimize_batch_max_segments = 12
        try:
            self.optimize_batch_min_segments = max(
                2,
                int(self.optimize_cfg.get("batch_min_segments", 3)),
            )
        except (TypeError, ValueError):
            self.optimize_batch_min_segments = 3

        if not self.api_key:
            self.disabled_reason = "missing API key in env and config"
            logger.warning(
                "LLM API key not set, disabling LLM "
                "(set llm.api_key or DASHSCOPE_API_KEY/OPENAI_API_KEY)"
            )
            self.enabled = False
            self.analysis_enabled = False
            self.translation_enabled = False
            self.optimize_enabled = False
            self.speaker_arbitration_enabled = False
            return

        if self.speaker_arbitration_enabled and not self.analysis_enabled:
            self.speaker_arbitration_enabled = False
            logger.warning(
                "Speaker semantic arbitration requires llm.enabled=true; "
                "disabling llm.speaker_arbitration.enabled for this run."
            )

        self.enabled = bool(
            self.analysis_enabled
            or self.translation_enabled
            or self.optimize_enabled
            or self.speaker_arbitration_enabled
        )
        if not self.enabled:
            self.disabled_reason = "no_active_llm_mode"
            logger.info("LLM processor disabled: no active analysis/translation/optimization/arbitration mode.")
            return

        if self.speaker_arbitration_enabled:
            self.speaker_arbiter = SpeakerSemanticArbiter(
                self.config,
                llm_call=self._call_api,
                logger=logger,
            )

        logger.info(
            f"LLM Processor: model={self.model}, "
            f"concurrency={self.max_concurrent}, "
            f"provider_concurrency={self.provider_max_concurrent}, "
            f"candidates={','.join(self.model_candidates[:3])}, "
            f"default_thinking={'ON' if self.enable_thinking else 'OFF'}, "
            "analysis_thinking=ON, translation_thinking=OFF, "
            f"speaker_arbitration={'ON' if self.speaker_arbitration_enabled else 'OFF'}, "
            f"mode=ANALYSIS_ONLY (原文不修改), "
            f"api_key_source={self.api_key_source or 'unknown'}"
        )

    @staticmethod
    def _get_env_api_key() -> str:
        return (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()

    def _resolve_api_key(self):
        env_key = self._get_env_api_key()
        if env_key:
            return env_key, "env"

        cfg_key = str(self.llm_cfg.get("api_key") or "").strip()
        if cfg_key:
            return cfg_key, "config"

        return "", ""

    @staticmethod
    def _dedupe_strings(items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    def _is_dashscope_compatible_mode(self) -> bool:
        url = str(self.base_url or "").strip().lower()
        return "dashscope.aliyuncs.com" in url and "compatible-mode" in url

    def _default_provider_max_concurrent(self, fallback: int) -> int:
        if self._is_dashscope_compatible_mode():
            return max(1, min(int(fallback or 1), 2))
        return max(1, int(fallback or 1))

    def _build_model_candidates(self) -> List[str]:
        configured = self.llm_cfg.get("model_candidates", []) or []
        candidates: List[str] = []
        if isinstance(configured, (list, tuple)):
            candidates.extend([str(item or "").strip() for item in configured])
        elif str(configured or "").strip():
            candidates.append(str(configured).strip())
        candidates.insert(0, str(self.model or "").strip())
        if self._is_dashscope_compatible_mode():
            candidates.extend(["qwen-plus", "qwen-max"])
        return self._dedupe_strings(candidates)

    @staticmethod
    def _is_retryable_server_error(exc: Exception) -> bool:
        text = str(exc or "").strip().lower()
        if not text:
            return False
        markers = (
            "error code: 500",
            "status code: 500",
            "http/1.1 500",
            "internal server error",
            "internalerror.algo",
            "<503>",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "error code: 502",
            "error code: 503",
            "error code: 504",
        )
        return any(marker in text for marker in markers)

    def _pick_model_for_attempt(
        self,
        attempt: int,
        last_error: Optional[Exception],
    ) -> str:
        retryable_server = self._is_retryable_server_error(last_error) if last_error else False
        if not self.model_candidates:
            return str(self.model or "qwen3.5-plus")
        if attempt <= 1 or not retryable_server:
            return self.model_candidates[0]
        idx = min(max(1, attempt - 1), len(self.model_candidates) - 1)
        return self.model_candidates[idx]

    def _build_request_params(
        self,
        messages: List[Dict[str, Any]],
        *,
        model_name: str,
        thinking: bool,
        conservative: bool = False,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "temperature": 0.0 if conservative else self.temperature,
            "max_tokens": min(int(self.max_tokens), 2048) if conservative else int(self.max_tokens),
        }
        if not conservative:
            params["extra_body"] = {"enable_thinking": bool(thinking)}
        return params

    @staticmethod
    def _messages_char_count(messages: List[Dict[str, Any]]) -> int:
        total = 0
        for message in messages or []:
            content = message.get("content", "")
            if isinstance(content, str):
                total += len(content)
                continue
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        total += len(str(item.get("text", "")))
                    else:
                        total += len(str(item))
                continue
            total += len(str(content))
        return total

    @staticmethod
    def _find_soft_split_index(text: str, target_chars: int) -> int:
        source = str(text or "")
        if not source:
            return 0
        if len(source) <= target_chars:
            return len(source)

        window = min(240, max(32, target_chars // 6))
        start = max(1, target_chars - window)
        stop = min(len(source), target_chars + window)
        for marker in ("\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " "):
            idx = source.rfind(marker, start, stop)
            if idx > 0:
                return idx + 1
        return max(1, min(len(source), target_chars))

    def _split_text_for_request(
        self,
        text: str,
        *,
        max_chars: int,
    ) -> List[str]:
        source = str(text or "").strip()
        if not source:
            return []
        if max_chars <= 0 or len(source) <= max_chars:
            return [source]

        units = [line.strip() for line in source.splitlines() if line.strip()]
        if len(units) <= 1:
            units = [part.strip() for part in re.split(r"(?<=[。！？!?；;])", source) if part.strip()]
        if len(units) <= 1:
            units = [part.strip() for part in re.split(r"(?<=[，,])", source) if part.strip()]
        if len(units) <= 1:
            units = [source]

        chunks: List[str] = []
        current = ""
        for unit in units:
            piece = str(unit or "").strip()
            if not piece:
                continue
            while len(piece) > max_chars:
                split_at = self._find_soft_split_index(piece, max_chars)
                head = piece[:split_at].strip()
                if head:
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(head)
                piece = piece[split_at:].strip()
            if not piece:
                continue
            candidate = piece if not current else f"{current}\n{piece}"
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate

        if current:
            chunks.append(current)
        return chunks or [source]

    @staticmethod
    def _clone_analysis(analysis: LLMAnalysis) -> LLMAnalysis:
        return LLMAnalysis(
            language=str(getattr(analysis, "language", "") or "zh"),
            summary=str(getattr(analysis, "summary", "") or ""),
            key_points=list(getattr(analysis, "key_points", []) or []),
            action_items=list(getattr(analysis, "action_items", []) or []),
            topics=list(getattr(analysis, "topics", []) or []),
            error=str(getattr(analysis, "error", "") or ""),
            metadata=dict(getattr(analysis, "metadata", {}) or {}),
        )

    @staticmethod
    def _normalize_language_code(language: str) -> str:
        return (language or "").strip().lower().replace("_", "-")

    def _record_request_success(self, total_tokens: int = 0) -> int:
        with self._stats_lock:
            recovered_after = self._failed
            self._total_requests += 1
            self._successful += 1
            self._failed = 0
            if total_tokens > 0:
                self._total_tokens += int(total_tokens)
        return recovered_after

    def _record_request_failure(self) -> int:
        with self._stats_lock:
            self._total_requests += 1
            self._failed += 1
            return self._failed

    def _snapshot_request_stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return {
                "total_requests": int(self._total_requests),
                "successful": int(self._successful),
                "failed": int(self._failed),
                "total_tokens": int(self._total_tokens),
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 后台事件循环
    # ═══════════════════════════════════════════════════════════════════════

    def start(self):
        """启动后台异步事件循环"""
        if not self.enabled or self._started:
            return

        self._loop_ready.clear()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="LLM-Loop",
        )
        self._thread.start()
        if not self._loop_ready.wait(timeout=5):
            logger.error("LLM event loop failed to start within timeout")
            return
        self._started = True
        logger.info("LLM background loop started")

    def _run_loop(self):
        """后台线程：设置并运行事件循环"""
        try:
            asyncio.set_event_loop(self._loop)
            self._analysis_semaphore = asyncio.Semaphore(self.max_concurrent)
            self._translation_semaphore = asyncio.Semaphore(
                max(1, int(self.translation_max_concurrent))
            )
            self._optimize_semaphore = asyncio.Semaphore(
                max(1, int(self.optimize_max_concurrent))
            )
            self._provider_semaphore = asyncio.Semaphore(
                max(1, int(self.provider_max_concurrent))
            )
            self._loop_ready.set()
            self._loop.run_forever()
        finally:
            self._analysis_semaphore = None
            self._translation_semaphore = None
            self._optimize_semaphore = None
            self._provider_semaphore = None
            self._loop_ready.clear()

    def stop(self):
        """停止后台事件循环"""
        if not self.enabled and not self._started and self._loop is None:
            logger.info(
                "LLM stop skipped: disabled "
                f"({self.disabled_reason or 'llm.enabled=false'})"
            )
            return

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._started = False
        self._loop = None
        self._provider_cooldown_until = 0.0
        with self._lock:
            self._jobs.clear()
        stats = self._snapshot_request_stats()
        logger.info(
            "LLM stopped: %d/%d OK, consecutive_failed=%d, ~%d tokens",
            stats["successful"],
            stats["total_requests"],
            stats["failed"],
            stats["total_tokens"],
        )

    # ═══════════════════════════════════════════════════════════════════════
    # API 调用（关闭思考模式）
    # ═══════════════════════════════════════════════════════════════════════

    async def _call_api(
        self,
        messages: List[Dict],
        enable_thinking: Optional[bool] = None,
        request_label: str = "",
    ) -> str:
        """调用Qwen API，关闭思考模式"""
        payload_chars = self._messages_char_count(messages)
        request_name = str(request_label or "unnamed").strip() or "unnamed"
        if payload_chars >= self.request_payload_warn_chars:
            logger.info(
                "LLM request payload: label=%s chars=%d warn=%d hard=%d",
                request_name,
                payload_chars,
                self.request_payload_warn_chars,
                self.request_payload_hard_chars,
            )
        else:
            logger.debug(
                "LLM request payload: label=%s chars=%d",
                request_name,
                payload_chars,
            )
        if payload_chars > self.request_payload_hard_chars:
            logger.warning(
                "LLM request skipped: payload too large (label=%s, chars=%d, hard=%d)",
                request_name,
                payload_chars,
                self.request_payload_hard_chars,
            )
            return ""

        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.warning("openai not installed, pip install openai")
            return ""

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
        )

        thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        label_suffix = f" [{request_label}]" if str(request_label or "").strip() else ""
        total_attempts = self.max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            model_name = self._pick_model_for_attempt(attempt, last_error)
            conservative = attempt > 0 and self._is_retryable_server_error(last_error)
            params = self._build_request_params(
                messages,
                model_name=model_name,
                thinking=thinking,
                conservative=conservative,
            )
            try:
                cooldown_until = float(self._provider_cooldown_until or 0.0)
                now = time.time()
                if cooldown_until > now:
                    await asyncio.sleep(cooldown_until - now)

                if self._provider_semaphore is not None:
                    async with self._provider_semaphore:
                        resp = await asyncio.wait_for(
                            client.chat.completions.create(**params),
                            timeout=self.timeout,
                        )
                else:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(**params),
                        timeout=self.timeout,
                    )
                usage_tokens = 0
                if resp.usage:
                    usage_tokens = int(getattr(resp.usage, "total_tokens", 0) or 0)
                recovered_after = self._record_request_success(total_tokens=usage_tokens)
                self._provider_cooldown_until = 0.0

                result = ""
                if resp.choices:
                    content = resp.choices[0].message.content
                    if isinstance(content, str):
                        result = content.strip()
                    elif isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict):
                                parts.append(str(item.get("text", "")))
                            else:
                                parts.append(str(item))
                        result = "".join(parts).strip()
                await client.close()
                if recovered_after > 0:
                    logger.info(
                        "LLM request recovered%s after %d consecutive failed requests",
                        label_suffix,
                        recovered_after,
                    )
                return result

            except asyncio.TimeoutError:
                logger.warning(
                    "LLM timeout%s (attempt %d/%d, model=%s)",
                    label_suffix,
                    attempt + 1,
                    total_attempts,
                    model_name,
                )
                last_error = None
            except Exception as e:
                last_error = e
                if self._is_retryable_server_error(e):
                    cooldown = max(
                        float(self.server_error_cooldown_sec),
                        float(self.retry_delay) * (2 ** attempt),
                    )
                    self._provider_cooldown_until = max(
                        float(self._provider_cooldown_until or 0.0),
                        time.time() + min(12.0, cooldown),
                    )
                logger.warning(
                    "LLM error%s (attempt %d/%d, model=%s%s): %s",
                    label_suffix,
                    attempt + 1,
                    total_attempts,
                    model_name,
                    ", conservative" if conservative else "",
                    str(e)[:120],
                )

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        failed_streak = self._record_request_failure()
        logger.warning(
            "LLM request failed%s after %d attempts; consecutive_failed=%d",
            label_suffix,
            total_attempts,
            failed_streak,
        )
        try:
            await client.close()
        except Exception:
            pass
        return ""

    # ═══════════════════════════════════════════════════════════════════════
    # 语言检测（轻量规则优先）
    # ═══════════════════════════════════════════════════════════════════════

    def detect_language_sync(self, text: str) -> str:
        """同步语言检测，规则优先，LLM兜底"""
        lang = self._rule_based_detect(text)
        if lang != "unknown":
            return lang

        if not self.enabled:
            return "zh"

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._llm_detect_language(text[:500]))
            return result if result else "zh"
        except Exception:
            return "zh"
        finally:
            loop.close()

    @staticmethod
    def _rule_based_detect(text: str) -> str:
        """基于Unicode范围的快速语言检测"""
        if not text.strip():
            return "zh"

        cn = 0
        en = 0
        jp_kana = 0
        jp_marks = 0
        kr = 0
        yue_markers = 0
        total = 0
        yue_chars = set("咩嘅喺佢哋冇唔咗啲啦咁嚟呢嗰噉")
        jp_punct = set("々〆〇ヶー。、「」『』【】")

        for ch in text:
            if ch.isspace() or ord(ch) < 0x21:
                continue
            total += 1
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                cn += 1
                if ch in yue_chars:
                    yue_markers += 1
            elif 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A:
                en += 1
            elif 0x3040 <= cp <= 0x30FF:
                jp_kana += 1
            elif ch in jp_punct:
                jp_marks += 1
            elif 0xAC00 <= cp <= 0xD7AF:
                kr += 1

        if total == 0:
            return "zh"

        cjk_ratio = cn / total
        en_ratio = en / total
        jp_ratio = jp_kana / total
        kr_ratio = kr / total

        if (jp_ratio > 0.02 and jp_kana >= 2) or (jp_kana >= 1 and jp_marks >= 2 and cjk_ratio > 0.2):
            return "ja"
        if kr_ratio > 0.08:
            return "ko"
        if yue_markers >= 2 and cjk_ratio > 0.25:
            return "yue"
        if en_ratio > 0.65 and cjk_ratio < 0.2:
            return "en"
        if cjk_ratio > 0.3:
            return "zh"
        return "unknown"

    async def _llm_detect_language(self, text: str) -> str:
        """使用LLM检测语言"""
        prompt = (
            "判断以下文本的主要语言，只返回一个语言代码，不要解释：\n"
            "zh=中文 en=英语 yue=粤语 ja=日语 ko=韩语 fr=法语 de=德语 "
            "es=西班牙语 ru=俄语 other=其他\n\n"
            f"文本：{text}"
        )
        result = await self._call_api([{"role": "user", "content": prompt}])
        if result:
            for code in ["zh", "en", "yue", "ja", "ko", "fr", "de", "es", "ru"]:
                if code in result.lower():
                    return code
        return "zh"

    # ═══════════════════════════════════════════════════════════════════════
    # 会议分析（不修改原文）
    # ═══════════════════════════════════════════════════════════════════════

    async def _analyze_transcript(
        self,
        full_text: str,
        response_language: str = "",
    ) -> LLMAnalysis:
        """兼容旧入口，统一走分层摘要路径。"""
        return await self._analyze_transcript_hierarchical(
            full_text,
            response_language=response_language,
        )

    def _build_analysis_messages(
        self,
        text_for_llm: str,
        *,
        response_language: str = "",
        stage: str = "single_pass",
        chunk_index: int = 0,
        total_chunks: int = 0,
    ) -> List[Dict[str, str]]:
        stage_line = ""
        if stage == "chunk" and total_chunks > 0:
            stage_line = (
                f"This is chunk {chunk_index}/{total_chunks} of a long transcript. "
                "Analyze only this chunk faithfully.\n\n"
            )

        prompt = (
            f"{stage_line}"
            "Analyze the following meeting transcript and output exactly these sections:\n\n"
            "## Summary\n"
            "- 2 to 4 sentences.\n\n"
            "## Key Points\n"
            "- 4 to 8 bullet points, one per line, each starting with '- '.\n\n"
            "## Action Items\n"
            "- Bullet points starting with '- '. If there are none, write '- None'.\n\n"
            "## Topics\n"
            "- Bullet points starting with '- '. If there are none, write '- None'.\n\n"
            "Do not rewrite or beautify the original transcript. Summarize faithfully.\n"
            "Do not invent names, numbers, decisions, or tasks.\n\n"
            f"Transcript:\n{text_for_llm}"
        )
        if self.summary_prompt:
            prompt += f"\n\nAdditional instruction:\n{self.summary_prompt}"

        response_lang = self._normalize_language_code(response_language)
        if response_lang:
            prompt += (
                "\n\nOutput requirements:\n"
                f"- Respond in language: {response_lang}\n"
                "- Keep the exact section structure."
            )

        return [
            {
                "role": "system",
                "content": (
                    "You are a meeting transcript analysis assistant. "
                    "Summarize faithfully and keep the output structured."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _build_reduce_messages(
        self,
        chunk_analyses: List[LLMAnalysis],
        *,
        response_language: str = "",
    ) -> List[Dict[str, str]]:
        blocks: List[str] = []
        for idx, item in enumerate(chunk_analyses, start=1):
            summary = str(item.summary or "").strip() or "(empty summary)"
            points = item.key_points or []
            actions = item.action_items or []
            topics = item.topics or []
            block = [
                f"Chunk {idx}",
                f"Summary: {summary}",
                "Key Points:",
            ]
            block.extend([f"- {one}" for one in points[:8]] or ["- None"])
            block.append("Action Items:")
            block.extend([f"- {one}" for one in actions[:8]] or ["- None"])
            block.append("Topics:")
            block.extend([f"- {one}" for one in topics[:8]] or ["- None"])
            blocks.append("\n".join(block))

        prompt = (
            "You are given chunk-level analyses from a long meeting transcript.\n"
            "Merge them into one final report and deduplicate repeated points.\n"
            "Output exactly these sections:\n\n"
            "## Summary\n"
            "## Key Points\n"
            "## Action Items\n"
            "## Topics\n\n"
            "Rules:\n"
            "- Keep only information supported by the chunk analyses.\n"
            "- Deduplicate repeated items.\n"
            "- Prefer concise, high-signal bullets.\n"
            "- If there are no action items or topics, write '- None'.\n\n"
            "Chunk analyses:\n\n"
            + "\n\n".join(blocks)
        )
        response_lang = self._normalize_language_code(response_language)
        if response_lang:
            prompt += (
                "\n\nOutput requirements:\n"
                f"- Respond in language: {response_lang}\n"
                "- Keep the exact section structure."
            )

        return [
            {
                "role": "system",
                "content": (
                    "You are a meeting transcript analysis assistant. "
                    "Merge chunk summaries into one faithful final summary."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def _tail_lines_for_overlap(lines: List[str], overlap_chars: int) -> List[str]:
        if overlap_chars <= 0 or not lines:
            return []
        tail: List[str] = []
        total = 0
        for line in reversed(lines):
            line_len = len(line) + (1 if tail else 0)
            if not tail and line_len > overlap_chars:
                break
            if tail and total + line_len > overlap_chars:
                break
            tail.append(line)
            total += line_len
        tail.reverse()
        return tail

    @staticmethod
    def _analysis_line_break_priority(line: str) -> int:
        text = str(line or "").strip()
        if not text:
            return 0
        if re.search(r"[。！？!?；;][\"'”’）】》」』]*$", text):
            return 4
        if re.search(r"[，,、：:][\"'”’）】》」』]*$", text):
            return 3
        return 1

    @classmethod
    def _choose_analysis_split_index(
        cls,
        lines: List[str],
        target_chars: int,
    ) -> int:
        if len(lines) <= 1:
            return len(lines)

        cumulative = 0
        candidates: List[tuple[int, int, int]] = []
        lower_bound = max(1, int(target_chars * 0.72))
        for idx, line in enumerate(lines[:-1], start=1):
            cumulative += len(line) + (1 if idx > 1 else 0)
            if cumulative < lower_bound:
                continue
            priority = cls._analysis_line_break_priority(line)
            distance = abs(cumulative - target_chars)
            candidates.append((priority, -distance, idx))

        if not candidates:
            return max(1, len(lines) - 1)
        candidates.sort(reverse=True)
        return int(candidates[0][2])

    def _split_text_for_analysis(self, full_text: str) -> List[str]:
        text = str(full_text or "").strip()
        if not text:
            return []
        if (
            not self.hierarchical_summary_enabled
            or len(text) <= self.analysis_single_pass_chars
        ):
            return [text]

        adaptive_target = min(
            self.analysis_chunk_target_chars,
            max(
                self.analysis_single_pass_chars,
                int((len(text) + self.analysis_max_chunks - 1) / self.analysis_max_chunks),
            ),
        )
        adaptive_target = max(2000, adaptive_target)
        lines: List[str] = []
        for raw_line in text.splitlines():
            line = str(raw_line or "").rstrip()
            if not line.strip():
                continue
            lines.extend(
                self._split_text_for_request(
                    line,
                    max_chars=adaptive_target,
                )
            )
        if not lines:
            return [text]

        chunks: List[str] = []
        current_lines: List[str] = []
        current_len = 0

        for line in lines:
            extra_len = len(line) + (1 if current_lines else 0)
            current_lines.append(line)
            current_len += extra_len
            while (
                len(current_lines) > 1
                and current_len > adaptive_target
                and len(chunks) < self.analysis_max_chunks - 1
            ):
                split_at = self._choose_analysis_split_index(
                    current_lines,
                    adaptive_target,
                )
                split_at = max(1, min(len(current_lines) - 1, split_at))
                emit_lines = current_lines[:split_at]
                chunk_text = "\n".join(emit_lines).strip()
                if not chunk_text:
                    break
                chunks.append(chunk_text)
                overlap_lines = self._tail_lines_for_overlap(
                    emit_lines,
                    self.analysis_chunk_overlap_chars,
                )
                current_lines = list(overlap_lines) + current_lines[split_at:]
                current_len = sum(len(one) for one in current_lines) + max(
                    0, len(current_lines) - 1
                )

        if current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks or [text]

    @staticmethod
    def _estimate_reduce_input_chars(chunk_analyses: List[LLMAnalysis]) -> int:
        total = 0
        for idx, item in enumerate(chunk_analyses, start=1):
            total += len(f"Chunk {idx}\nSummary: \nKey Points:\nAction Items:\nTopics:\n")
            total += len(str(item.summary or "").strip())
            total += sum(len(str(one or "").strip()) + 4 for one in list(item.key_points or [])[:8])
            total += sum(len(str(one or "").strip()) + 4 for one in list(item.action_items or [])[:8])
            total += sum(len(str(one or "").strip()) + 4 for one in list(item.topics or [])[:8])
        return total

    def _split_chunk_analyses_for_reduce(
        self,
        chunk_analyses: List[LLMAnalysis],
    ) -> List[List[LLMAnalysis]]:
        groups: List[List[LLMAnalysis]] = []
        current: List[LLMAnalysis] = []
        current_chars = 0
        target_chars = max(2000, self.analysis_reduce_target_chars)

        for item in chunk_analyses:
            item_chars = max(120, self._estimate_reduce_input_chars([item]))
            if current and current_chars + item_chars > target_chars:
                groups.append(list(current))
                current = [item]
                current_chars = item_chars
                continue
            current.append(item)
            current_chars += item_chars

        if current:
            groups.append(list(current))
        return groups

    @staticmethod
    def _unique_strings(items: List[str], *, limit: int = 8) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _normalize_report_line(text: str, *, limit: int = 120) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(value) <= limit:
            return value
        trimmed = value[: max(0, limit - 1)].rstrip(" ,;:，；：")
        return f"{trimmed}..."

    def build_report_fallback_analysis(
        self,
        segments: List[Any],
        *,
        response_language: str = "",
        reason: str = "",
    ) -> LLMAnalysis:
        texts: List[str] = []
        speakers: set[str] = set()
        action_markers = (
            "待办",
            "跟进",
            "安排",
            "确认",
            "发送",
            "提交",
            "处理",
            "完成",
            "需要",
            "尽快",
            "todo",
            "action",
            "follow up",
            "follow-up",
            "next step",
        )

        for seg in segments or []:
            text = self._normalize_report_line(getattr(seg, "text", ""))
            if not text:
                continue
            texts.append(text)
            speaker = str(getattr(seg, "speaker", "") or "").strip()
            if speaker:
                speakers.add(speaker)

        if not texts:
            return LLMAnalysis(
                language=self._normalize_language_code(response_language) or "zh",
                error=str(reason or "report_fallback_no_text").strip(),
                metadata={
                    "mode": "local_report_fallback",
                    "segment_count": len(list(segments or [])),
                },
            )

        summary_seed = self._unique_strings(texts, limit=3)
        summary_body = "；".join(summary_seed)
        summary_prefix = (
            f"本次转写共 {len(texts)} 段"
            f"，涉及 {max(1, len(speakers))} 位说话人。"
        )
        summary = summary_prefix
        if summary_body:
            summary = f"{summary_prefix} 主要内容包括：{summary_body}"
        summary = self._normalize_report_line(summary, limit=260)

        key_points = self._unique_strings(
            [self._normalize_report_line(text, limit=110) for text in texts],
            limit=5,
        )

        action_items = self._unique_strings(
            [
                self._normalize_report_line(text, limit=110)
                for text in texts
                if any(marker in text.lower() for marker in action_markers)
            ],
            limit=5,
        )

        topic_candidates = []
        for text in texts:
            head = re.split(r"[。！？!?；;，,]", text, maxsplit=1)[0].strip()
            head = self._normalize_report_line(head, limit=40)
            if head:
                topic_candidates.append(head)
        topics = self._unique_strings(topic_candidates, limit=4)

        return LLMAnalysis(
            language=self._normalize_language_code(response_language) or "zh",
            summary=summary,
            key_points=key_points,
            action_items=action_items,
            topics=topics,
            error="",
            metadata={
                "mode": "local_report_fallback",
                "request_strategy": "local",
                "segment_count": len(texts),
                "speaker_count": max(1, len(speakers)),
                "reason": str(reason or "report_content_missing").strip(),
            },
        )

    @staticmethod
    def _analysis_brief(analysis: LLMAnalysis) -> Dict[str, Any]:
        summary = str(getattr(analysis, "summary", "") or "").strip()
        return {
            "summary_preview": summary[:160],
            "summary_chars": len(summary),
            "key_points": len(list(getattr(analysis, "key_points", []) or [])),
            "action_items": len(list(getattr(analysis, "action_items", []) or [])),
            "topics": len(list(getattr(analysis, "topics", []) or [])),
            "error": str(getattr(analysis, "error", "") or "").strip(),
        }

    def _merge_analyses_locally(
        self,
        chunk_analyses: List[LLMAnalysis],
        *,
        response_language: str = "",
        reason: str = "",
    ) -> LLMAnalysis:
        summaries = [
            str(item.summary or "").strip()
            for item in chunk_analyses
            if str(item.summary or "").strip()
        ]
        key_points: List[str] = []
        action_items: List[str] = []
        topics: List[str] = []
        for item in chunk_analyses:
            key_points.extend(list(item.key_points or []))
            action_items.extend(list(item.action_items or []))
            topics.extend(list(item.topics or []))

        return LLMAnalysis(
            language=self._normalize_language_code(response_language) or "zh",
            summary=" ".join(summaries[:4]).strip(),
            key_points=self._unique_strings(key_points, limit=8),
            action_items=self._unique_strings(action_items, limit=8),
            topics=self._unique_strings(topics, limit=8),
            error=str(reason or "").strip(),
            metadata={
                "mode": "hierarchical",
                "request_strategy": "chunk_then_reduce",
                "reduce_stage": "local_fallback",
                "chunk_count": len(chunk_analyses),
                "chunk_briefs": [self._analysis_brief(item) for item in chunk_analyses],
            },
        )

    async def _analyze_transcript_single(
        self,
        full_text: str,
        *,
        response_language: str = "",
        stage: str = "single_pass",
        chunk_index: int = 0,
        total_chunks: int = 0,
    ) -> LLMAnalysis:
        analysis = LLMAnalysis()
        source_text = str(full_text or "").strip()
        if not source_text:
            return analysis

        if stage == "chunk" and total_chunks > 0:
            request_label = f"analysis_chunk {chunk_index}/{total_chunks}"
        else:
            request_label = f"analysis_{stage}"

        result = await self._call_api(
            self._build_analysis_messages(
                source_text,
                response_language=response_language,
                stage=stage,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            ),
            enable_thinking=True,
            request_label=request_label,
        )

        if result:
            analysis = self._parse_analysis_structured(result)
        else:
            analysis.error = "empty_response"

        analysis.metadata.update(
            {
                "stage": stage,
                "source_chars": len(source_text),
                "chunk_index": int(chunk_index),
                "total_chunks": int(total_chunks),
                "request_strategy": "chunk_then_reduce" if stage == "chunk" else "single_pass",
            }
        )
        return analysis

    async def _reduce_chunk_analyses(
        self,
        chunk_analyses: List[LLMAnalysis],
        *,
        response_language: str = "",
        reduce_depth: int = 0,
    ) -> LLMAnalysis:
        if not chunk_analyses:
            return LLMAnalysis(error="no_chunk_analyses")
        if len(chunk_analyses) == 1:
            analysis = self._clone_analysis(chunk_analyses[0])
            analysis.metadata.update(
                {
                    "stage": "reduce",
                    "reduce_stage": "passthrough",
                    "chunk_count": 1,
                    "request_strategy": "chunk_then_reduce",
                    "chunk_briefs": [self._analysis_brief(chunk_analyses[0])],
                    "reduce_depth": int(reduce_depth),
                }
            )
            return analysis
        if reduce_depth >= 3:
            logger.warning(
                "LLM analysis reduce depth limit reached; fallback to local merge (%d chunks)",
                len(chunk_analyses),
            )
            return self._merge_analyses_locally(
                chunk_analyses,
                response_language=response_language,
                reason="reduce_depth_limit",
            )

        estimated_chars = self._estimate_reduce_input_chars(chunk_analyses)
        reduce_groups = self._split_chunk_analyses_for_reduce(chunk_analyses)
        if len(reduce_groups) > 1:
            logger.info(
                "LLM analysis reduce will recurse: source_chunks=%d groups=%d est_chars=%d target=%d depth=%d",
                len(chunk_analyses),
                len(reduce_groups),
                estimated_chars,
                self.analysis_reduce_target_chars,
                reduce_depth,
            )
            if all(len(group) <= 1 for group in reduce_groups):
                logger.warning(
                    "LLM analysis reduce inputs are individually over budget; fallback to local merge"
                )
                return self._merge_analyses_locally(
                    chunk_analyses,
                    response_language=response_language,
                    reason="reduce_input_over_budget",
                )

            condensed: List[LLMAnalysis] = []
            for group_index, group in enumerate(reduce_groups, start=1):
                if len(group) == 1:
                    condensed.append(self._clone_analysis(group[0]))
                    continue
                logger.info(
                    "LLM analysis reduce group %d/%d: chunks=%d est_chars=%d",
                    group_index,
                    len(reduce_groups),
                    len(group),
                    self._estimate_reduce_input_chars(group),
                )
                group_analysis = await self._reduce_chunk_analyses(
                    group,
                    response_language=response_language,
                    reduce_depth=reduce_depth + 1,
                )
                if (
                    group_analysis.summary
                    or group_analysis.key_points
                    or group_analysis.action_items
                    or group_analysis.topics
                ):
                    condensed.append(group_analysis)
                else:
                    condensed.append(
                        self._merge_analyses_locally(
                            group,
                            response_language=response_language,
                            reason="reduce_group_empty_response",
                        )
                    )
            return await self._reduce_chunk_analyses(
                condensed,
                response_language=response_language,
                reduce_depth=reduce_depth + 1,
            )

        logger.info(
            "LLM analysis reduce request submitted: source_chunks=%d est_chars=%d depth=%d",
            len(chunk_analyses),
            estimated_chars,
            reduce_depth,
        )
        result = await self._call_api(
            self._build_reduce_messages(
                chunk_analyses,
                response_language=response_language,
            ),
            enable_thinking=True,
            request_label=f"analysis_reduce {len(chunk_analyses)} chunks",
        )
        if not result:
            logger.warning(
                "LLM analysis reduce returned empty response; fallback to local merge"
            )
            return self._merge_analyses_locally(
                chunk_analyses,
                response_language=response_language,
                reason="reduce_empty_response",
            )

        analysis = self._parse_analysis_structured(result)
        analysis.metadata.update(
            {
                "stage": "reduce",
                "reduce_stage": "llm",
                "chunk_count": len(chunk_analyses),
                "request_strategy": "chunk_then_reduce",
                "chunk_briefs": [self._analysis_brief(item) for item in chunk_analyses],
                "reduce_depth": int(reduce_depth),
                "reduce_estimated_chars": int(estimated_chars),
            }
        )
        logger.info(
            "LLM analysis reduce complete: summary=%d chars, points=%d, actions=%d, topics=%d",
            len(analysis.summary),
            len(analysis.key_points),
            len(analysis.action_items),
            len(analysis.topics),
        )
        return analysis

    async def _analyze_transcript_hierarchical(
        self,
        full_text: str,
        response_language: str = "",
        *,
        job_id: str = "",
    ) -> LLMAnalysis:
        source_text = str(full_text or "").strip()
        if not source_text:
            return LLMAnalysis()

        total_chars = len(source_text)
        chunks = self._split_text_for_analysis(source_text)
        response_lang = self._normalize_language_code(response_language)
        chunk_sizes = [len(chunk) for chunk in chunks]
        if len(chunks) <= 1:
            logger.info(
                "LLM analysis mode=single_pass chars=%d lang=%s",
                total_chars,
                response_lang or "auto",
            )
            if job_id:
                self._update_job_status(
                    job_id,
                    analysis_mode="single_pass",
                    phase="single_pass",
                    total_chars=total_chars,
                    total_chunks=1,
                    completed_chunks=0,
                    reduce_started=False,
                    reduce_finished=True,
                )
            analysis = await self._analyze_transcript_single(
                source_text,
                response_language=response_lang,
                stage="single_pass",
            )
            analysis.metadata.update(
                {
                    "mode": "single_pass",
                    "hierarchical": False,
                    "total_chars": total_chars,
                    "chunk_count": 1,
                    "completed_chunks": 1,
                    "chunk_sizes": chunk_sizes or [total_chars],
                    "request_strategy": "single_pass",
                }
            )
            return analysis

        total_chunks = len(chunks)
        logger.info(
            "LLM analysis mode=hierarchical chars=%d chunks=%d target=%d overlap=%d lang=%s",
            total_chars,
            total_chunks,
            self.analysis_chunk_target_chars,
            self.analysis_chunk_overlap_chars,
            response_lang or "auto",
        )
        logger.info(
            "LLM analysis chunk plan: %s",
            ", ".join(
                f"{index + 1}:{size}" for index, size in enumerate(chunk_sizes)
            ),
        )
        if job_id:
            self._update_job_status(
                job_id,
                analysis_mode="hierarchical",
                phase="chunk_analysis",
                total_chars=total_chars,
                total_chunks=total_chunks,
                completed_chunks=0,
                reduce_started=False,
                reduce_finished=False,
                chunk_sizes=chunk_sizes,
            )

        chunk_analyses: List[LLMAnalysis] = []
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            logger.info(
                "LLM analysis chunk request %d/%d chars=%d",
                chunk_index,
                total_chunks,
                len(chunk_text),
            )
            if job_id:
                self._update_job_status(
                    job_id,
                    phase="chunk_analysis",
                    current_chunk=chunk_index,
                )

            chunk_analysis = await self._analyze_transcript_single(
                chunk_text,
                response_language=response_lang,
                stage="chunk",
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
            if (
                chunk_analysis.summary
                or chunk_analysis.key_points
                or chunk_analysis.action_items
                or chunk_analysis.topics
            ):
                chunk_analyses.append(chunk_analysis)
            else:
                logger.warning(
                    "LLM analysis chunk %d/%d returned no structured content",
                    chunk_index,
                    total_chunks,
                )
            logger.info(
                "LLM analysis chunk complete %d/%d: summary=%d chars, points=%d, actions=%d, topics=%d, error=%s",
                chunk_index,
                total_chunks,
                len(chunk_analysis.summary),
                len(chunk_analysis.key_points),
                len(chunk_analysis.action_items),
                len(chunk_analysis.topics),
                str(chunk_analysis.error or "") or "<none>",
            )

            if job_id:
                self._append_job_chunk_analysis(job_id, chunk_analysis)
                self._update_job_status(job_id, completed_chunks=chunk_index)

        if not chunk_analyses:
            return LLMAnalysis(
                error="all_chunk_requests_failed",
                metadata={
                    "mode": "hierarchical",
                    "hierarchical": True,
                    "total_chars": total_chars,
                    "chunk_count": total_chunks,
                    "completed_chunks": 0,
                    "chunk_sizes": chunk_sizes,
                    "request_strategy": "chunk_then_reduce",
                },
            )

        logger.info(
            "LLM analysis reduce stage: chunk_results=%d/%d",
            len(chunk_analyses),
            total_chunks,
        )
        if job_id:
            self._update_job_status(
                job_id,
                phase="reduce",
                reduce_started=True,
            )

        analysis = await self._reduce_chunk_analyses(
            chunk_analyses,
            response_language=response_lang,
        )
        analysis.metadata.update(
            {
                "mode": "hierarchical",
                "hierarchical": True,
                "total_chars": total_chars,
                "chunk_count": total_chunks,
                "completed_chunks": len(chunk_analyses),
                "chunk_sizes": chunk_sizes,
                "request_strategy": "chunk_then_reduce",
            }
        )
        if job_id:
            self._update_job_status(
                job_id,
                phase="done",
                reduce_finished=True,
            )
        return analysis

    @staticmethod
    def _parse_analysis(text: str) -> LLMAnalysis:
        """解析LLM分析结果"""
        analysis = LLMAnalysis()
        current_section = ""

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            lower = stripped.lower()
            if "摘要" in stripped or "summary" in lower:
                current_section = "summary"
                continue
            elif "关键" in stripped or "要点" in stripped or "key" in lower:
                current_section = "key_points"
                continue
            elif "待办" in stripped or "action" in lower or "todo" in lower:
                current_section = "action_items"
                continue
            elif "主题" in stripped or "topic" in lower:
                current_section = "topics"
                continue

            # 去掉列表前缀
            content = stripped
            for prefix in ["- ", "• ", "* ", "· "]:
                if content.startswith(prefix):
                    content = content[len(prefix):]
                    break

            # 去掉数字编号
            if len(content) > 2 and content[0].isdigit() and content[1] in ".、)）":
                content = content[2:].strip()
            elif (
                len(content) > 3
                and content[:2].isdigit()
                and content[2] in ".、)）"
            ):
                content = content[3:].strip()

            if not content:
                continue

            if current_section == "summary":
                if analysis.summary:
                    analysis.summary += " " + content
                else:
                    analysis.summary = content
            elif current_section == "key_points":
                if content != "无":
                    analysis.key_points.append(content)
            elif current_section == "action_items":
                if content != "无":
                    analysis.action_items.append(content)
            elif current_section == "topics":
                if content != "无":
                    analysis.topics.append(content)

        return analysis

    # ═══════════════════════════════════════════════════════════════════════
    # 流水线接口（异步提交，不阻塞ASR）
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_analysis_structured(text: str) -> LLMAnalysis:
        analysis = LLMAnalysis()
        current_section = ""
        section_map = {
            "summary": "summary",
            "鎽樿": "summary",
            "key points": "key_points",
            "key point": "key_points",
            "鍏抽敭瑕佺偣": "key_points",
            "瑕佺偣": "key_points",
            "action items": "action_items",
            "action item": "action_items",
            "todo": "action_items",
            "to do": "action_items",
            "寰呭姙浜嬮」": "action_items",
            "topics": "topics",
            "topic": "topics",
            "涓婚": "topics",
            "璁ㄨ涓婚": "topics",
        }

        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            header = stripped.lstrip("#").strip().strip(":：").lower()
            mapped = section_map.get(header)
            if mapped:
                current_section = mapped
                continue

            content = stripped
            for prefix in ["- ", "鈥?", "* ", "路 "]:
                if content.startswith(prefix):
                    content = content[len(prefix):]
                    break

            if (
                len(content) > 2
                and content[0].isdigit()
                and content[1] in ".)"
            ):
                content = content[2:].strip()
            elif (
                len(content) > 3
                and content[:2].isdigit()
                and content[2] in ".)"
            ):
                content = content[3:].strip()

            if not content:
                continue

            if content.lower() in {"none", "n/a", "no action items", "no topics"}:
                continue

            if current_section == "summary":
                analysis.summary = (analysis.summary + " " + content).strip()
            elif current_section == "key_points":
                analysis.key_points.append(content)
            elif current_section == "action_items":
                analysis.action_items.append(content)
            elif current_section == "topics":
                analysis.topics.append(content)

        return analysis

    def submit_analysis(
        self,
        segments: List,
        file_id: str = "",
        response_language: str = "",
    ) -> str:
        """
        提交转写结果进行异步分析（不修改原文）。
        Returns: job_id
        """
        if not self.analysis_enabled:
            logger.debug(
                "LLM analysis disabled, skip analysis submit: "
                f"{self.disabled_reason or 'llm.enabled=false'}"
            )
            return ""

        if not self._started:
            self.start()
        if not self._started or self._loop is None:
            logger.warning("LLM loop unavailable, skipping async analysis submission")
            return ""

        response_lang = self._normalize_language_code(response_language)
        job_seed = f"{file_id}:{len(segments)}:{time.time_ns()}"
        job_hash = hashlib.sha1(job_seed.encode("utf-8")).hexdigest()[:10]
        job_id = f"{file_id or 'job'}_{job_hash}"

        with self._lock:
            self._jobs[job_id] = {
                "segments": segments,
                "analysis": None,
                "done": False,
                "response_language": response_lang,
                "analysis_mode": "pending",
                "request_strategy": "single_pass",
                "phase": "queued",
                "total_chars": 0,
                "total_chunks": 0,
                "completed_chunks": 0,
                "current_chunk": 0,
                "reduce_started": False,
                "reduce_finished": False,
                "chunk_sizes": [],
                "chunk_analyses": [],
                "chunk_briefs": [],
                "future": None,
                "submitted_at": time.time(),
            }

        # 组装全文
        full_text = "\n".join(
            f"[{seg.speaker}] {seg.text}"
            for seg in segments
            if seg.text.strip()
        )
        planned_chunks = self._split_text_for_analysis(full_text)
        self._update_job_status(
            job_id,
            total_chars=len(full_text),
            total_chunks=max(1, len(planned_chunks)),
            analysis_mode="hierarchical" if len(planned_chunks) > 1 else "single_pass",
            request_strategy="chunk_then_reduce" if len(planned_chunks) > 1 else "single_pass",
            chunk_sizes=[len(chunk) for chunk in planned_chunks],
        )

        future = asyncio.run_coroutine_threadsafe(
            self._run_analysis_job(
                job_id,
                full_text,
                response_language=response_lang,
            ),
            self._loop,
        )
        self._update_job_status(job_id, future=future)

        logger.debug(f"Submitted analysis job: {job_id}")
        return job_id

    def _update_job_status(self, job_id: str, **values) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(values)

    def _append_job_chunk_analysis(self, job_id: str, analysis: LLMAnalysis) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            bucket = job.setdefault("chunk_analyses", [])
            if isinstance(bucket, list):
                bucket.append(analysis)
            brief_bucket = job.setdefault("chunk_briefs", [])
            if isinstance(brief_bucket, list):
                brief_bucket.append(self._analysis_brief(analysis))

    def describe_analysis_job(self, job_id: str) -> Dict[str, Any]:
        if not job_id:
            return {}
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return {}
            return {
                "job_id": job_id,
                "analysis_mode": str(job.get("analysis_mode", "") or ""),
                "request_strategy": str(job.get("request_strategy", "") or ""),
                "phase": str(job.get("phase", "") or ""),
                "total_chars": int(job.get("total_chars", 0) or 0),
                "total_chunks": int(job.get("total_chunks", 0) or 0),
                "completed_chunks": int(job.get("completed_chunks", 0) or 0),
                "current_chunk": int(job.get("current_chunk", 0) or 0),
                "reduce_started": bool(job.get("reduce_started", False)),
                "reduce_finished": bool(job.get("reduce_finished", False)),
                "chunk_sizes": list(job.get("chunk_sizes", []) or []),
                "chunk_briefs": list(job.get("chunk_briefs", []) or []),
            }

    def estimate_analysis_wait_timeout(self, total_chars: int) -> float:
        source_text = "x" * max(0, int(total_chars or 0))
        planned_chunks = self._split_text_for_analysis(source_text)
        request_count = max(1, len(planned_chunks))
        reduce_requests = 1 if request_count > 1 else 0
        per_call_budget = (
            max(10.0, float(self.timeout))
            + max(0.0, float(self.retry_delay)) * max(0, int(self.max_retries))
            + 8.0
        )
        estimated = 20.0 + (request_count + reduce_requests) * per_call_budget
        return min(3600.0, max(90.0, estimated))

    def _build_timeout_partial_analysis(
        self,
        job_id: str,
        snapshot: Dict[str, Any],
    ) -> LLMAnalysis:
        chunk_analyses = list(snapshot.get("chunk_analyses") or [])
        phase = str(snapshot.get("phase", "") or "waiting")
        completed = int(snapshot.get("completed_chunks", 0) or 0)
        total_chunks = int(snapshot.get("total_chunks", 0) or 0)
        mode = str(snapshot.get("analysis_mode", "") or "unknown")
        reason = (
            f"timeout during {phase} "
            f"(mode={mode}, completed_chunks={completed}/{max(1, total_chunks)})"
        )
        if not chunk_analyses:
            return LLMAnalysis(
                error=reason,
                metadata={
                    "mode": mode,
                    "request_strategy": str(
                        snapshot.get("request_strategy", "") or "single_pass"
                    ),
                    "phase": phase,
                    "chunk_count": total_chunks,
                    "completed_chunks": completed,
                    "job_id": job_id,
                    "chunk_sizes": list(snapshot.get("chunk_sizes", []) or []),
                    "chunk_briefs": list(snapshot.get("chunk_briefs", []) or []),
                    "reduce_started": bool(snapshot.get("reduce_started", False)),
                    "reduce_finished": bool(snapshot.get("reduce_finished", False)),
                },
            )

        partial = self._merge_analyses_locally(
            chunk_analyses,
            response_language=str(snapshot.get("response_language", "") or ""),
            reason=reason,
        )
        partial.metadata.update(
            {
                "phase": phase,
                "job_id": job_id,
                "completed_chunks": completed,
                "chunk_count": total_chunks,
                "request_strategy": str(
                    snapshot.get("request_strategy", "") or "chunk_then_reduce"
                ),
                "chunk_sizes": list(snapshot.get("chunk_sizes", []) or []),
                "chunk_briefs": list(snapshot.get("chunk_briefs", []) or []),
                "reduce_started": bool(snapshot.get("reduce_started", False)),
                "reduce_finished": bool(snapshot.get("reduce_finished", False)),
            }
        )
        return partial

    def _cancel_job_future(self, snapshot: Dict[str, Any]) -> None:
        future = snapshot.get("future")
        if isinstance(future, concurrent.futures.Future) and not future.done():
            future.cancel()

    async def _run_analysis_job(
        self,
        job_id: str,
        full_text: str,
        response_language: str = "",
    ):
        """后台处理单个分析任务"""
        try:
            if self._analysis_semaphore is not None:
                async with self._analysis_semaphore:
                    analysis = await self._analyze_transcript_hierarchical(
                        full_text,
                        response_language=response_language,
                        job_id=job_id,
                    )
            else:
                analysis = await self._analyze_transcript_hierarchical(
                    full_text,
                    response_language=response_language,
                    job_id=job_id,
                )
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["analysis"] = analysis
                    self._jobs[job_id]["done"] = True
                    self._jobs[job_id]["phase"] = "done"
                    self._jobs[job_id]["future"] = None
        except asyncio.CancelledError:
            logger.info("Analysis job %s cancelled", job_id)
            raise
        except Exception as e:
            logger.error(f"Analysis job {job_id} failed: {e}")
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["done"] = True
                    self._jobs[job_id]["phase"] = "error"
                    self._jobs[job_id]["future"] = None
                    self._jobs[job_id]["analysis"] = LLMAnalysis(error=str(e))

    def wait_for_analysis(
        self,
        job_id: str,
        timeout: float = 120,
        pause_checker=None,
        cancel_checker=None,
    ) -> Optional[LLMAnalysis]:
        """等待分析结果"""
        if not job_id or not self.analysis_enabled:
            return None

        deadline = time.time() + timeout
        last_progress_log = 0.0
        while time.time() < deadline:
            if cancel_checker:
                try:
                    if cancel_checker():
                        break
                except Exception:
                    pass
            if pause_checker:
                try:
                    pause_checker()
                except Exception:
                    pass
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.get("done"):
                    analysis = job.get("analysis")
                    self._cancel_job_future(job)
                    del self._jobs[job_id]
                    return analysis
                snapshot = dict(job or {})
            now = time.time()
            if snapshot and now - last_progress_log >= 8.0:
                logger.info(
                    "LLM analysis progress: job=%s mode=%s strategy=%s phase=%s current_chunk=%d chunks=%d/%d reduce_started=%s",
                    job_id,
                    str(snapshot.get("analysis_mode", "") or "pending"),
                    str(snapshot.get("request_strategy", "") or "single_pass"),
                    str(snapshot.get("phase", "") or "queued"),
                    int(snapshot.get("current_chunk", 0) or 0),
                    int(snapshot.get("completed_chunks", 0) or 0),
                    max(1, int(snapshot.get("total_chunks", 0) or 0)),
                    bool(snapshot.get("reduce_started", False)),
                )
                last_progress_log = now
            time.sleep(0.1)

        with self._lock:
            snapshot = dict(self._jobs.get(job_id) or {})
            self._jobs.pop(job_id, None)
        if snapshot:
            self._cancel_job_future(snapshot)
            logger.warning(
                "Analysis job %s timed out: mode=%s strategy=%s phase=%s current_chunk=%d chunks=%d/%d reduce_started=%s",
                job_id,
                str(snapshot.get("analysis_mode", "") or "pending"),
                str(snapshot.get("request_strategy", "") or "single_pass"),
                str(snapshot.get("phase", "") or "queued"),
                int(snapshot.get("current_chunk", 0) or 0),
                int(snapshot.get("completed_chunks", 0) or 0),
                max(1, int(snapshot.get("total_chunks", 0) or 0)),
                bool(snapshot.get("reduce_started", False)),
            )
            return self._build_timeout_partial_analysis(job_id, snapshot)
        logger.warning(f"Analysis job {job_id} timed out")
        return None

    @staticmethod
    def _speaker_profile_payload(
        profiles: Dict[str, SpeakerProfile],
    ) -> Dict[str, Dict[str, Any]]:
        payload: Dict[str, Dict[str, Any]] = {}
        for speaker, profile in (profiles or {}).items():
            payload[str(speaker)] = {
                "role": str(getattr(profile, "role", "") or "").strip(),
                "style": str(getattr(profile, "style", "") or "").strip(),
                "evidence": [
                    str(item or "").strip()
                    for item in list(getattr(profile, "evidence", []) or [])
                    if str(item or "").strip()
                ][:3],
            }
        return payload

    @staticmethod
    def _analysis_summary_for_arbitration(analysis: Any) -> str:
        if analysis is None:
            return ""
        summary = str(getattr(analysis, "summary", "") or "").strip()
        if summary:
            return summary
        points = [
            str(item or "").strip()
            for item in list(getattr(analysis, "key_points", []) or [])
            if str(item or "").strip()
        ][:4]
        return " ".join(points).strip()

    @staticmethod
    def _speaker_constraints_payload(
        speaker_constraints: Optional[Dict[str, Any]],
        allowed_speakers: List[str],
    ) -> Dict[str, Any]:
        payload = dict(speaker_constraints or {})
        payload["allowed_speakers"] = list(allowed_speakers)
        return payload

    @staticmethod
    def _segment_gap_seconds(left: Any, right: Any) -> float:
        try:
            left_end = float(getattr(left, "end", 0.0) or 0.0)
        except Exception:
            left_end = 0.0
        try:
            right_start = float(getattr(right, "start", left_end) or left_end)
        except Exception:
            right_start = left_end
        return max(0.0, right_start - left_end)

    @staticmethod
    def _join_segment_texts(texts: List[str]) -> str:
        joined = ""
        for raw_part in texts:
            part = str(raw_part or "").strip()
            if not part:
                continue
            if not joined:
                joined = part
                continue
            prev_char = joined[-1]
            next_char = part[0]
            if (
                prev_char.isascii()
                and prev_char.isalnum()
                and next_char.isascii()
                and next_char.isalnum()
            ):
                joined = f"{joined} {part}"
            elif prev_char.isspace() or next_char.isspace():
                joined = f"{joined}{part}"
            else:
                joined = f"{joined}{part}"
        return joined.strip()

    @staticmethod
    def _merge_transcription_segments(group: List[Any]) -> Any:
        first = group[0]
        last = group[-1]
        merged_words: List[Dict[str, Any]] = []
        merged_highlights: List[str] = []
        seen_highlights: set[str] = set()
        speaker_role = ""
        language = ""
        speaker = str(getattr(first, "speaker", "") or "").strip()
        confidences: List[float] = []
        arbitration_confidences: List[float] = []

        for seg in group:
            if not language:
                language = str(getattr(seg, "language", "") or "").strip()
            if not speaker_role:
                speaker_role = str(getattr(seg, "speaker_role", "") or "").strip()
            try:
                conf_value = float(getattr(seg, "confidence", 0.0) or 0.0)
                if conf_value > 0.0:
                    confidences.append(conf_value)
            except Exception:
                pass
            try:
                arb_value = float(getattr(seg, "arbitration_confidence", 0.0) or 0.0)
                if arb_value > 0.0:
                    arbitration_confidences.append(arb_value)
            except Exception:
                pass
            for highlight in list(getattr(seg, "semantic_highlights", []) or []):
                text = str(highlight or "").strip()
                if not text:
                    continue
                key = text.casefold()
                if key in seen_highlights:
                    continue
                seen_highlights.add(key)
                merged_highlights.append(text)
            for word in list(getattr(seg, "words", []) or []):
                if not isinstance(word, dict):
                    continue
                entry = dict(word)
                if speaker:
                    entry["speaker"] = speaker
                merged_words.append(entry)

        merged_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        merged_arb_confidence = (
            sum(arbitration_confidences) / len(arbitration_confidences)
            if arbitration_confidences
            else 0.0
        )
        return first.__class__(
            start=float(getattr(first, "start", 0.0) or 0.0),
            end=float(getattr(last, "end", getattr(first, "end", 0.0)) or 0.0),
            text=LLMProcessor._join_segment_texts(
                [str(getattr(seg, "text", "") or "") for seg in group]
            ),
            speaker=speaker,
            language=language,
            confidence=merged_confidence,
            words=merged_words,
            semantic_highlights=merged_highlights,
            arbitration_confidence=merged_arb_confidence,
            speaker_role=speaker_role,
        )

    def _apply_speaker_arbitration_merges(
        self,
        segments: List[Any],
        arb_result: SpeakerArbitrationResult,
        *,
        allowed_speakers: List[str],
    ) -> Dict[str, Any]:
        merge_groups = list(getattr(arb_result, "merge_groups", []) or [])
        if (
            not self.speaker_arbitration_enable_segment_merge
            or len(segments) < 2
            or not merge_groups
        ):
            return {"merged_group_count": 0, "merged_segments_removed": 0}

        decisions = dict(getattr(arb_result, "decisions", {}) or {})
        candidates: List[Dict[str, Any]] = []
        for merge_group in merge_groups:
            raw_indices = [int(index) for index in list(getattr(merge_group, "indices", []) or [])]
            indices = sorted(dict.fromkeys(raw_indices))
            if (
                len(indices) < 2
                or len(indices) > self.speaker_arbitration_max_merge_group_size
                or indices[0] < 0
                or indices[-1] >= len(segments)
            ):
                continue
            if any((left + 1) != right for left, right in zip(indices, indices[1:])):
                continue
            group_segments = [segments[index] for index in indices]
            total_duration = max(
                0.0,
                float(getattr(group_segments[-1], "end", 0.0) or 0.0)
                - float(getattr(group_segments[0], "start", 0.0) or 0.0),
            )
            if total_duration > self.speaker_arbitration_max_merged_duration_sec:
                continue
            if any(
                self._segment_gap_seconds(group_segments[pos], group_segments[pos + 1])
                > self.speaker_arbitration_max_merge_gap_sec
                for pos in range(len(group_segments) - 1)
            ):
                continue
            target_speaker = str(getattr(merge_group, "speaker", "") or "").strip()
            if target_speaker and target_speaker not in allowed_speakers:
                target_speaker = ""
            final_speakers = [
                str(getattr(group_segments[pos], "speaker", "") or "").strip()
                for pos in range(len(group_segments))
            ]
            if target_speaker:
                if any(speaker != target_speaker for speaker in final_speakers):
                    continue
            else:
                unique_speakers = {speaker for speaker in final_speakers if speaker}
                if len(unique_speakers) != 1:
                    continue
                target_speaker = next(iter(unique_speakers))
            decision_confidences = [
                float(getattr(decisions.get(index), "confidence", 0.0) or 0.0)
                for index in indices
                if decisions.get(index) is not None
            ]
            base_confidence = float(getattr(merge_group, "confidence", 0.0) or 0.0)
            effective_confidence = max(
                base_confidence,
                (sum(decision_confidences) / len(decision_confidences))
                if decision_confidences
                else 0.0,
            )
            if effective_confidence < self.speaker_arbitration_merge_min_confidence:
                continue
            candidates.append(
                {
                    "indices": indices,
                    "confidence": effective_confidence,
                    "speaker": target_speaker,
                }
            )

        if not candidates:
            return {"merged_group_count": 0, "merged_segments_removed": 0}

        candidates.sort(
            key=lambda item: (
                -len(list(item.get("indices") or [])),
                -float(item.get("confidence", 0.0) or 0.0),
                int((list(item.get("indices") or [0]) or [0])[0]),
            )
        )
        selected: List[Dict[str, Any]] = []
        occupied: set[int] = set()
        for candidate in candidates:
            indices = list(candidate.get("indices") or [])
            if any(index in occupied for index in indices):
                continue
            selected.append(candidate)
            occupied.update(indices)

        if not selected:
            return {"merged_group_count": 0, "merged_segments_removed": 0}

        group_map = {
            int(list(item["indices"])[0]): list(item["indices"])
            for item in selected
        }
        merged_segments: List[Any] = []
        merged_group_count = 0
        removed_segments = 0
        index = 0
        while index < len(segments):
            group_indices = group_map.get(index)
            if group_indices:
                group = [segments[group_index] for group_index in group_indices]
                merged_segments.append(self._merge_transcription_segments(group))
                merged_group_count += 1
                removed_segments += max(0, len(group_indices) - 1)
                index = int(group_indices[-1]) + 1
                continue
            merged_segments.append(segments[index])
            index += 1

        if merged_group_count > 0:
            segments[:] = merged_segments
        return {
            "merged_group_count": merged_group_count,
            "merged_segments_removed": removed_segments,
        }

    def _apply_speaker_arbitration_result(
        self,
        segments: List[Any],
        arb_result: SpeakerArbitrationResult,
        *,
        speaker_constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "applied": False,
            "changed_segments": 0,
            "reason": str(getattr(arb_result, "reason", "") or ""),
            "speaker_profiles": self._speaker_profile_payload(
                getattr(arb_result, "speaker_profiles", {}) or {}
            ),
            "window_count": int(
                (getattr(arb_result, "metadata", {}) or {}).get("window_count", 0) or 0
            ),
            "decision_count": len(getattr(arb_result, "decisions", {}) or {}),
            "merge_group_candidates": len(getattr(arb_result, "merge_groups", []) or []),
            "merged_segments_removed": 0,
        }
        if not segments:
            result["reason"] = result["reason"] or "no_segments"
            return result

        decisions = dict(getattr(arb_result, "decisions", {}) or {})
        merge_groups = list(getattr(arb_result, "merge_groups", []) or [])
        if not decisions and not merge_groups:
            result["reason"] = result["reason"] or "no_decisions"
            return result

        allowed_speakers = sorted(
            {
                str(getattr(seg, "speaker", "") or "").strip()
                for seg in segments
                if str(getattr(seg, "speaker", "") or "").strip()
            }
        )
        speaker_constraints_payload = self._speaker_constraints_payload(
            speaker_constraints,
            allowed_speakers,
        )
        max_changed_segments = max(
            0,
            int(round(len(segments) * self.speaker_arbitration_max_changed_ratio)),
        )
        change_runs: List[Dict[str, Any]] = []
        index = 0
        while index < len(segments):
            decision = decisions.get(index)
            if decision is None:
                index += 1
                continue
            proposed_speaker = str(getattr(decision, "speaker", "") or "").strip()
            current_speaker = str(getattr(segments[index], "speaker", "") or "").strip()
            confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
            if (
                not proposed_speaker
                or proposed_speaker not in allowed_speakers
                or proposed_speaker == current_speaker
                or confidence < self.speaker_arbitration_min_confidence
            ):
                index += 1
                continue
            run_indices = [index]
            run_confidences = [confidence]
            cursor = index + 1
            while cursor < len(segments):
                next_decision = decisions.get(cursor)
                if next_decision is None:
                    break
                next_speaker = str(getattr(next_decision, "speaker", "") or "").strip()
                old_speaker = str(getattr(segments[cursor], "speaker", "") or "").strip()
                next_confidence = float(getattr(next_decision, "confidence", 0.0) or 0.0)
                if (
                    next_speaker != proposed_speaker
                    or next_speaker not in allowed_speakers
                    or next_speaker == old_speaker
                    or next_confidence < self.speaker_arbitration_min_confidence
                ):
                    break
                run_indices.append(cursor)
                run_confidences.append(next_confidence)
                cursor += 1
            avg_confidence = sum(run_confidences) / len(run_confidences)
            score = avg_confidence + min(0.08, 0.02 * max(0, len(run_indices) - 1))
            change_runs.append(
                {
                    "indices": run_indices,
                    "speaker": proposed_speaker,
                    "avg_confidence": avg_confidence,
                    "score": score,
                }
            )
            index = cursor

        allowed_change_indices: set[int] = set()
        if max_changed_segments > 0 and change_runs:
            change_runs.sort(
                key=lambda item: (
                    -float(item.get("score", 0.0) or 0.0),
                    -len(list(item.get("indices") or [])),
                    int((list(item.get("indices") or [0]) or [0])[0]),
                )
            )
            remaining = max_changed_segments
            for run in change_runs:
                run_indices = list(run.get("indices") or [])
                if not run_indices or len(run_indices) > remaining:
                    continue
                allowed_change_indices.update(run_indices)
                remaining -= len(run_indices)
                if remaining <= 0:
                    break

        changed_segments = 0
        for index, seg in enumerate(segments):
            decision = decisions.get(index)
            current_speaker = str(getattr(seg, "speaker", "") or "").strip()
            proposed_speaker = ""
            confidence = 0.0
            highlight_terms = [
                str(item or "").strip()
                for item in list(getattr(decision, "highlight_terms", []) or [])
                if str(item or "").strip()
            ] if decision is not None else []
            if decision is not None:
                proposed_speaker = str(getattr(decision, "speaker", "") or "").strip()
                confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
            profile = (getattr(arb_result, "speaker_profiles", {}) or {}).get(
                proposed_speaker or current_speaker
            )
            speaker_role = str(getattr(profile, "role", "") or "").strip() if profile else ""

            if (
                proposed_speaker
                and proposed_speaker in allowed_speakers
                and proposed_speaker != current_speaker
                and index in allowed_change_indices
                and confidence >= self.speaker_arbitration_min_confidence
            ):
                previous_speaker = current_speaker
                current_speaker = proposed_speaker
                changed_segments += 1
                try:
                    seg.speaker = current_speaker
                except Exception:
                    pass
                for word in list(getattr(seg, "words", []) or []):
                    if not isinstance(word, dict):
                        continue
                    word["speaker"] = current_speaker
                    word["speaker_arbitrated"] = True
                    if previous_speaker:
                        word["speaker_before_arbitration"] = previous_speaker

            try:
                seg.semantic_highlights = list(highlight_terms)
            except Exception:
                pass
            try:
                seg.arbitration_confidence = confidence
            except Exception:
                pass
            if speaker_role:
                try:
                    seg.speaker_role = speaker_role
                except Exception:
                    pass

        merge_meta = self._apply_speaker_arbitration_merges(
            segments,
            arb_result,
            allowed_speakers=allowed_speakers,
        )
        merged_group_count = int(merge_meta.get("merged_group_count", 0) or 0)
        merged_segments_removed = int(
            merge_meta.get("merged_segments_removed", 0) or 0
        )

        result["applied"] = bool(changed_segments > 0 or merged_group_count > 0)
        result["changed_segments"] = changed_segments
        result["merged_group_count"] = merged_group_count
        result["merged_segments_removed"] = merged_segments_removed
        result["speaker_constraints"] = speaker_constraints_payload
        return result

    def semantic_arbitrate_segments_sync(
        self,
        segments: List[Any],
        *,
        analysis: Any,
        response_language: str = "",
        speaker_constraints: Optional[Dict[str, Any]] = None,
        timeout: float = 360.0,
        pause_checker=None,
        cancel_checker=None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "applied": False,
            "changed_segments": 0,
            "merged_group_count": 0,
            "merged_segments_removed": 0,
            "reason": "",
            "speaker_profiles": {},
            "speaker_constraints": dict(speaker_constraints or {}),
        }
        if not segments:
            result["reason"] = "no_segments"
            return result
        if not self.speaker_arbitration_enabled or self.speaker_arbiter is None:
            result["reason"] = "speaker_arbitration_disabled"
            return result
        if not self.enabled:
            result["reason"] = self.disabled_reason or "llm_disabled"
            return result

        analysis_summary = self._analysis_summary_for_arbitration(analysis)
        analysis_points = [
            str(item or "").strip()
            for item in list(getattr(analysis, "key_points", []) or [])
            if str(item or "").strip()
        ]
        if not analysis_summary and not analysis_points:
            result["reason"] = "missing_analysis_summary"
            return result

        if not self._started:
            self.start()
        if not self._started or self._loop is None:
            result["reason"] = "loop_unavailable"
            return result

        fut = asyncio.run_coroutine_threadsafe(
            self.speaker_arbiter.arbitrate(
                segments,
                analysis_summary=analysis_summary,
                analysis_points=analysis_points,
                response_language=self._normalize_language_code(response_language),
                speaker_constraints=speaker_constraints,
            ),
            self._loop,
        )

        deadline = time.time() + max(15.0, float(timeout))
        while not fut.done():
            if cancel_checker:
                try:
                    if cancel_checker():
                        fut.cancel()
                        result["reason"] = "cancelled"
                        return result
                except Exception:
                    pass
            if pause_checker:
                try:
                    pause_checker()
                except Exception:
                    pass
            if time.time() >= deadline:
                fut.cancel()
                result["reason"] = "timeout"
                return result
            time.sleep(0.1)

        try:
            arb_result = fut.result()
        except Exception as e:
            result["reason"] = f"speaker_arbitration_error:{e}"
            logger.warning(f"Speaker semantic arbitration failed: {e}")
            return result

        result.update(
            self._apply_speaker_arbitration_result(
                segments,
                arb_result,
                speaker_constraints=speaker_constraints,
            )
        )
        return result

    @staticmethod
    def _join_request_parts(parts: List[str]) -> str:
        cleaned = [str(part or "").strip() for part in parts if str(part or "").strip()]
        return "\n".join(cleaned).strip()

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        value = str(text or "").strip()
        if not value.startswith("```"):
            return value
        lines = value.splitlines()
        if not lines:
            return value
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @classmethod
    def _extract_json_payload(cls, text: str) -> Any:
        raw = str(text or "").strip()
        if not raw:
            return None

        candidates = [raw]
        stripped = cls._strip_markdown_code_fence(raw)
        if stripped and stripped not in candidates:
            candidates.insert(0, stripped)

        decoder = json.JSONDecoder()
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                pass
            for idx, ch in enumerate(candidate):
                if ch not in "[{":
                    continue
                try:
                    payload, _end = decoder.raw_decode(candidate[idx:])
                    return payload
                except Exception:
                    continue
        return None

    def _plan_optimize_groups(self, texts: List[str]) -> List[List[str]]:
        clean_texts = [str(text or "") for text in texts if str(text or "").strip()]
        if not clean_texts:
            return []

        target_chars = max(1600, int(self.optimize_batch_target_chars))
        max_segments = max(1, int(self.optimize_batch_max_segments))
        min_segments = max(2, int(self.optimize_batch_min_segments))

        groups: List[List[str]] = []
        current: List[str] = []
        current_chars = 0
        for text in clean_texts:
            text_len = len(text)
            oversized = text_len > min(self.segment_request_max_chars, target_chars)
            projected_chars = current_chars + text_len + (24 if current else 0)
            if current and (
                oversized
                or len(current) >= max_segments
                or projected_chars > target_chars
            ):
                groups.append(list(current))
                current = []
                current_chars = 0

            if oversized:
                groups.append([text])
                continue

            current.append(text)
            current_chars += text_len + 24

        if current:
            groups.append(list(current))

        if len(groups) >= 2 and len(groups[-1]) < min_segments:
            merged = list(groups[-2]) + list(groups[-1])
            merged_chars = sum(len(item) + 24 for item in merged)
            if (
                len(merged) <= max_segments
                and merged_chars <= int(target_chars * 1.15)
            ):
                groups[-2] = merged
                groups.pop()

        return groups

    def estimate_optimize_wait_timeout(self, texts: List[str]) -> float:
        groups = self._plan_optimize_groups(texts)
        if not groups:
            return 120.0
        effective_concurrency = max(
            1,
            min(
                int(self.max_concurrent or 1),
                int(self.provider_max_concurrent or 1),
                int(self.optimize_max_concurrent or 1),
            ),
        )
        waves = max(1, (len(groups) + effective_concurrency - 1) // effective_concurrency)
        per_group_budget = (
            max(12.0, min(45.0, float(self.timeout)))
            + max(0.0, float(self.retry_delay)) * max(0, int(self.max_retries))
            + 6.0
        )
        estimated = 30.0 + waves * per_group_budget
        return min(2400.0, max(180.0, estimated))

    def _build_optimize_batch_messages(
        self,
        texts: List[str],
        *,
        source_language: str,
        strict_keep_text: bool,
        infer_missing_words: bool,
    ) -> List[Dict[str, str]]:
        source_lang = source_language or "auto"
        strict_rule = (
            "Keep wording as close as possible; only fix obvious punctuation, "
            "spacing, casing, and clear ASR artifacts."
            if strict_keep_text
            else "You may lightly rewrite awkward phrases for readability."
        )
        infer_rule = (
            "If a missing short word is obvious from context, you may restore it."
            if infer_missing_words
            else "Do not add or infer missing words."
        )
        items = [
            {"index": idx + 1, "text": str(text or "")}
            for idx, text in enumerate(texts)
        ]
        prompt = (
            "Optimize each transcript segment independently in the original language.\n"
            f"Source language: {source_lang}\n"
            "Preserve speaker intent and factual meaning.\n"
            "Keep names, numbers, and key facts unchanged.\n"
            "Do not merge, split, drop, or reorder segments.\n"
            f"{strict_rule}\n"
            f"{infer_rule}\n"
            "Return JSON only with this exact shape:\n"
            '{"items":[{"index":1,"text":"..."},{"index":2,"text":"..."}]}\n\n'
            "Input JSON:\n"
            f"{json.dumps(items, ensure_ascii=False)}"
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a transcript post-editor. "
                    "Return JSON only and keep segment boundaries unchanged."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _parse_optimized_batch_response(
        self,
        raw_text: str,
        *,
        expected_count: int,
    ) -> Optional[List[str]]:
        payload = self._extract_json_payload(raw_text)
        if payload is None:
            return None

        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return None
        if len(items) != expected_count:
            return None

        parsed: List[Tuple[int, str]] = []
        for idx, item in enumerate(items, start=1):
            if isinstance(item, str):
                parsed.append((idx, str(item)))
                continue
            if not isinstance(item, dict):
                return None
            try:
                item_index = int(item.get("index", idx))
            except (TypeError, ValueError):
                item_index = idx
            parsed.append((item_index, str(item.get("text", ""))))

        parsed.sort(key=lambda part: part[0])
        texts = [text for _idx, text in parsed]
        if len(texts) != expected_count:
            return None
        return texts

    async def _optimize_group(
        self,
        texts: List[str],
        source_language: str,
        strict_keep_text: bool,
        infer_missing_words: bool,
    ) -> List[str]:
        clean_texts = [str(text or "") for text in texts]
        if not clean_texts:
            return []
        if len(clean_texts) == 1:
            return [
                await self._optimize_text(
                    clean_texts[0],
                    source_language=source_language,
                    strict_keep_text=strict_keep_text,
                    infer_missing_words=infer_missing_words,
                )
            ]

        messages = self._build_optimize_batch_messages(
            clean_texts,
            source_language=source_language,
            strict_keep_text=strict_keep_text,
            infer_missing_words=infer_missing_words,
        )
        payload_chars = self._messages_char_count(messages)
        if payload_chars > self.request_payload_hard_chars:
            midpoint = max(1, len(clean_texts) // 2)
            left = await self._optimize_group(
                clean_texts[:midpoint],
                source_language=source_language,
                strict_keep_text=strict_keep_text,
                infer_missing_words=infer_missing_words,
            )
            right = await self._optimize_group(
                clean_texts[midpoint:],
                source_language=source_language,
                strict_keep_text=strict_keep_text,
                infer_missing_words=infer_missing_words,
            )
            return left + right

        result = await self._call_api(
            messages,
            enable_thinking=False,
            request_label=f"optimize_batch_{len(clean_texts)}",
        )
        parsed = self._parse_optimized_batch_response(
            result,
            expected_count=len(clean_texts),
        )
        if parsed is not None:
            return [
                text if str(text or "").strip() else original
                for text, original in zip(parsed, clean_texts)
            ]

        logger.warning(
            "Optimize batch response parse failed; splitting batch (segments=%d, payload_chars=%d)",
            len(clean_texts),
            payload_chars,
        )
        midpoint = max(1, len(clean_texts) // 2)
        left = await self._optimize_group(
            clean_texts[:midpoint],
            source_language=source_language,
            strict_keep_text=strict_keep_text,
            infer_missing_words=infer_missing_words,
        )
        right = await self._optimize_group(
            clean_texts[midpoint:],
            source_language=source_language,
            strict_keep_text=strict_keep_text,
            infer_missing_words=infer_missing_words,
        )
        return left + right

    async def _optimize_group_with_limit(
        self,
        texts: List[str],
        source_language: str,
        strict_keep_text: bool,
        infer_missing_words: bool,
    ) -> List[str]:
        try:
            if self._optimize_semaphore is not None:
                async with self._optimize_semaphore:
                    return await self._optimize_group(
                        texts=texts,
                        source_language=source_language,
                        strict_keep_text=strict_keep_text,
                        infer_missing_words=infer_missing_words,
                    )
            return await self._optimize_group(
                texts=texts,
                source_language=source_language,
                strict_keep_text=strict_keep_text,
                infer_missing_words=infer_missing_words,
            )
        except Exception as e:
            logger.warning(f"Segment optimization batch failed, keep original text: {e}")
            return [str(text or "") for text in texts]

    async def _translate_text_once(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        part_index: int = 1,
        total_parts: int = 1,
    ) -> str:
        source_lang = source_language or "auto"
        target_lang = target_language or self.translation_target_language or "zh"
        part_hint = ""
        if total_parts > 1:
            part_hint = (
                f"This is part {part_index}/{total_parts} of one long transcript segment. "
                "Translate only this part faithfully.\n"
            )
        prompt = (
            f"{part_hint}"
            "Translate the following transcript segment.\n"
            f"Source language: {source_lang}\n"
            f"Target language: {target_lang}\n"
            "Keep meaning, speaker intent, and tone.\n"
            "Return translated text only, without any explanation.\n\n"
            f"Text:\n{text}"
        )

        result = await self._call_api(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a professional subtitle translator. "
                        "Return only the translated text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            enable_thinking=False,
            request_label=(
                f"translate_segment {part_index}/{total_parts}"
                if total_parts > 1
                else "translate_segment"
            ),
        )
        translated = str(result or "").strip()
        return translated if translated else text

    async def _translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        if not text.strip():
            return text

        pieces = self._split_text_for_request(
            text,
            max_chars=self.segment_request_max_chars,
        )
        if len(pieces) <= 1:
            return await self._translate_text_once(
                text=text,
                source_language=source_language,
                target_language=target_language,
            )

        logger.info(
            "LLM translation split long segment: chars=%d parts=%d limit=%d",
            len(text),
            len(pieces),
            self.segment_request_max_chars,
        )
        translated_parts: List[str] = []
        for idx, piece in enumerate(pieces, start=1):
            translated_parts.append(
                await self._translate_text_once(
                    text=piece,
                    source_language=source_language,
                    target_language=target_language,
                    part_index=idx,
                    total_parts=len(pieces),
                )
            )
        joined = self._join_request_parts(translated_parts)
        return joined if joined else text

    async def _translate_text_with_limit(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        try:
            if self._translation_semaphore is not None:
                async with self._translation_semaphore:
                    return await self._translate_text(
                        text=text,
                        source_language=source_language,
                        target_language=target_language,
                    )
            return await self._translate_text(
                text=text,
                source_language=source_language,
                target_language=target_language,
            )
        except Exception as e:
            logger.warning(f"Segment translation failed, keep original text: {e}")
            return text

    async def _translate_batch_async(
        self,
        texts: List[str],
        source_language: str,
        target_language: str,
    ) -> List[str]:
        if not texts:
            return []
        tasks = [
            self._translate_text_with_limit(
                text=t,
                source_language=source_language,
                target_language=target_language,
            )
            for t in texts
        ]
        return await asyncio.gather(*tasks)

    async def _optimize_text(
        self,
        text: str,
        source_language: str,
        strict_keep_text: bool = True,
        infer_missing_words: bool = True,
    ) -> str:
        if not text.strip():
            return text

        pieces = self._split_text_for_request(
            text,
            max_chars=self.segment_request_max_chars,
        )
        if len(pieces) > 1:
            logger.info(
                "LLM optimization split long segment: chars=%d parts=%d limit=%d",
                len(text),
                len(pieces),
                self.segment_request_max_chars,
            )
            optimized_parts: List[str] = []
            for idx, piece in enumerate(pieces, start=1):
                optimized_parts.append(
                    await self._optimize_text(
                        piece,
                        source_language=source_language,
                        strict_keep_text=strict_keep_text,
                        infer_missing_words=infer_missing_words,
                    )
                )
            joined = self._join_request_parts(optimized_parts)
            return joined if joined else text

        source_lang = source_language or "auto"
        strict_rule = (
            "Keep wording as close as possible; only fix obvious punctuation, "
            "spacing, casing, and clear ASR artifacts."
            if strict_keep_text
            else "You may lightly rewrite awkward phrases for readability."
        )
        infer_rule = (
            "If a missing short word is obvious from context, you may restore it."
            if infer_missing_words
            else "Do not add or infer missing words."
        )
        prompt = (
            "Optimize the following transcript segment in the original language.\n"
            f"Source language: {source_lang}\n"
            "Preserve speaker intent and factual meaning.\n"
            "Keep names, numbers, and key facts unchanged.\n"
            f"{strict_rule}\n"
            f"{infer_rule}\n"
            "Return optimized text only, without any explanation.\n\n"
            f"Text:\n{text}"
        )

        result = await self._call_api(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a transcript post-editor. "
                        "Keep meaning intact and return only edited text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            enable_thinking=False,
            request_label="optimize_segment",
        )
        optimized = str(result or "").strip()
        return optimized if optimized else text

    async def _optimize_text_with_limit(
        self,
        text: str,
        source_language: str,
        strict_keep_text: bool,
        infer_missing_words: bool,
    ) -> str:
        try:
            if self._optimize_semaphore is not None:
                async with self._optimize_semaphore:
                    return await self._optimize_text(
                        text=text,
                        source_language=source_language,
                        strict_keep_text=strict_keep_text,
                        infer_missing_words=infer_missing_words,
                    )
            return await self._optimize_text(
                text=text,
                source_language=source_language,
                strict_keep_text=strict_keep_text,
                infer_missing_words=infer_missing_words,
            )
        except Exception as e:
            logger.warning(f"Segment optimization failed, keep original text: {e}")
            return text

    async def _optimize_batch_async(
        self,
        texts: List[str],
        source_language: str,
        strict_keep_text: bool,
        infer_missing_words: bool,
    ) -> List[str]:
        if not texts:
            return []
        groups = self._plan_optimize_groups(texts)
        logger.info(
            "LLM optimization batch plan: segments=%d groups=%d target_chars=%d max_segments=%d",
            len(texts),
            len(groups),
            int(self.optimize_batch_target_chars),
            int(self.optimize_batch_max_segments),
        )
        tasks = [
            self._optimize_group_with_limit(
                texts=group,
                source_language=source_language,
                strict_keep_text=strict_keep_text,
                infer_missing_words=infer_missing_words,
            )
            for group in groups
        ]
        grouped_results = await asyncio.gather(*tasks)
        flattened: List[str] = []
        for group_result in grouped_results:
            flattened.extend(list(group_result or []))
        return flattened

    def translate_segments_sync(
        self,
        segments: List[Any],
        source_language: str = "",
        target_language: str = "",
        timeout: float = 300.0,
        pause_checker=None,
        cancel_checker=None,
    ) -> Dict[str, Any]:
        source_lang = self._normalize_language_code(source_language)
        target_lang = self._normalize_language_code(
            target_language or self.translation_target_language
        )
        if not target_lang:
            target_lang = "zh"

        result: Dict[str, Any] = {
            "applied": False,
            "translated_count": 0,
            "skipped_count": 0,
            "source_language": source_lang,
            "target_language": target_lang,
            "reason": "",
        }

        if not segments:
            result["reason"] = "no_segments"
            return result
        if not self.translation_enabled:
            result["reason"] = "translation_disabled"
            return result
        if not self.enabled:
            result["reason"] = self.disabled_reason or "llm_disabled"
            return result

        if (
            self.translation_skip_same_language
            and source_lang
            and source_lang == target_lang
        ):
            non_empty = sum(
                1 for seg in segments if str(getattr(seg, "text", "") or "").strip()
            )
            result["skipped_count"] = non_empty
            result["reason"] = "same_language"
            return result

        text_indices: List[int] = []
        text_payload: List[str] = []
        for idx, seg in enumerate(segments):
            text = str(getattr(seg, "text", "") or "")
            if not text.strip():
                continue
            text_indices.append(idx)
            text_payload.append(text)

        if not text_payload:
            result["reason"] = "no_text"
            return result

        if not self._started:
            self.start()
        if not self._started or self._loop is None:
            result["reason"] = "loop_unavailable"
            return result

        fut = asyncio.run_coroutine_threadsafe(
            self._translate_batch_async(
                texts=text_payload,
                source_language=source_lang or "auto",
                target_language=target_lang,
            ),
            self._loop,
        )

        deadline = time.time() + max(5.0, float(timeout))
        while not fut.done():
            if cancel_checker:
                try:
                    if cancel_checker():
                        fut.cancel()
                        result["reason"] = "cancelled"
                        return result
                except Exception:
                    pass
            if pause_checker:
                try:
                    pause_checker()
                except Exception:
                    pass
            if time.time() >= deadline:
                fut.cancel()
                result["reason"] = "timeout"
                return result
            time.sleep(0.1)

        try:
            translated_texts = list(fut.result())
        except Exception as e:
            result["reason"] = f"translate_error:{e}"
            logger.warning(f"Translation failed: {e}")
            return result

        translated_count = 0
        for idx, new_text in zip(text_indices, translated_texts):
            seg = segments[idx]
            original_text = str(getattr(seg, "text", "") or "")
            final_text = str(new_text or "")
            if not final_text.strip():
                final_text = original_text
            if final_text != original_text:
                translated_count += 1
            try:
                seg.text = final_text
                seg.language = target_lang
            except Exception:
                pass

        result["applied"] = True
        result["translated_count"] = translated_count
        result["skipped_count"] = max(0, len(text_payload) - translated_count)
        result["reason"] = "ok"
        return result

    def optimize_segments_sync(
        self,
        segments: List[Any],
        source_language: str = "",
        timeout: float = 300.0,
        pause_checker=None,
        cancel_checker=None,
    ) -> Dict[str, Any]:
        source_lang = self._normalize_language_code(source_language)
        result: Dict[str, Any] = {
            "applied": False,
            "optimized_count": 0,
            "skipped_count": 0,
            "source_language": source_lang,
            "reason": "",
        }

        if not segments:
            result["reason"] = "no_segments"
            return result
        if not self.optimize_enabled:
            result["reason"] = "optimize_disabled"
            return result
        if not self.enabled:
            result["reason"] = self.disabled_reason or "llm_disabled"
            return result

        text_indices: List[int] = []
        text_payload: List[str] = []
        for idx, seg in enumerate(segments):
            text = str(getattr(seg, "text", "") or "")
            if not text.strip():
                continue
            text_indices.append(idx)
            text_payload.append(text)

        if not text_payload:
            result["reason"] = "no_text"
            return result

        if not self._started:
            self.start()
        if not self._started or self._loop is None:
            result["reason"] = "loop_unavailable"
            return result

        fut = asyncio.run_coroutine_threadsafe(
            self._optimize_batch_async(
                texts=text_payload,
                source_language=source_lang or "auto",
                strict_keep_text=self.optimize_strict_keep_text,
                infer_missing_words=self.optimize_infer_missing_words,
            ),
            self._loop,
        )

        deadline = time.time() + max(5.0, float(timeout))
        while not fut.done():
            if cancel_checker:
                try:
                    if cancel_checker():
                        fut.cancel()
                        result["reason"] = "cancelled"
                        return result
                except Exception:
                    pass
            if pause_checker:
                try:
                    pause_checker()
                except Exception:
                    pass
            if time.time() >= deadline:
                fut.cancel()
                result["reason"] = "timeout"
                return result
            time.sleep(0.1)

        try:
            optimized_texts = list(fut.result())
        except Exception as e:
            result["reason"] = f"optimize_error:{e}"
            logger.warning(f"Optimization failed: {e}")
            return result

        optimized_count = 0
        for idx, new_text in zip(text_indices, optimized_texts):
            seg = segments[idx]
            original_text = str(getattr(seg, "text", "") or "")
            final_text = str(new_text or "")
            if not final_text.strip():
                final_text = original_text
            if final_text != original_text:
                optimized_count += 1
            try:
                seg.text = final_text
                if source_lang:
                    seg.language = source_lang
            except Exception:
                pass

        result["applied"] = True
        result["optimized_count"] = optimized_count
        result["skipped_count"] = max(0, len(text_payload) - optimized_count)
        result["reason"] = "ok"
        return result
