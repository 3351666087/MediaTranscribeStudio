"""
ui_app.py - 现代化桌面端转录流水线用户界面（深度优化版）
"""

import copy
import gc
import ctypes
import ctypes.util
import functools
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time
import yaml
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    QSize,
    Qt,
    QtMsgType,
    QThread,
    QTimer,
    QUrl,
    Slot,
    qInstallMessageHandler,
)
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QKeySequence, QMovie, QPalette, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QSizePolicy,
)

from config import ALL_MEDIA_EXTENSIONS, Config, DEFAULT_CONFIG
from output_layout import (
    RUNTIME_ARTIFACTS_DIRNAME,
    resolve_output_subdir,
    resolve_runtime_artifact_path,
)
from runtime_paths import APP_ROOT, resolve_app_writable_path
from utils import smart_empty_cache


def _get_nested(data: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value = data
    for part in dotted_key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _set_nested(data: Dict[str, Any], dotted_key: str, value: Any):
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        node = current.get(part)
        if not isinstance(node, dict):
            node = {}
            current[part] = node
        current = node
    current[parts[-1]] = value


DEFAULT_HF_TOKEN = str(
    ((DEFAULT_CONFIG.get("asr", {}) or {}).get("faster_whisper", {}) or {}).get(
        "hf_token", ""
    )
).strip()


_QT_SUPPRESSED_MESSAGE_PATTERNS = (
    "QPainter::begin: A paint device can only be painted by one painter at a time.",
    "QPainter::translate: Painter not active",
    "QPainter::worldTransform: Painter not active",
    "QPainter::setWorldTransform: Painter not active",
    "QPropertyAnimation::updateState (opacity): Changing state of an animation without target",
    "Unknown property box-shadow",
    "libtorchcodec",
    "onelogger disabled",
)
_qt_previous_message_handler = None
_qt_message_filter_installed = False


def _qt_message_handler(mode, context, message) -> None:
    text = str(message or "")
    lower_text = text.lower()
    if any(pattern.lower() in lower_text for pattern in _QT_SUPPRESSED_MESSAGE_PATTERNS):
        return

    previous = _qt_previous_message_handler
    if previous is not None:
        try:
            previous(mode, context, message)
            return
        except Exception:
            pass

    is_err = mode in {QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg}
    stream = sys.stderr if is_err else sys.stdout
    try:
        print(text, file=stream)
    except Exception:
        pass


def _install_qt_message_filter() -> None:
    global _qt_previous_message_handler, _qt_message_filter_installed
    if _qt_message_filter_installed:
        return
    try:
        _qt_previous_message_handler = qInstallMessageHandler(_qt_message_handler)
        _qt_message_filter_installed = True
    except Exception:
        _qt_previous_message_handler = None


@functools.lru_cache(maxsize=1)
def _macos_objc_runtime():
    if sys.platform != "darwin":
        return None
    objc_path = ctypes.util.find_library("objc")
    appkit_path = ctypes.util.find_library("AppKit")
    if not objc_path or not appkit_path:
        return None
    try:
        ctypes.cdll.LoadLibrary(appkit_path)
        objc = ctypes.cdll.LoadLibrary(objc_path)
    except Exception:
        return None

    get_class = objc.objc_getClass
    get_class.restype = ctypes.c_void_p
    get_class.argtypes = [ctypes.c_char_p]

    register_sel = objc.sel_registerName
    register_sel.restype = ctypes.c_void_p
    register_sel.argtypes = [ctypes.c_char_p]

    send_id = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )(("objc_msgSend", objc))
    send_id_id = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )(("objc_msgSend", objc))
    send_double = ctypes.CFUNCTYPE(
        ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p
    )(("objc_msgSend", objc))
    send_void_bool = ctypes.CFUNCTYPE(
        None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool
    )(("objc_msgSend", objc))
    send_void_integer = ctypes.CFUNCTYPE(
        None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long
    )(("objc_msgSend", objc))
    return (
        get_class,
        register_sel,
        send_id,
        send_id_id,
        send_double,
        send_void_bool,
        send_void_integer,
    )


from mts_ui.event_feed import BurstEventFeed
from mts_ui.onboarding import StepOnboardingOverlay
from mts_ui.widgets import (
    AnimatedSticker,
    ColorToggleButton,
    DropArea,
    HoldToDeleteButton,
    PictureBackgroundFrame,
)
from mts_ui.worker import PipelineWorker, PosteriorFusionTrainerWorker


class MainWindow(QMainWindow):
    """Main application window."""

    SWITCH_SPECS: List[Tuple[str, str, str]] = [
        ("启用 NeMo MSDD 分角色", "asr.nemo_msdd.enabled", "说话人分离与角色标注"),
        ("启用 LLM 分析", "llm.enabled", "摘要、要点、待办提示"),
        ("优化语言（仅纠错）", "llm.optimize_language.enabled", "保持原意，仅修正错别字与病句"),
        ("启用翻译模块", "translation.enabled", "将转录文本翻译成目标语言"),
        ("启用 Taichi 加速", "taichi.enabled", "预处理加速"),
        ("生成 HTML 报告", "report.generate_html", "输出可视化时间轴"),
        ("生成 PDF 报告", "report.generate_pdf", "输出可打印报告"),
        ("启用文本回写视频", "video_text_overlay.enabled", "将转录文本烧录回原视频"),
        ("显示进度条", "logging.show_progress", "终端 CLI 进度显示"),
        ("硬件媒体解码", "audio.use_gpu_decode", "macOS 优先使用 VideoToolbox"),
        ("音频预处理加速", "audio.use_gpu_audio_ops", "macOS 优先使用 Apple Metal (MPS)"),
        ("Pin Memory 优化", "audio.pin_memory", "优化主机到显存的数据传输"),
        ("无阻塞传输", "audio.non_blocking", "降低传输阻塞开销"),
        ("启用 Torch Compile", "performance.use_torch_compile", "提升模型推理性能"),
        ("启用 CUDA Graph", "performance.use_cuda_graph", "稳定高频推理开销"),
        ("cuDNN 基准测试", "performance.use_cudnn_benchmark", "固定输入尺寸时提速"),
    ]

    MACOS_HIDDEN_SWITCH_KEYS = {
        "audio.pin_memory",
        "audio.non_blocking",
        "performance.use_cuda_graph",
        "performance.use_cudnn_benchmark",
    }

    OUTPUT_KEYS = [
        ("txt_path", "纯文本 (TXT)"),
        ("json_path", "数据格式 (JSON)"),
        ("html_path", "网页版 (HTML)"),
        ("pdf_path", "打印版 (PDF)"),
        ("subtitle_srt_path", "字幕文件 (SRT)"),
        ("subtitle_ass_path", "字幕文件 (ASS)"),
        ("burned_video_path", "回写视频 (字幕烧录)"),
    ]

    FINAL_OUTPUT_SUFFIXES = {
        ".txt",
        ".json",
        ".html",
        ".htm",
        ".pdf",
        ".srt",
        ".ass",
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".webm",
        ".mp3",
        ".wav",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
    }

    LANGUAGE_CHOICES: List[Tuple[str, str]] = [
        ("自动检测（所有语言）", "auto"),
        ("强制中文", "zh"),
        ("强制英文", "en"),
    ]

    TRANSLATION_LANGUAGE_CHOICES: List[Tuple[str, str]] = [
        ("中文", "zh"),
        ("英文", "en"),
        ("粤语", "yue"),
        ("日语", "ja"),
        ("韩语", "ko"),
        ("法语", "fr"),
        ("德语", "de"),
        ("西班牙语", "es"),
        ("俄语", "ru"),
    ]

    READABLE_OUTPUT_SUFFIXES = {
        ".txt",
        ".json",
        ".html",
        ".htm",
        ".md",
        ".log",
        ".yaml",
        ".yml",
        ".csv",
        ".srt",
        ".ass",
    }

    OUTPUT_KIND_BY_SUFFIX = {
        ".txt": "TXT",
        ".json": "JSON",
        ".html": "HTML",
        ".htm": "HTML",
        ".pdf": "PDF",
        ".srt": "SRT",
        ".ass": "ASS",
        ".mp4": "MP4",
        ".mkv": "MKV",
        ".mov": "MOV",
        ".avi": "AVI",
        ".webm": "WEBM",
    }

    CONFIG_HELP_TEXT = (
        "⚙️ 配置修改将自动保存在 config.yaml。"
        "支持 DASHSCOPE_API_KEY / HF_TOKEN 环境变量覆盖。"
        "支持 NeMo MSDD 分角色人数自动检测/手动固定。"
        "支持可选翻译模块与目标语言配置。"
    )

    USAGE_GUIDE_TEXT = (
        "快速上手\n"
        "\n"
        "1. 点击“添加文件”选择音频或视频文件，也可以直接拖拽到窗口。\n"
        "2. 在“系统运行配置”里先设置识别语言、翻译目标（可选）和密钥（如需）。\n"
        "3. 点击“开始处理”，任务会按队列顺序执行，并在日志区显示进度。\n"
        "4. 处理完成后，在“处理结果产物”中双击可打开目录或文件。\n"
        "5. 如需释放空间，可使用“清理临时文件”或“长按全清理”。\n"
        "\n"
        "配置建议\n"
        "\n"
        "- 新手建议先保持默认，仅修改“识别语言 / 翻译目标 / 是否启用翻译”。\n"
        "- 机器内存紧张或模型频繁回退时，降低“并发文件数”并保留默认保护阈值。\n"
        "- 分角色效果不稳定时，可在高级配置中限制最少/最多说话人数。\n"
        "- 视频字幕回写失败时，优先检查 FFmpeg 路径、编码器配置和输出容器格式。\n"
        "\n"
        "快捷键（可直接操作）\n"
        "\n"
        "- Ctrl+O：添加文件\n"
        "- Ctrl+Enter：开始处理\n"
        "- Space：暂停 / 恢复任务\n"
        "- Ctrl+L：显示 / 隐藏调试日志\n"
        "- Ctrl+Shift+L：清空日志记录\n"
        "- Delete：删除当前选中的队列文件\n"
        "- Ctrl+Shift+T：清理临时文件\n"
        "- Ctrl+,：打开 / 关闭高级配置\n"
        "- F1：打开 / 关闭使用说明\n"
        "\n"
        "常见问题\n"
        "\n"
        "- 处理中按钮被禁用：表示当前有工序正在运行，等待完成后会自动恢复。\n"
        "- 模型下载慢：可配置 HF Token，并检查网络环境。\n"
        "- Posterior Fusion 训练助手：点配置区的独立弹窗按钮；开启样本导出后会同步打开“自动跑两次”，先导样本训练，再自动第二遍。\n"
        "- 关闭窗口前会自动尝试保存当前配置改动到 config.yaml。\n"
    )

    POSTERIOR_FUSION_HELP_TEXT = (
        "Posterior Fusion 训练助手\n"
        "\n"
        "这套按钮是给小白准备的，目标是把“融合校准”变成固定三步：\n"
        "\n"
        "1. 先点“开启样本导出”。\n"
        "   这会同步打开“自动跑两次”；你正常点击“开始处理”后，程序会先导出 posterior fusion 的帧级样本。\n"
        "\n"
        "2. 第一遍结束后会自动读取 RTTM 并训练校准器。\n"
        "   样本目录里会出现 `.json + .npz` 文件；RTTM 目录里放同名或同前缀的 `.rttm` 文件。\n"
        "\n"
        "3. 训练完成后会自动重跑第二遍。\n"
        "   第二遍会直接使用新校准器，不需要你手动回来再点一次开始。\n"
        "\n"
        "如果你不想自动双跑，也可以单独关闭“自动跑两次”，只保留样本导出和手动训练。\n"
        "\n"
        "手动点“开始训练校准器”时，程序会自动联合搜索活动阈值、主标签损失权重、活动损失权重和 overlap 强化权重，保存成一个校准器文件。\n"
        "\n"
        "训练完成后会发生什么\n"
        "- 新的校准器文件会写到“校准器输出”。\n"
        "- 主流程下次运行时会自动读取它，不需要手工拷贝参数。\n"
        "- 如果你的开发集里 overlap 较多，新校准器会更重视双说话人帧，而不是只顾单说话人准确率。\n"
        "\n"
        "最常见的失败原因\n"
        "- 样本目录里没有 `.json/.npz`：说明你还没开启样本导出，或还没跑开发集。\n"
        "- RTTM 对不上：文件名要尽量和源音频同名或同前缀。\n"
        "- 开发集太少：建议至少准备几段真实双人/多人音频，最好包含重叠讲话。"
    )

    CONFIG_ITEM_HELP: Dict[str, str] = {
        "ui.recognition_language": (
            "设置语音识别时的语言策略。\n\n"
            "如何选择（新手版）\n"
            "- 不确定音频里是什么语言、或者可能中英混说：选“自动检测”。\n"
            "- 明确全程基本都是中文：选“强制中文”，通常更稳、更快。\n"
            "- 明确全程基本都是英文：选“强制英文”，可减少误判。\n\n"
            "影响\n"
            "- 自动检测更灵活，但会增加一次语言判断，极短音频或噪声较大的片段可能误判。\n"
            "- 强制语言会减少模型搜索范围，通常速度更快，也更容易保持术语稳定。\n\n"
            "常见现象\n"
            "- 说中文却被识别成英文：优先改成“强制中文”。\n"
            "- 中英文混说但总是偏一种语言：先用自动检测，再看是否需要拆文件处理。"
        ),
        "translation.target_language": (
            "设置翻译输出的目标语言。仅在“启用翻译模块”打开时生效。\n\n"
            "新手建议\n"
            "- 只需要原文转写：关闭翻译模块即可，这一项不用管。\n"
            "- 面向中文阅读：目标语言选中文。\n"
            "- 做国际分享/字幕二次编辑：按受众选择英文或其他语言。\n\n"
            "说明\n"
            "- 该项不会影响“语音识别”本身，只影响识别完成后的翻译文本。\n"
            "- 如果翻译结果不理想，可先确认识别原文是否准确，再调整翻译模型/提示词。"
        ),
        "llm.api_key": (
            "DashScope（通义千问）API Key，用于摘要、要点提取、待办分析、部分翻译增强等 LLM 功能。\n\n"
            "什么时候必须填\n"
            "- 你要用 LLM 分析功能（摘要/要点/行动项）时需要。\n"
            "- 只做纯转录（不启用 LLM、不启用相关增强）时可以留空。\n\n"
            "新手排查\n"
            "- 提示鉴权失败/401：检查 Key 是否复制完整、前后是否有空格。\n"
            "- 公司网络环境下调用失败：先确认网络能访问对应云服务。\n"
            "- 明明填了仍报错：检查是否被环境变量覆盖（程序支持环境变量覆盖配置）。"
        ),
        "asr.faster_whisper.hf_token": (
            "Hugging Face Token，用于访问需要授权的模型或提升下载稳定性/速度。\n\n"
            "你什么时候需要它\n"
            "- 首次下载模型较慢、失败较多时。\n"
            "- 使用需要授权访问的模型时。\n"
            "- 已经把模型缓存到本机并且当前功能都正常时，通常可以暂时不填。\n\n"
            "新手建议\n"
            "- 先不填也能试跑；如果模型下载报权限/限流/连接问题，再补填。\n"
            "- Token 不要公开分享，建议只保存在本机配置或环境变量。"
        ),
        "asr.nemo_msdd.enabled": (
            "是否启用说话人分角色（Speaker Diarization）。开启后，结果里会尝试区分“谁在说话”，例如 A / B / C。\n\n"
            "适合场景\n"
            "- 会议录音、访谈、播客、双人对话、客服通话。\n"
            "- 需要按说话人整理纪要、统计发言内容的场景。\n\n"
            "不建议开启的场景\n"
            "- 单人讲话（如课程录屏、个人口播）：开启后收益很小，耗时会增加。\n"
            "- 噪声很大或大量背景音乐：分角色稳定性可能下降。\n\n"
            "程序实际路径说明\n"
            "- 开启后会按顺序尝试多种分角色方案（NeMo / pyannote.audio / torchaudio 回退）。\n"
            "- 你可以在日志里看到“实际命中的分角色路径”。"
        ),
        "asr.nemo_msdd.speaker_mode": (
            "设置分角色时“说话人数”的策略。\n\n"
            "自动检测说话人数（推荐给新手）\n"
            "- 适合不知道有几个人讲话的录音。\n"
            "- 程序会在你设置的最少/最多人数范围内尝试判断。\n"
            "- 更省心，但耗时通常比固定人数略高。\n\n"
            "固定说话人数\n"
            "- 适合你非常确定人数的场景（例如明确是 2 人访谈、3 人会议）。\n"
            "- 通常更稳定，也可能更快。\n"
            "- 如果填错人数（比如实际 3 人却填 2），就容易把两个人合并成一个角色。"
        ),
        "asr.nemo_msdd.num_speakers": (
            "当“说话人数模式”选择“固定说话人数”时使用这个值。\n\n"
            "怎么填（新手版）\n"
            "- 双人访谈：填 2\n"
            "- 三人会议（发言都比较均衡）：填 3\n"
            "- 不确定：不要硬填，改用“自动检测说话人数”。\n\n"
            "常见错误\n"
            "- 填得过大：一个人可能被拆成多个角色。\n"
            "- 填得过小：多个人会被合并成同一个角色。\n"
            "- 录音里有人只说一句话：即使人数填对，也可能被弱化或合并。"
        ),
        "audio.overlap_sec": (
            "音频分块之间的重叠时长（秒）。程序把长音频切块处理时，会让相邻块保留一小段重叠，减少句子被截断。\n\n"
            "新手建议\n"
            "- 保持默认值通常最好。\n"
            "- 如果发现句子经常在块边界被切断、前后句不连贯，可适当增大一点。\n"
            "- 如果追求速度且结果已经稳定，可适当减小一点。\n\n"
            "注意\n"
            "- 重叠越大，重复计算越多，处理时间会增加。\n"
            "- 过小可能导致边界处漏词、断句奇怪。"
        ),
        "asr.nemo_msdd.min_speakers": (
            "自动分角色模式下的“最少说话人数”下限。\n\n"
            "作用\n"
            "- 告诉程序：这段录音至少有多少人说话。\n"
            "- 可以缩小搜索范围，提高稳定性并减少误判。\n\n"
            "新手建议\n"
            "- 不确定时保持默认 1。\n"
            "- 确定不是单人（例如双人访谈），可以设为 2，让结果更稳。\n"
            "- 不要把下限设得过高，否则会强行拆出不存在的角色。"
        ),
        "asr.nemo_msdd.max_speakers": (
            "自动分角色模式下的“最多说话人数”上限。\n\n"
            "作用\n"
            "- 告诉程序：这段录音最多不太可能超过多少人。\n"
            "- 上限越合理，自动判断越容易稳定。\n\n"
            "新手建议\n"
            "- 普通会议/访谈先用 4~8 的范围通常够用。\n"
            "- 如果实际就 2 人对话，可以把上限收紧到 2 或 3。\n"
            "- 上限设太大可能增加耗时，也更容易把噪声/语气变化误当成新角色。"
        ),
        "performance.max_concurrent_files": (
            "同时处理的文件数量（并发文件数）。\n\n"
            "怎么理解\n"
            "- 值越大，总吞吐可能更高，但会同时占用更多内存和磁盘带宽。\n"
            "- 值越小，单任务更稳，特别适合长音频或内存较紧的机器。\n\n"
            "新手建议\n"
            "- 先从 1 开始，确认稳定后再尝试 2。\n"
            "- 如果出现内存不足（OOM）、程序变慢或频繁回退模型，就把它调小。\n"
            "- 批量跑很多短音频时，适当增大可能更有收益。"
        ),
        "performance.gpu_memory_fraction": (
            "资源保护比例（0~1）。主要用于控制 CUDA 机器上的显存占用；在 macOS 上通常保留默认即可。\n\n"
            "怎么调（新手版）\n"
            "- 默认值一般可用。\n"
            "- 容易 OOM 时：调低到 0.8~0.9。\n"
            "- 专用机器、只跑这个程序且资源充足时：可适当提高，但不建议直接拉满到 1.0。\n\n"
            "常见现象\n"
            "- 太高：速度可能快一点，但更容易 OOM。\n"
            "- 太低：更稳，但可能触发更保守的回退策略，整体耗时变长。"
        ),
        "video_text_overlay.renderer": (
            "字幕回写引擎选择。\n\n"
            "当前仅保留 FFmpeg。\n"
            "- 兼容性最好，环境要求最低。\n"
            "- 程序会自动选择当前机器可用的编码器；macOS 优先使用 VideoToolbox。\n\n"
            "FFmpeg\n"
            "- 兼容性最好，环境要求最低。\n"
            "- 速度取决于可用编码器和机器环境。"
        ),
        "video_text_overlay.style_effect": (
            "字幕视觉效果。\n\n"
            "自动（推荐）\n"
            "- 有词级时间戳时使用逐词高亮。\n"
            "- 没有词级时间戳时自动退到关键词变色。\n\n"
            "逐词高亮\n"
            "- 更接近 TikTok / Shorts 的 karaoke 风格。\n"
            "- 依赖 faster-whisper 的词级时间戳。\n\n"
            "关键词变色\n"
            "- 不依赖词级时间戳。\n"
            "- 更稳，但动效弱一些。"
        ),
        "video_text_overlay.output_suffix": (
            "回写视频文件名后缀，用于区分原视频与字幕视频。\n\n"
            "示例\n"
            "- 原文件：meeting.mp4\n"
            "- 后缀：.captioned\n"
            "- 输出：meeting.captioned.mp4\n\n"
            "建议\n"
            "- 建议保留点号前缀（如 .captioned）。\n"
            "- 不要和原文件同名，避免覆盖原始素材。"
        ),
        "video_text_overlay.font_name": (
            "字幕字体名称（必须是系统已安装字体）。\n\n"
            "建议\n"
            "- 中文场景优先用“Microsoft YaHei”或“微软雅黑”。\n"
            "- 字体名写错时会回退默认字体，可能导致字形和排版变化。\n\n"
            "排查\n"
            "- 结果字体不对：先确认目标系统是否安装该字体。"
        ),
        "video_text_overlay.font_size": (
            "字幕字号（像素）。\n\n"
            "参考区间\n"
            "- 1080p 短视频常用 18~24。\n"
            "- 4K 常用 26~38。\n\n"
            "风险\n"
            "- 太小：手机端难以阅读。\n"
            "- 太大：容易遮挡画面主体。"
        ),
        "video_text_overlay.max_line_chars": (
            "单行最大字符数，超过自动换行。\n\n"
            "建议\n"
            "- 中文短视频常用 16~22。\n"
            "- 英文会按更窄字符宽度自动折算，可比中文适当更长。\n\n"
            "风险\n"
            "- 值过大：单行太长，阅读压力大。\n"
            "- 值过小：换行过于频繁，画面跳动感增强。"
        ),
        "video_text_overlay.bottom_margin_px": (
            "字幕到底边的像素距离。\n\n"
            "建议\n"
            "- 常见范围 44~84。\n"
            "- 有播放器控制栏遮挡时可适当增大。\n\n"
            "风险\n"
            "- 过小会贴底，可能被播放器 UI 遮住。\n"
            "- 过大会抬得太高，干扰主体画面。"
        ),
        "video_text_overlay.ffmpeg_video_codec": (
            "FFmpeg 回写时使用的视频编码器。\n\n"
            "常见选项\n"
            "- h264_videotoolbox：macOS 原生硬件编码，速度快，兼容较好。\n"
            "- hevc_videotoolbox：压缩率更高，但对播放端要求更高。\n"
            "- libx264：CPU 编码，最通用但通常更慢。\n\n"
            "建议\n"
            "- macOS 优先选 VideoToolbox；兼容优先选 h264。"
        ),
        "video_text_overlay.ffmpeg_crf": (
            "FFmpeg 质量参数（CRF）。仅对 x264/x265 等 CRF 路径生效。\n\n"
            "规则\n"
            "- 数值越小，画质越高，文件越大。\n"
            "- 数值越大，画质越低，文件越小。\n\n"
            "建议区间\n"
            "- 18~23：画质与体积平衡较好。"
        ),
        "video_text_overlay.ffmpeg_path": (
            "自定义 ffmpeg 可执行文件路径。\n\n"
            "留空时\n"
            "- 使用系统 PATH 中的 ffmpeg，或程序内置检测结果。\n\n"
            "建议\n"
            "- 多版本共存时可显式指定，避免命令冲突。\n"
            "- 路径错误会导致回写阶段失败，可在日志中核对实际调用路径。"
        ),
        "llm.optimize_language.enabled": (
            "开启后会在转写/翻译后做“仅纠错”优化。\n\n"
            "处理原则\n"
            "- 尽量保持原文和原意不变。\n"
            "- 仅修正常见错别字、标点和明显病句。\n"
            "- 仅在必要时补全极少量功能词，增强可读性。\n\n"
            "不会做的事\n"
            "- 不会翻译、改写、扩写或总结正文。"
        ),
    }

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.is_macos = sys.platform == "darwin"
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("glassWindow", True)
        self.setWindowTitle("AI 语音转录工作站")
        self._root_layout: Optional[QVBoxLayout] = None
        self._root_base_margins = (18, 18, 18, 18)
        self._macos_safe_area_handle = None
        self._macos_native_titlebar_ready = False
        if self.is_macos:
            self._enable_macos_native_titlebar_hints()
        self.resize(1500, 950)
        self.setMinimumSize(1200, 800)

        self.config_path = config_path
        self.config = Config(config_path=config_path)

        self._language_dirty = False
        self._translation_dirty = False
        self._dashscope_dirty = False
        self._hf_token_dirty = False
        self._switch_dirty = False
        self._advanced_dirty = False
        self._status_state = "info"
        self._is_dark_theme = False
        self._progress_total = 0
        self._progress_done = 0
        self._current_file_percent = 0.0
        self._overall_percent = 0.0
        self._download_inflight = False
        self._download_last_percent: Optional[float] = None
        self._download_started_at = 0.0
        self._download_last_event_at = 0.0
        self._download_anim_frame = 0
        self._last_file_progress_label = ""
        self._last_pipeline_step = ""
        self._accent_color = QColor("#4fa2ff")
        self._log_buffer: Deque[str] = deque()
        self._log_flush_limit = 28
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(34)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        self._download_ui_timer = QTimer(self)
        self._download_ui_timer.setInterval(420)
        self._download_ui_timer.timeout.connect(self._tick_download_progress_ui)
        self._progress_pet_phase = 0.0
        self._progress_pet_timer = QTimer(self)
        self._progress_pet_timer.setInterval(48)
        self._progress_pet_timer.timeout.connect(self._tick_progress_pet_motion)
        self._progress_pet_movie: Optional[QMovie] = None
        self._progress_pet_pos = [0.0, 0.0]
        self._progress_pet_roam_target = [0.0, 0.0]
        self._progress_pet_return_speed = 0.24
        self._progress_pet_roam_speed = 0.13
        self._progress_pet_roam_pause_ticks = 0
        self._progress_pet_last_mode = ""
        self.background_root: Optional[PictureBackgroundFrame] = None

        self.selected_files: List[str] = []
        self.file_items: Dict[str, QListWidgetItem] = {}
        self.running = False
        self.paused = False
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[PipelineWorker] = None
        self.posterior_trainer_thread: Optional[QThread] = None
        self.posterior_trainer_worker: Optional[PosteriorFusionTrainerWorker] = None
        self._animations: List[QPropertyAnimation] = []
        self._widget_fade_anims: Dict[int, QPropertyAnimation] = {}
        self._dialog_fade_anims: Dict[int, QPropertyAnimation] = {}
        self.advanced_dialog: Optional[QDialog] = None
        self.posterior_dialog: Optional[QDialog] = None
        self.usage_help_dialog: Optional[QDialog] = None
        self._layout_sync_pending = False
        self._splitter_layout_initialized = False
        self._pause_toggle_cooldown = False
        self._ui_action_locks: Set[str] = set()
        self._posterior_auto_cycle_phase = ""
        self._posterior_auto_cycle_first_pass_calibrator_path = ""
        self._switch_row_widgets: List[QWidget] = []
        self._switch_buttons: List[ColorToggleButton] = []
        self._grid_help_label_widgets: List[QWidget] = []
        self._switch_grid_columns = 0
        self._current_step = 1
        self._step_buttons: List[QPushButton] = []
        self._step_transition_anim: Optional[QPropertyAnimation] = None
        self._tour_overlay: Optional[StepOnboardingOverlay] = None
        self._tour_pending = not bool(
            _get_nested(self.config._data, "ui.onboarding.stepper_v1_done", False)
        )
        self._tour_shown = False
        self._resize_reflow_timer = QTimer(self)
        self._resize_reflow_timer.setSingleShot(True)
        self._resize_reflow_timer.setInterval(28)
        self._resize_reflow_timer.timeout.connect(self._sync_responsive_layout)
        self._theme_refresh_timer = QTimer(self)
        self._theme_refresh_timer.setSingleShot(True)
        self._theme_refresh_timer.setInterval(56)
        self._theme_refresh_timer.timeout.connect(self._refresh_theme_visuals)
        self._macos_menu_actions: Dict[str, QAction] = {}

        self._init_ui()
        self._setup_macos_menu_bar()
        self._setup_shortcuts()
        self._apply_styles()
        self._animate_cards()
        self._apply_wizard_step(1, animate=False)

    def _active_switch_specs(self) -> List[Tuple[str, str, str]]:
        if not self.is_macos:
            return list(self.SWITCH_SPECS)
        return [
            spec
            for spec in self.SWITCH_SPECS
            if spec[1] not in self.MACOS_HIDDEN_SWITCH_KEYS
        ]

    def _enable_macos_native_titlebar_hints(self) -> None:
        if not self.is_macos:
            return
        for flag_name in ("ExpandedClientAreaHint", "NoTitleBarBackgroundHint"):
            flag = getattr(Qt, flag_name, None)
            if flag is None:
                continue
            try:
                self.setWindowFlag(flag, True)
            except Exception:
                pass
        try:
            self.setUnifiedTitleAndToolBarOnMac(True)
        except Exception:
            pass

    def _macos_titlebar_top_inset(self) -> int:
        if not self.is_macos:
            return 0
        fallback = self._scaled_px(30, min_px=24, max_px=44)
        handle = self.windowHandle()
        if handle is None:
            return fallback
        try:
            margins = handle.safeAreaMargins()
        except Exception:
            margins = None
        if margins is None:
            return fallback
        top = max(0, int(margins.top()))
        return top if top > 0 else fallback

    def _update_root_layout_margins(self) -> None:
        if self._root_layout is None:
            return
        left, top, right, bottom = self._root_base_margins
        if self.is_macos:
            top += self._macos_titlebar_top_inset()
        self._root_layout.setContentsMargins(left, top, right, bottom)

    def _apply_macos_native_titlebar(self) -> None:
        if not self.is_macos:
            return
        runtime = _macos_objc_runtime()
        if runtime is None:
            return
        try:
            (
                _get_class,
                register_sel,
                send_id,
                _send_id_id,
                _send_double,
                send_void_bool,
                send_void_integer,
            ) = runtime
            ns_view = int(self.winId())
            if ns_view <= 0:
                return
            ns_window = send_id(ns_view, register_sel(b"window"))
            if not ns_window:
                return
            send_void_integer(ns_window, register_sel(b"setTitleVisibility:"), 1)
            send_void_bool(ns_window, register_sel(b"setTitlebarAppearsTransparent:"), True)
            # Prevent splitter drags from being interpreted as whole-window drags.
            send_void_bool(ns_window, register_sel(b"setMovableByWindowBackground:"), False)
            self._macos_native_titlebar_ready = True
        except Exception:
            return

    def _ensure_macos_titlebar_tracking(self) -> None:
        if not self.is_macos:
            return
        handle = self.windowHandle()
        if handle is None:
            self._update_root_layout_margins()
            return
        if handle is not self._macos_safe_area_handle:
            signal = getattr(handle, "safeAreaMarginsChanged", None)
            if signal is not None:
                try:
                    signal.connect(self._update_root_layout_margins)
                except Exception:
                    pass
            self._macos_safe_area_handle = handle
            self._macos_native_titlebar_ready = False
        if not self._macos_native_titlebar_ready:
            self._apply_macos_native_titlebar()
        self._update_root_layout_margins()

    def _ui_scale_factor(self) -> float:
        try:
            dpi = max(float(self.logicalDpiX()), float(self.logicalDpiY()))
        except Exception:
            dpi = 96.0
        if dpi <= 0:
            dpi = 96.0
        return max(1.0, min(2.25, dpi / 96.0))

    def _scaled_px(self, value: int, min_px: Optional[int] = None, max_px: Optional[int] = None) -> int:
        scaled = int(round(float(value) * self._ui_scale_factor()))
        if min_px is not None:
            scaled = max(int(min_px), scaled)
        if max_px is not None:
            scaled = min(int(max_px), scaled)
        return max(0, scaled)

    @staticmethod
    def _resolve_picture_asset(name: str) -> Path:
        return (APP_ROOT / "pictures" / name).resolve()

    def _apply_picture_background(self) -> None:
        if self.background_root is None:
            return
        bg_name = "night.jpg" if self._is_dark_theme else "day.jpg"
        self.background_root.set_background_theme(
            self._resolve_picture_asset(bg_name),
            dark_mode=self._is_dark_theme,
            accent=self._accent_color,
        )

    def _refresh_progress_pet_size(self) -> None:
        if not hasattr(self, "progress_pet") or not hasattr(self, "progress_bar"):
            return
        bar_h = max(18, self.progress_bar.height())
        bar_w = max(120, self.progress_bar.width())
        source = self.progress_pet.current_source_size()
        if not source.isValid() or source.width() <= 0 or source.height() <= 0:
            pet_w = pet_h = max(42, min(86, int(round(bar_h * 2.85))))
        else:
            target_h = max(42, min(86, int(round(bar_h * 2.85))))
            pet_w = max(36, int(round(target_h * (source.width() / max(1, source.height())))))
            pet_h = target_h
        max_w = max(44, int(round(bar_w * 0.24)))
        if pet_w > max_w:
            scale = max_w / float(max(1, pet_w))
            pet_w = max_w
            pet_h = max(32, int(round(pet_h * scale)))
        self.progress_pet.resize(pet_w, pet_h)

    def _progress_pet_ratio(self) -> float:
        if self._download_inflight and self._download_last_percent is None:
            return 1.0 - abs((self._progress_pet_phase * 2.0) - 1.0)

        minimum = int(self.progress_bar.minimum())
        maximum = int(self.progress_bar.maximum())
        span = maximum - minimum
        if span <= 0:
            return max(0.0, min(1.0, float(self._overall_percent) / 100.0))
        return max(
            0.0,
            min(1.0, float(self.progress_bar.value() - minimum) / float(span)),
        )

    def _progress_pet_mode(self) -> str:
        if self._download_inflight and self._download_last_percent is None:
            return "roam"
        return "track"

    def _progress_pet_track_target(self) -> tuple[float, float] | None:
        if not hasattr(self, "progress_bar") or not self.progress_bar.isVisible():
            return None
        bar_rect = self.progress_bar.geometry()
        if bar_rect.width() <= 0 or bar_rect.height() <= 0:
            return None

        pet_w = max(1, self.progress_pet.width())
        pet_h = max(1, self.progress_pet.height())
        ratio = self._progress_pet_ratio()
        travel = max(1, bar_rect.width() - pet_w)
        x = bar_rect.left() + int(round(travel * ratio))
        x = max(bar_rect.left(), min(bar_rect.right() - pet_w + 1, x))
        y_anchor = max(10, int(round(bar_rect.height() * 0.92)))
        y = max(4, bar_rect.top() - pet_h + y_anchor + self._scaled_px(2, min_px=2, max_px=6))
        return float(x), float(y)

    def _progress_pet_roam_bounds(self) -> tuple[float, float, float, float] | None:
        parent = self.progress_pet.parentWidget()
        if parent is None:
            return None
        rect = parent.contentsRect()
        pet_w = max(1, self.progress_pet.width())
        pet_h = max(1, self.progress_pet.height())
        min_x = float(rect.left())
        max_x = float(max(rect.left(), rect.right() - pet_w + 1))
        min_y = float(rect.top())
        max_y = float(max(rect.top(), rect.bottom() - pet_h + 1))
        if max_x < min_x or max_y < min_y:
            return None
        return min_x, max_x, min_y, max_y

    def _choose_progress_pet_roam_target(self) -> None:
        bounds = self._progress_pet_roam_bounds()
        if bounds is None:
            return
        min_x, max_x, min_y, max_y = bounds
        x = random.uniform(min_x, max_x) if max_x > min_x else min_x
        y = random.uniform(min_y, max_y) if max_y > min_y else min_y
        self._progress_pet_roam_target = [x, y]
        self._progress_pet_roam_pause_ticks = random.randint(6, 24)

    def _move_progress_pet_towards(self, target_x: float, target_y: float, *, speed: float) -> bool:
        current_x, current_y = self._progress_pet_pos
        dx = float(target_x) - float(current_x)
        dy = float(target_y) - float(current_y)
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= 1.0:
            self._progress_pet_pos = [float(target_x), float(target_y)]
            return True

        step = max(1.0, distance * float(speed))
        if step >= distance:
            self._progress_pet_pos = [float(target_x), float(target_y)]
            return True

        scale = step / distance
        self._progress_pet_pos[0] = float(current_x) + (dx * scale)
        self._progress_pet_pos[1] = float(current_y) + (dy * scale)
        return False

    def _apply_progress_pet_position(self) -> None:
        self.progress_pet.move(
            int(round(self._progress_pet_pos[0])),
            int(round(self._progress_pet_pos[1])),
        )
        self.progress_pet.raise_()

    def _sync_progress_pet(self) -> None:
        if not hasattr(self, "progress_pet") or not hasattr(self, "progress_bar"):
            return
        should_show = bool(
            self.running
            or self._download_inflight
            or self._overall_percent > 0.0
            or self.progress_bar.value() > 0
        )
        if not self.progress_pet.has_media():
            should_show = False
        was_visible = self.progress_pet.isVisible()
        self.progress_pet.setVisible(should_show)
        if should_show:
            self._refresh_progress_pet_size()
            self._update_progress_pet_position(force_snap=not was_visible)
            self.progress_pet.raise_()

        if should_show:
            if not self._progress_pet_timer.isActive():
                self._progress_pet_timer.start()
        elif self._progress_pet_timer.isActive():
            self._progress_pet_timer.stop()

    def _update_progress_pet_position(self, *, force_snap: bool = False) -> None:
        if not hasattr(self, "progress_pet") or not self.progress_pet.isVisible():
            return

        self._refresh_progress_pet_size()
        mode = self._progress_pet_mode()
        target = self._progress_pet_track_target()
        if target is None and mode != "roam":
            return

        if force_snap or self._progress_pet_last_mode == "":
            if mode == "roam":
                bounds = self._progress_pet_roam_bounds()
                if bounds is None:
                    return
                min_x, _max_x, min_y, _max_y = bounds
                self._progress_pet_pos = [min_x, min_y]
                self._choose_progress_pet_roam_target()
            elif target is not None:
                self._progress_pet_pos = [target[0], target[1]]
            self._progress_pet_last_mode = mode

        if mode != self._progress_pet_last_mode:
            if mode == "roam":
                self._choose_progress_pet_roam_target()
            self._progress_pet_last_mode = mode

        if mode == "track" and target is not None:
            self._move_progress_pet_towards(target[0], target[1], speed=1.0 if force_snap else self._progress_pet_return_speed)
            self._apply_progress_pet_position()
            return

        if mode == "roam":
            bounds = self._progress_pet_roam_bounds()
            if bounds is None:
                return
            min_x, max_x, min_y, max_y = bounds
            self._progress_pet_pos[0] = min(max(self._progress_pet_pos[0], min_x), max_x)
            self._progress_pet_pos[1] = min(max(self._progress_pet_pos[1], min_y), max_y)
            if force_snap or (
                self._progress_pet_roam_target[0] == 0.0
                and self._progress_pet_roam_target[1] == 0.0
            ):
                self._choose_progress_pet_roam_target()
            self._apply_progress_pet_position()

    def _tick_progress_pet_motion(self) -> None:
        if not hasattr(self, "progress_pet") or not self.progress_pet.isVisible():
            return
        self._progress_pet_phase = (self._progress_pet_phase + 0.024) % 1.0
        mode = self._progress_pet_mode()
        if mode == "roam":
            if self._progress_pet_roam_pause_ticks > 0:
                self._progress_pet_roam_pause_ticks -= 1
            else:
                arrived = self._move_progress_pet_towards(
                    self._progress_pet_roam_target[0],
                    self._progress_pet_roam_target[1],
                    speed=self._progress_pet_roam_speed,
                )
                if arrived:
                    self._choose_progress_pet_roam_target()
        else:
            target = self._progress_pet_track_target()
            if target is not None:
                self._move_progress_pet_towards(
                    target[0],
                    target[1],
                    speed=self._progress_pet_return_speed,
                )
        self._apply_progress_pet_position()

    def _init_ui(self):
        root = PictureBackgroundFrame(self)
        root.setObjectName("Root")
        root.setAttribute(Qt.WA_StyledBackground, True)
        self.setCentralWidget(root)
        self.background_root = root

        main_layout = QVBoxLayout(root)
        main_layout.setSpacing(14)
        self._root_layout = main_layout
        self._update_root_layout_margins()

        root_splitter = QSplitter(Qt.Vertical)
        root_splitter.setChildrenCollapsible(False)
        root_splitter.setHandleWidth(8)
        root_splitter.setObjectName("VerticalSplitter")
        self.root_splitter = root_splitter

        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(20, 12, 20, 12)

        title_col = QVBoxLayout()
        title = QLabel("AI 音视频转录处理中心")
        title.setObjectName("Title")
        subtitle = QLabel("音视频与文本处理流水线  //  V11.45.14  //  MediaTranscribeStudio")
        subtitle.setObjectName("Subtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        self.status_chip = QLabel("系统已就绪")
        self.status_chip.setObjectName("StatusChip")
        self.status_chip.setAlignment(Qt.AlignCenter)
        self.status_chip.setMinimumWidth(self._scaled_px(160, min_px=148, max_px=320))

        header_layout.addLayout(title_col)
        header_layout.addStretch(1)
        header_layout.addWidget(self.status_chip)
        header_card.setMinimumHeight(80)
        root_splitter.addWidget(header_card)

        progress_card = QFrame()
        progress_card.setObjectName("ProgressCard")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(18, 16, 18, 18)
        progress_layout.setSpacing(10)

        self.progress_label = QLabel("准备就绪")
        self.progress_label.setObjectName("ProgressLabel")
        self.progress_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("TaskProgress")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")

        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        self.progress_pet = AnimatedSticker(progress_card)
        self.progress_pet.setObjectName("ProgressPet")
        self.progress_pet.hide()

        gif_path = self._resolve_picture_asset("1.gif")
        if gif_path.exists():
            movie = QMovie(str(gif_path))
            if movie.isValid():
                movie.setCacheMode(QMovie.CacheAll)
                movie.start()
                self.progress_pet.set_movie(movie)
                self._progress_pet_movie = movie

        progress_card.setMinimumHeight(114)
        root_splitter.addWidget(progress_card)

        stepper_card = QFrame()
        stepper_card.setObjectName("StepCard")
        stepper_card.setMinimumHeight(52)
        stepper_layout = QHBoxLayout(stepper_card)
        stepper_layout.setContentsMargins(12, 8, 12, 8)
        stepper_layout.setSpacing(8)
        stepper_layout.addWidget(self._section_label("工作流", "cyantext"))

        for step, title in ((1, "导入文件"), (2, "配置参数"), (3, "开始处理")):
            btn = QPushButton(f"{step} · {title}")
            btn.setObjectName("StepBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked=False, s=step: self._go_to_step(s))
            self._step_buttons.append(btn)
            stepper_layout.addWidget(btn)

        stepper_layout.addStretch(1)

        self.prev_step_button = QPushButton("上一步")
        self.prev_step_button.setObjectName("GhostBtnMin")
        self.prev_step_button.clicked.connect(self._go_prev_step)
        stepper_layout.addWidget(self.prev_step_button, 0, Qt.AlignRight)

        self.next_step_button = QPushButton("下一步")
        self.next_step_button.setObjectName("PrimaryBtn")
        self.next_step_button.clicked.connect(self._go_next_step)
        stepper_layout.addWidget(self.next_step_button, 0, Qt.AlignRight)

        root_splitter.addWidget(stepper_card)
        self.stepper_card = stepper_card

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        self.main_splitter = splitter

        # 核心改动：将左右区域替换为QSplitter以实现可拖动调节高度
        left_panel = QSplitter(Qt.Vertical)
        left_panel.setChildrenCollapsible(False)
        left_panel.setHandleWidth(6)
        left_panel.setObjectName("VerticalSplitter")
        left_panel.setMinimumWidth(self._scaled_px(560, min_px=500, max_px=860))
        self.left_panel_splitter = left_panel

        right_panel = QSplitter(Qt.Vertical)
        right_panel.setChildrenCollapsible(False)
        right_panel.setHandleWidth(6)
        right_panel.setObjectName("VerticalSplitter")
        right_panel.setMinimumWidth(self._scaled_px(420, min_px=340, max_px=760))
        self.right_panel_splitter = right_panel

        # 拖拽区域
        self.drop_area = DropArea()
        self.drop_area.setObjectName("Card")
        self.drop_area.files_dropped.connect(self.add_files)
        self.drop_area.browse_requested.connect(self.browse_files)
        left_panel.addWidget(self.drop_area)

        # 队列区域
        queue_card = QFrame()
        queue_card.setObjectName("Card")
        queue_layout = QVBoxLayout(queue_card)
        queue_layout.setContentsMargins(12, 12, 12, 12)
        queue_layout.setSpacing(8)
        queue_layout.addWidget(self._section_label("待处理任务队列", "cyantext"))

        self.file_list = QListWidget()
        self.file_list.setObjectName("ListSurface")
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        queue_layout.addWidget(self.file_list, 1)
        queue_card.setMinimumHeight(140)
        left_panel.addWidget(queue_card)
        left_panel.setStretchFactor(1, 1)  # 修改：降低队列区域的默认伸展比例，为配置区留出空间
        # 控制按钮区域
        controls_card = QFrame()
        controls_card.setObjectName("Card")
        controls_card.setMinimumHeight(90)
        controls_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.controls_card = controls_card
        controls_shell_layout = QVBoxLayout(controls_card)
        controls_shell_layout.setContentsMargins(12, 8, 12, 8)
        controls_shell_layout.setSpacing(0)

        self.controls_scroll = QScrollArea(controls_card)
        self.controls_scroll.setObjectName("ControlScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.controls_scroll.setMinimumHeight(64)
        self.controls_scroll.setMaximumHeight(82)

        controls_row = QWidget(self.controls_scroll)
        controls_layout = QHBoxLayout(controls_row)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        self.controls_scroll.setWidget(controls_row)
        controls_shell_layout.addWidget(self.controls_scroll)
        self.controls_row_widget = controls_row
        self.controls_row_layout = controls_layout

        self.add_button = QPushButton("添加文件")
        self.add_button.setObjectName("GhostBtn")
        self.add_button.clicked.connect(self.browse_files)

        self.start_button = QPushButton("开始处理")
        self.start_button.setObjectName("PrimaryBtn")
        self.start_button.clicked.connect(self.start_processing)

        self.pause_button = QPushButton("暂停任务")
        self.pause_button.setObjectName("WarningBtn")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.clear_button = QPushButton("清空队列")
        self.clear_button.setObjectName("GhostBtn")
        self.clear_button.clicked.connect(self.clear_queue)

        self.remove_checked_button = QPushButton("删除选中")
        self.remove_checked_button.setObjectName("GhostBtn")
        self.remove_checked_button.clicked.connect(self.remove_selected_queue_items)

        self.clear_temp_button = QPushButton("清理临时文件")
        self.clear_temp_button.setObjectName("GhostBtn")
        self.clear_temp_button.clicked.connect(self.delete_temp_files)

        self.full_cleanup_button = HoldToDeleteButton("长按全清理", hold_ms=1200)
        self.full_cleanup_button.setObjectName("DangerHoldBtn")
        self.full_cleanup_button.setToolTip(
            "按住约 1.2 秒执行全清理：删除选中、清空队列、临时文件、日志、全部产物"
        )
        self.full_cleanup_button.confirmed.connect(self.run_full_cleanup)

        for btn in (
            self.add_button,
            self.start_button,
            self.pause_button,
            self.remove_checked_button,
            self.clear_button,
            self.clear_temp_button,
            self.full_cleanup_button,
        ):
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        controls_layout.addWidget(self.add_button)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.remove_checked_button)
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.clear_temp_button)
        controls_layout.addWidget(self.full_cleanup_button)
        controls_layout.addStretch(1)
        self._sync_controls_row_width()
        left_panel.addWidget(controls_card)

        # --- 深度优化的系统配置面板 ---
        config_card = QFrame()
        config_card.setObjectName("Card")
        config_layout = QVBoxLayout(config_card)
        # 进一步缩减配置区域的边距
        config_layout.setContentsMargins(12, 10, 12, 8)
        config_layout.setSpacing(6)
        
        cfg_header_layout = QHBoxLayout()
        cfg_header_layout.addWidget(self._section_label("系统运行配置", "cyantext"))
        self.toggle_all_switches_button = QPushButton("一键全开")
        self.toggle_all_switches_button.setObjectName("GhostBtnMin")
        self.toggle_all_switches_button.setCheckable(True)
        self.toggle_all_switches_button.clicked.connect(self.toggle_all_switches)
        cfg_header_layout.addWidget(self.toggle_all_switches_button, 0, Qt.AlignRight)
        self.usage_help_button = QPushButton("打开使用说明")
        self.usage_help_button.setObjectName("GhostBtnMin")
        self.usage_help_button.setCheckable(True)
        self.usage_help_button.setChecked(False)
        self.usage_help_button.toggled.connect(self._on_usage_help_toggled)
        cfg_header_layout.addWidget(self.usage_help_button, 0, Qt.AlignRight)
        self.save_config_button = QPushButton("保存配置到本地")
        self.save_config_button.setObjectName("GhostBtnMin")
        self.save_config_button.clicked.connect(self.save_config_overrides)
        cfg_header_layout.addWidget(self.save_config_button, 0, Qt.AlignRight)
        config_layout.addLayout(cfg_header_layout)

        config_help = QLabel(self.CONFIG_HELP_TEXT)
        config_help.setObjectName("ConfigGuide")
        config_help.setText("⚙️ 配置自动保存到 config.yaml；支持环境变量覆盖、翻译和分角色设置。")
        config_help.setToolTip(self.CONFIG_HELP_TEXT)
        config_help.setWordWrap(True)
        config_help.setTextInteractionFlags(Qt.TextSelectableByMouse)
        config_layout.addWidget(config_help)
        self.config_help_label = config_help

        # 核心改动：使用紧凑网格布局，排布语言、DashScope 和 HF Token
        inputs_layout = QGridLayout()
        inputs_layout.setSpacing(8)
        inputs_layout.setContentsMargins(0, 4, 0, 4)
        inputs_layout.setColumnMinimumWidth(0, 76)
        inputs_layout.setColumnMinimumWidth(2, 76)
        inputs_layout.setColumnStretch(1, 1)
        inputs_layout.setColumnStretch(3, 1)
        self.inputs_layout = inputs_layout

        # 识别语言设置
        self.language_combo = QComboBox()
        self.language_combo.setObjectName("ConfigInput")
        for label, code in self.LANGUAGE_CHOICES:
            self.language_combo.addItem(label, code)
        language_code = self._resolve_language_choice()
        lang_index = self.language_combo.findData(language_code)
        self.language_combo.setCurrentIndex(lang_index if lang_index >= 0 else 0)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        
        lbl_lang = self._make_grid_help_label(
            "识别语言:", "ui.recognition_language", "识别语言"
        )

        self.translation_target_combo = QComboBox()
        self.translation_target_combo.setObjectName("ConfigInput")
        for label, code in self.TRANSLATION_LANGUAGE_CHOICES:
            self.translation_target_combo.addItem(label, code)
        translation_code = self._resolve_translation_target_choice()
        trans_index = self.translation_target_combo.findData(translation_code)
        self.translation_target_combo.setCurrentIndex(trans_index if trans_index >= 0 else 0)
        self.translation_target_combo.currentIndexChanged.connect(
            self._on_translation_target_changed
        )

        lbl_translate = self._make_grid_help_label(
            "翻译目标:", "translation.target_language", "翻译目标"
        )
        
        # DashScope设置
        self.dashscope_key_input = QLineEdit()
        self.dashscope_key_input.setObjectName("ConfigInput")
        self.dashscope_key_input.setPlaceholderText("sk-xxxx (用于LLM分析/翻译)")
        self.dashscope_key_input.setText(str(_get_nested(self.config._data, "llm.api_key", "")))
        self.dashscope_key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.dashscope_key_input.textEdited.connect(self._on_dashscope_key_edited)
        
        lbl_dash = self._make_grid_help_label("千问 Key:", "llm.api_key", "千问 Key")
        eye_size = self._scaled_px(24, min_px=22, max_px=34)
        
        self.dash_eye_btn = QPushButton("👁")
        self.dash_eye_btn.setObjectName("EyeBtn")
        self.dash_eye_btn.setFixedSize(eye_size, eye_size)
        self.dash_eye_btn.setCheckable(True)
        self.dash_eye_btn.toggled.connect(lambda checked: self.dashscope_key_input.setEchoMode(QLineEdit.Normal if checked else QLineEdit.PasswordEchoOnEdit))

        # HF Token设置
        self.hf_token_input = QLineEdit()
        self.hf_token_input.setObjectName("ConfigInput")
        self.hf_token_input.setPlaceholderText("hf_xxxx (用于加速下载)")
        self.hf_token_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        hf_token = str(_get_nested(self.config._data, "asr.faster_whisper.hf_token", "") or "").strip()
        if not hf_token:
            hf_token = DEFAULT_HF_TOKEN
        self.hf_token_input.setText(hf_token)
        self.hf_token_input.textEdited.connect(self._on_hf_token_edited)
        
        lbl_hf = self._make_grid_help_label(
            "HF Token:", "asr.faster_whisper.hf_token", "HF Token"
        )

        self.hf_eye_btn = QPushButton("👁")
        self.hf_eye_btn.setObjectName("EyeBtn")
        self.hf_eye_btn.setFixedSize(eye_size, eye_size)
        self.hf_eye_btn.setCheckable(True)
        self.hf_eye_btn.toggled.connect(lambda checked: self.hf_token_input.setEchoMode(QLineEdit.Normal if checked else QLineEdit.PasswordEchoOnEdit))

        # 将这些输入框在三排紧凑放置
        # 第一排：识别语言 + 翻译目标
        inputs_layout.addWidget(lbl_lang, 0, 0)
        inputs_layout.addWidget(self.language_combo, 0, 1)
        inputs_layout.addWidget(lbl_translate, 0, 2)
        inputs_layout.addWidget(self.translation_target_combo, 0, 3)

        # 第二排：DashScope
        inputs_layout.addWidget(lbl_dash, 1, 0)
        inputs_layout.addWidget(self.dashscope_key_input, 1, 1, 1, 3)
        inputs_layout.addWidget(self.dash_eye_btn, 1, 4)

        # 第三排：HF Token（拉长）
        inputs_layout.addWidget(lbl_hf, 2, 0)
        inputs_layout.addWidget(self.hf_token_input, 2, 1, 1, 3)
        inputs_layout.addWidget(self.hf_eye_btn, 2, 4)

        config_layout.addLayout(inputs_layout)
        diarization_layout = QGridLayout()
        diarization_layout.setSpacing(8)
        diarization_layout.setContentsMargins(0, 0, 0, 2)

        nemo_num_speakers = int(_get_nested(self.config._data, "asr.nemo_msdd.num_speakers", 0) or 0)
        nemo_mode = "manual" if nemo_num_speakers > 0 else "auto"

        self.nemo_speaker_mode_combo = QComboBox()
        self.nemo_speaker_mode_combo.setObjectName("ConfigInput")
        self.nemo_speaker_mode_combo.addItem("自动检测说话人数", "auto")
        self.nemo_speaker_mode_combo.addItem("固定说话人数", "manual")
        idx_nemo_mode = self.nemo_speaker_mode_combo.findData(nemo_mode)
        self.nemo_speaker_mode_combo.setCurrentIndex(idx_nemo_mode if idx_nemo_mode >= 0 else 0)
        self.nemo_speaker_mode_combo.currentIndexChanged.connect(self._on_nemo_speaker_mode_changed)

        self.nemo_fixed_speakers_input = QLineEdit()
        self.nemo_fixed_speakers_input.setObjectName("ConfigInput")
        self.nemo_fixed_speakers_input.setText(str(nemo_num_speakers if nemo_num_speakers > 0 else 2))
        self.nemo_fixed_speakers_input.setToolTip("固定说话人数，仅在“固定说话人数”模式下生效。")
        self.nemo_fixed_speakers_input.textEdited.connect(self._on_advanced_setting_edited)

        diarization_layout.addWidget(
            self._make_grid_help_label(
                "NeMo 分角色人数:",
                "asr.nemo_msdd.speaker_mode",
                "NeMo 分角色人数模式",
            ),
            0,
            0,
        )
        diarization_layout.addWidget(self.nemo_speaker_mode_combo, 0, 1)
        diarization_layout.addWidget(
            self._make_grid_help_label(
                "固定人数:",
                "asr.nemo_msdd.num_speakers",
                "固定人数",
            ),
            0,
            2,
        )
        diarization_layout.addWidget(self.nemo_fixed_speakers_input, 0, 3)
        config_layout.addLayout(diarization_layout)

        tool_button_row = QHBoxLayout()
        tool_button_row.setContentsMargins(0, 0, 0, 0)
        tool_button_row.setSpacing(8)

        self.advanced_toggle_button = QPushButton("打开高级配置弹窗")
        self.advanced_toggle_button.setObjectName("GhostBtnMin")
        self.advanced_toggle_button.setCheckable(True)
        self.advanced_toggle_button.setChecked(False)
        self.advanced_toggle_button.toggled.connect(self._on_advanced_panel_toggled)
        tool_button_row.addWidget(self.advanced_toggle_button, 0, Qt.AlignLeft)

        self.posterior_assistant_toggle_button = QPushButton("打开 Posterior Fusion 助手")
        self.posterior_assistant_toggle_button.setObjectName("GhostBtnMin")
        self.posterior_assistant_toggle_button.setCheckable(True)
        self.posterior_assistant_toggle_button.setChecked(False)
        self.posterior_assistant_toggle_button.toggled.connect(self._on_posterior_assistant_toggled)
        tool_button_row.addWidget(self.posterior_assistant_toggle_button, 0, Qt.AlignLeft)
        tool_button_row.addStretch(1)
        config_layout.addLayout(tool_button_row)

        self.advanced_panel = QFrame()
        self.advanced_panel.setObjectName("AdvancedPanel")
        advanced_panel_layout = QVBoxLayout(self.advanced_panel)
        advanced_panel_layout.setContentsMargins(0, 0, 0, 0)
        advanced_panel_layout.setSpacing(6)

        advanced_hint = QLabel(
            "高级可配置项（默认收起）：控制分块策略、NeMo 搜索范围、并发与内存保护，以及字幕回写参数。"
        )
        advanced_hint.setObjectName("ConfigGuide")
        advanced_hint.setWordWrap(True)
        advanced_panel_layout.addWidget(advanced_hint)

        advanced_layout = QGridLayout()
        advanced_layout.setSpacing(8)
        advanced_layout.setContentsMargins(0, 2, 0, 2)

        self.overlap_input = QLineEdit()
        self.overlap_input.setObjectName("ConfigInput")
        self.overlap_input.setText(str(_get_nested(self.config._data, "audio.overlap_sec", 1.0)))
        self.overlap_input.setToolTip("分块重叠秒数，用于降低跨块句子截断。")
        self.overlap_input.textEdited.connect(self._on_advanced_setting_edited)

        self.nemo_min_speakers_input = QLineEdit()
        self.nemo_min_speakers_input.setObjectName("ConfigInput")
        self.nemo_min_speakers_input.setText(str(_get_nested(self.config._data, "asr.nemo_msdd.min_speakers", 1)))
        self.nemo_min_speakers_input.setToolTip("自动检测模式下的最少说话人数。")
        self.nemo_min_speakers_input.textEdited.connect(self._on_advanced_setting_edited)

        self.nemo_max_speakers_input = QLineEdit()
        self.nemo_max_speakers_input.setObjectName("ConfigInput")
        self.nemo_max_speakers_input.setText(str(_get_nested(self.config._data, "asr.nemo_msdd.max_speakers", 8)))
        self.nemo_max_speakers_input.setToolTip("自动检测模式下的最多说话人数。")
        self.nemo_max_speakers_input.textEdited.connect(self._on_advanced_setting_edited)

        self.max_concurrent_input = QLineEdit()
        self.max_concurrent_input.setObjectName("ConfigInput")
        self.max_concurrent_input.setText(str(_get_nested(self.config._data, "performance.max_concurrent_files", 1)))
        self.max_concurrent_input.setToolTip("并发处理文件数，增大会提速但更占内存。")
        self.max_concurrent_input.textEdited.connect(self._on_advanced_setting_edited)

        self.gpu_mem_fraction_input = QLineEdit()
        self.gpu_mem_fraction_input.setObjectName("ConfigInput")
        self.gpu_mem_fraction_input.setText(str(_get_nested(self.config._data, "performance.gpu_memory_fraction", 0.98)))
        self.gpu_mem_fraction_input.setToolTip("资源保护比例，macOS 下通常保持默认即可。")
        self.gpu_mem_fraction_input.textEdited.connect(self._on_advanced_setting_edited)

        self.overlay_renderer_combo = QComboBox()
        self.overlay_renderer_combo.setObjectName("ConfigInput")
        self.overlay_renderer_combo.addItem("FFmpeg", "ffmpeg")
        overlay_renderer = str(_get_nested(self.config._data, "video_text_overlay.renderer", "ffmpeg") or "ffmpeg")
        idx_overlay_renderer = self.overlay_renderer_combo.findData(overlay_renderer)
        self.overlay_renderer_combo.setCurrentIndex(idx_overlay_renderer if idx_overlay_renderer >= 0 else 0)
        self.overlay_renderer_combo.currentIndexChanged.connect(self._on_advanced_setting_edited)

        self.overlay_effect_combo = QComboBox()
        self.overlay_effect_combo.setObjectName("ConfigInput")
        self.overlay_effect_combo.addItem("自动（逐词高亮优先）", "auto")
        self.overlay_effect_combo.addItem("逐词高亮", "karaoke")
        self.overlay_effect_combo.addItem("关键词变色", "keyword")
        self.overlay_effect_combo.addItem("纯样式无特效", "plain")
        overlay_effect = str(_get_nested(self.config._data, "video_text_overlay.style_effect", "auto") or "auto")
        idx_overlay_effect = self.overlay_effect_combo.findData(overlay_effect)
        self.overlay_effect_combo.setCurrentIndex(idx_overlay_effect if idx_overlay_effect >= 0 else 0)
        self.overlay_effect_combo.currentIndexChanged.connect(self._on_advanced_setting_edited)

        self.overlay_suffix_input = QLineEdit()
        self.overlay_suffix_input.setObjectName("ConfigInput")
        self.overlay_suffix_input.setText(str(_get_nested(self.config._data, "video_text_overlay.output_suffix", ".captioned") or ".captioned"))
        self.overlay_suffix_input.setToolTip("回写视频文件后缀，例如 .captioned。")
        self.overlay_suffix_input.textEdited.connect(self._on_advanced_setting_edited)

        self.overlay_font_name_input = QLineEdit()
        self.overlay_font_name_input.setObjectName("ConfigInput")
        default_overlay_font = "PingFang SC" if self.is_macos else "Microsoft YaHei"
        self.overlay_font_name_input.setText(str(_get_nested(self.config._data, "video_text_overlay.font_name", default_overlay_font) or default_overlay_font))
        self.overlay_font_name_input.setToolTip(f"字幕字体名称，例如 {default_overlay_font}。")
        self.overlay_font_name_input.textEdited.connect(self._on_advanced_setting_edited)

        self.overlay_font_size_input = QLineEdit()
        self.overlay_font_size_input.setObjectName("ConfigInput")
        self.overlay_font_size_input.setText(str(_get_nested(self.config._data, "video_text_overlay.font_size", 22)))
        self.overlay_font_size_input.setToolTip("字幕字号（像素）。")
        self.overlay_font_size_input.textEdited.connect(self._on_advanced_setting_edited)

        self.overlay_max_chars_input = QLineEdit()
        self.overlay_max_chars_input.setObjectName("ConfigInput")
        self.overlay_max_chars_input.setText(str(_get_nested(self.config._data, "video_text_overlay.max_line_chars", 18)))
        self.overlay_max_chars_input.setToolTip("每行最大字符数，超过会自动换行。")
        self.overlay_max_chars_input.textEdited.connect(self._on_advanced_setting_edited)

        self.overlay_margin_input = QLineEdit()
        self.overlay_margin_input.setObjectName("ConfigInput")
        self.overlay_margin_input.setText(str(_get_nested(self.config._data, "video_text_overlay.bottom_margin_px", 56)))
        self.overlay_margin_input.setToolTip("字幕到底边距离（像素）。")
        self.overlay_margin_input.textEdited.connect(self._on_advanced_setting_edited)

        self.overlay_codec_combo = QComboBox()
        self.overlay_codec_combo.setObjectName("ConfigInput")
        if self.is_macos:
            self.overlay_codec_combo.addItem("h264_videotoolbox（Apple）", "h264_videotoolbox")
            self.overlay_codec_combo.addItem("hevc_videotoolbox（Apple H265）", "hevc_videotoolbox")
            self.overlay_codec_combo.addItem("libx264（通用）", "libx264")
            overlay_codec_default = "h264_videotoolbox"
        else:
            self.overlay_codec_combo.addItem("h264_nvenc（NVIDIA）", "h264_nvenc")
            self.overlay_codec_combo.addItem("libx264（通用）", "libx264")
            self.overlay_codec_combo.addItem("hevc_nvenc（H265）", "hevc_nvenc")
            overlay_codec_default = "h264_nvenc"
        overlay_codec = str(_get_nested(self.config._data, "video_text_overlay.ffmpeg_video_codec", overlay_codec_default) or overlay_codec_default)
        idx_overlay_codec = self.overlay_codec_combo.findData(overlay_codec)
        self.overlay_codec_combo.setCurrentIndex(idx_overlay_codec if idx_overlay_codec >= 0 else 0)
        self.overlay_codec_combo.currentIndexChanged.connect(self._on_advanced_setting_edited)

        self.overlay_quality_input = QLineEdit()
        self.overlay_quality_input.setObjectName("ConfigInput")
        self.overlay_quality_input.setText(str(_get_nested(self.config._data, "video_text_overlay.ffmpeg_crf", 20)))
        self.overlay_quality_input.setToolTip("质量参数：数值越小画质越高（文件更大）。")
        self.overlay_quality_input.textEdited.connect(self._on_advanced_setting_edited)

        self.overlay_ffmpeg_input = QLineEdit()
        self.overlay_ffmpeg_input.setObjectName("ConfigInput")
        self.overlay_ffmpeg_input.setText(str(_get_nested(self.config._data, "video_text_overlay.ffmpeg_path", "") or ""))
        self.overlay_ffmpeg_input.setPlaceholderText("可选：自定义 ffmpeg 可执行文件路径")
        self.overlay_ffmpeg_input.textEdited.connect(self._on_advanced_setting_edited)

        advanced_layout.addWidget(
            self._make_grid_help_label("重叠秒数:", "audio.overlap_sec", "重叠秒数"),
            0,
            0,
        )
        advanced_layout.addWidget(self.overlap_input, 0, 1)
        advanced_layout.addWidget(
            self._make_grid_help_label(
                "最少说话人数:",
                "asr.nemo_msdd.min_speakers",
                "最少说话人数",
            ),
            0,
            2,
        )
        advanced_layout.addWidget(self.nemo_min_speakers_input, 0, 3)

        advanced_layout.addWidget(
            self._make_grid_help_label(
                "最多说话人数:",
                "asr.nemo_msdd.max_speakers",
                "最多说话人数",
            ),
            1,
            0,
        )
        advanced_layout.addWidget(self.nemo_max_speakers_input, 1, 1)
        advanced_layout.addWidget(
            self._make_grid_help_label(
                "并发文件数:",
                "performance.max_concurrent_files",
                "并发文件数",
            ),
            1,
            2,
        )
        advanced_layout.addWidget(self.max_concurrent_input, 1, 3)
        advanced_layout.addWidget(
            self._make_grid_help_label(
                "资源保护比例:",
                "performance.gpu_memory_fraction",
                "资源保护比例",
            ),
            1,
            4,
        )
        advanced_layout.addWidget(self.gpu_mem_fraction_input, 1, 5)

        advanced_layout.addWidget(
            self._make_grid_help_label(
                "回写引擎:",
                "video_text_overlay.renderer",
                "回写引擎",
            ),
            2,
            0,
        )
        advanced_layout.addWidget(self.overlay_renderer_combo, 2, 1, 1, 2)
        advanced_layout.addWidget(
            self._make_grid_help_label(
                "字幕效果:",
                "video_text_overlay.style_effect",
                "字幕效果",
            ),
            2,
            3,
        )
        advanced_layout.addWidget(self.overlay_effect_combo, 2, 4, 1, 2)

        advanced_layout.addWidget(
            self._make_grid_help_label(
                "输出后缀:",
                "video_text_overlay.output_suffix",
                "输出后缀",
            ),
            3,
            0,
        )
        advanced_layout.addWidget(self.overlay_suffix_input, 3, 1)
        advanced_layout.addWidget(
            self._make_grid_help_label(
                "字体名:",
                "video_text_overlay.font_name",
                "字体名",
            ),
            3,
            2,
        )
        advanced_layout.addWidget(self.overlay_font_name_input, 3, 3, 1, 3)

        advanced_layout.addWidget(
            self._make_grid_help_label(
                "字号(px):",
                "video_text_overlay.font_size",
                "字号",
            ),
            4,
            0,
        )
        advanced_layout.addWidget(self.overlay_font_size_input, 4, 1)
        advanced_layout.addWidget(
            self._make_grid_help_label(
                "每行字数:",
                "video_text_overlay.max_line_chars",
                "每行字数",
            ),
            4,
            2,
        )
        advanced_layout.addWidget(self.overlay_max_chars_input, 4, 3)
        advanced_layout.addWidget(
            self._make_grid_help_label(
                "底边距(px):",
                "video_text_overlay.bottom_margin_px",
                "底边距",
            ),
            4,
            4,
        )
        advanced_layout.addWidget(self.overlay_margin_input, 4, 5)

        advanced_layout.addWidget(
            self._make_grid_help_label(
                "视频编码器:",
                "video_text_overlay.ffmpeg_video_codec",
                "视频编码器",
            ),
            5,
            0,
        )
        advanced_layout.addWidget(self.overlay_codec_combo, 5, 1, 1, 2)
        advanced_layout.addWidget(
            self._make_grid_help_label(
                "质量参数:",
                "video_text_overlay.ffmpeg_crf",
                "质量参数",
            ),
            5,
            3,
        )
        advanced_layout.addWidget(self.overlay_quality_input, 5, 4, 1, 2)

        advanced_layout.addWidget(
            self._make_grid_help_label(
                "FFmpeg 路径:",
                "video_text_overlay.ffmpeg_path",
                "FFmpeg 路径",
            ),
            6,
            0,
        )
        advanced_layout.addWidget(self.overlay_ffmpeg_input, 6, 1, 1, 5)

        advanced_panel_layout.addLayout(advanced_layout)
        # Avoid forcing visibility here; in some PySide6/Qt builds this can
        # crash during layout activation after dynamic label wrapping.
        self.advanced_dialog = QDialog(self)
        self.advanced_dialog.setWindowTitle("高级配置")
        self.advanced_dialog.setModal(False)
        self.advanced_dialog.setWindowFlag(Qt.Tool, False)
        self.advanced_dialog.setWindowFlag(Qt.Dialog, True)
        self.advanced_dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.advanced_dialog.setAttribute(Qt.WA_StyledBackground, True)
        self.advanced_dialog.setProperty("glassWindow", True)
        if self.is_macos:
            self.advanced_dialog.setAttribute(Qt.WA_TranslucentBackground, True)
        advanced_dialog_layout = QVBoxLayout(self.advanced_dialog)
        advanced_dialog_layout.setContentsMargins(12, 12, 12, 12)
        advanced_dialog_layout.setSpacing(8)
        advanced_dialog_layout.addWidget(self.advanced_panel)
        advanced_close_row = QHBoxLayout()
        advanced_close_row.addStretch(1)
        advanced_close_btn = QPushButton("关闭")
        advanced_close_btn.setObjectName("GhostBtnMin")
        advanced_close_btn.clicked.connect(lambda: self._sync_advanced_toggle(False))
        advanced_close_btn.clicked.connect(lambda: self._animate_tool_dialog(self.advanced_dialog, False))
        advanced_close_row.addWidget(advanced_close_btn, 0, Qt.AlignRight)
        advanced_dialog_layout.addLayout(advanced_close_row)
        self.advanced_dialog.resize(980, 460)
        self.advanced_dialog.finished.connect(lambda _code: self._sync_advanced_toggle(False))

        self.usage_help_dialog = QDialog(self)
        self.usage_help_dialog.setWindowTitle("使用说明")
        self.usage_help_dialog.setModal(False)
        self.usage_help_dialog.setWindowFlag(Qt.Tool, False)
        self.usage_help_dialog.setWindowFlag(Qt.Dialog, True)
        self.usage_help_dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.usage_help_dialog.setAttribute(Qt.WA_StyledBackground, True)
        self.usage_help_dialog.setProperty("glassWindow", True)
        if self.is_macos:
            self.usage_help_dialog.setAttribute(Qt.WA_TranslucentBackground, True)
        usage_dialog_layout = QVBoxLayout(self.usage_help_dialog)
        usage_dialog_layout.setContentsMargins(12, 12, 12, 12)
        usage_dialog_layout.setSpacing(8)
        usage_intro = QLabel("新手可先阅读本说明，再按配置项右侧的问号查看单项解释。")
        usage_intro.setObjectName("ConfigGuide")
        usage_intro.setWordWrap(True)
        usage_dialog_layout.addWidget(usage_intro)
        self.usage_help_text = QPlainTextEdit()
        self.usage_help_text.setObjectName("LogSurface")
        self.usage_help_text.setReadOnly(True)
        self.usage_help_text.setPlainText(self.USAGE_GUIDE_TEXT)
        usage_dialog_layout.addWidget(self.usage_help_text, 1)
        usage_close_row = QHBoxLayout()
        usage_close_row.addStretch(1)
        usage_close_btn = QPushButton("关闭")
        usage_close_btn.setObjectName("GhostBtnMin")
        usage_close_btn.clicked.connect(lambda: self._sync_usage_help_toggle(False))
        usage_close_btn.clicked.connect(lambda: self._animate_tool_dialog(self.usage_help_dialog, False))
        usage_close_row.addWidget(usage_close_btn, 0, Qt.AlignRight)
        usage_dialog_layout.addLayout(usage_close_row)
        self.usage_help_dialog.resize(820, 560)
        self.usage_help_dialog.finished.connect(lambda _code: self._sync_usage_help_toggle(False))

        nemo_enabled = bool(_get_nested(self.config._data, "asr.nemo_msdd.enabled", True))
        self.nemo_speaker_mode_combo.setEnabled(nemo_enabled)
        self.nemo_fixed_speakers_input.setEnabled(
            nemo_enabled and str(self.nemo_speaker_mode_combo.currentData() or "auto") == "manual"
        )

        posterior_panel = QFrame()
        posterior_panel.setObjectName("AdvancedPanel")
        posterior_panel_layout = QVBoxLayout(posterior_panel)
        posterior_panel_layout.setContentsMargins(10, 10, 10, 10)
        posterior_panel_layout.setSpacing(8)

        posterior_header = QHBoxLayout()
        posterior_header.setContentsMargins(0, 0, 0, 0)
        posterior_header.setSpacing(8)
        posterior_header.addWidget(self._section_label("Posterior Fusion 训练助手", "cyantext"))
        posterior_header.addStretch(1)
        self.posterior_help_button = QPushButton("查看说明")
        self.posterior_help_button.setObjectName("GhostBtnMin")
        self.posterior_help_button.clicked.connect(self._show_posterior_fusion_help)
        posterior_header.addWidget(self.posterior_help_button, 0, Qt.AlignRight)
        posterior_panel_layout.addLayout(posterior_header)

        posterior_intro = QLabel(
            "Posterior Fusion 的自动双跑默认关闭。只有当你明确开启“样本导出”时，"
            "才会同步打开“自动跑两次”：第一遍导样本并训练，第二遍再重跑当前队列。"
        )
        posterior_intro.setObjectName("ConfigGuide")
        posterior_intro.setWordWrap(True)
        posterior_panel_layout.addWidget(posterior_intro)

        posterior_examples_default = str(
            _get_nested(
                self.config._data,
                "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.examples_dir",
                _get_nested(
                    self.config._data,
                    "asr.nemo_msdd.hybrid_fusion.posterior_decoder.dump_examples.output_dir",
                    self._default_posterior_examples_relpath(),
                ),
            )
            or self._default_posterior_examples_relpath()
        )
        posterior_rttm_default = str(
            _get_nested(
                self.config._data,
                "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.rttm_dir",
                self._default_posterior_rttm_relpath(),
            )
            or self._default_posterior_rttm_relpath()
        )
        posterior_output_default = str(
            _get_nested(
                self.config._data,
                "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.output_path",
                _get_nested(
                    self.config._data,
                    "asr.nemo_msdd.hybrid_fusion.posterior_decoder.calibrator.persist_path",
                    self._default_posterior_output_relpath(),
                ),
            )
            or self._default_posterior_output_relpath()
        )
        posterior_epochs_default = int(
            _get_nested(
                self.config._data,
                "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.epochs",
                6,
            )
            or 6
        )
        posterior_auto_rerun_enabled = bool(
            _get_nested(self.config._data, "ui.posterior_fusion.auto_run_twice", False)
        )
        posterior_dump_enabled = bool(
            _get_nested(
                self.config._data,
                "asr.nemo_msdd.hybrid_fusion.posterior_decoder.dump_examples.enabled",
                False,
            )
        ) and posterior_auto_rerun_enabled

        posterior_grid = QGridLayout()
        posterior_grid.setContentsMargins(0, 0, 0, 0)
        posterior_grid.setSpacing(8)

        self.posterior_examples_dir_input = QLineEdit()
        self.posterior_examples_dir_input.setObjectName("ConfigInput")
        self.posterior_examples_dir_input.setText(posterior_examples_default)
        self.posterior_examples_dir_input.setPlaceholderText("导出的 posterior fusion 样本目录")
        self.posterior_examples_dir_input.textEdited.connect(self._on_advanced_setting_edited)
        self.posterior_examples_dir_button = QPushButton("浏览")
        self.posterior_examples_dir_button.setObjectName("GhostBtnMin")
        self.posterior_examples_dir_button.clicked.connect(self._browse_posterior_examples_dir)

        self.posterior_rttm_dir_input = QLineEdit()
        self.posterior_rttm_dir_input.setObjectName("ConfigInput")
        self.posterior_rttm_dir_input.setText(posterior_rttm_default)
        self.posterior_rttm_dir_input.setPlaceholderText("人工标注 RTTM 目录")
        self.posterior_rttm_dir_input.textEdited.connect(self._on_advanced_setting_edited)
        self.posterior_rttm_dir_button = QPushButton("浏览")
        self.posterior_rttm_dir_button.setObjectName("GhostBtnMin")
        self.posterior_rttm_dir_button.clicked.connect(self._browse_posterior_rttm_dir)

        self.posterior_output_path_input = QLineEdit()
        self.posterior_output_path_input.setObjectName("ConfigInput")
        self.posterior_output_path_input.setText(posterior_output_default)
        self.posterior_output_path_input.setPlaceholderText("校准器输出 JSON 路径")
        self.posterior_output_path_input.textEdited.connect(self._on_advanced_setting_edited)
        self.posterior_output_path_button = QPushButton("浏览")
        self.posterior_output_path_button.setObjectName("GhostBtnMin")
        self.posterior_output_path_button.clicked.connect(self._browse_posterior_output_path)

        self.posterior_epochs_input = QLineEdit()
        self.posterior_epochs_input.setObjectName("ConfigInput")
        self.posterior_epochs_input.setText(str(max(1, posterior_epochs_default)))
        self.posterior_epochs_input.setToolTip("离线训练轮数。默认 6；开发集较小时可先用 4~6。")
        self.posterior_epochs_input.textEdited.connect(self._on_advanced_setting_edited)

        posterior_grid.addWidget(QLabel("样本目录:"), 0, 0)
        posterior_grid.addWidget(self.posterior_examples_dir_input, 0, 1)
        posterior_grid.addWidget(self.posterior_examples_dir_button, 0, 2)
        posterior_grid.addWidget(QLabel("RTTM 目录:"), 1, 0)
        posterior_grid.addWidget(self.posterior_rttm_dir_input, 1, 1)
        posterior_grid.addWidget(self.posterior_rttm_dir_button, 1, 2)
        posterior_grid.addWidget(QLabel("校准器输出:"), 2, 0)
        posterior_grid.addWidget(self.posterior_output_path_input, 2, 1)
        posterior_grid.addWidget(self.posterior_output_path_button, 2, 2)
        posterior_grid.addWidget(QLabel("训练轮数:"), 3, 0)
        posterior_grid.addWidget(self.posterior_epochs_input, 3, 1)
        posterior_panel_layout.addLayout(posterior_grid)

        posterior_actions = QHBoxLayout()
        posterior_actions.setContentsMargins(0, 0, 0, 0)
        posterior_actions.setSpacing(8)
        self.posterior_dump_toggle_button = QPushButton()
        self.posterior_dump_toggle_button.setObjectName("GhostBtnMin")
        self.posterior_dump_toggle_button.setCheckable(True)
        self.posterior_dump_toggle_button.setChecked(posterior_dump_enabled)
        self.posterior_dump_toggle_button.toggled.connect(self._on_posterior_dump_toggled)
        self.posterior_auto_rerun_toggle_button = QPushButton()
        self.posterior_auto_rerun_toggle_button.setObjectName("GhostBtnMin")
        self.posterior_auto_rerun_toggle_button.setCheckable(True)
        self.posterior_auto_rerun_toggle_button.setChecked(posterior_auto_rerun_enabled)
        self.posterior_auto_rerun_toggle_button.toggled.connect(self._on_posterior_auto_rerun_toggled)
        self.posterior_train_button = QPushButton("开始训练校准器")
        self.posterior_train_button.setObjectName("PrimaryBtn")
        self.posterior_train_button.clicked.connect(self.start_posterior_fusion_training)
        posterior_actions.addWidget(self.posterior_dump_toggle_button, 0, Qt.AlignLeft)
        posterior_actions.addWidget(self.posterior_auto_rerun_toggle_button, 0, Qt.AlignLeft)
        posterior_actions.addWidget(self.posterior_train_button, 0, Qt.AlignLeft)
        posterior_actions.addStretch(1)
        posterior_panel_layout.addLayout(posterior_actions)
        self.posterior_panel = posterior_panel
        self._sync_posterior_dump_button_text()
        self._sync_posterior_auto_rerun_button_text()

        self.posterior_dialog = QDialog(self)
        self.posterior_dialog.setWindowTitle("Posterior Fusion 训练助手")
        self.posterior_dialog.setModal(False)
        self.posterior_dialog.setWindowFlag(Qt.Tool, False)
        self.posterior_dialog.setWindowFlag(Qt.Dialog, True)
        self.posterior_dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.posterior_dialog.setAttribute(Qt.WA_StyledBackground, True)
        self.posterior_dialog.setProperty("glassWindow", True)
        if self.is_macos:
            self.posterior_dialog.setAttribute(Qt.WA_TranslucentBackground, True)
        posterior_dialog_layout = QVBoxLayout(self.posterior_dialog)
        posterior_dialog_layout.setContentsMargins(12, 12, 12, 12)
        posterior_dialog_layout.setSpacing(8)
        posterior_dialog_layout.addWidget(self.posterior_panel)
        posterior_close_row = QHBoxLayout()
        posterior_close_row.addStretch(1)
        posterior_close_btn = QPushButton("关闭")
        posterior_close_btn.setObjectName("GhostBtnMin")
        posterior_close_btn.clicked.connect(lambda: self._sync_posterior_assistant_toggle(False))
        posterior_close_btn.clicked.connect(lambda: self._animate_tool_dialog(self.posterior_dialog, False))
        posterior_close_row.addWidget(posterior_close_btn, 0, Qt.AlignRight)
        posterior_dialog_layout.addLayout(posterior_close_row)
        self.posterior_dialog.resize(920, 360)
        self.posterior_dialog.finished.connect(lambda _code: self._sync_posterior_assistant_toggle(False))

        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QFrame.NoFrame)
        config_scroll.setObjectName("ConfigScroll")
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        config_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        config_scroll.viewport().setObjectName("ConfigViewport")
        config_scroll.viewport().setAutoFillBackground(False)
        config_scroll.setMinimumHeight(220)
        self.config_scroll = config_scroll

        config_content = QWidget()
        config_content.setObjectName("ConfigScrollContent")
        cfg_rows = QGridLayout(config_content)
        cfg_rows.setContentsMargins(0, 4, 0, 8)
        cfg_rows.setSpacing(6)
        cfg_rows.setAlignment(Qt.AlignTop)
        self.config_switch_grid = cfg_rows
        row_idx = 0
        col_idx = 0
        self._switch_row_widgets = []
        self._switch_buttons = []
        switch_palettes = [
            ("#2f9dff", "#778592"),
            ("#26b39a", "#7b8793"),
            ("#4e7aff", "#78849a"),
            ("#2aa66a", "#7d8590"),
            ("#d18a00", "#7c8794"),
            ("#a068ff", "#7c8696"),
            ("#0f9fa8", "#7b8696"),
            ("#cc6f00", "#7d8895"),
            ("#6f7d8f", "#7f8a95"),
            ("#2c8be0", "#788796"),
            ("#17839e", "#788795"),
            ("#4f8fda", "#7b8797"),
            ("#4f6bd8", "#7a8495"),
            ("#4f8ad8", "#7d8894"),
            ("#6d7ea2", "#7f8b96"),
        ]
        for title_text, key, desc in self._active_switch_specs():
            row_widget = QFrame()
            row_widget.setObjectName("SwitchRow")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(8, 2, 8, 2)
            row_layout.setSpacing(6)

            text_col = QVBoxLayout()
            text_col.setSpacing(0)
            text_col.setAlignment(Qt.AlignVCenter)
            title_label = QLabel(title_text)
            title_label.setObjectName("SwitchTitle")
            title_label.setWordWrap(True)
            desc_label = QLabel(desc)
            desc_label.setObjectName("SwitchDesc")
            desc_label.setWordWrap(True)
            text_col.addWidget(title_label)
            text_col.addWidget(desc_label)

            checked = bool(_get_nested(self.config._data, key, False))
            switch = ColorToggleButton(checked=checked)
            palette = switch_palettes[len(self._switch_buttons) % len(switch_palettes)]
            switch.set_palette(palette[0], palette[1])
            switch.toggled.connect(
                lambda v, dotted=key: self._on_switch_toggled(dotted, bool(v))
            )
            help_btn = self._make_help_dot_button(key, title_text)

            row_layout.addLayout(text_col, 1)
            row_layout.addWidget(help_btn, 0, Qt.AlignVCenter)
            row_layout.addWidget(switch, 0, Qt.AlignVCenter)
            self._switch_row_widgets.append(row_widget)
            self._switch_buttons.append(switch)
            cfg_rows.addWidget(row_widget, row_idx, col_idx)
            col_idx += 1
            if col_idx > 1:  # 每行显示2个
                col_idx = 0
                row_idx += 1
        cfg_rows.setRowStretch(row_idx + 1, 1)
        config_scroll.setWidget(config_content)
        config_layout.addWidget(config_scroll, 3)
        config_card.setMinimumHeight(240)
        self.config_card = config_card
        left_panel.addWidget(config_card)
        left_panel.setStretchFactor(3, 14)
        results_card = QFrame()
        results_card.setObjectName("Card")
        results_card.setMinimumHeight(180)
        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(12, 12, 12, 12)
        results_layout.setSpacing(8)
        results_header = QHBoxLayout()
        results_header.setContentsMargins(0, 0, 0, 0)
        results_header.setSpacing(8)
        results_header.addWidget(self._section_label("处理结果产物", "cyantext"))
        results_header.addStretch(1)

        self.reload_results_button = QPushButton("读取产物")
        self.reload_results_button.setObjectName("GhostBtnMin")
        self.reload_results_button.clicked.connect(self.reload_results_from_disk)

        self.read_result_button = QPushButton("读取文件")
        self.read_result_button.setObjectName("GhostBtnMin")
        self.read_result_button.clicked.connect(self.read_selected_result_file)

        self.delete_result_button = HoldToDeleteButton("长按删除", hold_ms=1200)
        self.delete_result_button.setObjectName("DangerHoldBtn")
        self.delete_result_button.setToolTip("按住约 1.2 秒删除；中途松开不会删除")
        self.delete_result_button.confirmed.connect(self.delete_selected_result_item)

        results_header.addWidget(self.reload_results_button, 0, Qt.AlignRight)
        results_header.addWidget(self.read_result_button, 0, Qt.AlignRight)
        results_header.addWidget(self.delete_result_button, 0, Qt.AlignRight)
        results_layout.addLayout(results_header)

        self.result_tree = QTreeWidget()
        self.result_tree.setObjectName("TreeSurface")
        self.result_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.result_tree.setHeaderLabels(["项目 / 文件名称", "状态", "本地磁盘路径"])
        self.result_tree.setColumnWidth(0, 280)
        self.result_tree.setColumnWidth(1, 90)
        self.result_tree.itemDoubleClicked.connect(self.open_item_path)
        self.result_tree.itemSelectionChanged.connect(self._update_result_action_state)
        results_layout.addWidget(self.result_tree, 1)
        right_panel.addWidget(results_card)
        right_panel.setStretchFactor(0, 4)

        event_card = QFrame()
        event_card.setObjectName("Card")
        event_card.setMinimumHeight(150)
        event_layout = QVBoxLayout(event_card)
        event_layout.setContentsMargins(12, 12, 12, 12)
        event_layout.setSpacing(8)

        event_header = QHBoxLayout()
        event_header.setContentsMargins(0, 0, 0, 0)
        event_header.setSpacing(8)
        event_header.addWidget(self._section_label("关键进度流", "cyantext"))
        event_header.addStretch(1)
        self.toggle_debug_log_button = QPushButton("显示详细日志")
        self.toggle_debug_log_button.setObjectName("GhostBtnMin")
        self.toggle_debug_log_button.setCheckable(True)
        self.toggle_debug_log_button.toggled.connect(self._toggle_debug_log)
        event_header.addWidget(self.toggle_debug_log_button, 0, Qt.AlignRight)
        event_layout.addLayout(event_header)

        self.event_feed = BurstEventFeed(
            event_card,
            max_items=14,
            role="event",
            default_ttl_ms=9400,
            overlay_stack=True,
        )
        event_layout.addWidget(self.event_feed, 1)
        right_panel.addWidget(event_card)
        right_panel.setStretchFactor(1, 2)

        # 调试日志区域（默认折叠）
        log_card = QFrame()
        log_card.setObjectName("Card")
        log_card.setMinimumHeight(160)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(12, 12, 12, 12)
        log_layout.setSpacing(8)
        
        log_header = QHBoxLayout()
        log_header.setContentsMargins(0, 0, 0, 0)
        log_header.setSpacing(8)
        log_header.addWidget(self._section_label("系统运行消息流", "cyantext"))
        log_header.addStretch(1)
        self.clear_log_file_button = QPushButton("清空日志记录")
        self.clear_log_file_button.setObjectName("GhostBtnMin")
        self.clear_log_file_button.clicked.connect(self.delete_pipeline_log)
        log_header.addWidget(self.clear_log_file_button, 0, Qt.AlignRight)
        log_layout.addLayout(log_header)

        self.log_feed = BurstEventFeed(log_card, max_items=26, role="log", default_ttl_ms=0)
        log_layout.addWidget(self.log_feed, 1)
        right_panel.addWidget(log_card)
        right_panel.setStretchFactor(2, 2)
        log_card.setVisible(False)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        root_splitter.addWidget(splitter)
        root_splitter.setStretchFactor(0, 0)
        root_splitter.setStretchFactor(1, 0)
        root_splitter.setStretchFactor(2, 0)
        root_splitter.setStretchFactor(3, 1)
        main_layout.addWidget(root_splitter, 1)
        self.header_card = header_card
        self.progress_card = progress_card
        self.queue_card = queue_card
        self.config_card = config_card
        self.results_card = results_card
        self.event_card = event_card
        self.log_card = log_card

        for idx in range(root_splitter.count()):
            root_splitter.setCollapsible(idx, False)
        for idx in range(left_panel.count()):
            left_panel.setCollapsible(idx, False)
        for idx in range(right_panel.count()):
            right_panel.setCollapsible(idx, False)
        for idx in range(splitter.count()):
            splitter.setCollapsible(idx, False)

        root_splitter.splitterMoved.connect(self._on_splitter_moved)
        left_panel.splitterMoved.connect(self._on_splitter_moved)
        right_panel.splitterMoved.connect(self._on_splitter_moved)
        splitter.splitterMoved.connect(self._on_splitter_moved)

        self._cards = [
            header_card,
            progress_card,
            stepper_card,
            self.drop_area,
            queue_card,
            controls_card,
            config_card,
            results_card,
            event_card,
            log_card,
        ]
        self._refresh_action_button_states()
        self._sync_global_switch_button_state()
        self._update_result_action_state()
        self.reload_results_from_disk(silent=True)
        self._schedule_layout_sync(0)
        self._schedule_layout_sync(120)

    def _setup_macos_menu_bar(self) -> None:
        if not self.is_macos:
            return
        menu_bar = self.menuBar()
        if menu_bar is None:
            return
        try:
            menu_bar.setNativeMenuBar(True)
        except Exception:
            pass
        menu_bar.clear()
        self._macos_menu_actions = {}

        file_menu = menu_bar.addMenu("文件")
        run_menu = menu_bar.addMenu("运行")
        window_menu = menu_bar.addMenu("窗口")
        help_menu = menu_bar.addMenu("帮助")

        self._create_macos_menu_action(
            file_menu,
            "add_files",
            "添加文件…",
            self.browse_files,
            shortcut="Meta+O",
        )
        self._create_macos_menu_action(
            file_menu,
            "save_config",
            "保存配置",
            self.save_config_overrides,
            shortcut="Meta+S",
        )
        self._create_macos_menu_action(
            file_menu,
            "open_output_dir",
            "打开产物目录",
            self.open_output_dir,
            shortcut="Meta+Shift+O",
        )
        self._create_macos_menu_action(
            file_menu,
            "read_selected_result",
            "读取选中产物",
            self.read_selected_result_file,
            shortcut="Meta+Shift+R",
        )
        self._create_macos_menu_action(
            file_menu,
            "reload_results",
            "刷新产物列表",
            self.reload_results_from_disk,
            shortcut="Meta+R",
        )
        file_menu.addSeparator()
        self._create_macos_menu_action(
            file_menu,
            "preferences_role",
            "偏好设置…",
            self._open_preferences_dialog,
            shortcut="Meta+,",
            menu_role=QAction.PreferencesRole,
        )
        self._create_macos_menu_action(
            file_menu,
            "quit_app",
            "退出 AI 语音转录工作站",
            self.close,
            menu_role=QAction.QuitRole,
        )

        self._create_macos_menu_action(
            run_menu,
            "start_processing",
            "开始处理",
            self.start_processing,
            shortcut="Meta+Return",
        )
        self._create_macos_menu_action(
            run_menu,
            "pause_processing",
            "暂停任务",
            self.toggle_pause,
        )
        run_menu.addSeparator()
        self._create_macos_menu_action(
            run_menu,
            "clear_temp",
            "清理临时文件",
            self.delete_temp_files,
            shortcut="Meta+Shift+T",
        )
        self._create_macos_menu_action(
            run_menu,
            "full_cleanup",
            "执行全清理",
            self.run_full_cleanup,
        )

        self._create_macos_menu_action(
            window_menu,
            "show_advanced_window",
            "高级配置窗口",
            lambda checked: self.advanced_toggle_button.setChecked(bool(checked)),
            checkable=True,
        )
        self._create_macos_menu_action(
            window_menu,
            "show_posterior_assistant",
            "Posterior Fusion 助手",
            lambda checked: self.posterior_assistant_toggle_button.setChecked(bool(checked)),
            checkable=True,
        )
        self._create_macos_menu_action(
            window_menu,
            "show_debug_log",
            "详细日志",
            lambda checked: self.toggle_debug_log_button.setChecked(bool(checked)),
            checkable=True,
            shortcut="Meta+Shift+L",
        )

        self._create_macos_menu_action(
            help_menu,
            "open_usage_help",
            "使用说明",
            self._open_usage_help_dialog,
        )
        self._create_macos_menu_action(
            help_menu,
            "about_app",
            "关于 AI 语音转录工作站",
            self._show_about_dialog,
            menu_role=QAction.AboutRole,
        )

        self._sync_macos_menu_state()

    def _create_macos_menu_action(
        self,
        menu,
        key: str,
        text: str,
        slot,
        *,
        shortcut: Optional[str] = None,
        checkable: bool = False,
        menu_role: Optional[QAction.MenuRole] = None,
    ) -> QAction:
        action = QAction(text, self)
        action.setCheckable(bool(checkable))
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if menu_role is not None:
            try:
                action.setMenuRole(menu_role)
            except Exception:
                pass
        if slot is not None:
            if checkable:
                action.toggled.connect(slot)
            else:
                action.triggered.connect(slot)
        menu.addAction(action)
        self._macos_menu_actions[key] = action
        return action

    def _setup_shortcuts(self) -> None:
        shortcuts: List[QShortcut] = []

        def bind(key: str, callback):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(callback)
            shortcuts.append(shortcut)

        bind("Ctrl+O", self.browse_files)
        bind("Ctrl+Return", self.start_processing)
        bind("Ctrl+Enter", self.start_processing)
        bind("Space", self.toggle_pause)
        bind("Ctrl+L", lambda: self.toggle_debug_log_button.toggle())
        bind("Ctrl+Shift+L", lambda: self.delete_pipeline_log())
        bind("Delete", self.remove_selected_queue_items)
        bind("Ctrl+Shift+T", lambda: self.delete_temp_files())
        bind("Ctrl+,", lambda: self.advanced_toggle_button.toggle())
        bind("F1", lambda: self.usage_help_button.toggle())

        self._shortcuts = shortcuts

    def _set_macos_action_checked(self, key: str, checked: bool) -> None:
        action = self._macos_menu_actions.get(key)
        if action is None or not action.isCheckable():
            return
        action.blockSignals(True)
        try:
            action.setChecked(bool(checked))
        finally:
            action.blockSignals(False)

    def _sync_macos_menu_state(self) -> None:
        if not self.is_macos or not self._macos_menu_actions:
            return

        def _mirror_enabled(action_key: str, widget_name: str) -> None:
            action = self._macos_menu_actions.get(action_key)
            widget = getattr(self, widget_name, None)
            if action is not None and widget is not None:
                action.setEnabled(bool(widget.isEnabled()))

        _mirror_enabled("add_files", "add_button")
        _mirror_enabled("save_config", "save_config_button")
        _mirror_enabled("read_selected_result", "read_result_button")
        _mirror_enabled("reload_results", "reload_results_button")
        _mirror_enabled("start_processing", "start_button")
        _mirror_enabled("pause_processing", "pause_button")
        _mirror_enabled("clear_temp", "clear_temp_button")
        _mirror_enabled("full_cleanup", "full_cleanup_button")

        pause_action = self._macos_menu_actions.get("pause_processing")
        if pause_action is not None:
            pause_action.setText("恢复任务" if self.paused else "暂停任务")

        output_action = self._macos_menu_actions.get("open_output_dir")
        if output_action is not None:
            output_action.setEnabled(True)

        self._set_macos_action_checked(
            "show_advanced_window",
            bool(getattr(self, "advanced_toggle_button", None) and self.advanced_toggle_button.isChecked()),
        )
        self._set_macos_action_checked(
            "show_posterior_assistant",
            bool(
                getattr(self, "posterior_assistant_toggle_button", None)
                and self.posterior_assistant_toggle_button.isChecked()
            ),
        )
        self._set_macos_action_checked(
            "show_debug_log",
            bool(
                getattr(self, "toggle_debug_log_button", None)
                and self.toggle_debug_log_button.isChecked()
            ),
        )
        _mirror_enabled("show_advanced_window", "advanced_toggle_button")
        _mirror_enabled("show_posterior_assistant", "posterior_assistant_toggle_button")
        _mirror_enabled("show_debug_log", "toggle_debug_log_button")

    def _open_preferences_dialog(self) -> None:
        if hasattr(self, "advanced_toggle_button") and self.advanced_toggle_button is not None:
            self.advanced_toggle_button.setChecked(True)

    def _open_usage_help_dialog(self) -> None:
        if hasattr(self, "usage_help_button") and self.usage_help_button is not None:
            self.usage_help_button.setChecked(True)

    def open_output_dir(self, _checked: bool = False) -> None:
        output_root = self._resolve_output_dir()
        if not output_root.exists():
            self.set_status_chip("产物目录尚未生成", "info")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_root)))
        self.set_status_chip(f"已打开产物目录: {output_root.name}", "info")

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "关于 AI 语音转录工作站",
            "AI 语音转录工作站\n\n"
            "音视频转录、分角色、翻译、报告和字幕回写的一体化桌面界面。\n"
            "当前 macOS 版本已补充原生菜单栏入口，便于打包后的日常使用。",
        )

    def _refresh_stepper_buttons(self) -> None:
        for idx, btn in enumerate(self._step_buttons, start=1):
            btn.setChecked(idx == self._current_step)
            btn.setProperty("activeStep", idx == self._current_step)
            btn.setEnabled((not self.running) or idx == 3)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.prev_step_button.setEnabled(self._current_step > 1 and not self.running)
        self.next_step_button.setEnabled(self._current_step < 3 and not self.running)
        if self._current_step == 1:
            self.next_step_button.setText("下一步")
        elif self._current_step == 2:
            self.next_step_button.setText("进入第 3 步")
        else:
            self.next_step_button.setText("下一步")

    def _finish_widget_fade(
        self,
        key: int,
        widget: QWidget,
        effect: QGraphicsOpacityEffect,
        anim: QPropertyAnimation,
        visible: bool,
    ) -> None:
        current = self._widget_fade_anims.get(key)
        if current is not anim:
            return
        self._widget_fade_anims.pop(key, None)
        if not visible:
            widget.hide()
        if widget.graphicsEffect() is effect:
            widget.setGraphicsEffect(None)
        self._schedule_layout_sync(0)

    def _set_widget_visible_animated(
        self,
        widget: Optional[QWidget],
        visible: bool,
        animate: bool = True,
        duration_ms: int = 170,
    ) -> None:
        if widget is None:
            return
        key = id(widget)
        existing = self._widget_fade_anims.pop(key, None)
        if existing is not None:
            existing.stop()

        if not animate:
            if isinstance(widget.graphicsEffect(), QGraphicsOpacityEffect):
                widget.setGraphicsEffect(None)
            widget.setVisible(bool(visible))
            return

        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(1.0 if widget.isVisible() else 0.0)
            widget.setGraphicsEffect(effect)

        if visible:
            if not widget.isVisible():
                widget.show()
                effect.setOpacity(0.0)
            start = float(effect.opacity())
            if start >= 0.98:
                widget.setGraphicsEffect(None)
                return
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(max(80, int(duration_ms)))
            anim.setStartValue(max(0.0, min(1.0, start)))
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.finished.connect(
                lambda k=key, w=widget, e=effect, a=anim: self._finish_widget_fade(k, w, e, a, True)
            )
        else:
            if not widget.isVisible():
                widget.setGraphicsEffect(None)
                return
            start = float(effect.opacity())
            if start <= 0.02:
                widget.hide()
                widget.setGraphicsEffect(None)
                return
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(max(80, int(duration_ms)))
            anim.setStartValue(max(0.0, min(1.0, start)))
            anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.finished.connect(
                lambda k=key, w=widget, e=effect, a=anim: self._finish_widget_fade(k, w, e, a, False)
            )

        self._widget_fade_anims[key] = anim
        anim.start()

    def _finish_step_transition_anim(
        self,
        target: QWidget,
        effect: QGraphicsOpacityEffect,
        anim: QPropertyAnimation,
    ) -> None:
        if self._step_transition_anim is not anim:
            return
        self._step_transition_anim = None
        if target.graphicsEffect() is effect:
            target.setGraphicsEffect(None)

    def _animate_step_transition(self) -> None:
        target = getattr(self, "main_splitter", None)
        if target is None:
            return
        effect = target.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(target)
            effect.setOpacity(1.0)
            target.setGraphicsEffect(effect)
        if self._step_transition_anim is not None:
            self._step_transition_anim.stop()
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(220)
        anim.setStartValue(0.22 if effect.opacity() >= 0.98 else max(0.0, float(effect.opacity())))
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.finished.connect(
            lambda t=target, e=effect, a=anim: self._finish_step_transition_anim(t, e, a)
        )
        anim.start()
        self._step_transition_anim = anim

    def _apply_wizard_step(self, step: int, animate: bool = True) -> None:
        step = max(1, min(3, int(step)))
        self._current_step = step

        if step == 1:
            self._set_widget_visible_animated(self.drop_area, True, animate=animate)
            self._set_widget_visible_animated(self.queue_card, True, animate=animate)
            self._set_widget_visible_animated(self.config_card, False, animate=animate)
            self._set_widget_visible_animated(self.results_card, False, animate=animate)
            self._set_widget_visible_animated(self.event_card, False, animate=animate)
            self._set_widget_visible_animated(self.log_card, False, animate=animate)
            self._set_widget_visible_animated(self.right_panel_splitter, False, animate=animate)
            self.add_button.setVisible(True)
            self.remove_checked_button.setVisible(True)
            self.clear_button.setVisible(True)
            self.start_button.setVisible(False)
            self.pause_button.setVisible(False)
            self.clear_temp_button.setVisible(False)
            self.full_cleanup_button.setVisible(False)
            self.set_status_chip("步骤 1/3：添加待处理文件", "info")
        elif step == 2:
            self._set_widget_visible_animated(self.drop_area, False, animate=animate)
            self._set_widget_visible_animated(self.queue_card, False, animate=animate)
            self._set_widget_visible_animated(self.config_card, True, animate=animate)
            self._set_widget_visible_animated(self.results_card, False, animate=animate)
            self._set_widget_visible_animated(self.event_card, False, animate=animate)
            self._set_widget_visible_animated(self.log_card, False, animate=animate)
            self._set_widget_visible_animated(self.right_panel_splitter, False, animate=animate)
            self.add_button.setVisible(False)
            self.remove_checked_button.setVisible(False)
            self.clear_button.setVisible(False)
            self.start_button.setVisible(False)
            self.pause_button.setVisible(False)
            self.clear_temp_button.setVisible(False)
            self.full_cleanup_button.setVisible(False)
            self.set_status_chip("步骤 2/3：检查并确认参数配置", "info")
        else:
            self._set_widget_visible_animated(self.drop_area, False, animate=animate)
            self._set_widget_visible_animated(self.queue_card, True, animate=animate)
            self._set_widget_visible_animated(self.config_card, False, animate=animate)
            self._set_widget_visible_animated(self.results_card, True, animate=animate)
            self._set_widget_visible_animated(self.event_card, True, animate=animate)
            self._set_widget_visible_animated(self.right_panel_splitter, True, animate=animate)
            debug_visible = bool(self.toggle_debug_log_button.isChecked())
            self._set_widget_visible_animated(self.log_card, debug_visible, animate=animate)
            if debug_visible:
                self._flush_log_buffer(flush_all=False)
                if self._log_buffer and not self._log_flush_timer.isActive():
                    self._log_flush_timer.start()
            self.add_button.setVisible(True)
            self.remove_checked_button.setVisible(True)
            self.clear_button.setVisible(True)
            self.start_button.setVisible(True)
            self.pause_button.setVisible(True)
            self.clear_temp_button.setVisible(True)
            self.full_cleanup_button.setVisible(True)
            if not self.running:
                self.set_status_chip("步骤 3/3：开始处理并关注关键进度", "info")

        self._refresh_stepper_buttons()
        self._refresh_action_button_states()
        self._sync_controls_row_width()
        self._schedule_layout_sync(0)
        if animate:
            self._animate_step_transition()

    def _go_to_step(self, step: int) -> None:
        prev_step = int(self._current_step)
        if self.running and step != 3:
            return
        if step == 2 and not self.selected_files:
            QMessageBox.information(self, "先添加文件", "请先在步骤 1 添加至少一个文件。")
            self._apply_wizard_step(1, animate=True)
            return
        self._apply_wizard_step(step, animate=True)
        if int(step) > prev_step:
            self._force_refresh_progress_after_step_change()

    def _go_prev_step(self) -> None:
        self._go_to_step(self._current_step - 1)

    def _go_next_step(self) -> None:
        self._go_to_step(self._current_step + 1)

    def _force_refresh_progress_after_step_change(self) -> None:
        if self._download_inflight:
            now = time.monotonic()
            if self._download_last_event_at > 0:
                stale_sec = max(0.0, now - self._download_last_event_at)
            elif self._download_started_at > 0:
                stale_sec = max(0.0, now - self._download_started_at)
            else:
                stale_sec = 0.0

            if stale_sec >= 8.0:
                self.append_log(
                    "[模型下载] 检测到下载状态长时间未更新，已在切换步骤时强制恢复任务进度显示。"
                )
                self._restore_file_progress_view(self._last_file_progress_label)
                return

            self._set_download_busy(
                self.progress_label.text(),
                progress_percent=self._download_last_percent,
            )
            return

        self._set_file_progress(
            self._progress_done,
            self._progress_total,
            self._last_file_progress_label or self.progress_label.text(),
            file_percent=self._current_file_percent,
            overall_percent=self._overall_percent,
        )

    def _toggle_debug_log(self, checked: bool) -> None:
        self.toggle_debug_log_button.setText("隐藏详细日志" if checked else "显示详细日志")
        if self._current_step == 3:
            self._set_widget_visible_animated(self.log_card, bool(checked), animate=True, duration_ms=150)
            if checked:
                self._flush_log_buffer(flush_all=False)
                if self._log_buffer and not self._log_flush_timer.isActive():
                    self._log_flush_timer.start()
            self._schedule_layout_sync(0)
        self._sync_macos_menu_state()

    def _push_event(self, text: str, level: str = "info", ttl_ms: Optional[int] = None) -> None:
        feed = getattr(self, "event_feed", None)
        if feed is None:
            return
        clean = str(text or "").strip()
        if not clean:
            return
        if level not in {"info", "ok", "warn", "err"}:
            level = "info"
        if ttl_ms is None:
            ttl_ms = {
                "err": 22000,
                "warn": 14000,
                "ok": 12000,
                "info": 10000,
            }.get(level, 10000)
        feed.push_event(clean, level=level, ttl_ms=ttl_ms)

    def _sync_global_switch_button_state(self) -> None:
        if not hasattr(self, "toggle_all_switches_button"):
            return
        if not self._switch_buttons:
            self.toggle_all_switches_button.setEnabled(False)
            return
        all_enabled = all(btn.isChecked() for btn in self._switch_buttons)
        any_enabled = any(btn.isChecked() for btn in self._switch_buttons)
        self.toggle_all_switches_button.blockSignals(True)
        try:
            self.toggle_all_switches_button.setChecked(all_enabled)
            self.toggle_all_switches_button.setText("一键全关" if all_enabled else "一键全开")
        finally:
            self.toggle_all_switches_button.blockSignals(False)

        mode = "mixed"
        if all_enabled:
            mode = "all"
        elif not any_enabled:
            mode = "none"
        self.config_card.setProperty("switchMode", mode)
        self.config_card.style().unpolish(self.config_card)
        self.config_card.style().polish(self.config_card)

    def toggle_all_switches(self, checked: bool = False) -> None:
        target = bool(checked)
        for switch in self._switch_buttons:
            switch.setChecked(target)
        self._switch_dirty = True
        self._sync_global_switch_button_state()
        if target:
            self.set_status_chip("已批量启用全部开关（绿色）", "ok")
        else:
            self.set_status_chip("已批量关闭全部开关（灰色）", "info")

    def _maybe_show_first_run_tour(self) -> None:
        if not self._tour_pending or self._tour_shown or not self.isVisible():
            return
        self._tour_shown = True
        if self._tour_overlay is None:
            self._tour_overlay = StepOnboardingOverlay(self)
        self._tour_overlay.set_accent_color(self._accent_color)
        step_one_target = self._step_buttons[0] if len(self._step_buttons) >= 1 else self.stepper_card
        step_two_target = self._step_buttons[1] if len(self._step_buttons) >= 2 else self.stepper_card
        step_three_target = self._step_buttons[2] if len(self._step_buttons) >= 3 else self.stepper_card
        steps = [
            (
                self.stepper_card,
                "第一次使用：三步工作流",
                "顶部的 1/2/3 是完整流程。你可以逐步操作，也可以点击任一步快速跳转。",
            ),
            (
                step_one_target,
                "步骤 1：导入文件",
                "先点“1 · 导入文件”，把音视频拖进大加号区域。快捷键：Ctrl+O 添加文件，Delete 删除选中。",
            ),
            (
                step_two_target,
                "步骤 2：配置参数",
                "点“2 · 配置参数”，调整语言、翻译和高级项。问号会说明用途、好处、坏处和适合场景。",
            ),
            (
                step_three_target,
                "步骤 3：开始处理",
                "点“3 · 开始处理”后再点击开始。快捷键：Ctrl+Enter 开始，Space 暂停/恢复，Ctrl+L 切换调试日志。",
            ),
        ]
        self._tour_overlay.start(steps, on_finished=self._mark_first_run_tour_done)

    def _mark_first_run_tour_done(self) -> None:
        self._tour_pending = False
        _set_nested(self.config._data, "ui.onboarding.stepper_v1_done", True)
        save_path = self._config_save_path()
        user_cfg: Dict[str, Any] = {}
        try:
            if save_path.exists():
                with open(save_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    user_cfg = loaded
            _set_nested(user_cfg, "ui.onboarding.stepper_v1_done", True)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(user_cfg, f, allow_unicode=True, sort_keys=False)
        except Exception:
            pass

    @Slot(int, int)
    def _on_splitter_moved(self, _pos: int, _index: int):
        self._schedule_layout_sync(0)

    def _schedule_layout_sync(self, delay_ms: int = 0):
        if not hasattr(self, "_resize_reflow_timer"):
            return
        self._layout_sync_pending = True
        delay = max(0, int(delay_ms))
        if not self._resize_reflow_timer.isActive():
            self._resize_reflow_timer.start(delay)
            return
        remaining = self._resize_reflow_timer.remainingTime()
        if remaining < 0 or delay < remaining:
            self._resize_reflow_timer.start(delay)

    def _sync_responsive_layout(self):
        self._layout_sync_pending = False
        self._reflow_switch_rows()
        self._sync_controls_row_width()
        self._rebalance_splitters(force=not self._splitter_layout_initialized)
        self._splitter_layout_initialized = True
        self._keep_aux_dialogs_in_view()

    def _reflow_switch_rows(self):
        if not hasattr(self, "config_switch_grid") or not self._switch_row_widgets:
            return
        width = 0
        if hasattr(self, "config_scroll") and self.config_scroll is not None:
            width = max(width, int(self.config_scroll.viewport().width()))
        if hasattr(self, "config_card") and self.config_card is not None:
            width = max(width, int(self.config_card.width()))
        if width <= 0:
            return

        fold_threshold = self._scaled_px(760, min_px=680, max_px=980)
        columns = 1 if width < fold_threshold else 2
        if columns == self._switch_grid_columns:
            return

        for widget in self._switch_row_widgets:
            self.config_switch_grid.removeWidget(widget)

        for i, widget in enumerate(self._switch_row_widgets):
            row = i // columns
            col = i % columns
            self.config_switch_grid.addWidget(widget, row, col)

        self.config_switch_grid.setColumnStretch(0, 1)
        self.config_switch_grid.setColumnStretch(1, 1 if columns == 2 else 0)
        row_count = (len(self._switch_row_widgets) + columns - 1) // columns
        for row in range(max(1, len(self._switch_row_widgets) + 2)):
            self.config_switch_grid.setRowStretch(row, 0)
        self.config_switch_grid.setRowStretch(row_count, 1)
        self._switch_grid_columns = columns

    def _sync_controls_row_width(self):
        row = getattr(self, "controls_row_widget", None)
        layout = getattr(self, "controls_row_layout", None)
        if row is None or layout is None:
            return
        try:
            target_width = max(0, int(layout.sizeHint().width()) + 8)
            row.setMinimumWidth(target_width)
        except Exception:
            pass

    @staticmethod
    def _fit_splitter_sizes(
        base_sizes: List[int],
        min_sizes: List[int],
        total: int,
        grow_priority: Optional[int] = None,
    ) -> List[int]:
        count = min(len(base_sizes), len(min_sizes))
        if count <= 0:
            return []
        total = max(0, int(total))
        mins = [max(0, int(v)) for v in min_sizes[:count]]
        if total <= 0:
            return mins

        min_total = sum(mins)
        if min_total >= total:
            if min_total <= 0:
                base = total // count
                sizes = [base] * count
                for i in range(total - base * count):
                    sizes[i % count] += 1
                return sizes
            raw = [m * total / min_total for m in mins]
            sizes = [int(v) for v in raw]
            remainder = total - sum(sizes)
            fracs = sorted(
                range(count),
                key=lambda i: (raw[i] - sizes[i], mins[i]),
                reverse=True,
            )
            for i in fracs[:remainder]:
                sizes[i] += 1
            return sizes

        sizes = [max(mins[i], int(base_sizes[i])) for i in range(count)]
        if grow_priority is None or not (0 <= int(grow_priority) < count):
            grow_priority = count - 1
        grow_priority = int(grow_priority)

        current_total = sum(sizes)
        if current_total < total:
            sizes[grow_priority] += total - current_total
            return sizes

        overflow = current_total - total
        order = sorted(
            range(count),
            key=lambda i: (i == grow_priority, sizes[i] - mins[i]),
        )
        for idx in order:
            if overflow <= 0:
                break
            room = max(0, sizes[idx] - mins[idx])
            if room <= 0:
                continue
            take = min(room, overflow)
            sizes[idx] -= take
            overflow -= take

        if overflow > 0:
            # Fallback: enforce exact total even if it means shrinking below min (extreme small windows).
            for idx in sorted(range(count), key=lambda i: sizes[i], reverse=True):
                if overflow <= 0:
                    break
                take = min(max(0, sizes[idx] - 1), overflow)
                sizes[idx] -= take
                overflow -= take

        if sum(sizes) != total:
            delta = total - sum(sizes)
            sizes[grow_priority] = max(0, sizes[grow_priority] + delta)
        return sizes

    def _rebalance_splitters(self, force: bool = False):
        if not hasattr(self, "root_splitter") or not hasattr(self, "main_splitter"):
            return
        self._rebalance_root_splitter(force=force)
        self._rebalance_main_splitter(force=force)
        self._rebalance_side_splitters(force=force)

    @staticmethod
    def _apply_splitter_sizes(splitter: QSplitter, target_sizes: List[int]):
        current = splitter.sizes()
        if len(current) == len(target_sizes) and all(
            abs(int(current[i]) - int(target_sizes[i])) <= 1 for i in range(len(target_sizes))
        ):
            return
        splitter.setSizes(target_sizes)

    def _rebalance_root_splitter(self, force: bool = False):
        splitter = getattr(self, "root_splitter", None)
        if splitter is None or splitter.count() != 4:
            return
        current = splitter.sizes()
        if len(current) != 4:
            return
        total = sum(current)
        if total <= 0:
            return
        min_sizes = [
            max(getattr(self, "header_card").minimumHeight(), 78),
            max(getattr(self, "progress_card").minimumHeight(), 72),
            max(getattr(self, "stepper_card").minimumHeight(), 54),
            320,
        ]
        if not force and all(current[i] >= min_sizes[i] - 2 for i in range(4)):
            return
        base = [
            min_sizes[0],
            min_sizes[1],
            min_sizes[2],
            max(min_sizes[3], total - min_sizes[0] - min_sizes[1] - min_sizes[2]),
        ]
        self._apply_splitter_sizes(
            splitter,
            self._fit_splitter_sizes(base, min_sizes, total, grow_priority=3),
        )

    def _rebalance_main_splitter(self, force: bool = False):
        splitter = getattr(self, "main_splitter", None)
        if splitter is None or splitter.count() != 2:
            return
        current = splitter.sizes()
        if len(current) != 2:
            return
        total = sum(current)
        if total <= 0:
            return
        left_min = max(getattr(self, "left_panel_splitter").minimumWidth(), 520)
        right_visible = bool(getattr(self, "right_panel_splitter").isVisible())
        right_min = max(getattr(self, "right_panel_splitter").minimumWidth(), 380) if right_visible else 0
        min_sizes = [left_min, right_min]
        if not force and all(current[i] >= min_sizes[i] - 2 for i in range(2)):
            return
        left_pref = int(total * 0.56) if right_visible else total
        base = [left_pref, total - left_pref]
        self._apply_splitter_sizes(
            splitter,
            self._fit_splitter_sizes(base if force else current, min_sizes, total, grow_priority=0),
        )

    def _rebalance_side_splitters(self, force: bool = False):
        left_splitter = getattr(self, "left_panel_splitter", None)
        if left_splitter is not None and left_splitter.count() == 4:
            current = left_splitter.sizes()
            total = sum(current)
            if total > 0:
                control_min = max(getattr(self, "controls_card").minimumHeight(), 58)
                min_sizes = [
                    0 if not self.drop_area.isVisible() else 110,
                    0 if not self.queue_card.isVisible() else 120,
                    control_min,
                    0 if not self.config_card.isVisible() else max(getattr(self, "config_card").minimumHeight(), 220),
                ]
                need_fix = force or any(
                    len(current) != 4 or current[i] < min_sizes[i] - 2 for i in range(4)
                )
                if need_fix:
                    base = current
                    if force or len(current) != 4:
                        base = [
                            max(min_sizes[0], int(total * 0.16)),
                            max(min_sizes[1], int(total * 0.20)),
                            control_min,
                            max(min_sizes[3], int(total * 0.48)),
                        ]
                    self._apply_splitter_sizes(
                        left_splitter,
                        self._fit_splitter_sizes(base, min_sizes, total, grow_priority=3),
                    )

        right_splitter = getattr(self, "right_panel_splitter", None)
        if right_splitter is not None and right_splitter.count() in {2, 3}:
            current = right_splitter.sizes()
            total = sum(current)
            if total > 0:
                if right_splitter.count() == 3:
                    result_min = 0 if not self.results_card.isVisible() else max(self.results_card.minimumHeight(), 160)
                    event_min = 0 if not self.event_card.isVisible() else max(self.event_card.minimumHeight(), 130)
                    log_min = 0 if not self.log_card.isVisible() else max(self.log_card.minimumHeight(), 140)
                    min_sizes = [
                        result_min,
                        event_min,
                        log_min,
                    ]
                    if force or any(current[i] < min_sizes[i] - 2 for i in range(3)):
                        base = current
                        if force or len(current) != 3:
                            remain = max(0, total - min_sizes[2])
                            top = int(remain * 0.64)
                            mid = max(min_sizes[1], remain - top)
                            base = [top, mid, total - top - mid]
                        self._apply_splitter_sizes(
                            right_splitter,
                            self._fit_splitter_sizes(base, min_sizes, total, grow_priority=0),
                        )
                else:
                    min_sizes = [
                        max(getattr(self, "results_card").minimumHeight(), 160),
                        max(getattr(self, "log_card").minimumHeight(), 140),
                    ]
                    if force or any(current[i] < min_sizes[i] - 2 for i in range(2)):
                        base = current
                        if force or len(current) != 2:
                            top = int(total * 0.58)
                            base = [top, total - top]
                        self._apply_splitter_sizes(
                            right_splitter,
                            self._fit_splitter_sizes(base, min_sizes, total, grow_priority=0),
                        )

    def _try_acquire_action_lock(self, name: str) -> bool:
        key = str(name).strip()
        if not key:
            return True
        if key in self._ui_action_locks:
            return False
        self._ui_action_locks.add(key)
        self._refresh_action_button_states()
        return True

    def _release_action_lock(self, name: str):
        self._ui_action_locks.discard(str(name).strip())
        self._refresh_action_button_states()

    def _config_item_help_text(self, help_key: str, title: str = "") -> str:
        detail_text = str(self.CONFIG_ITEM_HELP.get(help_key, "") or "").strip()
        switch_title = ""
        switch_desc = ""
        for item_title, dotted_key, desc in self._active_switch_specs():
            if dotted_key == help_key:
                switch_title = item_title
                switch_desc = desc
                break

        display_name = title or switch_title or help_key
        if not detail_text and switch_title:
            detail_text = (
                f"{switch_title}\n\n"
                f"作用：{switch_desc}。\n\n"
                "开启后：该能力会进入处理流程，结果会包含对应增强输出。\n"
                "关闭后：流程会跳过该能力，速度和资源占用通常更可控。\n\n"
                "建议：仅在明确需要该能力时开启，批量任务前先用单文件验证效果。"
            )
        if not detail_text:
            detail_text = (
                f"{display_name}\n\n"
                f"配置路径：{help_key}\n\n"
                "该项暂未提供专门文案，请按当前任务目标进行单项调节并复测结果。"
            )

        return f"{display_name}\n配置项：{help_key}\n\n{detail_text}"

    def _config_item_help_summary(self, help_key: str, title: str = "") -> str:
        raw = str(self.CONFIG_ITEM_HELP.get(help_key, "") or "").strip()
        text = raw if raw else self._config_item_help_text(help_key, title)
        for line in text.splitlines():
            clean = line.strip().lstrip("-").strip()
            if not clean:
                continue
            if clean.startswith("配置项："):
                continue
            return clean[:64]
        return f"查看“{title or help_key}”详细说明"

    def _show_config_item_help(self, help_key: str, title: str):
        display_title = str(title or help_key)
        QMessageBox.information(
            self,
            f"配置说明 - {display_title}",
            self._config_item_help_text(help_key, display_title),
        )

    def _make_help_dot_button(self, help_key: str, title: str) -> QPushButton:
        btn = QPushButton("?")
        btn.setObjectName("HelpDotBtn")
        side = self._scaled_px(18, min_px=18, max_px=28)
        btn.setFixedSize(side, side)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(self._config_item_help_summary(help_key, title))
        btn.clicked.connect(
            lambda _checked=False, dotted=help_key, help_title=title: self._show_config_item_help(
                dotted, help_title
            )
        )
        return btn

    def _make_grid_help_label(self, text: str, help_key: str, title: str) -> QWidget:
        wrapper = QWidget()
        lay = QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        label = QLabel(text, wrapper)
        label.setObjectName("InputLabel")
        lay.addWidget(label, 0, Qt.AlignVCenter)
        lay.addWidget(self._make_help_dot_button(help_key, title), 0, Qt.AlignVCenter)
        lay.addStretch(1)
        # Keep Python refs to avoid PySide wrapper GC edge cases on some builds.
        self._grid_help_label_widgets.append(wrapper)
        return wrapper

    def _replace_grid_label_with_help(
        self,
        grid: QGridLayout,
        row: int,
        col: int,
        help_key: str,
        title: str,
    ):
        # Disabled: dynamic QLabel->wrapper replacement can crash PySide6/Qt
        # during later layout activation on some Windows builds (native exit,
        # no Python traceback). Keep original labels to preserve startup
        # stability; switch-row help buttons still work.
        return

    def _dialog_screen_geometry(self):
        screen = None
        handle = self.windowHandle()
        if handle is not None:
            try:
                screen = handle.screen()
            except Exception:
                screen = None
        if screen is None:
            app = QApplication.instance()
            try:
                screen = app.primaryScreen() if app is not None else None
            except Exception:
                screen = None
        if screen is None:
            return None
        try:
            return screen.availableGeometry().adjusted(12, 12, -12, -12)
        except Exception:
            return None

    def _position_dialog_near_main(self, dialog: Optional[QDialog]) -> None:
        if dialog is None:
            return
        area = self._dialog_screen_geometry()
        if area is None or not area.isValid():
            return
        width = min(max(320, dialog.width()), area.width())
        height = min(max(220, dialog.height()), area.height())
        if dialog.width() != width or dialog.height() != height:
            dialog.resize(width, height)

        frame = self.frameGeometry()
        x = frame.left() + max(0, (frame.width() - dialog.width()) // 2)
        y = frame.top() + max(0, (frame.height() - dialog.height()) // 2)
        x = max(area.left(), min(area.right() - dialog.width() + 1, x))
        y = max(area.top(), min(area.bottom() - dialog.height() + 1, y))
        dialog.move(int(x), int(y))

    def _keep_aux_dialogs_in_view(self) -> None:
        for dlg in (self.advanced_dialog, self.posterior_dialog, self.usage_help_dialog):
            if dlg is not None and dlg.isVisible():
                self._position_dialog_near_main(dlg)

    def _finish_dialog_fade(
        self,
        key: int,
        dialog: QDialog,
        anim: QPropertyAnimation,
        visible: bool,
    ) -> None:
        current = self._dialog_fade_anims.get(key)
        if current is not anim:
            return
        self._dialog_fade_anims.pop(key, None)
        if not visible:
            dialog.hide()
        try:
            dialog.setWindowOpacity(1.0)
        except Exception:
            pass

    def _animate_tool_dialog(self, dialog: Optional[QDialog], visible: bool) -> None:
        if dialog is None:
            return
        key = id(dialog)
        old = self._dialog_fade_anims.pop(key, None)
        if old is not None:
            old.stop()

        try:
            if visible:
                self._position_dialog_near_main(dialog)
                if not dialog.isVisible():
                    dialog.setWindowOpacity(0.0)
                    dialog.show()
                dialog.raise_()
                dialog.activateWindow()
                start = float(dialog.windowOpacity())
                if start >= 0.98:
                    dialog.setWindowOpacity(1.0)
                    return
                anim = QPropertyAnimation(dialog, b"windowOpacity", self)
                anim.setDuration(180)
                anim.setStartValue(max(0.0, min(1.0, start)))
                anim.setEndValue(1.0)
                anim.setEasingCurve(QEasingCurve.OutCubic)
                anim.finished.connect(
                    lambda k=key, d=dialog, a=anim: self._finish_dialog_fade(k, d, a, True)
                )
            else:
                if not dialog.isVisible():
                    return
                start = max(0.02, float(dialog.windowOpacity()))
                anim = QPropertyAnimation(dialog, b"windowOpacity", self)
                anim.setDuration(150)
                anim.setStartValue(max(0.0, min(1.0, start)))
                anim.setEndValue(0.0)
                anim.setEasingCurve(QEasingCurve.OutCubic)
                anim.finished.connect(
                    lambda k=key, d=dialog, a=anim: self._finish_dialog_fade(k, d, a, False)
                )
        except Exception:
            dialog.setVisible(bool(visible))
            return

        self._dialog_fade_anims[key] = anim
        anim.start()

    def _on_usage_help_toggled(self, checked: bool):
        expanded = bool(checked)
        if hasattr(self, "usage_help_button") and self.usage_help_button is not None:
            self.usage_help_button.setText("关闭使用说明" if expanded else "打开使用说明")
        if self.usage_help_dialog is None:
            return
        if expanded:
            self._animate_tool_dialog(self.usage_help_dialog, True)
        else:
            self._animate_tool_dialog(self.usage_help_dialog, False)
        self._sync_macos_menu_state()

    def _sync_usage_help_toggle(self, checked: bool):
        if not hasattr(self, "usage_help_button") or self.usage_help_button is None:
            return
        self.usage_help_button.blockSignals(True)
        try:
            self.usage_help_button.setChecked(bool(checked))
            self.usage_help_button.setText("关闭使用说明" if checked else "打开使用说明")
        finally:
            self.usage_help_button.blockSignals(False)
        self._sync_macos_menu_state()

    def _is_ui_action_busy(self) -> bool:
        return bool(self.running or self._ui_action_locks)

    def _refresh_action_button_states(self):
        busy = self._is_ui_action_busy()

        for attr in (
            "add_button",
            "start_button",
            "clear_button",
            "remove_checked_button",
            "full_cleanup_button",
            "save_config_button",
            "advanced_toggle_button",
            "posterior_assistant_toggle_button",
            "usage_help_button",
            "dash_eye_btn",
            "hf_eye_btn",
            "reload_results_button",
            "clear_log_file_button",
            "toggle_all_switches_button",
            "prev_step_button",
            "next_step_button",
            "posterior_help_button",
            "posterior_examples_dir_button",
            "posterior_rttm_dir_button",
            "posterior_output_path_button",
            "posterior_dump_toggle_button",
            "posterior_auto_rerun_toggle_button",
            "posterior_train_button",
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    widget.setEnabled(not busy)
                except Exception:
                    pass

        for attr in (
            "posterior_examples_dir_input",
            "posterior_rttm_dir_input",
            "posterior_output_path_input",
            "posterior_epochs_input",
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    widget.setEnabled(not busy)
                except Exception:
                    pass

        if hasattr(self, "clear_temp_button") and self.clear_temp_button is not None:
            allow_when_paused = bool(self.running and self.paused and not self._ui_action_locks)
            self.clear_temp_button.setEnabled((not busy) or allow_when_paused)

        if hasattr(self, "pause_button") and self.pause_button is not None:
            self.pause_button.setEnabled(bool(self.running) and not self._pause_toggle_cooldown)
        if hasattr(self, "toggle_debug_log_button") and self.toggle_debug_log_button is not None:
            self.toggle_debug_log_button.setEnabled(self._current_step == 3)
        if hasattr(self, "start_button") and self.start_button is not None:
            can_start = (not busy) and bool(self.selected_files)
            self.start_button.setEnabled(can_start)
        self._refresh_stepper_buttons()

        self._update_result_action_state()
        self._sync_macos_menu_state()

    @staticmethod
    def _section_label(text: str, extra_class: str = "") -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionLabel" if not extra_class else extra_class)
        return label

    @staticmethod
    def _path_hint(path: Path, depth: int = 2) -> str:
        parts = list(path.parts)
        if not parts:
            return str(path)
        tail = parts[-depth:] if len(parts) > depth else parts
        return "/".join(tail)

    def _resolve_output_dir(self) -> Path:
        configured = str(_get_nested(self.config._data, "paths.output_dir", "output_files") or "output_files")
        p = resolve_app_writable_path(configured)
        return p.resolve()

    def _queue_display_name(self, file_path: str) -> str:
        p = Path(file_path)
        file_name = p.name
        counts = Counter(Path(fp).name.lower() for fp in self.selected_files)
        if counts.get(file_name.lower(), 0) <= 1:
            return file_name
        return f"{file_name} ({self._path_hint(p.parent)})"

    def _queue_item_text(self, file_path: str, status: str = "") -> str:
        title = self._queue_display_name(file_path)
        if status:
            title = f"{title}  [ {status} ]"
        return f"{title}\n{file_path}"

    def _update_result_action_state(self):
        has_selection = bool(self._selected_result_items()) if hasattr(self, "result_tree") else False
        busy = self._is_ui_action_busy()
        if hasattr(self, "read_result_button"):
            self.read_result_button.setEnabled((not busy) and has_selection)
        if hasattr(self, "delete_result_button"):
            self.delete_result_button.setEnabled((not busy) and has_selection)
        self._sync_macos_menu_state()

    def _selected_result_items(self) -> List[QTreeWidgetItem]:
        if not hasattr(self, "result_tree"):
            return []
        selected = list(self.result_tree.selectedItems())
        if not selected:
            current = self.result_tree.currentItem()
            return [current] if current is not None else []

        selected_ids = {id(item) for item in selected}
        filtered: List[QTreeWidgetItem] = []
        for item in selected:
            parent = item.parent()
            skip = False
            while parent is not None:
                if id(parent) in selected_ids:
                    skip = True
                    break
                parent = parent.parent()
            if not skip:
                filtered.append(item)
        return filtered

    def _runtime_artifacts_dir(self) -> Path:
        return resolve_runtime_artifact_path(self._resolve_output_dir())

    def _default_posterior_examples_relpath(self) -> str:
        return str(resolve_runtime_artifact_path(self._resolve_output_dir(), "posterior_fusion_examples"))

    def _default_posterior_rttm_relpath(self) -> str:
        return str(resolve_runtime_artifact_path(self._resolve_output_dir(), "posterior_fusion_rttm"))

    def _default_posterior_output_relpath(self) -> str:
        return str(resolve_runtime_artifact_path(self._resolve_output_dir(), "posterior_fusion_calibrator.json"))

    @classmethod
    def _is_runtime_artifact_path(cls, path: Path, output_root: Path) -> bool:
        try:
            rel = path.resolve().relative_to(output_root.resolve())
            parts = rel.parts
            if not parts:
                return False
            head = str(parts[0])
            return head.startswith(".") or head == RUNTIME_ARTIFACTS_DIRNAME
        except Exception:
            return False

    @classmethod
    def _is_final_result_file(
        cls,
        path: Path,
        *,
        output_root: Path,
        summary_path: Path,
    ) -> bool:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        try:
            if resolved == summary_path.resolve():
                return True
        except Exception:
            pass
        if not path.is_file():
            return False
        if cls._is_runtime_artifact_path(path, output_root):
            return False
        return path.suffix.lower() in cls.FINAL_OUTPUT_SUFFIXES

    @staticmethod
    def _result_item_path(item: QTreeWidgetItem) -> str:
        path = item.data(0, Qt.UserRole)
        if not path:
            path = item.text(2).strip()
        return str(path or "")

    def _remove_result_tree_item(self, item: QTreeWidgetItem):
        parent = item.parent()
        if parent is None:
            idx = self.result_tree.indexOfTopLevelItem(item)
            if idx >= 0:
                self.result_tree.takeTopLevelItem(idx)
            return

        parent.removeChild(item)
        if parent.childCount() == 0:
            idx = self.result_tree.indexOfTopLevelItem(parent)
            if idx >= 0:
                self.result_tree.takeTopLevelItem(idx)

    @Slot(bool)
    def reload_results_from_disk(self, silent: bool = False):
        if not self._try_acquire_action_lock("reload_results"):
            return
        try:
            output_root = self._resolve_output_dir()
            summary_name = str(
                _get_nested(self.config._data, "paths.summary_file", "all_results_summary.txt")
                or "all_results_summary.txt"
            )
            summary_path = output_root / summary_name

            self.result_tree.clear()
            if not output_root.exists():
                self._update_result_action_state()
                if not silent:
                    self.set_status_chip("产物目录不存在", "info")
                return

            groups: Dict[str, List[Path]] = {}
            total_files = 0
            summary_found = False

            for file_path in sorted(output_root.rglob("*")):
                if not file_path.is_file():
                    continue
                if not self._is_final_result_file(
                    file_path,
                    output_root=output_root,
                    summary_path=summary_path,
                ):
                    continue
                if file_path.resolve() == summary_path.resolve():
                    summary_found = True
                    continue
                parent_key = str(file_path.parent.resolve())
                groups.setdefault(parent_key, []).append(file_path)
                total_files += 1

            for parent_key in sorted(groups):
                parent_path = Path(parent_key)
                try:
                    rel_parent = parent_path.relative_to(output_root)
                    group_label = str(rel_parent).replace("\\", "/")
                except ValueError:
                    group_label = str(parent_path)
                top = QTreeWidgetItem([group_label, "目录", str(parent_path)])
                top.setData(0, Qt.UserRole, str(parent_path))
                for child_path in sorted(groups[parent_key], key=lambda p: p.name.lower()):
                    suffix = child_path.suffix.lower()
                    kind = self.OUTPUT_KIND_BY_SUFFIX.get(suffix, suffix.upper().lstrip(".") or "FILE")
                    child = QTreeWidgetItem([child_path.name, kind, str(child_path)])
                    child.setData(0, Qt.UserRole, str(child_path))
                    top.addChild(child)
                self.result_tree.addTopLevelItem(top)

            if summary_found and summary_path.exists():
                summary_item = QTreeWidgetItem([summary_path.name, "汇总", str(summary_path)])
                summary_item.setData(0, Qt.UserRole, str(summary_path))
                self.result_tree.addTopLevelItem(summary_item)

            self._update_result_action_state()
            if not silent:
                if total_files == 0 and not summary_found:
                    self.set_status_chip("未发现可读取的产物文件", "info")
                else:
                    self.set_status_chip(f"已读取产物: {total_files} 个文件", "ok")
        finally:
            self._release_action_lock("reload_results")

    @Slot(bool)
    def read_selected_result_file(self, _checked: bool = False):
        if not self._try_acquire_action_lock("read_result"):
            return
        try:
            items = self._selected_result_items()
            if not items:
                self.set_status_chip("请先选择产物项", "info")
                return
            opened_count = 0
            read_count = 0
            for item in items:
                path_str = self._result_item_path(item)
                if not path_str:
                    continue
                path = Path(path_str)
                if not path.exists():
                    self._remove_result_tree_item(item)
                    continue

                if path.is_dir():
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
                    opened_count += 1
                    continue

                suffix = path.suffix.lower()
                if suffix not in self.READABLE_OUTPUT_SUFFIXES:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
                    opened_count += 1
                    continue

                try:
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    QMessageBox.warning(self, "读取失败", f"无法读取文件：\n{path}\n\n{e}")
                    return

                max_chars = 120000
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n\n...(内容过长，已截断)"

                self.append_log(f"[读取产物] {path}")
                self.append_log("-" * 60)
                self.append_log(content)
                self.append_log("-" * 60)
                read_count += 1

            if read_count > 0 and opened_count > 0:
                self.set_status_chip(f"已读取 {read_count} 个文件，并打开 {opened_count} 个目标", "ok")
            elif read_count > 0:
                self.set_status_chip(f"已读取 {read_count} 个文件", "ok")
            elif opened_count > 0:
                self.set_status_chip(f"已打开 {opened_count} 个目标", "info")
            else:
                self.set_status_chip("未找到可读取的产物项", "warn")
        finally:
            self._release_action_lock("read_result")

    @Slot()
    def delete_selected_result_item(self):
        if not self._try_acquire_action_lock("delete_result"):
            return
        try:
            items = self._selected_result_items()
            if not items:
                self.set_status_chip("请先选择要删除的产物", "info")
                return
            removed_count = 0
            for item in items:
                path_str = self._result_item_path(item)
                if not path_str:
                    continue
                target = Path(path_str)
                try:
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    self._remove_result_tree_item(item)
                    removed_count += 1
                except Exception as e:
                    QMessageBox.warning(self, "删除失败", f"无法删除目标：\n{target}\n\n{e}")
                    return
            self._update_result_action_state()
            self.set_status_chip(f"已删除 {removed_count} 个产物项", "ok")
        finally:
            self._release_action_lock("delete_result")

    def _resolve_language_choice(self) -> str:
        auto_detect = bool(_get_nested(self.config._data, "language.auto_detect", True))
        if auto_detect:
            return "auto"

        fw_lang = str(_get_nested(self.config._data, "asr.faster_whisper.language", "") or "")
        fallback = str(_get_nested(self.config._data, "language.fallback_language", "zh") or "")
        current = (fw_lang or fallback).strip().lower()
        if current.startswith("en"):
            return "en"
        return "zh"

    def _resolve_translation_target_choice(self) -> str:
        target = str(
            _get_nested(self.config._data, "translation.target_language", "zh") or "zh"
        ).strip().lower().replace("_", "-")
        if not target:
            target = "zh"
        return target

    def _language_payload_from_editor(self) -> Dict[str, Any]:
        selected = str(self.language_combo.currentData() or "auto").strip().lower()
        fallback = str(
            _get_nested(self.config._data, "language.fallback_language", "zh") or ""
        ).strip().lower()
        if not fallback or fallback == "auto":
            fallback = "zh"
        if selected == "auto":
            return {
                "auto_detect": True,
                "fallback_language": fallback,
                "primary_languages": [],
                "fw_language": None,
            }

        language_code = "en" if selected.startswith("en") else "zh"
        return {
            "auto_detect": False,
            "fallback_language": language_code,
            "primary_languages": [language_code],
            "fw_language": language_code,
        }

    @staticmethod
    def _write_language_payload(target_cfg: Dict[str, Any], payload: Dict[str, Any]):
        _set_nested(target_cfg, "language.auto_detect", bool(payload["auto_detect"]))
        _set_nested(target_cfg, "language.fallback_language", payload["fallback_language"])
        _set_nested(target_cfg, "language.primary_languages", list(payload["primary_languages"]))
        _set_nested(target_cfg, "asr.faster_whisper.language", payload["fw_language"])

    @Slot(int)
    def _on_language_changed(self, _index: int):
        self._language_dirty = True
        self.set_status_chip("配置已修改", "info")

    @Slot(int)
    def _on_translation_target_changed(self, _index: int):
        self._translation_dirty = True
        self.set_status_chip("配置已修改", "info")

    @Slot(str)
    def _on_dashscope_key_edited(self, _text: str):
        self._dashscope_dirty = True

    @Slot(str)
    def _on_hf_token_edited(self, _text: str):
        self._hf_token_dirty = True

    def _on_switch_toggled(self, dotted_key: str, checked: bool):
        _set_nested(self.config._data, dotted_key, bool(checked))
        if dotted_key == "asr.nemo_msdd.enabled":
            self.nemo_speaker_mode_combo.setEnabled(bool(checked))
            manual = str(self.nemo_speaker_mode_combo.currentData() or "auto") == "manual"
            self.nemo_fixed_speakers_input.setEnabled(bool(checked) and manual)
        self._switch_dirty = True
        self._sync_global_switch_button_state()
        self.set_status_chip("配置已修改", "info")

    @Slot(int)
    def _on_nemo_speaker_mode_changed(self, _index: int):
        manual = str(self.nemo_speaker_mode_combo.currentData() or "auto") == "manual"
        nemo_enabled = bool(_get_nested(self.config._data, "asr.nemo_msdd.enabled", True))
        self.nemo_fixed_speakers_input.setEnabled(nemo_enabled and manual)
        self._on_advanced_setting_edited()

    @Slot(bool)
    def _on_advanced_panel_toggled(self, checked: bool):
        expanded = bool(checked)
        self.advanced_toggle_button.setText(
            "关闭高级配置弹窗" if expanded else "打开高级配置弹窗"
        )
        if self.advanced_dialog is None:
            return
        if expanded:
            self._animate_tool_dialog(self.advanced_dialog, True)
        else:
            self._animate_tool_dialog(self.advanced_dialog, False)
        self._sync_macos_menu_state()

    def _sync_advanced_toggle(self, checked: bool):
        self.advanced_toggle_button.blockSignals(True)
        try:
            self.advanced_toggle_button.setChecked(bool(checked))
            self.advanced_toggle_button.setText(
                "关闭高级配置弹窗" if checked else "打开高级配置弹窗"
            )
        finally:
            self.advanced_toggle_button.blockSignals(False)
        self._sync_macos_menu_state()

    @Slot(bool)
    def _on_posterior_assistant_toggled(self, checked: bool):
        expanded = bool(checked)
        if hasattr(self, "posterior_assistant_toggle_button") and self.posterior_assistant_toggle_button is not None:
            self.posterior_assistant_toggle_button.setText(
                "关闭 Posterior Fusion 助手" if expanded else "打开 Posterior Fusion 助手"
            )
        if self.posterior_dialog is None:
            return
        if expanded:
            self._animate_tool_dialog(self.posterior_dialog, True)
        else:
            self._animate_tool_dialog(self.posterior_dialog, False)
        self._sync_macos_menu_state()

    def _sync_posterior_assistant_toggle(self, checked: bool):
        if not hasattr(self, "posterior_assistant_toggle_button") or self.posterior_assistant_toggle_button is None:
            return
        self.posterior_assistant_toggle_button.blockSignals(True)
        try:
            self.posterior_assistant_toggle_button.setChecked(bool(checked))
            self.posterior_assistant_toggle_button.setText(
                "关闭 Posterior Fusion 助手" if checked else "打开 Posterior Fusion 助手"
            )
        finally:
            self.posterior_assistant_toggle_button.blockSignals(False)
        self._sync_macos_menu_state()

    def _posterior_auto_run_twice_enabled(self) -> bool:
        button = getattr(self, "posterior_auto_rerun_toggle_button", None)
        return bool(button is not None and button.isChecked())

    @staticmethod
    def _parse_int_value(
        text: str,
        default: int,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        try:
            value = int(float(str(text).strip()))
        except (TypeError, ValueError):
            value = int(default)
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value

    @staticmethod
    def _parse_float_value(
        text: str,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            value = float(default)
        if min_value is not None:
            value = max(float(min_value), value)
        if max_value is not None:
            value = min(float(max_value), value)
        return value

    def _posterior_fusion_examples_dir(self) -> str:
        raw = (
            self.posterior_examples_dir_input.text().strip()
            or str(
                _get_nested(
                    self.config._data,
                    "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.examples_dir",
                    self._default_posterior_examples_relpath(),
                )
                or self._default_posterior_examples_relpath()
            )
        )
        normalized = raw.replace("\\", "/")
        if normalized in {
            "",
            "output_files/posterior_fusion_examples",
            f"{RUNTIME_ARTIFACTS_DIRNAME}/posterior_fusion_examples",
        }:
            return self._default_posterior_examples_relpath()
        return raw

    def _posterior_fusion_rttm_dir(self) -> str:
        raw = (
            self.posterior_rttm_dir_input.text().strip()
            or str(
                _get_nested(
                    self.config._data,
                    "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.rttm_dir",
                    self._default_posterior_rttm_relpath(),
                )
                or self._default_posterior_rttm_relpath()
            )
        )
        normalized = raw.replace("\\", "/")
        if normalized in {
            "",
            "output_files/posterior_fusion_rttm",
            f"{RUNTIME_ARTIFACTS_DIRNAME}/posterior_fusion_rttm",
        }:
            return self._default_posterior_rttm_relpath()
        return raw

    def _posterior_fusion_output_path(self) -> str:
        raw = (
            self.posterior_output_path_input.text().strip()
            or str(
                _get_nested(
                    self.config._data,
                    "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.output_path",
                    self._default_posterior_output_relpath(),
                )
                or self._default_posterior_output_relpath()
            )
        )
        normalized = raw.replace("\\", "/")
        if normalized in {
            "",
            "output_files/.posterior_fusion_calibrator.json",
            ".posterior_fusion_calibrator.json",
            f"{RUNTIME_ARTIFACTS_DIRNAME}/posterior_fusion_calibrator.json",
        }:
            return self._default_posterior_output_relpath()
        return raw

    def _posterior_fusion_epochs(self) -> int:
        default_epochs = int(
            _get_nested(
                self.config._data,
                "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.epochs",
                6,
            )
            or 6
        )
        return self._parse_int_value(
            self.posterior_epochs_input.text(),
            default=max(1, default_epochs),
            min_value=1,
            max_value=200,
        )

    def _sync_posterior_dump_button_text(self) -> None:
        if not hasattr(self, "posterior_dump_toggle_button") or self.posterior_dump_toggle_button is None:
            return
        checked = bool(self.posterior_dump_toggle_button.isChecked())
        self.posterior_dump_toggle_button.setText("关闭样本导出" if checked else "开启样本导出")
        self.posterior_dump_toggle_button.setToolTip(
            "开启后，下一次正常处理会把 posterior fusion 的开发集样本自动导出到样本目录。"
            if checked
            else "建议先开启一次，跑完开发集后再回来训练校准器。"
        )

    def _sync_posterior_auto_rerun_button_text(self) -> None:
        if (
            not hasattr(self, "posterior_auto_rerun_toggle_button")
            or self.posterior_auto_rerun_toggle_button is None
        ):
            return
        checked = bool(self.posterior_auto_rerun_toggle_button.isChecked())
        self.posterior_auto_rerun_toggle_button.setText(
            "关闭自动跑两次" if checked else "开启自动跑两次"
        )
        self.posterior_auto_rerun_toggle_button.setToolTip(
            "第一遍先导出样本并自动训练，训练完成后会重跑同一批文件。"
            if checked
            else "只跑一遍；同时关闭 UI 触发的样本导出逻辑。"
        )

    def _show_posterior_fusion_help(self) -> None:
        QMessageBox.information(self, "Posterior Fusion 训练说明", self.POSTERIOR_FUSION_HELP_TEXT)

    def _browse_posterior_examples_dir(self) -> None:
        current = str(Path(self._posterior_fusion_examples_dir()).expanduser())
        selected = QFileDialog.getExistingDirectory(self, "选择 posterior fusion 样本目录", current)
        if not selected:
            return
        self.posterior_examples_dir_input.setText(selected)
        self._on_advanced_setting_edited()

    def _browse_posterior_rttm_dir(self) -> None:
        current = str(Path(self._posterior_fusion_rttm_dir()).expanduser())
        selected = QFileDialog.getExistingDirectory(self, "选择 RTTM 目录", current)
        if not selected:
            return
        self.posterior_rttm_dir_input.setText(selected)
        self._on_advanced_setting_edited()

    def _browse_posterior_output_path(self) -> None:
        current = str(Path(self._posterior_fusion_output_path()).expanduser())
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "选择校准器输出路径",
            current,
            "JSON Files (*.json);;All Files (*)",
        )
        if not selected:
            return
        self.posterior_output_path_input.setText(selected)
        self._on_advanced_setting_edited()

    def _on_posterior_dump_toggled(self, _checked: bool) -> None:
        self._sync_posterior_dump_button_text()
        if self.posterior_dump_toggle_button.isChecked():
            if not self._posterior_auto_run_twice_enabled():
                self.posterior_auto_rerun_toggle_button.blockSignals(True)
                try:
                    self.posterior_auto_rerun_toggle_button.setChecked(True)
                finally:
                    self.posterior_auto_rerun_toggle_button.blockSignals(False)
                self._sync_posterior_auto_rerun_button_text()
        elif self._posterior_auto_run_twice_enabled():
            self.posterior_auto_rerun_toggle_button.blockSignals(True)
            try:
                self.posterior_auto_rerun_toggle_button.setChecked(False)
            finally:
                self.posterior_auto_rerun_toggle_button.blockSignals(False)
            self._sync_posterior_auto_rerun_button_text()
        self._on_advanced_setting_edited()
        if self.posterior_dump_toggle_button.isChecked():
            state = "已开启样本导出，并同步开启自动跑两次"
        else:
            state = "已关闭样本导出，并同步关闭自动跑两次"
        self.set_status_chip(state, "info")

    def _on_posterior_auto_rerun_toggled(self, _checked: bool) -> None:
        if (
            not self._posterior_auto_run_twice_enabled()
            and self.posterior_dump_toggle_button.isChecked()
        ):
            self.posterior_dump_toggle_button.blockSignals(True)
            try:
                self.posterior_dump_toggle_button.setChecked(False)
            finally:
                self.posterior_dump_toggle_button.blockSignals(False)
            self._sync_posterior_dump_button_text()
        self._sync_posterior_auto_rerun_button_text()
        self._on_advanced_setting_edited()
        state = (
            "已开启自动跑两次：第一遍导样本，训练后自动第二遍"
            if self._posterior_auto_run_twice_enabled()
            else "已关闭自动跑两次，并严格回退为单次处理"
        )
        self.set_status_chip(state, "info")

    def _shutdown_posterior_trainer(self, timeout_ms: int = 3000) -> bool:
        thread = self.posterior_trainer_thread
        if thread is not None and thread.isRunning():
            timeout_ms = max(500, int(timeout_ms))
            deadline = time.time() + (timeout_ms / 1000.0)
            while thread.isRunning() and time.time() < deadline:
                thread.wait(120)
        if thread is not None and thread.isRunning():
            return False
        if thread is not None:
            thread.deleteLater()
            self.posterior_trainer_thread = None
        if self.posterior_trainer_worker is not None:
            self.posterior_trainer_worker.deleteLater()
            self.posterior_trainer_worker = None
        return True

    def _abort_posterior_auto_cycle(self, reason: str, level: str = "warn") -> None:
        if not self._posterior_auto_cycle_phase:
            return
        self._posterior_auto_cycle_phase = ""
        self._posterior_auto_cycle_first_pass_calibrator_path = ""
        text = f"Posterior Fusion 自动双跑已中止：{reason}"
        self.append_log(text)
        self.set_status_chip(text, level)
        self._push_event(text, level, ttl_ms=22000)

    def _posterior_auto_cycle_first_pass_output_path(self) -> str:
        cached = str(self._posterior_auto_cycle_first_pass_calibrator_path or "").strip()
        if cached:
            return cached
        base_name = Path(self._posterior_fusion_output_path() or "posterior_fusion_calibrator.json").name
        stem = Path(base_name).stem or "posterior_fusion_calibrator"
        suffix = Path(base_name).suffix or ".json"
        temp_path = Path(tempfile.gettempdir()) / f"{stem}.first_pass_{int(time.time() * 1000)}{suffix}"
        self._posterior_auto_cycle_first_pass_calibrator_path = str(temp_path)
        return self._posterior_auto_cycle_first_pass_calibrator_path

    def _posterior_auto_cycle_run_options(self) -> Dict[str, Any]:
        phase = str(self._posterior_auto_cycle_phase or "").strip()
        if phase == "first_pass":
            return {
                "disable_llm_processing": True,
                "disable_resume": True,
                "config_overrides": {
                    "asr.nemo_msdd.hybrid_fusion.posterior_decoder.calibrator.persist_path": self._posterior_auto_cycle_first_pass_output_path(),
                    "asr.nemo_msdd.hybrid_fusion.posterior_decoder.calibrator.enabled": False,
                },
            }
        if phase == "second_pass":
            return {
                "disable_resume": True,
            }
        return {}

    def _clear_posterior_auto_cycle_first_pass_artifacts(self) -> None:
        if not self.selected_files:
            return
        output_root = Path(
            str(_get_nested(self.config._data, "paths.output_dir", "output_files") or "output_files")
        ).expanduser()
        if not output_root.is_absolute():
            output_root = resolve_app_writable_path(output_root)
        input_root = Path(self._common_input_root(self.selected_files))

        removed_dirs = 0
        for raw_file in self.selected_files:
            source_file = Path(raw_file)
            target_dir = resolve_output_subdir(
                output_root=output_root,
                source_file=source_file,
                input_dir=input_root,
            )
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
                removed_dirs += 1

        summary_name = str(
            _get_nested(self.config._data, "paths.summary_file", "all_results_summary.txt")
            or "all_results_summary.txt"
        )
        summary_path = output_root / summary_name
        try:
            if summary_path.exists():
                summary_path.unlink()
        except Exception:
            pass

        resume_dir = resolve_runtime_artifact_path(output_root, "resume")
        if resume_dir.exists():
            shutil.rmtree(resume_dir, ignore_errors=True)

        self.result_tree.clear()
        self.append_log(
            f"Posterior Fusion 自动双跑：已清除第一遍输出与 resume 检查点（目录 {removed_dirs} 个）。"
        )

    def start_posterior_fusion_training(self, automated: bool = False) -> None:
        if not self._try_acquire_action_lock("posterior_train"):
            return
        started = False
        try:
            self._apply_editor_settings_to_config()
            try:
                self._persist_modified_config_fields()
            except Exception as e:
                self.append_log(f"警告：Posterior Fusion 训练前保存配置失败，继续使用当前内存配置: {e}")

            examples_dir = Path(self._posterior_fusion_examples_dir()).expanduser()
            rttm_dir = Path(self._posterior_fusion_rttm_dir()).expanduser()
            output_path = Path(self._posterior_fusion_output_path()).expanduser()
            if not examples_dir.is_absolute():
                examples_dir = resolve_app_writable_path(examples_dir)
            if not rttm_dir.is_absolute():
                rttm_dir = resolve_app_writable_path(rttm_dir)
            if not output_path.is_absolute():
                output_path = resolve_app_writable_path(output_path)
            epochs = self._posterior_fusion_epochs()

            if not examples_dir.exists():
                QMessageBox.warning(
                    self,
                    "样本目录不存在",
                    "找不到 posterior fusion 样本目录。\n\n请先点击“开启样本导出”，再跑一遍开发集。",
                )
                if automated:
                    self._abort_posterior_auto_cycle("未找到样本目录")
                return
            if not any(examples_dir.glob("*.json")):
                QMessageBox.warning(
                    self,
                    "样本目录为空",
                    "当前样本目录里没有 `.json` 样本。\n\n请先开启样本导出并正常处理一遍开发集。",
                )
                if automated:
                    self._abort_posterior_auto_cycle("样本目录为空")
                return
            if not rttm_dir.exists():
                QMessageBox.warning(
                    self,
                    "RTTM 目录不存在",
                    "找不到 RTTM 目录。\n\n请把人工标注的 `.rttm` 文件放到同一个目录后再试。",
                )
                if automated:
                    self._abort_posterior_auto_cycle("RTTM 目录不存在")
                return

            self.append_log(
                "Posterior Fusion 训练启动: "
                f"examples={examples_dir}, rttm={rttm_dir}, output={output_path}, epochs={epochs}"
            )
            if automated:
                self.set_status_chip("Posterior Fusion 自动双跑：校准训练中...", "warn")
                self._push_event("Posterior Fusion 自动双跑：校准训练已启动", "warn", ttl_ms=20000)
            else:
                self.set_status_chip("Posterior Fusion 校准训练中...", "warn")
                self._push_event("Posterior Fusion 校准训练已启动", "warn", ttl_ms=20000)

            config_data = copy.deepcopy(self.config._data)
            self.posterior_trainer_thread = QThread(self)
            self.posterior_trainer_worker = PosteriorFusionTrainerWorker(
                config_data=config_data,
                examples_dir=str(examples_dir),
                rttm_dir=str(rttm_dir),
                output_path=str(output_path),
                epochs=epochs,
            )
            self.posterior_trainer_worker.moveToThread(self.posterior_trainer_thread)
            self.posterior_trainer_thread.started.connect(self.posterior_trainer_worker.run)
            self.posterior_trainer_worker.log_message.connect(self.append_log, Qt.QueuedConnection)
            self.posterior_trainer_worker.status_message.connect(self._on_worker_status_message)
            self.posterior_trainer_worker.finished.connect(self.posterior_trainer_thread.quit)
            self.posterior_trainer_worker.failed.connect(self.posterior_trainer_thread.quit)
            self.posterior_trainer_worker.finished.connect(self.on_posterior_fusion_training_finished)
            self.posterior_trainer_worker.failed.connect(self.on_posterior_fusion_training_failed)
            self.posterior_trainer_thread.start()
            started = True
        except Exception as e:
            QMessageBox.critical(self, "无法启动训练", str(e))
            if automated:
                self._abort_posterior_auto_cycle("训练线程启动失败", level="err")
        finally:
            if not started:
                self._release_action_lock("posterior_train")

    @Slot(dict)
    def on_posterior_fusion_training_finished(self, metrics: Dict[str, Any]) -> None:
        auto_cycle_training = self._posterior_auto_cycle_phase == "training"
        output_path = str(metrics.get("output_path", "") or "").strip()
        if output_path:
            self.posterior_output_path_input.setText(output_path)
        self._apply_editor_settings_to_config()
        _set_nested(
            self.config._data,
            "asr.nemo_msdd.hybrid_fusion.posterior_decoder.calibrator.persist_path",
            output_path or self._posterior_fusion_output_path(),
        )
        self._advanced_dirty = True
        try:
            self._persist_modified_config_fields()
        except Exception as e:
            self.append_log(f"警告：训练完成后保存配置失败: {e}")
        summary = (
            "Posterior Fusion 训练完成。\n\n"
            f"样本数: {int(metrics.get('examples', 0) or 0)}\n"
            f"额外自适应样本: {int(metrics.get('unlabeled_examples', 0) or 0)}\n"
            f"阈值: {float(metrics.get('activity_threshold', 0.0) or 0.0):.3f}\n"
            f"主标签准确率: {float(metrics.get('primary_accuracy', 0.0) or 0.0):.3f}\n"
            f"活动 F1: {float(metrics.get('activity_f1', 0.0) or 0.0):.3f}\n"
            f"Overlap F1: {float(metrics.get('overlap_f1', 0.0) or 0.0):.3f}\n"
            f"综合目标: {float(metrics.get('objective_score', 0.0) or 0.0):.3f}\n"
            f"Consensus 更新: {int(((metrics.get('pseudo_adaptation', {}) or {}).get('updates', 0) or 0))}\n"
            f"输出: {output_path or '未返回'}"
        )
        if auto_cycle_training:
            self.append_log(summary)
            self.set_status_chip("Posterior Fusion 自动双跑：校准完成，准备第二遍处理...", "ok")
            self._push_event("Posterior Fusion 自动双跑：校准完成，准备第二遍处理", "ok", ttl_ms=22000)
        else:
            self.set_status_chip("Posterior Fusion 校准器训练完成", "ok")
            self._push_event("Posterior Fusion 校准器训练完成", "ok", ttl_ms=22000)
            QMessageBox.information(self, "训练完成", summary)
        self._shutdown_posterior_trainer()
        self._release_action_lock("posterior_train")
        if auto_cycle_training:
            self._posterior_auto_cycle_phase = "second_pass"
            QTimer.singleShot(180, self.start_processing)

    @Slot(str)
    def on_posterior_fusion_training_failed(self, trace_text: str) -> None:
        auto_cycle_training = self._posterior_auto_cycle_phase == "training"
        self.append_log(trace_text)
        friendly = self._friendly_error_text(trace_text)
        if auto_cycle_training:
            self.set_status_chip(f"Posterior Fusion 自动双跑训练失败：{friendly}", "err")
            self._push_event("Posterior Fusion 自动双跑训练失败", "err", ttl_ms=22000)
        else:
            self.set_status_chip(f"Posterior Fusion 训练失败：{friendly}", "err")
            self._push_event("Posterior Fusion 训练失败", "err", ttl_ms=22000)
        QMessageBox.critical(
            self,
            "训练失败",
            "Posterior Fusion 校准器训练失败。\n\n"
            f"{friendly}\n\n"
            "常见排查：样本目录是否真的有 `.json/.npz`，RTTM 文件名是否和音频前缀匹配。",
        )
        self._shutdown_posterior_trainer()
        self._release_action_lock("posterior_train")
        if auto_cycle_training:
            self._abort_posterior_auto_cycle("校准训练失败", level="err")

    def _advanced_payload_from_editor(self) -> Dict[str, Any]:
        overlap_default = float(_get_nested(self.config._data, "audio.overlap_sec", 1.0) or 1.0)
        nemo_num_default = int(_get_nested(self.config._data, "asr.nemo_msdd.num_speakers", 0) or 0)
        if nemo_num_default <= 0:
            nemo_num_default = 2
        nemo_min_default = int(_get_nested(self.config._data, "asr.nemo_msdd.min_speakers", 1) or 1)
        nemo_max_default = int(_get_nested(self.config._data, "asr.nemo_msdd.max_speakers", 8) or 8)
        max_concurrent_default = int(_get_nested(self.config._data, "performance.max_concurrent_files", 1) or 1)
        gpu_fraction_default = float(_get_nested(self.config._data, "performance.gpu_memory_fraction", 0.98) or 0.98)
        font_size_default = int(_get_nested(self.config._data, "video_text_overlay.font_size", 22) or 22)
        max_chars_default = int(_get_nested(self.config._data, "video_text_overlay.max_line_chars", 18) or 18)
        margin_default = int(_get_nested(self.config._data, "video_text_overlay.bottom_margin_px", 56) or 56)
        crf_default = int(_get_nested(self.config._data, "video_text_overlay.ffmpeg_crf", 20) or 20)
        speaker_mode = str(self.nemo_speaker_mode_combo.currentData() or "auto")
        fixed_num_speakers = self._parse_int_value(
            self.nemo_fixed_speakers_input.text(),
            default=nemo_num_default,
            min_value=1,
            max_value=32,
        )
        min_speakers = self._parse_int_value(
            self.nemo_min_speakers_input.text(),
            default=nemo_min_default,
            min_value=1,
            max_value=32,
        )
        max_speakers = self._parse_int_value(
            self.nemo_max_speakers_input.text(),
            default=max(nemo_max_default, min_speakers),
            min_value=min_speakers,
            max_value=32,
        )

        suffix = self.overlay_suffix_input.text().strip() or ".captioned"
        if suffix and not suffix.startswith("."):
            suffix = f".{suffix}"

        return {
            "audio.overlap_sec": self._parse_float_value(
                self.overlap_input.text(),
                default=overlap_default,
                min_value=0.0,
                max_value=30.0,
            ),
            "asr.nemo_msdd.num_speakers": (
                fixed_num_speakers if speaker_mode == "manual" else 0
            ),
            "asr.nemo_msdd.min_speakers": min_speakers,
            "asr.nemo_msdd.max_speakers": max_speakers,
            "performance.max_concurrent_files": self._parse_int_value(
                self.max_concurrent_input.text(),
                default=max_concurrent_default,
                min_value=1,
                max_value=16,
            ),
            "performance.gpu_memory_fraction": self._parse_float_value(
                self.gpu_mem_fraction_input.text(),
                default=gpu_fraction_default,
                min_value=0.1,
                max_value=1.0,
            ),
            "video_text_overlay.renderer": str(self.overlay_renderer_combo.currentData() or "ffmpeg"),
            "video_text_overlay.style_effect": str(self.overlay_effect_combo.currentData() or "auto"),
            "video_text_overlay.output_suffix": suffix,
            "video_text_overlay.font_name": self.overlay_font_name_input.text().strip() or ("PingFang SC" if self.is_macos else "Microsoft YaHei"),
            "video_text_overlay.font_size": self._parse_int_value(
                self.overlay_font_size_input.text(),
                default=font_size_default,
                min_value=8,
                max_value=120,
            ),
            "video_text_overlay.max_line_chars": self._parse_int_value(
                self.overlay_max_chars_input.text(),
                default=max_chars_default,
                min_value=6,
                max_value=120,
            ),
            "video_text_overlay.bottom_margin_px": self._parse_int_value(
                self.overlay_margin_input.text(),
                default=margin_default,
                min_value=0,
                max_value=200,
            ),
            "video_text_overlay.ffmpeg_video_codec": str(
                self.overlay_codec_combo.currentData() or ("h264_videotoolbox" if self.is_macos else "h264_nvenc")
            ),
            "video_text_overlay.ffmpeg_crf": self._parse_int_value(
                self.overlay_quality_input.text(),
                default=crf_default,
                min_value=0,
                max_value=51,
            ),
            "video_text_overlay.ffmpeg_path": self.overlay_ffmpeg_input.text().strip(),
            "asr.nemo_msdd.hybrid_fusion.posterior_decoder.dump_examples.enabled": bool(
                self.posterior_dump_toggle_button.isChecked()
                and self._posterior_auto_run_twice_enabled()
            ),
            "asr.nemo_msdd.hybrid_fusion.posterior_decoder.dump_examples.output_dir": self._posterior_fusion_examples_dir(),
            "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.examples_dir": self._posterior_fusion_examples_dir(),
            "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.rttm_dir": self._posterior_fusion_rttm_dir(),
            "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.output_path": self._posterior_fusion_output_path(),
            "asr.nemo_msdd.hybrid_fusion.posterior_decoder.trainer.epochs": self._posterior_fusion_epochs(),
            "asr.nemo_msdd.hybrid_fusion.posterior_decoder.calibrator.persist_path": self._posterior_fusion_output_path(),
            "ui.posterior_fusion.auto_run_twice": self._posterior_auto_run_twice_enabled(),
        }

    @staticmethod
    def _write_advanced_payload(target_cfg: Dict[str, Any], payload: Dict[str, Any]):
        for dotted_key, value in payload.items():
            _set_nested(target_cfg, dotted_key, value)

    def _on_advanced_setting_edited(self, *_args):
        self._advanced_dirty = True
        self.set_status_chip("配置已修改", "info")

    def _apply_editor_settings_to_config(self):
        if self._language_dirty:
            self._write_language_payload(
                self.config._data,
                self._language_payload_from_editor(),
            )
        if self._translation_dirty:
            _set_nested(
                self.config._data,
                "translation.target_language",
                str(self.translation_target_combo.currentData() or "zh"),
            )
        if self._advanced_dirty:
            self._write_advanced_payload(
                self.config._data,
                self._advanced_payload_from_editor(),
            )

        _set_nested(self.config._data, "llm.api_key", self.dashscope_key_input.text().strip())
        _set_nested(
            self.config._data,
            "asr.faster_whisper.hf_token",
            self.hf_token_input.text().strip(),
        )

    def _config_save_path(self) -> Path:
        if self.config_path:
            return Path(self.config_path).resolve()
        return resolve_app_writable_path("config.yaml")

    def _persist_modified_config_fields(self) -> bool:
        self._apply_editor_settings_to_config()

        if not (
            self._language_dirty
            or self._translation_dirty
            or self._dashscope_dirty
            or self._hf_token_dirty
            or self._switch_dirty
            or self._advanced_dirty
        ):
            return False

        save_path = self._config_save_path()
        user_cfg: Dict[str, Any] = {}
        if save_path.exists():
            with open(save_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                user_cfg = loaded

        if self._language_dirty:
            self._write_language_payload(
                user_cfg,
                self._language_payload_from_editor(),
            )

        if self._translation_dirty:
            _set_nested(
                user_cfg,
                "translation.target_language",
                str(self.translation_target_combo.currentData() or "zh"),
            )

        if self._dashscope_dirty:
            _set_nested(user_cfg, "llm.api_key", self.dashscope_key_input.text().strip())

        if self._hf_token_dirty:
            _set_nested(
                user_cfg,
                "asr.faster_whisper.hf_token",
                self.hf_token_input.text().strip(),
            )

        if self._switch_dirty:
            for _, dotted_key, _ in self._active_switch_specs():
                _set_nested(user_cfg, dotted_key, _get_nested(self.config._data, dotted_key, False))

        if self._advanced_dirty:
            self._write_advanced_payload(
                user_cfg,
                self._advanced_payload_from_editor(),
            )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(user_cfg, f, allow_unicode=True, sort_keys=False)

        self._language_dirty = False
        self._translation_dirty = False
        self._dashscope_dirty = False
        self._hf_token_dirty = False
        self._switch_dirty = False
        self._advanced_dirty = False
        return True

    def save_config_overrides(self):
        try:
            updated = self._persist_modified_config_fields()
            if updated:
                self.set_status_chip("系统：配置已成功保存", "ok")
            else:
                self.set_status_chip("系统：未发现更改", "info")
        except Exception as e:
            QMessageBox.warning(self, "写入错误", f"保存配置文件失败：\n{e}")

    def _is_dark_mode(self) -> bool:
        app = QApplication.instance()
        style_hints = app.styleHints() if app is not None else None
        color_scheme_enum = getattr(Qt, "ColorScheme", None)
        if style_hints is not None and color_scheme_enum is not None:
            try:
                scheme = style_hints.colorScheme()
                if scheme == color_scheme_enum.Light:
                    return False
            except Exception:
                pass
        return True 

    @staticmethod
    def _mix_color(base: QColor, other: QColor, ratio: float) -> QColor:
        t = max(0.0, min(1.0, float(ratio)))
        inv = 1.0 - t
        return QColor(
            int(base.red() * inv + other.red() * t),
            int(base.green() * inv + other.green() * t),
            int(base.blue() * inv + other.blue() * t),
        )

    @staticmethod
    def _rgba(color: QColor, alpha: int) -> str:
        a = max(0, min(255, int(alpha)))
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {a})"

    def _ui_font_css_stack(self) -> str:
        if self.is_macos:
            return '"PingFang SC", "Helvetica Neue"'
        return '"Microsoft YaHei UI", "Segoe UI Variable Text", "Segoe UI"'

    def _display_font_css_stack(self) -> str:
        if self.is_macos:
            return '"PingFang SC", "Helvetica Neue"'
        return '"Microsoft YaHei UI", "Segoe UI Variable Display", "Segoe UI"'

    def _mono_font_css_stack(self) -> str:
        if self.is_macos:
            return '"SF Mono", "Menlo", "Monaco", monospace'
        return '"Consolas", "Cascadia Code", monospace'

    def _macos_native_accent_color(self) -> Optional[QColor]:
        runtime = _macos_objc_runtime()
        if runtime is None:
            return None

        try:
            (
                get_class,
                register_sel,
                send_id,
                send_id_id,
                send_double,
                _send_void_bool,
                _send_void_integer,
            ) = runtime
            ns_color = get_class(b"NSColor")
            ns_color_space = get_class(b"NSColorSpace")
            if not ns_color or not ns_color_space:
                return None

            accent = send_id(ns_color, register_sel(b"controlAccentColor"))
            if not accent:
                return None

            srgb_space = send_id(ns_color_space, register_sel(b"sRGBColorSpace"))
            if srgb_space:
                converted = send_id_id(
                    accent,
                    register_sel(b"colorUsingColorSpace:"),
                    srgb_space,
                )
                if converted:
                    accent = converted

            red = max(0.0, min(1.0, float(send_double(accent, register_sel(b"redComponent")))))
            green = max(0.0, min(1.0, float(send_double(accent, register_sel(b"greenComponent")))))
            blue = max(0.0, min(1.0, float(send_double(accent, register_sel(b"blueComponent")))))
            color = QColor(
                int(round(red * 255)),
                int(round(green * 255)),
                int(round(blue * 255)),
            )
            return color if color.isValid() else None
        except Exception:
            return None

    def _system_accent_color(self) -> QColor:
        app = QApplication.instance()
        palette = app.palette() if app is not None else self.palette()

        candidates: List[QColor] = []
        native_accent = self._macos_native_accent_color() if self.is_macos else None
        if native_accent is not None and native_accent.isValid():
            candidates.append(QColor(native_accent))
        accent_role = getattr(QPalette, "Accent", None)
        for role in (accent_role, QPalette.Highlight, QPalette.Link):
            if role is None:
                continue
            try:
                candidates.append(QColor(palette.color(role)))
            except Exception:
                continue

        accent = None
        for color in candidates:
            if color.isValid() and color.alpha() > 0:
                accent = QColor(color)
                break
        if accent is None:
            accent = QColor("#4fa2ff")

        if accent.saturation() < 72 and native_accent is None:
            accent = self._mix_color(accent, QColor("#4fa2ff"), 0.45)
        if accent.value() < 96:
            accent = accent.lighter(150)
        if accent.value() > 225:
            accent = accent.darker(130)
        accent.setAlpha(255)
        return accent

    def _build_glass_override_css(self) -> str:
        accent = QColor(self._accent_color)
        if not accent.isValid():
            accent = QColor("#4fa2ff")
        ui_font_stack = self._ui_font_css_stack()
        display_font_stack = self._display_font_css_stack()
        mono_font_stack = self._mono_font_css_stack()
        candy = self._mix_color(accent, QColor("#ffbfd6"), 0.54 if self._is_dark_theme else 0.64)
        cream = self._mix_color(accent, QColor("#fff8ef"), 0.74 if self._is_dark_theme else 0.88)
        mint = self._mix_color(accent, QColor("#d6ffee"), 0.44 if self._is_dark_theme else 0.56)
        accent_hover = accent.lighter(112) if self._is_dark_theme else accent.darker(103)
        accent_soft = (
            self._mix_color(accent, QColor("#ffffff"), 0.48)
            if self._is_dark_theme
            else self._mix_color(accent, QColor("#000000"), 0.16)
        )

        if self._is_dark_theme:
            text_color = QColor("#eef4ff")
            secondary_text = QColor("#ccd7e7")
            tertiary_text = QColor("#9cabc0")
            surface = QColor(22, 28, 38)
            surface_alt = QColor(14, 18, 27)
            field = QColor(25, 31, 43)
            pop_surface = QColor(17, 21, 30)
            progress_bg = QColor(16, 21, 31)
            card_alpha = 154
            card_alt_alpha = 138
            field_alpha = 150
            popup_alpha = 196
            bubble_alpha = 184
            white_border = self._rgba(QColor("#ffffff"), 84)
            soft_border = self._rgba(accent_soft, 116)
            focus_ring = self._rgba(candy, 176)
            selection_bg = self._rgba(candy, 92)
            selection_hover = self._rgba(candy, 62)
            menu_bg = self._rgba(pop_surface, popup_alpha)
        else:
            text_color = QColor("#2f3848")
            secondary_text = QColor("#667187")
            tertiary_text = QColor("#8994a8")
            surface = QColor("#ffffff")
            surface_alt = QColor("#f4f7fb")
            field = QColor("#fffafc")
            pop_surface = QColor("#fffdfd")
            progress_bg = QColor("#f4f6fb")
            card_alpha = 194
            card_alt_alpha = 176
            field_alpha = 222
            popup_alpha = 238
            bubble_alpha = 230
            white_border = self._rgba(QColor("#ffffff"), 154)
            soft_border = self._rgba(accent_soft, 92)
            focus_ring = self._rgba(candy, 154)
            selection_bg = self._rgba(candy, 112)
            selection_hover = self._rgba(candy, 74)
            menu_bg = self._rgba(pop_surface, popup_alpha)

        title_color = cream if self._is_dark_theme else self._mix_color(accent, QColor("#1c3149"), 0.78)
        subtitle_color = self._mix_color(secondary_text, accent, 0.28)
        soft_glass = self._rgba(surface, card_alpha)
        soft_glass_alt = self._rgba(surface_alt, card_alt_alpha)
        field_glass = self._rgba(field, field_alpha)
        bubble_glass = self._rgba(surface, bubble_alpha)
        chip_text = "#fffefd" if self._is_dark_theme else "#5c4456"

        return f"""
            QMainWindow,
            QWidget#Root,
            QDialog[glassWindow="true"],
            QMessageBox,
            QFileDialog {{
                background: transparent;
                color: {text_color.name()};
                font-family: {ui_font_stack};
                font-size: 13px;
            }}
            QWidget {{
                color: {text_color.name()};
                selection-background-color: {selection_bg};
            }}
            QFrame#HeaderCard,
            QFrame#ProgressCard,
            QFrame#StepCard,
            QFrame#Card,
            QFrame#AdvancedPanel,
            QFrame#OnboardingPanel,
            QDialog[glassWindow="true"],
            QMessageBox,
            QFileDialog {{
                background: {soft_glass};
                border: 1px solid {white_border};
                border-radius: 24px;
            }}
            QDialog[glassWindow="true"],
            QMessageBox,
            QFileDialog {{
                background: {self._rgba(pop_surface, popup_alpha)};
            }}
            QFrame#HeaderCard {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {self._rgba(surface, min(255, card_alpha + 12))},
                    stop:0.55 {soft_glass},
                    stop:1 {self._rgba(surface_alt, min(255, card_alt_alpha + 14))}
                );
                border-top: 1px solid {self._rgba(candy, 148 if self._is_dark_theme else 132)};
            }}
            QFrame#ProgressCard,
            QFrame#StepCard,
            QFrame#Card,
            QFrame#AdvancedPanel {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {soft_glass},
                    stop:1 {soft_glass_alt}
                );
            }}
            QFrame#Card[switchMode="all"] {{
                border: 1px solid {self._rgba(QColor("#72d99a"), 166 if self._is_dark_theme else 146)};
            }}
            QFrame#Card[switchMode="mixed"] {{
                border: 1px solid {self._rgba(accent, 164 if self._is_dark_theme else 138)};
            }}
            QFrame#Card[switchMode="none"] {{
                border: 1px solid {self._rgba(QColor("#93a0b4"), 144 if self._is_dark_theme else 126)};
            }}
            QSplitter::handle {{
                background: {self._rgba(candy, 54 if self._is_dark_theme else 70)};
                border: 1px solid {self._rgba(accent_soft, 70 if self._is_dark_theme else 88)};
                border-radius: 7px;
            }}
            QSplitter::handle:hover {{
                background: {self._rgba(candy, 96 if self._is_dark_theme else 112)};
                border: 1px solid {self._rgba(accent_hover, 118 if self._is_dark_theme else 128)};
            }}
            QLabel#Title {{
                font-family: {display_font_stack};
                font-weight: 700;
                font-size: 28px;
                color: {title_color.name()};
                letter-spacing: {"0.12px" if self.is_macos else "0.35px"};
            }}
            QLabel#Subtitle {{
                font-family: {mono_font_stack};
                color: {subtitle_color.name()};
                font-size: 11px;
                letter-spacing: {"0.1px" if self.is_macos else "0.25px"};
            }}
            QLabel#cyantext {{
                font-family: {display_font_stack};
                color: {self._mix_color(candy, text_color, 0.30).name()};
                font-size: 14px;
                font-weight: 700;
                letter-spacing: {"0.1px" if self.is_macos else "0.24px"};
            }}
            QLabel#InputLabel {{
                color: {secondary_text.name()};
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#ProgressLabel,
            QLabel#SwitchTitle,
            QLabel#OnboardingTitle {{
                color: {text_color.name()};
                font-weight: 600;
            }}
            QLabel#SwitchDesc,
            QLabel#OnboardingBody {{
                color: {tertiary_text.name()};
            }}
            QLabel#ConfigGuide {{
                background: {self._rgba(candy, 42 if self._is_dark_theme else 38)};
                border: 1px solid {self._rgba(accent_soft, 118 if self._is_dark_theme else 96)};
                border-left: 4px solid {candy.name()};
                border-radius: 16px;
                color: {secondary_text.name()};
                padding: 8px 12px;
            }}
            QPushButton#StepBtn {{
                min-height: 32px;
                padding: 4px 14px;
                border-radius: 11px;
                border: 1px solid {self._rgba(accent_soft, 94 if self._is_dark_theme else 88)};
                background: {self._rgba(surface_alt, 126 if self._is_dark_theme else 180)};
                color: {secondary_text.name()};
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#StepBtn:hover {{
                background: {selection_hover};
                color: {text_color.name()};
            }}
            QPushButton#StepBtn[activeStep="true"] {{
                border: 1px solid {focus_ring};
                background: {selection_bg};
                color: {text_color.name()};
            }}
            QListWidget#ListSurface,
            QTreeWidget#TreeSurface,
            QWidget#ConfigViewport,
            QFrame#BurstFeed,
            QPlainTextEdit#LogSurface {{
                background: {field_glass};
                border: 1px solid {soft_border};
                border-radius: 20px;
            }}
            QFrame#BurstFeed[role="log"] {{
                background: {self._rgba(field, min(255, field_alpha + 26))};
                border-radius: 22px;
            }}
            QListWidget#ListSurface::item {{
                padding: 10px 12px;
                margin: 2px 6px;
                border-radius: 14px;
                color: {text_color.name()};
            }}
            QListWidget#ListSurface::item:hover,
            QTreeWidget#TreeSurface::item:hover {{
                background: {selection_hover};
            }}
            QListWidget#ListSurface::item:selected,
            QTreeWidget#TreeSurface::item:selected {{
                background: {selection_bg};
                color: {text_color.name()};
                border: none;
            }}
            QTreeWidget#TreeSurface {{
                alternate-background-color: {self._rgba(surface_alt, 80 if self._is_dark_theme else 112)};
            }}
            QTreeWidget#TreeSurface::item {{
                padding: 5px 6px;
            }}
            QHeaderView::section {{
                background: {self._rgba(surface_alt, 138 if self._is_dark_theme else 204)};
                color: {tertiary_text.name()};
                border: none;
                border-bottom: 1px solid {self._rgba(accent_soft, 86 if self._is_dark_theme else 74)};
                padding: 8px 10px;
                font-family: {mono_font_stack};
                font-size: 11px;
                font-weight: 600;
            }}
            QPlainTextEdit#LogSurface {{
                padding: 8px;
                background: {self._rgba(field, min(255, field_alpha + 4))};
                color: {text_color.name()};
                font-family: {mono_font_stack};
                font-size: 11.5px;
            }}
            QFrame#SwitchRow {{
                background: {self._rgba(surface_alt, 118 if self._is_dark_theme else 188)};
                border: 1px solid {self._rgba(accent_soft, 80 if self._is_dark_theme else 70)};
                border-radius: 18px;
            }}
            QComboBox#ConfigInput,
            QLineEdit#ConfigInput {{
                min-height: 30px;
                border: 1px solid {self._rgba(accent_soft, 102 if self._is_dark_theme else 88)};
                background: {self._rgba(field, min(255, field_alpha + 8))};
                border-radius: 14px;
                padding: 3px 12px;
                color: {text_color.name()};
                selection-background-color: {selection_bg};
            }}
            QComboBox#ConfigInput:hover,
            QLineEdit#ConfigInput:hover {{
                border: 1px solid {self._rgba(accent_soft, 130 if self._is_dark_theme else 112)};
            }}
            QComboBox#ConfigInput:focus,
            QLineEdit#ConfigInput:focus {{
                border: 1px solid {focus_ring};
                background: {self._rgba(field, min(255, field_alpha + 22))};
            }}
            QComboBox#ConfigInput::drop-down {{
                border: none;
                width: 24px;
                background: transparent;
            }}
            QComboBox#ConfigInput QAbstractItemView,
            QListView,
            QMenu {{
                background: {menu_bg};
                border: 1px solid {white_border};
                border-radius: 18px;
                outline: none;
                padding: 6px;
            }}
            QComboBox#ConfigInput QAbstractItemView::item,
            QListView::item,
            QMenu::item {{
                padding: 8px 12px;
                margin: 2px 4px;
                border-radius: 12px;
                background: transparent;
            }}
            QComboBox#ConfigInput QAbstractItemView::item:selected,
            QListView::item:selected,
            QMenu::item:selected {{
                background: {selection_bg};
                color: {text_color.name()};
            }}
            QScrollArea#ConfigScroll,
            QScrollArea#ControlScroll {{
                border: none;
                background: transparent;
            }}
            QWidget#ConfigScrollContent {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 9px;
                margin: 4px 2px 4px 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {self._rgba(candy, 90 if self._is_dark_theme else 108)};
                min-height: 24px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {self._rgba(accent_hover, 126 if self._is_dark_theme else 138)};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
                border: none;
                height: 0px;
            }}
            QProgressBar#TaskProgress {{
                min-height: 24px;
                padding: 3px;
                background: {self._rgba(progress_bg, 222 if self._is_dark_theme else 236)};
                border: 1px solid {self._rgba(accent_soft, 124 if self._is_dark_theme else 106)};
                border-radius: 14px;
                color: {cream.name() if self._is_dark_theme else "#5a4658"};
                text-align: center;
                font-weight: 600;
            }}
            QProgressBar#TaskProgress::chunk {{
                border-radius: 11px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {candy.name()},
                    stop:0.5 {accent.name()},
                    stop:1 {mint.name()}
                );
            }}
            QFrame#OnboardingPanel {{
                background: {bubble_glass};
                border: 1px solid {focus_ring};
                border-radius: 24px;
            }}
            QFrame#BurstItem_info,
            QFrame#BurstItem_ok,
            QFrame#BurstItem_warn,
            QFrame#BurstItem_err {{
                border-radius: 22px;
                border: 1px solid {self._rgba(QColor("#ffffff"), 54 if self._is_dark_theme else 94)};
            }}
            QFrame#BurstItem_info {{
                background: {self._rgba(self._mix_color(accent, QColor("#ffffff"), 0.18), 124 if self._is_dark_theme else 198)};
            }}
            QFrame#BurstItem_ok {{
                background: {self._rgba(self._mix_color(QColor("#86d7b4"), surface, 0.18), 126 if self._is_dark_theme else 202)};
            }}
            QFrame#BurstItem_warn {{
                background: {self._rgba(self._mix_color(QColor("#ffc995"), surface, 0.20), 128 if self._is_dark_theme else 206)};
            }}
            QFrame#BurstItem_err {{
                background: {self._rgba(self._mix_color(QColor("#ff9eb2"), surface, 0.22), 132 if self._is_dark_theme else 210)};
            }}
            QFrame#BurstItem_info[overlayStack="true"],
            QFrame#BurstItem_ok[overlayStack="true"],
            QFrame#BurstItem_warn[overlayStack="true"],
            QFrame#BurstItem_err[overlayStack="true"] {{
                background: transparent;
                border: none;
            }}
            QLabel#BurstChip {{
                padding: 2px 10px;
                border-radius: 10px;
                color: {chip_text};
                font-size: 11px;
                font-weight: 700;
            }}
            QLabel#BurstChip[level="info"] {{
                background: {self._rgba(accent, 152 if self._is_dark_theme else 118)};
            }}
            QLabel#BurstChip[level="ok"] {{
                background: {self._rgba(QColor("#69c997"), 150 if self._is_dark_theme else 126)};
            }}
            QLabel#BurstChip[level="warn"] {{
                background: {self._rgba(QColor("#ffb25f"), 156 if self._is_dark_theme else 134)};
            }}
            QLabel#BurstChip[level="err"] {{
                background: {self._rgba(QColor("#ff7c9a"), 164 if self._is_dark_theme else 146)};
            }}
            QLabel#BurstStamp {{
                color: {secondary_text.name()};
                font-family: {mono_font_stack};
                font-size: 10px;
            }}
            QLabel#BurstText {{
                color: {text_color.name()};
                font-size: 12px;
            }}
            QLabel#BurstText[role="log"] {{
                color: {text_color.name()};
                font-family: {mono_font_stack};
                font-size: 11.5px;
            }}
            QToolTip {{
                background: {menu_bg};
                color: {text_color.name()};
                border: 1px solid {white_border};
                border-radius: 12px;
                padding: 6px 8px;
            }}
            QLabel#ProgressPet {{
                background: transparent;
            }}
        """

    def _build_windows11_button_css(self) -> str:
        accent = QColor(self._accent_color)
        if not accent.isValid():
            accent = QColor("#4fa2ff")
        accent_soft = self._mix_color(accent, QColor("#ffffff"), 0.38) if self._is_dark_theme else self._mix_color(accent, QColor("#000000"), 0.18)
        accent_hover = accent.lighter(116) if self._is_dark_theme else accent.darker(108)
        candy = self._mix_color(accent, QColor("#ffbfd6"), 0.56 if self._is_dark_theme else 0.66)
        base_text = "#edf4ff" if self._is_dark_theme else "#21364b"
        ghost_bg = self._rgba(QColor(24, 30, 42) if self._is_dark_theme else QColor(255, 255, 255), 178 if self._is_dark_theme else 214)
        ghost_hover = self._rgba(candy, 70 if self._is_dark_theme else 76)
        ghost_pressed = self._rgba(candy, 94 if self._is_dark_theme else 102)
        warn = QColor("#ffb769")
        danger = QColor("#ff7e9c")
        small_h = self._scaled_px(24, min_px=22, max_px=34)
        normal_h = self._scaled_px(36, min_px=32, max_px=46)
        round_normal = self._scaled_px(12, min_px=10, max_px=16)
        round_small = self._scaled_px(10, min_px=8, max_px=14)
        help_side = self._scaled_px(18, min_px=18, max_px=28)
        help_radius = max(7, help_side // 2)

        return f"""
            QPushButton#PrimaryBtn,
            QPushButton#GhostBtn,
            QPushButton#WarningBtn,
            QPushButton#DangerHoldBtn,
            QMessageBox QPushButton,
            QFileDialog QPushButton {{
                min-height: {normal_h}px;
                border-radius: {round_normal}px;
                padding: 6px 14px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton#GhostBtn,
            QPushButton#GhostBtnMin,
            QMessageBox QPushButton,
            QFileDialog QPushButton {{
                background: {ghost_bg};
                color: {base_text};
                border: 1px solid {self._rgba(accent_soft, 150)};
            }}
            QPushButton#GhostBtn:hover,
            QPushButton#GhostBtnMin:hover,
            QMessageBox QPushButton:hover,
            QFileDialog QPushButton:hover {{
                background: {ghost_hover};
                border: 1px solid {self._rgba(accent_soft, 210)};
            }}
            QPushButton#GhostBtn:pressed,
            QPushButton#GhostBtnMin:pressed,
            QMessageBox QPushButton:pressed,
            QFileDialog QPushButton:pressed {{
                background: {ghost_pressed};
            }}
            QPushButton#PrimaryBtn {{
                color: {"#fffdfd" if self._is_dark_theme else "#573a49"};
                border: 1px solid {self._rgba(accent_soft, 205)};
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {self._rgba(candy, 196)},
                    stop:0.55 {self._rgba(accent, 184)},
                    stop:1 {self._rgba(accent_hover, 168)}
                );
            }}
            QPushButton#PrimaryBtn:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {self._rgba(candy, 224)},
                    stop:0.55 {self._rgba(accent, 206)},
                    stop:1 {self._rgba(accent_hover, 188)}
                );
            }}
            QPushButton#PrimaryBtn:pressed {{
                background: {self._rgba(accent, 220)};
            }}
            QPushButton#PrimaryBtn:disabled,
            QPushButton#GhostBtn:disabled,
            QPushButton#GhostBtnMin:disabled,
            QPushButton#WarningBtn:disabled,
            QPushButton#DangerHoldBtn:disabled,
            QPushButton#EyeBtn:disabled,
            QPushButton#HelpDotBtn:disabled {{
                color: {self._rgba(QColor("#8c9aad"), 230)};
                border: 1px solid {self._rgba(QColor("#8c9aad"), 112)};
                background: {self._rgba(QColor("#7d8795"), 62)};
            }}
            QPushButton#GhostBtnMin,
            QPushButton#EyeBtn,
            QPushButton#HelpDotBtn {{
                min-height: {small_h}px;
                border-radius: {round_small}px;
                font-size: 11px;
                padding: 2px 8px;
            }}
            QPushButton#EyeBtn {{
                min-width: {small_h}px;
                max-width: {small_h}px;
                min-height: {small_h}px;
                max-height: {small_h}px;
                padding: 0;
            }}
            QPushButton#HelpDotBtn {{
                min-width: {help_side}px;
                max-width: {help_side}px;
                min-height: {help_side}px;
                max-height: {help_side}px;
                border-radius: {help_radius}px;
                font-weight: 700;
                padding: 0;
            }}
            QPushButton#HelpDotBtn:hover,
            QPushButton#EyeBtn:hover {{
                background: {self._rgba(candy, 78 if self._is_dark_theme else 84)};
                border: 1px solid {self._rgba(accent_soft, 220)};
            }}
            QPushButton#WarningBtn {{
                background: {self._rgba(warn, 86 if self._is_dark_theme else 74)};
                color: {"#fff7ef" if self._is_dark_theme else "#7b4b16"};
                border: 1px solid {self._rgba(warn, 182 if self._is_dark_theme else 146)};
            }}
            QPushButton#WarningBtn:hover {{
                background: {self._rgba(warn, 114 if self._is_dark_theme else 92)};
            }}
            QPushButton#DangerHoldBtn {{
                background: {self._rgba(danger, 62 if self._is_dark_theme else 58)};
                color: {"#fff6fa" if self._is_dark_theme else "#8d2f4e"};
                border: 1px solid {self._rgba(danger, 168 if self._is_dark_theme else 138)};
            }}
            QPushButton#DangerHoldBtn:hover {{
                background: {self._rgba(danger, 92 if self._is_dark_theme else 82)};
            }}
        """

    def _apply_windows11_backdrop(self) -> None:
        if os.name != "nt":
            return
        try:
            hwnd = int(self.winId())
        except Exception:
            return
        if hwnd <= 0:
            return
        try:
            dwmapi = ctypes.windll.dwmapi
        except Exception:
            return

        try:
            use_dark_mode_attr = 20
            dark_value = ctypes.c_int(1 if self._is_dark_theme else 0)
            dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(hwnd),
                use_dark_mode_attr,
                ctypes.byref(dark_value),
                ctypes.sizeof(dark_value),
            )
        except Exception:
            pass

        try:
            backdrop_attr = 38
            # Prefer Tabbed (Mica Alt) for better inactive-window stability.
            for backdrop_kind in (4, 2):
                backdrop_value = ctypes.c_int(backdrop_kind)
                result = dwmapi.DwmSetWindowAttribute(
                    ctypes.c_void_p(hwnd),
                    backdrop_attr,
                    ctypes.byref(backdrop_value),
                    ctypes.sizeof(backdrop_value),
                )
                if int(result) == 0:
                    break
        except Exception:
            pass

    def _apply_styles(self):
        updates_enabled = self.updatesEnabled()
        if updates_enabled:
            self.setUpdatesEnabled(False)
        self._is_dark_theme = self._is_dark_mode()
        self._accent_color = self._system_accent_color()
        
        if self._is_dark_theme:
            self.setStyleSheet(
                """
                QWidget#Root {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #050a10, stop:1 #0c1420);
                    color: #d8e5f5;
                    font-family: "Microsoft YaHei UI", "Segoe UI Variable Text", "Segoe UI";
                    font-size: 13px;
                }
                QSplitter::handle {
                    background: rgba(0, 240, 255, 0.15);
                    border: 1px solid rgba(0, 240, 255, 0.05);
                    border-radius: 3px;
                }
                QSplitter::handle:hover {
                    background: rgba(0, 240, 255, 0.8);
                    border: 1px solid rgba(0, 240, 255, 0.65);
                }
                QSplitter::handle:vertical {
                    margin: 2px 20px;
                }
                QSplitter::handle:horizontal {
                    margin: 20px 2px;
                }
                QFrame#HeaderCard {
                    background: rgba(12, 20, 32, 0.85);
                    border: 1px solid rgba(0, 240, 255, 0.3);
                    border-radius: 8px;
                    border-top: 2px solid #00f0ff;
                }
                QFrame#ProgressCard {
                    background: rgba(10, 18, 30, 0.78);
                    border: 1px solid rgba(0, 240, 255, 0.24);
                    border-radius: 10px;
                }
                QFrame#Card {
                    background: rgba(16, 26, 40, 0.65);
                    border: 1px solid rgba(0, 240, 255, 0.15);
                    border-radius: 10px;
                }
                QLabel#Title {
                    font-family: "Microsoft YaHei UI", "Orbitron";
                    font-weight: bold;
                    font-size: 24px;
                    color: #ffffff;
                    letter-spacing: 1px;
                }
                QLabel#Subtitle {
                    font-family: "Consolas", "Cascadia Code";
                    font-size: 11px;
                    color: #00f0ff;
                    letter-spacing: 1px;
                }
                QLabel#cyantext {
                    font-weight: bold;
                    font-size: 14px;
                    color: #00f0ff;
                    letter-spacing: 1px;
                }
                QLabel#InputLabel {
                    color: #8da5c7;
                    font-size: 12px;
                    font-weight: bold;
                }
                QLabel#ProgressLabel {
                    color: #8ed6ea;
                    font-size: 12px;
                    font-weight: bold;
                }
                QProgressBar#TaskProgress {
                    min-height: 16px;
                    border: 1px solid rgba(0, 240, 255, 0.45);
                    border-radius: 8px;
                    background: rgba(0, 30, 40, 0.75);
                    color: #d6f9ff;
                    text-align: center;
                    font-size: 11px;
                    font-weight: bold;
                }
                QProgressBar#TaskProgress::chunk {
                    border-radius: 7px;
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #00c8ff,
                        stop:0.5 #00f0ff,
                        stop:1 #00ffbf
                    );
                }
                QPushButton {
                    border-radius: 6px;
                    min-height: 36px;
                    padding: 6px 16px;
                    font-weight: bold;
                    font-size: 13px;
                    letter-spacing: 1px;
                }
                QPushButton#PrimaryBtn {
                    background: rgba(0, 240, 255, 0.15);
                    color: #00f0ff;
                    border: 1px solid #00f0ff;
                }
                QPushButton#PrimaryBtn:hover { 
                    background: rgba(0, 240, 255, 0.3); 
                    border: 1px solid #33f5ff;
                }
                QPushButton#PrimaryBtn:disabled {
                    background: rgba(40, 60, 80, 0.4);
                    border: 1px solid #2a4158;
                    color: #59738f;
                }
                QPushButton#WarningBtn {
                    background: rgba(255, 0, 85, 0.15);
                    color: #ff0055;
                    border: 1px solid #ff0055;
                }
                QPushButton#WarningBtn:hover { background: rgba(255, 0, 85, 0.3); }
                QPushButton#WarningBtn:disabled {
                    background: rgba(40, 60, 80, 0.4);
                    border: 1px solid #2a4158;
                    color: #59738f;
                }
                QPushButton#GhostBtn {
                    background: rgba(255, 255, 255, 0.05);
                    color: #a9c2e3;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                QPushButton#GhostBtn:hover { 
                    background: rgba(255, 255, 255, 0.1); 
                    color: #ffffff;
                }
                QPushButton#GhostBtnMin {
                    background: transparent;
                    color: #00f0ff;
                    border: 1px solid rgba(0, 240, 255, 0.4);
                    border-radius: 4px;
                    min-height: 24px;
                    padding: 4px 10px;
                    font-size: 11px;
                }
                QPushButton#GhostBtnMin:hover {
                    background: rgba(0, 240, 255, 0.2);
                }
                QPushButton#DangerHoldBtn {
                    background: rgba(255, 0, 85, 0.08);
                    color: #ff7a9f;
                    border: 1px solid rgba(255, 0, 85, 0.45);
                    border-radius: 4px;
                    min-height: 24px;
                    padding: 4px 10px;
                    font-size: 11px;
                }
                QPushButton#DangerHoldBtn:hover {
                    background: rgba(255, 0, 85, 0.18);
                    color: #ffb0c3;
                }
                QPushButton#DangerHoldBtn:disabled {
                    background: rgba(40, 60, 80, 0.4);
                    color: #657b95;
                    border: 1px solid rgba(101, 123, 149, 0.45);
                }
                QPushButton#EyeBtn {
                    background: rgba(0, 240, 255, 0.05);
                    color: #00f0ff;
                    border: 1px solid rgba(0, 240, 255, 0.2);
                    border-radius: 4px;
                    font-size: 12px;
                    padding: 0;
                    min-height: 24px;
                }
                QPushButton#EyeBtn:hover {
                    background: rgba(0, 240, 255, 0.2);
                    border: 1px solid rgba(0, 240, 255, 0.5);
                }
                QPushButton#EyeBtn:checked {
                    background: rgba(0, 240, 255, 0.3);
                    border: 1px solid #00f0ff;
                }
                QPushButton#HelpDotBtn {
                    min-height: 18px;
                    max-height: 18px;
                    min-width: 18px;
                    max-width: 18px;
                    padding: 0;
                    border-radius: 9px;
                    font-size: 11px;
                    font-weight: bold;
                    color: #00f0ff;
                    background: rgba(0, 240, 255, 0.08);
                    border: 1px solid rgba(0, 240, 255, 0.35);
                }
                QPushButton#HelpDotBtn:hover {
                    background: rgba(0, 240, 255, 0.2);
                    border: 1px solid rgba(0, 240, 255, 0.7);
                }
                QPushButton#HelpDotBtn:disabled {
                    color: #59738f;
                    background: rgba(40, 60, 80, 0.35);
                    border: 1px solid rgba(89, 115, 143, 0.35);
                }
                QListWidget#ListSurface,
                QTreeWidget#TreeSurface,
                QPlainTextEdit#LogSurface {
                    background: rgba(5, 10, 15, 0.45);
                    border: 1px solid rgba(0, 240, 255, 0.1);
                    border-radius: 6px;
                }
                QListWidget#ListSurface::item {
                    padding: 8px;
                    border-bottom: 1px solid rgba(0, 240, 255, 0.05);
                    color: #c9daf0;
                }
                QListWidget#ListSurface::item:selected {
                    background: rgba(0, 240, 255, 0.2);
                    color: #ffffff;
                    border-left: 3px solid #00f0ff;
                }
                QTreeWidget#TreeSurface {
                    alternate-background-color: rgba(0, 240, 255, 0.02);
                    color: #c9daf0;
                }
                QTreeWidget#TreeSurface::item {
                    padding: 4px;
                }
                QTreeWidget#TreeSurface::item:selected {
                    background: rgba(0, 240, 255, 0.2);
                    color: #ffffff;
                }
                QHeaderView::section {
                    background: rgba(5, 10, 15, 0.8);
                    color: #00f0ff;
                    border: none;
                    border-bottom: 1px solid rgba(0, 240, 255, 0.2);
                    padding: 6px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPlainTextEdit#LogSurface {
                    font-family: "Consolas", "Cascadia Code";
                    font-size: 12px;
                    color: #00f0ff;
                }
                QFrame#SwitchRow {
                    background: rgba(5, 10, 15, 0.4);
                    border: 1px solid rgba(0, 240, 255, 0.1);
                    border-radius: 6px;
                }
                QLabel#SwitchTitle {
                    font-weight: bold;
                    color: #eef3ff;
                    font-size: 12px;
                }
                QLabel#SwitchDesc {
                    color: #597e9c;
                    font-size: 10px;
                }
                QLabel#ConfigGuide {
                    background: rgba(0, 240, 255, 0.05);
                    border: 1px solid rgba(0, 240, 255, 0.2);
                    border-left: 3px solid #00f0ff;
                    border-radius: 4px;
                    color: #8db5cc;
                    font-size: 11px;
                    padding: 4px 8px;
                }
                QComboBox#ConfigInput,
                QLineEdit#ConfigInput {
                    min-height: 26px;
                    background: rgba(5, 10, 15, 0.6);
                    border: 1px solid rgba(0, 240, 255, 0.2);
                    border-radius: 4px;
                    padding: 1px 8px;
                    color: #e8eef8;
                    font-size: 12px;
                }
                QComboBox#ConfigInput::drop-down {
                    border: none;
                    width: 20px;
                }
                QComboBox#ConfigInput:focus,
                QLineEdit#ConfigInput:focus {
                    border: 1px solid #00f0ff;
                    background: rgba(0, 240, 255, 0.05);
                }
                QScrollArea#ConfigScroll {
                    border: none;
                    background: transparent;
                }
                QWidget#ConfigViewport {
                    background: rgba(5, 10, 15, 0.26);
                    border-radius: 6px;
                }
                QWidget#ConfigScrollContent {
                    background: transparent;
                }
                QScrollBar:vertical {
                    background: transparent;
                    width: 6px;
                    margin: 2px;
                }
                QScrollBar::handle:vertical {
                    background: rgba(0, 240, 255, 0.3);
                    min-height: 20px;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:hover {
                    background: rgba(0, 240, 255, 0.6);
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QWidget#Root {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e6eef5, stop:1 #cfdbe6);
                    color: #1a2a3a;
                    font-family: "Microsoft YaHei UI", "Segoe UI Variable Text", "Segoe UI";
                    font-size: 13px;
                }
                QSplitter::handle {
                    background: rgba(0, 195, 217, 0.15);
                    border: 1px solid rgba(0, 195, 217, 0.05);
                    border-radius: 3px;
                }
                QSplitter::handle:hover {
                    background: rgba(0, 195, 217, 0.6);
                    border: 1px solid rgba(0, 195, 217, 0.55);
                }
                QSplitter::handle:vertical {
                    margin: 2px 20px;
                }
                QSplitter::handle:horizontal {
                    margin: 20px 2px;
                }
                QFrame#HeaderCard {
                    background: rgba(255, 255, 255, 0.7);
                    border: 1px solid rgba(255, 255, 255, 0.9);
                    border-radius: 8px;
                    border-top: 2px solid #00c3d9;
                }
                QFrame#ProgressCard {
                    background: rgba(255, 255, 255, 0.65);
                    border: 1px solid rgba(0, 195, 217, 0.35);
                    border-radius: 10px;
                }
                QFrame#Card {
                    background: rgba(255, 255, 255, 0.5);
                    border: 1px solid rgba(255, 255, 255, 0.8);
                    border-radius: 10px;
                }
                QLabel#Title {
                    font-family: "Microsoft YaHei UI", "Orbitron";
                    font-weight: bold;
                    font-size: 24px;
                    color: #0f1f33;
                    letter-spacing: 1px;
                }
                QLabel#Subtitle {
                    font-family: "Consolas", "Cascadia Code";
                    font-size: 11px;
                    color: #0099ab;
                    letter-spacing: 1px;
                }
                QLabel#cyantext {
                    font-weight: bold;
                    font-size: 14px;
                    color: #008ba3;
                    letter-spacing: 1px;
                }
                QLabel#InputLabel {
                    color: #4b627a;
                    font-size: 12px;
                    font-weight: bold;
                }
                QLabel#ProgressLabel {
                    color: #166a87;
                    font-size: 12px;
                    font-weight: bold;
                }
                QProgressBar#TaskProgress {
                    min-height: 16px;
                    border: 1px solid rgba(0, 160, 180, 0.55);
                    border-radius: 8px;
                    background: rgba(210, 235, 244, 0.85);
                    color: #13485e;
                    text-align: center;
                    font-size: 11px;
                    font-weight: bold;
                }
                QProgressBar#TaskProgress::chunk {
                    border-radius: 7px;
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #00abd0,
                        stop:0.5 #00c3d9,
                        stop:1 #35d6a4
                    );
                }
                QPushButton {
                    border-radius: 6px;
                    min-height: 36px;
                    padding: 6px 16px;
                    font-weight: bold;
                    font-size: 13px;
                    letter-spacing: 1px;
                }
                QPushButton#PrimaryBtn {
                    background: rgba(0, 195, 217, 0.15);
                    color: #008ba3;
                    border: 1px solid #00c3d9;
                }
                QPushButton#PrimaryBtn:hover { 
                    background: rgba(0, 195, 217, 0.3); 
                }
                QPushButton#PrimaryBtn:disabled {
                    background: rgba(200, 210, 220, 0.4);
                    border: 1px solid #aebac9;
                    color: #798b9e;
                }
                QPushButton#WarningBtn {
                    background: rgba(255, 0, 85, 0.1);
                    color: #cc0044;
                    border: 1px solid #ff0055;
                }
                QPushButton#WarningBtn:hover { background: rgba(255, 0, 85, 0.2); }
                QPushButton#WarningBtn:disabled {
                    background: rgba(200, 210, 220, 0.4);
                    border: 1px solid #aebac9;
                    color: #798b9e;
                }
                QPushButton#GhostBtn {
                    background: rgba(255, 255, 255, 0.5);
                    color: #3d5875;
                    border: 1px solid rgba(180, 200, 220, 0.6);
                }
                QPushButton#GhostBtn:hover { 
                    background: rgba(255, 255, 255, 0.8); 
                }
                QPushButton#GhostBtnMin {
                    background: transparent;
                    color: #008ba3;
                    border: 1px solid rgba(0, 195, 217, 0.6);
                    border-radius: 4px;
                    min-height: 24px;
                    padding: 4px 10px;
                    font-size: 11px;
                }
                QPushButton#GhostBtnMin:hover {
                    background: rgba(0, 195, 217, 0.1);
                }
                QPushButton#DangerHoldBtn {
                    background: rgba(255, 0, 85, 0.07);
                    color: #bf244e;
                    border: 1px solid rgba(200, 30, 70, 0.5);
                    border-radius: 4px;
                    min-height: 24px;
                    padding: 4px 10px;
                    font-size: 11px;
                }
                QPushButton#DangerHoldBtn:hover {
                    background: rgba(255, 0, 85, 0.15);
                    color: #9f1038;
                }
                QPushButton#DangerHoldBtn:disabled {
                    background: rgba(200, 210, 220, 0.4);
                    color: #7d8896;
                    border: 1px solid rgba(140, 150, 165, 0.45);
                }
                QPushButton#EyeBtn {
                    background: rgba(0, 195, 217, 0.05);
                    color: #008ba3;
                    border: 1px solid rgba(0, 195, 217, 0.3);
                    border-radius: 4px;
                    font-size: 12px;
                    padding: 0;
                    min-height: 24px;
                }
                QPushButton#EyeBtn:hover {
                    background: rgba(0, 195, 217, 0.2);
                    border: 1px solid rgba(0, 195, 217, 0.6);
                }
                QPushButton#EyeBtn:checked {
                    background: rgba(0, 195, 217, 0.3);
                    border: 1px solid #00c3d9;
                }
                QPushButton#HelpDotBtn {
                    min-height: 18px;
                    max-height: 18px;
                    min-width: 18px;
                    max-width: 18px;
                    padding: 0;
                    border-radius: 9px;
                    font-size: 11px;
                    font-weight: bold;
                    color: #008ba3;
                    background: rgba(0, 195, 217, 0.05);
                    border: 1px solid rgba(0, 195, 217, 0.3);
                }
                QPushButton#HelpDotBtn:hover {
                    background: rgba(0, 195, 217, 0.2);
                    border: 1px solid rgba(0, 195, 217, 0.6);
                }
                QPushButton#HelpDotBtn:disabled {
                    color: #7d8896;
                    background: rgba(200, 210, 220, 0.32);
                    border: 1px solid rgba(140, 150, 165, 0.45);
                }
                QListWidget#ListSurface,
                QTreeWidget#TreeSurface,
                QPlainTextEdit#LogSurface {
                    background: rgba(255, 255, 255, 0.6);
                    border: 1px solid rgba(255, 255, 255, 0.8);
                    border-radius: 6px;
                }
                QListWidget#ListSurface::item {
                    padding: 8px;
                    border-bottom: 1px solid rgba(0, 195, 217, 0.1);
                    color: #1a2a3a;
                }
                QListWidget#ListSurface::item:selected {
                    background: rgba(0, 195, 217, 0.15);
                    color: #0f1f33;
                    border-left: 3px solid #00c3d9;
                }
                QTreeWidget#TreeSurface {
                    alternate-background-color: rgba(0, 195, 217, 0.03);
                    color: #1a2a3a;
                }
                QTreeWidget#TreeSurface::item {
                    padding: 4px;
                }
                QTreeWidget#TreeSurface::item:selected {
                    background: rgba(0, 195, 217, 0.15);
                    color: #0f1f33;
                }
                QHeaderView::section {
                    background: rgba(255, 255, 255, 0.8);
                    color: #008ba3;
                    border: none;
                    border-bottom: 1px solid rgba(0, 195, 217, 0.3);
                    padding: 6px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPlainTextEdit#LogSurface {
                    font-family: "Consolas", "Cascadia Code";
                    font-size: 12px;
                    color: #2b4563;
                }
                QFrame#SwitchRow {
                    background: rgba(255, 255, 255, 0.5);
                    border: 1px solid rgba(255, 255, 255, 0.9);
                    border-radius: 6px;
                }
                QLabel#SwitchTitle {
                    font-weight: bold;
                    color: #1a2a3a;
                    font-size: 12px;
                }
                QLabel#SwitchDesc {
                    color: #5c7996;
                    font-size: 10px;
                }
                QLabel#ConfigGuide {
                    background: rgba(0, 195, 217, 0.05);
                    border: 1px solid rgba(0, 195, 217, 0.3);
                    border-left: 3px solid #00c3d9;
                    border-radius: 4px;
                    color: #3b5b75;
                    font-size: 11px;
                    padding: 4px 8px;
                }
                QComboBox#ConfigInput,
                QLineEdit#ConfigInput {
                    min-height: 26px;
                    background: rgba(255, 255, 255, 0.7);
                    border: 1px solid rgba(180, 200, 220, 0.8);
                    border-radius: 4px;
                    padding: 1px 8px;
                    color: #1a2a3a;
                    font-size: 12px;
                }
                QComboBox#ConfigInput::drop-down {
                    border: none;
                    width: 20px;
                }
                QComboBox#ConfigInput:focus,
                QLineEdit#ConfigInput:focus {
                    border: 1px solid #00c3d9;
                    background: rgba(255, 255, 255, 0.9);
                }
                QScrollArea#ConfigScroll {
                    border: none;
                    background: transparent;
                }
                QWidget#ConfigViewport {
                    background: rgba(255, 255, 255, 0.46);
                    border-radius: 6px;
                }
                QWidget#ConfigScrollContent {
                    background: transparent;
                }
                QScrollBar:vertical {
                    background: transparent;
                    width: 6px;
                    margin: 2px;
                }
                QScrollBar::handle:vertical {
                    background: rgba(0, 195, 217, 0.25);
                    min-height: 20px;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:hover {
                    background: rgba(0, 195, 217, 0.45);
                }
                """
            )

        if self._is_dark_theme:
            extra_css = """
            QFrame#StepCard {
                background: rgba(18, 24, 33, 0.92);
                border: 1px solid rgba(120, 138, 160, 0.35);
                border-radius: 10px;
            }
            QPushButton#StepBtn {
                min-height: 30px;
                padding: 4px 12px;
                border-radius: 8px;
                border: 1px solid rgba(108, 124, 145, 0.5);
                background: rgba(34, 42, 54, 0.85);
                color: #bdcadb;
                font-size: 12px;
            }
            QPushButton#StepBtn[activeStep="true"] {
                border: 1px solid rgba(97, 160, 255, 0.85);
                background: rgba(49, 96, 168, 0.35);
                color: #eaf2ff;
            }
            QFrame#Card[switchMode="all"] {
                border: 1px solid rgba(72, 196, 122, 0.75);
            }
            QFrame#Card[switchMode="none"] {
                border: 1px solid rgba(130, 139, 152, 0.6);
            }
            QFrame#Card[switchMode="mixed"] {
                border: 1px solid rgba(106, 149, 214, 0.72);
            }
            QFrame#BurstFeed {
                background: rgba(11, 16, 24, 0.72);
                border: 1px solid rgba(96, 111, 130, 0.35);
                border-radius: 8px;
            }
            QFrame#BurstItem_info {
                background: rgba(49, 109, 190, 0.22);
                border: 1px solid rgba(86, 151, 236, 0.42);
                border-radius: 8px;
            }
            QFrame#BurstItem_ok {
                background: rgba(50, 146, 92, 0.22);
                border: 1px solid rgba(84, 192, 129, 0.42);
                border-radius: 8px;
            }
            QFrame#BurstItem_warn {
                background: rgba(161, 118, 33, 0.24);
                border: 1px solid rgba(218, 165, 62, 0.45);
                border-radius: 8px;
            }
            QFrame#BurstItem_err {
                background: rgba(167, 66, 66, 0.24);
                border: 1px solid rgba(219, 104, 104, 0.45);
                border-radius: 8px;
            }
            QLabel#BurstText {
                color: #d8e4f4;
                font-size: 12px;
            }
            QFrame#OnboardingPanel {
                background: rgba(18, 24, 33, 0.97);
                border: 1px solid rgba(101, 167, 255, 0.6);
                border-radius: 12px;
            }
            QLabel#OnboardingTitle {
                color: #e8f0ff;
                font-size: 15px;
                font-weight: 600;
            }
            QLabel#OnboardingBody {
                color: #b8c6da;
                font-size: 12px;
            }
            """
        else:
            extra_css = """
            QFrame#StepCard {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(188, 199, 214, 0.88);
                border-radius: 10px;
            }
            QPushButton#StepBtn {
                min-height: 30px;
                padding: 4px 12px;
                border-radius: 8px;
                border: 1px solid rgba(183, 197, 212, 0.95);
                background: rgba(246, 249, 253, 0.96);
                color: #314862;
                font-size: 12px;
            }
            QPushButton#StepBtn[activeStep="true"] {
                border: 1px solid rgba(64, 134, 232, 0.95);
                background: rgba(212, 230, 255, 0.95);
                color: #103e74;
            }
            QFrame#Card[switchMode="all"] {
                border: 1px solid rgba(68, 174, 109, 0.86);
            }
            QFrame#Card[switchMode="none"] {
                border: 1px solid rgba(143, 152, 165, 0.86);
            }
            QFrame#Card[switchMode="mixed"] {
                border: 1px solid rgba(88, 139, 208, 0.86);
            }
            QFrame#BurstFeed {
                background: rgba(245, 249, 255, 0.94);
                border: 1px solid rgba(188, 199, 214, 0.88);
                border-radius: 8px;
            }
            QFrame#BurstItem_info {
                background: rgba(88, 152, 235, 0.18);
                border: 1px solid rgba(88, 152, 235, 0.42);
                border-radius: 8px;
            }
            QFrame#BurstItem_ok {
                background: rgba(77, 176, 116, 0.18);
                border: 1px solid rgba(77, 176, 116, 0.42);
                border-radius: 8px;
            }
            QFrame#BurstItem_warn {
                background: rgba(218, 164, 65, 0.19);
                border: 1px solid rgba(218, 164, 65, 0.42);
                border-radius: 8px;
            }
            QFrame#BurstItem_err {
                background: rgba(210, 86, 86, 0.2);
                border: 1px solid rgba(210, 86, 86, 0.45);
                border-radius: 8px;
            }
            QLabel#BurstText {
                color: #1d3653;
                font-size: 12px;
            }
            QFrame#OnboardingPanel {
                background: rgba(255, 255, 255, 0.98);
                border: 1px solid rgba(86, 147, 230, 0.9);
                border-radius: 12px;
            }
            QLabel#OnboardingTitle {
                color: #16395f;
                font-size: 15px;
                font-weight: 600;
            }
            QLabel#OnboardingBody {
                color: #355474;
                font-size: 12px;
            }
            """
        self.setStyleSheet(
            self.styleSheet()
            + extra_css
            + self._build_glass_override_css()
            + self._build_windows11_button_css()
        )

        self._apply_picture_background()
        self.drop_area.setProperty("darkMode", self._is_dark_theme)
        self.drop_area.set_accent_color(self._accent_color)
        self.drop_area.update()
        off_color = "#748290" if self._is_dark_theme else "#8f9bab"
        for switch in self.findChildren(ColorToggleButton):
            switch.setProperty("darkMode", self._is_dark_theme)
            switch.set_palette(self._accent_color.name(), off_color)
            switch.update()
        if self._tour_overlay is not None:
            self._tour_overlay.set_accent_color(self._accent_color)
        self._refresh_status_chip_style()
        self._apply_windows11_backdrop()
        self._sync_progress_pet()
        self._keep_aux_dialogs_in_view()
        if updates_enabled:
            self.setUpdatesEnabled(True)
        self.update()

    def _finish_card_intro_animation(
        self,
        card: QWidget,
        effect: QGraphicsOpacityEffect,
        anim: QPropertyAnimation,
    ) -> None:
        if card.graphicsEffect() is effect:
            card.setGraphicsEffect(None)
        if anim in self._animations:
            self._animations.remove(anim)

    def _animate_cards(self):
        delay = 0
        for card in self._cards:
            effect = QGraphicsOpacityEffect(card)
            effect.setOpacity(0.0)
            card.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(320)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.finished.connect(
                lambda c=card, e=effect, a=anim: self._finish_card_intro_animation(c, e, a)
            )
            self._animations.append(anim)
            QTimer.singleShot(delay, anim.start)
            delay += 45

    @staticmethod
    def _split_log_lines(text: str) -> List[str]:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines: List[str] = []
        for line in raw.split("\n"):
            clean = line.rstrip()
            if clean.strip():
                lines.append(clean)
        return lines

    @staticmethod
    def _extract_log_level(line: str) -> str:
        text = str(line or "").strip()
        if not text:
            return ""
        match = re.match(r"^\d{2}:\d{2}:\d{2}\s+\|\s*([A-Za-z]+)\s*\|", text)
        if not match:
            return ""
        return str(match.group(1) or "").strip().upper()

    @classmethod
    def _classify_log_line(cls, line: str) -> str:
        explicit_level = cls._extract_log_level(line)
        if explicit_level in {"CRITICAL", "ERROR", "EXCEPTION"}:
            return "err"
        if explicit_level in {"WARNING", "WARN"}:
            return "warn"
        if explicit_level in {"INFO", "DEBUG"}:
            return "info"

        upper = str(line or "").upper()
        if any(token in upper for token in ("CRITICAL", "ERROR", "EXCEPTION", "TRACEBACK", "FAILED")):
            return "err"
        if "WARNING" in upper or "WARN" in upper:
            return "warn"
        if any(token in upper for token in ("完成", "COMPLETE", "READY", "SUCCESS", "已完成")):
            return "ok"
        return "info"

    def append_log(self, text: str):
        lines = self._split_log_lines(str(text or ""))
        if not lines:
            return
        for clean in lines:
            self._emit_log_event_hint(clean)
            self._log_buffer.append(clean)
        if len(self._log_buffer) > 12000:
            while len(self._log_buffer) > 12000:
                self._log_buffer.popleft()
        debug_visible = bool(self.toggle_debug_log_button.isChecked()) and self._current_step == 3
        if debug_visible and not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _emit_log_event_hint(self, line: str) -> None:
        explicit_level = self._extract_log_level(line)
        upper = line.upper()
        if explicit_level in {"ERROR", "CRITICAL", "EXCEPTION"}:
            self._push_event(line.splitlines()[-1][-140:], "err")
        elif explicit_level in {"WARNING", "WARN"}:
            self._push_event(line.splitlines()[-1][-140:], "warn")
        elif not explicit_level and ("ERROR" in upper or "CRITICAL" in upper):
            self._push_event(line.splitlines()[-1][-140:], "err")
        elif not explicit_level and ("WARNING" in upper or "WARN" in upper):
            self._push_event(line.splitlines()[-1][-140:], "warn")

    def _flush_log_buffer(self, flush_all: bool = False) -> None:
        if not self._log_buffer:
            if self._log_flush_timer.isActive():
                self._log_flush_timer.stop()
            return

        debug_visible = bool(self.toggle_debug_log_button.isChecked()) and self._current_step == 3 and self.log_card.isVisible()
        if not flush_all and not debug_visible:
            if self._log_flush_timer.isActive():
                self._log_flush_timer.stop()
            return

        take = len(self._log_buffer) if flush_all else min(self._log_flush_limit, len(self._log_buffer))
        batch: List[str] = []
        for _ in range(max(0, int(take))):
            if not self._log_buffer:
                break
            batch.append(self._log_buffer.popleft())
        if not batch:
            return

        for line in batch:
            self.log_feed.push_event(line, level=self._classify_log_line(line), ttl_ms=0)

        if not self._log_buffer and self._log_flush_timer.isActive():
            self._log_flush_timer.stop()

    def _set_file_progress(
        self,
        done: int,
        total: int,
        label_text: str = "",
        file_percent: Optional[float] = None,
        overall_percent: Optional[float] = None,
    ):
        total = max(0, int(total))
        done = max(0, int(done))
        if total > 0:
            done = min(done, total)

        if file_percent is not None:
            try:
                self._current_file_percent = max(0.0, min(100.0, float(file_percent)))
            except (TypeError, ValueError):
                self._current_file_percent = 0.0
        elif total <= 0 or done >= total:
            self._current_file_percent = 0.0

        if overall_percent is None:
            if total > 0:
                active_ratio = 0.0 if done >= total else (self._current_file_percent / 100.0)
                overall_percent = ((done + active_ratio) / total) * 100.0
            else:
                overall_percent = 0.0
        try:
            self._overall_percent = max(0.0, min(100.0, float(overall_percent)))
        except (TypeError, ValueError):
            self._overall_percent = 0.0

        self._progress_total = total
        self._progress_done = done

        if label_text:
            self._last_file_progress_label = label_text

        if self._download_inflight:
            if label_text:
                self.progress_label.setText(label_text)
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(round(self._overall_percent)))

        if total <= 0:
            self.progress_bar.setFormat(f"{self._overall_percent:.1f}%")
        elif done >= total:
            self.progress_bar.setFormat(f"{self._overall_percent:.1f}%  |  已完成 {done}/{total}")
        else:
            self.progress_bar.setFormat(
                f"{self._overall_percent:.1f}%  |  已完成 {done}/{total}  |  当前文件 {self._current_file_percent:.1f}%"
            )
        if label_text:
            self.progress_label.setText(label_text)
        self._sync_progress_pet()

    def _set_download_busy(self, text: str, progress_percent: Optional[float] = None):
        if not self._download_inflight:
            self._download_started_at = time.monotonic()
            self._download_anim_frame = 0
            self._progress_pet_phase = 0.0
        self._download_inflight = True
        self._download_last_event_at = time.monotonic()
        self.progress_label.setText(text)
        if progress_percent is None:
            self._download_last_percent = None
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("模型下载中...")
            if not self._download_ui_timer.isActive():
                self._download_ui_timer.start()
            self._sync_progress_pet()
            return
        try:
            pct = max(0.0, min(100.0, float(progress_percent)))
        except (TypeError, ValueError):
            self._download_last_percent = None
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("模型下载中...")
            if not self._download_ui_timer.isActive():
                self._download_ui_timer.start()
            return
        self._download_last_percent = pct
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(round(pct)))
        self.progress_bar.setFormat(f"模型下载 {pct:.1f}%")
        if not self._download_ui_timer.isActive():
            self._download_ui_timer.start()
        self._sync_progress_pet()

    def _tick_download_progress_ui(self):
        if not self._download_inflight:
            if self._download_ui_timer.isActive():
                self._download_ui_timer.stop()
            return
        self._download_anim_frame = (self._download_anim_frame + 1) % 3
        dots = "." * (self._download_anim_frame + 1)
        elapsed_sec = 0.0
        if self._download_started_at > 0:
            elapsed_sec = max(0.0, time.monotonic() - self._download_started_at)

        if self._download_last_percent is None:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat(f"模型下载中{dots}  {elapsed_sec:.0f}s")
            return

        pct = max(0.0, min(100.0, float(self._download_last_percent)))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(round(pct)))
        self.progress_bar.setFormat(f"模型下载 {pct:.1f}%  |  {elapsed_sec:.0f}s")
        self._update_progress_pet_position()

    def _restore_file_progress_view(self, text: str = ""):
        self._download_inflight = False
        if self._download_ui_timer.isActive():
            self._download_ui_timer.stop()
        self._download_last_percent = None
        self._download_started_at = 0.0
        self._download_last_event_at = 0.0
        self._download_anim_frame = 0
        label = text or self._last_file_progress_label or self.progress_label.text()
        self._set_file_progress(self._progress_done, self._progress_total, label)
        self._sync_progress_pet()

    @Slot(int)
    def on_run_started(self, total: int):
        total = max(0, int(total))
        self._last_pipeline_step = ""
        self._set_file_progress(0, total, f"任务已启动，待处理文件: {total}", file_percent=0.0)
        self._apply_wizard_step(3, animate=True)
        if self.event_feed is not None:
            self.event_feed.clear()
        self._push_event(f"任务启动，已接收 {total} 个文件", "info", ttl_ms=22000)

    @Slot(dict)
    def on_model_download(self, payload: Dict[str, Any]):
        phase = str(payload.get("phase", "")).strip().lower()
        family = str(payload.get("family", "asr")).strip().lower()
        kind = str(payload.get("kind", "")).strip()
        repo = str(payload.get("repo", "model")).strip()
        source = str(payload.get("source", "")).strip().lower()
        endpoint = str(payload.get("endpoint", "")).strip()
        error_text = str(payload.get("error", "")).strip()
        attempt = int(payload.get("attempt", 0) or 0)
        total_attempts = int(payload.get("total_attempts", 0) or 0)
        progress_percent = payload.get("progress_percent")
        phase_detail = str(payload.get("phase_detail", "") or "").strip()
        try:
            elapsed_sec = float(payload.get("elapsed_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            elapsed_sec = 0.0

        source_text = "官方源"
        if source == "mirror":
            source_text = "国内镜像"
        elif source == "official":
            source_text = "官方源"
        elif source:
            source_text = source

        if progress_percent is None and phase == "done":
            progress_percent = 100.0

        family_text = "NeMo 模型" if family.startswith("nemo") else "ASR 模型"
        kind_text = f"[{kind}] " if kind else ""
        endpoint_text = f" | {endpoint}" if endpoint else ""
        attempt_text = f" {attempt}/{max(total_attempts, 1)}" if total_attempts > 0 else ""
        detail_text = f" | {phase_detail}" if phase_detail else ""

        if phase in {"start", "attempt", "retry"}:
            if phase == "start":
                msg = f"准备下载 {family_text}: {kind_text}{repo}"
            elif phase == "retry":
                msg = (
                    f"模型下载重试{attempt_text}: {kind_text}{repo} | "
                    f"{source_text}{endpoint_text}{detail_text}"
                )
            else:
                msg = (
                    f"正在下载模型{attempt_text}: {kind_text}{repo} | "
                    f"{source_text}{endpoint_text}{detail_text}"
                )
            self.append_log(f"[模型下载] {msg}")
            self._set_download_busy(msg, progress_percent=progress_percent)
            self._push_event(msg, "err" if phase == "retry" else "warn", ttl_ms=18000)
            return

        if phase == "done":
            elapsed_text = f"（{elapsed_sec:.1f}s）" if elapsed_sec > 0 else ""
            msg = f"{family_text}下载完成: {kind_text}{repo}{elapsed_text}"
            is_final_done = True
            try:
                if (
                    progress_percent is not None
                    and float(progress_percent) < 99.9
                    and total_attempts > 0
                ):
                    is_final_done = False
            except (TypeError, ValueError):
                is_final_done = True
            self.append_log(f"[模型下载] {msg}")
            if not is_final_done:
                self._set_download_busy(msg, progress_percent=progress_percent)
                self._push_event(msg, "info", ttl_ms=12000)
                return
            self._restore_file_progress_view(msg)
            self._push_event(msg, "ok", ttl_ms=20000)
            return

        if phase == "failed":
            err_tail = f" | {error_text[:120]}" if error_text else ""
            msg = f"{family_text}下载失败: {kind_text}{repo}{err_tail}"
            self.append_log(f"[模型下载] {msg}")
            self._restore_file_progress_view(msg)
            self._push_event(msg, "err", ttl_ms=26000)
            return

    def _status_chip_css(self, state: str) -> str:
        accent = QColor(self._accent_color)
        if not accent.isValid():
            accent = QColor("#4fa2ff")
        candy = self._mix_color(accent, QColor("#ffbfd6"), 0.52 if self._is_dark_theme else 0.64)
        info_fg = accent.lighter(132) if self._is_dark_theme else accent.darker(145)
        info_border = self._mix_color(accent, QColor("#ffffff"), 0.34) if self._is_dark_theme else self._mix_color(accent, QColor("#000000"), 0.22)
        if self._is_dark_theme:
            palette = {
                "ok": "background:rgba(105, 201, 151, 0.18);color:#dfffee;border:1px solid rgba(105, 201, 151, 0.46);",
                "warn": "background:rgba(255, 183, 105, 0.20);color:#fff0d5;border:1px solid rgba(255, 183, 105, 0.48);",
                "err": "background:rgba(255, 124, 154, 0.20);color:#ffe9f0;border:1px solid rgba(255, 124, 154, 0.48);",
                "info": (
                    f"background:{self._rgba(candy, 68)};"
                    f"color:{info_fg.name()};"
                    f"border:1px solid {self._rgba(info_border, 156)};"
                ),
            }
        else:
            palette = {
                "ok": "background:rgba(105, 201, 151, 0.14);color:#287a57;border:1px solid rgba(105, 201, 151, 0.40);",
                "warn": "background:rgba(255, 183, 105, 0.15);color:#8f5a18;border:1px solid rgba(255, 183, 105, 0.42);",
                "err": "background:rgba(255, 124, 154, 0.14);color:#a6375b;border:1px solid rgba(255, 124, 154, 0.44);",
                "info": (
                    f"background:{self._rgba(candy, 52)};"
                    f"color:{info_fg.name()};"
                    f"border:1px solid {self._rgba(info_border, 128)};"
                ),
            }
        return (
            palette.get(state, palette["info"])
            + "border-radius:14px;padding:7px 13px;font-weight:600;font-size:12px;letter-spacing:0.2px;"
        )

    def _refresh_status_chip_style(self):
        self.status_chip.setStyleSheet(self._status_chip_css(self._status_state))

    def set_status_chip(self, text: str, state: str = "info"):
        self.status_chip.setText(text)
        self._status_state = state
        self._refresh_status_chip_style()

    def _refresh_theme_visuals(self):
        if not hasattr(self, "status_chip"):
            return
        dark_now = self._is_dark_mode()
        accent_now = self._system_accent_color()
        accent_changed = int(accent_now.rgb()) != int(self._accent_color.rgb())
        if dark_now != self._is_dark_theme or accent_changed:
            self._apply_styles()
        else:
            self._refresh_status_chip_style()
            self._apply_windows11_backdrop()
        self._schedule_layout_sync(0)

    def changeEvent(self, event):
        event_type = event.type()
        theme_change = getattr(QEvent, "ThemeChange", None)
        watch_types = {
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
        }
        if theme_change is not None:
            watch_types.add(theme_change)

        if event_type in watch_types:
            self._theme_refresh_timer.start(0)
        elif event_type in {QEvent.WindowStateChange, QEvent.ActivationChange}:
            if event_type == QEvent.ActivationChange:
                self._theme_refresh_timer.start(0)
            self._apply_windows11_backdrop()
            if self.is_macos:
                self._ensure_macos_titlebar_tracking()
            self._schedule_layout_sync(0)
            self._keep_aux_dialogs_in_view()

        super().changeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self.is_macos:
            self._ensure_macos_titlebar_tracking()
            QTimer.singleShot(0, self._ensure_macos_titlebar_tracking)
        self._theme_refresh_timer.start(0)
        self._schedule_layout_sync(0)
        self._schedule_layout_sync(90)
        self._sync_progress_pet()
        self._keep_aux_dialogs_in_view()
        if self._tour_pending and not self._tour_shown:
            QTimer.singleShot(260, self._maybe_show_first_run_tour)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_layout_sync(22)
        self._update_progress_pet_position()
        self._keep_aux_dialogs_in_view()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._schedule_layout_sync(18)
        self._keep_aux_dialogs_in_view()

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择多媒体文件",
            "",
            "音视频文件 (*.mp4 *.mkv *.mov *.avi *.wmv *.flv *.webm *.mp3 *.wav *.flac *.m4a *.aac *.ogg)",
        )
        if files:
            self.add_files(files)

    @Slot(list)
    def add_files(self, files: List[str]):
        added = 0
        for raw in files:
            p = Path(raw).resolve()
            if not p.exists() or not p.is_file():
                continue
            if p.suffix.lower() not in ALL_MEDIA_EXTENSIONS:
                continue
            key = str(p)
            if key in self.file_items:
                continue
            item = QListWidgetItem(self._queue_item_text(key))
            item.setData(Qt.UserRole, key)
            self.file_list.addItem(item)
            self.file_items[key] = item
            self.selected_files.append(key)
            added += 1

        if added > 0:
            for key, item in self.file_items.items():
                item.setText(self._queue_item_text(key))
            self.set_status_chip(f"已排队 {len(self.selected_files)} 个文件", "info")
            self._push_event(f"已添加 {added} 个文件（队列共 {len(self.selected_files)} 个）", "ok")
            self._refresh_action_button_states()

    def _selected_queue_keys(self) -> List[str]:
        selected: List[str] = []
        for item in self.file_list.selectedItems():
            key = str(item.data(Qt.UserRole) or "").strip()
            if key and key in self.file_items:
                selected.append(key)
        return selected

    def remove_selected_queue_items(self):
        if self.running:
            QMessageBox.information(self, "系统繁忙", "任务正在运行中，无法删除队列项。")
            return
        selected_keys = self._selected_queue_keys()
        if not selected_keys:
            self.set_status_chip("请先选中要删除的任务", "info")
            return

        removed_set = set(selected_keys)
        for key in selected_keys:
            item = self.file_items.pop(key, None)
            if not item:
                continue
            row = self.file_list.row(item)
            if row >= 0:
                self.file_list.takeItem(row)

        self.selected_files = [fp for fp in self.selected_files if fp not in removed_set]
        for key, item in self.file_items.items():
            item.setText(self._queue_item_text(key))
        self.set_status_chip(f"已删除 {len(selected_keys)} 个选中任务", "info")
        self._push_event(f"已删除 {len(selected_keys)} 个队列文件", "warn")
        self._refresh_action_button_states()

    def remove_checked_queue_items(self):
        self.remove_selected_queue_items()

    def clear_queue(self):
        if self.running:
            QMessageBox.information(self, "系统繁忙", "任务正在运行中，无法清空队列。")
            return
        self._clear_queue_internal()
        self.set_status_chip("待处理队列已清空", "info")
        self._push_event("待处理队列已清空", "warn")
        self._refresh_action_button_states()

    def _clear_queue_internal(self) -> int:
        count = len(self.selected_files)
        self.file_list.clear()
        self.file_items.clear()
        self.selected_files.clear()
        return count

    def _resolve_log_file_path(self) -> Path:
        configured = str(_get_nested(self.config._data, "logging.log_file", "pipeline.log") or "pipeline.log")
        p = resolve_app_writable_path(configured, kind="log")
        return p.resolve()

    @staticmethod
    def _normalize_fs_path(path: Path) -> str:
        return os.path.normcase(os.path.abspath(str(path)))

    def _collect_log_file_handler_bindings(self, log_path: Path) -> List[Tuple[logging.Logger, logging.Handler]]:
        target = self._normalize_fs_path(log_path)
        bindings: List[Tuple[logging.Logger, logging.Handler]] = []
        logger_candidates: List[logging.Logger] = [logging.getLogger()]
        logger_candidates.extend(
            logger_obj
            for logger_obj in logging.root.manager.loggerDict.values()
            if isinstance(logger_obj, logging.Logger)
        )
        for logger_obj in logger_candidates:
            for handler in list(logger_obj.handlers):
                base_filename = getattr(handler, "baseFilename", "")
                if not base_filename:
                    continue
                try:
                    handler_path = self._normalize_fs_path(Path(str(base_filename)))
                except Exception:
                    continue
                if handler_path == target:
                    bindings.append((logger_obj, handler))
        return bindings

    @staticmethod
    def _force_remove_log_file(log_path: Path) -> bool:
        try:
            if log_path.exists():
                log_path.unlink()
            return not log_path.exists()
        except Exception:
            pass

        try:
            tomb = log_path.with_name(f"{log_path.stem}.deleting.{os.getpid()}{log_path.suffix}")
            os.replace(log_path, tomb)
            try:
                tomb.unlink()
            except Exception:
                pass
            return not log_path.exists()
        except Exception:
            pass

        try:
            with open(log_path, "w", encoding="utf-8"):
                pass
            return True
        except Exception:
            pass

        if os.name == "nt" and log_path.exists():
            try:
                movefile_delay_until_reboot = 0x4
                ok = ctypes.windll.kernel32.MoveFileExW(
                    str(log_path),
                    None,
                    movefile_delay_until_reboot,
                )
                if ok != 0:
                    return True
            except Exception:
                pass

        return not log_path.exists()

    def delete_pipeline_log(self, silent: bool = False) -> bool:
        if not self._try_acquire_action_lock("delete_log"):
            return False
        try:
            log_path = self._resolve_log_file_path()
            self.log_feed.clear()
            self._log_buffer.clear()
            self._log_flush_timer.stop()
            if not log_path.exists():
                if not silent:
                    self.set_status_chip("未找到日志文件", "info")
                return False

            bindings = self._collect_log_file_handler_bindings(log_path)
            removed = False
            try:
                for logger_obj, handler in bindings:
                    if handler in logger_obj.handlers:
                        logger_obj.removeHandler(handler)
                    try:
                        handler.flush()
                    except Exception:
                        pass
                    try:
                        handler.close()
                    except Exception:
                        pass

                removed = self._force_remove_log_file(log_path)
                if not silent:
                    if removed:
                        self.set_status_chip("日志已强制清理", "ok")
                    else:
                        self.set_status_chip("日志正在被占用，已尝试强制清理", "warn")
            finally:
                for logger_obj, handler in bindings:
                    if handler not in logger_obj.handlers:
                        logger_obj.addHandler(handler)
            return removed
        finally:
            self._release_action_lock("delete_log")

    def delete_temp_files(self, silent: bool = False) -> Dict[str, Any]:
        if not self._try_acquire_action_lock("delete_temp"):
            return {"removed_count": 0, "removed_bytes": 0, "failed_count": 0}
        if self.running and not self.paused:
            try:
                QMessageBox.information(self, "系统繁忙", "任务运行中，请先暂停任务再清理临时文件。")
                return {"removed_count": 0, "removed_bytes": 0, "failed_count": 0}
            finally:
                self._release_action_lock("delete_temp")

        try:
            removed_count = 0
            removed_bytes = 0
            failed_count = 0

            def try_remove_file(path: Path):
                nonlocal removed_count, removed_bytes, failed_count
                try:
                    if not path.exists() or not path.is_file():
                        return
                    size = path.stat().st_size
                    path.unlink()
                    removed_count += 1
                    removed_bytes += int(size)
                except Exception:
                    failed_count += 1

            temp_dir = Path(tempfile.gettempdir())
            for wav_path in temp_dir.glob("audio_*.wav"):
                try_remove_file(wav_path)

            output_root = self._resolve_output_dir()
            resume_dir = resolve_runtime_artifact_path(output_root, "resume")
            if resume_dir.exists():
                entries = sorted(
                    resume_dir.rglob("*"),
                    key=lambda p: len(p.parts),
                    reverse=True,
                )
                for entry in entries:
                    if entry.is_file():
                        try_remove_file(entry)
                for entry in entries:
                    if entry.is_dir():
                        try:
                            entry.rmdir()
                        except OSError:
                            pass
                try:
                    resume_dir.rmdir()
                except OSError:
                    pass

            if removed_count > 0:
                size_mb = removed_bytes / (1024 * 1024)
                if not silent:
                    self.set_status_chip(
                        f"已清理临时文件 {removed_count} 个（约 {size_mb:.1f} MB）",
                        "ok",
                    )
            elif failed_count > 0:
                if not silent:
                    self.set_status_chip("临时文件正在占用，清理未完成", "warn")
            else:
                if not silent:
                    self.set_status_chip("未发现可清理的临时文件", "info")

            return {
                "removed_count": removed_count,
                "removed_bytes": removed_bytes,
                "failed_count": failed_count,
            }
        finally:
            self._release_action_lock("delete_temp")

    def delete_all_result_items(self, silent: bool = False) -> Dict[str, int]:
        if not self._try_acquire_action_lock("delete_all_results"):
            return {"removed_count": 0, "failed_count": 0}
        try:
            removed_count = 0
            failed_count = 0
            output_root = self._resolve_output_dir()
            if output_root.exists():
                for entry in sorted(output_root.iterdir(), key=lambda p: p.name.lower()):
                    if entry.name.startswith(".") or entry.name == RUNTIME_ARTIFACTS_DIRNAME:
                        continue
                    try:
                        if entry.is_dir():
                            shutil.rmtree(entry)
                        else:
                            entry.unlink()
                        removed_count += 1
                    except Exception:
                        failed_count += 1

            self.result_tree.clear()
            self._update_result_action_state()

            if not silent:
                if removed_count > 0 and failed_count == 0:
                    self.set_status_chip(f"已删除全部产物项 {removed_count} 个", "ok")
                elif removed_count > 0:
                    self.set_status_chip(
                        f"已删除产物项 {removed_count} 个，失败 {failed_count} 个",
                        "warn",
                    )
                elif failed_count > 0:
                    self.set_status_chip("产物删除失败，请查看占用情况", "warn")
                else:
                    self.set_status_chip("未发现可删除的处理结果", "info")
            return {"removed_count": removed_count, "failed_count": failed_count}
        finally:
            self._release_action_lock("delete_all_results")

    def run_full_cleanup(self):
        if not self._try_acquire_action_lock("full_cleanup"):
            return
        try:
            if self.running:
                QMessageBox.information(self, "系统繁忙", "任务正在运行中，无法执行全清理。")
                return

            selected_removed = len(self._selected_queue_keys())
            queue_removed = self._clear_queue_internal()
            temp_stats = self.delete_temp_files(silent=True)
            log_removed = self.delete_pipeline_log(silent=True)
            result_stats = self.delete_all_result_items(silent=True)
            self.log_feed.clear()

            temp_removed = int(temp_stats.get("removed_count", 0) or 0)
            temp_failed = int(temp_stats.get("failed_count", 0) or 0)
            result_removed = int(result_stats.get("removed_count", 0) or 0)
            result_failed = int(result_stats.get("failed_count", 0) or 0)
            status_bits = [
                f"队列清理 {queue_removed} 项（选中 {selected_removed}）",
                f"临时文件 {temp_removed} 项",
                f"处理结果 {result_removed} 项",
                "日志已清空" if log_removed else "日志未找到/未完全清空",
            ]
            if temp_failed > 0 or result_failed > 0:
                status_bits.append(f"失败 {temp_failed + result_failed} 项")

            self.set_status_chip("全清理完成：" + "；".join(status_bits), "ok")
            self._push_event("全清理已完成", "ok")
        finally:
            self._release_action_lock("full_cleanup")

    @staticmethod
    def _common_input_root(files: List[str]) -> str:
        if not files:
            return str(Path.cwd())
        parents = [str(Path(f).resolve().parent) for f in files]
        try:
            return os.path.commonpath(parents)
        except ValueError:
            return str(Path(files[0]).resolve().parent)

    def _on_worker_status_message(self, message: str) -> None:
        self.set_status_chip(message, "warn")
        self._push_event(message, "warn")

    def start_processing(self):
        if self.running:
            return
        if not self.selected_files:
            QMessageBox.warning(self, "提示", "请先添加至少一个需要转录的文件。")
            return

        self._apply_wizard_step(3, animate=True)
        if self.toggle_debug_log_button.isChecked():
            self.toggle_debug_log_button.setChecked(False)
        self.log_card.setVisible(False)
        self.log_feed.clear()
        self._log_buffer.clear()
        self._log_flush_timer.stop()
        if self.event_feed is not None:
            self.event_feed.clear()
        self.set_status_chip("正在启动处理...", "warn")
        self._push_event("正在初始化处理流水线...", "info", ttl_ms=18000)
        self._download_inflight = False
        self._set_file_progress(0, len(self.selected_files), "正在初始化流水线...", file_percent=0.0)

        self._apply_editor_settings_to_config()
        try:
            self._persist_modified_config_fields()
        except Exception as e:
            self.append_log(f"警告：无法保存配置到磁盘: {e}")
            self.set_status_chip("当前为临时运行模式", "warn")
            self._push_event("配置保存失败，已按当前临时配置继续执行", "warn")

        if not self._posterior_auto_cycle_phase:
            if self.posterior_dump_toggle_button.isChecked() and self._posterior_auto_run_twice_enabled():
                self._posterior_auto_cycle_first_pass_calibrator_path = ""
                self._posterior_auto_cycle_phase = "first_pass"
                self.append_log(
                    "Posterior Fusion 自动双跑已启用：第一遍将导出样本，随后自动训练并重跑当前队列。"
                )
                self._push_event("Posterior Fusion 自动双跑：开始第一遍处理", "info", ttl_ms=18000)
        elif self._posterior_auto_cycle_phase == "second_pass":
            self.append_log("Posterior Fusion 自动双跑：开始第二遍处理。")
            self._push_event("Posterior Fusion 自动双跑：开始第二遍处理", "warn", ttl_ms=18000)
            try:
                self._clear_posterior_auto_cycle_first_pass_artifacts()
            except Exception as e:
                self.append_log(f"警告：清理第一遍输出失败，将继续第二遍处理: {e}")

        config_data = copy.deepcopy(self.config._data)
        input_root = self._common_input_root(self.selected_files)
        run_options = self._posterior_auto_cycle_run_options()

        self.worker_thread = QThread(self)
        self.worker = PipelineWorker(
            config_data=config_data,
            files=list(self.selected_files),
            input_root=input_root,
            run_options=run_options,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self.append_log, Qt.QueuedConnection)
        self.worker.status_message.connect(self._on_worker_status_message)
        self.worker.run_started.connect(self.on_run_started)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_step.connect(self.on_file_step)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.model_download.connect(self.on_model_download)
        self.worker.run_finished.connect(self.on_run_finished)
        self.worker.run_failed.connect(self.on_run_failed)

        self.worker_thread.start()

        self.running = True
        self.paused = False
        self.pause_button.setText("暂停任务")
        self._refresh_action_button_states()
        self._sync_progress_pet()

    def _shutdown_worker(self, request_stop: bool = False, timeout_ms: int = 4000) -> bool:
        if self.worker is not None and request_stop:
            try:
                self.worker.request_stop()
            except Exception:
                pass

        thread = self.worker_thread
        if thread is not None and thread.isRunning():
            timeout_ms = max(500, int(timeout_ms))
            deadline = time.time() + (timeout_ms / 1000.0)
            while thread.isRunning() and time.time() < deadline:
                thread.wait(180)
            if thread.isRunning():
                thread.quit()
                thread.wait(1800)

        if thread is not None and thread.isRunning():
            return False

        if thread is not None:
            thread.deleteLater()
            self.worker_thread = None
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        return True

    def _cleanup_runtime_caches_on_exit(self) -> None:
        try:
            temp_dir = Path(tempfile.gettempdir())
            now = time.time()
            for wav_path in temp_dir.glob("audio_*.wav"):
                try:
                    if not wav_path.is_file():
                        continue
                    # Avoid touching files that could still be in use by a just-stopped run.
                    if (now - float(wav_path.stat().st_mtime)) < 30.0:
                        continue
                    wav_path.unlink()
                except Exception:
                    continue
        except Exception:
            pass

        try:
            smart_empty_cache(force=True)
        except Exception:
            pass
        gc.collect()

    @Slot(int, int, str)
    def on_file_started(self, index: int, total: int, file_path: str):
        name = self._queue_display_name(file_path)
        self.set_status_chip(f"[{index}/{total}] 处理中: {name}", "warn")
        self._push_event(f"[{index}/{total}] 开始处理：{name}", "info", ttl_ms=18000)
        self._set_file_progress(
            max(0, int(index) - 1),
            max(0, int(total)),
            f"处理中: {name}",
            file_percent=0.0,
        )
        item = self.file_items.get(str(Path(file_path).resolve()))
        if item:
            item.setText(self._queue_item_text(file_path, "正在处理"))

    @Slot(dict)
    def on_file_step(self, payload: Dict[str, Any]):
        file_path = str(payload.get("file_path", ""))
        step = str(payload.get("step", "")).strip().lower()
        phase = str(payload.get("phase", "start")).strip().lower()
        file_percent = payload.get("file_percent")
        overall_percent = payload.get("overall_percent")
        try:
            step_index = int(payload.get("step_index", 0) or 0)
        except (TypeError, ValueError):
            step_index = 0
        try:
            step_total = int(payload.get("step_total", 0) or 0)
        except (TypeError, ValueError):
            step_total = 0
        try:
            chunk_index = int(payload.get("chunk_index", 0) or 0)
        except (TypeError, ValueError):
            chunk_index = 0
        try:
            chunk_total = int(payload.get("chunk_total", 0) or 0)
        except (TypeError, ValueError):
            chunk_total = 0
        phase_detail = str(payload.get("phase_detail", "") or "").strip()

        step_map = {
            "extract": "提取音频中",
            "preprocess": "预处理中",
            "transcribe": "AI语音识别中",
            "translate": "文本翻译中",
            "optimize_language": "语言优化中",
            "write_text": "写入文本文件",
            "report": "生成图文报告",
            "render_video": "字幕回写视频中",
        }
        name = self._queue_display_name(file_path)
        mapped = step_map.get(step, step)
        if step == "transcribe" and chunk_total > 0 and chunk_index > 0:
            mapped = f"{mapped}（分块 {chunk_index}/{chunk_total}）"
        suffix = ""
        try:
            if file_percent is not None:
                suffix = f" · {float(file_percent):.1f}%"
        except (TypeError, ValueError):
            suffix = ""
        prefix = f"[{step_index}/{step_total}] " if step_total > 0 and step_index > 0 else ""
        if phase_detail:
            mapped = f"{mapped}（{phase_detail}）"

        step_advanced = (
            bool(step)
            and phase in {"start", "resume"}
            and step != self._last_pipeline_step
        )
        if step_advanced:
            self._last_pipeline_step = step
            if self._download_inflight:
                self.append_log(
                    "[模型下载] 检测到流程进入下一步骤，已强制恢复任务进度显示。"
                )
                self._restore_file_progress_view(f"{name}：{mapped}")

        level = "err" if phase in {"failed", "error"} else "warn"
        self.set_status_chip(f"{name} | {prefix}{mapped}{suffix}", level)
        self._set_file_progress(
            self._progress_done,
            self._progress_total,
            f"{name}：{mapped}",
            file_percent=file_percent if file_percent is not None else None,
            overall_percent=overall_percent if overall_percent is not None else None,
        )
        if phase in {"start", "resume", "skip", "retry", "failed", "error"} and step in {
            "extract",
            "preprocess",
            "transcribe",
            "translate",
            "optimize_language",
            "write_text",
            "report",
            "render_video",
        }:
            ttl_ms = 20000 if phase in {"failed", "error"} else 16000
            self._push_event(
                f"{name}：{prefix}{mapped}",
                "err" if phase in {"failed", "error"} else "warn",
                ttl_ms=ttl_ms,
            )

    @Slot(dict)
    def on_file_finished(self, result: Dict[str, Any]):
        if self._progress_total <= 0:
            self._progress_total = max(1, len(self.selected_files))
        completed = min(self._progress_done + 1, self._progress_total)
        self._set_file_progress(
            completed,
            self._progress_total,
            f"已完成 {completed}/{self._progress_total}",
            file_percent=0.0,
        )

        source_path = str(Path(result.get("source_path", "")).resolve())
        source_file = result.get("source_file", "未知文件")
        status = result.get("status", "未知状态")
        if status == "OK":
            status = "处理成功"

        item = self.file_items.get(source_path)
        if item:
            item.setText(self._queue_item_text(source_path, status))
        self._push_event(f"{self._queue_display_name(source_path)}：{status}", "ok" if status == "处理成功" else "warn")

        top = QTreeWidgetItem([self._queue_display_name(source_path), status, ""])
        top.setExpanded(True)

        open_target = ""
        for key, label in self.OUTPUT_KEYS:
            file_path = result.get(key)
            if not file_path:
                continue
            p = Path(file_path)
            if not open_target:
                open_target = str(p.parent)
            suffix = p.suffix.lower()
            child_status = self.OUTPUT_KIND_BY_SUFFIX.get(suffix, "文件")
            child = QTreeWidgetItem([f"{label}: {p.name}", child_status, str(p)])
            child.setData(0, Qt.UserRole, str(p))
            top.addChild(child)

        if open_target:
            top.setData(0, Qt.UserRole, open_target)
            top.setText(2, open_target)

        self.result_tree.addTopLevelItem(top)
        self._update_result_action_state()

    @Slot(dict)
    def on_run_finished(self, payload: Dict[str, Any]):
        self._flush_log_buffer(flush_all=False)
        ok = int(payload.get("ok", 0))
        total = int(payload.get("total", len(self.selected_files)))
        self._last_pipeline_step = ""
        self._download_inflight = False
        self._download_last_event_at = 0.0
        self._set_file_progress(total, total, f"全部完成: 成功 {ok}/{total}", file_percent=100.0)
        self.set_status_chip(f"全部完成：成功 {ok} / 总数 {total}", "ok")
        self._push_event(f"处理完成：成功 {ok}/{total}", "ok", ttl_ms=26000)
        self.running = False
        self.paused = False
        self.pause_button.setText("暂停任务")
        self._refresh_action_button_states()
        self._shutdown_worker()
        if self._posterior_auto_cycle_phase == "first_pass":
            self.append_log("Posterior Fusion 自动双跑：第一遍处理完成，开始训练校准器。")
            self.set_status_chip("Posterior Fusion 自动双跑：开始训练校准器...", "warn")
            self._push_event("Posterior Fusion 自动双跑：开始训练校准器", "warn", ttl_ms=22000)
            self._posterior_auto_cycle_phase = "training"
            QTimer.singleShot(180, lambda: self.start_posterior_fusion_training(automated=True))
        elif self._posterior_auto_cycle_phase == "second_pass":
            self._posterior_auto_cycle_phase = ""
            self._posterior_auto_cycle_first_pass_calibrator_path = ""
            self.append_log("Posterior Fusion 自动双跑已完成。")
            self.set_status_chip("Posterior Fusion 自动双跑已完成", "ok")
            self._push_event("Posterior Fusion 自动双跑已完成", "ok", ttl_ms=24000)

    @staticmethod
    def _friendly_error_text(trace_text: str) -> str:
        t = (trace_text or "").lower()
        if "out of memory" in t or "cuda out of memory" in t or "cublas_status_alloc_failed" in t:
            return "内存不足，建议在步骤 2 降低并发文件数或关闭部分增强功能后重试。"
        if "permissionerror" in t or "access is denied" in t:
            return "没有足够权限访问文件，请关闭占用程序后再试。"
        if "filenotfounderror" in t or "no such file" in t:
            return "部分输入或输出路径不存在，请检查文件是否被移动或删除。"
        if "connection" in t or "timeout" in t or "dns" in t:
            return "网络连接异常，模型下载或在线服务调用失败，请检查网络后重试。"
        if "ffmpeg" in t:
            return "视频处理依赖不可用或执行失败，请检查 ffmpeg 配置后重试。"
        return "处理过程中出现异常，请根据关键进度和调试日志排查。"

    @Slot(str)
    def on_run_failed(self, trace_text: str):
        auto_cycle_phase = self._posterior_auto_cycle_phase
        self.append_log(trace_text)
        self._flush_log_buffer(flush_all=False)
        self._last_pipeline_step = ""
        self._download_inflight = False
        self._download_last_event_at = 0.0
        if self._progress_total > 0:
            self._set_file_progress(
                self._progress_done,
                self._progress_total,
                f"执行失败: 已完成 {self._progress_done}/{self._progress_total}",
                file_percent=0.0,
            )
        else:
            self._set_file_progress(0, 0, "执行失败", file_percent=0.0)
        friendly = self._friendly_error_text(trace_text)
        self.set_status_chip(friendly, "err")
        self._push_event(friendly, "err", ttl_ms=28000)
        QMessageBox.critical(self, "处理失败", friendly)
        self.running = False
        self.paused = False
        self.pause_button.setText("暂停任务")
        self._refresh_action_button_states()
        self._shutdown_worker()
        if auto_cycle_phase:
            self._abort_posterior_auto_cycle("处理流程失败", level="err")

    def toggle_pause(self):
        if not self.running or not self.worker:
            return
        if self._pause_toggle_cooldown:
            return
        self._pause_toggle_cooldown = True
        self.pause_button.setEnabled(False)
        if not self.paused:
            self.worker.pause()
            self.paused = True
            self.pause_button.setText("恢复任务")
            self.set_status_chip("正在挂起...", "warn")
            self._push_event("任务已暂停", "warn")
        else:
            self.worker.resume()
            self.paused = False
            self.pause_button.setText("暂停任务")
            self.set_status_chip("正在恢复处理...", "warn")
            self._push_event("任务已恢复", "info")
        QTimer.singleShot(180, self._release_pause_toggle_guard)

    def _release_pause_toggle_guard(self):
        self._pause_toggle_cooldown = False
        self._refresh_action_button_states()

    def open_item_path(self, item: QTreeWidgetItem, _column: int):
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(self, "文件丢失", f"找不到目标路径：\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def closeEvent(self, event):
        try:
            self._persist_modified_config_fields()
        except Exception:
            pass
        if self.advanced_dialog is not None:
            self.advanced_dialog.close()
        if self.posterior_dialog is not None:
            self.posterior_dialog.close()
        if self.usage_help_dialog is not None:
            self.usage_help_dialog.close()
        if self._tour_overlay is not None:
            self._tour_overlay.hide()
        if self.posterior_trainer_thread is not None and self.posterior_trainer_thread.isRunning():
            QMessageBox.warning(
                self,
                "训练仍在进行",
                "Posterior Fusion 校准器仍在训练中，请等待完成后再关闭程序。",
            )
            event.ignore()
            return
        if self.running and self.worker:
            reply = QMessageBox.question(
                self,
                "Task Running",
                "A task is still running.\n\n"
                "Stop now and exit? Progress checkpoints will be kept for resume.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.set_status_chip("正在停止任务并清理缓存...", "warn")
            stopped = self._shutdown_worker(request_stop=True, timeout_ms=14000)
            if not stopped:
                QMessageBox.warning(
                    self,
                    "正在停止任务",
                    "后台任务仍在收尾，请稍候几秒再关闭程序。",
                )
                event.ignore()
                return
        else:
            stopped = self._shutdown_worker(request_stop=False, timeout_ms=1800)
            if not stopped:
                QMessageBox.warning(
                    self,
                    "稍后重试",
                    "后台线程仍在退出中，请稍候再关闭程序。",
                )
                event.ignore()
                return
        self._cleanup_runtime_caches_on_exit()
        super().closeEvent(event)


def _set_windows_app_user_model_id(app_id: str) -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        pass


def _hide_console_window_if_present() -> None:
    if os.name != "nt":
        return
    try:
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


def _resolve_qt_icon() -> Optional[QIcon]:
    candidates: List[Path] = [
        APP_ROOT / "assets" / "app.icns",
        APP_ROOT / "assets" / "app.png",
        APP_ROOT / "app.ico",
        APP_ROOT / "app.icns",
        APP_ROOT / "app.png",
        APP_ROOT / "assets" / "app.ico",
        APP_ROOT / "_internal" / "app.icns",
        APP_ROOT / "_internal" / "app.png",
        APP_ROOT / "_internal" / "app.ico",
    ]
    if getattr(sys, "frozen", False):
        try:
            # Prefer packaged .ico resources; use executable icon as last fallback.
            candidates.append(Path(sys.executable))
        except Exception:
            pass
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


def _default_ui_font_family() -> str:
    if sys.platform == "darwin":
        return "PingFang SC"
    if os.name == "nt":
        return "Microsoft YaHei UI"
    return "Noto Sans CJK SC"


def launch_ui_app(config_path: Optional[str] = None):
    _hide_console_window_if_present()
    _set_windows_app_user_model_id("MediaTranscribeStudio.Main")
    _install_qt_message_filter()
    rounding_policy = getattr(Qt, "HighDpiScaleFactorRoundingPolicy", None)
    if rounding_policy is not None:
        try:
            QApplication.setHighDpiScaleFactorRoundingPolicy(rounding_policy.PassThrough)
        except Exception:
            pass
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("AI 语音转录工作站")
    try:
        app.setApplicationDisplayName("AI 语音转录工作站")
    except Exception:
        pass
    app.setFont(QFont(_default_ui_font_family(), 10))
    icon = _resolve_qt_icon()
    if icon is not None:
        try:
            app.setWindowIcon(icon)
        except Exception:
            pass
    window = MainWindow(config_path=config_path)
    if icon is not None:
        try:
            window.setWindowIcon(icon)
        except Exception:
            pass
    window.show()
    app.exec()

if __name__ == "__main__":
    launch_ui_app()
