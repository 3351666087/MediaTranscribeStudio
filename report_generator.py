"""
report_generator.py - HTML/PDF报告生成器

PDF引擎优先级：
  1. Playwright (Chromium内核) - 浏览器打印级别质量
  2. pdfkit (wkhtmltopdf) - 降级方案

特性：
  - 说话人颜色动态生成（HSL色环，支持任意数量）
  - 不硬编码任何说话人标签
  - 精美时间轴CSS
  - 可选嵌入LLM分析摘要
  - Chrome级PDF渲染质量
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from output_layout import resolve_output_subdir
from runtime_paths import APP_ROOT
from utils import format_timestamp

logger = logging.getLogger(__name__)
_PLAYWRIGHT_ENV_CONFIGURED = False


def _hidden_subprocess_kwargs() -> dict:
    if os.name != "nt":
        return {}
    kwargs = {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = startupinfo
    except Exception:
        pass
    return kwargs


def _configure_playwright_runtime_env() -> None:
    """
    Prefer a bundled Playwright browser cache inside the frozen onedir package.

    This prevents runtime fallback to user-profile cache and preserves Playwright
    priority over wkhtmltopdf when Chromium was packed into the payload.
    """
    global _PLAYWRIGHT_ENV_CONFIGURED
    if _PLAYWRIGHT_ENV_CONFIGURED:
        return
    _PLAYWRIGHT_ENV_CONFIGURED = True

    def _has_browser_payload(browser_dir: Path) -> bool:
        try:
            if not browser_dir.exists():
                return False
            for child in browser_dir.iterdir():
                if child.is_dir() and child.name.startswith(
                    ("chromium-", "chromium_headless_shell-", "chrome-", "msedge-")
                ):
                    return True
        except Exception:
            return False
        return False

    def _package_local_browser_dirs() -> List[Path]:
        try:
            import playwright
        except Exception:
            return []

        pkg_root = Path(playwright.__file__).resolve().parent
        return [
            pkg_root / "driver" / "package" / ".local-browsers",
            pkg_root / ".local-browsers",
        ]

    env_browser_path = str(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "") or "").strip()
    if env_browser_path and env_browser_path != "0":
        env_dir = Path(env_browser_path).expanduser()
        if _has_browser_payload(env_dir):
            os.environ.setdefault("PLAYWRIGHT_SKIP_BROWSER_GC", "1")
            logger.info(f"Playwright browser cache detected from env: {env_dir}")
            return

    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(
            APP_ROOT / "_internal" / "playwright" / "driver" / "package" / ".local-browsers"
        )
    candidates.append(APP_ROOT / "playwright" / "driver" / "package" / ".local-browsers")
    candidates.extend(_package_local_browser_dirs())
    if sys.platform == "darwin":
        candidates.append(Path.home() / "Library" / "Caches" / "ms-playwright")
    else:
        candidates.append(Path.home() / ".cache" / "ms-playwright")

    for browser_dir in candidates:
        if not _has_browser_payload(browser_dir):
            continue
        target = str(browser_dir)
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = target
        os.environ.setdefault("PLAYWRIGHT_SKIP_BROWSER_GC", "1")
        logger.info(f"Playwright bundled browser cache detected: {browser_dir}")
        return


# ═══════════════════════════════════════════════════════════════════════════
# 动态颜色生成器
# ═══════════════════════════════════════════════════════════════════════════

def generate_speaker_color(index: int, total: int) -> Dict[str, str]:
    """
    基于HSL色环为每个说话人生成独特颜色。
    黄金角度分布确保视觉区分度。
    """
    golden_angle = 137.508
    hue = (index * golden_angle) % 360
    saturation = 65
    lightness = 48

    h = hue / 360
    s = saturation / 100
    l_val = lightness / 100

    r, g, b = _hsl_to_rgb(h, s, l_val)
    hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    r_l, g_l, b_l = _hsl_to_rgb(h, s * 0.4, 0.95)
    light_color = f"#{int(r_l*255):02x}{int(g_l*255):02x}{int(b_l*255):02x}"

    return {
        "hex": hex_color,
        "hsl": f"hsl({int(hue)}, {saturation}%, {lightness}%)",
        "light": light_color,
        "hue": hue,
    }


def _hsl_to_rgb(h: float, s: float, l_val: float):
    if s == 0:
        return l_val, l_val, l_val

    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l_val * (1 + s) if l_val < 0.5 else l_val + s - l_val * s
    p = 2 * l_val - q

    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)
    return r, g, b


def build_speaker_styles(speakers: List[str]) -> Dict[str, Dict[str, str]]:
    """为所有说话人动态生成颜色映射"""
    color_map = {}
    total = len(speakers)
    for i, spk in enumerate(speakers):
        color = generate_speaker_color(i, total)
        safe_class = f"spk-{i}"
        color["css_class"] = safe_class
        color["label"] = spk
        color_map[spk] = color
    return color_map


def build_dynamic_css(speaker_styles: Dict[str, Dict]) -> str:
    """构建动态说话人颜色CSS规则"""
    dot_rules = []
    speaker_rules = []
    for spk, style in speaker_styles.items():
        cls = style["css_class"]
        hex_c = style["hex"]
        dot_rules.append(f"        .dot-{cls} {{ background: {hex_c}; }}")
        speaker_rules.append(
            f"        .speaker-{cls} {{ color: {hex_c}; }}"
        )
    return "\n".join(dot_rules + speaker_rules)


# ═══════════════════════════════════════════════════════════════════════════
# HTML 模板
# ═══════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{filename}</title>
    <style>
        @page {{
            size: A4;
            margin: 15mm;
        }}
        body {{
            font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC",
                         "Segoe UI", sans-serif;
            background-color: #f4f6f9;
            color: #333;
            margin: 0;
            padding: 40px;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #fff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 22px;
        }}
        h2 {{
            color: #34495e;
            font-size: 17px;
            margin-top: 30px;
            margin-bottom: 12px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .header-divider {{
            border: none;
            border-bottom: 2px solid #eaeaea;
            margin-bottom: 25px;
        }}
        .meta-info {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 14px;
            color: #666;
            gap: 10px;
        }}
        .meta-item {{
            text-align: center;
            min-width: 80px;
        }}
        .meta-item strong {{
            display: block;
            font-size: 18px;
            color: #2c3e50;
            margin-top: 4px;
        }}
        .analysis-box {{
            background: #fafbfc;
            border: 1px solid #e8eaed;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            font-size: 14px;
            line-height: 1.7;
        }}
        .analysis-box .summary {{
            color: #2c3e50;
            margin-bottom: 12px;
        }}
        .analysis-box ul {{
            margin: 5px 0;
            padding-left: 20px;
        }}
        .analysis-box li {{
            margin-bottom: 4px;
            color: #444;
        }}
        .speaker-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: center;
            margin-bottom: 20px;
            padding: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            color: #555;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .timeline {{
            position: relative;
            padding: 20px 0;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            top: 0;
            bottom: 0;
            left: 100px;
            width: 2px;
            background: #e0e0e0;
        }}
        .entry {{
            display: flex;
            margin-bottom: 25px;
            position: relative;
            page-break-inside: avoid;
        }}
        .time-box {{
            width: 85px;
            text-align: right;
            padding-right: 30px;
            flex-shrink: 0;
        }}
        .timestamp {{
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 14px;
            color: #666;
            background: #f4f6f9;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
            margin-right: 2px;
        }}
        .dot {{
            position: absolute;
            left: 95px;
            top: 6px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 3px solid #fff;
            box-shadow: 0 0 0 1px #e0e0e0;
            z-index: 1;
        }}
        .content-box {{
            flex-grow: 1;
            padding-left: 20px;
        }}
        .speaker {{
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 4px;
            letter-spacing: 0.5px;
        }}
        .text {{
            font-size: 16px;
            line-height: 1.6;
            color: #2c3e50;
            text-align: justify;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            font-size: 12px;
            color: #aaa;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }}
{dynamic_css}
        @media print {{
            body {{
                background: #fff;
                padding: 0;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .container {{
                box-shadow: none;
                border-radius: 0;
                padding: 20px;
            }}
            .entry {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ {title}</h1>
        <hr class="header-divider">
        <div class="meta-info">
            <div class="meta-item">时长<strong>{duration}</strong></div>
            <div class="meta-item">说话人<strong>{num_speakers}</strong></div>
            <div class="meta-item">段落<strong>{num_segments}</strong></div>
            <div class="meta-item">语言<strong>{language}</strong></div>
        </div>
{speaker_legend}
{analysis_section}
        <div class="timeline">
{content}
        </div>
        <div class="footer">
            Generated by AI Transcription Pipeline | {date}
        </div>
    </div>
</body>
</html>"""

ENTRY_TEMPLATE = """            <div class="entry">
                <div class="time-box">
                    <span class="timestamp">{start}</span>
                </div>
                <div class="dot dot-{css_class}"></div>
                <div class="content-box">
                    <div class="speaker speaker-{css_class}">{speaker}</div>
                    <div class="text">{text}</div>
                </div>
            </div>"""

LANG_NAMES = {
    "zh": "中文", "en": "English", "yue": "粤语",
    "ja": "日本語", "ko": "한국어", "fr": "Français",
    "de": "Deutsch", "es": "Español", "ru": "Русский",
}


# ═══════════════════════════════════════════════════════════════════════════
# PDF 引擎抽象
# ═══════════════════════════════════════════════════════════════════════════

class PDFEngine:
    """PDF生成引擎基类"""
    name: str = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def generate(
        self, html_content: str, pdf_path: Path, options: dict
    ) -> bool:
        raise NotImplementedError


class PlaywrightPDFEngine(PDFEngine):
    """
    Playwright (Chromium) PDF引擎

    优势：
      - 最新 Chromium 内核，CSS 渲染完美
      - 中文字体、Flexbox、CSS Grid 全支持
      - 打印颜色保留（-webkit-print-color-adjust: exact）
      - 页眉页脚、页边距精确控制
    """

    name = "playwright"

    def __init__(self):
        self._playwright_module = None
        self._checked = False
        self._available = False

    def _ensure_chromium_ready(self) -> bool:
        global _PLAYWRIGHT_ENV_CONFIGURED
        _PLAYWRIGHT_ENV_CONFIGURED = False
        _configure_playwright_runtime_env()
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return False

        def _can_launch() -> bool:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-gpu"],
                )
                browser.close()
            return True

        try:
            return _can_launch()
        except Exception as e:
            msg = str(e).lower()
            need_install = (
                "executable doesn't exist" in msg
                or "download new browsers" in msg
                or "playwright install" in msg
            )
            if not need_install:
                logger.debug(f"Playwright launch check failed: {e}")
                return False

            logger.info("Chromium not found for Playwright, installing automatically...")
            install_error = None

            # Frozen app cannot act like `python -m playwright`; invoke Playwright CLI entrypoint in-process.
            try:
                import playwright.__main__ as pw_main
                argv_backup = list(sys.argv)
                pw_env_backup = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
                try:
                    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")
                    os.environ.setdefault("PLAYWRIGHT_SKIP_BROWSER_GC", "1")
                    sys.argv = ["playwright", "install", "chromium"]
                    pw_main.main()
                except SystemExit as se:
                    code = int(se.code) if isinstance(se.code, int) else 0
                    if code not in (0, None):
                        raise RuntimeError(f"playwright install exited with code {code}") from se
                finally:
                    sys.argv = argv_backup
                    if pw_env_backup is None:
                        os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
                    else:
                        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = pw_env_backup
            except Exception as e_inproc:
                install_error = e_inproc
                # Development/non-frozen fallback path.
                try:
                    install_env = os.environ.copy()
                    # In development/build environments, install into playwright package-local cache
                    # so PyInstaller can collect it via driver/package/.local-browsers.
                    install_env.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")
                    install_env.setdefault("PLAYWRIGHT_SKIP_BROWSER_GC", "1")
                    subprocess.run(
                        [sys.executable, "-m", "playwright", "install", "chromium"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=900,
                        env=install_env,
                        **_hidden_subprocess_kwargs(),
                    )
                    install_error = None
                except Exception as e_subproc:
                    install_error = e_subproc

            if install_error is not None:
                logger.warning(f"Chromium auto-install failed: {install_error}")
                return False

            try:
                _PLAYWRIGHT_ENV_CONFIGURED = False
                _configure_playwright_runtime_env()
                return _can_launch()
            except Exception as retry_err:
                logger.warning(f"Chromium launch still failed after install: {retry_err}")
                return False

    def is_available(self) -> bool:
        if self._checked:
            return self._available

        self._checked = True
        try:
            # 检查 playwright 包
            import importlib.util
            if importlib.util.find_spec("playwright.sync_api") is None:
                raise ImportError("playwright.sync_api not found")
            self._playwright_module = True
            self._available = self._ensure_chromium_ready()
            if self._available:
                logger.info("Playwright (Chromium) PDF engine: available")
            return self._available

        except ImportError:
            logger.info(
                "Playwright not installed. "
                "Install for Chrome-quality PDF:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
            return False
        except Exception as e:
            logger.debug(f"Playwright check failed: {e}")
            return False

    def generate(
        self, html_content: str, pdf_path: Path, options: dict
    ) -> bool:
        """
        使用 Playwright Chromium 生成 PDF

        完全同步执行，不依赖外部事件循环。
        每次生成都启动/关闭浏览器，确保无资源泄漏。
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return False

        if not self._ensure_chromium_ready():
            return False

        margin = {
            "top": options.get("margin_top", "15mm"),
            "bottom": options.get("margin_bottom", "15mm"),
            "left": options.get("margin_left", "15mm"),
            "right": options.get("margin_right", "15mm"),
        }

        page_format = options.get("page_size", "A4")

        # 页眉页脚模板（可选）
        header_template = options.get("header_template", "")
        footer_template = options.get(
            "footer_template",
            '<div style="font-size:9px; color:#aaa; '
            'text-align:center; width:100%;">'
            '<span class="pageNumber"></span> / '
            '<span class="totalPages"></span>'
            '</div>',
        )
        display_header_footer = bool(
            options.get("display_header_footer", False)
        )
        print_background = bool(options.get("print_background", True))

        try:
            with sync_playwright() as p:
                # 启动 chromium（headless）
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-gpu",
                        "--disable-dev-shm-usage",
                        "--font-render-hinting=none",
                    ],
                )

                try:
                    page = browser.new_page()

                    # 设置内容，等待渲染完成
                    page.set_content(
                        html_content,
                        wait_until="networkidle",
                        timeout=30000,
                    )

                    # 等待字体加载
                    page.wait_for_timeout(500)

                    # 生成 PDF
                    pdf_kwargs = {
                        "path": str(pdf_path),
                        "format": page_format,
                        "margin": margin,
                        "print_background": print_background,
                        "prefer_css_page_size": False,
                    }

                    if display_header_footer:
                        pdf_kwargs["display_header_footer"] = True
                        if header_template:
                            pdf_kwargs["header_template"] = (
                                header_template
                            )
                        pdf_kwargs["footer_template"] = footer_template

                    page.pdf(**pdf_kwargs)

                finally:
                    browser.close()

            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                size_kb = pdf_path.stat().st_size / 1024
                logger.info(
                    f"PDF generated via Playwright (Chromium): "
                    f"{size_kb:.0f}KB"
                )
                return True
            else:
                logger.warning("Playwright PDF: output file empty")
                return False

        except Exception as e:
            logger.warning(f"Playwright PDF failed: {e}")
            # 清理失败产物
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except OSError:
                    pass
            return False


class PdfkitPDFEngine(PDFEngine):
    """pdfkit (wkhtmltopdf) PDF引擎 - 降级方案"""

    name = "pdfkit"

    def __init__(self):
        self._pdfkit = None
        self._config = None
        self._checked = False
        self._available = False

    def is_available(self) -> bool:
        if self._checked:
            return self._available

        self._checked = True
        try:
            import pdfkit
            self._pdfkit = pdfkit

            wk_path = self._find_wkhtmltopdf()
            if wk_path:
                self._config = pdfkit.configuration(
                    wkhtmltopdf=wk_path
                )
                self._available = True
                logger.info(f"pdfkit PDF engine: available ({wk_path})")
            else:
                logger.info(
                    "wkhtmltopdf not found. "
                    "Install the system wkhtmltopdf binary and ensure it is on PATH."
                )
            return self._available

        except ImportError:
            logger.info("pdfkit not installed (pip install pdfkit)")
            return False

    @staticmethod
    def _find_wkhtmltopdf() -> Optional[str]:
        found = shutil.which("wkhtmltopdf")
        if found:
            return found
        if sys.platform == "win32":
            for p in [
                r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
                r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
            ]:
                if Path(p).exists():
                    return p
        return None

    def generate(
        self, html_content: str, pdf_path: Path, options: dict
    ) -> bool:
        if not self._available or not self._pdfkit:
            return False

        display_header_footer = bool(
            options.get("display_header_footer", False)
        )
        print_background = bool(options.get("print_background", True))

        pdfkit_options = {
            "page-size": options.get("page_size", "A4"),
            "margin-top": options.get("margin_top", "15mm"),
            "margin-bottom": options.get("margin_bottom", "15mm"),
            "margin-left": options.get("margin_left", "15mm"),
            "margin-right": options.get("margin_right", "15mm"),
            "encoding": "UTF-8",
            "enable-local-file-access": "",
            "print-media-type": "",
            "quiet": "",
        }
        if print_background:
            pdfkit_options["background"] = ""
        else:
            pdfkit_options["no-background"] = ""

        if display_header_footer:
            pdfkit_options["footer-center"] = "[page] / [topage]"
            pdfkit_options["footer-font-size"] = "9"
            pdfkit_options["footer-spacing"] = "4"

        try:
            kwargs = {"options": pdfkit_options}
            if self._config:
                kwargs["configuration"] = self._config

            self._pdfkit.from_string(
                html_content, str(pdf_path), **kwargs
            )

            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                size_kb = pdf_path.stat().st_size / 1024
                logger.info(
                    f"PDF generated via pdfkit (wkhtmltopdf): "
                    f"{size_kb:.0f}KB"
                )
                return True
            return False

        except Exception as e:
            logger.warning(f"pdfkit PDF failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# 主报告生成器
# ═══════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    HTML/PDF报告生成器

    PDF引擎优先级：
      1. Playwright (Chromium) - 浏览器打印级别质量
      2. pdfkit (wkhtmltopdf) - 降级方案
    """

    def __init__(self, config):
        self.config = config
        self.report_cfg = config.get("report", {})
        if isinstance(self.report_cfg, type(None)):
            self.report_cfg = {}
        self.ts_fmt = config.get("output", {}).get("timestamp_format", "HH:MM:SS.mmm")
        self.output_dir = Path(config["paths"]["output_dir"])

        self.generate_html = self.report_cfg.get("generate_html", True)
        self.generate_pdf = self.report_cfg.get("generate_pdf", True)
        self.pdf_options = self.report_cfg.get("pdf_options", {})

        # 初始化 PDF 引擎（按优先级）
        self._pdf_engine: Optional[PDFEngine] = None
        if self.generate_pdf:
            self._init_pdf_engine()

    def resolve_output_subdir(self, source_file: Path, input_dir: Path) -> Path:
        return resolve_output_subdir(
            output_root=self.output_dir,
            source_file=source_file,
            input_dir=input_dir,
        )

    def _init_pdf_engine(self):
        """按优先级初始化 PDF 引擎"""
        # 用户可在配置中强制指定引擎
        preferred = self.pdf_options.get("engine", "auto")

        engines = []
        if preferred == "playwright":
            engines = [PlaywrightPDFEngine()]
        elif preferred in ("pdfkit", "wkhtmltopdf"):
            engines = [PdfkitPDFEngine()]
        else:
            # auto: 优先 Playwright
            engines = [PlaywrightPDFEngine(), PdfkitPDFEngine()]

        for engine in engines:
            if engine.is_available():
                self._pdf_engine = engine
                logger.info(f"PDF engine: {engine.name}")
                return

        logger.warning(
            "No PDF engine available. PDF generation disabled.\n"
            "  Option 1 (recommended): "
            "pip install playwright && playwright install chromium\n"
            "  Option 2: pip install pdfkit + install wkhtmltopdf"
        )
        self.generate_pdf = False

    # ═══════════════════════════════════════════════════════════════════════
    # 主生成入口
    # ═══════════════════════════════════════════════════════════════════════

    def generate(
        self,
        segments: List,
        source_file: Path,
        input_dir: Path,
        duration: float = 0.0,
        metadata: Optional[Dict] = None,
        analysis=None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """生成HTML和PDF报告"""
        result = {}
        metadata = metadata or {}

        output_subdir = (
            Path(output_dir)
            if output_dir is not None
            else self.resolve_output_subdir(source_file=source_file, input_dir=input_dir)
        )
        output_subdir.mkdir(parents=True, exist_ok=True)
        stem = source_file.stem

        # 渲染 HTML
        html_content = self._render_html(
            segments, source_file.name, duration, metadata, analysis
        )

        # 写 HTML
        if self.generate_html:
            html_path = output_subdir / f"{stem}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            result["html"] = html_path
            logger.info(f"HTML: {html_path.name}")

        # 生成 PDF
        if self.generate_pdf and self._pdf_engine:
            pdf_path = output_subdir / f"{stem}.pdf"
            try:
                ok = self._pdf_engine.generate(
                    html_content, pdf_path, self.pdf_options
                )
                if ok:
                    result["pdf"] = pdf_path
                else:
                    logger.warning(
                        f"PDF generation returned False for "
                        f"{pdf_path.name}"
                    )
            except Exception as e:
                logger.warning(f"PDF generation failed: {e}")

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # HTML 渲染
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_report_duration(
        duration: float,
        segments: List,
        metadata: Dict,
    ) -> float:
        """Resolve a non-zero duration for report metadata."""
        try:
            dur = float(duration or 0.0)
        except (TypeError, ValueError):
            dur = 0.0
        if dur > 0:
            return dur

        md_duration = metadata.get("duration_sec", metadata.get("duration", 0.0))
        try:
            md_duration = float(md_duration or 0.0)
        except (TypeError, ValueError):
            md_duration = 0.0
        if md_duration > 0:
            return md_duration

        max_end = 0.0
        for seg in segments:
            try:
                end_val = float(getattr(seg, "end", 0.0) or 0.0)
            except (TypeError, ValueError):
                end_val = 0.0
            if end_val > max_end:
                max_end = end_val
        return max_end

    def _render_html(
        self,
        segments: List,
        filename: str,
        duration: float,
        metadata: Dict,
        analysis=None,
    ) -> str:
        """渲染完整HTML"""
        duration = self._resolve_report_duration(duration, segments, metadata)

        # 1. 收集说话人（按出场顺序）
        speakers = []
        seen = set()
        detected_langs = set()

        for seg in segments:
            spk = seg.speaker.strip() if seg.speaker else "Unknown"
            if spk not in seen:
                speakers.append(spk)
                seen.add(spk)
            if hasattr(seg, "language") and seg.language:
                detected_langs.add(seg.language)

        # 2. 动态颜色
        speaker_styles = build_speaker_styles(speakers)
        dynamic_css = build_dynamic_css(speaker_styles)

        # 3. 说话人图例
        legend_html = self._render_legend(speakers, speaker_styles)

        # 4. LLM分析（如果有）
        analysis_html = self._render_analysis(analysis)

        # 5. 时间轴条目
        entries = []
        for seg in segments:
            spk = seg.speaker.strip() if seg.speaker else "Unknown"
            style = speaker_styles.get(spk, {"css_class": "spk-0"})
            entry = ENTRY_TEMPLATE.format(
                start=format_timestamp(seg.start, self.ts_fmt),
                css_class=style["css_class"],
                speaker=_escape(spk),
                text=_escape(seg.text),
            )
            entries.append(entry)

        content = "\n".join(entries)

        # 6. 语言
        language_str = self._resolve_report_language(metadata, detected_langs)

        # 7. 组装
        html = HTML_TEMPLATE.format(
            filename=_escape(filename),
            title=_escape(filename),
            duration=format_timestamp(duration, self.ts_fmt),
            num_speakers=len(speakers),
            num_segments=len(segments),
            language=language_str,
            dynamic_css=dynamic_css,
            speaker_legend=legend_html,
            analysis_section=analysis_html,
            content=content,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        return html

    @staticmethod
    def _resolve_report_language(metadata: Dict, detected_langs: set) -> str:
        def _normalize_lang(value: object) -> str:
            text = str(value or "").strip()
            if not text:
                return ""
            if text.lower() in {"auto", "unknown", "none", "null"}:
                return ""
            return text

        preferred = _normalize_lang(
            metadata.get("language", metadata.get("translated_language", ""))
        )
        if not preferred:
            preferred = _normalize_lang(metadata.get("pre_detected_language", ""))
        if preferred:
            return LANG_NAMES.get(preferred, preferred)

        if detected_langs:
            lang_parts = [LANG_NAMES.get(lang, lang) for lang in sorted(detected_langs)]
            return " / ".join(lang_parts)

        return "Auto"

    @staticmethod
    def _render_legend(
        speakers: List[str], styles: Dict
    ) -> str:
        if len(speakers) <= 1:
            return ""

        items = []
        for spk in speakers:
            color = styles.get(spk, {}).get("hex", "#95a5a6")
            items.append(
                f'            <div class="legend-item">'
                f'<div class="legend-dot" '
                f'style="background:{color}"></div>'
                f"<span>{_escape(spk)}</span>"
                f"</div>"
            )

        return (
            '        <div class="speaker-legend">\n'
            + "\n".join(items)
            + "\n        </div>"
        )

    @staticmethod
    def _render_analysis(analysis) -> str:
        if analysis is None:
            return ""

        parts = []

        if analysis.summary:
            parts.append("            <h2>📋 摘要</h2>")
            parts.append(
                f'            <div class="summary">'
                f"{_escape(analysis.summary)}</div>"
            )

        if analysis.key_points:
            parts.append("            <h2>🔑 关键要点</h2>")
            parts.append("            <ul>")
            for pt in analysis.key_points:
                parts.append(
                    f"                <li>{_escape(pt)}</li>"
                )
            parts.append("            </ul>")

        if analysis.action_items:
            parts.append("            <h2>✅ 待办事项</h2>")
            parts.append("            <ul>")
            for item in analysis.action_items:
                parts.append(
                    f"                <li>{_escape(item)}</li>"
                )
            parts.append("            </ul>")

        if analysis.topics:
            parts.append("            <h2>💬 讨论主题</h2>")
            parts.append("            <ul>")
            for topic in analysis.topics:
                parts.append(
                    f"                <li>{_escape(topic)}</li>"
                )
            parts.append("            </ul>")

        if not parts:
            return ""

        return (
            '        <div class="analysis-box">\n'
            + "\n".join(parts)
            + "\n        </div>"
        )


def _escape(text: str) -> str:
    """HTML转义"""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
