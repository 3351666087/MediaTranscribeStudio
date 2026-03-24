from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_MACOS_PYTHON = PROJECT_ROOT / ".venv-macos" / "bin" / "python"
SIGNABLE_BUNDLE_SUFFIXES = {".app", ".framework", ".xpc", ".appex", ".plugin", ".bundle", ".kext"}
SIGNABLE_FILE_SUFFIXES = {".dylib", ".so"}
MACHO_MAGIC_VALUES = {
    0xFEEDFACE,
    0xCEFAEDFE,
    0xFEEDFACF,
    0xCFFAEDFE,
    0xCAFEBABE,
    0xBEBAFECA,
    0xCAFEBABF,
    0xBFBACEFA,
}


def ensure_preferred_python_runtime() -> None:
    if sys.platform != "darwin":
        return

    preferred = PREFERRED_MACOS_PYTHON
    if not preferred.exists():
        return
    if os.environ.get("MTS_BUILD_MACOS_REEXECED") == "1":
        return

    try:
        current = Path(sys.executable).resolve()
        preferred_resolved = preferred.resolve()
    except Exception:
        return

    if current == preferred_resolved:
        return

    print(f"Re-launching build with project Python: {preferred_resolved}")
    relaunched_env = os.environ.copy()
    relaunched_env["MTS_BUILD_MACOS_REEXECED"] = "1"
    os.execve(
        str(preferred_resolved),
        [str(preferred_resolved), str(Path(__file__).resolve()), *sys.argv[1:]],
        relaunched_env,
    )


def prepare_hf_upload_environment() -> str:
    try:
        import upload_hf_dmg as hf_upload
    except Exception:
        return "missing"

    try:
        _, source = hf_upload.resolve_upload_token(interactive=False)
    except Exception:
        return "missing"
    return source or "missing"

from one_click_build import (
    DIST_DIR,
    ROOT,
    build_app_pyinstaller_env,
    dir_size_bytes,
    human_bytes,
    load_dist_config,
    playwright_package_local_browsers_dir,
    prepare_bundled_playwright_chromium,
    prompt_url_with_default,
    pyinstaller_cmd,
    run,
    sha256sum,
    update_dist_config_macos_defaults,
)


def author_name(cfg=None) -> str:
    for candidate in (
        getattr(cfg, "APP_AUTHOR", None) if cfg is not None else None,
        os.environ.get("APP_AUTHOR"),
        os.environ.get("APP_PUBLISHER"),
        "MediaTranscribeStudio Team",
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return "MediaTranscribeStudio Team"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def release_version() -> str:
    raw = str(os.environ.get("RELEASE_VERSION", "")).strip()
    if raw:
        return raw
    return datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M")


def macos_offline_bundle_enabled() -> bool:
    raw = os.environ.get("INCLUDE_RUNTIME_MODEL_CACHES")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def model_cache_bundle_mode() -> str:
    raw = str(os.environ.get("MODEL_CACHE_BUNDLE_MODE", "") or "").strip().lower()
    if raw in {"full", "curated", "minimal", "selected"}:
        return raw
    if macos_offline_bundle_enabled():
        return "full"
    return "full"


def _setdefault_bundle_env(env: dict[str, str], name: str, values: list[str]) -> None:
    if env.get(name):
        return
    clean = [str(value or "").strip() for value in values if str(value or "").strip()]
    if clean:
        env[name] = os.pathsep.join(clean)


def apply_default_curated_model_bundle_env(env: dict[str, str]) -> None:
    """
    Bundle only the models needed for the default macOS feature set.

    This keeps the offline package materially smaller than mirroring the
    developer machine's entire HF/ModelScope/Torch caches while still allowing
    zero-download first run on a fresh machine.
    """
    _setdefault_bundle_env(
        env,
        "BUNDLE_HF_MODEL_REPOS",
        [
            "nvidia/diar_streaming_sortformer_4spk-v2.1",
            "Systran/faster-whisper-medium",
            "Systran/faster-whisper-small",
            "Systran/faster-whisper-tiny",
            "mlx-community/whisper-large-v3-mlx",
            "mlx-community/whisper-large-v3-turbo",
            "mlx-community/whisper-small-mlx",
            "pyannote/speaker-diarization-community-1",
            "pyannote/speaker-diarization-3.1",
            "pyannote/overlapped-speech-detection",
            "pyannote/speech-separation-ami-1.0",
            "microsoft/wavlm-large",
            "speechbrain/spkrec-ecapa-voxceleb",
            "alibabasglab/MossFormer2_SS_16K",
            "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0",
        ],
    )
    _setdefault_bundle_env(
        env,
        "BUNDLE_MODELSCOPE_CACHE_SUBDIRS",
        [
            "hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "hub/models/iic/speech_campplus_sv_zh-cn_16k-common",
            "hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        ],
    )
    _setdefault_bundle_env(
        env,
        "BUNDLE_TORCH_CACHE_SUBDIRS",
        [
            "pyannote/models--pyannote--speaker-diarization-community-1",
            "pyannote/models--pyannote--speaker-diarization-3.1",
            "pyannote/models--pyannote--overlapped-speech-detection",
            "pyannote/models--pyannote--speech-separation-ami-1.0",
            "pyannote/models--pyannote--separation-ami-1.0",
            "pyannote/models--pyannote--segmentation-3.0",
            "pyannote/models--pyannote--segmentation",
            "pyannote/models--pyannote--wespeaker-voxceleb-resnet34-LM",
            "pyannote/speechbrain",
        ],
    )

    env["MODEL_CACHE_BUNDLE_MODE"] = model_cache_bundle_mode()


def bundle_identifier(app_name: str) -> str:
    explicit = str(os.environ.get("MACOS_BUNDLE_ID", "")).strip()
    if explicit:
        return explicit
    normalized = "".join(ch for ch in app_name.lower() if ch.isalnum())
    return f"cn.wisemodel.{normalized}"


def codesign_identity() -> str:
    return str(
        os.environ.get("MAC_CODESIGN_IDENTITY")
        or os.environ.get("CODESIGN_IDENTITY")
        or ""
    ).strip()


def installer_identity() -> str:
    return str(
        os.environ.get("MAC_INSTALLER_IDENTITY")
        or os.environ.get("INSTALLER_IDENTITY")
        or ""
    ).strip()


def macos_dmg_payload_kind() -> str:
    raw = str(os.environ.get("MACOS_DMG_PAYLOAD", "") or "").strip().lower()
    if raw in {"pkg", "app"}:
        return raw
    return "app"


def keep_bootstrapper_app_artifact() -> bool:
    return _env_flag("KEEP_MACOS_BOOTSTRAPPER_APP", False)


def keep_macos_intermediate_artifacts() -> bool:
    return _env_flag("KEEP_MACOS_INTERMEDIATES", False)


def keep_macos_hosted_payload_artifact() -> bool:
    return _env_flag("KEEP_MACOS_HOSTED_PAYLOAD", True)


def _ask_reuse_existing_app_bundle(app_bundle: Path) -> bool:
    if not app_bundle.exists():
        return False

    override = str(os.environ.get("MACOS_REUSE_DIST_APP", "") or "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False

    if not sys.stdin.isatty():
        print("\n== Existing macOS App Bundle Detected ==")
        print(f"App bundle      : {app_bundle.resolve()}")
        print("Selection       : reuse existing app bundle (non-interactive default)")
        return True

    print("\n== Existing macOS App Bundle Detected ==")
    print(f"App bundle      : {app_bundle.resolve()}")
    print("Press Enter to skip rebuild and continue from signing/package/upload.")
    print("Type n and press Enter to rebuild the app from scratch.")
    while True:
        raw = input("Reuse this dist app bundle? [Y/n]: ").strip().lower()
        if raw in {"", "y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please press Enter to reuse the existing app bundle, or type n to rebuild.")


def can_reuse_existing_app_bundle(app_bundle: Path) -> tuple[bool, str]:
    try:
        validate_app_bundle(app_bundle)
    except Exception as exc:
        return False, str(exc)
    return True, ""


SUPPORTED_MACOS_HOSTED_PAYLOAD_EXTENSIONS = (
    ".tar.gz",
    ".tgz",
    ".dmg",
    ".zip",
    ".tar",
)


def macos_adhoc_sign_enabled() -> bool:
    return _env_flag("MACOS_ADHOC_SIGN", True)


def macos_signing_mode_label(sign_identity: str) -> str:
    if sign_identity:
        return "configured identity"
    if macos_adhoc_sign_enabled():
        return "ad-hoc"
    return "disabled"


def macos_hosted_payload_format(sign_identity: str) -> str:
    raw = str(os.environ.get("MACOS_HOSTED_PAYLOAD_FORMAT", "") or "").strip().lower()
    if raw in {"dmg", "zip"}:
        return raw
    return "zip"


def macos_auto_upload_hosted_payload() -> bool:
    return _env_flag("MACOS_AUTO_UPLOAD_HOSTED_PAYLOAD", True)


def macos_hosted_payload_label(path_or_url: Path | str) -> str:
    raw = str(path_or_url or "").strip().lower()
    if raw.endswith(".dmg"):
        return "DMG"
    if raw.endswith(".zip"):
        return "ZIP"
    if raw.endswith(".tar.gz") or raw.endswith(".tgz") or raw.endswith(".tar"):
        return "TAR"
    return "archive"


def is_supported_macos_hosted_payload_url(url: str) -> bool:
    parsed_path = urllib.parse.urlparse(str(url or "").strip()).path.lower()
    return any(parsed_path.endswith(ext) for ext in SUPPORTED_MACOS_HOSTED_PAYLOAD_EXTENSIONS)


def rewrite_hosted_payload_url_filename(configured_url: str, payload_name: str) -> str:
    raw = str(configured_url or "").strip()
    target_name = str(payload_name or "").strip()
    if not raw or not target_name:
        return raw

    parsed = urllib.parse.urlsplit(raw)
    if not parsed.scheme or not parsed.netloc or not parsed.path:
        return raw

    current_name = PurePosixPath(parsed.path).name
    if not current_name:
        return raw
    if current_name.lower() == target_name.lower():
        return raw
    if not any(current_name.lower().endswith(ext) for ext in SUPPORTED_MACOS_HOSTED_PAYLOAD_EXTENSIONS):
        return raw

    new_path = str(PurePosixPath(parsed.path).with_name(target_name))
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment))


def upload_hosted_payload_to_hf(hosted_full_payload: Path) -> str:
    try:
        import upload_hf_dmg as hf_upload
    except Exception as exc:
        raise RuntimeError(f"Failed to load Hugging Face upload helper: {exc}") from exc

    repo_id = str(os.environ.get("MACOS_HF_REPO_ID", "") or "").strip() or hf_upload.DEFAULT_REPO_ID
    repo_type = str(os.environ.get("MACOS_HF_REPO_TYPE", "") or "").strip().lower() or hf_upload.DEFAULT_REPO_TYPE
    revision = str(os.environ.get("MACOS_HF_REVISION", "") or "").strip() or hf_upload.DEFAULT_REVISION
    path_in_repo = str(os.environ.get("MACOS_HF_PATH_IN_REPO", "") or "").strip() or hosted_full_payload.name
    endpoint = (
        str(os.environ.get("MACOS_HF_ENDPOINT", "") or "").strip()
        or str(os.environ.get("MTS_HF_UPLOAD_ENDPOINT", "") or "").strip()
        or hf_upload.DEFAULT_ENDPOINT
    )
    state_dir_raw = str(os.environ.get("MACOS_HF_UPLOAD_STATE_DIR", "") or "").strip()
    state_dir = Path(state_dir_raw).expanduser() if state_dir_raw else None

    workers = max(
        1,
        int(str(os.environ.get("MACOS_HF_UPLOAD_WORKERS", hf_upload.DEFAULT_WORKERS) or hf_upload.DEFAULT_WORKERS)),
    )
    request_retries = int(
        str(
            os.environ.get("MACOS_HF_REQUEST_RETRIES", hf_upload.DEFAULT_REQUEST_RETRIES)
            or hf_upload.DEFAULT_REQUEST_RETRIES
        )
    )
    verify = not _env_flag("MACOS_HF_NO_VERIFY", False)
    private = _env_flag("MACOS_HF_PRIVATE", False)
    ensure_repo = _env_flag("MACOS_HF_ENSURE_REPO", True)
    fresh_session = _env_flag("MACOS_HF_FRESH_SESSION", False)
    auto_restart_expired = not _env_flag("MACOS_HF_NO_AUTO_RESTART_EXPIRED", False)
    commit_message = (
        str(os.environ.get("MACOS_HF_COMMIT_MESSAGE", "") or "").strip()
        or f"Upload {hosted_full_payload.name}"
    )
    commit_description = str(os.environ.get("MACOS_HF_COMMIT_DESCRIPTION", "") or "").strip()

    print("\n== Upload Hosted macOS Payload To Hugging Face ==")
    print(f"Payload       : {hosted_full_payload.resolve()}")
    print(f"Repo target   : {repo_type}/{repo_id}@{revision}:{path_in_repo}")
    print(f"Hub endpoint  : {endpoint}")

    upload_result = hf_upload.upload_file_to_hf(
        file_path=hosted_full_payload,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
        endpoint=endpoint,
        workers=workers,
        private=private,
        verify=verify,
        state_dir=state_dir,
        commit_message=commit_message,
        commit_description=commit_description,
        fresh_session=fresh_session,
        auto_restart_expired=auto_restart_expired,
        ensure_repo=ensure_repo,
        request_retries=request_retries,
    )
    hosted_url = str(upload_result.get("url") or "").strip()
    if not hosted_url:
        raise RuntimeError("Upload completed but no public payload URL was returned.")
    print(f"Hosted URL    : {hosted_url}")
    return hosted_url


def resolve_executable(name: str, *, use_xcrun: bool = False) -> str | None:
    venv_candidate = Path(sys.executable).resolve().parent / name
    if venv_candidate.exists():
        return str(venv_candidate)

    resolved = shutil.which(name)
    if resolved:
        return resolved

    if use_xcrun:
        xcrun = shutil.which("xcrun")
        if xcrun:
            probe = subprocess.run(
                [xcrun, "--find", name],
                check=False,
                capture_output=True,
                text=True,
            )
            location = probe.stdout.strip()
            if probe.returncode == 0 and location:
                return location
    return None


def _resolve_macos_icon() -> Optional[Path]:
    candidates = [
        ROOT / "assets" / "app.icns",
        ROOT / "assets" / "app.png",
        ROOT / "app.icns",
        ROOT / "app.png",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _resolve_macos_png_icon() -> Optional[Path]:
    candidates = [
        ROOT / "assets" / "app.png",
        ROOT / "app.png",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _resolve_scene_background() -> Optional[Path]:
    candidates = [
        ROOT / "pictures" / "scene.png",
        ROOT / "pictures" / "scene.jpg",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _create_bootstrapper_brand_assets() -> tuple[Optional[Path], Optional[Path]]:
    icon_png = _resolve_macos_png_icon()
    if icon_png is None:
        return None, None

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return icon_png, _resolve_macos_icon()

    build_dir = ROOT / "build" / "generated_bootstrapper_icon"
    build_dir.mkdir(parents=True, exist_ok=True)
    out_png = build_dir / "setup-icon-1024.png"
    out_icns = build_dir / "setup.icns"

    size = 1024
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for y in range(size):
        blend = y / max(size - 1, 1)
        r = int(228 + (194 - 228) * blend)
        g = int(214 + (170 - 214) * blend)
        b = int(192 + (122 - 192) * blend)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

    draw.rounded_rectangle(
        (46, 46, size - 46, size - 46),
        radius=224,
        fill=(244, 237, 228, 235),
        outline=(255, 255, 255, 160),
        width=4,
    )
    draw.rounded_rectangle(
        (84, 84, size - 84, size - 84),
        radius=190,
        outline=(133, 93, 56, 72),
        width=3,
    )

    try:
        app_icon = Image.open(icon_png).convert("RGBA")
        app_icon.thumbnail((640, 640))
        canvas.alpha_composite(app_icon, ((size - app_icon.width) // 2, 150))
    except Exception:
        pass

    arrow_fill = (122, 76, 36, 255)
    arrow_shadow = (255, 255, 255, 80)
    draw.rounded_rectangle((320, 700, 704, 860), radius=72, fill=(255, 248, 240, 212))
    draw.rounded_rectangle((320, 700, 704, 860), radius=72, outline=(168, 128, 86, 140), width=4)
    draw.rectangle((486, 732, 538, 808), fill=arrow_shadow)
    draw.polygon([(512, 874), (424, 774), (470, 774), (470, 700), (554, 700), (554, 774), (600, 774)], fill=arrow_shadow)
    draw.rectangle((486, 722, 538, 798), fill=arrow_fill)
    draw.polygon([(512, 864), (424, 764), (470, 764), (470, 690), (554, 690), (554, 764), (600, 764)], fill=arrow_fill)

    canvas.save(out_png)

    iconutil = resolve_executable("iconutil", use_xcrun=True)
    if not iconutil:
        return out_png, _resolve_macos_icon()

    with tempfile.TemporaryDirectory(prefix="mts-setup-iconset-") as tmp_dir:
        iconset_dir = Path(tmp_dir) / "setup.iconset"
        iconset_dir.mkdir(parents=True, exist_ok=True)
        try:
            base = Image.open(out_png).convert("RGBA")
        except Exception:
            return out_png, _resolve_macos_icon()

        for edge in (16, 32, 64, 128, 256, 512):
            normal = base.resize((edge, edge), Image.LANCZOS)
            normal.save(iconset_dir / f"icon_{edge}x{edge}.png")
            retina = base.resize((edge * 2, edge * 2), Image.LANCZOS)
            retina.save(iconset_dir / f"icon_{edge}x{edge}@2x.png")

        try:
            run([iconutil, "-c", "icns", str(iconset_dir), "-o", str(out_icns)])
        except Exception:
            return out_png, _resolve_macos_icon()

    if out_icns.exists():
        return out_png, out_icns
    return out_png, _resolve_macos_icon()


def _load_background_font(size: int, *, serif: bool = False, bold: bool = False):
    try:
        from PIL import ImageFont
    except Exception:
        return None

    candidates = []
    if serif:
        candidates.extend(
            [
                "/System/Library/Fonts/NewYork.ttf",
                "/System/Library/Fonts/Times.ttc",
                "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            ]
        )
    else:
        if bold:
            candidates.extend(
                [
                    "/System/Library/Fonts/SFNS.ttf",
                    "/System/Library/Fonts/HelveticaNeue.ttc",
                    "/System/Library/Fonts/Helvetica.ttc",
                ]
            )
        else:
            candidates.extend(
                [
                    "/System/Library/Fonts/SFNS.ttf",
                    "/System/Library/Fonts/Avenir Next.ttc",
                    "/System/Library/Fonts/HelveticaNeue.ttc",
                    "/System/Library/Fonts/Helvetica.ttc",
                ]
            )

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_centered(draw, box, text, font, fill):
    try:
        left, top, right, bottom = box
        text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, align="center")
        width = text_box[2] - text_box[0]
        height = text_box[3] - text_box[1]
        x = left + max(0, (right - left - width) / 2)
        y = top + max(0, (bottom - top - height) / 2)
        draw.multiline_text((x, y), text, font=font, fill=fill, spacing=6, align="center")
    except Exception:
        pass


def _create_dmg_background(
    background_path: Path,
    *,
    app_name: str,
    version: str,
    author: str,
    icon_png: Optional[Path] = None,
    installer_mode: bool = False,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        return

    width, height = 1200, 720
    scene_path = _resolve_scene_background()
    using_scene = False
    if scene_path is not None and scene_path.exists():
        try:
            fit_mode = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "LANCZOS", 1))
            image = ImageOps.fit(
                Image.open(scene_path).convert("RGBA"),
                (width, height),
                method=fit_mode,
            )
            using_scene = True
        except Exception:
            image = Image.new("RGBA", (width, height), "#f4efe7")
    else:
        image = Image.new("RGBA", (width, height), "#f4efe7")

    draw = ImageDraw.Draw(image)

    if using_scene:
        veil = Image.new("RGBA", (width, height), (4, 18, 38, 76 if installer_mode else 58))
        image.alpha_composite(veil)
        draw = ImageDraw.Draw(image)
        accent = (136, 214, 255, 255)
        soft = (235, 248, 255, 255)
        muted = (190, 225, 244, 255)
        draw.rounded_rectangle((48, 48, 1152, 672), radius=40, outline=(172, 223, 255, 148), width=2)
        draw.rounded_rectangle((74, 74, 1126, 646), radius=34, outline=(198, 235, 255, 78), width=1)
        draw.rounded_rectangle((74, 74, 1126, 190), radius=28, fill=(8, 28, 54, 142), outline=(182, 228, 255, 44), width=1)
        draw.rounded_rectangle((320, 252, 882, 378), radius=28, fill=(8, 32, 60, 132), outline=(182, 228, 255, 40), width=1)
        draw.rounded_rectangle((152, 530, 1048, 624), radius=26, fill=(7, 30, 58, 126), outline=(182, 228, 255, 36), width=1)
    else:
        for y in range(height):
            blend = y / max(height - 1, 1)
            r = int(15 + (6 - 15) * blend)
            g = int(35 + (52 - 35) * blend)
            b = int(58 + (96 - 58) * blend)
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

        accent = (122, 208, 255, 255)
        soft = (229, 246, 255, 255)
        muted = (177, 214, 235, 255)

        draw.rounded_rectangle((54, 52, 1146, 668), radius=36, outline=(139, 214, 255, 255), width=2)
        draw.rounded_rectangle((76, 74, 1124, 646), radius=30, outline=(187, 232, 255, 188), width=1)
        draw.rounded_rectangle((74, 74, 1126, 190), radius=28, fill=(10, 35, 66, 220), outline=(182, 228, 255, 76), width=1)
        draw.rounded_rectangle((320, 252, 882, 378), radius=28, fill=(10, 39, 72, 210), outline=(182, 228, 255, 70), width=1)
        draw.rounded_rectangle((152, 530, 1048, 624), radius=26, fill=(10, 35, 66, 204), outline=(182, 228, 255, 64), width=1)

        for offset in range(0, 340, 20):
            alpha = max(0, 42 - offset // 8)
            draw.line([(120 + offset, 120), (330 + offset, 330)], fill=(180, 228, 255, alpha), width=2)

    if icon_png is not None:
        try:
            from PIL import Image

            icon = Image.open(icon_png).convert("RGBA")
            icon.thumbnail((132, 132))
            image.alpha_composite(icon, (92, 92))
        except Exception:
            pass

    title_font = _load_background_font(56, serif=True, bold=True)
    subtitle_font = _load_background_font(22, serif=False, bold=False)
    label_font = _load_background_font(24, serif=False, bold=True)
    small_font = _load_background_font(18, serif=False, bold=False)
    badge_font = _load_background_font(18, serif=False, bold=True)

    subtitle = "Guided Setup Edition" if installer_mode else "Offline Studio Edition"
    body_line = (
        "Install this launcher, then auto-download the full studio package."
        if installer_mode
        else "Full models bundled. No warm-up. No downloads."
    )
    draw.text((246, 108), app_name, font=title_font, fill=soft)
    draw.text((248, 178), subtitle, font=subtitle_font, fill=muted)
    draw.text((248, 214), body_line, font=small_font, fill=muted)
    draw.text((248, 246), f"by {author}", font=small_font, fill=accent)

    arrow_y = 390
    arrow_start = 388
    arrow_end = 816
    draw.rounded_rectangle((430, 344, 776, 438), radius=42, fill=(9, 34, 62, 118))
    draw.rounded_rectangle((448, 368, 736, 412), radius=22, fill=(24, 112, 196, 224))
    draw.rounded_rectangle((448, 368, 736, 412), radius=22, outline=(214, 241, 255, 132), width=2)
    draw.polygon(
        [(724, 350), (810, arrow_y), (724, 430), (742, arrow_y)],
        fill=(24, 112, 196, 224),
    )
    draw.polygon(
        [(720, 354), (804, arrow_y), (720, 426), (742, arrow_y)],
        fill=accent,
    )
    draw.ellipse((530, 334, 552, 356), fill=(216, 242, 255, 132))
    draw.ellipse((566, 326, 580, 340), fill=(216, 242, 255, 120))
    draw.ellipse((596, 336, 608, 348), fill=(216, 242, 255, 108))

    hint_box = (360, 280, 840, 360)
    hint_text = "\u5c06\u5e94\u7528\u62d6\u5165 Applications\nDrag the app into Applications"
    _draw_centered(draw, hint_box, hint_text, label_font, soft)

    footer_box = (170, 548, 1030, 618)
    footer_text = (
        "\u62d6\u5165 Applications \u540e\uff0c\u9996\u6b21\u6253\u5f00\u5c06\u81ea\u52a8\u4ece Hugging Face \u4e0b\u8f7d\u5b8c\u6574\u7248"
        if installer_mode
        else "\u5b8c\u6574\u79bb\u7ebf\u7248\uff1a\u8bed\u97f3\u3001\u5206\u89d2\u8272\u3001\u62a5\u544a\u3001LLM \u5168\u90e8\u53ef\u7528"
    )
    _draw_centered(draw, footer_box, footer_text, small_font, muted)

    image.save(background_path)


def _build_dmg_with_dmgbuild(
    dmg_path: Path,
    *,
    payload: Path,
    payload_display_name: Optional[str],
    app_name: str,
    version: str,
    author: str,
    icon_path: Optional[Path] = None,
    background_icon_png: Optional[Path] = None,
    installer_mode: bool = False,
) -> Path:
    import dmgbuild

    display_name = str(payload_display_name or payload.name).strip() or payload.name

    with tempfile.TemporaryDirectory(prefix="mts-dmg-bg-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        background_path = tmp_root / "background.png"
        _create_dmg_background(
            background_path,
            app_name=app_name,
            version=version,
            author=author,
            icon_png=background_icon_png or _resolve_macos_png_icon(),
            installer_mode=installer_mode,
        )

        settings = {
            "format": "UDZO",
            "files": [(str(payload), display_name)],
            "window_rect": ((120, 120), (1200, 720)),
            "default_view": "icon-view",
            "show_status_bar": False,
            "show_tab_view": False,
            "show_toolbar": False,
            "show_pathbar": False,
            "show_sidebar": False,
            "icon_size": 156,
            "text_size": 16,
            "label_pos": "bottom",
            "background": str(background_path),
            "icon": str(icon_path) if icon_path and icon_path.suffix.lower() == ".icns" else None,
            "hide_extensions": [display_name] if payload.suffix.lower() == ".app" else [],
            "icon_locations": {
                display_name: (280, 390),
            },
        }

        if payload.suffix.lower() == ".app":
            settings["symlinks"] = {"Applications": "/Applications"}
            settings["icon_locations"]["Applications"] = (920, 390)

        dmgbuild.build_dmg(
            str(dmg_path),
            app_name,
            settings=settings,
            lookForHiDPI=True,
        )

    return dmg_path


def _apply_custom_icon_to_dmg_file(dmg_path: Path, icon_path: Optional[Path]) -> None:
    if icon_path is None or not icon_path.exists() or icon_path.suffix.lower() != ".icns":
        return

    derez = resolve_executable("DeRez", use_xcrun=True)
    rez = resolve_executable("Rez", use_xcrun=True)
    setfile = resolve_executable("SetFile", use_xcrun=True)
    if not derez or not rez or not setfile:
        return

    with tempfile.TemporaryDirectory(prefix="mts-dmg-icon-rsrc-") as tmp_dir:
        rsrc_path = Path(tmp_dir) / "icon.rsrc"
        with rsrc_path.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [derez, str(icon_path)],
                check=True,
                stdout=handle,
                stderr=subprocess.DEVNULL,
            )
        subprocess.run(
            [rez, "-append", str(rsrc_path), "-o", str(dmg_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [setfile, "-a", "C", str(dmg_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def build_native_helpers() -> list[Path]:
    native_root = ROOT / "native"
    cmake_lists = native_root / "CMakeLists.txt"
    if not cmake_lists.exists():
        return []

    cmake = resolve_executable("cmake")
    if not cmake:
        raise RuntimeError(
            "cmake was not found in the active Python environment or PATH, but native/CMakeLists.txt exists."
        )

    build_dir = ROOT / "build" / "native"
    install_dir = build_dir / "install"
    run(
        [
            cmake,
            "-S",
            str(native_root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
    )
    run([cmake, "--build", str(build_dir), "--config", "Release"])
    run([cmake, "--install", str(build_dir), "--config", "Release"])

    patterns = [
        "mts_media_helper*",
        "libmts_apple_runtime_shim*.dylib",
    ]
    outputs: list[Path] = []
    for pattern in patterns:
        outputs.extend(path for path in install_dir.rglob(pattern) if path.is_file())

    if not any("mts_media_helper" in path.name for path in outputs):
        raise RuntimeError("Native helper build finished but no mts_media_helper artifact was installed.")
    return outputs


def copy_playwright_local_browsers_into_app(app_bundle: Path) -> None:
    src = playwright_package_local_browsers_dir()
    if src is None or not src.exists():
        raise RuntimeError(
            "Playwright package-local .local-browsers directory was not found after preparation. "
            f"Expected: {src}"
        )

    package_root = app_bundle_internal_root(app_bundle) / "playwright" / "driver" / "package"
    if not package_root.exists():
        raise RuntimeError(
            "PyInstaller app bundle is missing the Playwright package directory needed for browser injection: "
            f"{package_root}"
        )

    dest = package_root / ".local-browsers"
    _remove_path_if_exists(dest)

    print("\n== Copying Playwright Chromium into app bundle ==")
    ditto = resolve_executable("ditto")
    if ditto:
        run([ditto, str(src), str(dest)])
    else:
        shutil.copytree(src, dest, symlinks=True)

    if not dest.exists():
        raise RuntimeError(f"Failed to copy Playwright .local-browsers into app bundle: {dest}")


def app_bundle_internal_root(app_bundle: Path) -> Path:
    contents_dir = app_bundle / "Contents"
    candidates = [
        contents_dir / "Resources" / "_internal",
        contents_dir / "MacOS" / "_internal",
        contents_dir / "Frameworks" / "_internal",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.resolve()
            except Exception:
                return candidate
    raise RuntimeError(f"PyInstaller app bundle is missing _internal runtime files: {app_bundle}")


def app_bundle_has_internal_runtime(app_bundle: Path) -> bool:
    contents_dir = app_bundle / "Contents"
    candidates = [
        contents_dir / "Resources" / "_internal",
        contents_dir / "MacOS" / "_internal",
        contents_dir / "Frameworks" / "_internal",
    ]
    return any(candidate.exists() for candidate in candidates)


def normalize_app_bundle_layout_for_signing(app_bundle: Path) -> Path:
    contents_dir = app_bundle / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    macos_internal = macos_dir / "_internal"
    resources_internal = resources_dir / "_internal"

    resources_dir.mkdir(parents=True, exist_ok=True)

    if resources_internal.exists():
        if macos_internal.exists() or macos_internal.is_symlink():
            if macos_internal.is_symlink():
                try:
                    if macos_internal.resolve() == resources_internal.resolve():
                        return resources_internal.resolve()
                except Exception:
                    pass
            _remove_path_if_exists(macos_internal)
    elif macos_internal.exists():
        if macos_internal.is_symlink():
            resolved = macos_internal.resolve()
            if resolved.exists():
                if resolved != resources_internal:
                    _remove_path_if_exists(resources_internal)
                    shutil.move(str(resolved), str(resources_internal))
                _remove_path_if_exists(macos_internal)
            else:
                _remove_path_if_exists(macos_internal)
                raise RuntimeError(
                    "macOS app bundle _internal symlink is broken and cannot be normalized for signing: "
                    f"{macos_internal}"
                )
        else:
            print("\n== Normalize macOS App Bundle Layout ==")
            print(f"Move runtime    : {macos_internal} -> {resources_internal}")
            shutil.move(str(macos_internal), str(resources_internal))
    else:
        raise RuntimeError(f"PyInstaller app bundle is missing _internal runtime files: {app_bundle}")

    relative_target = os.path.relpath(resources_internal, macos_internal.parent)
    if macos_internal.exists() or macos_internal.is_symlink():
        _remove_path_if_exists(macos_internal)
    os.symlink(relative_target, macos_internal)
    return resources_internal.resolve()


def assemble_pyinstaller_onedir_app_bundle(
    *,
    onedir_root: Path,
    app_bundle: Path,
    app_name: str,
    bundle_id: str,
    version: str,
    icon_path: Optional[Path] = None,
) -> Path:
    if not onedir_root.exists():
        raise RuntimeError(f"PyInstaller onedir output was not found: {onedir_root}")

    executable = onedir_root / app_name
    if not executable.exists():
        raise RuntimeError(
            "PyInstaller onedir output is missing the main executable needed for app bundle assembly: "
            f"{executable}"
        )

    _remove_path_if_exists(app_bundle)

    contents_dir = app_bundle / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    print("\n== Assembling macOS app bundle from PyInstaller onedir ==")
    resources_internal = resources_dir / "_internal"
    source_internal = onedir_root / "_internal"
    if not source_internal.exists():
        raise RuntimeError(
            "PyInstaller onedir output is missing the _internal runtime directory needed for app bundle assembly: "
            f"{source_internal}"
        )

    shutil.copy2(executable, macos_dir / app_name)

    ditto = resolve_executable("ditto")
    if ditto:
        run([ditto, str(source_internal), str(resources_internal)])
    else:
        shutil.copytree(source_internal, resources_internal, symlinks=True, dirs_exist_ok=True)

    for child in onedir_root.iterdir():
        if child.name in {app_name, "_internal"}:
            continue
        dest = resources_dir / child.name
        if child.is_dir():
            if ditto:
                run([ditto, str(child), str(dest)])
            else:
                shutil.copytree(child, dest, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dest)

    os.symlink("../Resources/_internal", macos_dir / "_internal")

    copied_executable = macos_dir / app_name
    if not copied_executable.exists():
        raise RuntimeError(f"App bundle assembly did not copy the main executable: {copied_executable}")

    copied_executable.chmod(copied_executable.stat().st_mode | 0o111)

    icon_file_name: Optional[str] = None
    if icon_path is not None and icon_path.exists():
        bundled_icon = resources_dir / icon_path.name
        shutil.copy2(icon_path, bundled_icon)
        if icon_path.suffix.lower() == ".icns":
            icon_file_name = icon_path.name

    info_plist = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": app_name,
        "CFBundleExecutable": app_name,
        "CFBundleIdentifier": bundle_id,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": app_name,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    }
    if icon_file_name:
        info_plist["CFBundleIconFile"] = icon_file_name

    with (contents_dir / "Info.plist").open("wb") as fp:
        plistlib.dump(info_plist, fp, sort_keys=True)
    (contents_dir / "PkgInfo").write_text("APPL????", encoding="ascii")
    return app_bundle


def validate_app_bundle(app_bundle: Path) -> None:
    if not app_bundle.exists():
        raise RuntimeError(f"Expected app bundle was not built: {app_bundle}")

    internal_root = app_bundle_internal_root(app_bundle)

    required = [
        app_bundle / "Contents" / "MacOS" / app_bundle.stem,
        internal_root / "playwright" / "driver" / "package",
        internal_root / "playwright" / "driver" / "package" / ".local-browsers",
    ]
    if sys.platform == "darwin":
        required.extend(
            [
                internal_root / "mlx",
                internal_root / "mlx_whisper",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("macOS bundle validation failed:\n- " + "\n- ".join(missing))

    if sys.platform == "darwin":
        mlx_core_candidates = sorted(internal_root.glob("mlx/core*.so"))
        mlx_lib_candidates = sorted((internal_root / "mlx" / "lib").glob("libmlx*.dylib"))
        if not mlx_core_candidates:
            raise RuntimeError(
                "macOS bundle validation failed:\n- missing MLX compiled core module under "
                f"{internal_root / 'mlx'}"
            )
        if not mlx_lib_candidates:
            raise RuntimeError(
                "macOS bundle validation failed:\n- missing MLX dylibs under "
                f"{internal_root / 'mlx' / 'lib'}"
            )

    ffmpeg_candidates = [
        internal_root / "tools" / "ffmpeg" / "ffmpeg",
        internal_root / "ffmpeg" / "ffmpeg",
    ]
    if not any(path.exists() for path in ffmpeg_candidates):
        raise RuntimeError("Bundled app is missing ffmpeg inside _internal/tools/ffmpeg.")

    native_helper_candidates = [
        path
        for path in internal_root.rglob("mts_media_helper*")
        if path.is_file()
    ]
    if not native_helper_candidates:
        raise RuntimeError(
            "Bundled app is missing native helper mts_media_helper under _internal/native."
        )

    runtime_shim_candidates = [
        path
        for path in internal_root.rglob("libmts_apple_runtime_shim*.dylib")
        if path.is_file()
    ]
    if not runtime_shim_candidates:
        raise RuntimeError(
            "Bundled app is missing runtime shim libmts_apple_runtime_shim under _internal/native."
        )


def _is_macho_file(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        with path.open("rb") as handle:
            header = handle.read(4)
    except Exception:
        return False
    if len(header) != 4:
        return False
    return int.from_bytes(header, "big", signed=False) in MACHO_MAGIC_VALUES


def _codesign_command(
    codesign: str,
    *,
    identity: str,
    target: Path,
    runtime: bool,
) -> list[str]:
    cmd = [codesign, "--force"]
    if identity:
        cmd.append("--timestamp")
        if runtime:
            cmd.extend(["--options", "runtime"])
        cmd.extend(["--sign", identity])
    else:
        cmd.extend(["--sign", "-"])
    cmd.append(str(target))
    return cmd


def _signable_nested_targets(app_bundle: Path) -> list[Path]:
    collected: list[Path] = []
    for path in app_bundle.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            if path != app_bundle and path.suffix.lower() in SIGNABLE_BUNDLE_SUFFIXES:
                collected.append(path)
            continue
        if path.suffix.lower() in SIGNABLE_FILE_SUFFIXES or _is_macho_file(path):
            collected.append(path)

    unique_targets = {path.resolve(): path for path in collected}
    return sorted(
        unique_targets.values(),
        key=lambda path: (len(path.resolve().parts), 0 if path.is_file() else 1),
        reverse=True,
    )


def _run_quiet(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    raise SystemExit(proc.returncode)


def sign_app_bundle(app_bundle: Path, identity: str) -> None:
    codesign = resolve_executable("codesign", use_xcrun=True)
    if not codesign:
        if identity or macos_adhoc_sign_enabled():
            raise RuntimeError("codesign is required for app signing but was not found.")
        print("\n[warn] No macOS code-signing identity configured; app bundle will remain unsigned.")
        return

    if identity:
        print(f"\n== Sign macOS App Bundle ==\nTarget        : {app_bundle.resolve()}\nSigning mode  : Developer ID")
    elif macos_adhoc_sign_enabled():
        print("\n[info] No Developer ID identity configured; applying ad-hoc macOS code signature.")
    else:
        print("\n[warn] No macOS code-signing identity configured; app bundle will remain unsigned.")
        return

    if app_bundle_has_internal_runtime(app_bundle):
        normalize_app_bundle_layout_for_signing(app_bundle)
    nested_targets = _signable_nested_targets(app_bundle)
    if nested_targets:
        print(f"Nested code objs: {len(nested_targets)}")
    for target in nested_targets:
        _run_quiet(
            _codesign_command(
                codesign,
                identity=identity,
                target=target,
                runtime=False,
            )
        )

    run(
        _codesign_command(
            codesign,
            identity=identity,
            target=app_bundle,
            runtime=bool(identity),
        )
    )

    executable = app_bundle / "Contents" / "MacOS" / app_bundle.stem
    if identity:
        run([codesign, "--verify", "--strict", "--verbose=2", str(app_bundle)])
    else:
        run([codesign, "--verify", "--verbose=2", str(executable)])
        run([codesign, "--verify", "--verbose=2", str(app_bundle)])
        print("[info] Ad-hoc macOS signing verified for the executable and app bundle.")


def build_component_pkg(app_bundle: Path, *, app_name: str, version: str, identifier: str) -> Path:
    component_pkg = DIST_DIR / f"{app_name}-component.pkg"
    if component_pkg.exists():
        component_pkg.unlink()
    pkgbuild = resolve_executable("pkgbuild", use_xcrun=True)
    if not pkgbuild:
        raise RuntimeError("pkgbuild is required to build the macOS component package.")
    run(
        [
            pkgbuild,
            "--component",
            str(app_bundle),
            "--install-location",
            "/Applications",
            "--identifier",
            identifier,
            "--version",
            version,
            str(component_pkg),
        ]
    )
    return component_pkg


def build_product_pkg(component_pkg: Path, *, app_name: str, identity: str) -> Path:
    final_pkg = DIST_DIR / f"{app_name}-Installer.pkg"
    if final_pkg.exists():
        final_pkg.unlink()

    productbuild = resolve_executable("productbuild", use_xcrun=True)
    if not productbuild:
        raise RuntimeError("productbuild is required to assemble the signed macOS installer.")

    cmd = [productbuild, "--package", str(component_pkg)]
    if identity:
        cmd.extend(["--sign", identity])
    cmd.append(str(final_pkg))
    run(cmd)
    return final_pkg


def build_dmg(
    payload: Path,
    *,
    app_name: str,
    version: str,
    author: str,
    dmg_name: Optional[str] = None,
    payload_display_name: Optional[str] = None,
    icon_path: Optional[Path] = None,
    background_icon_png: Optional[Path] = None,
    installer_mode: bool = False,
) -> Path:
    dmg_stem = str(dmg_name or app_name).strip() or app_name
    display_name = str(payload_display_name or payload.name).strip() or payload.name
    dmg_path = DIST_DIR / f"{dmg_stem}-macOS.dmg"
    if dmg_path.exists():
        dmg_path.unlink()

    if payload.is_dir() and payload.suffix.lower() == ".app":
        try:
            dmg_result = _build_dmg_with_dmgbuild(
                dmg_path,
                payload=payload,
                payload_display_name=display_name,
                app_name=app_name,
                version=version,
                author=author,
                icon_path=icon_path,
                background_icon_png=background_icon_png,
                installer_mode=installer_mode,
            )
            try:
                _apply_custom_icon_to_dmg_file(dmg_result, icon_path)
            except Exception as exc:
                print(f"\n[warn] Could not apply custom icon to DMG file: {exc}")
            return dmg_result
        except Exception as exc:
            print(f"\n[warn] dmgbuild custom layout failed, falling back to generic DMG: {exc}")

    create_dmg = resolve_executable("create-dmg")
    hdiutil = resolve_executable("hdiutil", use_xcrun=True)
    with tempfile.TemporaryDirectory(prefix="mts-dmg-") as tmp_dir:
        staging_dir = Path(tmp_dir) / app_name
        staging_dir.mkdir(parents=True, exist_ok=True)
        target = staging_dir / display_name
        if payload.is_dir():
            shutil.copytree(payload, target)
            try:
                os.symlink("/Applications", staging_dir / "Applications")
            except Exception:
                pass
        else:
            shutil.copy2(payload, target)

        if create_dmg:
            cmd = [
                create_dmg,
                "--overwrite",
                "--volname",
                app_name,
            ]
            if payload.is_dir() and payload.suffix.lower() == ".app":
                background_path = staging_dir / ".background.png"
                _create_dmg_background(
                    background_path,
                    app_name=app_name,
                    version=version,
                    author=author,
                    icon_png=background_icon_png or _resolve_macos_png_icon(),
                    installer_mode=installer_mode,
                )
                if background_path.exists():
                    cmd.extend(["--background", str(background_path)])
                cmd.extend(
                    [
                        "--window-size",
                        "1200",
                        "720",
                        "--icon-size",
                        "156",
                        "--icon",
                        display_name,
                        "280",
                        "390",
                        "--app-drop-link",
                        "920",
                        "390",
                    ]
                )
            cmd.extend([str(dmg_path), str(staging_dir)])
            run(cmd)
        else:
            if not hdiutil:
                raise RuntimeError("hdiutil is required to build a DMG when create-dmg is unavailable.")
            run(
                [
                    hdiutil,
                    "create",
                    "-volname",
                    app_name,
                    "-srcfolder",
                    str(staging_dir),
                    "-ov",
                    "-format",
                    "UDZO",
                    str(dmg_path),
                ]
            )
    try:
        _apply_custom_icon_to_dmg_file(dmg_path, icon_path)
    except Exception as exc:
        print(f"\n[warn] Could not apply custom icon to DMG file: {exc}")
    return dmg_path


def build_zip_payload(
    payload: Path,
    *,
    archive_name: Optional[str] = None,
) -> Path:
    print("\n== Build Hosted macOS ZIP Payload ==")
    print(f"Source        : {payload.resolve()}")
    zip_stem = str(archive_name or payload.stem).strip() or payload.stem
    zip_path = DIST_DIR / f"{zip_stem}-macOS.zip"
    if zip_path.exists():
        zip_path.unlink()

    ditto = resolve_executable("ditto")
    if not ditto:
        raise RuntimeError("ditto is required to build a macOS ZIP payload.")

    run(
        [
            ditto,
            "-c",
            "-k",
            "--sequesterRsrc",
            "--keepParent",
            str(payload),
            str(zip_path),
        ]
    )
    print(f"Hosted ZIP    : {zip_path.resolve()} ({human_bytes(zip_path.stat().st_size)})")
    return zip_path


def validate_simple_app_bundle(app_bundle: Path) -> None:
    if not app_bundle.exists():
        raise RuntimeError(f"Expected app bundle was not built: {app_bundle}")
    executable = app_bundle / "Contents" / "MacOS" / app_bundle.stem
    if not executable.exists():
        raise RuntimeError(f"App bundle is missing the main executable: {executable}")


def _remove_path_if_exists(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    except Exception:
        pass


def find_existing_hosted_full_payload(
    app_name: str,
    *,
    preferred_format: Optional[str] = None,
) -> Optional[Path]:
    zip_candidates = [
        DIST_DIR / f"{app_name}-Full-macOS.zip",
        *sorted(DIST_DIR.glob(f"{app_name}*Full*.zip")),
    ]
    dmg_candidates = [
        DIST_DIR / f"{app_name}-Full-macOS.dmg",
        *sorted(DIST_DIR.glob(f"{app_name}*Full*.dmg")),
    ]
    if preferred_format == "zip":
        candidates = zip_candidates
    elif preferred_format == "dmg":
        candidates = dmg_candidates
    else:
        candidates = [*zip_candidates, *dmg_candidates]

    existing = [path.resolve() for path in candidates if path.exists() and path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def bootstrapper_app_name(cfg, app_name: str) -> str:
    raw = str(getattr(cfg, "MACOS_BOOTSTRAPPER_APP_NAME", "") or "").strip()
    return raw or f"{app_name}-Setup.app"


def bootstrapper_bundle_candidates(cfg, app_name: str) -> list[Path]:
    app_name_with_suffix = bootstrapper_app_name(cfg, app_name)
    stem = Path(app_name_with_suffix).stem
    return [
        DIST_DIR / app_name_with_suffix,
        DIST_DIR / stem / app_name_with_suffix,
    ]


def build_bootstrapper_pyinstaller_env(*, bundle_id: str, icon_path: Optional[Path] = None) -> dict[str, str]:
    env = os.environ.copy()
    runtime_bin = str(Path(sys.executable).resolve().parent)
    path_parts = [part for part in str(env.get("PATH", "")).split(os.pathsep) if part]
    if runtime_bin not in path_parts:
        env["PATH"] = os.pathsep.join([runtime_bin, *path_parts]) if path_parts else runtime_bin

    pythonpath_parts = [part for part in str(env.get("PYTHONPATH", "")).split(os.pathsep) if part]
    root_str = str(ROOT)
    if root_str not in pythonpath_parts:
        env["PYTHONPATH"] = os.pathsep.join([root_str, *pythonpath_parts]) if pythonpath_parts else root_str

    env["MACOS_BOOTSTRAPPER_BUNDLE_ID"] = bundle_id
    if icon_path is not None and icon_path.exists():
        env["MACOS_BOOTSTRAPPER_ICON_PATH"] = str(icon_path.resolve())
    return env


def build_online_bootstrapper(
    *,
    cfg,
    app_name: str,
    version: str,
    bundle_id: str,
    sign_identity: str,
    hosted_full_payload: Path,
    pyi: list[str],
) -> tuple[Path, Path, str] | None:
    payload_sha256 = sha256sum(hosted_full_payload)
    payload_label = macos_hosted_payload_label(hosted_full_payload)

    print("\n== Prepare Hosted macOS Payload ==")
    print("The final lightweight DMG will embed the hosted payload URL and replace itself with the real app on first launch.")
    print(f"Hosted Payload  : {hosted_full_payload.resolve()} ({payload_label})")
    print(f"           SHA256: {payload_sha256}")

    hosted_url = ""
    if macos_auto_upload_hosted_payload():
        hosted_url = upload_hosted_payload_to_hf(hosted_full_payload)

    if not hosted_url:
        default_installer_url = rewrite_hosted_payload_url_filename(
            str(getattr(cfg, "DEFAULT_MACOS_INSTALLER_URL", "") or "").strip(),
            hosted_full_payload.name,
        )
        hosted_url = prompt_url_with_default(
            default_url=default_installer_url,
            accept_empty_default=True,
            env_override_name="MACOS_INSTALLER_URL_OVERRIDE",
            label="Hosted full macOS payload URL",
        )
    if not hosted_url:
        print("\n[warn] No hosted macOS payload URL was provided; the lightweight DMG was not built.")
        return None
    if not is_supported_macos_hosted_payload_url(hosted_url):
        raise RuntimeError(
            "The lightweight macOS launcher expects a hosted .zip/.dmg/.tar full-app payload. "
            f"Received URL: {hosted_url}"
        )

    updated_defaults = update_dist_config_macos_defaults(
        installer_url=hosted_url,
        installer_sha256=payload_sha256,
    )
    if updated_defaults:
        print("\n>>> Updated macOS installer defaults in dist_config.py")
    else:
        print("\n>>> macOS installer defaults were already up to date in dist_config.py")

    print("\n== Build Lightweight macOS Launcher App ==")
    bootstrapper_env = build_bootstrapper_pyinstaller_env(
        bundle_id=bundle_id,
        icon_path=_resolve_macos_icon(),
    )
    run([*pyi, "packaging/build_macos_bootstrapper.spec", "--noconfirm", "--clean"], env=bootstrapper_env)

    bootstrapper_app = next(
        (path for path in bootstrapper_bundle_candidates(cfg, app_name) if path.exists()),
        bootstrapper_bundle_candidates(cfg, app_name)[0],
    )
    validate_simple_app_bundle(bootstrapper_app)
    sign_app_bundle(bootstrapper_app, sign_identity)

    print("\n== Build Lightweight macOS DMG ==")
    bootstrapper_dmg = build_dmg(
        bootstrapper_app,
        app_name=app_name,
        version=version,
        author=author_name(cfg),
        dmg_name=app_name,
        payload_display_name=f"{app_name}.app",
        icon_path=_resolve_macos_icon(),
        background_icon_png=_resolve_macos_png_icon(),
        installer_mode=True,
    )
    print(f"Lightweight DMG: {bootstrapper_dmg.resolve()}")
    if not keep_bootstrapper_app_artifact() and bootstrapper_app.exists():
        shutil.rmtree(bootstrapper_app, ignore_errors=True)
    return bootstrapper_app, bootstrapper_dmg, hosted_url


def main() -> int:
    ensure_preferred_python_runtime()
    if sys.platform != "darwin":
        raise SystemExit("build_macos.py must be run on macOS.")

    cfg = load_dist_config()
    app_name = str(cfg.APP_NAME)
    app_bundle_candidates = [
        DIST_DIR / f"{app_name}.app",
        DIST_DIR / app_name / f"{app_name}.app",
    ]
    app_bundle = app_bundle_candidates[0]
    version = release_version()
    bundle_id = bundle_identifier(app_name)
    sign_identity = codesign_identity()
    pkg_identity = installer_identity()
    build_online_installer = _env_flag("BUILD_MACOS_ONLINE_INSTALLER", True)
    hosted_payload_format = macos_hosted_payload_format(sign_identity)
    auto_upload_enabled = build_online_installer and macos_auto_upload_hosted_payload()
    upload_token_source = prepare_hf_upload_environment() if auto_upload_enabled else ""

    print("== macOS Build Start ==")
    print(f"Repo root        : {ROOT}")
    print(f"Python runtime   : {sys.executable}")
    print(f"App name         : {app_name}")
    print(f"Bundle ID        : {bundle_id}")
    print(f"Code signing     : {macos_signing_mode_label(sign_identity)}")
    print(f"Installer signing: {'enabled' if pkg_identity else 'disabled'}")
    print(f"Offline bundle   : {'enabled' if macos_offline_bundle_enabled() else 'disabled'}")
    print(f"Model cache mode : {model_cache_bundle_mode()}")
    print(f"DMG payload      : {macos_dmg_payload_kind()}")
    print(f"Keep intermediates: {'yes' if keep_macos_intermediate_artifacts() else 'no'}")
    if build_online_installer:
        print(f"Keep hosted ZIP  : {'yes' if keep_macos_hosted_payload_artifact() else 'no'}")
    if build_online_installer:
        print(f"Hosted payload   : {hosted_payload_format}")
        print(f"Auto upload      : {'enabled' if auto_upload_enabled else 'disabled'}")
        if auto_upload_enabled:
            print(f"HF upload token  : {upload_token_source}")
            if upload_token_source == "missing" and not sys.stdin.isatty():
                raise RuntimeError(
                    "Automatic hosted payload upload is enabled but no Hugging Face token is available. "
                    "Set MACOS_HF_UPLOAD_TOKEN/HF_TOKEN, run `hf auth login`, or configure the packaging upload token before launching build_macos.py."
                )
    pyi = pyinstaller_cmd()

    existing_full_payload = (
        find_existing_hosted_full_payload(app_name, preferred_format=hosted_payload_format)
        if build_online_installer
        else None
    )
    if existing_full_payload is not None:
        print("\n== Reusing Existing Full macOS Payload ==")
        print(f"Hosted Payload  : {existing_full_payload.resolve()}")
        bootstrapper_result = build_online_bootstrapper(
            cfg=cfg,
            app_name=app_name,
            version=version,
            bundle_id=bundle_id,
            sign_identity=sign_identity,
            hosted_full_payload=existing_full_payload,
            pyi=pyi,
        )
        print("\n== macOS Build Complete ==")
        if bootstrapper_result is not None:
            bootstrapper_app, bootstrapper_dmg, hosted_url = bootstrapper_result
            print(f"Hosted URL: {hosted_url}")
            if bootstrapper_app.exists():
                print(f"Bootstrap App : {bootstrapper_app.resolve()}")
            print(f"Final DMG      : {bootstrapper_dmg.resolve()}")
            print(f"Hosted Payload : {existing_full_payload.resolve()}")
            return 0
        print(f"Hosted Payload : {existing_full_payload.resolve()}")
        return 0

    existing_app_bundle = next(
        (path.resolve() for path in app_bundle_candidates if path.exists() and path.is_dir()),
        None,
    )
    existing_app_bundle_reusable = False
    existing_app_bundle_reason = ""
    if existing_app_bundle is not None:
        existing_app_bundle_reusable, existing_app_bundle_reason = can_reuse_existing_app_bundle(
            existing_app_bundle
        )
        if not existing_app_bundle_reusable:
            print("\n== Existing macOS App Bundle Is Incomplete ==")
            print(f"App bundle      : {existing_app_bundle.resolve()}")
            print(f"Reason          : {existing_app_bundle_reason}")
            print("Selection       : rebuild from scratch")

    reuse_existing_app_bundle = (
        existing_app_bundle is not None
        and existing_app_bundle_reusable
        and _ask_reuse_existing_app_bundle(existing_app_bundle)
    )

    if reuse_existing_app_bundle and existing_app_bundle is not None:
        app_bundle = existing_app_bundle
        print("\n== Reusing Existing macOS App Bundle ==")
        print(f"App bundle      : {app_bundle.resolve()} ({human_bytes(dir_size_bytes(app_bundle))})")
    else:
        prepare_bundled_playwright_chromium()
        build_native_helpers()

        env = build_app_pyinstaller_env()
        env["MACOS_BUNDLE_ID"] = bundle_id
        env["INCLUDE_RUNTIME_MODEL_CACHES"] = "1" if macos_offline_bundle_enabled() else "0"
        env["PYI_SKIP_PLAYWRIGHT_LOCAL_BROWSERS"] = "1"
        env["PYI_SKIP_MACOS_BUNDLE"] = "1"
        apply_default_curated_model_bundle_env(env)

        onedir_root = DIST_DIR / app_name
        for candidate in app_bundle_candidates:
            _remove_path_if_exists(candidate)
        _remove_path_if_exists(onedir_root)
        run([*pyi, "packaging/build_app.spec", "--noconfirm", "--clean"], env=env)

        app_bundle = assemble_pyinstaller_onedir_app_bundle(
            onedir_root=onedir_root,
            app_bundle=app_bundle_candidates[0],
            app_name=app_name,
            bundle_id=bundle_id,
            version=version,
            icon_path=_resolve_macos_icon(),
        )
        if not keep_macos_intermediate_artifacts():
            _remove_path_if_exists(onedir_root)
        copy_playwright_local_browsers_into_app(app_bundle)

    validate_app_bundle(app_bundle)
    sign_app_bundle(app_bundle, sign_identity)

    component_pkg = build_component_pkg(
        app_bundle,
        app_name=app_name,
        version=version,
        identifier=bundle_id,
    )
    final_pkg = build_product_pkg(component_pkg, app_name=app_name, identity=pkg_identity)

    full_app_payload: Optional[Path] = None
    fallback_dmg_path: Optional[Path] = None
    if build_online_installer:
        if hosted_payload_format == "zip":
            full_app_payload = build_zip_payload(
                app_bundle,
                archive_name=f"{app_name}-Full",
            )
        else:
            full_app_payload = build_dmg(
                app_bundle,
                app_name=app_name,
                version=version,
                author=author_name(cfg),
                dmg_name=f"{app_name}-Full",
                icon_path=_resolve_macos_icon(),
                background_icon_png=_resolve_macos_png_icon(),
                installer_mode=False,
            )
    else:
        dmg_target = app_bundle if macos_dmg_payload_kind() == "app" else final_pkg
        fallback_dmg_path = build_dmg(
            dmg_target,
            app_name=app_name,
            version=version,
            author=author_name(cfg),
            icon_path=_resolve_macos_icon(),
            background_icon_png=_resolve_macos_png_icon(),
            installer_mode=False,
        )
    bootstrapper_result = None
    if build_online_installer and full_app_payload is not None:
        bootstrapper_result = build_online_bootstrapper(
            cfg=cfg,
            app_name=app_name,
            version=version,
            bundle_id=bundle_id,
            sign_identity=sign_identity,
            hosted_full_payload=full_app_payload,
            pyi=pyi,
        )

    print("\n== macOS Build Complete ==")
    if bootstrapper_result is not None:
        bootstrapper_app, bootstrapper_dmg, hosted_url = bootstrapper_result
        print(f"Hosted URL: {hosted_url}")
        if bootstrapper_app.exists():
            print(f"Bootstrap App : {bootstrapper_app.resolve()}")
        print(f"Final DMG      : {bootstrapper_dmg.resolve()}")
        if keep_macos_intermediate_artifacts():
            print(f"Hosted Payload : {full_app_payload.resolve()}")
            print(f"App bundle     : {app_bundle.resolve()} ({human_bytes(dir_size_bytes(app_bundle))})")
            print(f"Component PKG  : {component_pkg.resolve()}")
            print(f"Installer PKG  : {final_pkg.resolve()}")
        else:
            _remove_path_if_exists(app_bundle)
            _remove_path_if_exists(component_pkg)
            _remove_path_if_exists(final_pkg)
            if not keep_macos_hosted_payload_artifact():
                _remove_path_if_exists(full_app_payload)
                print("Intermediates  : cleaned (app bundle / pkg / hosted payload)")
            else:
                print("Intermediates  : cleaned (app bundle / pkg)")
                print(f"Hosted Payload : kept at {full_app_payload.resolve()}")
        return 0

    print(f"App bundle : {app_bundle.resolve()} ({human_bytes(dir_size_bytes(app_bundle))})")
    print(f"Component PKG: {component_pkg.resolve()}")
    print(f"Installer PKG: {final_pkg.resolve()}")
    if fallback_dmg_path is not None:
        print(f"DMG       : {fallback_dmg_path.resolve()}")
    elif full_app_payload is not None:
        print(f"Hosted Payload: {full_app_payload.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
