# MediaTranscribeStudio

MediaTranscribeStudio is a desktop-first audio and video transcription pipeline that combines ASR, speaker diarization, structured report generation, subtitle rendering, and release packaging tooling for macOS and Windows.

This GitHub-ready copy keeps the original source layout, removes local build/runtime artifacts, and replaces embedded secrets and personal distribution endpoints with safe placeholders.

## Highlights

- Multi-engine ASR workflow with `FunASR`, `faster-whisper`, and platform-aware runtime selection.
- Speaker diarization pipeline with NeMo MSDD, pyannote fallback, and posterior-fusion helpers.
- Rich outputs including plain text, JSON, HTML, PDF, SRT, ASS, and optional burned-in caption video.
- Desktop UI built with `PySide6` plus packaging scripts for Windows installers and macOS app delivery.
- Native helper modules under [`native/`](./native) for media/runtime support.

## Repository Layout

```text
.
├── main.py                     # CLI entry point and default UI launcher
├── pipeline.py                 # End-to-end transcription workflow
├── transcriber.py              # ASR, diarization, and model download logic
├── mts_ui/                     # Desktop UI implementation
├── diar_fusion/                # Speaker posterior fusion and calibration helpers
├── native/                     # Native C++ / Objective-C++ helper targets
├── packaging/                  # Windows / macOS packaging and installer scripts
├── assets/                     # Application icons
├── pictures/                   # UI/installer visual assets
├── tools/                      # Internal verification and calibration utilities
├── config.yaml                 # Safe default configuration tracked in Git
├── config.example.yaml         # Copy or diff against this for custom setups
├── requirements.txt            # Main Python dependency set
└── requirements-macos.txt      # macOS-focused dependency set
```

## Quick Start

### 1. Prepare a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For macOS-specific packaging/runtime work, install the extra set:

```bash
pip install -r requirements-macos.txt
```

### 2. Install required system tools

- `ffmpeg` for media decoding and caption video rendering
- `wkhtmltopdf` or a Chromium/Playwright-based PDF path for report export
- `cmake` if you plan to build native helpers from [`native/`](./native)

### 3. Configure secrets locally

Do not hardcode secrets into tracked files. Use environment variables or an untracked local override:

```bash
cp .env.example .env
```

Common variables:

- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`
- `NGC_API_KEY`
- `OPENAI_API_KEY` or `DASHSCOPE_API_KEY`
- `PAYLOAD_URL_OVERRIDE` for packaging/release flows

### 4. Run the application

Launch the desktop UI:

```bash
python main.py --ui
```

Run from the CLI:

```bash
python main.py \
  --config config.yaml \
  --input input_files \
  --output output_files \
  --engine auto
```

## Configuration

- [`config.yaml`](./config.yaml) is intentionally safe for Git and contains no live tokens.
- [`config.example.yaml`](./config.example.yaml) can be used as a reset baseline.
- Missing keys are merged with defaults from [`config.py`](./config.py), so the committed config stays compact.

## Packaging and Release

Packaging scripts live in [`packaging/`](./packaging):

- `python packaging/one_click_build.py`
- `python packaging/build_macos.py`
- `python packaging/upload_hf_dmg.py --repo-id your-org/your-repo`

Before running release builds, update:

- [`packaging/dist_config.py`](./packaging/dist_config.py)
- Environment variables from [`.env.example`](./.env.example)
- Any installer branding or host URLs needed for your distribution channel

## GitHub Upload Checklist

- Keep `checkpoints/`, model caches, outputs, and virtual environments out of Git.
- Review model licenses for `FunASR`, `NeMo`, `pyannote`, Whisper-family models, and bundled assets.
- Replace placeholder packaging URLs with your real release endpoints before shipping installers.
- Store API keys and download tokens in environment variables, not in tracked YAML or Python files.

## Notes

- This repository is application-oriented rather than a lightweight reusable SDK.
- Large pretrained models and generated reports are intentionally excluded from version control.
- Native packaging scripts are preserved, but release-specific secrets and private endpoints have been removed.
