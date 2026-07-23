"""CLI for the repository-local Design Pack quality gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .validator import audit_project, report_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.design_quality",
        description=(
            "Fail-closed static and evidence validation for the desktop "
            "Design Pack contract. This command does not claim external "
            "Design Pack passage."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="MediaTranscribeStudio repository root (default: current directory)",
    )
    parser.add_argument(
        "--screenshot-manifest",
        type=Path,
        help="real native screenshot evidence JSON",
    )
    parser.add_argument(
        "--ocr-manifest",
        type=Path,
        help="real day/night/background OCR result JSON",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="require screenshot and OCR evidence in addition to source checks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the JSON report to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_project(
            args.project_root,
            screenshot_manifest=args.screenshot_manifest,
            ocr_manifest=args.ocr_manifest,
            require_evidence=args.release,
        )
        rendered = report_json(report)
    except Exception as exc:
        print(
            "{\n"
            '  "schemaVersion": "1.0.0",\n'
            '  "kind": "media-transcribe-studio/design-quality-report",\n'
            '  "status": "fail",\n'
            '  "fatal": true,\n'
            f'  "error": {__import__("json").dumps(str(exc), ensure_ascii=False)}\n'
            "}",
            file=sys.stdout,
        )
        return 2

    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
