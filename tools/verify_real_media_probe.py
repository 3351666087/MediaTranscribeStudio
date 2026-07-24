"""Verify real local media through content-driven FFprobe/FFmpeg admission.

The checks intentionally exercise the same MOV and M4A bytes under their
original names, an unknown extension, and no extension.  Hard links avoid
duplicating the large source files while still changing only the filename
hint presented to the media boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.media_probe import MediaProbe, MediaProbeError, validated_media_probe_payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_case(
    probe: MediaProbe,
    *,
    source: Path,
    label: str,
    extension_hint: str,
) -> dict[str, Any]:
    try:
        result = probe.probe(source)
    except MediaProbeError as exc:
        return {
            "label": label,
            "sourcePath": str(source.resolve()),
            "extensionHint": extension_hint,
            "status": "rejected",
            "errorCode": exc.code.value,
            "error": str(exc),
        }
    payload = validated_media_probe_payload(result)
    return {
        "label": label,
        "sourcePath": str(source.resolve()),
        "extensionHint": extension_hint,
        "status": "accepted",
        "sourceSha256": _sha256(source),
        "mediaProbe": payload,
    }


def verify_sources(sources: Sequence[Path]) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one source is required")
    probe = MediaProbe()
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mts-real-media-probe-") as temp_root:
        temporary_root = Path(temp_root)
        for source in sources:
            source = source.expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            cases.append(
                _probe_case(
                    probe,
                    source=source,
                    label=f"{source.name}:original",
                    extension_hint=source.suffix.lower() or "<none>",
                )
            )
            for label, filename in (
                ("unknown-extension", f"{source.stem}.content-probe"),
                ("extensionless", f"{source.stem}-extensionless"),
            ):
                alias = temporary_root / filename
                os.link(source, alias)
                cases.append(
                    _probe_case(
                        probe,
                        source=alias,
                        label=f"{source.name}:{label}",
                        extension_hint=alias.suffix.lower() or "<none>",
                    )
                )
    return {
        "schemaVersion": "1.0.0",
        "status": "passed" if all(item["status"] == "accepted" for item in cases) else "failed",
        "sources": [
            {
                "path": str(path.expanduser().resolve()),
                "sha256": _sha256(path.expanduser().resolve()),
                "bytes": path.expanduser().resolve().stat().st_size,
            }
            for path in sources
        ],
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_sources(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
