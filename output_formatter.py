"""
output_formatter.py - TXT + JSON 输出（不含报告生成，报告由pipeline直接调用report_generator）
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from output_layout import resolve_output_subdir
from utils import format_timestamp

logger = logging.getLogger(__name__)


class OutputFormatter:
    """输出TXT和JSON格式"""

    def __init__(self, config):
        self.config = config
        self.output_cfg = config["output"]
        self.output_dir = Path(config["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ts_fmt = self.output_cfg.get("timestamp_format", "HH:MM:SS.mmm")
        self.json_indent = self.output_cfg.get("json_indent", 2)

    def resolve_output_subdir(self, source_file: Path, input_dir: Path) -> Path:
        return resolve_output_subdir(
            output_root=self.output_dir,
            source_file=source_file,
            input_dir=input_dir,
        )

    def write_results(
        self, segments: List, source_file: Path, input_dir: Path,
        duration: float = 0.0, metadata: Optional[dict] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        stem = source_file.stem
        file_subdir = (
            Path(output_dir)
            if output_dir is not None
            else self.resolve_output_subdir(source_file=source_file, input_dir=input_dir)
        )
        file_subdir.mkdir(parents=True, exist_ok=True)

        txt_path = file_subdir / f"{stem}.txt"
        json_path = file_subdir / f"{stem}.json"

        self._write_txt(segments, txt_path, source_file, duration)
        self._write_json(segments, json_path, source_file, duration, metadata)

        logger.info(f"Output: {txt_path.name}, {json_path.name}")
        return {"txt": txt_path, "json": json_path}

    def _write_txt(self, segments, path, source, duration):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{'='*72}\n")
            f.write(f"  Transcript: {source.name}\n")
            f.write(f"  Duration:   {format_timestamp(duration, self.ts_fmt)}\n")
            f.write(f"  Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Segments:   {len(segments)}\n")
            f.write(f"{'='*72}\n\n")

            prev_spk = None
            for seg in segments:
                st = format_timestamp(seg.start, self.ts_fmt)
                et = format_timestamp(seg.end, self.ts_fmt)
                if seg.speaker != prev_spk:
                    if prev_spk is not None:
                        f.write("\n")
                    prev_spk = seg.speaker
                f.write(f"[{st} --> {et}] {seg.speaker}: {seg.text}\n")

            f.write(f"\n{'='*72}\n  End of transcript\n{'='*72}\n")

    def _write_json(self, segments, path, source, duration, metadata):
        speakers = list(dict.fromkeys(seg.speaker for seg in segments))

        data = {
            "metadata": {
                "source_file": source.name,
                "source_path": str(source),
                "duration_sec": round(duration, 3),
                "duration_formatted": format_timestamp(duration, self.ts_fmt),
                "generated_at": datetime.now().isoformat(),
                "num_segments": len(segments),
                "num_speakers": len(speakers),
                "speakers": speakers,
                **(metadata or {}),
            },
            "segments": [s.to_dict() for s in segments],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=self.json_indent)

    def write_summary(self, all_results: List[Dict], path: Optional[Path] = None):
        if path is None:
            path = self.output_dir / self.config["paths"]["summary_file"]

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{'#'*72}\n")
            f.write("#  TRANSCRIPTION SUMMARY\n")
            f.write(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"#  Files: {len(all_results)}\n")
            f.write(f"{'#'*72}\n\n")

            t_dur = t_seg = 0
            t_time = 0.0

            for i, r in enumerate(sorted(all_results, key=lambda x: x.get("source_file", "")), 1):
                dur = r.get("duration", 0)
                ns = r.get("num_segments", 0)
                el = r.get("elapsed", 0)
                t_dur += dur
                t_seg += ns
                t_time += el

                f.write(f"{'─'*72}\n")
                f.write(f"[{i:3d}] {r.get('source_file','?')}\n")
                f.write(f"      Duration: {format_timestamp(dur, self.ts_fmt)} | "
                        f"Segments: {ns} | Time: {el:.1f}s | "
                        f"Status: {r.get('status','?')}\n")

                if r.get("error"):
                    f.write(f"      ERROR: {r['error']}\n")

                for key in [
                    "txt_path",
                    "json_path",
                    "html_path",
                    "pdf_path",
                    "burned_video_path",
                    "subtitle_srt_path",
                ]:
                    if key in r:
                        p = Path(r[key])
                        try:
                            shown = p.relative_to(self.output_dir)
                        except ValueError:
                            shown = p
                        f.write(f"      -> {shown}\n")
                if r.get("subtitle_backend"):
                    f.write(f"      -> subtitle_backend: {r['subtitle_backend']}\n")

                segs = r.get("segments", [])
                if segs:
                    f.write("      Preview:\n")
                    for s in segs[:3]:
                        txt = s.text[:60] + ("..." if len(s.text) > 60 else "")
                        f.write(f"        [{format_timestamp(s.start, self.ts_fmt)}] {s.speaker}: {txt}\n")
                    if len(segs) > 3:
                        f.write(f"        ... +{len(segs)-3} more\n")
                f.write("\n")

            f.write(f"{'═'*72}\n")
            f.write(f"  Total: {len(all_results)} files, "
                    f"{format_timestamp(t_dur, self.ts_fmt)} audio, "
                    f"{t_seg} segments, {t_time:.1f}s\n")
            if t_dur > 0:
                f.write(f"  RTF: {t_time/t_dur:.3f}\n")
            f.write(f"{'═'*72}\n")

        logger.info(f"Summary: {path}") 
