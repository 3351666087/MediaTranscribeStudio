from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from local_llm_bench.data import load_from_environment
from local_llm_bench.runner import run


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Privacy-preserving conservative Chinese cleanup benchmark."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--dev-ratio", type=float, default=0.70)
    parser.add_argument("--max-dev", type=_positive_or_none, default=48)
    parser.add_argument("--max-heldout", type=_positive_or_none, default=24)
    parser.add_argument("--max-safety", type=_positive_or_none, default=16)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model_name = "".join(
        character if character.isalnum() else "-"
        for character in args.model
    ).strip("-")
    output_json = (
        Path(args.output_json)
        if args.output_json
        else ROOT / "reports" / f"{timestamp}-{safe_model_name}.json"
    )
    output_markdown = (
        Path(args.output_markdown)
        if args.output_markdown
        else ROOT / "reports" / f"{timestamp}-{safe_model_name}.md"
    )
    dataset = load_from_environment(
        dev_ratio=args.dev_ratio,
        max_dev=args.max_dev,
        max_heldout=args.max_heldout,
        max_safety=args.max_safety,
    )
    report = run(
        root=ROOT,
        dataset=dataset,
        model=args.model,
        ollama_host=args.ollama_host,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        max_retries=args.max_retries,
        output_json=output_json,
        output_markdown=output_markdown,
    )
    overall = report["metrics"]["overall"]
    print(
        "completed "
        f"samples={report['run']['sampleCount']} "
        f"wall_s={report['run']['wallTimeMs'] / 1000:.3f} "
        f"contract_valid={overall['finalContractValidityRate']:.3f} "
        f"worsened={overall['worsenedAgainstTargetRate']:.3f} "
        f"tier={report['recommendation']['tier']}"
    )
    print(f"json_report={output_json.resolve()}")
    print(f"markdown_report={output_markdown.resolve()}")
    return 0


def _positive_or_none(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


if __name__ == "__main__":
    raise SystemExit(main())
