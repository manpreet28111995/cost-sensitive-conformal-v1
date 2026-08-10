"""Run benchmark sensitivity grid over costs and review capacity."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("sensitivity"))
    parser.add_argument("--fp-costs", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--fn-costs", nargs="+", type=float, default=[5.0, 10.0, 20.0, 50.0])
    parser.add_argument("--review-capacities", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--defer-costs", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    benchmark = root / "scripts" / "run_benchmark.py"
    for fp in args.fp_costs:
        for fn in args.fn_costs:
            for capacity in args.review_capacities:
                name = f"fp_{fp:g}_fn_{fn:g}_cap_{capacity:g}"
                output = args.output_dir / name
                command = [
                    sys.executable,
                    str(benchmark),
                    "--fp-cost",
                    str(fp),
                    "--fn-cost",
                    str(fn),
                    "--review-capacity",
                    str(capacity),
                    "--defer-costs",
                    *map(str, args.defer_costs),
                    "--jobs",
                    str(args.jobs),
                    "--output-dir",
                    str(output),
                    "--resume",
                    *args.benchmark_args,
                ]
                print("Running", name, flush=True)
                subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
