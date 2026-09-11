"""Command-line interface: `cacheadmit benchmark` and `cacheadmit sketch-demo`."""
from __future__ import annotations

import argparse
import random
import sys
from typing import List, Optional

from .benchmark import compare_policies, format_table
from .count_min_sketch import CountMinSketch
from .workload import scan_resistance_trace, shifting_zipfian, zipfian


def _build_stream(args: argparse.Namespace) -> List[int]:
    if args.workload == "zipf":
        return zipfian(args.requests, args.items, alpha=args.alpha, seed=args.seed)
    if args.workload == "shifting":
        return shifting_zipfian(
            args.requests, args.items, num_phases=args.phases, alpha=args.alpha, seed=args.seed
        )
    if args.workload == "scan":
        return scan_resistance_trace(
            args.items, args.requests // 2, args.scan_length, seed=args.seed
        )
    raise ValueError(f"unknown workload {args.workload!r}")  # pragma: no cover


def cmd_benchmark(args: argparse.Namespace) -> int:
    stream = _build_stream(args)
    results = compare_policies(stream, capacity=args.capacity, seed=args.seed)
    print(f"workload={args.workload} requests={len(stream)} capacity={args.capacity}\n")
    print(format_table(results))
    return 0


def cmd_sketch_demo(args: argparse.Namespace) -> int:
    sketch = CountMinSketch(
        width=args.width,
        depth=args.depth,
        conservative=not args.naive,
        max_counter=args.max_counter,
    )
    rng = random.Random(args.seed)
    true_counts: dict = {}
    for _ in range(args.requests):
        item = rng.randrange(args.items)
        sketch.add(item)
        true_counts[item] = true_counts.get(item, 0) + 1

    print(f"mode={'naive' if args.naive else 'conservative'} width={args.width} depth={args.depth}\n")
    print(f"{'item':<8}{'true':>8}{'estimate':>10}{'error':>8}")
    top = sorted(true_counts, key=lambda k: -true_counts[k])[:10]
    for item in top:
        est = sketch.estimate(item)
        print(f"{item:<8}{true_counts[item]:>8}{est:>10}{est - true_counts[item]:>8}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cacheadmit",
        description="TinyLFU-style cache admission policy benchmarking toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser(
        "benchmark", help="compare eviction/admission policies on a synthetic workload"
    )
    bench.add_argument("--workload", choices=["zipf", "shifting", "scan"], default="zipf")
    bench.add_argument("--requests", type=int, default=20000)
    bench.add_argument("--items", type=int, default=1000)
    bench.add_argument("--capacity", type=int, default=100)
    bench.add_argument("--alpha", type=float, default=1.0, help="Zipfian skew parameter")
    bench.add_argument("--phases", type=int, default=4, help="only used by --workload shifting")
    bench.add_argument("--scan-length", type=int, default=2000, help="only used by --workload scan")
    bench.add_argument("--seed", type=int, default=None)
    bench.set_defaults(func=cmd_benchmark)

    sketch = sub.add_parser(
        "sketch-demo", help="show Count-Min Sketch frequency estimates vs. ground truth"
    )
    sketch.add_argument("--requests", type=int, default=50000)
    sketch.add_argument("--items", type=int, default=2000)
    sketch.add_argument("--width", type=int, default=256)
    sketch.add_argument("--depth", type=int, default=4)
    sketch.add_argument("--naive", action="store_true", help="disable conservative update")
    sketch.add_argument(
        "--max-counter",
        type=int,
        default=1_000_000,
        help=(
            "counter saturation cap. Deliberately far above the 15 used by "
            "TinyLFUCache (which only needs relative ordering, not exact "
            "magnitude) so this demo's estimates vs. true counts stay "
            "meaningful instead of silently capping out."
        ),
    )
    sketch.add_argument("--seed", type=int, default=None)
    sketch.set_defaults(func=cmd_sketch_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
