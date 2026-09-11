"""Run multiple cache policies over the same request stream and compare hit
rates head-to-head.

The whole point of this toolkit is the comparison, not any one policy in
isolation: `compare_policies` takes one workload (a list of keys) and runs
every policy class in `DEFAULT_POLICIES` against it at the same capacity, so
their hit rates are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Type

from .policies import CachePolicy, LRUCache, LFUCache, RandomCache, TinyLFUCache

DEFAULT_POLICIES: Dict[str, Type[CachePolicy]] = {
    "random": RandomCache,
    "lru": LRUCache,
    "lfu": LFUCache,
    "tinylfu": TinyLFUCache,
}


@dataclass
class PolicyResult:
    name: str
    capacity: int
    hits: int
    misses: int
    hit_rate: float
    extra: dict = field(default_factory=dict)


def run_policy(policy: CachePolicy, stream: Sequence) -> PolicyResult:
    """Replay `stream` through `policy` and summarize the outcome."""
    for key in stream:
        policy.access(key)
    extra: dict = {}
    if isinstance(policy, TinyLFUCache):
        extra = {
            "admitted": policy.admitted,
            "rejected": policy.rejected,
            "window_capacity": policy.window_capacity,
            "main_capacity": policy.main_capacity,
        }
    return PolicyResult(
        name=policy.name,
        capacity=policy.capacity,
        hits=policy.hits,
        misses=policy.misses,
        hit_rate=policy.hit_rate,
        extra=extra,
    )


def compare_policies(
    stream: Sequence,
    capacity: int,
    policies: Optional[Dict[str, Type[CachePolicy]]] = None,
    seed: Optional[int] = None,
) -> List[PolicyResult]:
    """Run every policy in `policies` (default: random/LRU/LFU/TinyLFU) over
    the same `stream` at the same `capacity`, each built fresh (empty), and
    return one `PolicyResult` per policy in insertion order.
    """
    policies = policies or DEFAULT_POLICIES
    results = []
    for _name, cls in policies.items():
        kwargs: dict = {"capacity": capacity}
        if cls is RandomCache:
            kwargs["seed"] = seed
        elif cls is TinyLFUCache:
            kwargs["rng_seed"] = seed
        policy = cls(**kwargs)
        results.append(run_policy(policy, stream))
    return results


def format_table(results: List[PolicyResult]) -> str:
    """Render `compare_policies` output as a plain-text aligned table."""
    header = f"{'policy':<10}{'capacity':>10}{'hits':>10}{'misses':>10}{'hit_rate':>12}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.name:<10}{r.capacity:>10}{r.hits:>10}{r.misses:>10}{r.hit_rate:>12.4f}"
        )
    return "\n".join(lines)
