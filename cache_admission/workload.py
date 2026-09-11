"""Synthetic request-stream generators for benchmarking cache policies.

Real cache-replacement research almost always benchmarks against Zipfian
(power-law) popularity distributions, because real traffic (web requests,
database key access, CDN hits) is famously Zipfian-shaped: a small set of
keys gets most of the requests. This module also includes a "concept drift"
variant (popularity ranking changes over time) and the classic adversarial
scan-resistance trace used to demonstrate why pure-recency (LRU) eviction
can be fooled in ways frequency-aware policies are not.
"""
from __future__ import annotations

import random
from typing import List, Optional


def zipfian(
    num_requests: int,
    num_items: int,
    alpha: float = 1.0,
    seed: Optional[int] = None,
) -> List[int]:
    """`num_requests` keys drawn from `range(num_items)`, Zipfian-weighted so
    rank `r`'s request probability is proportional to `1 / r**alpha`.

    Which item *gets* which rank is itself randomized (via `seed`), so key
    values aren't trivially sorted by hotness the way a textbook example
    might be — a real trace's popular keys aren't conveniently labeled 0, 1,
    2, ...
    """
    if num_items < 1:
        raise ValueError("num_items must be >= 1")
    if num_requests < 0:
        raise ValueError("num_requests must be >= 0")
    rng = random.Random(seed)
    weights = [1.0 / (rank ** alpha) for rank in range(1, num_items + 1)]
    items = list(range(num_items))
    rng.shuffle(items)
    return rng.choices(items, weights=weights, k=num_requests)


def shifting_zipfian(
    num_requests: int,
    num_items: int,
    num_phases: int = 4,
    alpha: float = 1.0,
    seed: Optional[int] = None,
) -> List[int]:
    """Like `zipfian`, but which keys are hot is re-shuffled every
    `num_requests / num_phases` requests ("concept drift": what's popular
    changes over time). This is the scenario `CountMinSketch.reset()`
    (periodic aging) exists for — without it, keys that were hot in an early
    phase keep winning TinyLFU admission races long after they've gone
    cold.
    """
    if num_phases < 1:
        raise ValueError("num_phases must be >= 1")
    if num_items < 1:
        raise ValueError("num_items must be >= 1")
    rng = random.Random(seed)
    phase_len = num_requests // num_phases
    weights = [1.0 / (rank ** alpha) for rank in range(1, num_items + 1)]
    stream: List[int] = []
    for phase in range(num_phases):
        items = list(range(num_items))
        rng.shuffle(items)
        remaining_phases = num_phases - phase
        length = (num_requests - len(stream)) if remaining_phases == 1 else phase_len
        stream.extend(rng.choices(items, weights=weights, k=length))
    return stream


def scan_resistance_trace(
    hot_set_size: int,
    hot_requests: int,
    scan_length: int,
    seed: Optional[int] = None,
) -> List[int]:
    """The classic adversarial trace used to show TinyLFU's advantage over
    plain LRU: hammer a small "hot set" of keys, splice in one long
    sequential *scan* over keys that are each touched exactly once (a backup
    job, a full-table scan, a one-off batch read), then hammer the original
    hot set again.

    A pure-recency policy (LRU) is fooled into evicting the entire hot set
    to make room for the scan, even though not a single scanned key is ever
    requested again. A frequency-aware policy should refuse to admit scan
    keys over the proven-hot set, since the scan keys' estimated frequency
    (1) never beats the hot set's.

    Layout: `hot_requests` uniformly-random accesses to `range(hot_set_size)`,
    then `scan_length` strictly-increasing never-repeated keys starting at
    `hot_set_size`, then `hot_requests` more hot-set accesses so a
    "how much of the hot set survived the scan" hit rate can be measured on
    just that final segment.
    """
    if hot_set_size < 1:
        raise ValueError("hot_set_size must be >= 1")
    if hot_requests < 0 or scan_length < 0:
        raise ValueError("hot_requests and scan_length must be >= 0")
    rng = random.Random(seed)
    warm_up = [rng.randrange(hot_set_size) for _ in range(hot_requests)]
    scan = list(range(hot_set_size, hot_set_size + scan_length))
    cool_down = [rng.randrange(hot_set_size) for _ in range(hot_requests)]
    return warm_up + scan + cool_down
