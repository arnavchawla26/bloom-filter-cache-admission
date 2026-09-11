"""Fixed-capacity cache eviction/admission policies, compared head-to-head by
`cache_admission.benchmark`.

Every policy implements the same tiny interface (`access(key) -> bool`, plus
`hits`/`misses`/`hit_rate`), so any of them can be dropped into a benchmark
loop and run over the exact same request stream for an apples-to-apples hit
rate comparison.
"""
from __future__ import annotations

import random
from collections import OrderedDict
from typing import Dict, Hashable, Optional

from .count_min_sketch import CountMinSketch


class CachePolicy:
    """Common interface for a fixed-capacity cache policy.

    Subclasses implement `access`, which records one request for `key` and
    returns True on a cache hit, False on a miss (handling any eviction or
    admission bookkeeping as a side effect). `hit_rate` over a workload is
    the metric used to compare policies.
    """

    name = "base"

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def access(self, key: Hashable) -> bool:
        raise NotImplementedError

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, key: Hashable) -> bool:
        raise NotImplementedError


class LRUCache(CachePolicy):
    """Evicts the least-recently-used key. Admits every new arrival."""

    name = "LRU"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._data: "OrderedDict[Hashable, None]" = OrderedDict()

    def access(self, key: Hashable) -> bool:
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        if len(self._data) >= self.capacity:
            self._data.popitem(last=False)
        self._data[key] = None
        return False

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._data


class LFUCache(CachePolicy):
    """Evicts the key with the lowest *exact* access count seen so far.

    Ties are broken by recency (oldest-inserted-among-tied loses), so this
    needs an `_order` structure alongside the frequency counts. Exact LFU is
    a reasonable baseline but has a well-known weakness versus TinyLFU: a key
    that was hot long ago keeps a high count forever and can keep winning
    eviction battles against genuinely-currently-hot keys, since there is no
    aging/decay.
    """

    name = "LFU"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._freq: Dict[Hashable, int] = {}
        self._order: "OrderedDict[Hashable, None]" = OrderedDict()

    def access(self, key: Hashable) -> bool:
        if key in self._freq:
            self._freq[key] += 1
            self.hits += 1
            return True
        self.misses += 1
        if len(self._freq) >= self.capacity:
            victim = min(self._order, key=lambda k: self._freq[k])
            del self._freq[victim]
            del self._order[victim]
        self._freq[key] = 1
        self._order[key] = None
        return False

    def __len__(self) -> int:
        return len(self._freq)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._freq


class RandomCache(CachePolicy):
    """Evicts a uniformly random resident key. The naive baseline."""

    name = "Random"

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        super().__init__(capacity)
        self._data: Dict[Hashable, None] = {}
        self._rng = random.Random(seed)

    def access(self, key: Hashable) -> bool:
        if key in self._data:
            self.hits += 1
            return True
        self.misses += 1
        if len(self._data) >= self.capacity:
            victim = self._rng.choice(list(self._data.keys()))
            del self._data[victim]
        self._data[key] = None
        return False

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._data


class TinyLFUCache(CachePolicy):
    """A window-based TinyLFU cache (the policy behind Caffeine's cache).

    The capacity is split into a small LRU *window* (admits every new key
    unconditionally) and a larger *main* LRU segment. Whatever key the
    window evicts is only *promoted* into main if a Count-Min Sketch
    frequency estimate says it is at least as hot as main's own LRU victim;
    otherwise the candidate is discarded outright and never occupies any
    cache slot. That's the whole admission policy: frequency-gate what gets
    past the small recency-only front door.

    This is a simplified version of the real Caffeine TinyLFU, which splits
    `main` itself into a segmented LRU (probation + protected regions); here
    `main` is a single plain LRU segment, which keeps the implementation
    small while preserving the part being benchmarked — the admission
    decision itself.
    """

    name = "TinyLFU"

    def __init__(
        self,
        capacity: int,
        window_ratio: float = 0.01,
        sketch_width: int = 256,
        sketch_depth: int = 4,
        aging_multiplier: int = 100,
        rng_seed: Optional[int] = None,
    ) -> None:
        """`aging_multiplier` controls how often the frequency sketch halves
        itself: every `aging_multiplier * capacity` accesses. This has to be
        tuned relative to how *long* an adversarial burst (e.g. a sequential
        scan) can plausibly run versus the cache's own capacity, not just
        picked for looking like a sensible round number. An earlier default
        of 10 reset the sketch roughly every 10 scan-length-equivalents for a
        small cache, which repeatedly halved the hot set's earned frequency
        advantage *during* a 100x-capacity scan — by partway through, the
        hot set's decayed estimate had fallen to the same ballpark as a
        scan key's estimate of 1, so a large fraction of the scan got
        admitted into `main` anyway, defeating the entire point of the
        policy (measured in
        `tests/test_policies.py::test_tinylfu_resists_a_scan_dozens_of_times_longer_than_capacity`).
        100 was chosen as a default that survives a scan roughly two orders
        of magnitude longer than capacity without materially hurting
        adaptation to real popularity shifts (see
        `test_tinylfu_adapts_after_aging_reset`, which overrides this
        explicitly to exercise aging directly on a short stream).
        """
        super().__init__(capacity)
        if not 0 < window_ratio < 1:
            raise ValueError("window_ratio must be in (0, 1)")
        self.window_capacity = max(1, round(capacity * window_ratio))
        self.main_capacity = max(1, capacity - self.window_capacity)
        self._window: "OrderedDict[Hashable, None]" = OrderedDict()
        self._main: "OrderedDict[Hashable, None]" = OrderedDict()
        self._sketch = CountMinSketch(width=sketch_width, depth=sketch_depth)
        self._aging_threshold = aging_multiplier * capacity
        self._rng = random.Random(rng_seed)
        self.admitted = 0
        self.rejected = 0

    def _record_access(self, key: Hashable) -> None:
        self._sketch.add(key)
        if self._sketch.additions >= self._aging_threshold:
            self._sketch.reset()

    def access(self, key: Hashable) -> bool:
        self._record_access(key)

        if key in self._window:
            self._window.move_to_end(key)
            self.hits += 1
            return True
        if key in self._main:
            self._main.move_to_end(key)
            self.hits += 1
            return True

        self.misses += 1
        self._window[key] = None
        if len(self._window) > self.window_capacity:
            candidate, _ = self._window.popitem(last=False)
            self._try_promote(candidate)
        return False

    def _try_promote(self, candidate: Hashable) -> None:
        """Decide whether `candidate` (just evicted from the window) earns a
        slot in `main`, admitting it directly if main has room, or running
        the TinyLFU admission race against main's own LRU victim otherwise.
        """
        if len(self._main) < self.main_capacity:
            self._main[candidate] = None
            self.admitted += 1
            return

        main_victim = next(iter(self._main))  # main's current LRU victim
        candidate_freq = self._sketch.estimate(candidate)
        victim_freq = self._sketch.estimate(main_victim)
        # Strict win admits outright; an exact tie admits with small
        # probability so an unlucky-but-equally-hot key isn't locked out
        # forever (mirrors Caffeine's randomized tie-break for low counts).
        if candidate_freq > victim_freq or (
            candidate_freq == victim_freq and self._rng.random() < 0.01
        ):
            del self._main[main_victim]
            self._main[candidate] = None
            self.admitted += 1
        else:
            self.rejected += 1

    def __len__(self) -> int:
        return len(self._window) + len(self._main)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._window or key in self._main
