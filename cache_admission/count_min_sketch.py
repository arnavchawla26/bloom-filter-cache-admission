"""Count-Min Sketch frequency estimator with conservative update and aging.

A Count-Min Sketch (CMS) estimates how many times each item in a stream has
been seen, using sub-linear space by hashing each item into `depth`
independent rows of `width` counters and taking the *minimum* counter across
rows as the estimate. Collisions can only ever push the estimate up, never
down, so a CMS never underestimates a true count — but it may overestimate.

This sketch is the frequency source behind the TinyLFU admission policy in
`cache_admission.policies.TinyLFUCache`: deciding "is this new item hotter
than what it would evict?" only needs an *approximate* frequency comparison,
which is exactly the cheap, constant-space operation a CMS provides.
"""
from __future__ import annotations

import hashlib
from typing import Hashable, Iterator, Tuple


def _to_bytes(item: Hashable) -> bytes:
    """Turn any hashable item into a stable byte string for hashing.

    Using `repr()` (rather than Python's built-in `hash()`) keeps hashing
    deterministic across runs/processes regardless of `PYTHONHASHSEED`,
    which matters for reproducible benchmarks and for tests that assert on
    exact estimate values.
    """
    if isinstance(item, bytes):
        return item
    if isinstance(item, str):
        return item.encode("utf-8")
    return repr(item).encode("utf-8")


class CountMinSketch:
    """A Count-Min Sketch with optional conservative update and manual aging.

    Parameters
    ----------
    width:
        Number of counters per row. Larger width lowers overestimation error.
    depth:
        Number of independent hash rows. Larger depth lowers the probability
        that every row collides badly for a given item.
    conservative:
        If True, use "conservative update" (Estan & Varghese): on `add`,
        only counters that are *below* the new post-increment estimate get
        raised to that value, instead of unconditionally incrementing every
        row. This strictly never produces a *larger* overestimate than naive
        CMS for the same stream — see
        `tests/test_count_min_sketch.py::test_conservative_reduces_overestimation`
        for a measured comparison. The tradeoff: a conservative sketch is no
        longer exactly mergeable/subtractable the way a naive CMS is.
    max_counter:
        Counters saturate at this value instead of growing unboundedly. Real
        TinyLFU implementations (e.g. Caffeine) use 4-bit counters (max 15)
        to keep the sketch tiny; this toolkit defaults to the same bound for
        realism, but it is just a constructor argument.
    """

    def __init__(
        self,
        width: int = 256,
        depth: int = 4,
        conservative: bool = True,
        max_counter: int = 15,
    ) -> None:
        if width < 1:
            raise ValueError("width must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if max_counter < 1:
            raise ValueError("max_counter must be >= 1")
        self.width = width
        self.depth = depth
        self.conservative = conservative
        self.max_counter = max_counter
        self._rows = [[0] * width for _ in range(depth)]
        self._additions = 0  # count of .add() calls since the last .reset()

    def _indices(self, item: Hashable) -> Iterator[Tuple[int, int]]:
        data = _to_bytes(item)
        for row in range(self.depth):
            digest = hashlib.blake2b(
                data, digest_size=8, person=str(row).encode("ascii")
            ).digest()
            yield row, int.from_bytes(digest, "big") % self.width

    def add(self, item: Hashable, count: int = 1) -> None:
        """Record `count` (default 1) more occurrences of `item`."""
        if count <= 0:
            raise ValueError("count must be positive")
        positions = list(self._indices(item))
        if self.conservative:
            current_min = min(self._rows[row][col] for row, col in positions)
            new_value = min(current_min + count, self.max_counter)
            for row, col in positions:
                if self._rows[row][col] < new_value:
                    self._rows[row][col] = new_value
        else:
            for row, col in positions:
                cell = self._rows[row]
                cell[col] = min(cell[col] + count, self.max_counter)
        self._additions += 1

    def estimate(self, item: Hashable) -> int:
        """Return the estimated count for `item` (>= the true count)."""
        return min(self._rows[row][col] for row, col in self._indices(item))

    def reset(self) -> None:
        """Halve every counter (integer division) and zero the add counter.

        This "aging" step keeps the sketch biased toward recent activity
        instead of accumulating frequency forever. Without it, an item that
        was extremely popular early in a long-running stream could keep
        winning admission races against genuinely hot *current* items long
        after it stopped being requested — see
        `cache_admission.workload.shifting_zipfian` and
        `tests/test_policies.py::test_tinylfu_adapts_after_aging_reset`.
        """
        for row in self._rows:
            for i in range(len(row)):
                row[i] //= 2
        self._additions = 0

    @property
    def additions(self) -> int:
        """Number of `add()` calls since the sketch was created or reset."""
        return self._additions
