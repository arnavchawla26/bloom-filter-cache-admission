"""bloom-filter-cache-admission: a TinyLFU-style, frequency-aware cache
admission policy toolkit.

The core idea (popularized by Caffeine's TinyLFU): don't just evict by
recency (LRU) or exact frequency counts (LFU, which is expensive and slow to
adapt) — keep a small, cheap, probabilistic frequency sketch (a Count-Min
Sketch) and use it to decide whether a *newly arriving* item is hot enough
to be worth admitting over whatever it would evict.

See `cache_admission.count_min_sketch.CountMinSketch` for the sketch,
`cache_admission.policies` for the cache policies (LRU / LFU / Random /
TinyLFU) being benchmarked against each other, and `cache_admission.workload`
for the synthetic traffic generators used to compare them.
"""

from .count_min_sketch import CountMinSketch
from .policies import CachePolicy, LRUCache, LFUCache, RandomCache, TinyLFUCache

__all__ = [
    "CountMinSketch",
    "CachePolicy",
    "LRUCache",
    "LFUCache",
    "RandomCache",
    "TinyLFUCache",
]

__version__ = "0.1.0"
