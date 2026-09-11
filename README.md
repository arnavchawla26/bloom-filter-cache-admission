# bloom-filter-cache-admission

A from-scratch, TinyLFU-style **cache admission policy** toolkit: a
Count-Min Sketch frequency estimator used to decide whether a newly-arrived
cache key is "hot enough" to be worth admitting over whatever it would
evict — plus a benchmark harness that pits it head-to-head against plain
LRU, exact LFU, and random eviction on synthetic Zipfian and adversarial
workloads.

## What it does

Most caches evict by recency (LRU) or exact frequency (LFU). Both have
weaknesses: LRU is trivially fooled by a one-off sequential scan (a backup
job, a full-table read) into evicting a genuinely hot working set for keys
that are never touched again; exact LFU needs an unbounded counter per
distinct key ever seen and never forgets old popularity, so a key that was
hot a long time ago can keep winning eviction battles forever.

[TinyLFU](https://arxiv.org/abs/1512.00727) (the policy behind the popular
[Caffeine](https://github.com/ben-manes/caffeine) Java cache) sidesteps both
problems with an **admission** policy layered on top of a small LRU window:
every new key is let into a small window unconditionally, but the key the
window evicts only gets promoted into the larger main cache if a cheap,
constant-space frequency sketch says it's at least as hot as whatever it
would evict from main. Everything else is discarded before it ever occupies
a cache slot.

This repo implements that idea from scratch:

- **`count_min_sketch.py`** — a [Count-Min
  Sketch](https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch) with
  conservative update (Estan & Varghese) and manual aging (periodic
  halving), the frequency source behind admission decisions. Never
  underestimates a true count (collisions only push estimates *up*); the
  conservative-update mode measurably reduces overestimation versus naive
  CMS (see `tests/test_count_min_sketch.py`).
- **`policies.py`** — four cache policies sharing one interface
  (`access(key) -> bool`, `hit_rate`): `RandomCache`, `LRUCache`,
  `LFUCache` (exact frequency), and `TinyLFUCache` (window + main LRU
  segments, admission gated by the sketch).
- **`workload.py`** — synthetic request-stream generators: `zipfian`
  (the standard power-law cache-research benchmark), `shifting_zipfian`
  (popularity ranking drifts over time, exercising the sketch's aging), and
  `scan_resistance_trace` (the classic adversarial trace: hammer a small
  hot set, splice in one long sequential scan, hammer the hot set again).
- **`benchmark.py`** — runs every policy over the same stream at the same
  capacity and reports hit rates side by side.
- **`cli.py`** — a `cacheadmit` command-line tool wrapping both the
  benchmark and a standalone sketch-accuracy demo.

## Tech stack

Python 3.9+, standard library only (`hashlib`, `collections.OrderedDict`,
`random`, `argparse`) — no runtime dependencies. `pytest` for tests.

## How to run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Compare all four policies on a Zipfian workload
cacheadmit benchmark --workload zipf --requests 20000 --items 1000 --capacity 100 --seed 1

# The headline demo: a sequential scan 100x longer than the cache
cacheadmit benchmark --workload scan --items 20 --requests 2800 --scan-length 2000 --capacity 20 --seed 42

# Popularity that shifts over time (exercises sketch aging)
cacheadmit benchmark --workload shifting --requests 20000 --items 500 --phases 5 --capacity 80 --seed 1

# Count-Min Sketch accuracy vs. exact ground-truth counts
cacheadmit sketch-demo --requests 50000 --items 2000 --seed 1

pytest                 # 40 tests
python -m pyflakes cache_admission tests   # lint, should be silent
```

### Library usage

```python
from cache_admission import TinyLFUCache
from cache_admission.workload import zipfian
from cache_admission.benchmark import compare_policies, format_table

stream = zipfian(num_requests=20000, num_items=1000, alpha=1.0, seed=1)
print(format_table(compare_policies(stream, capacity=100, seed=1)))

cache = TinyLFUCache(capacity=100)
for key in stream:
    hit = cache.access(key)
```

## A real bug this project caught

The first version of the scan-resistance test measured LRU's hit rate over
the *entire* post-scan "cool_down" segment and got a suspiciously high
`0.95` instead of the expected near-zero. Tracing it by hand: with
`capacity == hot_set_size`, once `cool_down` starts cycling back over
exactly those `capacity`-many keys, LRU only needs `capacity` misses to
fully rebuild itself with that exact key set — after which *every* policy
hits on every further request, since there's nowhere else a miss could come
from. The hit-rate-over-the-whole-segment metric was hiding the real signal
(how much work a policy needs to *recover* from the scan) inside that
self-healing tail; counting misses in `cool_down` instead isolates it
cleanly (see `test_tinylfu_resists_a_scan_dozens_of_times_longer_than_capacity`
in `tests/test_policies.py`).

Fixing that test surfaced a second, more interesting problem: with the
corrected metric, **TinyLFU was losing to LRU** (more cool_down misses, not
fewer) at the default `aging_multiplier=10`. The frequency sketch was
resetting (halving) every `10 * capacity` accesses — for a capacity-20
cache, that's every 200 accesses, i.e. roughly every 10 items of a
2000-item scan. Repeatedly halving the hot set's earned frequency
advantage *during* the scan let its estimate decay down to the same
ballpark as a scan key's estimate of 1 partway through, so a large fraction
of the scan got admitted into `main` anyway — defeating the entire point of
the policy. The default was changed to `aging_multiplier=100`, which
survives a scan roughly two orders of magnitude longer than capacity
without materially hurting the sketch's ability to adapt to genuine
popularity shifts (`test_tinylfu_adapts_after_aging_reset` covers that
separately with an explicit low multiplier on a short stream). Measured
result with real numbers, same seed: LRU pays 20 misses to recover its
20-key hot set from a 2000-key scan; TinyLFU at `aging_multiplier=100` pays
2.

A smaller, separate catch: the CLI's `sketch-demo` subcommand initially
reused `CountMinSketch`'s class default of `max_counter=15` (the right
choice *for TinyLFU's admission use case*, which only needs relative
ordering, not exact magnitude — real TinyLFU implementations use 4-bit
counters to stay tiny). For a demo whose entire purpose is showing estimate
accuracy against ground truth, that silently capped every moderately-popular
item's estimate at 15 regardless of its real count, making the demo look
badly broken (`error: -55` for an item seen 70 times). `sketch-demo` now
takes its own `--max-counter` (default `1,000,000`), independent of the
cache-admission default.

## Design notes / simplifications

- `TinyLFUCache`'s `main` segment is a single plain LRU, not Caffeine's
  real segmented LRU (probation + protected regions). The simplification
  keeps the implementation small while preserving the part actually being
  benchmarked: the admission decision at the window/main boundary.
- The Count-Min Sketch hashes via `hashlib.blake2b` seeded by row index
  rather than Python's built-in `hash()`, so results are deterministic
  across processes/runs regardless of `PYTHONHASHSEED` — required for
  reproducible benchmarks and exact-value tests.

## Current status

Complete, tested v1. All four policies, all three workload generators, the
benchmark harness, and the `cacheadmit` CLI (`benchmark` + `sketch-demo`)
are implemented and covered by 40 passing tests (`pytest`), plus a clean
`pyflakes` run. Verified via a fresh-venv install, the full test suite, a
clone-vs-local diff, and manual CLI smoke tests.

Possible future extensions (not started): a real segmented-LRU `main`
region (probation/protected, matching Caffeine exactly); reading/replaying
a real access-log trace file instead of only synthetic workloads; plotting
hit-rate-vs-capacity curves across a sweep rather than single-point
comparisons.

## License

MIT (see `LICENSE`).
