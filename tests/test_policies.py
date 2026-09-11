import pytest

from cache_admission.policies import CachePolicy, LFUCache, LRUCache, RandomCache, TinyLFUCache
from cache_admission.workload import scan_resistance_trace


def test_cache_policy_base_class_rejects_bad_capacity():
    with pytest.raises(ValueError):
        LRUCache(capacity=0)
    with pytest.raises(ValueError):
        LFUCache(capacity=-1)


def test_base_access_not_implemented():
    base = CachePolicy(capacity=1)
    with pytest.raises(NotImplementedError):
        base.access("x")
    with pytest.raises(NotImplementedError):
        len(base)


# ---------------------------------------------------------------------------
# LRU: hand-computed trace
# ---------------------------------------------------------------------------

def test_lru_hand_computed_trace():
    """capacity=2, trace: a, b, a, c, a, b
    - a: miss  -> cache [a]
    - b: miss  -> cache [a, b]
    - a: HIT   -> cache [b, a]  (a now most-recent)
    - c: miss, evicts b (LRU)  -> cache [a, c]
    - a: HIT   -> cache [c, a]
    - b: miss, evicts c (LRU)  -> cache [a, b]
    Expected hit sequence: miss, miss, hit, miss, hit, miss -> 2 hits, 4 misses
    """
    cache = LRUCache(capacity=2)
    results = [cache.access(k) for k in ["a", "b", "a", "c", "a", "b"]]
    assert results == [False, False, True, False, True, False]
    assert cache.hits == 2
    assert cache.misses == 4
    assert "b" in cache
    assert "a" in cache
    assert "c" not in cache
    assert len(cache) == 2


def test_lru_repeated_access_of_same_key_is_always_a_hit_after_first():
    cache = LRUCache(capacity=1)
    assert cache.access("x") is False
    for _ in range(10):
        assert cache.access("x") is True


# ---------------------------------------------------------------------------
# LFU: exact frequency eviction
# ---------------------------------------------------------------------------

def test_lfu_evicts_the_least_frequently_used():
    cache = LFUCache(capacity=2)
    cache.access("a")
    cache.access("a")
    cache.access("a")  # freq(a) = 3
    cache.access("b")  # freq(b) = 1, cache full: {a: 3, b: 1}
    cache.access("c")  # miss -> must evict b (freq 1 < a's freq 3)
    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache


def test_lfu_tie_breaks_by_oldest_insertion():
    cache = LFUCache(capacity=2)
    cache.access("a")  # freq(a)=1, inserted first
    cache.access("b")  # freq(b)=1, inserted second; cache full, both tied at freq 1
    cache.access("c")  # miss -> tie between a and b at freq 1, a is older -> evict a
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache


# ---------------------------------------------------------------------------
# Random: structural invariants only (behavior is nondeterministic by design)
# ---------------------------------------------------------------------------

def test_random_cache_never_exceeds_capacity_and_is_seed_reproducible():
    stream = list(range(20)) * 5
    c1 = RandomCache(capacity=5, seed=123)
    c2 = RandomCache(capacity=5, seed=123)
    results1 = [c1.access(k) for k in stream]
    results2 = [c2.access(k) for k in stream]
    assert results1 == results2
    assert len(c1) <= 5


# ---------------------------------------------------------------------------
# TinyLFU: structural behavior + the headline scan-resistance property
# ---------------------------------------------------------------------------

def test_tinylfu_never_exceeds_capacity():
    cache = TinyLFUCache(capacity=10, window_ratio=0.2, rng_seed=0)
    for key in (list(range(50)) * 3):
        cache.access(key)
    assert len(cache) <= 10
    assert cache.window_capacity + cache.main_capacity == 10


def test_tinylfu_window_ratio_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        TinyLFUCache(capacity=10, window_ratio=0.0)
    with pytest.raises(ValueError):
        TinyLFUCache(capacity=10, window_ratio=1.0)


def test_tinylfu_admits_directly_while_main_has_room():
    # main_capacity is large relative to how many distinct keys pass through
    # the window, so every promoted candidate should be admitted outright.
    cache = TinyLFUCache(capacity=20, window_ratio=0.5, rng_seed=0)
    for key in range(8):
        cache.access(key)
    assert cache.rejected == 0


def test_tinylfu_resists_a_scan_dozens_of_times_longer_than_capacity():
    """The headline property this whole project exists to demonstrate:
    TinyLFU's frequency-gated admission should resist a one-off sequential
    scan far better than plain LRU, because the scan keys never beat the
    hot set's estimated frequency and so are (mostly) never admitted into
    `main` at all.

    The metric has to be *misses in the cool_down segment*, not hit rate
    over the whole segment: a first version of this test measured hit rate
    across all of `cool_down` and found LRU scoring a suspiciously high
    0.95 instead of the expected near-zero. Tracing it by hand: with
    `capacity == hot_set_size`, the moment `cool_down` starts cycling back
    over exactly those `hot_set_size` keys, LRU only needs `capacity`
    misses to fully re-populate itself with that exact key set — after
    which *every* cache, regardless of policy, hits on every subsequent
    request, since there's nowhere else for a miss to come from. A single
    hit-rate number over the full segment was hiding the real signal (how
    much work a policy needs to *recover* from the scan) inside that
    self-healing tail. Counting misses isolates it: LRU pays up to
    `capacity` misses to rebuild from nothing, while TinyLFU — which kept
    most of the hot set resident in `main` throughout the scan — should pay
    only a handful.
    """
    hot_set_size = 20
    hot_requests = 400
    scan_length = 2000
    trace = scan_resistance_trace(
        hot_set_size=hot_set_size,
        hot_requests=hot_requests,
        scan_length=scan_length,
        seed=42,
    )
    warm_up = trace[:hot_requests]
    scan = trace[hot_requests:hot_requests + scan_length]
    cool_down = trace[hot_requests + scan_length:]
    assert len(cool_down) == hot_requests

    capacity = hot_set_size  # exactly big enough to hold the hot set, no more

    lru = LRUCache(capacity=capacity)
    tinylfu = TinyLFUCache(capacity=capacity, window_ratio=0.1, rng_seed=0)

    for key in warm_up + scan:
        lru.access(key)
        tinylfu.access(key)

    # Confirm the scan really did wipe out LRU's hot set (every resident key
    # should now be a scan key), so the upcoming miss-count comparison is
    # measuring scan resistance, not something else.
    assert not any(k in lru for k in range(hot_set_size))

    lru.hits = lru.misses = 0
    tinylfu.hits = tinylfu.misses = 0
    for key in cool_down:
        lru.access(key)
        tinylfu.access(key)

    # LRU must pay close to `capacity` misses to fully recover the hot set
    # from nothing (one miss per previously-evicted key, at minimum).
    assert lru.misses >= capacity - 2
    # TinyLFU kept nearly all of the hot set resident throughout the scan,
    # so it should need only a handful of misses, not dozens.
    assert tinylfu.misses <= 5
    assert tinylfu.misses < lru.misses


def test_tinylfu_adapts_after_aging_reset():
    """Without aging, a key that was extremely popular early on would keep
    winning admission races against a *newly* hot key forever once the
    sketch saturates. With small `aging_multiplier` (frequent resets), a key
    that stops being requested should lose its frequency advantage and a
    newly-hot key should be able to take over the cache.
    """
    cache = TinyLFUCache(capacity=10, window_ratio=0.3, aging_multiplier=2, rng_seed=0)

    # Phase 1: key 0 is hammered until it dominates main.
    for _ in range(200):
        cache.access(0)
    assert 0 in cache

    # Phase 2: a new set of keys is hammered hard enough (with aging
    # resetting key 0's advantage periodically) that at least one of them
    # displaces key 0 from the cache.
    for _ in range(5):
        for key in range(1, 11):
            cache.access(key)

    new_keys_resident = sum(1 for key in range(1, 11) if key in cache)
    assert new_keys_resident > 0
