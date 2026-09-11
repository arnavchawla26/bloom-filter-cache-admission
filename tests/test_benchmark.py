from cache_admission.benchmark import DEFAULT_POLICIES, compare_policies, format_table, run_policy
from cache_admission.policies import LRUCache
from cache_admission.workload import zipfian


def test_run_policy_reports_consistent_totals():
    stream = zipfian(num_requests=500, num_items=50, seed=1)
    cache = LRUCache(capacity=10)
    result = run_policy(cache, stream)
    assert result.hits + result.misses == len(stream)
    assert 0.0 <= result.hit_rate <= 1.0
    assert result.name == "LRU"
    assert result.extra == {}


def test_compare_policies_runs_every_default_policy_on_same_stream():
    stream = zipfian(num_requests=2000, num_items=100, alpha=1.0, seed=1)
    results = compare_policies(stream, capacity=20, seed=7)
    assert len(results) == len(DEFAULT_POLICIES)
    names = {r.name for r in results}
    assert names == {"Random", "LRU", "LFU", "TinyLFU"}
    for r in results:
        assert r.hits + r.misses == len(stream)
        assert 0.0 <= r.hit_rate <= 1.0


def test_compare_policies_is_reproducible_with_a_seed():
    stream = zipfian(num_requests=1000, num_items=80, seed=1)
    r1 = compare_policies(stream, capacity=15, seed=11)
    r2 = compare_policies(stream, capacity=15, seed=11)
    assert [(r.name, r.hits, r.misses) for r in r1] == [(r.name, r.hits, r.misses) for r in r2]


def test_tinylfu_skewed_workload_beats_random_baseline():
    """Sanity check that the benchmark plumbing reflects real quality
    differences: on a strongly skewed Zipfian workload with a tight cache,
    any frequency-aware policy should clearly outperform pure-random
    eviction (which ignores all history).
    """
    stream = zipfian(num_requests=20000, num_items=500, alpha=1.2, seed=1)
    results = compare_policies(stream, capacity=30, seed=1)
    by_name = {r.name: r for r in results}
    assert by_name["TinyLFU"].hit_rate > by_name["Random"].hit_rate


def test_format_table_contains_every_policy_name():
    stream = zipfian(num_requests=200, num_items=30, seed=1)
    results = compare_policies(stream, capacity=10, seed=1)
    table = format_table(results)
    for r in results:
        assert r.name in table
