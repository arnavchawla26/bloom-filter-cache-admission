from collections import Counter

import pytest

from cache_admission.workload import scan_resistance_trace, shifting_zipfian, zipfian


def test_zipfian_length_and_range():
    stream = zipfian(num_requests=1000, num_items=50, seed=1)
    assert len(stream) == 1000
    assert all(0 <= k < 50 for k in stream)


def test_zipfian_rejects_bad_params():
    with pytest.raises(ValueError):
        zipfian(num_requests=10, num_items=0)
    with pytest.raises(ValueError):
        zipfian(num_requests=-1, num_items=10)


def test_zipfian_is_seed_reproducible():
    s1 = zipfian(num_requests=500, num_items=30, alpha=1.2, seed=99)
    s2 = zipfian(num_requests=500, num_items=30, alpha=1.2, seed=99)
    assert s1 == s2


def test_zipfian_is_actually_skewed():
    """With alpha=1.0 over many requests, the single most-requested item
    should dominate far more than a uniform distribution would predict
    (uniform over 100 items -> ~1% each; Zipfian rank-1 should be well
    above that).
    """
    stream = zipfian(num_requests=20000, num_items=100, alpha=1.0, seed=5)
    counts = Counter(stream)
    most_common_count = counts.most_common(1)[0][1]
    assert most_common_count / len(stream) > 0.05  # >> the ~1% uniform baseline


def test_shifting_zipfian_changes_the_hot_set_across_phases():
    """The key property 'shifting' adds over plain zipfian: the identity of
    the most-requested item should generally differ between early and late
    phases, since each phase re-shuffles which item gets rank 1.
    """
    stream = shifting_zipfian(
        num_requests=20000, num_items=50, num_phases=4, alpha=1.0, seed=3
    )
    phase_len = len(stream) // 4
    phases = [stream[i * phase_len:(i + 1) * phase_len] for i in range(4)]
    top_per_phase = [Counter(p).most_common(1)[0][0] for p in phases]
    # Not every adjacent pair is guaranteed to differ (shuffle could repeat),
    # but across 4 independently-shuffled phases it would be a vanishingly
    # unlikely coincidence for all four top keys to match.
    assert len(set(top_per_phase)) > 1


def test_shifting_zipfian_total_length_matches_request():
    for num_phases in (1, 3, 7):
        stream = shifting_zipfian(num_requests=1003, num_items=20, num_phases=num_phases, seed=1)
        assert len(stream) == 1003


def test_scan_resistance_trace_layout():
    trace = scan_resistance_trace(hot_set_size=10, hot_requests=20, scan_length=15, seed=1)
    assert len(trace) == 20 + 15 + 20
    warm_up, scan, cool_down = trace[:20], trace[20:35], trace[35:]
    assert all(0 <= k < 10 for k in warm_up)
    assert all(0 <= k < 10 for k in cool_down)
    # Scan portion is the strictly-increasing never-repeated range [10, 25).
    assert scan == list(range(10, 25))
    assert len(set(scan)) == len(scan)


def test_scan_resistance_trace_rejects_bad_params():
    with pytest.raises(ValueError):
        scan_resistance_trace(hot_set_size=0, hot_requests=1, scan_length=1)
    with pytest.raises(ValueError):
        scan_resistance_trace(hot_set_size=1, hot_requests=-1, scan_length=1)
