import random

import pytest

from cache_admission.count_min_sketch import CountMinSketch


def test_never_underestimates():
    """The core CMS correctness property: estimate(x) >= true_count(x),
    always, regardless of hash collisions. Checked against an exact dict of
    true counts over a biggish random stream.

    This only holds for a sketch whose counters don't saturate below the
    true counts involved — `max_counter` is a deliberate design choice (real
    TinyLFU implementations cap counters at 4 bits to stay tiny), and a
    saturated counter *can* legitimately read below a true count that
    exceeded the cap (see `test_counters_saturate_at_max_counter`). An
    earlier version of this test used the class default (`max_counter=15`)
    and failed nondeterministically once an item's true count climbed past
    15 — that was a test-design mistake, not an implementation bug, so here
    `max_counter` is set far above anything this stream can produce.
    """
    sketch = CountMinSketch(width=64, depth=4, max_counter=1_000_000)
    rng = random.Random(1)
    true_counts: dict = {}
    for _ in range(5000):
        item = rng.randrange(500)
        sketch.add(item)
        true_counts[item] = true_counts.get(item, 0) + 1

    for item, true_count in true_counts.items():
        assert sketch.estimate(item) >= true_count


def test_unseen_item_estimates_zero_with_enough_width():
    """With a wide-enough sketch relative to the stream, an item that was
    never added should estimate to exactly 0 (no accidental collisions).
    """
    sketch = CountMinSketch(width=4096, depth=4)
    for item in range(50):
        sketch.add(item)
    assert sketch.estimate("never-added") == 0


def test_deterministic_across_instances():
    """Two freshly constructed sketches fed the identical sequence of adds
    must produce identical estimates — hashing must not depend on
    PYTHONHASHSEED or any other per-process randomness.
    """
    def build():
        s = CountMinSketch(width=32, depth=3)
        for item in ["a", "b", "a", 42, (1, 2), "a", "b"]:
            s.add(item)
        return s

    s1, s2 = build(), build()
    for item in ["a", "b", 42, (1, 2), "never-seen"]:
        assert s1.estimate(item) == s2.estimate(item)


def test_add_rejects_non_positive_count():
    sketch = CountMinSketch()
    with pytest.raises(ValueError):
        sketch.add("x", count=0)
    with pytest.raises(ValueError):
        sketch.add("x", count=-1)


def test_constructor_validates_dimensions():
    with pytest.raises(ValueError):
        CountMinSketch(width=0)
    with pytest.raises(ValueError):
        CountMinSketch(depth=0)
    with pytest.raises(ValueError):
        CountMinSketch(max_counter=0)


def test_counters_saturate_at_max_counter():
    sketch = CountMinSketch(width=16, depth=2, max_counter=5)
    for _ in range(100):
        sketch.add("hot")
    assert sketch.estimate("hot") == 5


def test_reset_halves_counters_and_zeros_addition_count():
    sketch = CountMinSketch(width=16, depth=2, max_counter=100)
    for _ in range(10):
        sketch.add("x")
    before = sketch.estimate("x")
    assert before == 10
    assert sketch.additions == 10

    sketch.reset()
    assert sketch.additions == 0
    # Integer-halved, not necessarily exact if collisions inflated counters,
    # but for a dedicated wide sketch with one item it is exact here.
    assert sketch.estimate("x") == before // 2


def test_reset_can_eventually_zero_out_a_cold_item():
    """Repeated halving of a small count converges to (and stays at) zero,
    which is exactly the 'forget old popularity' behavior aging exists for.
    """
    sketch = CountMinSketch(width=16, depth=2)
    sketch.add("x")
    sketch.add("x")
    sketch.add("x")  # count = 3
    for _ in range(5):
        sketch.reset()
    assert sketch.estimate("x") == 0


def test_conservative_reduces_overestimation():
    """Conservative update must never produce a *larger* overestimate than
    naive CMS for the same stream, and on a skewed stream with real
    collisions it should produce a strictly smaller total overestimation
    error. This is the measured justification for defaulting to
    conservative=True.
    """
    width, depth = 32, 3  # deliberately narrow/shallow to force collisions
    naive = CountMinSketch(width=width, depth=depth, conservative=False, max_counter=10_000)
    conservative = CountMinSketch(width=width, depth=depth, conservative=True, max_counter=10_000)

    rng = random.Random(7)
    stream = [rng.randrange(200) for _ in range(4000)]
    true_counts: dict = {}
    for item in stream:
        naive.add(item)
        conservative.add(item)
        true_counts[item] = true_counts.get(item, 0) + 1

    naive_error = sum(naive.estimate(k) - v for k, v in true_counts.items())
    conservative_error = sum(conservative.estimate(k) - v for k, v in true_counts.items())

    assert conservative_error <= naive_error
    assert conservative_error < naive_error  # strictly better on this seeded, collision-heavy stream
    for k in true_counts:
        assert conservative.estimate(k) <= naive.estimate(k)
