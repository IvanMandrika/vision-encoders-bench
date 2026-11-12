import numpy as np
import pytest

from veb.selection import DETERMINISTIC, METHODS, USES_POOL_LABELS, class_quotas, kcenter, random

N_CLASSES = 5
PER_CLASS = 200
BALANCED = ("random_balanced", "herding", "kmeans_balanced")


def blobs(seed: int = 0, dim: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(N_CLASSES, dim))
    y = np.repeat(np.arange(N_CLASSES), PER_CLASS)
    x = centers[y] + rng.normal(size=(len(y), dim))
    perm = rng.permutation(len(y))
    return x[perm].astype(np.float32), y[perm]


def max_min_distance(x: np.ndarray, idx: np.ndarray) -> float:
    return float(np.linalg.norm(x[:, None] - x[idx][None], axis=2).min(1).max())


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("budget", [5, 50, 123])
def test_count_range_and_no_duplicates(method, budget):
    x, y = blobs()
    idx = METHODS[method](x, y, budget, np.random.default_rng(0))
    assert idx.shape == (budget,)
    assert idx.dtype == np.int64
    assert len(np.unique(idx)) == budget
    assert idx.min() >= 0 and idx.max() < len(y)


@pytest.mark.parametrize("method", BALANCED)
def test_class_balance(method):
    x, y = blobs()
    idx = METHODS[method](x, y, 50, np.random.default_rng(1))
    assert np.bincount(y[idx], minlength=N_CLASSES).tolist() == [10] * N_CLASSES


@pytest.mark.parametrize("method", BALANCED)
def test_uneven_budget_differs_by_at_most_one(method):
    x, y = blobs()
    counts = np.bincount(y[METHODS[method](x, y, 23, np.random.default_rng(2))], minlength=N_CLASSES)
    assert counts.sum() == 23
    assert counts.max() - counts.min() == 1


@pytest.mark.parametrize("method", METHODS)
def test_deterministic_given_seed(method):
    x, y = blobs()
    a = METHODS[method](x, y, 40, np.random.default_rng(7))
    b = METHODS[method](x, y, 40, np.random.default_rng(7))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("method", ["random", "random_balanced", "kcenter", "kmeans", "kmeans_balanced"])
def test_seed_changes_selection(method):
    x, y = blobs()
    a = METHODS[method](x, y, 40, np.random.default_rng(1))
    b = METHODS[method](x, y, 40, np.random.default_rng(2))
    assert not np.array_equal(np.sort(a), np.sort(b))


def test_kcenter_covers_better_than_random():
    x, y = blobs()
    for seed in range(5):
        k = max_min_distance(x, kcenter(x, y, 50, np.random.default_rng(seed)))
        r = max_min_distance(x, random(x, y, 50, np.random.default_rng(seed)))
        assert k < r


def test_kcenter_is_prefix_nested():
    x, y = blobs()
    small = kcenter(x, y, 20, np.random.default_rng(3))
    large = kcenter(x, y, 60, np.random.default_rng(3))
    assert np.array_equal(small, large[:20])


def test_kcenter_survives_duplicate_points():
    x = np.repeat(np.eye(4, dtype=np.float32), 3, axis=0)
    idx = kcenter(x, np.zeros(12, dtype=np.int64), 12, np.random.default_rng(0))
    assert sorted(idx.tolist()) == list(range(12))


def test_herding_mean_tracks_class_mean_better_than_random():
    x, y = blobs()
    def gap(idx):
        return np.mean([np.linalg.norm(x[idx][y[idx] == c].mean(0) - x[y == c].mean(0)) for c in range(N_CLASSES)])
    herd = gap(METHODS["herding"](x, y, 50, np.random.default_rng(0)))
    rand = gap(METHODS["random_balanced"](x, y, 50, np.random.default_rng(0)))
    assert herd < rand


def test_kmeans_takes_whole_class_when_quota_equals_class_size():
    x, y = blobs()
    idx = METHODS["kmeans_balanced"](x, y, len(y), np.random.default_rng(0))
    assert sorted(idx.tolist()) == list(range(len(y)))


def test_quota_larger_than_class_raises():
    y = np.array([0, 0, 0, 1])
    with pytest.raises(ValueError):
        class_quotas(y, 4, np.random.default_rng(0))


@pytest.mark.parametrize("method", [m for m, uses in USES_POOL_LABELS.items() if not uses])
def test_label_free_methods_ignore_labels(method):
    x, y = blobs()
    a = METHODS[method](x, y, 30, np.random.default_rng(3))
    b = METHODS[method](x, np.random.default_rng(9).permutation(y), 30, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_kmeans_covers_every_blob():
    x, y = blobs()
    assert set(y[METHODS["kmeans"](x, y, 10, np.random.default_rng(4))]) == set(range(N_CLASSES))


def test_herding_is_seed_independent_for_divisible_budget():
    x, y = blobs()
    assert "herding" in DETERMINISTIC
    a = METHODS["herding"](x, y, 50, np.random.default_rng(0))
    b = METHODS["herding"](x, y, 50, np.random.default_rng(1))
    assert np.array_equal(np.sort(a), np.sort(b))
