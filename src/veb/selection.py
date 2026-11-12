from collections.abc import Callable

import numpy as np
import torch
from sklearn.cluster import KMeans

Selector = Callable[[np.ndarray, np.ndarray, int, np.random.Generator], np.ndarray]


def class_quotas(y: np.ndarray, budget: int, rng: np.random.Generator) -> dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    base, extra = divmod(budget, len(classes))
    quotas = np.full(len(classes), base)
    quotas[rng.permutation(len(classes))[:extra]] += 1
    if (quotas > counts).any():
        raise ValueError(f"budget {budget} exceeds the size of at least one class")
    return dict(zip(classes.tolist(), quotas.tolist()))


def _per_class(y: np.ndarray, budget: int, rng: np.random.Generator, pick: Callable[[np.ndarray, int], np.ndarray]) -> np.ndarray:
    parts = []
    for cls, quota in class_quotas(y, budget, rng).items():
        members = np.flatnonzero(y == cls)
        parts.append(members[pick(members, quota)])
    return np.concatenate(parts).astype(np.int64)


def random(x: np.ndarray, y: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(y), size=budget, replace=False).astype(np.int64)


def random_balanced(x: np.ndarray, y: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    return _per_class(y, budget, rng, lambda members, k: rng.choice(len(members), size=k, replace=False))


def kcenter(x: np.ndarray, y: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    sq_norm = np.einsum("ij,ij->i", x, x)
    min_dist = np.full(len(x), np.inf, dtype=np.float32)
    chosen = np.empty(budget, dtype=np.int64)
    chosen[0] = rng.integers(len(x))
    for i in range(1, budget):
        last = chosen[i - 1]
        np.minimum(min_dist, sq_norm + sq_norm[last] - 2.0 * (x @ x[last]), out=min_dist)
        min_dist[last] = -np.inf
        chosen[i] = min_dist.argmax()
    return chosen


def _herd(z: np.ndarray, k: int) -> np.ndarray:
    z = z.astype(np.float64)
    mean = z.mean(0)
    half_sq_norm = 0.5 * np.einsum("ij,ij->i", z, z)
    running_sum = np.zeros_like(mean)
    available = np.ones(len(z), dtype=bool)
    chosen = np.empty(k, dtype=np.int64)
    for t in range(k):
        score = z @ ((t + 1) * mean - running_sum) - half_sq_norm
        chosen[t] = np.where(available, score, -np.inf).argmax()
        available[chosen[t]] = False
        running_sum += z[chosen[t]]
    return chosen


def herding(x: np.ndarray, y: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    return _per_class(y, budget, rng, lambda members, k: _herd(x[members], k))


def _kmeans_medoids(z: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k >= len(z):
        return np.arange(len(z))
    km = KMeans(n_clusters=k, n_init=1, random_state=int(rng.integers(2**31 - 1))).fit(z)
    return _nearest_members(z, km.labels_, km.cluster_centers_, k, rng)


def _nearest_members(z: np.ndarray, labels: np.ndarray, centres: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    dist = np.linalg.norm(z - centres[labels], axis=1)
    order = np.lexsort((dist, labels))
    first = np.r_[True, np.diff(labels[order]) > 0]
    chosen = order[first]
    if len(chosen) < k:
        rest = np.setdiff1d(np.arange(len(z)), chosen)
        chosen = np.r_[chosen, rng.choice(rest, size=k - len(chosen), replace=False)]
    return chosen


def _lloyd(z: np.ndarray, k: int, rng: np.random.Generator, iters: int) -> tuple[np.ndarray, np.ndarray]:
    zt = torch.as_tensor(z, dtype=torch.float32)
    centres = zt[torch.as_tensor(rng.choice(len(z), size=k, replace=False))].clone()
    for _ in range(iters):
        labels = torch.cdist(zt, centres).argmin(1)
        counts = torch.bincount(labels, minlength=k)
        sums = torch.zeros_like(centres).index_add_(0, labels, zt)
        filled = counts > 0
        centres[filled] = sums[filled] / counts[filled, None]
    labels = torch.cdist(zt, centres).argmin(1)
    return labels.numpy(), centres.numpy()


def kmeans(x: np.ndarray, y: np.ndarray, budget: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    labels, centres = _lloyd(x, budget, rng, iters)
    return _nearest_members(np.asarray(x, dtype=np.float32), labels, centres, budget, rng).astype(np.int64)


def kmeans_balanced(x: np.ndarray, y: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    return _per_class(y, budget, rng, lambda members, k: _kmeans_medoids(x[members], k, rng))


METHODS: dict[str, Selector] = {
    "random": random,
    "random_balanced": random_balanced,
    "kcenter": kcenter,
    "kmeans": kmeans,
    "herding": herding,
    "kmeans_balanced": kmeans_balanced,
}
USES_POOL_LABELS = {"random": False, "random_balanced": True, "kcenter": False, "kmeans": False, "herding": True, "kmeans_balanced": True}
DETERMINISTIC = {"herding"}
