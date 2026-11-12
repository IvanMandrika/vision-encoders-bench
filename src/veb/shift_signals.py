import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp, softmax
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[rng.choice(len(x), size=n, replace=False)]


def gaussian_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    cov, _ = ledoit_wolf(x)
    return x.mean(0), cov


def psd_sqrt(m: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(m)
    return (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    mu_a, cov_a = gaussian_fit(a)
    mu_b, cov_b = gaussian_fit(b)
    root_a = psd_sqrt(cov_a)
    cross = np.sqrt(np.clip(np.linalg.eigvalsh(root_a @ cov_b @ root_a), 0, None)).sum()
    return float(np.square(mu_a - mu_b).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * cross)


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :] - 2 * a @ b.T
    return np.clip(d, 0, None)


def median_bandwidth(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> float:
    x = np.asarray(subsample(x, n, rng), dtype=np.float64)
    d = squared_distances(x, x)
    return float(np.sqrt(np.median(d[np.triu_indices(len(x), k=1)])))


def mmd_rbf(a: np.ndarray, b: np.ndarray, bandwidth: float, rng: np.random.Generator, n: int = 2000) -> float:
    a = np.asarray(subsample(a, n, rng), dtype=np.float64)
    b = np.asarray(subsample(b, n, rng), dtype=np.float64)
    gamma = 1.0 / (2 * bandwidth**2)
    k_aa = np.exp(-gamma * squared_distances(a, a))
    k_bb = np.exp(-gamma * squared_distances(b, b))
    k_ab = np.exp(-gamma * squared_distances(a, b))
    n_a, n_b = len(a), len(b)
    within_a = (k_aa.sum() - np.trace(k_aa)) / (n_a * (n_a - 1))
    within_b = (k_bb.sum() - np.trace(k_bb)) / (n_b * (n_b - 1))
    return float(within_a + within_b - 2 * k_ab.mean())


def domain_classifier_auc(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = 2000, folds: int = 5) -> float:
    a = subsample(a, n, rng)
    b = subsample(b, n, rng)
    x = np.concatenate([a, b]).astype(np.float64)
    y = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=200))
    cv = StratifiedKFold(folds, shuffle=True, random_state=int(rng.integers(2**31)))
    scores = cross_val_predict(model, x, y, cv=cv, method="decision_function")
    return float(roc_auc_score(y, scores))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    rows = np.arange(len(y))

    def nll(log_t: float) -> float:
        return float(-log_softmax(logits / np.exp(log_t), axis=1)[rows, y].mean())

    return float(np.exp(minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded").x))


def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1)


def max_confidence(probs: np.ndarray) -> np.ndarray:
    return probs.max(1)


def top_log_odds(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    top = z.argmax(1)
    rest = z - z[np.arange(len(z)), top][:, None]
    rest[np.arange(len(z)), top] = -np.inf
    return -logsumexp(rest, axis=1)


def negative_entropy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    log_p = log_softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1)
    return (np.exp(log_p) * log_p).sum(1)


def average_confidence(probs: np.ndarray) -> float:
    return float(max_confidence(probs).mean())


def atc_threshold(source_scores: np.ndarray, source_accuracy: float) -> float:
    return float(np.quantile(source_scores, 1.0 - source_accuracy))


def atc_estimate(target_scores: np.ndarray, threshold: float) -> float:
    return float((target_scores > threshold).mean())


def doc_estimate(source_probs: np.ndarray, source_accuracy: float, target_probs: np.ndarray) -> float:
    return float(source_accuracy - (average_confidence(source_probs) - average_confidence(target_probs)))
