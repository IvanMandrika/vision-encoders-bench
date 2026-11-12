from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EMB = DATA / "embeddings"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ENCODERS = ("clip", "dinov2")


def seed_everything(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def load_split(encoder: str, name: str) -> dict:
    with np.load(EMB / encoder / f"{name}.npz") as f:
        return {k: f[k] for k in f.files}


def save_split(encoder: str, name: str, **arrays) -> None:
    path = EMB / encoder / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


class Probe:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, n_iter: int = 0, grad_norm: float = 0.0):
        self.weight = weight
        self.bias = bias
        self.n_iter = n_iter
        self.grad_norm = grad_norm

    def logits(self, x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(np.array(x, dtype=np.float32))
        return (xt @ self.weight.T + self.bias).numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.logits(x).argmax(1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == y).mean())


def fit_probe(x: np.ndarray, y: np.ndarray, n_classes: int, weight_decay: float = 1e-3, max_iter: int = 1000) -> Probe:
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    weight = torch.zeros(n_classes, xt.shape[1], requires_grad=True)
    bias = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([weight, bias], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ weight.T + bias, yt) + weight_decay * weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    closure()
    grad_norm = float(torch.cat([weight.grad.flatten(), bias.grad.flatten()]).norm())
    return Probe(weight.detach(), bias.detach(), int(opt.state[weight]["n_iter"]), grad_norm)


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
