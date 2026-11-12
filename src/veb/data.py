import hashlib
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from veb.common import DATA

URLS = (
    "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz",
)
SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
ARCHIVE = DATA / "cifar-100-python.tar.gz"
CACHE = DATA / "cifar100.npz"
N_CLASSES = 100


@dataclass(frozen=True)
class Cifar100:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    class_names: list[str]


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _download(archive: Path) -> None:
    partial = archive.with_suffix(".partial")
    for url in URLS:
        try:
            urllib.request.urlretrieve(url, partial)
        except OSError:
            continue
        if _sha256(partial) == SHA256:
            partial.replace(archive)
            return
    raise RuntimeError(f"could not download a CIFAR-100 archive with the expected checksum from {URLS}")


def _read_member(tar: tarfile.TarFile, name: str) -> dict:
    return pickle.load(tar.extractfile(f"cifar-100-python/{name}"), encoding="latin1")


def _to_images(flat: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(flat, dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))


def _build_cache(archive: Path, cache: Path) -> None:
    with tarfile.open(archive) as tar:
        train, test, meta = (_read_member(tar, name) for name in ("train", "test", "meta"))
    np.savez(
        cache,
        train_x=_to_images(train["data"]),
        train_y=np.asarray(train["fine_labels"], dtype=np.int64),
        test_x=_to_images(test["data"]),
        test_y=np.asarray(test["fine_labels"], dtype=np.int64),
        class_names=np.asarray(meta["fine_label_names"]),
    )


def load_cifar100() -> Cifar100:
    if not CACHE.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        if not ARCHIVE.exists():
            _download(ARCHIVE)
        if _sha256(ARCHIVE) != SHA256:
            raise RuntimeError(f"{ARCHIVE} does not match the expected CIFAR-100 checksum")
        _build_cache(ARCHIVE, CACHE)
    with np.load(CACHE) as f:
        return Cifar100(f["train_x"], f["train_y"], f["test_x"], f["test_y"], [str(c) for c in f["class_names"]])


def calibration_indices(n_test: int) -> np.ndarray:
    return np.arange(0, n_test, 2, dtype=np.int64)


def evaluation_indices(n_test: int) -> np.ndarray:
    return np.arange(1, n_test, 2, dtype=np.int64)
