import argparse
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from veb import common, plotting
from veb.corruptions import CORRUPTIONS, SEVERITIES, corrupt
from veb.data import Cifar100, evaluation_indices, load_cifar100
from veb.encoders import Encoder, retry_on_oom

GRID_IMAGE_INDEX = 1


@dataclass(frozen=True)
class Split:
    name: str
    corruption: str | None = None
    severity: int = 0


def all_splits() -> list[Split]:
    corrupted = [Split(f"test_{c}_{s}", c, s) for c in CORRUPTIONS for s in SEVERITIES]
    return [Split("train"), Split("test_clean"), *corrupted]


def split_arrays(data: Cifar100, split: Split, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split.name == "train":
        return data.train_x, data.train_y, np.arange(len(data.train_y), dtype=np.int64)
    if split.corruption is None:
        return data.test_x, data.test_y, np.arange(len(data.test_y), dtype=np.int64)
    idx = evaluation_indices(len(data.test_y))
    return corrupt(data.test_x[idx], split.corruption, split.severity, seed), data.test_y[idx], idx


def split_path(encoder_name: str, split_name: str):
    return common.EMB / encoder_name / f"{split_name}.npz"


def write_split(encoder: Encoder, data: Cifar100, split: Split, batch_size: int, seed: int) -> tuple[int, float]:
    images, y, idx = split_arrays(data, split, seed)
    start = time.perf_counter()
    x = retry_on_oom(lambda bs: encoder.embed(images, bs), batch_size)
    seconds = time.perf_counter() - start
    partial = f"{split.name}.partial"
    common.save_split(encoder.name, partial, x=x, y=y, idx=idx, seed=np.int64(seed))
    split_path(encoder.name, partial).replace(split_path(encoder.name, split.name))
    return len(x), seconds


def is_current(encoder_name: str, split: Split, seed: int) -> bool:
    path = split_path(encoder_name, split.name)
    if not path.exists():
        return False
    with np.load(path) as f:
        return split.corruption is None or ("seed" in f.files and int(f["seed"]) == seed)


def extract(encoder_name: str, data: Cifar100, batch_size: int, seed: int) -> None:
    todo = [s for s in all_splits() if not is_current(encoder_name, s, seed)]
    print(f"[{encoder_name}] {len(todo)} of {len(all_splits())} files to write", flush=True)
    if not todo:
        return
    encoder = Encoder(encoder_name)
    total_images, total_seconds = 0, 0.0
    for split in todo:
        n, seconds = write_split(encoder, data, split, batch_size, seed)
        total_images += n
        total_seconds += seconds
        print(f"[{encoder_name}] {split.name}: {n} images, {seconds:.1f} s, {n / seconds:.0f} img/s", flush=True)
    print(
        f"[{encoder_name}] total: {total_images} images, {total_seconds:.1f} s, "
        f"{total_images / total_seconds:.0f} img/s on {encoder.device}",
        flush=True,
    )
    del encoder
    torch.cuda.empty_cache()


def corruptions_grid(data: Cifar100, seed: int) -> None:
    plotting.setup()
    idx = evaluation_indices(len(data.test_y))[GRID_IMAGE_INDEX]
    image = data.test_x[idx : idx + 1]
    fig, axes = plt.subplots(len(CORRUPTIONS), 6, figsize=(8.4, 8.6))
    for row, name in enumerate(CORRUPTIONS):
        for severity in range(6):
            ax = axes[row, severity]
            shown = image[0] if severity == 0 else corrupt(image, name, severity, seed)[0]
            ax.imshow(shown, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.spines[:].set_visible(False)
            if row == 0:
                ax.set_title("clean" if severity == 0 else f"severity {severity}", loc="center", fontweight="normal")
        axes[row, 0].set_ylabel(name.replace("_", " "))
    fig.suptitle(
        f"The six corruptions on one 32x32 test image ({data.class_names[data.test_y[idx]]})",
        x=0.02, ha="left", fontweight="bold", fontsize=11,
    )
    fig.tight_layout()
    plotting.save(fig, "corruptions_grid.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen-encoder embeddings for CIFAR-100 and its corrupted copies.")
    parser.add_argument("--encoders", nargs="+", default=list(common.ENCODERS), choices=common.ENCODERS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    common.seed_everything(args.seed)
    data = load_cifar100()
    corruptions_grid(data, args.seed)
    start = time.perf_counter()
    for name in args.encoders:
        extract(name, data, args.batch_size, args.seed)
    print(f"wall time: {time.perf_counter() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
