import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from veb import plotting
from veb.common import ENCODERS, RESULTS, fit_probe, load_split, seed_everything
from veb.selection import DETERMINISTIC, METHODS, USES_POOL_LABELS, random, random_balanced

BUDGETS = (100, 200, 500, 1000, 2500, 5000)
WD_GRID = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
LABELS = {
    "random": "random",
    "random_balanced": "random, class-balanced*",
    "kcenter": "k-center greedy",
    "kmeans": "k-means medoids",
    "herding": "herding per class*",
    "kmeans_balanced": "k-means medoids per class*",
}
TITLES = {"clip": "CLIP ViT-B/32", "dinov2": "DINOv2 ViT-S/14"}
WD_STREAM, RUN_STREAM = 0, 1


def select_weight_decay(
    fit: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    budget: int | None,
    n_classes: int,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, dict[float, float]]:
    x, y = fit
    subsets = [np.arange(len(y))] if budget is None else [random(x, y, budget, rng) for _ in range(draws)]
    scores = {
        wd: float(np.mean([fit_probe(x[s], y[s], n_classes, weight_decay=wd).accuracy(*val) for s in subsets]))
        for wd in WD_GRID
    }
    return max(scores, key=scores.get), scores


def weight_decay_table(
    encoder: str, x: np.ndarray, y: np.ndarray, budgets: list[int], n_classes: int, draws: int, seed: int
) -> pd.DataFrame:
    stream = [seed, WD_STREAM, ENCODERS.index(encoder)]
    val_idx = random_balanced(x, y, len(y) // 10, np.random.default_rng(stream))
    fit_mask = np.ones(len(y), dtype=bool)
    fit_mask[val_idx] = False
    fit, val = (x[fit_mask], y[fit_mask]), (x[val_idx], y[val_idx])
    rows = []
    for budget in [*budgets, None]:
        best, scores = select_weight_decay(fit, val, budget, n_classes, draws, np.random.default_rng([*stream, budget or 0]))
        label = len(y) if budget is None else budget
        rows += [
            {"encoder": encoder, "budget": label, "weight_decay": wd, "val_accuracy": acc, "selected": wd == best}
            for wd, acc in scores.items()
        ]
        print(f"[{encoder}] budget {label}: weight decay {best:g} (train-internal val acc {scores[best]:.4f})", flush=True)
    return pd.DataFrame(rows)


def run_encoder(encoder: str, budgets: list[int], seeds: list[int], draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = load_split(encoder, "train"), load_split(encoder, "test_clean")
    x, y = train["x"], train["y"]
    n_classes = int(y.max()) + 1
    wd_table = weight_decay_table(encoder, x, y, budgets, n_classes, draws, seed)
    chosen = wd_table[wd_table.selected].set_index("budget").weight_decay
    full_acc = fit_probe(x, y, n_classes, weight_decay=chosen[len(y)]).accuracy(test["x"], test["y"])
    print(f"[{encoder}] full data ({len(y)}): test acc {full_acc:.4f}", flush=True)
    rows = [{"encoder": encoder, "method": "full", "budget": len(y), "seed": seed, "accuracy": full_acc, "weight_decay": chosen[len(y)]}]
    for budget in budgets:
        for m, (method, select) in enumerate(METHODS.items()):
            for s in seeds[:1] if method in DETERMINISTIC else seeds:
                rng = np.random.default_rng([s, RUN_STREAM, ENCODERS.index(encoder), m, budget])
                idx = select(x, y, budget, rng)
                acc = fit_probe(x[idx], y[idx], n_classes, weight_decay=chosen[budget]).accuracy(test["x"], test["y"])
                rows.append({"encoder": encoder, "method": method, "budget": budget, "seed": s, "accuracy": acc, "weight_decay": chosen[budget]})
            accs = [r["accuracy"] for r in rows if r["method"] == method and r["budget"] == budget]
            print(f"[{encoder}] budget {budget:5d} {method:16s} {np.mean(accs):.4f} +- {np.std(accs, ddof=1) if len(accs) > 1 else 0:.4f}", flush=True)
    return pd.DataFrame(rows), wd_table


def summarise(runs: pd.DataFrame, n_test: int) -> pd.DataFrame:
    full = runs[runs.method == "full"].set_index("encoder").accuracy
    subset = runs[runs.method != "full"]
    out = subset.groupby(["encoder", "method", "budget", "weight_decay"], sort=False).accuracy.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"count": "n_seeds"})
    out.insert(2, "uses_pool_labels", out.method.map(USES_POOL_LABELS))
    out["test_se"] = np.sqrt(out["mean"] * (1 - out["mean"]) / n_test)
    out["full_accuracy"] = out.encoder.map(full)
    out["retained"] = out["mean"] / out.full_accuracy
    baseline = out[out.method == "random"].set_index(["encoder", "budget"])["mean"]
    out["delta_vs_random"] = out["mean"] - baseline.reindex(pd.MultiIndex.from_frame(out[["encoder", "budget"]])).to_numpy()
    return out


def plot(summary: pd.DataFrame) -> None:
    plotting.setup()
    encoders = [e for e in ENCODERS if e in set(summary.encoder)]
    fig, axes = plt.subplots(1, len(encoders), figsize=(5.6 * len(encoders), 4.4), sharey=True, squeeze=False)
    for ax, encoder in zip(axes[0], encoders):
        part = summary[summary.encoder == encoder]
        for colour, method in zip(plotting.SERIES, METHODS):
            line = part[part.method == method].sort_values("budget")
            ax.plot(line.budget, line["mean"], marker="o", color=colour, label=LABELS[method])
            ax.fill_between(line.budget, line["mean"] - line["std"], line["mean"] + line["std"], color=colour, alpha=0.18, linewidth=0)
        full = part.full_accuracy.iloc[0]
        ax.axhline(full, color=plotting.INK_SECONDARY, linestyle="--", linewidth=1.2, label="full training set")
        budgets = sorted(part.budget.unique())
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_xlabel("training subset size (log scale)")
        ax.set_title(f"{TITLES[encoder]}: full-data accuracy {full:.3f}")
    axes[0, 0].set_ylabel("clean test accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Linear probe accuracy vs training subset size (band = std over 5 seeds; herding is deterministic)\n"
                 "* uses the class labels of all 50 000 pool images to balance or to compute class means", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    plotting.save(fig, "coreset_accuracy.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Data subset selection for linear probes on frozen features")
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=ENCODERS)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wd-draws", type=int, default=3)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        seed_everything(args.seed)
        seeds = list(range(args.seed, args.seed + args.n_seeds))
        runs, wd_tables = zip(*(run_encoder(e, args.budgets, seeds, args.wd_draws, args.seed) for e in args.encoders))
        pd.concat(runs).to_csv(RESULTS / "coreset.csv", index=False)
        pd.concat(wd_tables).to_csv(RESULTS / "coreset_weight_decay.csv", index=False)
    summary = summarise(pd.read_csv(RESULTS / "coreset.csv"), len(load_split(ENCODERS[0], "test_clean")["y"]))
    summary.to_csv(RESULTS / "coreset_summary.csv", index=False)
    plot(summary)
    print(summary.pivot_table(index=["encoder", "budget"], columns="method", values="mean", sort=False).round(4).to_string())


if __name__ == "__main__":
    main()
