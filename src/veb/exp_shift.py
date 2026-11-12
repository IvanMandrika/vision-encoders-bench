import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr

from veb import common, plotting
from veb import shift_signals as ss
from veb.corruptions import CORRUPTIONS, SEVERITIES
from veb.data import calibration_indices, evaluation_indices

DRIFT = ("frechet", "mmd", "domain_auc")
ESTIMATORS = ("avg_conf", "atc_maxconf", "atc_negent", "doc")
SHOWN_ESTIMATOR = "atc_negent"
WEIGHT_DECAYS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
AUC_SATURATED = 0.999
LABELS = {
    "frechet": "Frechet distance",
    "mmd": "MMD^2 (RBF)",
    "domain_auc": "Domain-classifier AUC",
    "avg_conf": "average confidence",
    "atc_maxconf": "ATC (max confidence)",
    "atc_negent": "ATC (negative entropy)",
    "doc": "DoC",
}


def select_probe(train: dict, seed: int, max_iter: int) -> tuple[common.Probe, float, float]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train["y"]))
    val, fit = order[: len(order) // 10], order[len(order) // 10 :]
    n_classes = int(train["y"].max()) + 1
    scores = {}
    for wd in WEIGHT_DECAYS:
        probe = common.fit_probe(train["x"][fit], train["y"][fit], n_classes, wd, max_iter)
        scores[wd] = probe.accuracy(train["x"][val], train["y"][val])
    best = max(scores, key=scores.get)
    return common.fit_probe(train["x"], train["y"], n_classes, best, max_iter), best, scores[best]


def drift_scores(reference: np.ndarray, target: np.ndarray, bandwidth: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "frechet": ss.frechet_distance(reference, target),
        "mmd": ss.mmd_rbf(reference, target, bandwidth, rng),
        "domain_auc": ss.domain_classifier_auc(reference, target, rng),
    }


def null_split_scores(pooled: np.ndarray, bandwidth: float, seed: int) -> dict:
    order = np.random.default_rng(seed).permutation(len(pooled))
    half = len(pooled) // 2
    return drift_scores(pooled[order[:half]], pooled[order[half:]], bandwidth, seed)


class Calibration:
    def __init__(self, logits: np.ndarray, y: np.ndarray):
        self.accuracy = float((logits.argmax(1) == y).mean())
        self.temperature = ss.fit_temperature(logits, y)
        self.raw_probs = ss.probabilities(logits)
        self.t_maxconf = ss.atc_threshold(ss.top_log_odds(logits, self.temperature), self.accuracy)
        self.t_negent = ss.atc_threshold(ss.negative_entropy(logits, self.temperature), self.accuracy)

    def estimates(self, logits: np.ndarray) -> dict:
        return {
            "avg_conf": ss.average_confidence(ss.probabilities(logits, self.temperature)),
            "atc_maxconf": ss.atc_estimate(ss.top_log_odds(logits, self.temperature), self.t_maxconf),
            "atc_negent": ss.atc_estimate(ss.negative_entropy(logits, self.temperature), self.t_negent),
            "doc": ss.doc_estimate(self.raw_probs, self.accuracy, ss.probabilities(logits)),
        }


def condition_row(probe: common.Probe, calib: Calibration, reference: np.ndarray, bandwidth: float,
                  split: dict, seed: int) -> dict:
    logits = probe.logits(split["x"])
    accuracy = float((logits.argmax(1) == split["y"]).mean())
    row = {"accuracy": accuracy, **drift_scores(reference, split["x"], bandwidth, seed)}
    for name, value in calib.estimates(logits).items():
        row[f"est_{name}"] = value
        row[f"err_{name}"] = value - accuracy
    return row


def half(split: dict, indices: np.ndarray) -> dict:
    mask = np.isin(split["idx"], indices)
    return {k: v[mask] for k, v in split.items() if k in ("x", "y", "idx")}


def corrupted_split(encoder: str, corruption: str, severity: int, seed: int) -> dict:
    split = common.load_split(encoder, f"test_{corruption}_{severity}")
    if "seed" not in split or int(split["seed"]) != seed:
        raise ValueError(f"{encoder}/test_{corruption}_{severity} was not built with seed {seed}; run veb.embed --seed {seed}")
    return split


def evaluation_splits(encoder: str, clean: dict, seed: int) -> list[tuple[str, int, dict]]:
    splits = [("clean", 0, half(clean, evaluation_indices(len(clean["y"]))))]
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            splits.append((corruption, severity, corrupted_split(encoder, corruption, severity, seed)))
    return splits


def run_encoder(encoder: str, seed: int, max_iter: int, n_null: int, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    probe, weight_decay, val_accuracy = select_probe(common.load_split(encoder, "train"), seed, max_iter)
    clean = common.load_split(encoder, "test_clean")
    calibration = half(clean, calibration_indices(len(clean["y"])))
    reference = calibration["x"]
    calib = Calibration(probe.logits(reference), calibration["y"])
    bandwidth = ss.median_bandwidth(reference, np.random.default_rng(seed))
    splits = evaluation_splits(encoder, clean, seed)
    rows = Parallel(n_jobs=jobs)(
        delayed(condition_row)(probe, calib, reference, bandwidth, split, seed + 1 + i)
        for i, (_, _, split) in enumerate(splits)
    )
    conditions = pd.DataFrame(
        [{"encoder": encoder, "seed": seed, "corruption": c, "severity": s, **row} for (c, s, _), row in zip(splits, rows)]
    )
    clean_accuracy = conditions.loc[conditions.severity == 0, "accuracy"].item()
    conditions.insert(5, "accuracy_drop", clean_accuracy - conditions["accuracy"])
    null = pd.DataFrame(
        Parallel(n_jobs=jobs)(delayed(null_split_scores)(clean["x"], bandwidth, seed + 1000 + i) for i in range(n_null))
    )
    null.insert(0, "encoder", encoder)
    probe_info = {
        "encoder": encoder, "seed": seed, "weight_decay": weight_decay, "val_accuracy": val_accuracy,
        "lbfgs_iterations": probe.n_iter, "final_grad_norm": probe.grad_norm, "max_iter": max_iter,
        "calibration_accuracy": calib.accuracy, "temperature": calib.temperature, "mmd_bandwidth": bandwidth,
    }
    return conditions, null, probe_info


def corruption_blocks(groups: np.ndarray, seed: int, n_boot: int = 2000) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    members = [np.flatnonzero(groups == name) for name in names]
    return [np.concatenate([members[i] for i in rng.integers(0, len(names), size=len(names))]) for _ in range(n_boot)]


def block_ci(statistic, blocks: list[np.ndarray], *arrays: np.ndarray) -> tuple[float, float, float]:
    boot = np.array([statistic(*(a[b] for a in arrays)) for b in blocks])
    return float(statistic(*arrays)), float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def mean_abs(x: np.ndarray) -> float:
    return float(np.abs(x).mean())


def within_severity_spearman(values: np.ndarray, drop: np.ndarray, severity: np.ndarray) -> float:
    return float(np.mean([spearman(values[severity == s], drop[severity == s]) for s in np.unique(severity)]))


def summarise(conditions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for encoder, frame in conditions.groupby("encoder", sort=False):
        shifted = frame[frame.severity > 0]
        clean = frame[frame.severity == 0].iloc[0]
        drop = shifted["accuracy_drop"].to_numpy()
        severity = shifted["severity"].to_numpy()
        blocks = corruption_blocks(shifted["corruption"].to_numpy(), seed)
        base = {"encoder": encoder, "seed": seed, "n_conditions": len(shifted)}
        for signal in DRIFT + ESTIMATORS:
            kind = "drift" if signal in DRIFT else "estimator"
            values = shifted[signal].to_numpy() if signal in DRIFT else -shifted[f"est_{signal}"].to_numpy()
            rho, lo, hi = block_ci(spearman, blocks, values, drop)
            rows.append({**base, "signal": signal, "kind": kind, "metric": "spearman_vs_drop", "value": rho, "ci_lo": lo, "ci_hi": hi})
            rows.append({**base, "signal": signal, "kind": kind, "metric": "spearman_within_severity",
                         "value": within_severity_spearman(values, drop, severity), "ci_lo": np.nan, "ci_hi": np.nan})
        rows.append({**base, "signal": "domain_auc", "kind": "drift", "metric": f"n_auc_above_{AUC_SATURATED}",
                     "value": float((shifted["domain_auc"] > AUC_SATURATED).sum()), "ci_lo": np.nan, "ci_hi": np.nan})
        rows.append({**base, "signal": "severity", "kind": "baseline", "metric": "spearman_vs_drop",
                     "value": spearman(severity, drop), "ci_lo": np.nan, "ci_hi": np.nan})
        for signal in ESTIMATORS:
            mae, lo, hi = block_ci(mean_abs, blocks, 100 * shifted[f"err_{signal}"].to_numpy())
            common_part = {**base, "signal": signal, "kind": "estimator"}
            rows.append({**common_part, "metric": "mae_pp", "value": mae, "ci_lo": lo, "ci_hi": hi})
            rows.append({**common_part, "metric": "bias_pp", "value": 100 * shifted[f"err_{signal}"].mean(), "ci_lo": np.nan, "ci_hi": np.nan})
            rows.append({**common_part, "n_conditions": 1, "metric": "clean_abs_error_pp",
                         "value": 100 * abs(clean[f"err_{signal}"]), "ci_lo": np.nan, "ci_hi": np.nan})
    return pd.DataFrame(rows)


def first_exceeding(frame: pd.DataFrame, signal: str, threshold: float) -> float:
    hits = frame.loc[frame[signal] > threshold, "severity"]
    return float(hits.min()) if len(hits) else np.nan


def permutation_p(null_values: np.ndarray, score: float) -> float:
    return float((1 + np.sum(null_values >= score)) / (1 + len(null_values)))


def detection(conditions: pd.DataFrame, null: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for encoder, frame in conditions.groupby("encoder", sort=False):
        clean = frame[frame.severity == 0].iloc[0]
        for signal in DRIFT:
            scores = null.loc[null.encoder == encoder, signal]
            q99 = float(scores.quantile(0.99))
            null_values = scores.to_numpy()
            for corruption in CORRUPTIONS:
                sub = frame[frame.corruption == corruption]
                rows.append({
                    "encoder": encoder, "seed": seed, "signal": signal, "corruption": corruption,
                    "null_mean": scores.mean(), "null_q99": q99, "n_null": len(scores),
                    "clean_score": clean[signal], "clean_exceeds_q99": bool(clean[signal] > q99),
                    "score_severity_1": sub.loc[sub.severity == 1, signal].item(),
                    "p_value_severity_1": permutation_p(null_values, sub.loc[sub.severity == 1, signal].item()),
                    "first_detected_severity": first_exceeding(sub, signal, q99),
                })
    return pd.DataFrame(rows)


def plot_accuracy(conditions: pd.DataFrame, encoders: list[str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True, layout="constrained")
    for ax, corruption in zip(axes.ravel(), CORRUPTIONS):
        for encoder in encoders:
            color = plotting.SERIES[common.ENCODERS.index(encoder)]
            frame = conditions[(conditions.encoder == encoder) & conditions.corruption.isin([corruption, "clean"])]
            frame = frame.sort_values("severity")
            ax.plot(frame.severity, 100 * frame.accuracy, marker="o", color=color, label=encoder)
        ax.set_title(corruption.replace("_", " "))
        ax.set_ylim(0, None)
    for ax in axes[-1]:
        ax.set_xlabel("severity (0 = clean)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy, %")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Linear-probe accuracy on CIFAR-100 under corruption (evaluation half, n=5000)", x=0.01, ha="left", fontweight="bold")
    plotting.save(fig, "shift_accuracy.png")


def plot_estimation(conditions: pd.DataFrame, summary: pd.DataFrame, encoders: list[str]) -> None:
    fig, (scatter, bars) = plt.subplots(1, 2, figsize=(11, 4.9), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]})
    scatter.plot([0, 100], [0, 100], color=plotting.MUTED, linewidth=1, linestyle="--", label="identity")
    mae = summary[summary.metric == "mae_pp"]
    offsets = np.linspace(-0.12, 0.12, len(encoders)) if len(encoders) > 1 else [0.0]
    for offset, encoder in zip(offsets, encoders):
        color = plotting.SERIES[common.ENCODERS.index(encoder)]
        frame = conditions[conditions.encoder == encoder]
        shifted, clean = frame[frame.severity > 0], frame[frame.severity == 0]
        scatter.scatter(100 * shifted.accuracy, 100 * shifted[f"est_{SHOWN_ESTIMATOR}"], color=color, s=26, label=f"{encoder}, corrupted")
        scatter.scatter(100 * clean.accuracy, 100 * clean[f"est_{SHOWN_ESTIMATOR}"], facecolor=plotting.SURFACE, edgecolor=color,
                        s=60, linewidth=1.8, label=f"{encoder}, clean", zorder=3)
        part = mae[mae.encoder == encoder].set_index("signal").loc[list(ESTIMATORS)]
        x = np.arange(len(ESTIMATORS)) + offset
        bars.errorbar(x, part["value"], yerr=[part["value"] - part["ci_lo"], part["ci_hi"] - part["value"]], fmt="o",
                      color=color, capsize=3, label=encoder)
    scatter.set_xlim(0, 100)
    scatter.set_ylim(0, 100)
    scatter.set_box_aspect(1)
    scatter.set_xlabel("true accuracy, %")
    scatter.set_ylabel("predicted accuracy (no target labels), %")
    scatter.set_title(f"{LABELS[SHOWN_ESTIMATOR]}, fixed in advance")
    scatter.legend(loc="upper left", fontsize=9)
    bars.set_xticks(range(len(ESTIMATORS)), [LABELS[e].replace(" (", "\n(") for e in ESTIMATORS])
    bars.set_ylim(0, None)
    bars.set_ylabel("mean absolute error over 30 conditions, pp")
    bars.set_title("All four estimators (95% CI, bootstrap over corruptions)")
    bars.legend(loc="upper right")
    fig.suptitle("Label-free accuracy estimation under corruption", x=0.01, ha="left", fontweight="bold")
    plotting.save(fig, "shift_estimation.png")


def drift_outlier(shifted: pd.DataFrame, signal: str) -> pd.Series | None:
    gaps = []
    for _, frame in shifted.groupby("encoder", sort=False):
        n = len(frame)
        gaps.append(pd.Series((rankdata(frame[signal]) - rankdata(frame["accuracy_drop"])) / n, index=frame.index))
    gap = pd.concat(gaps)
    row = shifted.loc[gap.idxmax()]
    tied = (shifted.loc[shifted.encoder == row.encoder, signal] == row[signal]).sum() > 1
    return row if gap.max() >= 0.3 and not tied else None


def plot_drift_vs_drop(conditions: pd.DataFrame, summary: pd.DataFrame, encoders: list[str]) -> None:
    shifted = conditions[conditions.severity > 0]
    fig, axes = plt.subplots(1, len(DRIFT), figsize=(12.5, 5.0), sharey=True, layout="constrained")
    for ax, signal in zip(axes, DRIFT):
        for encoder in encoders:
            color = plotting.SERIES[common.ENCODERS.index(encoder)]
            frame = shifted[shifted.encoder == encoder]
            rho = summary[(summary.encoder == encoder) & (summary.signal == signal) & (summary.metric == "spearman_vs_drop")]["value"].item()
            ax.scatter(frame[signal], 100 * frame.accuracy_drop, color=color, s=28, alpha=0.85, label=f"{encoder} (Spearman {rho:.2f})")
        if signal != "domain_auc":
            ax.set_xscale("log")
        outlier = drift_outlier(shifted, signal)
        if outlier is not None:
            ax.scatter([outlier[signal]], [100 * outlier.accuracy_drop], facecolor="none", edgecolor=plotting.INK, s=140, linewidth=1.5)
            ax.annotate(f"{outlier.corruption.replace('_', ' ')} s{outlier.severity} ({outlier.encoder})",
                        (outlier[signal], 100 * outlier.accuracy_drop), xytext=(-8, 14), textcoords="offset points",
                        ha="right", fontsize=9, color=plotting.INK)
        ax.set_title(LABELS[signal])
        ax.set_xlabel(f"{LABELS[signal]}, calibration half vs condition")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=9)
    axes[0].set_ylabel("accuracy drop vs clean, pp")
    rho = summary[(summary.kind == "drift") & (summary.metric == "spearman_vs_drop")]["value"]
    fig.suptitle(f"Within an encoder, drift scores rank accuracy loss well (Spearman {rho.min():.2f} to {rho.max():.2f}); "
                 "domain AUC saturates at 1", x=0.01, ha="left", fontweight="bold")
    plotting.save(fig, "shift_drift_vs_drop.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy, drift signals and label-free accuracy estimates under CIFAR-100 corruptions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoders", nargs="+", default=list(common.ENCODERS), choices=common.ENCODERS)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    common.seed_everything(args.seed)
    plotting.setup()
    results = [run_encoder(e, args.seed, args.max_iter, args.n_null, args.jobs) for e in args.encoders]
    conditions = pd.concat([r[0] for r in results], ignore_index=True)
    null = pd.concat([r[1] for r in results], ignore_index=True)
    summary = summarise(conditions, args.seed)
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(common.RESULTS / "shift_conditions.csv", index=False)
    summary.to_csv(common.RESULTS / "shift_summary.csv", index=False)
    detection(conditions, null, args.seed).to_csv(common.RESULTS / "shift_detection.csv", index=False)
    pd.DataFrame([r[2] for r in results]).to_csv(common.RESULTS / "shift_probe.csv", index=False)
    encoders = [e for e in common.ENCODERS if e in args.encoders]
    plot_accuracy(conditions, encoders)
    plot_estimation(conditions, summary, encoders)
    plot_drift_vs_drop(conditions, summary, encoders)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
