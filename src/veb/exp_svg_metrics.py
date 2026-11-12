import argparse
import difflib
import xml.etree.ElementTree as ET
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata
from skimage.metrics import structural_similarity
from tqdm import tqdm

from veb import plotting
from veb.common import RESULTS, bootstrap_ci, seed_everything
from veb.encoders import retry_on_oom
from veb.svg_corrupt import CONTROL, CORRUPTIONS, SEVERITIES, redundant
from veb.svg_data import Icon, load_icons, local_name, parse, render, serialise

RASTER = ("clip", "dinov2", "lpips", "ssim", "mse")
CODE = ("element_count", "token_edit")
METRICS = RASTER + CODE
RERENDER = "rerender"
LABELS = {
    "clip": "CLIP ViT-B/32 (1 - cos)",
    "dinov2": "DINOv2 ViT-S/14 (1 - cos)",
    "lpips": "LPIPS (alex)",
    "ssim": "1 - SSIM",
    "mse": "pixel MSE",
    "element_count": "element count change (code)",
    "token_edit": "token edit distance (code)",
}
COLOURS = dict(zip(RASTER, plotting.SERIES))
FLOOR_QUANTILE = 0.95
SUMMARY_COLUMNS = ["table", "metric", "corruption", "severity", "other_metric", "stat", "value", "ci_lo", "ci_hi", "seed"]


@dataclass(frozen=True)
class Variant:
    corruption: str
    severity: int
    svg: str
    supersample: int = 1


def variant_rng(seed: int, icon: str, corruption: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{icon}/{corruption}".encode())])


def build_variants(icon: Icon, seed: int) -> list[Variant]:
    root = parse(icon.svg)
    variants = [
        Variant(name, severity, serialise(corrupt(root, severity, variant_rng(seed, icon.name, name))))
        for name, corrupt in CORRUPTIONS.items()
        for severity in SEVERITIES
    ]
    variants.append(Variant(CONTROL, 0, serialise(redundant(root))))
    variants.append(Variant(RERENDER, 0, icon.svg, supersample=2))
    return variants


def tokens(element: ET.Element) -> list[str]:
    out = [f"<{local_name(element)}"]
    for key, value in element.attrib.items():
        out += [key, value]
    for child in element:
        out += tokens(child)
    return out + ["/>"]


def token_edit_distance(a: str, b: str) -> float:
    return 1.0 - difflib.SequenceMatcher(None, tokens(parse(a)), tokens(parse(b)), autojunk=False).ratio()


def element_count(svg: str) -> int:
    return sum(1 for element in parse(svg).iter() if len(element) == 0)


def element_count_change(a: str, b: str) -> float:
    return abs(element_count(b) - element_count(a)) / element_count(a)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2))


def ssim_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(structural_similarity(a, b, channel_axis=2, data_range=255))


def process_icon(job: tuple[Icon, int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    icon, seed = job
    original = render(icon.svg)
    images, rows = [], []
    for variant in build_variants(icon, seed):
        image = render(variant.svg, supersample=variant.supersample)
        images.append(image)
        rows.append({
            "icon": icon.name,
            "n_elements": element_count(icon.svg),
            "corruption": variant.corruption,
            "severity": variant.severity,
            "mse": mse(original, image),
            "ssim": ssim_distance(original, image),
            "element_count": element_count_change(icon.svg, variant.svg),
            "token_edit": token_edit_distance(icon.svg, variant.svg),
        })
    return original, np.stack(images), rows


def render_all(icons: list[Icon], seed: int, workers: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    jobs = [(icon, seed) for icon in icons]
    with ProcessPoolExecutor(workers) as pool:
        results = list(tqdm(pool.map(process_icon, jobs, chunksize=4), total=len(jobs), desc="render + pixel + code"))
    originals = np.stack([r[0] for r in results])
    variants = np.concatenate([r[1] for r in results])
    table = pd.DataFrame([row for r in results for row in r[2]])
    return originals, variants, table


def embed(name: str, images: np.ndarray, batch_size: int) -> np.ndarray:
    from veb.encoders import Encoder

    encoder = Encoder(name)
    out = retry_on_oom(lambda bs: encoder.embed(images, batch_size=bs), batch_size)
    del encoder
    torch.cuda.empty_cache()
    return out


class Lpips:
    def __init__(self):
        import lpips

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    def _tensor(self, images: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(images, device=self.device).permute(0, 3, 1, 2).float() / 127.5 - 1.0

    @torch.no_grad()
    def distances(self, a: np.ndarray, b: np.ndarray, index_a: np.ndarray, index_b: np.ndarray, batch_size: int) -> np.ndarray:
        out = []
        for i in range(0, len(index_a), batch_size):
            ia, ib = index_a[i : i + batch_size], index_b[i : i + batch_size]
            out.append(self.model(self._tensor(a[ia]), self._tensor(b[ib])).flatten().cpu().numpy())
        return np.concatenate(out)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(a * b, axis=1)


def random_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = rng.integers(n, size=n_pairs)
    b = (a + rng.integers(1, n, size=n_pairs)) % n
    return a, b


def compute_distances(icons: list[Icon], seed: int, n_pairs: int, batch_size: int, workers: int) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = seed_everything(seed)
    originals, variants, table = render_all(icons, seed, workers)
    owner = np.repeat(np.arange(len(icons)), len(variants) // len(icons))
    index = np.arange(len(variants))
    pair_a, pair_b = random_pairs(len(icons), n_pairs, rng)

    reference = {
        "mse": np.array([mse(originals[i], originals[j]) for i, j in zip(pair_a, pair_b)]),
        "ssim": np.array([ssim_distance(originals[i], originals[j]) for i, j in zip(pair_a, pair_b)]),
    }
    for name in ("clip", "dinov2"):
        base, changed = embed(name, originals, batch_size), embed(name, variants, batch_size)
        table[name] = cosine_distance(base[owner], changed)
        reference[name] = cosine_distance(base[pair_a], base[pair_b])
    model = Lpips()
    table["lpips"] = retry_on_oom(lambda bs: model.distances(originals, variants, owner, index, bs), batch_size)
    reference["lpips"] = retry_on_oom(lambda bs: model.distances(originals, originals, pair_a, pair_b, bs), batch_size)
    del model
    torch.cuda.empty_cache()
    return table, {name: float(values.mean()) for name, values in reference.items()}


def to_long(table: pd.DataFrame, normaliser: dict[str, float], seed: int) -> pd.DataFrame:
    long = table.melt(id_vars=["icon", "n_elements", "corruption", "severity"], value_vars=list(METRICS), var_name="metric", value_name="distance")
    long["normalised"] = long["distance"] / long["metric"].map(normaliser)
    long["seed"] = seed
    return long


def with_value(long: pd.DataFrame) -> pd.DataFrame:
    out = long.copy()
    out["value"] = out["normalised"].where(out["metric"].isin(RASTER), out["distance"])
    return out


def row(table: str, stat: str, value: float, lo: float = np.nan, hi: float = np.nan, metric: str = "", corruption: str = "", severity: int | str = "", other_metric: str = "") -> dict:
    return {"table": table, "metric": metric, "corruption": corruption, "severity": severity, "other_metric": other_metric, "stat": stat, "value": value, "ci_lo": lo, "ci_hi": hi}


def response_rows(data: pd.DataFrame) -> list[dict]:
    rows = []
    for (metric, corruption, severity), group in data.groupby(["metric", "corruption", "severity"], sort=False):
        rows.append(row("response", "mean", *bootstrap_ci(group["value"]), metric=metric, corruption=corruption, severity=severity))
    return rows


def severity_matrix(data: pd.DataFrame, metric: str, corruption: str) -> np.ndarray:
    subset = data[(data["metric"] == metric) & (data["corruption"] == corruption)]
    return subset.pivot(index="icon", columns="severity", values="value")[list(SEVERITIES)].to_numpy()


def spearman_with_severity(matrix: np.ndarray) -> np.ndarray:
    ranks = rankdata(matrix, axis=1)
    ranks = ranks - ranks.mean(1, keepdims=True)
    target = np.arange(matrix.shape[1]) - (matrix.shape[1] - 1) / 2
    norm = np.linalg.norm(ranks, axis=1) * np.linalg.norm(target)
    return np.divide(ranks @ target, norm, out=np.zeros(len(matrix)), where=norm > 0)


def ordering_accuracy(matrix: np.ndarray) -> np.ndarray:
    lower, upper = np.triu_indices(matrix.shape[1], k=1)
    diff = matrix[:, upper] - matrix[:, lower]
    return ((diff > 0) + 0.5 * (diff == 0)).mean(1)


def monotonicity_rows(data: pd.DataFrame) -> list[dict]:
    rows = []
    for metric in METRICS:
        for corruption in CORRUPTIONS:
            matrix = severity_matrix(data, metric, corruption)
            keys = {"metric": metric, "corruption": corruption}
            rows.append(row("monotonicity", "spearman", *bootstrap_ci(spearman_with_severity(matrix)), **keys))
            rows.append(row("monotonicity", "ordering_accuracy", *bootstrap_ci(ordering_accuracy(matrix)), **keys))
            rows.append(row("monotonicity", "frac_constant", float(np.mean(np.ptp(matrix, axis=1) == 0)), **keys))
    return rows


def noise_floors(data: pd.DataFrame) -> pd.Series:
    rerender = data[(data["corruption"] == RERENDER) & data["metric"].isin(RASTER)]
    return rerender.groupby("metric")["value"].quantile(FLOOR_QUANTILE)


def exceed_fraction(values: pd.Series, floor: float) -> tuple[float, float, float]:
    return bootstrap_ci((values > floor).astype(float))


def sensitivity_rows(data: pd.DataFrame, floors: pd.Series) -> list[dict]:
    rows = [row("noise_floor", f"q{round(FLOOR_QUANTILE * 100)}", floors[metric], metric=metric, corruption=RERENDER) for metric in RASTER]
    first = data[data["severity"] == 1]
    for metric in METRICS:
        floor = floors[metric] if metric in RASTER else 0.0
        for corruption in CORRUPTIONS:
            values = first[(first["metric"] == metric) & (first["corruption"] == corruption)]["value"]
            rows.append(row("sensitivity", "frac_above_floor", *exceed_fraction(values, floor), metric=metric, corruption=corruption, severity=1))
    return rows


def blind_spot_rows(data: pd.DataFrame, floors: pd.Series) -> list[dict]:
    rows = []
    for corruption, severity in ((CONTROL, 0), ("remove", 1)):
        subset = data[(data["corruption"] == corruption) & (data["severity"] == severity)]
        for metric in METRICS:
            values = subset[subset["metric"] == metric]["value"]
            floor = floors[metric] if metric in RASTER else 0.0
            keys = {"metric": metric, "corruption": corruption, "severity": severity}
            rows.append(row("blind_spot", "mean", *bootstrap_ci(values), **keys))
            rows.append(row("blind_spot", "median", float(values.median()), **keys))
            rows.append(row("blind_spot", "frac_above_rerender_floor", *exceed_fraction(values, floor), **keys))
    return rows


def rank_agreement_rows(data: pd.DataFrame) -> list[dict]:
    samples = data[data["corruption"].isin(CORRUPTIONS)].pivot(index=["icon", "corruption", "severity"], columns="metric", values="value")
    corr = samples[list(METRICS)].corr(method="spearman")
    return [row("rank_agreement", "spearman", corr.loc[a, b], metric=a, other_metric=b) for a in METRICS for b in METRICS]


def summarise(long: pd.DataFrame, normaliser: dict[str, float], seed: int) -> pd.DataFrame:
    data = with_value(long)
    floors = noise_floors(data)
    rows = [row("normaliser", "mean_random_pair_distance", normaliser[metric], metric=metric) for metric in RASTER]
    rows += response_rows(data) + monotonicity_rows(data) + sensitivity_rows(data, floors) + blind_spot_rows(data, floors) + rank_agreement_rows(data)
    summary = pd.DataFrame(rows)
    summary["seed"] = seed
    return summary[SUMMARY_COLUMNS]


def pick_example(icons: list[Icon], name: str | None) -> Icon:
    if name:
        return next(icon for icon in icons if icon.name == name)
    return min(icons, key=lambda icon: (abs(element_count(icon.svg) - 10), icon.name))


def plot_examples(icon: Icon, seed: int) -> None:
    import matplotlib.pyplot as plt

    by_key = {(v.corruption, v.severity): v for v in build_variants(icon, seed)}
    names = [*CORRUPTIONS, CONTROL]
    fig, axes = plt.subplots(len(names), 6, figsize=(8.4, 1.42 * len(names)), gridspec_kw={"wspace": 0.04, "hspace": 0.04})
    for r, name in enumerate(names):
        for c in range(6):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            variant = by_key.get((name, c)) if c else None
            if c == 0 or (name == CONTROL and c == 1):
                ax.imshow(render(icon.svg if c == 0 else by_key[(CONTROL, 0)].svg))
                if c == 1:
                    ax.set_xlabel("control (no severity)", fontsize=8, color=plotting.INK_SECONDARY)
            elif variant:
                ax.imshow(render(variant.svg))
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", color=plotting.INK)
            if r == 0:
                ax.set_title("original" if c == 0 else f"severity {c}", fontsize=9, loc="center", fontweight="normal")
    fig.suptitle(f"SVG-level corruptions of one Twemoji icon ({icon.name}); redundant = identical pixels, twice the code", x=0.5, y=0.93, fontsize=10)
    plotting.save(fig, "svg_examples.png")


def plot_response(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    response = summary[(summary["table"] == "response") & summary["metric"].isin(RASTER)]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey=True)
    for ax, corruption in zip(axes.flat, CORRUPTIONS):
        for metric in RASTER:
            line = response[(response["metric"] == metric) & (response["corruption"] == corruption)].sort_values("severity")
            x = line["severity"].astype(int)
            ax.plot(x, line["value"], marker="o", color=COLOURS[metric], label=LABELS[metric])
            ax.fill_between(x, line["ci_lo"].astype(float), line["ci_hi"].astype(float), color=COLOURS[metric], alpha=0.18, linewidth=0)
        ax.set_title(corruption)
        ax.set_xticks(SEVERITIES)
    for ax in axes[-1]:
        ax.set_xlabel("severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("distance / mean distance\nbetween different icons")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(RASTER), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Raster metric response to SVG corruptions (1.0 = as different as another icon; band = 95% bootstrap CI over icons)", x=0.5, fontsize=10.5)
    fig.tight_layout()
    plotting.save(fig, "svg_metric_response.png")


def plot_monotonicity(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    monotonicity = summary[summary["table"] == "monotonicity"]

    def grid_of(stat: str) -> np.ndarray:
        part = monotonicity[monotonicity["stat"] == stat]
        return part.pivot(index="metric", columns="corruption", values="value").loc[list(METRICS), list(CORRUPTIONS)].astype(float).to_numpy()

    grid = np.where(grid_of("frac_constant") == 1.0, np.nan, grid_of("spearman"))
    cmap = LinearSegmentedColormap.from_list("veb_sequential", plotting.SEQUENTIAL)
    low = min(0.0, float(np.nanmin(grid)))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    image = ax.imshow(grid, cmap=cmap, vmin=low, vmax=1.0, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(CORRUPTIONS)), list(CORRUPTIONS))
    ax.set_yticks(range(len(METRICS)), [LABELS[m] for m in METRICS])
    for (r, c), value in np.ndenumerate(grid):
        if np.isnan(value):
            ax.text(c, r, "constant", ha="center", va="center", color=plotting.MUTED, fontsize=8)
            continue
        dark = (value - low) / (1.0 - low) > 0.55
        ax.text(c, r, f"{value:.2f}", ha="center", va="center", color="white" if dark else plotting.INK, fontsize=9)
    fig.colorbar(image, ax=ax, label="mean per-icon Spearman", fraction=0.04, pad=0.02)
    ax.set_title("Monotonicity: Spearman(severity, distance) per icon, averaged\n(constant = metric does not change with severity, Spearman undefined)")
    plotting.save(fig, "svg_monotonicity.png")


def report(summary: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    for table, stat in (("normaliser", "mean_random_pair_distance"), ("noise_floor", f"q{round(FLOOR_QUANTILE * 100)}"), ("monotonicity", "spearman"), ("monotonicity", "ordering_accuracy"), ("sensitivity", "frac_above_floor")):
        part = summary[(summary["table"] == table) & (summary["stat"] == stat)]
        columns = "corruption" if part["corruption"].ne("").any() else "stat"
        print(f"\n{table} / {stat}\n{part.pivot(index='metric', columns=columns, values='value').round(4)}")
    blind = summary[summary["table"] == "blind_spot"]
    print(f"\nblind_spot\n{blind.pivot(index='metric', columns=['corruption', 'stat'], values='value').round(4)}")
    agreement = summary[summary["table"] == "rank_agreement"]
    print(f"\nrank_agreement\n{agreement.pivot(index='metric', columns='other_metric', values='value').loc[list(METRICS), list(METRICS)].round(2)}")
    response = summary[summary["table"] == "response"]
    print(f"\nresponse\n{response.pivot(index=['corruption', 'severity'], columns='metric', values='value')[list(METRICS)].round(3)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Image-similarity metrics on SVG-level corruptions of Twemoji icons")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-icons", type=int, default=400)
    parser.add_argument("--n-pairs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--example-icon", default=None)
    parser.add_argument("--reuse", action="store_true", help="rebuild summary and figures from results/svg_metrics.csv.gz and the normaliser in svg_summary.csv")
    args = parser.parse_args()

    icons = load_icons(args.n_icons, args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS / "svg_metrics.csv.gz"
    if args.reuse:
        long = pd.read_csv(metrics_path, dtype={"icon": str})
        previous = pd.read_csv(RESULTS / "svg_summary.csv")
        normaliser = previous[previous["table"] == "normaliser"].set_index("metric")["value"].to_dict()
    else:
        table, normaliser = compute_distances(icons, args.seed, args.n_pairs, args.batch_size, args.workers)
        long = to_long(table, normaliser, args.seed)
        long.to_csv(metrics_path, index=False)
    summary = summarise(long, normaliser, args.seed)
    summary.to_csv(RESULTS / "svg_summary.csv", index=False)

    plotting.setup()
    plot_examples(pick_example(icons, args.example_icon), args.seed)
    plot_response(summary)
    plot_monotonicity(summary)
    report(summary)


if __name__ == "__main__":
    main()
