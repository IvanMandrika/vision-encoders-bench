import numpy as np
import pandas as pd
import pytest

from veb import common, exp_coreset, exp_shift
from veb.data import calibration_indices, evaluation_indices
from veb.exp_svg_metrics import element_count_change, ordering_accuracy, spearman_with_severity, token_edit_distance

SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36"><rect fill="#000" width="3" height="3"/><circle fill="#FFF" r="2"/></svg>'


def test_fit_probe_separates_blobs_and_converges():
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=3.0, size=(4, 8))
    y = rng.integers(0, 4, size=400)
    x = centers[y] + rng.normal(size=(400, 8))
    probe = common.fit_probe(x, y, 4, weight_decay=1e-3)
    assert probe.accuracy(x, y) > 0.95
    assert probe.n_iter < 1000 and probe.grad_norm < 1e-3


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = common.bootstrap_ci(np.arange(100.0))
    assert lo < mean == 49.5 < hi
    assert common.bootstrap_ci(np.ones(10)) == (1.0, 1.0, 1.0)


def test_test_halves_partition_the_split():
    cal, ev = calibration_indices(10), evaluation_indices(10)
    assert np.array_equal(np.sort(np.r_[cal, ev]), np.arange(10))
    assert (cal % 2 == 0).all() and (ev % 2 == 1).all()


def test_spearman_with_severity_signs():
    rows = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 2, 2, 2, 2]], dtype=float)
    assert np.allclose(spearman_with_severity(rows), [1.0, -1.0, 0.0])


def test_ordering_accuracy_counts_ties_as_half():
    rows = np.array([[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]], dtype=float)
    assert np.allclose(ordering_accuracy(rows), [1.0, 0.5, 0.0])


def test_code_distances():
    assert token_edit_distance(SVG, SVG) == 0.0
    assert element_count_change(SVG, SVG) == 0.0
    fewer = SVG.replace('<circle fill="#FFF" r="2"/>', "")
    assert element_count_change(SVG, fewer) == 0.5
    assert 0.0 < token_edit_distance(SVG, fewer) < 1.0


def toy_conditions() -> pd.DataFrame:
    rows = [{"encoder": "clip", "corruption": "clean", "severity": 0, "accuracy": 0.8}]
    for c, corruption in enumerate(("a", "b", "c")):
        for severity in (1, 2, 3):
            rows.append({"encoder": "clip", "corruption": corruption, "severity": severity, "accuracy": 0.8 - 0.05 * severity - 0.01 * c})
    frame = pd.DataFrame(rows)
    frame["accuracy_drop"] = 0.8 - frame["accuracy"]
    for signal in exp_shift.DRIFT:
        frame[signal] = frame["accuracy_drop"]
    for name in exp_shift.ESTIMATORS:
        frame[f"est_{name}"] = frame["accuracy"] + 0.02
        frame[f"err_{name}"] = 0.02
    return frame


def test_shift_summary_on_toy_conditions():
    summary = exp_shift.summarise(toy_conditions(), seed=0).set_index(["signal", "metric"])
    assert summary.loc[("frechet", "spearman_vs_drop"), "value"] == pytest.approx(1.0)
    assert summary.loc[("frechet", "spearman_within_severity"), "value"] == pytest.approx(1.0)
    assert summary.loc[("atc_negent", "mae_pp"), "value"] == pytest.approx(2.0)
    assert summary.loc[("atc_negent", "bias_pp"), "value"] == pytest.approx(2.0)


def test_corruption_blocks_keep_whole_corruptions():
    groups = np.repeat(np.array(["a", "b", "c"]), 4)
    for block in exp_shift.corruption_blocks(groups, seed=0, n_boot=20):
        assert len(block) == 12
        for name in np.unique(groups[block]):
            assert np.sum(groups[block] == name) % 4 == 0


def test_permutation_p_value():
    null = np.arange(99.0)
    assert exp_shift.permutation_p(null, 1000.0) == pytest.approx(0.01)
    assert exp_shift.permutation_p(null, -1.0) == pytest.approx(1.0)


def test_drift_outlier_ignores_ties():
    frame = toy_conditions()
    frame = frame[frame.severity > 0].copy()
    frame["frechet"] = 1.0
    assert exp_shift.drift_outlier(frame, "frechet") is None


def test_coreset_summary_deltas_and_retained():
    runs = pd.DataFrame([
        {"encoder": "clip", "method": "full", "budget": 1000, "seed": 0, "accuracy": 0.8, "weight_decay": 0.0},
        *({"encoder": "clip", "method": "random", "budget": 10, "seed": s, "accuracy": a, "weight_decay": 0.0} for s, a in enumerate((0.3, 0.5))),
        {"encoder": "clip", "method": "herding", "budget": 10, "seed": 0, "accuracy": 0.6, "weight_decay": 0.0},
    ])
    summary = exp_coreset.summarise(runs, n_test=100).set_index("method")
    assert summary.loc["random", "mean"] == pytest.approx(0.4)
    assert summary.loc["herding", "delta_vs_random"] == pytest.approx(0.2)
    assert summary.loc["herding", "retained"] == pytest.approx(0.75)
    assert summary.loc["herding", "n_seeds"] == 1
    assert bool(summary.loc["herding", "uses_pool_labels"]) and not bool(summary.loc["random", "uses_pool_labels"])
