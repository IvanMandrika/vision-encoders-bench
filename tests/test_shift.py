import numpy as np

from veb import shift_signals as ss


def gaussian(rng, n, d, shift=0.0):
    return rng.normal(size=(n, d)) + shift


def classification_problem(rng, n, n_classes=5, d=10):
    centers = np.random.default_rng(100).normal(size=(n_classes, d))
    y = rng.integers(0, n_classes, size=n)
    x = centers[y] + rng.normal(size=(n, d)) * 1.5
    logits = -0.5 * np.square(x[:, None, :] - centers[None]).sum(-1) / 1.5**2
    return logits, y


def test_frechet_near_zero_on_same_gaussian_and_grows_with_shift():
    rng = np.random.default_rng(0)
    a = gaussian(rng, 3000, 20)
    same = ss.frechet_distance(a, gaussian(rng, 3000, 20))
    small = ss.frechet_distance(a, gaussian(rng, 3000, 20, 0.2))
    large = ss.frechet_distance(a, gaussian(rng, 3000, 20, 1.0))
    assert same < 0.1
    assert same < small < large
    assert abs(large - 20.0) < 1.5


def test_frechet_sees_covariance_change():
    rng = np.random.default_rng(1)
    a = gaussian(rng, 3000, 20)
    wide = gaussian(rng, 3000, 20) * 2.0
    assert abs(ss.frechet_distance(a, wide) - 20.0) < 2.0


def test_mmd_zero_on_same_distribution_and_positive_under_shift():
    rng = np.random.default_rng(2)
    a = gaussian(rng, 1500, 10)
    bandwidth = ss.median_bandwidth(a, rng)
    same = ss.mmd_rbf(a, gaussian(rng, 1500, 10), bandwidth, rng)
    shifted = ss.mmd_rbf(a, gaussian(rng, 1500, 10, 0.5), bandwidth, rng)
    assert abs(same) < 2e-3
    assert shifted > 10 * abs(same)


def test_domain_auc_is_chance_on_identical_distributions():
    rng = np.random.default_rng(3)
    same = ss.domain_classifier_auc(gaussian(rng, 1500, 10), gaussian(rng, 1500, 10), rng)
    shifted = ss.domain_classifier_auc(gaussian(rng, 1500, 10), gaussian(rng, 1500, 10, 0.5), rng)
    assert abs(same - 0.5) < 0.05
    assert shifted > 0.8


def test_temperature_scaling_undoes_logit_scaling():
    rng = np.random.default_rng(4)
    logits, y = classification_problem(rng, 4000)
    assert abs(ss.fit_temperature(logits * 3.0, y) - 3.0) < 0.3


def test_atc_recovers_in_distribution_accuracy():
    rng = np.random.default_rng(5)
    source_logits, source_y = classification_problem(rng, 4000)
    target_logits, target_y = classification_problem(rng, 4000)
    source_acc = float((source_logits.argmax(1) == source_y).mean())
    target_acc = float((target_logits.argmax(1) == target_y).mean())
    temperature = ss.fit_temperature(source_logits, source_y)
    for score in (ss.top_log_odds, ss.negative_entropy):
        threshold = ss.atc_threshold(score(source_logits, temperature), source_acc)
        assert abs(ss.atc_estimate(score(source_logits, temperature), threshold) - source_acc) < 0.005
        assert abs(ss.atc_estimate(score(target_logits, temperature), threshold) - target_acc) < 0.03


def test_atc_scores_do_not_saturate_on_confident_logits():
    rng = np.random.default_rng(6)
    logits = rng.normal(size=(2000, 10))
    logits[np.arange(2000), rng.integers(0, 10, 2000)] += 60.0
    assert np.allclose(ss.probabilities(logits).max(1), 1.0)
    assert len(np.unique(ss.top_log_odds(logits))) == 2000
    threshold = ss.atc_threshold(ss.top_log_odds(logits), 0.9)
    assert abs(ss.atc_estimate(ss.top_log_odds(logits), threshold) - 0.9) < 0.005


def test_top_log_odds_is_monotone_in_max_confidence():
    logits = np.random.default_rng(8).normal(size=(500, 7)) * 2
    order_odds = np.argsort(ss.top_log_odds(logits, 1.5))
    order_conf = np.argsort(ss.max_confidence(ss.probabilities(logits, 1.5)))
    assert np.array_equal(order_odds, order_conf)


def test_confidence_estimators_track_accuracy_when_calibrated():
    logits, y = classification_problem(np.random.default_rng(7), 6000)
    cal, tgt = slice(0, 3000), slice(3000, 6000)
    cal_acc = float((logits[cal].argmax(1) == y[cal]).mean())
    tgt_acc = float((logits[tgt].argmax(1) == y[tgt]).mean())
    temperature = ss.fit_temperature(logits[cal], y[cal])
    cal_probs = ss.probabilities(logits[cal], temperature)
    tgt_probs = ss.probabilities(logits[tgt], temperature)
    assert abs(ss.average_confidence(tgt_probs) - tgt_acc) < 0.03
    assert abs(ss.doc_estimate(cal_probs, cal_acc, tgt_probs) - tgt_acc) < 0.03
