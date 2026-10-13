import numpy as np
import pytest

from src.calibrate import (
    apply_calibration,
    conformal_qhat,
    ece_top_label,
    fit_calibration,
    prediction_sets,
    set_metrics,
)


def _synthetic(n=6000, k=4, true_t=2.5, seed=0):
    """Labels drawn from softmax(z), but the 'model' reports z * true_t (over-confident)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, size=(n, k))
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p])
    return z * true_t, y


def test_apply_calibration_identity_is_softmax():
    logp = np.log(np.array([[0.7, 0.2, 0.05, 0.05]]))
    assert np.allclose(apply_calibration(logp), [[0.7, 0.2, 0.05, 0.05]])
    out = apply_calibration(logp, temperature=3.0, bias=[0.0, 1.0, 0.0, 0.0])
    assert np.isclose(out.sum(), 1.0) and out[0, 1] > 0.2


def test_temperature_scaling_recovers_overconfidence():
    logits, y = _synthetic()
    t, b = fit_calibration(logits, y, with_bias=False)
    assert b is None
    assert t == pytest.approx(2.5, rel=0.15)
    assert ece_top_label(apply_calibration(logits, t), y) < ece_top_label(apply_calibration(logits), y)


def test_bias_correction_fixes_prior_shift():
    logits, y = _synthetic(true_t=1.0)
    shifted = logits + np.array([0.0, 2.0, 0.0, 0.0])  # model over-predicts class 1 (like class weighting)
    t, b = fit_calibration(shifted, y, with_bias=True)
    assert b[1] < min(b[0], b[2], b[3]) - 1.0  # learns to push class 1 back down
    assert abs(sum(b)) < 1e-6  # centred


def test_mondrian_conformal_reaches_per_class_coverage():
    logits, y = _synthetic(n=8000, true_t=1.0, seed=1)
    probs = apply_calibration(logits)
    cal, test = slice(0, 4000), slice(4000, None)
    qhat = conformal_qhat(probs[cal], y[cal], alpha=0.1)
    sets = prediction_sets(probs[test], qhat)
    m = set_metrics(sets, y[test])
    for c in "NSVF":
        assert m[f"{c}_coverage"] >= 0.86  # 90% target, finite-sample slack
    assert 1.0 <= m["mean_set_size"] <= 4.0


def test_prediction_sets_membership_rule():
    probs = np.array([[0.9, 0.05, 0.03, 0.02]])
    sets = prediction_sets(probs, qhat=[0.2, 0.96, 0.5, 0.5])
    assert sets.tolist() == [[True, True, False, False]]
