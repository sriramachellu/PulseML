import numpy as np
import pandas as pd

from src import config
from src.preprocess import beat_features, zscore
from src.validate import check_amplitude, check_duplicates, check_flat, check_missing, run_checks


def make_df(windows, record="900", labels=None) -> pd.DataFrame:
    rows = []
    for i, w in enumerate(windows):
        w = np.asarray(w, dtype=float)
        rows.append(
            {
                "record_id": record,
                "beat_idx": i,
                "label": 0 if labels is None else labels[i],
                "waveform": zscore(w).astype(np.float32).tolist(),
                **beat_features(w, 300, 300),
            }
        )
    return pd.DataFrame(rows)


def healthy_windows(n=5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(config.WINDOW_SIZE)
    return [np.exp(-0.5 * ((t - 140) / 5) ** 2) + rng.normal(0, 0.02, len(t)) for _ in range(n)]


def test_clean_data_passes():
    res = run_checks(make_df(healthy_windows()))
    assert res["passed"]
    assert res["amplitude"]["n_flagged"] == 0
    assert res["flat"]["n_flat"] == 0


def test_all_zero_signal_flagged_as_flat():
    df = make_df(healthy_windows(3) + [np.zeros(config.WINDOW_SIZE)])
    res = run_checks(df)
    assert res["flat"]["n_flat"] == 1
    assert not res["passed"]
    # z-scoring a flat beat must not introduce NaNs
    assert check_missing(df)["waveform"] == 0


def test_extreme_values_flagged():
    big = healthy_windows(1)[0] * 12.0  # 12 mV peak
    neg = -healthy_windows(1, seed=1)[0] * 8.0
    res = check_amplitude(make_df(healthy_windows(2) + [big, neg]))
    assert res["n_flagged"] == 2
    assert res["flagged_records"] == {"900": 2}


def test_empty_record():
    empty = pd.DataFrame(columns=["record_id", "beat_idx", "label", "waveform", *config.FEATURE_COLUMNS])
    res = run_checks(empty)
    assert res["empty"] and not res["passed"]
    assert res["n_beats"] == 0
    assert check_flat(empty)["n_flat"] == 0
    assert check_duplicates(empty)["duplicate_keys"] == 0


def test_duplicates_detected():
    df = make_df(healthy_windows(3))
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    dup = check_duplicates(df)
    assert dup["duplicate_keys"] == 1
    assert dup["duplicate_waveforms"] == 1
    assert not run_checks(df)["passed"]


def test_missing_values_and_invalid_labels():
    df = make_df(healthy_windows(3), labels=[0, 1, 7])
    df.loc[0, "rr_ratio"] = np.nan
    res = run_checks(df)
    assert res["missing"]["rr_ratio"] == 1
    assert res["labels"]["n_invalid_labels"] == 1
    assert not res["passed"]


def test_wrong_waveform_length():
    df = make_df(healthy_windows(2))
    df.at[1, "waveform"] = [0.0] * 100
    assert run_checks(df)["waveform_length"]["n_wrong_length"] == 1


def test_long_rr_intervals_flagged_as_warning():
    from src.validate import check_rr

    df = make_df(healthy_windows(3))
    df.loc[1, "rr_prev"] = 360 * 5  # 5 s gap
    res = check_rr(df)
    assert res["n_flagged"] == 1 and res["flagged_records"] == {"900": 1}
    assert res["max_rr_ms"] == 5000.0
    assert run_checks(df)["passed"]  # warning only, not a hard failure


def test_build_patient_template_matches_training_definition():
    from src.dataset import template_matrix
    from src.preprocess import build_patient_template

    rng = np.random.default_rng(0)
    raw = [rng.normal(size=280) * (i + 1) for i in range(20)]
    df = make_df(raw)  # stores z-scored waveforms, like the feature store
    train_tmpl = template_matrix(df)[17]  # template for beat 17 = median of z-scored beats 1..16
    assert np.allclose(build_patient_template(np.stack(raw[1:17])), train_tmpl, atol=1e-5)
