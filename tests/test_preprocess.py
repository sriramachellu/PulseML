import numpy as np
import pytest

from src import config
from src.preprocess import (
    bandpass_filter,
    beat_features,
    extract_window,
    lead_index,
    map_aami,
    process_signal,
    select_records,
    zscore,
)

FS = config.FS


def synthetic_ecg(n_beats=20, rr=300, baseline=True, seed=0):
    # gaussian "QRS" pulses + slow baseline wander + a little noise
    rng = np.random.default_rng(seed)
    n = rr * (n_beats + 1)
    t = np.arange(n)
    peaks = np.arange(rr, n - rr // 2, rr)
    sig = np.zeros(n)
    for p in peaks:
        sig += 1.2 * np.exp(-0.5 * ((t - p) / 4.0) ** 2)
    clean = sig.copy()
    if baseline:
        sig = sig + 0.8 * np.sin(2 * np.pi * 0.15 * t / FS)
    sig += rng.normal(0, 0.01, n)
    return sig, clean, peaks


def test_filter_preserves_length_and_peak_positions():
    sig, _, peaks = synthetic_ecg()
    filt = bandpass_filter(sig)
    assert filt.shape == sig.shape
    # zero-phase filtering must not shift R-peaks
    for p in peaks[2:-2]:
        local = filt[p - 20 : p + 20]
        assert abs(int(np.argmax(local)) - 20) <= 1


def test_filter_removes_baseline_wander_but_keeps_shape():
    sig, clean, peaks = synthetic_ecg()
    filt = bandpass_filter(sig)
    clean_f = bandpass_filter(clean)
    # Baseline wander (0.15 Hz) is removed: the filtered noisy signal ~ filtered clean signal
    core = slice(FS, len(sig) - FS)
    assert np.corrcoef(filt[core], clean_f[core])[0, 1] > 0.99
    # QRS morphology survives: beat windows strongly correlate with the original pulse shape
    for p in peaks[3:-3]:
        w_raw, w_f = clean[p - 30 : p + 30], filt[p - 30 : p + 30]
        assert np.corrcoef(w_raw, w_f)[0, 1] > 0.9


def test_filter_attenuates_out_of_band_sinusoids():
    t = np.arange(FS * 20) / FS
    in_band = np.sin(2 * np.pi * 10 * t)
    low = np.sin(2 * np.pi * 0.05 * t)
    high = np.sin(2 * np.pi * 120 * t)
    core = slice(FS * 2, -FS * 2)
    assert np.std(bandpass_filter(in_band)[core]) > 0.95 * np.std(in_band)
    assert np.std(bandpass_filter(low)[core]) < 0.05
    assert np.std(bandpass_filter(high)[core]) < 0.05


def test_window_size_is_280():
    x = np.arange(1000, dtype=float)
    w = extract_window(x, 500)
    assert len(w) == config.WINDOW_SIZE == 280
    assert w[0] == 500 - 140 and w[-1] == 500 + 139


@pytest.mark.parametrize("r", [0, 139, 861, 999])
def test_window_skips_boundaries(r):
    assert extract_window(np.zeros(1000), r) is None


def test_window_exact_boundaries_allowed():
    x = np.zeros(1000)
    assert extract_window(x, 140) is not None
    assert extract_window(x, 860) is not None


def test_zscore():
    w = zscore(np.random.default_rng(1).normal(3, 2, 280))
    assert abs(w.mean()) < 1e-6 and abs(w.std() - 1) < 1e-6
    assert np.all(np.isfinite(zscore(np.zeros(280))))


def test_normalised_rr_features():
    from src.preprocess import rr_features

    f = rr_features(rr_prev=200, rr_next=400, rr_local_avg=300)
    assert list(f) == config.RR_FEATURES
    assert f["rr_prev_norm"] == pytest.approx(2 / 3)
    assert f["rr_next_norm"] == pytest.approx(4 / 3)
    # without history the local average falls back to rr_prev
    assert rr_features(300, 300)["rr_prev_norm"] == 1.0


def test_beat_features():
    w = np.zeros(280)
    w[140] = 2.0
    f = beat_features(w, rr_prev=360, rr_next=180)
    assert f["rr_ratio"] == 2.0
    assert f["local_hr"] == 60.0
    assert f["qrs_proxy"] == 2.0
    assert f["pre_energy"] == 0.0 and f["post_energy"] == 0.0
    assert set(f) == set(config.FEATURE_COLUMNS)


@pytest.mark.parametrize(
    "symbols,expected",
    [
        (["N", "L", "R", "e", "j"], 0),
        (["A", "a", "J", "S"], 1),
        (["V", "E"], 2),
        (["F"], 3),
    ],
)
def test_aami_mapping(symbols, expected):
    for s in symbols:
        assert map_aami(s) == expected


@pytest.mark.parametrize("symbol", ["/", "f", "Q", "+", "~", "|", "!", "[", "]", '"', "x", "B", "?"])
def test_excluded_and_non_beat_symbols_unmapped(symbol):
    assert map_aami(symbol) is None


def test_paced_records_excluded():
    usable = select_records(config.ALL_RECORDS)
    for r in ["102", "104", "107", "217"]:
        assert r not in usable
        assert r not in config.DS1_RECORDS and r not in config.DS2_RECORDS
    assert len(usable) == 44


def test_ds1_ds2_are_disjoint_and_cover_usable_records():
    assert not set(config.DS1_RECORDS) & set(config.DS2_RECORDS)
    assert set(config.DS1_RECORDS) | set(config.DS2_RECORDS) == set(config.USABLE_RECORDS)
    assert len(config.DS1_RECORDS) == len(config.DS2_RECORDS) == 22


def test_lead_index_prefers_mlii():
    assert lead_index(["MLII", "V1"]) == 0
    assert lead_index(["V5", "MLII"]) == 1  # record 114
    assert lead_index(["V1", "V2"]) == 0


def test_process_signal_labels_counts_and_exclusions():
    sig, _, peaks = synthetic_ecg(n_beats=12)
    filt = bandpass_filter(sig)
    symbols = ["N", "A", "V", "F", "/", "N", "Q", "N", "E", "j", "N", "N"][: len(peaks)]
    samples = list(peaks)
    # interleave a non-beat rhythm annotation
    samples.insert(3, peaks[2] + 50)
    symbols.insert(3, "+")
    rows, st = process_signal(filt, np.array(samples), symbols, "999")

    assert st["total_annotations"] == len(symbols)
    assert st["non_beat_annotations"] == 1
    assert st["excluded_class"] == 2  # "/" and "Q"
    assert st["excluded_no_rr"] == 2  # first and last beats
    assert st["beats_extracted"] == len(rows)
    assert st["beats_extracted"] + st["beats_excluded"] == st["beat_annotations"]
    labels = [r["label"] for r in rows]
    assert labels == [1, 2, 3, 0, 0, 2, 0, 0]
    for r in rows:
        assert r["waveform"].shape == (280,)
        assert abs(r["waveform"].mean()) < 1e-4
        assert r["rr_prev"] > 0 and r["rr_next"] > 0
