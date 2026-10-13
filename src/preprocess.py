"""Filter, segment, label and featurise MIT-BIH beats into the Parquet feature store."""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

from src import config
from src.utils import ensure_dirs, get_logger, save_json, utc_now

log = get_logger("preprocess")

# Every annotation that marks a beat (used for RR). B, r, n, ? are rare and end up excluded.
BEAT_SYMBOLS = set(config.AAMI_MAP) | config.AAMI_EXCLUDED | {"B", "r", "n", "?"}


def bandpass_filter(x, fs=config.FS, low=config.BANDPASS_LOW, high=config.BANDPASS_HIGH, order=config.BANDPASS_ORDER):
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))  # zero-phase, so R-peaks don't move


def map_aami(symbol):
    return config.AAMI_MAP.get(symbol)  # None = excluded or not a beat


def select_records(records):
    return [r for r in records if str(r) not in config.PACED_RECORDS]


def extract_window(x, r_peak, before=config.WINDOW_BEFORE, after=config.WINDOW_AFTER):
    start, end = r_peak - before, r_peak + after
    return x[start:end] if start >= 0 and end <= len(x) else None


def zscore(w, eps=1e-8):
    w = np.asarray(w, dtype=np.float64)
    return (w - w.mean()) / (w.std() + eps)


def build_patient_template(previous_beats, k=config.TEMPLATE_BEATS):
    """Median of the patient's last k beats (each z-scored), same as in training."""
    beats = np.atleast_2d(np.asarray(previous_beats, dtype=np.float64))[-k:]
    return np.median([zscore(b) for b in beats], axis=0).astype(np.float32)


def _div(a, b):
    return float(a / b) if b > 0 else np.nan


def rr_features(rr_prev, rr_next, rr_local_avg=None, fs=config.FS):
    """RR features (in samples). Also used by the API so serving matches training."""
    avg = float(rr_prev if rr_local_avg is None else rr_local_avg)
    return {
        "rr_prev": float(rr_prev),
        "rr_next": float(rr_next),
        "rr_ratio": _div(rr_prev, rr_next),
        "local_hr": _div(fs * 60.0, rr_prev),
        "rr_local_avg": avg,
        "rr_prev_norm": _div(rr_prev, avg),
        "rr_next_norm": _div(rr_next, avg),
    }


def beat_features(window_mv, rr_prev, rr_next, fs=config.FS, rr_local_avg=None):
    # waveform stats use the un-normalised window; a z-scored one always has mean 0, std 1
    w = np.asarray(window_mv, dtype=np.float64)
    c = len(w) // 2
    flat = np.ptp(w) < 1e-12
    return {
        **rr_features(rr_prev, rr_next, rr_local_avg, fs),
        "wf_mean": w.mean(),
        "wf_std": w.std(),
        "wf_min": w.min(),
        "wf_max": w.max(),
        "wf_skew": 0.0 if flat else float(stats.skew(w)),
        "wf_kurtosis": 0.0 if flat else float(stats.kurtosis(w)),
        "qrs_proxy": np.abs(w[c - config.QRS_HALF_WIDTH : c + config.QRS_HALF_WIDTH]).max(),
        "pre_energy": np.sum(w[slice(*config.PRE_ENERGY_SLICE)] ** 2),
        "post_energy": np.sum(w[slice(*config.POST_ENERGY_SLICE)] ** 2),
    }


def process_signal(sig_mv, samples, symbols, record_id, fs=config.FS):
    """Cut one filtered record into labelled beats. Returns (rows, lineage counts)."""
    symbols = list(symbols)
    is_beat = np.array([s in BEAT_SYMBOLS for s in symbols], dtype=bool)
    peaks = np.asarray(samples)[is_beat]
    beat_syms = [s for s, b in zip(symbols, is_beat) if b]
    ann_idx = np.flatnonzero(is_beat)
    rr = np.diff(peaks).astype(np.float64)  # rr[k-1] is the interval ending at beat k

    rows, counts = [], Counter()
    for k, (r, sym) in enumerate(zip(peaks, beat_syms)):
        label = map_aami(sym)
        if label is None:
            counts["excluded_class"] += 1
            continue
        if k in (0, len(peaks) - 1):
            counts["excluded_no_rr"] += 1
            continue
        win = extract_window(sig_mv, int(r))
        if win is None:
            counts["excluded_boundary"] += 1
            continue
        local_avg = rr[max(0, k - config.RR_LOCAL_WINDOW) : k].mean()  # causal: only past intervals
        rows.append(
            {
                "record_id": record_id,
                "beat_idx": int(ann_idx[k]),
                "r_peak_sample": int(r),
                "symbol": sym,
                "label": label,
                "waveform": zscore(win).astype(np.float32),
                **beat_features(win, float(r - peaks[k - 1]), float(peaks[k + 1] - r), fs, float(local_avg)),
            }
        )
        counts[config.CLASS_NAMES[label]] += 1

    excluded = ["excluded_class", "excluded_no_rr", "excluded_boundary"]
    stats_row = {
        "record_id": record_id,
        "total_annotations": len(symbols),
        "non_beat_annotations": int((~is_beat).sum()),
        "beat_annotations": int(is_beat.sum()),
        "beats_extracted": len(rows),
        "beats_excluded": sum(counts[e] for e in excluded),
        **{e: counts[e] for e in excluded},
        **{f"n_{c}": counts[c] for c in config.CLASS_NAMES},
    }
    return rows, stats_row


def lead_index(sig_names, preferred=config.PREFERRED_LEAD):
    # record 114 has MLII on channel 1, so look it up by name
    return sig_names.index(preferred) if preferred in sig_names else 0


def process_record(record_id, raw_dir=config.RAW_DIR):
    import wfdb

    rec = wfdb.rdrecord(str(raw_dir / record_id))
    ann = wfdb.rdann(str(raw_dir / record_id), "atr")
    ch = lead_index(list(rec.sig_name))
    if rec.sig_name[ch] != config.PREFERRED_LEAD:
        log.warning("Record %s has no %s, using %s", record_id, config.PREFERRED_LEAD, rec.sig_name[ch])
    rows, st = process_signal(
        bandpass_filter(rec.p_signal[:, ch], fs=rec.fs), ann.sample, ann.symbol, record_id, rec.fs
    )
    st["lead"] = rec.sig_name[ch]
    return rows, st


def build_feature_store(raw_dir=config.RAW_DIR, out_dir=config.PARQUET_DIR):
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert not set(config.DS1_RECORDS) & set(config.DS2_RECORDS), "DS1/DS2 overlap"
    records = select_records(config.ALL_RECORDS)
    log.info(
        "%d records, %d paced dropped -> %d used", len(config.ALL_RECORDS), len(config.PACED_RECORDS), len(records)
    )

    split_of = {r: "DS1" for r in config.DS1_RECORDS} | {r: "DS2" for r in config.DS2_RECORDS}
    beats = {"DS1": [], "DS2": []}
    stats_rows = []
    for rid in records:
        rows, st = process_record(rid, raw_dir)
        st["split"] = split_of[rid]
        stats_rows.append(st)
        beats[st["split"]].extend(rows)
        log.info(
            "%s [%s] %d annotations -> %d beats (N=%d S=%d V=%d F=%d)",
            rid, st["split"], st["total_annotations"], st["beats_extracted"], st["n_N"], st["n_S"], st["n_V"], st["n_F"],
        )  # fmt: skip

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_parquet(out_dir / config.STATS_PARQUET.name, index=False)

    summary = {"created_at": utc_now(), "records_used": len(stats_rows), "splits": {}}
    for split, path in (("DS1", config.DS1_PARQUET), ("DS2", config.DS2_PARQUET)):
        df = pd.DataFrame(beats[split])
        df["waveform"] = df["waveform"].map(np.ndarray.tolist)
        df.to_parquet(out_dir / path.name, index=False)
        dist = df["label"].map(dict(enumerate(config.CLASS_NAMES))).value_counts().to_dict()
        summary["splits"][split] = {"records": df["record_id"].nunique(), "beats": len(df), "class_counts": dist}
        log.info("%s: %d beats from %d records %s", split, len(df), df["record_id"].nunique(), dist)

    totals = stats_df[["total_annotations", "beat_annotations", "beats_extracted", "beats_excluded"]].sum()
    log.info("Lineage: %s", totals.to_dict())
    summary["lineage"] = totals.to_dict()
    save_json(summary, config.REPORTS_DIR / "preprocess_summary.json")
    return summary


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build_feature_store()
