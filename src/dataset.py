from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config


def load_beats(path, records=None):
    df = pd.read_parquet(path)
    return df if records is None else df[df["record_id"].isin(records)].reset_index(drop=True)


def waveform_matrix(df):
    if df.empty:
        return np.zeros((0, config.WINDOW_SIZE), dtype=np.float32)
    return np.stack(df["waveform"].map(lambda w: np.asarray(w, dtype=np.float32)))


def template_matrix(df, k=config.TEMPLATE_BEATS):
    """Median of the previous k beats of the same record (causal, label-free).
    The first beat of a record is its own template."""
    wf = waveform_matrix(df)
    out = np.empty_like(wf)
    rec, beat = df["record_id"].to_numpy(), df["beat_idx"].to_numpy()
    for r in pd.unique(rec):
        idx = np.flatnonzero(rec == r)
        idx = idx[np.argsort(beat[idx], kind="stable")]
        w = wf[idx]
        out[idx[0]] = w[0]
        for i in range(1, len(w)):
            out[idx[i]] = np.median(w[max(0, i - k) : i], axis=0)
    return out


def feature_matrix(df, columns=None):
    x = df[columns or config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class BeatDataset(Dataset):
    """(waveform, features, label) per beat.

    waveform is (1, 280), or (2, 280) with relative=True: the beat plus the beat minus the patient's template.
    """

    def __init__(self, source, records=None, feature_stats=None, feature_columns=None, relative=False):
        df = pd.read_parquet(source) if isinstance(source, (str, Path)) else source
        if records is not None:
            df = df[df["record_id"].isin(records)]
        df = df.reset_index(drop=True)

        wf = waveform_matrix(df)
        wf = np.stack([wf, wf - template_matrix(df)], axis=1) if relative else wf[:, None]
        feats = feature_matrix(df, feature_columns)
        if feature_stats is not None:
            mean, std = feature_stats
            feats = (feats - mean) / np.where(std > 0, std, 1.0)

        self.record_ids = df["record_id"].to_numpy()
        self.waveforms = torch.from_numpy(np.ascontiguousarray(wf))
        self.features = torch.from_numpy(feats.astype(np.float32))
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.waveforms[i], self.features[i], self.labels[i]

    def class_counts(self):
        return np.bincount(self.labels.numpy(), minlength=config.NUM_CLASSES)
