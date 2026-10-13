import numpy as np
import pandas as pd
import pytest
import torch

from src import config
from src.dataset import BeatDataset
from src.models import ECGCNN, HybridCNN, count_parameters, load_cnn, make_logreg, make_xgb
from src.train import augment_batch, class_weights, patient_level_val_split


def test_cnn_forward_shape():
    model = ECGCNN().eval()
    out = model(torch.randn(8, 1, config.WINDOW_SIZE))
    assert out.shape == (8, config.NUM_CLASSES)
    assert torch.isfinite(out).all()


def test_cnn_handles_single_sample_in_eval_mode():
    assert ECGCNN().eval()(torch.randn(1, 1, 280)).shape == (1, 4)


def test_hybrid_forward_shape_and_requires_features():
    model = HybridCNN().eval()
    out = model(torch.randn(6, 1, 280), torch.randn(6, len(config.RR_FEATURES)))
    assert out.shape == (6, config.NUM_CLASSES)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 280))


def test_hybrid_checkpoint_roundtrip_keeps_feature_stats(hybrid_checkpoint):
    loaded, meta = load_cnn(hybrid_checkpoint)
    assert isinstance(loaded, HybridCNN)
    assert meta["model_type"] == "hybrid"
    assert torch.isclose(loaded.feat_mean[0], torch.tensor(300.0))
    assert torch.isclose(loaded.feat_std[3], torch.tensor(15.0))


def test_augmentation_preserves_shape_and_is_seeded():
    wf = torch.randn(16, 1, 280)
    a = augment_batch(wf, torch.Generator().manual_seed(0))
    b = augment_batch(wf, torch.Generator().manual_seed(0))
    assert a.shape == wf.shape
    assert torch.allclose(a, b)
    assert not torch.allclose(a, wf)


def test_cnn_parameter_count_reasonable():
    assert 30_000 < count_parameters(ECGCNN()) < 100_000


def test_model_loads_from_checkpoint(cnn_checkpoint):
    torch.manual_seed(0)
    reference = ECGCNN().eval()
    loaded, meta = load_cnn(cnn_checkpoint)
    x = torch.randn(4, 1, 280)
    with torch.no_grad():
        assert torch.allclose(reference(x), loaded(x), atol=1e-6)
    assert not loaded.training
    assert meta["best_epoch"] == 0


def _toy_df(n_records=10, beats=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_records):
        for b in range(beats):
            label = int(rng.choice(4, p=[0.85, 0.05, 0.07, 0.03]))
            rows.append(
                {
                    "record_id": str(100 + r),
                    "beat_idx": b,
                    "label": label,
                    "waveform": rng.normal(size=280).astype(np.float32).tolist(),
                    **{c: float(rng.normal()) for c in config.FEATURE_COLUMNS},
                }
            )
    return pd.DataFrame(rows)


def test_dataset_shapes():
    ds = BeatDataset(_toy_df())
    wf, feat, y = ds[0]
    assert wf.shape == (1, 280)
    assert feat.shape == (len(config.FEATURE_COLUMNS),)
    assert y.dtype == torch.int64
    assert len(ds) == 400


def test_patient_level_split_has_no_record_overlap():
    df = _toy_df()
    train, val = patient_level_val_split(df)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(df["record_id"])
    assert 1 <= len(val) <= 4


def test_val_split_keeps_concentrated_minority_class_in_training():
    # Like MIT-BIH record 208: one patient holds almost all F beats.
    df = _toy_df(n_records=20)
    df.loc[df["label"] == 3, "label"] = 0
    df.loc[(df["record_id"] == "105") & (df["beat_idx"] < 30), "label"] = 3
    df.loc[(df["record_id"] == "110") & (df["beat_idx"] < 3), "label"] = 3
    train, val = patient_level_val_split(df)
    assert "105" in train
    f_train = ((df["label"] == 3) & df["record_id"].isin(train)).sum()
    assert f_train >= 0.5 * (df["label"] == 3).sum()


def test_class_weights_inverse_frequency():
    w = class_weights(np.array([900, 50, 40, 10]))
    assert w.argmax() == 3 and w.argmin() == 0
    assert torch.allclose(class_weights(np.array([5, 5, 5, 5])), torch.ones(4))


def test_baselines_fit_and_predict_proba():
    df = _toy_df()
    X, y = df[config.FEATURE_COLUMNS].to_numpy(), df["label"].to_numpy()
    for est in (make_logreg(), make_xgb(n_estimators=10, max_depth=2)):
        est.fit(X, y)
        p = est.predict_proba(X[:5])
        assert p.shape == (5, 4)
        assert np.allclose(p.sum(1), 1, atol=1e-5)


def test_patient_template_is_causal_and_per_record():
    from src.dataset import template_matrix

    df = _toy_df(n_records=2, beats=30)
    base = template_matrix(df)
    # first beat of each record uses itself as template
    first = df.groupby("record_id")["beat_idx"].idxmin()
    for i in first:
        assert np.allclose(base[i], np.asarray(df.loc[i, "waveform"]))
    # editing a later beat must not change templates of earlier beats (no look-ahead)
    edited = df.copy()
    edited.at[25, "waveform"] = (np.asarray(df.loc[25, "waveform"]) + 100).tolist()
    after = template_matrix(edited)
    assert np.allclose(base[:26], after[:26])
    # templates never mix records
    assert np.allclose(base[40], np.median(np.stack(df.loc[30:39, "waveform"].map(np.asarray)), axis=0))


def test_relative_dataset_has_two_channels():
    ds = BeatDataset(_toy_df(n_records=3, beats=20), relative=True)
    assert ds[5][0].shape == (2, 280)


def test_seed_ensemble_roundtrip_averages_members(hybrid_rel_ensemble_checkpoint):
    from src.models import SeedEnsemble

    model, meta = load_cnn(hybrid_rel_ensemble_checkpoint)
    assert isinstance(model, SeedEnsemble) and len(model.members) == 2
    assert meta["model_type"] == "hybrid_rel" and meta["n_members"] == 2
    x, f = torch.randn(3, 2, 280), torch.randn(3, len(config.RR_FEATURES)) * 50 + 300
    with torch.no_grad():
        expected = torch.stack([torch.softmax(m(x, f), 1) for m in model.members]).mean(0)
        assert torch.allclose(torch.softmax(model(x, f), 1), expected, atol=1e-5)
