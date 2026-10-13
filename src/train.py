"""Train LogReg, XGBoost, the 1D CNN and the hybrid CNN ensemble on DS1 (patient-level splits only)."""

import argparse
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.dataset import BeatDataset, feature_matrix, load_beats
from src.models import SeedEnsemble, build_model, count_parameters, make_logreg, make_xgb, save_cnn
from src.utils import ensure_dirs, file_sha256, get_logger, git_commit, load_json, save_json, set_seed, utc_now

plt.switch_backend("Agg")
log = get_logger("train")

ARCHITECTURES = {
    "cnn": "ECGCNN (3x Conv1d-BN-ReLU + MLP head)",
    "hybrid": "HybridCNN (ECGCNN conv trunk + RR-feature MLP)",
    "hybrid_rel": "HybridCNN-rel (2-channel trunk: beat + deviation from patient template; RR MLP)",
}


def patient_level_val_split(df, val_fraction=config.VAL_RECORD_FRACTION, seed=config.SEED):
    """Hold out ~val_fraction of the records.

    Rare classes sit in a few patients (record 208 has ~90% of DS1's F beats), so a random grouped
    split can take a whole class out of training. Instead, try 5000 random record subsets and keep
    the one whose per-class share is closest to val_fraction, never taking more than half of a class.
    """
    counts = pd.crosstab(df["record_id"], df["label"]).reindex(columns=range(config.NUM_CLASSES), fill_value=0)
    records = counts.index.to_numpy()
    totals = counts.sum().to_numpy().astype(float)
    present = totals > 0
    n_val = max(1, round(len(records) * val_fraction))
    rng = np.random.default_rng(seed)

    best, best_score = None, np.inf
    for _ in range(5000):
        val = rng.choice(len(records), size=n_val, replace=False)
        share = (counts.iloc[val].sum().to_numpy() / np.maximum(totals, 1))[present]
        if share.max() > 0.5:
            continue
        score = np.abs(share - val_fraction).max()
        if score < best_score:
            best, best_score = val, score

    if best is None:  # nothing valid found, fall back to a plain grouped split
        sgkf = StratifiedGroupKFold(n_splits=max(2, round(1 / val_fraction)), shuffle=True, random_state=seed)
        _, val_idx = next(sgkf.split(df, df["label"], groups=df["record_id"]))
        best = np.flatnonzero(np.isin(records, df.iloc[val_idx]["record_id"].unique()))
    val_recs = sorted(records[best].tolist())
    return sorted(set(records) - set(val_recs)), val_recs


def train_baselines(ds1):
    X, y, groups = feature_matrix(ds1), ds1["label"].to_numpy(), ds1["record_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    specs = [
        ("logreg", make_logreg(), {f"clf__{k}": v for k, v in config.LOGREG_GRID.items()}, {}, config.LOGREG_MODEL),
        (
            "xgboost",
            make_xgb(),
            config.XGB_GRID,
            {"sample_weight": compute_sample_weight("balanced", y)},
            config.XGB_MODEL,
        ),
    ]
    results = {}
    for name, est, grid, fit_params, path in specs:
        t0 = time.time()
        gs = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=1, refit=True)
        gs.fit(X, y, groups=groups, **fit_params)
        joblib.dump(gs.best_estimator_, path)
        cv_res = gs.cv_results_
        rows = [
            {"params": p, "mean": float(m), "std": float(s)}
            for p, m, s in zip(cv_res["params"], cv_res["mean_test_score"], cv_res["std_test_score"])
        ]
        for r in rows:
            log.info("[%s] %s -> macro-F1 %.4f ± %.4f", name, r["params"], r["mean"], r["std"])
        log.info(
            "[%s] best %s (%.4f), saved %s [%.0fs]", name, gs.best_params_, gs.best_score_, path.name, time.time() - t0
        )
        results[name] = {
            "best_params": gs.best_params_,
            "cv_macro_f1": float(gs.best_score_),
            "cv_results": rows,
            "artifact": path.name,
            "features": config.FEATURE_COLUMNS,
        }
    return results


def class_weights(counts):
    # inverse frequency, scaled so a balanced dataset gets all ones
    counts = np.maximum(counts, 1).astype(np.float64)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def augment_batch(wf, gen=None):
    """Random R-peak shift (±5 samples), gain (±15%), noise and slow baseline drift."""
    b, c, n = wf.shape
    dev = wf.device

    def rand(*shape):
        return torch.rand(*shape, generator=gen).to(dev)

    shifts = (rand(b) * 11).long() - 5
    idx = (torch.arange(n, device=dev)[None] - shifts[:, None]).clamp(0, n - 1)
    out = torch.gather(wf, 2, idx[:, None].expand(-1, c, -1))
    out = out * (0.85 + 0.3 * rand(b, 1, 1))
    out = out + 0.05 * torch.randn(out.shape, generator=gen).to(dev)
    t = torch.linspace(0, 1, n, device=dev)
    drift = 0.2 * (rand(b, 1, 1) - 0.5) * torch.sin(2 * torch.pi * (0.5 * rand(b, 1, 1) * t + rand(b, 1, 1)))
    return out + drift


def run_epoch(model, loader, criterion, device, optimizer=None, augment=False, gen=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for wf, feat, y in loader:
            if training and augment:
                wf = augment_batch(wf, gen)
            wf, feat, y = wf.to(device), feat.to(device), y.to(device)
            logits = model(wf, feat)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(y.cpu().numpy())
    p, t = np.concatenate(preds), np.concatenate(labels)
    f1 = f1_score(t, p, average="macro", labels=list(range(config.NUM_CLASSES)), zero_division=0)
    return {"loss": total_loss / len(t), "acc": accuracy_score(t, p), "macro_f1": float(f1)}


def plot_curves(history, path, title):
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, key, name in zip(axes, ("loss", "acc", "macro_f1"), ("Weighted CE loss", "Accuracy", "Macro-F1")):
        ax.plot(h["epoch"], h[f"train_{key}"], label="train", color="#1f5fa8")
        ax.plot(h["epoch"], h[f"val_{key}"], label="val (held-out records)", color="#e8710a")
        ax.set(title=name, xlabel="epoch")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].axvline(h.loc[h["val_macro_f1"].idxmax(), "epoch"], ls="--", color="grey", lw=1)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train_torch_model(
    ds1, model_type="cnn", augment=False, seed=config.SEED, max_epochs=None, checkpoint=None, curves=None,
    tag=None, split=None, fixed_epochs=None,
):  # fmt: skip
    """Train a CNN / hybrid on DS1 with early stopping on a record-level validation split.

    With fixed_epochs it instead refits on all DS1 records for exactly that many epochs
    (no validation; the epoch count comes from CV). Returns metadata plus the model under "model".
    """
    set_seed(seed)
    tag = tag or model_type
    params = config.CNN_PARAMS | ({"max_epochs": max_epochs} if max_epochs else {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if fixed_epochs is not None:
        train_recs, val_recs = sorted(ds1["record_id"].unique()), []
    else:
        train_recs, val_recs = split or patient_level_val_split(ds1)
    rel = model_type == "hybrid_rel"
    train_ds = BeatDataset(ds1, train_recs, feature_columns=config.RR_FEATURES, relative=rel)
    val_ds = BeatDataset(ds1, val_recs, feature_columns=config.RR_FEATURES, relative=rel) if val_recs else None
    log.info("[%s] %s, augment=%s, seed=%d", tag, device, augment, seed)
    log.info("[%s] train %s: %d beats %s", tag, ",".join(train_recs), len(train_ds), train_ds.class_counts().tolist())
    if val_ds:
        log.info("[%s] val %s: %d beats %s", tag, ",".join(val_recs), len(val_ds), val_ds.class_counts().tolist())

    weights = class_weights(np.bincount(ds1["label"], minlength=config.NUM_CLASSES))  # from all of DS1
    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(val_ds, batch_size=1024) if val_ds else None
    aug_gen = torch.Generator().manual_seed(seed + 1)

    model = build_model(model_type)
    if model_type.startswith("hybrid"):
        f = train_ds.features.numpy()  # RR stats from training records only
        model.set_feature_stats(f.mean(0), f.std(0))
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history, best_f1, best_epoch, bad_epochs, best_state = [], -1.0, 0, 0, None
    t0 = time.time()
    for epoch in range(1, (fixed_epochs or params["max_epochs"]) + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer, augment, aug_gen)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}}
        if val_loader is None:
            history.append(row)
            log.info("[%s] epoch %02d  train loss %.4f F1 %.4f", tag, epoch, tr["loss"], tr["macro_f1"])
            best_epoch, best_f1, best_state = epoch, None, snapshot()
            continue

        va = run_epoch(model, val_loader, criterion, device)
        history.append(row | {f"val_{k}": v for k, v in va.items()})
        improved = va["macro_f1"] > best_f1
        log.info(
            "[%s] epoch %02d  train loss %.4f F1 %.4f | val loss %.4f F1 %.4f%s",
            tag, epoch, tr["loss"], tr["macro_f1"], va["loss"], va["macro_f1"], " *" if improved else "",
        )  # fmt: skip
        if improved:
            best_f1, best_epoch, bad_epochs, best_state = va["macro_f1"], epoch, 0, snapshot()
        else:
            bad_epochs += 1
            if bad_epochs >= params["patience"]:
                log.info("[%s] early stop at epoch %d", tag, epoch)
                break

    model.load_state_dict(best_state)
    model.cpu().eval()
    meta = {
        "model_type": model_type,
        "architecture": ARCHITECTURES[model_type],
        "rr_features": config.RR_FEATURES if model_type.startswith("hybrid") else [],
        "input_channels": 2 if rel else 1,
        "augmentation": augment,
        "seed": seed,
        "n_parameters": count_parameters(model),
        "hyperparameters": params,
        "optimizer": "Adam",
        "loss": "class-weighted cross-entropy",
        "class_weights": dict(zip(config.CLASS_NAMES, weights.tolist())),
        "train_records": train_recs,
        "val_records": val_recs,
        "train_beats": len(train_ds),
        "val_beats": len(val_ds) if val_ds else 0,
        "refit_fixed_epochs": fixed_epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_seconds": round(time.time() - t0, 1),
        "device": device,
        "trained_at": utc_now(),
        "model_version": config.MODEL_VERSION,
        "class_names": config.CLASS_NAMES,
    }
    if checkpoint:
        save_cnn(model, checkpoint, meta)
        log.info("[%s] best epoch %d (val F1 %s) -> %s", tag, best_epoch, best_f1, checkpoint)
    if curves and val_loader is not None:
        plot_curves(history, curves, f"{meta['architecture']}: training curves (patient-level validation)")
    return meta | {"history": history, "model": model}


def train_cnn(ds1, max_epochs=None):
    return train_torch_model(
        ds1, "cnn", max_epochs=max_epochs, checkpoint=config.CNN_CHECKPOINT,
        curves=config.FIGURES_DIR / "cnn_training_curves.png",
    )  # fmt: skip


def train_hybrid(ds1, max_epochs=None, augment=config.HYBRID_AUGMENT):
    """Final model: HybridCNN-rel refit on all of DS1 with several seeds, averaged."""
    epochs = max_epochs or config.HYBRID_REFIT_EPOCHS
    kind = config.HYBRID_MODEL_TYPE
    runs = [
        train_torch_model(ds1, kind, augment=augment, seed=s, fixed_epochs=epochs, tag=f"{kind}/s{s}")
        for s in config.ENSEMBLE_SEEDS
    ]
    ensemble = SeedEnsemble([r.pop("model") for r in runs]).eval()
    meta = {k: v for k, v in runs[0].items() if k not in ("history", "seed", "train_seconds")} | {
        "architecture": f"{runs[0]['architecture']}, {len(runs)}-seed ensemble",
        "ensemble_seeds": config.ENSEMBLE_SEEDS,
        "n_parameters": count_parameters(ensemble),
        "selection": "5-fold record-grouped CV on DS1 (pooled out-of-fold macro-F1); see docs/EXPERIMENTS.md",
        "train_seconds": round(sum(r["train_seconds"] for r in runs), 1),
        "trained_at": utc_now(),
    }
    save_cnn(ensemble, config.HYBRID_CHECKPOINT, meta)
    log.info("[%s] %d-seed ensemble, %d epochs each -> %s", kind, len(runs), epochs, config.HYBRID_CHECKPOINT)
    return meta | {"history": [r["history"] for r in runs], "model": ensemble}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, help="override epochs for the torch models")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if not config.DS1_PARQUET.exists():
        raise FileNotFoundError("Run `python -m src.preprocess` first.")
    ds1 = load_beats(config.DS1_PARQUET)
    assert not set(ds1["record_id"]) & set(config.DS2_RECORDS), "DS2 records in training data"
    log.info(
        "DS1: %d beats, %d records, classes %s",
        len(ds1),
        ds1["record_id"].nunique(),
        np.bincount(ds1["label"]).tolist(),
    )

    meta = {
        "trained_at": utc_now(),
        "model_version": config.MODEL_VERSION,
        "seed": config.SEED,
        "git_commit": git_commit(),
        "data": {
            "ds1_sha256": file_sha256(config.DS1_PARQUET),
            "ds1_beats": len(ds1),
            "ds1_records": sorted(ds1["record_id"].unique()),
            "features": config.FEATURE_COLUMNS,
        },
        "split_method": "de Chazal inter-patient (DS1 train / DS2 test); CV and validation grouped by record_id",
    }
    if config.TRAINING_METADATA.exists():  # keep results of the parts we skip
        prev = load_json(config.TRAINING_METADATA)
        meta |= {k: prev.get(k) for k in ("baselines", "cnn", "hybrid")}

    if not args.skip_baselines:
        set_seed(config.SEED)
        meta["baselines"] = train_baselines(ds1)
    if not args.skip_cnn:
        meta["cnn"] = {k: v for k, v in train_cnn(ds1, args.max_epochs).items() if k != "model"}
    if not args.skip_hybrid:
        meta["hybrid"] = {k: v for k, v in train_hybrid(ds1, args.max_epochs).items() if k != "model"}
    save_json(meta, config.TRAINING_METADATA)
    log.info("Saved %s", config.TRAINING_METADATA)


if __name__ == "__main__":
    main()
