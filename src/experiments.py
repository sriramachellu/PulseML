"""Ablations: which modelling choices actually help on unseen patients?

Variants are trained on DS1 only and chosen on validation (or grouped CV) only.
DS2 scores are reported for every variant for transparency, never used to choose.
  python -m src.experiments              # multi-seed deep ablation + classical feature ablation
  python -m src.experiments --cv-select  # 5-fold grouped CV between the hybrid variants
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.dataset import feature_matrix, load_beats, template_matrix, waveform_matrix
from src.evaluate import cnn_predict_proba
from src.models import make_logreg, make_xgb
from src.train import train_torch_model
from src.utils import ensure_dirs, get_logger, save_json, utc_now

plt.switch_backend("Agg")
log = get_logger("experiments")

LABELS = list(range(config.NUM_CLASSES))
SEEDS = [42, 7, 2024]
TORCH_VARIANTS = [
    ("cnn", False, "CNN (waveform)"),
    ("cnn", True, "CNN + augmentation"),
    ("hybrid", False, "Hybrid (waveform + RR)"),
    ("hybrid", True, "Hybrid + augmentation"),
    ("hybrid_rel", False, "Hybrid-rel (+ patient template)"),
    ("hybrid_rel", True, "Hybrid-rel + augmentation"),
]
CV_CANDIDATES = [v for v in TORCH_VARIANTS if v[0].startswith("hybrid")]
FEATURES_V1 = [c for c in config.FEATURE_COLUMNS if c not in ("rr_local_avg", "rr_prev_norm", "rr_next_norm")]


def scores(y, pred):
    rec = recall_score(y, pred, labels=LABELS, average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
    return {"macro_f1": float(f1)} | {f"{c}_recall": float(r) for c, r in zip(config.CLASS_NAMES, rec)}


def grouped_folds(df, n_splits=5):
    """(train records, held-out records) per fold, grouped by patient."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    recs = df["record_id"]
    return [
        (sorted(recs.iloc[tr].unique()), sorted(recs.iloc[va].unique()))
        for tr, va in cv.split(df, df["label"], groups=recs)
    ]


class Inputs:
    """Waveform / RR inputs for a dataframe, built once and reused by every variant."""

    def __init__(self, df):
        self.wf = waveform_matrix(df)
        self.wf_rel = np.stack([self.wf, self.wf - template_matrix(df)], axis=1)
        self.rr = feature_matrix(df, config.RR_FEATURES)

    def predict(self, model, model_type, mask=slice(None)):
        wf = self.wf_rel if model_type == "hybrid_rel" else self.wf
        rr = self.rr[mask] if model_type.startswith("hybrid") else None
        return cnn_predict_proba(model, wf[mask], rr)


def torch_ablation(ds1, ds2, max_epochs):
    test, y2 = Inputs(ds2), ds2["label"].to_numpy()
    rows = []
    for model_type, augment, name in TORCH_VARIANTS:
        for seed in SEEDS:
            t0 = time.time()
            tag = f"{model_type}{'+aug' if augment else ''}/s{seed}"
            res = train_torch_model(ds1, model_type, augment=augment, seed=seed, max_epochs=max_epochs, tag=tag)
            ds2_scores = scores(y2, test.predict(res["model"], model_type).argmax(1))
            rows.append(
                {
                    "family": "torch",
                    "variant": name,
                    "model_type": model_type,
                    "augment": augment,
                    "seed": seed,
                    "val_macro_f1": res["best_val_macro_f1"],
                    "best_epoch": res["best_epoch"],
                    **{f"ds2_{k}": v for k, v in ds2_scores.items()},
                    "seconds": round(time.time() - t0, 1),
                }
            )
            log.info(
                "%-32s seed %-4d val F1 %.4f | DS2 F1 %.4f",
                name,
                seed,
                res["best_val_macro_f1"],
                ds2_scores["macro_f1"],
            )
    return pd.DataFrame(rows)


def feature_ablation(ds1, ds2):
    y1, groups, y2 = ds1["label"].to_numpy(), ds1["record_id"].to_numpy(), ds2["label"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    models = [
        ("LogReg", lambda: make_logreg(C=0.01), {}),
        (
            "XGBoost",
            lambda: make_xgb(max_depth=3, n_estimators=200, learning_rate=0.05),
            {"sample_weight": compute_sample_weight("balanced", y1)},
        ),
    ]
    rows = []
    for fs_name, cols in (("13 features (v1)", FEATURES_V1), ("16 features (+ normalised RR)", config.FEATURE_COLUMNS)):
        X1, X2 = feature_matrix(ds1, cols), feature_matrix(ds2, cols)
        for name, make, fit_kw in models:
            cv_f1 = cross_val_score(make(), X1, y1, groups=groups, cv=cv, scoring="f1_macro", params=fit_kw)
            pred = make().fit(X1, y1, **fit_kw).predict(X2)
            row = {
                "family": "classical",
                "variant": f"{name} · {fs_name}",
                "cv_macro_f1": cv_f1.mean(),
                "cv_macro_f1_std": cv_f1.std(),
                **{f"ds2_{k}": v for k, v in scores(y2, pred).items()},
            }
            log.info("%-45s CV F1 %.4f | DS2 F1 %.4f", row["variant"], row["cv_macro_f1"], row["ds2_macro_f1"])
            rows.append(row)
    return pd.DataFrame(rows)


def cv_selection(ds1, max_epochs):
    """5-fold record-grouped CV over all 22 DS1 patients for the hybrid variants.

    The 4-patient validation split couldn't separate them (gaps under one seed-sd). Early stopping uses
    the held-out fold, which biases every variant the same way. Scored on the pooled out-of-fold
    predictions, which stays stable even when a fold is missing a class. DS2 isn't touched.
    """
    folds = grouped_folds(ds1)
    inputs, y = Inputs(ds1), ds1["label"].to_numpy()
    rows = []
    for model_type, augment, name in CV_CANDIDATES:
        oof, fold_f1 = np.full(len(ds1), -1), []
        for i, (tr, va) in enumerate(folds):
            tag = f"cv{i}/{model_type}{'+aug' if augment else ''}"
            res = train_torch_model(ds1, model_type, augment=augment, max_epochs=max_epochs, split=(tr, va), tag=tag)
            m = ds1["record_id"].isin(va).to_numpy()
            oof[m] = inputs.predict(res["model"], model_type, m).argmax(1)
            fold_f1.append(scores(y[m], oof[m])["macro_f1"])
        pooled = scores(y, oof)
        rows.append(
            {
                "variant": name,
                "cv_pooled_macro_f1": pooled.pop("macro_f1"),
                "cv_fold_macro_f1_mean": np.mean(fold_f1),
                "cv_fold_macro_f1_std": np.std(fold_f1),
                **{f"cv_{k}": v for k, v in pooled.items()},
            }
        )
        r = rows[-1]
        log.info(
            "%-34s pooled OOF F1 %.4f (folds %.4f ± %.4f)",
            name, r["cv_pooled_macro_f1"], r["cv_fold_macro_f1_mean"], r["cv_fold_macro_f1_std"],
        )  # fmt: skip
    return pd.DataFrame(rows), folds


def plot_ablation(torch_df, feat_df, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    agg = torch_df.groupby("variant", sort=False)[["val_macro_f1", "ds2_macro_f1"]].agg(["mean", "std"])
    x = np.arange(len(agg))
    bars = (("val_macro_f1", "validation (selection)", "#e8710a"), ("ds2_macro_f1", "DS2 test (reported)", "#1f5fa8"))
    for i, (col, label, color) in enumerate(bars):
        ax1.bar(
            x + (i - 0.5) * 0.38, agg[(col, "mean")], 0.38, yerr=agg[(col, "std")], capsize=4, color=color, label=label
        )
    ax1.set_xticks(x, agg.index, rotation=25, ha="right")
    ax1.set(ylabel="Macro-F1 (mean ± sd over 3 seeds)", ylim=(0, 1), title="Deep models")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    y = np.arange(len(feat_df))
    ax2.barh(y + 0.2, feat_df["cv_macro_f1"], 0.4, color="#e8710a", label="DS1 grouped CV")
    ax2.barh(y - 0.2, feat_df["ds2_macro_f1"], 0.4, color="#1f5fa8", label="DS2 test")
    ax2.set_yticks(y, feat_df["variant"])
    ax2.set(xlabel="Macro-F1", xlim=(0, 1), title="Classical models: feature ablation")
    ax2.legend(loc="lower right")
    ax2.grid(axis="x", alpha=0.3)
    fig.suptitle("PulseML ablation: selection on validation/CV only; DS2 shown for transparency")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--cv-select", action="store_true", help="only run the grouped CV between hybrid variants")
    args = parser.parse_args()
    ensure_dirs()
    ds1 = load_beats(config.DS1_PARQUET)

    if args.cv_select:
        cv_df, folds = cv_selection(ds1, args.max_epochs)
        winner = cv_df.loc[cv_df["cv_pooled_macro_f1"].idxmax(), "variant"]
        log.info("Selected by grouped CV: %s", winner)
        print(cv_df.to_string(index=False, float_format="{:.3f}".format))
        cv_df.to_csv(config.REPORTS_DIR / "experiments_cv.csv", index=False)
        out = {"created_at": utc_now(), "folds": folds, "selected_variant": winner, "results": cv_df.to_dict("records")}
        save_json(out, config.REPORTS_DIR / "experiments_cv.json")
        return

    ds2 = load_beats(config.DS2_PARQUET)
    feat_df = feature_ablation(ds1, ds2)
    torch_df = torch_ablation(ds1, ds2, args.max_epochs)
    summary = (
        torch_df.groupby("variant", sort=False)
        .agg(
            val_f1_mean=("val_macro_f1", "mean"),
            val_f1_std=("val_macro_f1", "std"),
            ds2_f1_mean=("ds2_macro_f1", "mean"),
            ds2_f1_std=("ds2_macro_f1", "std"),
            ds2_S_recall=("ds2_S_recall", "mean"),
            ds2_V_recall=("ds2_V_recall", "mean"),
            ds2_F_recall=("ds2_F_recall", "mean"),
        )
        .reset_index()
    )
    winner = summary.loc[summary["val_f1_mean"].idxmax(), "variant"]
    log.info("Selected by validation macro-F1: %s", winner)
    print(summary.to_string(index=False, float_format="{:.3f}".format))

    pd.concat([feat_df, torch_df], ignore_index=True).to_csv(config.REPORTS_DIR / "experiments.csv", index=False)
    save_json(
        {
            "created_at": utc_now(),
            "seeds": SEEDS,
            "selected_torch_variant": winner,
            "torch_summary": summary.to_dict("records"),
            "torch_runs": torch_df.to_dict("records"),
            "feature_ablation": feat_df.to_dict("records"),
        },
        config.REPORTS_DIR / "experiments.json",
    )
    plot_ablation(torch_df, feat_df, config.FIGURES_DIR / "ablation.png")


if __name__ == "__main__":
    main()
