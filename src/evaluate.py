"""Evaluate every trained model on the held-out DS2 patients, draw the figures and write the model card."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from src import config
from src.dataset import feature_matrix, load_beats, template_matrix, waveform_matrix
from src.models import load_cnn
from src.utils import ensure_dirs, file_sha256, get_logger, load_json, save_json, utc_now

plt.switch_backend("Agg")
log = get_logger("evaluate")

LABELS = list(range(config.NUM_CLASSES))
MODEL_COLORS = {"LogReg": "#7f8c9b", "XGBoost": "#e8710a", "CNN": "#8fb3e0", "Hybrid": "#1f5fa8"}
FLAG_ACC_DROP = 0.10  # flag records more than 10 points below the model's overall accuracy


def per_class_metrics(y_true, y_pred):
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fp = cm.sum(0) - np.diag(cm)
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    spec = np.divide(tn, tn + fp, out=np.zeros(len(LABELS)), where=(tn + fp) > 0)
    return pd.DataFrame(
        {"class": config.CLASS_NAMES, "precision": prec, "recall": rec, "f1": f1, "specificity": spec, "support": sup}
    )


def overall_metrics(y_true, y_pred, proba):
    aucs = {}
    for k, name in enumerate(config.CLASS_NAMES):
        yk = y_true == k
        aucs[name] = float(auc(*roc_curve(yk, proba[:, k])[:2])) if 0 < yk.sum() < len(yk) else None
    valid = [v for v in aucs.values() if v is not None]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "auc": aucs,
        "macro_auc": float(np.mean(valid)) if valid else None,
    }


def per_record_analysis(df, preds):
    y_all = df["label"].to_numpy()
    rows = []
    for rid, idx in df.groupby("record_id").indices.items():
        y = y_all[idx]
        abn = y != 0
        row = {
            "record_id": rid,
            "beats": len(idx),
            **{f"n_{c}": int((y == k).sum()) for k, c in enumerate(config.CLASS_NAMES)},
        }
        for name, p in preds.items():
            row[f"{name}_acc"] = float((p[idx] == y).mean())
            row[f"{name}_abn_sens"] = float((p[idx][abn] == y[abn]).mean()) if abn.any() else np.nan
        rows.append(row)
    rec = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    for name, p in preds.items():
        rec[f"{name}_flag"] = rec[f"{name}_acc"] < (p == y_all).mean() - FLAG_ACC_DROP
    return rec


@torch.no_grad()
def cnn_predict_proba(model, waveforms, feats=None, batch=2048):
    """Softmax output of a torch model. waveforms is (n, 280) or (n, C, 280); feats are raw RR features."""
    out = []
    for i in range(0, len(waveforms), batch):
        x = torch.from_numpy(waveforms[i : i + batch])
        x = x[:, None] if x.dim() == 2 else x
        f = None if feats is None else torch.from_numpy(feats[i : i + batch])
        out.append(model(x, f).softmax(1).numpy())
    return np.concatenate(out) if out else np.zeros((0, config.NUM_CLASSES))


def hybrid_inputs(df, meta):
    wf = waveform_matrix(df)
    if meta.get("model_type") == "hybrid_rel":
        wf = np.stack([wf, wf - template_matrix(df)], axis=1)
    return wf, feature_matrix(df, config.RR_FEATURES)


def predict_all(ds2):
    X = feature_matrix(ds2)
    probas = {}
    for name, path in (("LogReg", config.LOGREG_MODEL), ("XGBoost", config.XGB_MODEL)):
        if path.exists():
            probas[name] = joblib.load(path).predict_proba(X)
    if config.CNN_CHECKPOINT.exists():
        probas["CNN"] = cnn_predict_proba(load_cnn(config.CNN_CHECKPOINT)[0], waveform_matrix(ds2))
    if config.HYBRID_CHECKPOINT.exists():
        model, meta = load_cnn(config.HYBRID_CHECKPOINT)
        probas["Hybrid"] = cnn_predict_proba(model, *hybrid_inputs(ds2, meta))
    if not probas:
        raise FileNotFoundError("No trained models found, run `python -m src.train` first.")
    return probas


def write_demo_sample(ds2, pred_df, per_class=100):
    """A small committed DS2 sample (with templates) so the dashboard works without the raw data."""
    demo = pd.concat([ds2.reset_index(drop=True), pred_df.drop(columns=["record_id", "beat_idx", "label"])], axis=1)
    demo["template"] = list(template_matrix(ds2).astype(np.float32))
    picked = demo.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(per_class, len(g)), random_state=config.SEED), include_groups=False
    )
    demo = demo.loc[sorted(picked.index)].reset_index(drop=True)
    demo.to_parquet(config.DEMO_PARQUET, index=False)
    log.info("Demo sample: %d beats", len(demo))


def calibration_curve(y_true, prob, n_bins=10):
    """Quantile-binned reliability curve, like sklearn's strategy="quantile".

    Written here because sklearn.calibration loads a compiled module that some Windows
    Application Control policies block.
    """
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.array([y_true.mean()]), np.array([prob.mean()])
    bins = np.clip(np.searchsorted(edges[1:-1], prob, side="right"), 0, len(edges) - 2)
    used = [b for b in range(len(edges) - 1) if (bins == b).any()]
    return np.array([y_true[bins == b].mean() for b in used]), np.array([prob[bins == b].mean() for b in used])


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(y, probas, path, title, curve):
    """One panel per class; `curve(ax, yk, p, name)` draws a single model's line."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    fig.suptitle(title)
    for k, ax in enumerate(axes):
        yk = (y == k).astype(int)
        for name, p in probas.items():
            curve(ax, yk, p[:, k], name)
        ax.set_title(config.CLASS_NAMES[k])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _save(fig, path)


def roc_line(ax, yk, p, name):
    fpr, tpr, _ = roc_curve(yk, p)
    ax.plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate")


def pr_line(ax, yk, p, name):
    prec, rec, _ = precision_recall_curve(yk, p)
    ax.plot(rec, prec, color=MODEL_COLORS[name], label=f"{name} AP={average_precision_score(yk, p):.3f}")
    ax.axhline(yk.mean(), ls="--", color="grey", lw=1)
    ax.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))


def reliability_line(ax, yk, p, name):
    frac, mean = calibration_curve(yk, p)
    ax.plot(mean, frac, marker="o", ms=3, color=MODEL_COLORS[name], label=name)
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency")


def plot_confusion(y, preds, path):
    fig, axes = plt.subplots(len(preds), 2, figsize=(10, 4.2 * len(preds)), squeeze=False)
    for row, (name, p) in enumerate(preds.items()):
        cm = confusion_matrix(y, p, labels=LABELS)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        for ax, mat, fmt, kind in ((axes[row, 0], cm, "d", "absolute"), (axes[row, 1], cmn, ".2f", "row-normalised")):
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in LABELS:
                for j in LABELS:
                    color = "white" if cmn[i, j] > 0.5 else "black"
                    ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color=color, fontsize=9)
            ax.set_xticks(LABELS, config.CLASS_NAMES)
            ax.set_yticks(LABELS, config.CLASS_NAMES)
            ax.set(xlabel="Predicted", ylabel="True", title=f"{name} ({kind})")
    _save(fig, path)


def plot_per_record(rec, models, path):
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(rec))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar(x + i * w - 0.4 + w / 2, rec[f"{m}_acc"], width=w, color=MODEL_COLORS[m], label=m)
    ax.set_xticks(x, rec["record_id"], rotation=60)
    ax.set(ylabel="Accuracy", title="Per-record accuracy on DS2", ylim=(0, 1.05))
    ax.legend(ncol=len(models))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def md_table(df, floatfmt=".3f"):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(format(v, floatfmt) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def ablation_section():
    text = ""
    exp_path, cv_path = config.REPORTS_DIR / "experiments.json", config.REPORTS_DIR / "experiments_cv.json"
    if exp_path.exists() and (exp := load_json(exp_path)).get("torch_summary"):
        cols = ["variant", "val_f1_mean", "val_f1_std", "ds2_f1_mean", "ds2_f1_std"]
        text = (
            f"Deep-model variants, mean ± sd over seeds {exp.get('seeds')}. Selection used **validation macro-F1 only**; "
            f"selected: **{exp.get('selected_torch_variant')}**. DS2 is shown for transparency.\n\n"
            + md_table(pd.DataFrame(exp["torch_summary"])[cols])
        )
    if cv_path.exists():
        cv = load_json(cv_path)
        cols = ["variant", "cv_pooled_macro_f1", "cv_fold_macro_f1_mean", "cv_fold_macro_f1_std"]
        text += (
            "\n\nThe top variants were within one seed-sd on the 4-patient validation split, so the final choice "
            "used **5-fold record-grouped CV over all 22 DS1 patients** (pooled out-of-fold macro-F1). Selected: "
            f"**{cv['selected_variant']}**, refit on all of DS1 for {config.HYBRID_REFIT_EPOCHS} epochs "
            f"(median CV best epoch) as a {len(config.ENSEMBLE_SEEDS)}-seed ensemble.\n\n"
            + md_table(pd.DataFrame(cv["results"])[cols])
        )
    return text or "Run `python -m src.experiments` to populate."


def write_model_card(results, comparison, per_class, rec, train_meta):
    cnn, hyb, base = (train_meta.get(k) or {} for k in ("cnn", "hybrid", "baselines"))
    flagged = {m: rec.loc[rec[f"{m}_flag"], "record_id"].tolist() for m in results["models"]}
    template = (Path(__file__).parent / "templates" / "model_card.md").read_text(encoding="utf-8")
    card = template.format(
        evaluated_at=results["evaluated_at"],
        version=config.MODEL_VERSION,
        cnn_sha=results["artifacts"].get("cnn_sha256", "n/a"),
        hybrid_sha=results["artifacts"].get("hybrid_sha256", "n/a"),
        hyb_arch=hyb.get("architecture", "HybridCNN"),
        hyb_params=hyb.get("n_parameters", 0),
        hyb_aug=hyb.get("augmentation"),
        hyb_epochs=hyb.get("refit_fixed_epochs", "n/a"),
        hyb_seeds=hyb.get("ensemble_seeds"),
        cnn_params=cnn.get("n_parameters", 0),
        cnn_best=cnn.get("best_epoch", "n/a"),
        cnn_epochs=cnn.get("epochs_run", "n/a"),
        logreg_params=base.get("logreg", {}).get("best_params"),
        xgb_params=base.get("xgboost", {}).get("best_params"),
        seed=config.SEED,
        val_records=", ".join(cnn.get("val_records", [])),
        n_beats=results["n_test_beats"],
        comparison=md_table(comparison),
        per_class="\n\n".join(f"**{m}**\n\n{md_table(df)}" for m, df in per_class.items()),
        ablation=ablation_section(),
        flag_pp=int(FLAG_ACC_DROP * 100),
        flagged="\n".join(f"* **{m}:** {', '.join(v) or 'none'}" for m, v in flagged.items()),
    )
    config.MODEL_CARD.write_text(card, encoding="utf-8")
    log.info("Model card -> %s", config.MODEL_CARD)


def evaluate():
    ensure_dirs()
    ds2 = load_beats(config.DS2_PARQUET)
    assert set(ds2["record_id"]) <= set(config.DS2_RECORDS)
    y = ds2["label"].to_numpy()
    log.info("DS2: %d beats, %d records, classes %s", len(ds2), ds2["record_id"].nunique(), np.bincount(y).tolist())

    probas = predict_all(ds2)
    preds = {m: p.argmax(1) for m, p in probas.items()}
    results = {"evaluated_at": utc_now(), "n_test_beats": len(ds2), "models": {}, "artifacts": {}}
    per_class, rows = {}, []
    for m, p in probas.items():
        pc, ov = per_class_metrics(y, preds[m]), overall_metrics(y, preds[m], p)
        per_class[m] = pc
        results["models"][m] = {"overall": ov, "per_class": pc.to_dict(orient="records")}
        rows.append(
            {"model": m}
            | {k: ov[k] for k in ("accuracy", "macro_f1", "weighted_f1", "macro_auc")}
            | {f"{c}_recall": float(r) for c, r in zip(config.CLASS_NAMES, pc["recall"])}
            | {f"{c}_f1": float(f) for c, f in zip(config.CLASS_NAMES, pc["f1"])}
        )
        log.info(
            "%-8s acc %.4f  macro-F1 %.4f  macro-AUC %.4f", m, ov["accuracy"], ov["macro_f1"], ov["macro_auc"] or np.nan
        )
        print(pc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    comparison = pd.DataFrame(rows)
    rec = per_record_analysis(ds2, preds)
    for m in probas:
        log.info("%-8s flagged records: %s", m, rec.loc[rec[f"{m}_flag"], "record_id"].tolist() or "none")

    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    rec.to_csv(config.REPORTS_DIR / "per_record_metrics.csv", index=False)
    pred_df = ds2[["record_id", "beat_idx", "label"]].copy()
    for m, p in probas.items():
        for k, c in enumerate(config.CLASS_NAMES):
            pred_df[f"{m}_p{c}"] = p[:, k].astype(np.float32)
        pred_df[f"{m}_pred"] = preds[m]
    pred_df.to_parquet(config.PREDICTIONS_PARQUET, index=False)
    write_demo_sample(ds2, pred_df)

    fig_dir = config.FIGURES_DIR
    plot_per_class(y, probas, fig_dir / "roc_curves.png", "One-vs-rest ROC on DS2 (held-out patients)", roc_line)
    plot_per_class(y, probas, fig_dir / "pr_curves.png", "Precision–recall on DS2 (dashed = prevalence)", pr_line)
    plot_per_class(y, probas, fig_dir / "calibration.png", "Calibration (reliability) curves on DS2", reliability_line)
    plot_confusion(y, preds, fig_dir / "confusion_matrix.png")
    plot_per_record(rec, list(probas), fig_dir / "per_record_accuracy.png")

    for key, path in (("cnn_sha256", config.CNN_CHECKPOINT), ("hybrid_sha256", config.HYBRID_CHECKPOINT)):
        if path.exists():
            results["artifacts"][key] = file_sha256(path)
    results["artifacts"]["ds2_sha256"] = file_sha256(config.DS2_PARQUET)
    results |= {
        "comparison": comparison.to_dict(orient="records"),
        "per_record": rec.to_dict(orient="records"),
        "disclaimer": config.DISCLAIMER,
    }
    save_json(results, config.EVAL_METRICS)

    train_meta = load_json(config.TRAINING_METADATA) if config.TRAINING_METADATA.exists() else {}
    write_model_card(results, comparison, per_class, rec, train_meta)
    return results


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    evaluate()
