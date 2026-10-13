"""Calibrated probabilities and conformal prediction sets for the hybrid ensemble.

Class-weighted training inflates minority-class probabilities, and for triage a set of plausible
classes with a coverage target is safer than one label. Nothing here is fitted on DS2:

1. Out-of-fold predictions for all DS1 patients, using the exact final recipe inside 5-fold grouped CV.
2. Fit temperature scaling and BCTS (temperature + per-class bias) on them by NLL. Keep whichever gives
   the lowest NLL; only switch the predicted label to the calibrated argmax if that also improves macro-F1.
3. Class-conditional (Mondrian) split-conformal thresholds for 90% coverage per class.
4. Report calibration and coverage on DS2.
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from src import config
from src.dataset import load_beats
from src.evaluate import cnn_predict_proba, hybrid_inputs
from src.experiments import grouped_folds
from src.models import SeedEnsemble, apply_calibration, load_cnn, prediction_sets  # noqa: F401 (re-exported)
from src.train import train_torch_model
from src.utils import ensure_dirs, get_logger, load_json, save_json, utc_now

plt.switch_backend("Agg")
log = get_logger("calibrate")

LABELS = list(range(config.NUM_CLASSES))
CALIBRATION_FILE = config.MODELS_DIR / "calibration.json"
OOF_FILE = config.REPORTS_DIR / "oof_hybrid_logp.npz"
ALPHA = 0.10  # 90% coverage per class


def fit_calibration(logp, y, with_bias):
    """Fit temperature T (and a per-class bias) by minimising NLL with L-BFGS."""
    z, t = torch.tensor(logp, dtype=torch.float64), torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=with_bias)
    params = [log_t, b] if with_bias else [log_t]
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp() + b, t)
        loss.backward()
        return loss

    opt.step(closure)
    bias = (b - b.mean()).tolist() if with_bias else None  # softmax ignores a constant shift
    return log_t.exp().item(), bias


def conformal_qhat(probs, y, alpha=ALPHA):
    """Per-class conformal threshold on the score 1 - p_true."""
    qhat = []
    for k in LABELS:
        s = np.sort(1.0 - probs[y == k, k])
        n = len(s)
        qhat.append(float(s[min(n, math.ceil((n + 1) * (1 - alpha))) - 1]) if n else 1.0)
    return qhat


def ece_top_label(probs, y, n_bins=15):
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def prob_metrics(probs, y):
    pred = probs.argmax(1)
    kw = dict(labels=LABELS, average=None, zero_division=0)
    return {
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()),
        "brier": float(((probs - np.eye(config.NUM_CLASSES)[y]) ** 2).sum(1).mean()),
        "ece": ece_top_label(probs, y),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        **{f"{c}_precision": float(v) for c, v in zip(config.CLASS_NAMES, precision_score(y, pred, **kw))},
        **{f"{c}_recall": float(v) for c, v in zip(config.CLASS_NAMES, recall_score(y, pred, **kw))},
    }


def set_metrics(sets, y):
    covered, size = sets[np.arange(len(y)), y], sets.sum(1)
    per_class = {
        f"{c}_coverage": float(covered[y == k].mean()) if (y == k).any() else None
        for k, c in enumerate(config.CLASS_NAMES)
    }
    return {
        "coverage": float(covered.mean()),
        **per_class,
        "mean_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "empty_rate": float((size == 0).mean()),
    }


def oof_log_probs(ds1):
    cv_path = config.REPORTS_DIR / "experiments_cv.json"
    folds = load_json(cv_path)["folds"] if cv_path.exists() else grouped_folds(ds1)  # same folds as selection
    x, rr = hybrid_inputs(ds1, {"model_type": config.HYBRID_MODEL_TYPE})
    logp = np.zeros((len(ds1), config.NUM_CLASSES))
    for i, (tr, va) in enumerate(folds):
        sub = ds1[ds1["record_id"].isin(tr)].reset_index(drop=True)
        members = [
            train_torch_model(
                sub, config.HYBRID_MODEL_TYPE, augment=config.HYBRID_AUGMENT, seed=s,
                fixed_epochs=config.HYBRID_REFIT_EPOCHS, tag=f"oof{i}/s{s}",
            )["model"]
            for s in config.ENSEMBLE_SEEDS
        ]  # fmt: skip
        m = ds1["record_id"].isin(va).to_numpy()
        logp[m] = np.log(np.clip(cnn_predict_proba(SeedEnsemble(members).eval(), x[m], rr[m]), 1e-12, None))
        log.info("OOF fold %d/%d done (held out %s)", i + 1, len(folds), ", ".join(va))
    return logp, folds


def reliability(ax, probs, y, label, color):
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, 11)
    bins = [(conf > lo) & (conf <= hi) for lo, hi in zip(edges[:-1], edges[1:])]
    bins = [m for m in bins if m.sum() >= 20]
    xs, ys = [conf[m].mean() for m in bins], [correct[m].mean() for m in bins]
    ax.plot(xs, ys, marker="o", color=color, label=f"{label} (ECE {ece_top_label(probs, y):.3f})")


def plot_report(y2, raw2, cal2, sets2, path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.4))
    ax1.plot([0, 1], [0, 1], ls=":", color="grey")
    reliability(ax1, raw2, y2, "raw", "#e8710a")
    reliability(ax1, cal2, y2, "calibrated", "#1f5fa8")
    ax1.set(title="Top-label reliability on DS2", xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    covered = sets2[np.arange(len(y2)), y2]
    ax2.bar_label(ax2.bar(config.CLASS_NAMES, [covered[y2 == k].mean() for k in LABELS], color="#1f5fa8"), fmt="%.2f")
    ax2.axhline(1 - ALPHA, ls="--", color="#c5221f", label=f"target {1 - ALPHA:.0%}")
    ax2.set(title="Conformal set coverage by true class (DS2)", ylim=(0, 1.05), ylabel="coverage")
    ax2.legend(loc="lower right")

    sizes, counts = np.unique(sets2.sum(1), return_counts=True)
    ax3.bar(sizes.astype(str), counts / counts.sum(), color="#7f8c9b")
    ax3.set(title="Prediction-set size distribution (DS2)", xlabel="classes in set", ylabel="fraction of beats")
    for ax in (ax1, ax2, ax3):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Hybrid-rel ensemble: calibration and conformal sets (fitted on DS1 out-of-fold only)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def log_metrics(split, metrics):
    for k, m in metrics.items():
        log.info(
            "%s %-11s NLL %.4f  Brier %.4f  ECE %.4f  macro-F1 %.4f  S prec %.3f rec %.3f",
            split, k, m["nll"], m["brier"], m["ece"], m["macro_f1"], m["S_precision"], m["S_recall"],
        )  # fmt: skip


def run(reuse_oof=False):
    ensure_dirs()
    ds1 = load_beats(config.DS1_PARQUET)
    y1 = ds1["label"].to_numpy()
    if reuse_oof and OOF_FILE.exists():
        logp1 = np.load(OOF_FILE)["logp"]
    else:
        logp1, _ = oof_log_probs(ds1)
        np.savez_compressed(OOF_FILE, logp=logp1.astype(np.float32), record_id=ds1["record_id"].to_numpy(), y=y1)

    t_ts, _ = fit_calibration(logp1, y1, with_bias=False)
    t_bc, b_bc = fit_calibration(logp1, y1, with_bias=True)
    variants = {
        "raw": {"temperature": 1.0, "bias": None},
        "temperature": {"temperature": t_ts, "bias": None},
        "bcts": {"temperature": t_bc, "bias": b_bc},
    }
    oof = {k: prob_metrics(apply_calibration(logp1, **v), y1) for k, v in variants.items()}
    log_metrics("OOF", oof)

    # decided on DS1 out-of-fold data only
    chosen = min(variants, key=lambda k: oof[k]["nll"])
    use_cal_argmax = chosen != "raw" and oof[chosen]["macro_f1"] > oof["raw"]["macro_f1"]
    cal1 = apply_calibration(logp1, **variants[chosen])
    qhat = conformal_qhat(cal1, y1)
    oof_sets = set_metrics(prediction_sets(cal1, qhat), y1)
    log.info("Chose %s (calibrated argmax: %s), qhat %s", chosen, use_cal_argmax, np.round(qhat, 4).tolist())

    calib = {
        "fitted_at": utc_now(),
        "model": config.HYBRID_MODEL_TYPE,
        "method": chosen,
        **variants[chosen],
        "use_calibrated_argmax": use_cal_argmax,
        "alpha": ALPHA,
        "conformal": "Mondrian (class-conditional) split conformal, score = 1 - p_y",
        "qhat": qhat,
        "fit_data": "DS1 out-of-fold predictions (5-fold record-grouped CV, final recipe)",
    }
    save_json(calib, CALIBRATION_FILE)

    # report on the unseen DS2 patients
    ds2 = load_beats(config.DS2_PARQUET)
    y2 = ds2["label"].to_numpy()
    model, meta = load_cnn(config.HYBRID_CHECKPOINT)
    logp2 = np.log(np.clip(cnn_predict_proba(model, *hybrid_inputs(ds2, meta)), 1e-12, None))
    ds2_metrics = {k: prob_metrics(apply_calibration(logp2, **v), y2) for k, v in variants.items()}
    log_metrics("DS2", ds2_metrics)
    cal2 = apply_calibration(logp2, **variants[chosen])
    sets2 = prediction_sets(cal2, qhat)
    ds2_sets = set_metrics(sets2, y2)
    log.info("DS2 set coverage %.3f, mean size %.2f", ds2_sets["coverage"], ds2_sets["mean_set_size"])

    plot_report(y2, apply_calibration(logp2), cal2, sets2, config.FIGURES_DIR / "hybrid_calibration.png")
    report = {
        "calibration": calib,
        "oof": {"metrics": oof, "sets": oof_sets},
        "ds2": {"metrics": ds2_metrics, "sets": ds2_sets},
        "raw_oof_sanity": oof["raw"],
    }
    save_json(report, config.REPORTS_DIR / "calibration_report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse-oof", action="store_true", help="reuse saved out-of-fold predictions")
    run(parser.parse_args().reuse_oof)
