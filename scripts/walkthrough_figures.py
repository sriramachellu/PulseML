"""Pastel, minimal figures for the walkthrough PDF (run before build_walkthrough.py).

Needs the full feature store (python -m src.preprocess) and the raw records in data/raw.
Uses the Segoe UI font, so it is meant to run on Windows.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.preprocess import bandpass_filter  # noqa: E402

OUT = ROOT / "build" / "walkthrough_figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical palette (lightness band, chroma, CVD separation all pass)
BLUE, CORAL, LAV, SAGE = "#5B95E0", "#F0957A", "#A684D6", "#5DBB86"
CLASS = {"N": BLUE, "S": CORAL, "V": LAV, "F": SAGE}
INK, MUTED, GRID, SOFT = "#2E3440", "#6B7280", "#E7E5E0", "#CBD5E1"
BG = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "legend.frameon": False,
    }
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


ds1 = pd.read_parquet(config.DS1_PARQUET)
ds2 = pd.read_parquet(config.DS2_PARQUET)
ev = json.loads(config.EVAL_METRICS.read_text())
t_ms = (np.arange(280) - 140) / 360 * 1000

# 1. A real ECG strip with R-peaks and one RR interval
import wfdb  # noqa: E402

rec = wfdb.rdrecord(str(config.RAW_DIR / "100"), sampfrom=0, sampto=360 * 4)
ann = wfdb.rdann(str(config.RAW_DIR / "100"), "atr", sampfrom=0, sampto=360 * 4)
sig = bandpass_filter(rec.p_signal[:, 0])
t = np.arange(len(sig)) / 360
peaks = [s for s, sym in zip(ann.sample, ann.symbol) if sym in "NLRAV"]
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.plot(t, sig, color=BLUE, lw=1.6)
ax.scatter(np.array(peaks) / 360, sig[peaks], s=36, color=CORAL, zorder=3, edgecolor="white", linewidth=1.5)
a, b = peaks[1] / 360, peaks[2] / 360
y = sig[peaks].max() + 0.25
ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
ax.text((a + b) / 2, y + 0.08, f"RR interval  {1000 * (b - a):.0f} ms", ha="center", color=INK, fontsize=10)
ax.text(peaks[0] / 360 + 0.04, sig[peaks[0]], "R-peak", color=MUTED, fontsize=9, va="center")
ax.set(xlabel="seconds", yticks=[])
ax.spines["left"].set_visible(False)
ax.set_ylim(sig.min() - 0.2, y + 0.35)
save(fig, "ecg_strip")

# 2. Median beat shape per class
fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharey=True)
for ax, (k, c) in zip(axes, enumerate(config.CLASS_NAMES)):
    w = np.stack(ds1.loc[ds1.label == k, "waveform"].sample(400, random_state=0, replace=True).map(np.asarray))
    lo, med, hi = np.percentile(w, [25, 50, 75], axis=0)
    ax.fill_between(t_ms, lo, hi, color=CLASS[c], alpha=0.22, lw=0)
    ax.plot(t_ms, med, color=CLASS[c], lw=2)
    ax.set_title(f"{c}  ·  {config.CLASS_DESCRIPTIONS[c].split(' /')[0].split(' of')[0]}", fontsize=11, loc="left")
    ax.set_xticks([-300, 0, 300])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms from R-peak", fontsize=9)
save(fig, "class_shapes")

# 3. Class counts (DS1 vs DS2)
counts = pd.DataFrame(
    {s: d.label.value_counts().sort_index().values for s, d in (("DS1 · train", ds1), ("DS2 · test", ds2))},
    index=config.CLASS_NAMES,
)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
x = np.arange(4)
for i, (col, color) in enumerate(zip(counts.columns, (BLUE, LAV))):
    bars = ax.bar(x + (i - 0.5) * 0.36, counts[col], 0.34, color=color, label=col)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts[col]], fontsize=8.5, color=INK, padding=2)
ax.set_yscale("log")
ax.set_xticks(x, config.CLASS_NAMES)
ax.set_ylabel("beats (log scale)")
ax.set_ylim(100, 2e5)
ax.legend(loc="upper right", fontsize=9)
hgrid(ax)
save(fig, "class_counts")

# 4. Raw vs filtered signal
rec = wfdb.rdrecord(str(config.RAW_DIR / "105"), sampfrom=360 * 60, sampto=360 * 66)
raw = rec.p_signal[:, 0]
filt = bandpass_filter(raw)
t = np.arange(len(raw)) / 360
fig, axes = plt.subplots(2, 1, figsize=(9, 3.2), sharex=True)
axes[0].plot(t, raw, color=SOFT, lw=1.3)
axes[0].set_title("raw: wandering baseline + noise", loc="left", fontsize=10, color=MUTED)
axes[1].plot(t, filt, color=BLUE, lw=1.3)
axes[1].set_title("after 0.5–40 Hz band-pass", loc="left", fontsize=10, color=MUTED)
for ax in axes:
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
axes[1].set_xlabel("seconds")
save(fig, "filtering")

# 5. Prematurity by class: rr_prev_norm
fig, ax = plt.subplots(figsize=(6.2, 3.1))
data = [ds1.loc[ds1.label == k, "rr_prev_norm"].clip(0.3, 1.8) for k in range(4)]
parts = ax.violinplot(data, showextrema=False, widths=0.75)
for body, c in zip(parts["bodies"], config.CLASS_NAMES):
    body.set_facecolor(CLASS[c])
    body.set_alpha(0.55)
    body.set_edgecolor("none")
meds = [d.median() for d in data]
ax.scatter(range(1, 5), meds, color="white", edgecolor=INK, s=34, zorder=3, lw=1.2)
for i, m in enumerate(meds, 1):
    ax.text(i + 0.14, m - 0.06, f"{m:.2f}", va="center", fontsize=9, color=INK, bbox=dict(fc="white", ec="none", pad=1))
ax.axhline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(4.45, 1.02, "on time", color=MUTED, fontsize=8.5, va="bottom", ha="right")
ax.set_xticks(range(1, 5), config.CLASS_NAMES)
ax.set_ylabel("RR before beat ÷ patient's usual RR")
hgrid(ax)
save(fig, "rr_norm")

# 6. Patient template vs a V beat
from src.dataset import template_matrix  # noqa: E402

sub = ds1[ds1.record_id == "119"].reset_index(drop=True)
tm = template_matrix(sub)
i = int(np.flatnonzero(sub.label.to_numpy() == 2)[5])
beat = np.asarray(sub.waveform[i])
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
axes[0].plot(t_ms, tm[i], color=SOFT, lw=2.4, label="patient's usual beat (template)")
axes[0].plot(t_ms, beat, color=LAV, lw=2, label="this beat (V)")
axes[0].legend(fontsize=8.5, loc="upper left")
axes[0].set_title("channel 1: the beat", loc="left", fontsize=10, color=MUTED)
axes[1].plot(t_ms, beat - tm[i], color=CORAL, lw=2)
axes[1].axhline(0, color=GRID, lw=1)
axes[1].set_title("channel 2: beat − template  (what's different)", loc="left", fontsize=10, color=MUTED)
for ax in axes:
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms from R-peak", fontsize=9)
save(fig, "template")

# 7. Macro-F1 per model (emphasis on the recommended model)
comp = pd.DataFrame(ev["comparison"]).set_index("model")
order = ["CNN", "XGBoost", "LogReg", "Hybrid"]
fig, ax = plt.subplots(figsize=(6, 2.9))
vals = comp.loc[order, "macro_f1"]
bars = ax.barh(order, vals, color=[SOFT, SOFT, SOFT, BLUE], height=0.6)
ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=4, fontsize=10, color=INK)
ax.set_xlim(0, 0.75)
ax.set_xlabel("macro-F1 on 22 unseen patients")
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.spines["bottom"].set_visible(False)
save(fig, "macro_f1")

# 8. Recall per class per model
fig, ax = plt.subplots(figsize=(8.4, 3.1))
models = ["LogReg", "XGBoost", "CNN", "Hybrid"]
mcol = {"LogReg": SOFT, "XGBoost": SAGE, "CNN": LAV, "Hybrid": BLUE}
w = 0.2
for j, m in enumerate(models):
    v = [comp.loc[m, f"{c}_recall"] for c in config.CLASS_NAMES]
    bars = ax.bar(np.arange(4) + (j - 1.5) * w, v, w * 0.9, color=mcol[m], label=m)
    if m == "Hybrid":
        ax.bar_label(bars, labels=[f"{x:.2f}" for x in v], fontsize=8.5, padding=2, color=INK)
ax.set_xticks(range(4), [f"{c}" for c in config.CLASS_NAMES])
ax.set_ylim(0, 1.1)
ax.set_ylabel("recall (share found)")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9)
hgrid(ax)
save(fig, "recall")

# 9. Hybrid confusion matrix (row-normalised, single-hue sequential)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

pred = pd.read_parquet(config.PREDICTIONS_PARQUET, columns=["label", "Hybrid_pred"])
cm = pd.crosstab(pred.label, pred.Hybrid_pred).reindex(index=range(4), columns=range(4), fill_value=0).to_numpy()
cmn = cm / cm.sum(1, keepdims=True)
cmap = LinearSegmentedColormap.from_list("pastel_blue", ["#F4F8FD", "#C9DCF5", "#8DB5EA", BLUE])
fig, ax = plt.subplots(figsize=(4.4, 3.8))
ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(
            j,
            i,
            f"{cmn[i, j]:.2f}\n{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if cmn[i, j] > 0.6 else INK,
        )
ax.set_xticks(range(4), config.CLASS_NAMES)
ax.set_yticks(range(4), config.CLASS_NAMES)
ax.set_xlabel("predicted")
ax.set_ylabel("true (expert)")
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "confusion")

# 10. Model selection by grouped CV
cv = json.loads((config.REPORTS_DIR / "experiments_cv.json").read_text())
names = [r["variant"].replace(" (waveform + RR)", "").replace(" (+ patient template)", "") for r in cv["results"]]
vals = [r["cv_pooled_macro_f1"] for r in cv["results"]]
fig, ax = plt.subplots(figsize=(6.2, 2.7))
bars = ax.barh(names, vals, color=[SOFT] * 3 + [BLUE], height=0.6)
ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=4, fontsize=10, color=INK)
ax.set_xlim(0.4, 0.62)
ax.set_xlabel("pooled out-of-fold macro-F1 (5-fold, grouped by patient)")
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.spines["bottom"].set_visible(False)
save(fig, "cv_select")

# 11. Feature ablation (classical)
ex = json.loads((config.REPORTS_DIR / "experiments.json").read_text())
fa = {r["variant"]: r["ds2_macro_f1"] for r in ex["feature_ablation"]}
fig, ax = plt.subplots(figsize=(5.4, 2.7))
groups = ["LogReg", "XGBoost"]
v13 = [fa[f"{g} · 13 features (v1)"] for g in groups]
v16 = [fa[f"{g} · 16 features (+ normalised RR)"] for g in groups]
x = np.arange(2)
b1 = ax.bar(x - 0.19, v13, 0.36, color=SOFT, label="13 features")
b2 = ax.bar(x + 0.19, v16, 0.36, color=SAGE, label="+ patient-normalised RR")
for b in (b1, b2):
    ax.bar_label(b, labels=[f"{v:.3f}" for v in b.datavalues], fontsize=9, padding=2, color=INK)
ax.set_xticks(x, groups)
ax.set_ylim(0, 0.75)
ax.set_ylabel("DS2 macro-F1")
ax.legend(fontsize=9, loc="upper right")
hgrid(ax)
save(fig, "feature_ablation")

# 12. Calibration / conformal coverage
cal = json.loads((config.REPORTS_DIR / "calibration_report.json").read_text())
sets = cal["ds2"]["sets"]
fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
cov = [sets[f"{c}_coverage"] for c in config.CLASS_NAMES]
bars = axes[0].bar(config.CLASS_NAMES, cov, color=[CLASS[c] for c in config.CLASS_NAMES], width=0.6)
axes[0].bar_label(bars, labels=[f"{v:.2f}" for v in cov], fontsize=9.5, padding=2, color=INK)
axes[0].axhline(0.9, color=INK, lw=1, ls=(0, (3, 3)))
axes[0].text(-0.45, 1.06, "dashed line = 0.90 target", ha="left", va="bottom", fontsize=8.5, color=MUTED)
axes[0].set(ylim=(0, 1.12), ylabel="true class inside the set")
axes[0].set_title("prediction-set coverage on unseen patients", loc="left", fontsize=10, color=MUTED)
hgrid(axes[0])
nll = [cal["ds2"]["metrics"]["raw"]["nll"], cal["ds2"]["metrics"]["bcts"]["nll"]]
bars = axes[1].bar(["raw", "calibrated"], nll, color=[SOFT, BLUE], width=0.55)
axes[1].bar_label(bars, labels=[f"{v:.3f}" for v in nll], fontsize=9.5, padding=2, color=INK)
axes[1].set_title("log-loss (lower = more honest)", loc="left", fontsize=10, color=MUTED)
axes[1].set_ylim(0, 0.6)
hgrid(axes[1])
save(fig, "calibration")

# 13. Per-patient accuracy (hybrid)
pr = pd.read_csv(config.REPORTS_DIR / "per_record_metrics.csv", dtype={"record_id": str}).sort_values("Hybrid_acc")
overall = comp.loc["Hybrid", "accuracy"]
fig, ax = plt.subplots(figsize=(9, 2.8))
colors = [CORAL if f else BLUE for f in pr.Hybrid_flag]
ax.vlines(range(len(pr)), 0, pr.Hybrid_acc, color=colors, lw=2.5, alpha=0.5)
ax.scatter(range(len(pr)), pr.Hybrid_acc, color=colors, s=40, zorder=3, edgecolor="white", lw=1.5)
ax.axhline(overall, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(len(pr) - 0.5, overall + 0.02, f"overall {overall:.2f}", ha="right", fontsize=8.5, color=MUTED)
ax.set_xticks(range(len(pr)), pr.record_id, fontsize=8.5, rotation=0)
ax.set(ylim=(0, 1.05), ylabel="accuracy", xlabel="patient record (coral = >10 points below overall)")
hgrid(ax)
save(fig, "per_patient")

# 14. CNN training curve (shows early stopping)
tmeta = json.loads(config.TRAINING_METADATA.read_text())
h = pd.DataFrame(tmeta["cnn"]["history"])
fig, ax = plt.subplots(figsize=(5.6, 2.7))
ax.plot(h.epoch, h.train_macro_f1, color=SOFT, lw=2, label="train")
ax.plot(h.epoch, h.val_macro_f1, color=CORAL, lw=2, label="validation (held-out patients)")
best = h.loc[h.val_macro_f1.idxmax()]
ax.scatter([best.epoch], [best.val_macro_f1], color=CORAL, s=50, zorder=3, edgecolor="white", lw=1.5)
ax.annotate(
    "best epoch kept",
    (best.epoch, best.val_macro_f1),
    xytext=(8, 10),
    textcoords="offset points",
    fontsize=8.5,
    color=MUTED,
)
ax.set(xlabel="epoch", ylabel="macro-F1", ylim=(0, 1))
ax.legend(fontsize=8.5, loc="lower right")
hgrid(ax)
save(fig, "training_curve")

print("figures:", sorted(p.name for p in OUT.glob("*.png")))
