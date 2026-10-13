# Methodology

> ⚠️ Research demonstration only. Not for clinical use.

This document specifies exactly what PulseML computes, so results can be reproduced and criticised.
All constants live in [`src/config.py`](../src/config.py).

## 1. Data

**Source:** MIT-BIH Arrhythmia Database v1.0.0 (PhysioNet): 48 half-hour, two-channel ambulatory ECG
recordings from 47 subjects (records 201 and 202 are the same person), digitised at 360 Hz with
11-bit resolution over a 10 mV range. Two or more cardiologists annotated every beat independently.

**Record exclusion (AAMI EC57):** paced records 102, 104, 107 and 217 are removed, leaving **44 records**.

**Lead:** MLII, found by *name* rather than channel index, because record 114 has its MLII on channel 1.

## 2. Signal processing

| Step | Specification | Rationale |
|---|---|---|
| Band-pass | 4th-order Butterworth, 0.5–40 Hz, second-order sections, **zero-phase** (`sosfiltfilt`) | Removes baseline wander (<0.5 Hz) and EMG/mains noise (>40 Hz) without shifting R-peaks; tested in `test_preprocess.py` |
| Segmentation | 280 samples (0.78 s) per beat: 140 before + 140 after the annotated R-peak | Covers P-wave, QRS and most of the T-wave at normal heart rates |
| Boundary handling | Beats whose window crosses the start/end of the recording are dropped | Never pad with fabricated signal |
| Normalisation | Per-beat z-score (`(x − μ) / (σ + 1e-8)`) | Removes gain differences between patients and devices; flat beats stay finite |

## 3. Labels (AAMI EC57)

| AAMI class | MIT-BIH symbols | Label |
|---|---|---|
| N: normal & bundle branch block | `N L R e j` | 0 |
| S: supraventricular ectopic | `A a J S` | 1 |
| V: ventricular ectopic | `V E` | 2 |
| F: fusion | `F` | 3 |
| Q: paced / unclassifiable | `/ f Q` (+ rare `B r n ?`) | excluded, but still used for RR timing |

Non-beat annotations (`+ ~ | ! [ ] " x` …) are skipped. The first and last beat of each record are
dropped because they lack a neighbour for RR computation.

**Data-quality flags (warnings, not removals):** 80 beats have an RR interval > 3 s. Record 232 (62) has genuine
sinus pauses of up to 5.9 s. Records 207, 201 and 208 have annotation gaps over unreadable segments, up to 100 s in
record 207, which distort the RR features and local mean RR of the few affected beats. No missing values, flat beats,
amplitude violations (±5 mV), duplicate keys or DS1/DS2 record overlap.

**Lineage:** 103,724 annotations → 100,733 beat annotations → **100,630 beats** (15 excluded by class,
88 without an RR neighbour, 0 at boundaries). Per-record counts are in `outputs/parquet/pipeline_stats.parquet`.

## 4. Engineered features (16)

RR intervals are in **samples at 360 Hz** and are computed from *all* beat annotations, including excluded classes.

| Feature | Definition |
|---|---|
| `rr_prev` | R(k) − R(k−1) |
| `rr_next` | R(k+1) − R(k) |
| `rr_ratio` | rr_prev / rr_next |
| `local_hr` | 360 · 60 / rr_prev (bpm) |
| `rr_local_avg` | Mean of the last ≤32 RR intervals ending at beat k (**causal**) |
| `rr_prev_norm` | rr_prev / rr_local_avg: prematurity relative to *this patient's* rhythm |
| `rr_next_norm` | rr_next / rr_local_avg: compensatory pause relative to this patient |
| `wf_mean, wf_std, wf_min, wf_max, wf_skew, wf_kurtosis` | Statistics of the **filtered, un-normalised** window (mV); a z-scored window would always have mean 0 / sd 1 |
| `qrs_proxy` | max \|x\| in the central 40 samples |
| `pre_energy` | Σx² over samples 0–100 |
| `post_energy` | Σx² over samples 180–280 |

The three normalised-RR features were added in iteration 2 (see [EXPERIMENTS.md](EXPERIMENTS.md)). They use
only unlabelled timing from the same recording up to the current beat, so they are available at
inference time in a streaming setting and leak no labels.

## 5. Splits: the most important design decision

**Inter-patient split (de Chazal et al., 2004):**
* DS1 (train, 22 records): 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
* DS2 (test, 22 records): 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

Code asserts that the splits are disjoint and that no DS2 record reaches training. `validate.py` checks it again on the
written Parquet files. DS2 is read only by `evaluate.py` and by the reporting step of `experiments.py`.

> **Caveat:** records 201 and 202 are the same subject, and 201 is in DS1 while 202 is in DS2. This
> overlap is inherent to the standard de Chazal split and is kept so results compare with the literature.

**Why not a random beat split?** Beats from the same patient are highly correlated. A random split
puts near-duplicates of test beats in training and typically reports >99% accuracy that collapses on new patients.

**Hyperparameter search (LogReg/XGBoost):** `GridSearchCV` with 3-fold `StratifiedGroupKFold`, grouped
by `record_id`, scored by macro-F1, on DS1 only.

**Validation split for neural networks:** ~20% of DS1 *records*. Minority classes are concentrated
in a few patients (record 208 holds 372 of DS1's 414 F beats), so a naive grouped split can strip a
class from training. In iteration 1 that happened: validation took record 208 and left 37 F beats for training.
The split now searches 5,000 seeded record subsets and keeps the one whose per-class validation
share is closest to 20%, subject to every class keeping ≥50% of its beats in training. Result:
validation = records 114, 124, 205, 223 (8 k N, 119 S, 634 V, 34 F).

## 6. Models

| Model | Input | Details |
|---|---|---|
| Logistic regression | 16 features | StandardScaler → multinomial LR, `class_weight="balanced"`, C tuned |
| XGBoost | 16 features | `hist` trees, balanced sample weights, depth/trees/learning-rate tuned |
| **ECGCNN** (specified baseline) | z-scored beat (1×280) | Conv(1→32,k7)-BN-ReLU-MaxPool → Conv(32→64,k5)-BN-ReLU-MaxPool → Conv(64→128,k3)-BN-ReLU-GAP → Dropout .5 → FC128→64 → ReLU → Dropout .3 → FC64→4. 44,228 params |
| **HybridCNN** | beat + 7 RR features | Same conv trunk (128-d) ‖ RR MLP (7→32→32, BN) → Dropout .5 → FC160→64 → Dropout .3 → FC64→4. RR standardisation stats are stored as model buffers (computed on training records only) |
| **HybridCNN-rel** | 2-channel beat + RR | As HybridCNN, but channel 2 is the beat minus the median of the patient's previous 16 z-scored beats (causal template; `build_patient_template` reproduces it for API clients) |

Neural training: Adam (lr 1e-3, weight decay 1e-4), batch 256, inverse-frequency class-weighted
cross-entropy (weights from DS1: N 0.28, S 13.5, V 3.36, F 30.8).
* Specified CNN and ablation runs: at most 50 epochs, early stopping on validation macro-F1 (patience 7), best-epoch weights restored.
* **Recommended model** (HybridCNN-rel + augmentation): variant chosen by 5-fold record-grouped CV on DS1, then
  **refit on all 22 DS1 patients for 7 epochs** (median best epoch across CV folds) with seeds 42, 7 and 2024. The
  three members' softmax outputs are averaged (`SeedEnsemble`).

**Augmentation** (optional, train only): random R-peak shift ±5 samples, gain ×[0.85, 1.15], Gaussian
noise σ = 0.05, and a slow sinusoidal drift of amplitude ≤0.1, all seeded.

## 7. Metrics

* Per class: precision, recall (sensitivity), F1, specificity (TN / (TN + FP)), support.
* Overall: **macro-F1 (primary)**, weighted-F1, accuracy, one-vs-rest ROC-AUC per class and its mean.
* Curves: ROC, precision–recall (with prevalence baseline) and quantile-binned reliability (calibration), per class.
* Patient level: per-record accuracy and abnormal-beat sensitivity. A record is flagged when its
  accuracy is >10 percentage points below the model's overall DS2 accuracy.

**Calibration and uncertainty (`src/calibrate.py`):** the final recipe is re-run in the 5 record-grouped CV folds to
get out-of-fold log-probabilities for every DS1 beat. On these, temperature scaling and BCTS (temperature plus a
centred per-class bias) are fitted by minimising NLL with L-BFGS; BCTS is served. Mondrian split-conformal thresholds
qhat_c (score 1 − p_y, α = 0.1, finite-sample quantile ⌈(n_c+1)(1−α)⌉/n_c per class) define the prediction set
{c : 1 − p_c ≤ qhat_c}. DS2 is used only to report NLL, Brier, top-label ECE (15 bins), coverage and set size.

Accuracy is never reported alone: a model that predicts "N" for every beat scores 89% accuracy on DS2.

## 8. Reproducibility

* One seed (`config.SEED = 42`) for Python, NumPy and PyTorch (`cudnn.deterministic=True`); DataLoader
  shuffling and augmentation use seeded `torch.Generator`s. Re-running seed 42 reproduced the CNN's
  val/DS2 macro-F1 exactly (0.4640 / 0.4079).
* Every stage logs counts; `preprocess_summary.json`, `pipeline_stats.parquet`, `training_metadata.json`,
  `evaluation_metrics.json` and `experiments.json` record lineage, hyperparameters, split membership,
  data hashes (SHA-256 of the Parquet files) and model hashes.
* Ablations report mean ± sd over 3 seeds (42, 7, 2024). Variants were first compared on validation; the final
  decision used 5-fold record-grouped CV on DS1 (see EXPERIMENTS.md, iteration 4, including the disclosure).

## 9. Environment notes

* Tested on Windows 11 with Python 3.12 + CUDA 12.6 (RTX 3060) and on Linux CI with Python 3.11 CPU-only.
* `sklearn.calibration` is deliberately not imported (PulseML implements its own reliability curve),
  because that module loads a compiled extension blocked by some Windows Application Control policies.
